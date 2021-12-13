module DDMMigration

using Images
using SparseArrays
using BlockArrays
using Distances
using LinearAlgebra
using Hungarian
using NearestNeighbors
using SegmentationUtils
using RegionProps
using StatsBase
using DataFrames
using BioformatsLoader
using DDMFramework
using Random

using JSON

export MigrationState

include("ImageUtils.jl")
include("lazydf.jl")
include("query.jl")

struct FOV
    x::Float64
    y::Float64
    data::Array{DataFrame}
end
include("TrackUtils.jl")
struct MigrationState
    fov_arr::Vector{FOV}
    tracks::DataFrame
    config::Dict{String, Any}
end

Base.push!(x::FOV, df::DataFrame) = push!(x.data, df)
Base.push!(x::MigrationState, fov::FOV) = push!(x.fov_arr, fov)

function find_fov_id(x::Float64, y::Float64, fov_arr::Vector{FOV})
    findfirst(fov ->
        isapprox(x, fov.x, atol=1.0) &&
        isapprox(y, fov.y, atol=1.0),
    fov_arr
    )
end

function get_fov_id!(state::MigrationState, x::Float64, y::Float64)
    fov_id = find_fov_id(x,y,state.fov_arr)
    if isnothing(fov_id)
        push!(state, FOV(x,y,[]))
        return length(state.fov_arr)
    end
    return fov_id
end


function MigrationState(params::Dict{String, Any})
    MigrationState(
        FOV[],
        DataFrame(
            label_id = Int[],
            track_id = Int[],
            t = Int[],
            fov_id = Int[]
        ),
        params
    )
end

function push_new_lazy!(fun, d, k)
    if !haskey(d, k)
        push!(d, k => fun())
    end
    return d
end

function parse_image_meta!(state, data)
    push_new_lazy!(state.config, "image_meta") do
        Dict{String, Any}(
            "stage_pos_x" => [],
            "stage_pos_y" => []
        )
    end

    let (mdata, params) = (data["image"].Pixels, state.config["image_meta"])

        push!(params["stage_pos_x"], mdata[:Plane][1][:PositionX])
        push!(params["stage_pos_y"], mdata[:Plane][1][:PositionY])

        push_new_lazy!(params, "img_size") do
            (y = mdata[:SizeY], x = mdata[:SizeX], z = mdata[:SizeZ])
        end
        
    end
end

function drop_empty_dims(img::ImageMeta)
    dims = Tuple(findall(x -> x.val.stop == 1, img.data.axes))
    dropdims(img.data.data, dims=dims)
end

function DDMFramework.handle_update(state::MigrationState, data)
    parse_image_meta!(state,data) #subject to change
    update_state!(
        state,
        drop_empty_dims(data["image"]) #subject to change,
    )
end

function get_fov!(state::MigrationState, params)
    id = get_fov_id!(state, params["stage_pos_x"][end], params["stage_pos_y"][end])
    id, state.fov_arr[id]
end

to_named_tuple(dict::Dict{K,V}) where {K,V} = NamedTuple{Tuple(Iterators.map(Symbol,keys(dict))), NTuple{length(dict),V}}(values(dict))

function update_state!(state::MigrationState, image)
    display("analyzing image...")
    seg_params = state.config["analysis"]["segmentation"]
    ref_c = seg_params["reference_channel"]
    fov_id, fov = get_fov!(state, state.config["image_meta"])
    
    ref_props = regionprop_analysis(
        image[ref_c,:,:];
        to_named_tuple(seg_params)...
        ) |> DataFrame
    
    props = vcat(DataFrame(
            label_id = Int[],
            centroid_x = Float64[],
            centroid_y = Float64[]
            ),
        ref_props,
        cols=:union
    )
    push!(fov,props)
    
    if length(fov.data) > 1
        update_tracks!(state, fov, fov_id)
    end
    
    ("0", state)
end

function update_tracks!(state::MigrationState, fov::FOV, fov_id::Int64)
    label_mapping, start_id = get_labelmap(state,fov,fov_id)
    tracks = track(fov, fov_id, length(fov.data) .+ (-1:0), label_mapping, start_id)
    append!(state.tracks,tracks)
end

get_labelmap(state::MigrationState, fov::FOV, fov_id::Int64) = get_labelmap(state.tracks, fov, fov_id)

function get_labelmap(tracks::DataFrame, fov::FOV, fov_id::Int64)
    start_id = isempty(tracks) ? 0 : maximum(tracks.track_id)
    tracks_filt = filter([:t, :fov_id] => (x,v) -> (x == length(fov.data)-1) && (v == fov_id), tracks)
    
    Dict(zip(tracks_filt.label_id, tracks_filt.track_id)), start_id
end

table_filters = Dict(
     "bin" => function bin(data, sel, n)
         data = filter(!isnan, data)
         if sel == 1
             hi = quantile(data, sel/n)
             row -> row < hi
         elseif sel == n
             lo = quantile(data, (sel-1)/n)
             row -> row > lo
         else
             lo, hi = quantile(data, [(sel-1)/n, sel/n])
             row -> lo <= row < hi
         end
    end,
    ">" => function gt(data,v)
        >(v)
    end,
    "<" => function lt(data,v)
        <(v)
    end
)

to_stage_pos(xv,yv,stage_x,stage_y, p) = to_stage_pos(xv,
                                                      yv,
                                                      stage_x,
                                                      stage_y,
                                                      p["system"]["camera_M"],
                                                      p["system"]["pixelmicrons"],
                                                      p["image_meta"]["img_size"].y,
                                                      p["image_meta"]["img_size"].x
                                                  )
function to_stage_pos(xv,yv,stage_x,stage_y, camera_m, pixelmicrons, height, width)
    translation(x, p) =  x .* p.pixelmicrons .+ [p.y p.x]
    centre_coords(x, h, w) = ((x .- ([h w] ./2)) .* [-1 1])'
    camera_M = [camera_m["a11"] camera_m["a12"]; camera_m["a21"] camera_m["a22"]]
    p = (pixelmicrons=pixelmicrons, y=stage_y, x=stage_x, h=height, w=width)
    
    corr_coords = camera_M * centre_coords(hcat(yv,xv), p.h, p.w)
    translation(corr_coords', p)[:]
end

function sample_df(df, n::Int64, seed::Int64 = 1234)
    sel = df.selection
    n_tot = length(sel)
    index = randperm(MersenneTwister(seed), n_tot)[1:min(n, n_tot)]
    
    select(df, sel[index])
end

function proximity(pos,selected,r)
    reduced_pos =reduce(hcat, pos)
    tree = KDTree(@view reduced_pos[:,selected])
    idxs = inrange(tree, reduced_pos, r, true)
    length.(idxs)
end

function collect_objects(state)
    objects = DataFrame()
    for (fov_id, fov) in enumerate(state.fov_arr)
        for (t, prop) in enumerate(fov.data)
            df = copy(prop)
            df[!, :fov_id] .= fov_id
            df[!, :t] .= t
            df[!, :stage_x] .= fov.x
            df[!, :stage_y] .= fov.y
            append!(objects, df)
        end
    end
    objects
end

function collect_state(state::MigrationState, args)
    args = Dict(args)
    objects = collect_objects(state)
    df = innerjoin(state.tracks, objects, on = [:fov_id, :t, :label_id])
    index = [:fov_id, :track_id]
    distcols = [:centroid_x,:centroid_y]

    @time data = map(gdf for gdf in groupby(df, index)) do gdf
        (gdf=gdf, centroid_x=diff(gdf.centroid_x), centroid_y=diff(gdf.centroid_y))
    end |> DataFrame
    
    stage_pos_transform(x, y, gdf) = to_stage_pos(x,y, gdf.stage_x[1], gdf.stage_y[1], state.config)
    
    data = LazyDF(
        data;
        mean_speed=[:centroid_x, :centroid_y] => ByRow((x, y) -> mean(hypot.(x, y))),

        last_x=[:gdf] => ByRow(gdf -> gdf.centroid_x[end]),
        last_y=[:gdf] => ByRow(gdf -> gdf.centroid_y[end]),
        last_stage_pos= [:last_x, :last_y, :gdf] => ByRow(stage_pos_transform),
        top_speed_density    = [:last_stage_pos, :mean_speed] => (pos, speed) -> proximity(pos, speed .> 14.35, 150),
        bottom_speed_density = [:last_stage_pos, :mean_speed] => (pos, speed) -> proximity(pos, speed .< 6.66 , 150),
        fov_id = [:gdf] => ByRow(gdf -> gdf.fov_id[end])
    )
    
    
    data = if haskey(args, "filter")
        filters = mapfoldl(vcat, args["filter"]; init=Pair{String, Base.Callable}[]) do (column, filters)
            map(filters) do filt
                op = table_filters[filt["op"]](data[!,column], filt["args"]...)
                column => op
            end
        end
        reduce((d, f) -> filter(f, d), filters; init=data)
    else
        data
    end
    
    if haskey(args, "order")
        columns = [o == "asc" ? data[c] : -data[c] for (o, c) in args["order"]]
        key_type = Tuple{eltype.(columns)...}
        by(i) = key_type((c[i] for c in columns))
        data = select(data, sort(1:nrow(data); by))
    end
        
    if haskey(args, "sample")
        n = args["sample"]["n"]
        seed = args["sample"]["seed"]
        sample_df(data, n, seed)
    elseif haskey(args, "limit")
        n = args["limit"]
        limit(data, n)
    else
        data
    end
end

function get_fovs(state, args)
    return [Dict("x" => fov.x, "y" => fov.y, "n" => nrow.(fov.data)) for fov in state.fov_arr]
end

schema = Dict(
    "query" => "Query",
    "Query" => Dict(
        "migration" => "Migration",
        "fov" => "FOV"
    ),
    "FOV" => Dict(
    ),
    "Migration" => Dict(
        "centroid_x" => "Column",
        "centroid_y" => "Column",
        "last_x" => "Column",
        "last_y" => "Column",
        "last_stage_pos" => "Column",
        "top_speed_density" => "Column",
        "bottom_speed_density" => "Column",
        "mean_speed" => "Column",
        "fov_id" => "fov_id"
    ),
    "Column" => Dict(
        "name" => "String"
    )
    
)

resolvers(state) = Dict(
    "Query" => Dict(
        "migration" => (parent, args) -> collect_state(state, args),
        "fov" => (parent, args) -> get_fovs(state, args)
    )
)

function DDMFramework.query_state(state::MigrationState, query)
    execute_query(query["query"], schema, resolvers(state)) |> JSON.json
end

function Base.show(io::IO, mime::MIME"text/html", state::MigrationState)
    show(io, mime, collect_objects(state))
end

function readnd2(io)
    mktempdir() do d
        path = joinpath(d, "file.nd2")
        open(path, "w") do iow
            write(iow, read(io))
        end
        BioformatsLoader.bf_import(path)[1]
    end
end

function __init__()
    DDMFramework.register_mime_type("image/nd2", readnd2)
    DDMFramework.add_plugin("migration", MigrationState)
end


export handle_update
end # module
