module DDMMigration

using Images
using SparseArrays
using BlockArrays
using Distances
using LinearAlgebra
using Hungarian
using SegmentationUtils
using RegionProps
using StatsBase
using DataFrames
using DDMFramework

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
        )
    )
end

function DDMFramework.handle_update(state::MigrationState, data)
    update_state!(state, data["image"], data["params"])
end

function get_fov!(state::MigrationState, params)
    id = get_fov_id!(state, params["stage_pos_x"], params["stage_pos_y"])
    id, state.fov_arr[id]
end

function update_state!(state::MigrationState, image, params; shading = true)
    fov_id, fov = get_fov!(state, params["experiment_parameters"])
    image = organize_images(image, params)
    props = regionprop_analysis(image[:Nuclei]) |> DataFrame
    
    props = vcat(DataFrame(
            label_id = Int[],
            centroid_x = Float64[],
            centroid_y = Float64[]
            ),
        props,
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
     end
)

function collect_state(state::MigrationState, args)
    args = Dict(args)
    objects = DataFrame()
    for (fov_id, fov) in enumerate(state.fov_arr)
        for (t, prop) in enumerate(fov.data)
            df = copy(prop)
            df[!, :fov_id] .= fov_id
            df[!, :t] .= t
            append!(objects, df)
        end
    end

    df = innerjoin(state.tracks, objects, on = [:fov_id, :t, :label_id])

    index = [:fov_id, :track_id]
    distcols = [:centroid_x,:centroid_y]

    @time data = map(gdf for gdf in groupby(df, index)) do gdf
        (gdf=gdf, centroid_x=diff(gdf.centroid_x), centroid_y=diff(gdf.centroid_y))
    end |> DataFrame


    data = LazyDF(
        data;
        mean_speed=[:centroid_x, :centroid_y] => (x, y) -> mean(hypot.(x, y)),
        last_x=[:centroid_x] => x -> x[end]
    )
    if haskey(args, "filter")
        filters = mapfoldl(vcat, args["filter"]; init=Pair{String, Base.Callable}[]) do (column, filters)
            map(filters) do filt
                op = table_filters[filt["op"]](data[!,column], filt["args"]...)
                column => op
            end
        end
        foreach(f -> filter!(f, data), filters)
    end
    data
end

schema = Dict(
    "query" => "Query",
    "Query" => Dict(
        "migration" => "Migration"
    ),
    "Migration" => Dict(
        "centroid_x" => "Column",
        "centroid_y" => "Column",
        "mean_speed" => "Column"
    ),
    "Column" => Dict(
        "name" => "String"
    )
)

resolvers(state) = Dict(
    "Query" => Dict(
        "migration" => (parent, args) -> collect_state(state, args)
    )
)

function DDMFramework.query_state(state::MigrationState, query)
    execute_query(q, schema, resolvers)
end

end # module
