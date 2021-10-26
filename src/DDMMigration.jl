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
include("ImageUtils.jl")



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

function handle_update(state::MigrationState, data)
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



# sample_df
# save_DDA_results
# results_to_dict
# shift_to_centre_coords


function get_results(state::MigrationState, requested)
    response = prepare_results(state, requested)
    if isnothing(response)
        json(Dict(:response => false))
    else
        json(response)
    end
end

#filter_parser = Dict(
#    ".>" => .>,
#    ".<" => .<,
#    "&&" => &&,
#    "|" => |,
#)



function prepare_results(state::MigrationState, requested)
    n = requested["n"]
    
    props = get_latest_props(state.fov_array, 10)
    df = join(state.tracks, props, on = [:fov_id, :t, :label_id])
    
    speed_df = calculate_speed_stats(df)
    speed_df = by(speed_df, :fov_id) do fovdf
        filter(r -> r.t .== maximum(fovdf.t), fovdf)
    end
    
    speed_df[!,:speed_group] = discretize(speed_df.speed_mean, 10)
    
    #speed_groups = [1,10]
    selected_groups = [filter(x -> x.speed_group == n, speed_df) for n in speed_groups]
    
    if all(map(g -> size(g,1) >= n, selected_groups))
        results = vcat(map(g -> sample_df(g,n), selected_groups)...)
        results[!, :stage_pos_x] = (i -> state.fov_arr[i].x).(results.fov_id)
        results[!, :stage_pos_y] = (i -> state.fov_arr[i].y).(results.fov_id)
        #save???
        #response = result_to_dict
        #response[:response] = true
        response
    end
end

export MigrationState, handle_update
end # module
