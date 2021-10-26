function prepare_results(state::MigrationState, requested)
    props_df = get_latest_props(state.fov_array, 10)
    join_df = join(state.tracks, props_df, on=[:fov_id, :t, :label_id])
    
    stats_df = calculate_speed_stats(join_df)
    stats_df = by(stats_df, :fov_id) do fovdf
        filter(r -> r.t .== maximum(fovdf.t), fovdf)
    end
    
    stats_df[!, :speed_group] = discretize(stats_df.speed_mean, 10)
    
    selected_groups = [filter(x -> x.speed_group == n, stats_df) for n in [1,2,19,20]]
    
    n = requested["n"]
    if all(map(g -> size(g, 1) >= n, selected_groups))
        results = vcat(map(g -> sample_df(g, n), selected_groups)...)
        results[!, :stage_pos_x] = (i -> state.fov_array[i].x).(results.fov_id)
        results[!, :stage_pos_y] = (i -> state.fov_array[i].y).(results.fov_id)
        save_DDA_results(Dict("results" => results), requested["experiment_id"], requested)
        response = results_to_dict(shift_to_centre_coords(results, img_size))
        response[:response] = true
        response
    end
end

index = [:fov_id, :track_id]

taketail(x, n) = x[max(length(x)-n+1,1):end]

function get_latest_props(fov_array::Vector{FOV}, n)
    props_df = DataFrame()
    for (fov_id, fov) in enumerate(fov_array)
        for (t, prop) in taketail(collect(enumerate(fov.data)), n)
            df = copy(fov.data[t])
            df[!, :fov_id] .= fov_id
            df[!, :t] .= t
            append!(props_df, df)
        end
    end
    return props_df
end

function calculate_speed_stats(join_df)
    diff_df = combine(groupby(join_df, index)) do df
        DataFrame(Any[df.t[2:end], df.label_id[2:end], (diff(df[!, c]) for c in distcols)...], [:t, :label_id, (Symbol("diff_" * string(d)) for d in distcols)...])
    end
    
    df = select(diff_df, [:diff_centroid_x, :diff_centroid_y] => ((x,y) -> hypot.(x,y)) => :speed, [index..., :t, :label_id]);
    object_index = [:fov_id, :label_id, :t]
    df = innerjoin(df, join_df[:, [:centroid_x, :centroid_y, object_index...]], on=object_index)
    stats_df = combine(groupby(df, [:fov_id, :track_id]),
        :speed => mean,
        :centroid_x => last => :centroid_x,
        :centroid_y => last => :centroid_y,
        :t => last => :t)
end

vecbin(hist) = x -> StatsBase.binindex(hist, x)
function discretize(arr, n)
    hist = fit(Histogram, arr, quantile(arr, LinRange(0.0,1.0,n+1)[1:n]))
    vecbin(hist).(arr)
end