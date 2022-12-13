distcols = [:centroid_x,:centroid_y]

function safe_extrema(arr)
    foldl(arr, init=(10000, 0)) do (xmin, xmax), x
        (min(x, xmin), max(x,xmax))
    end
end

function track_blobs(positions, blocking::Float64=10000.0; maxdist::Float64=20.0)
    curind = size(positions[1], 2)
    ind = [i for i in 1:curind]
    inds = [[(1, i)] for i in 1:curind]
    for t = 2:length(positions)
        prev = positions[t-1]
        cur = positions[t]
        n = size(prev, 2)
        m = size(cur, 2)
        cost_mat = PseudoBlockArray{Float64}(undef, [n, m], [m, n])
        cost_mat .= blocking
        top_left = @view cost_mat[Block(1,1)]
        pairwise!(top_left, Euclidean(), prev, cur)
        cmin, cmax = safe_extrema(top_left)
        calt = cmax * 1.05
        bottom_right = @view cost_mat[Block(2,2)]
        for i in CartesianIndices(top_left)
            @inbounds if top_left[i] > maxdist
                top_left[i] = blocking
            else
                bottom_right[i.I[2],i.I[1]] = cmin
            end
        end
        top_right = @view cost_mat[Block(1,2)]
        top_right[diagind(top_right)] .= calt
        
        bot_left = @view cost_mat[Block(2,1)]
        bot_left[diagind(bot_left)] .= calt
        
        matching = Hungarian.munkres(cost_mat)
        is_star(x) = x == Hungarian.STAR
        row_ind = Int64[findfirst(is_star, r) for r in eachrow(matching)]
        indnxt = [-1 for _ in 1:m]
        for (r, c) in enumerate(row_ind[1:n])
            if c <= m
                push!(inds[ind[r]], (t, c))
                indnxt[c] = ind[r]
            end
        end
        for (r, c) in enumerate(row_ind[n+1:end])
            if c <= m
                push!(inds, [(t, c)])
                indnxt[c] = length(inds)
            end
        end
        ind = indnxt
    end
    return inds
end


function track(fov::FOV, fov_id, range, label_mapping = Dict(), start_id = 0,maxdist=20.0)
    trackdf = DataFrame()
    tracks = track_blobs([Matrix(props[:,distcols])' for props in fov.data[range]],maxdist=maxdist)
    curr_t(t) = t+first(range)-1
    get_label(t,c) = fov.data[curr_t(t)].label_id[c]
    
    for arr in tracks
        t, c = arr[1]
        label = get_label(t,c)
        track_id, new_points = if t == 1 && label ∈ keys(label_mapping)
            label_mapping[label], arr[2:end]
        else
            start_id += 1    
            start_id, arr
        end
          
        new_tracks = map(new_points) do (t, c)
            label = get_label(t,c) 
            (; fov_id, track_id, label_id = label, t = curr_t(t))
        end
        
        append!(trackdf, new_tracks)

    end
    @info "Tracked $(nrow(trackdf)) cells"
    trackdf
end

function get_latest_props(fov_arr::Vector{FOV}, n)
    props = DataFrame()
    
    for (fov_id, fov) in enumerate(fov_arr)
        for (t, prop) in taketail(collect(enumerate(fov.data)), n)
            df = copy(fov.data[t])
            df[!, :fov_id] .= fov_id
            df[!, :t] .= t
            append!(props, df)
        end
    end
    return props
end

index = [:fov_id, :track_id]
function calculate_speed_stats(df)
    
    diff_df = combine(groupby(df, index)) do gdf
        DataFrame(
            Any[
                gdf.t[2:end],
                gdf.label_id[2:end],
                (diff(df[!,c]) for c in distcols)...
                ],
            [
                :t,
                :label_id,
                (Symbol("diff_" * string(d)) for d in distcols)...
                ]
            )
    end
    
    speed_df = select(
        diff_df,
        [:diff_centroid, :diff_centroid_y] => ((x,y) -> hypot.(x,y)) => :speed, [index..., :t, :label_id]
    )
    object_index = vcat(index, :t)
    df = innerjoin(speed_df, df[:, [:centroid_x, :centroid_y, object_index...]], on = object_index)
    stats_df = combine(groupby(df, [:fov_id, :track_id]),
        :speed => mean,
        :centroid_x => last => :centroid_x,
        :centroid_y => last => :centroid_y,
        :t => last => :t)
end
            
vecbin(hist) = x -> StatsBase.binindex(hist, x)
function discretize(arr, n)
    hist = fit(Histogram, arr, quantile(arr, Linrange(0.0,1.0, n+1)[1:n]))
    vecbin(hist).(arr)
end
