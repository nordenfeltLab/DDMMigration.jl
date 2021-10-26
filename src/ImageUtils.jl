

function snr_binarize(img; sigma = 4)
    m,s = sigma_clipped_stats(img)
    
    img .> m + s*sigma
end


function organize_images(images, params)
    Dict(Symbol(p["definition"]) => @view(images[:,:,p["index"]]) for p in params["channels"])
end

function regionprop_analysis(img, minsize=150, maxsize=2000, n_sigma=3.0)
    seg =
        snr_binarize(img, sigma = n_sigma) |> 
        label_components |> 
        sparse
    
    counts = countmap(nonzeros(seg))
    for (i, j, v) in zip(findnz(seg)...)
        if counts[v] < minsize || counts[v] > maxsize
            seg[i,j] = 0
        end
    end
    dropzeros!(seg)
    ((;r...) for r in regionprops(img, seg; selected=unique(nonzeros(seg))))
end