
function snr_binarize(img; sigma = 4,kwargs...)
    m,s = sigma_clipped_stats(img)
    
    img .> m + s*sigma
end

otsu_close_seg(img) = closing(img .> otsu_threshold(img))

segmentation_lib = Dict(
    "sigma_clipped" => (x;kwargs...) -> snr_binarize(x;kwargs...),
    "otsu_close" => (x;kwargs...) -> otsu_close_seg(x)
    )

    
function regionprop_analysis(img;method="sigma_clipped",minsize=150, maxsize=2000, kwargs...)
    seg = segmentation_lib[method](img;kwargs...) |>
        label_components
    seg = sparse(seg)
    counts = countmap(nonzeros(seg))
    for (i, j, v) in zip(findnz(seg)...)
        if counts[v] < minsize || counts[v] > maxsize
            seg[i,j] = 0
        end
    end
    dropzeros!(seg)
    ((;r...) for r in regionprops(img, seg; selected=unique(nonzeros(seg))))
end