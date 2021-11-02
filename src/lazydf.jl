struct LazyDF
    df
    transforms
    selection
end

LazyDF(df; kwtf...) = LazyDF(df, Dict(String(k) => v for (k,v) in kwtf))
LazyDF(df, transforms) = LazyDF(df, transforms, trues(nrow(df)))
function Base.filter!(filt, df::LazyDF)
    df.selection[(!filt[2]).(df[filt[1]])] .= false
end

Base.getindex(t::LazyDF, ::typeof(!), col) = t[col]
function Base.getindex(t::LazyDF, col::String)
    if haskey(t.transforms, col)
        columns, op = t.transforms[col]
        map(op, (t.df[t.selection, c] for c in columns)...)
    else
        t.df[t.selection, col]
    end
end
