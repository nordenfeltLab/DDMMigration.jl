# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: julia4threads 1.6.0
#     language: julia
#     name: julia4threads-1.6
# ---

import Pkg
Pkg.activate(".")

using Revise
using DataFrames
using JLD2
using Chain
using Images
using JSON3
using Statistics

Pkg.activate("..")

using DDMMigration

# + tags=[]

path ="/tank/data/oscar/data/MicServer/old/Old server files/images/152/"
function get_sample_files()
    files = @chain path begin
        readdir
        filter(x -> isfile(path * x), _)
        split.(".")
        first.(_)
        sort(_, by = x -> x[end-3:end])
        unique
    end
    sample(x,n) = x[Int.(rand(1:length(x), n))]
    hashkeys = @chain files begin
        map(x -> x[1:end-3], _)
        unique
        sample(2)
    end


    let tmp = mapreduce(zip, hashkeys) do k
            filter(x -> occursin(k, x), files)
        end
        mapreduce(x -> [x[1], x[2]], vcat, tmp)
    end
end

function generate_state(file_iter)
    state = MigrationState(Dict{String, Any}(""=> 1))

    for f in file_iter
        params = path * f * ".jld2" |> load
        image = path * f * ".tif" |> load

        let p = Dict(
                "image" => image,
                "params" => params
            )
            DDMMigration.handle_update(state, p)
        end
    end
    return state
end
# -

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

function column_transform_macro_expr(expr)
    out_col = expr.args[1]
    right = expr.args[2]
    right.head == :call || throw(ArgumentError("Can only parse function calls"))
    fun = right.args[1]
    argset = Set{Symbol}()
    new_args = Any[fun]
    for e in right.args[2:end]
        if e isa QuoteNode
            push!(argset, e.value)
            push!(new_args, e.value)
        else
            push!(new_args, e)
        end
    end
    lambda = Expr(:(->), Expr(:tuple, argset...), Expr(:call, new_args...))
    col_vect = Expr(:vect, String.(argset)...)
    :($(String(out_col)) => ($col_vect => $lambda))
end
macro column_transform(expr)
    if expr.head == :(=)
        Expr(:call, :Dict, column_transform_macro_expr(expr))
    elseif expr.head == :block
        exprs = (column_transform_macro_expr(e) for e in expr.args if e isa Expr)
        Expr(:call, :Dict, exprs...)
    else
        throw(ArgumentError("Unable to parse"))
    end
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

# +
dtype(t, args...) = Dict("tipo" => t, args...)
fields(pairs...) = Dict(p[1] => p[2] for p in pairs)
returns(f, state) = (args...) -> f(state)

schema = Dict(
    "query" => "Query",
    "Query" => fields(
        "migration" => "Migration"
    ),
    "Migration" => fields(
        "centroid_x" => "Column",
        "centroid_y" => "Column",
        "mean_speed" => "Column"
    ),
    "Column" => fields(
        "name" => "String"
    )
)

resolvers(state) = Dict(
    "Query" => Dict(
        "migration" => (parent, args) -> collect_state(state, args)
    )
)

function resolve_field(parent, field, args, dtype, schema, resolvers)
    parent = if haskey(resolvers, dtype) && haskey(resolvers[dtype], field)
        resolvers[dtype][field](parent, args)
    else
        parent[field]
    end
    parent, schema[dtype][field]
end

function resolve_query(query, parent, dtype, schema, resolvers)
    (parent, dtype) = resolve_field(parent, query.field, query.args, dtype, schema, resolvers)
    if isempty(query.subquery)
        parent
    else
        execute_query(query.subquery, schema, resolvers; parent, dtype)
    end
end

function execute_query(query, schema, resolvers; parent=nothing, dtype=schema["query"])
    map(query) do q
        q.field => resolve_query(q, parent, dtype, schema, resolvers)
    end |> Dict
end

struct Query
    field::String
    args::Vector
    subquery::Vector{Query}
end

Query(name, queries...; args=[]) = Query(name, args, collect(queries))

# -

"""
{
    table(
        name: "migration"
        filter: {
            speed: [{
                op: "bin"
                args: [1,10]
                }]
        }
    )
    {
        position_x: column(name: "centroid_x")
        position_y: column(name: "centroid_y")
    }
}
"""


q = [Query("migration", Query("centroid_x"), Query("centroid_y"); args=["filter" => Dict()])]
println(JSON3.write(execute_query(q, schema, resolvers(state))))
