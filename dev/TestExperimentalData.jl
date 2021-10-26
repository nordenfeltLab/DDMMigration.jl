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

]activate .

using Revise
using DataFrames
using JLD2
using Chain
using Images
using Diana

]activate ../..

using DDMMigration

# + tags=[]
path ="../../../../storage/MicServer/old/Old server files/images/152/"
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


file_iter = let
    tmp = mapreduce(zip, hashkeys) do k
        filter(x -> occursin(k, x), files)
    end
    mapreduce(x -> [x[1], x[2]],vcat, tmp)
end
# -

params["channels"][2]

params["system_parameters"]

params["experiment_parameters"]

using DDMMigration

# +
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
# -

function collect_state(state::MigrationState)
    objects = DataFrame()
    for (fov_id, fov) in enumerate(state.fov_arr)
        for (t, prop) in enumerate(fov.data)
            df = copy(prop)
            df[!, :fov_id] .= fov_id
            df[!, :t] .= t
            append!(objects, df)
        end
    end
    
    table = innerjoin(state.tracks, objects, on = [:fov_id, :t, :label_id])
    
    Dict("data" => table)
end

# +
dtype(t, args...) = Dict("tipo" => t, args...)
fields(pairs...) = Dict(p[1] => dtype(p[2]) for p in pairs)
returns(f, state) = (args...) -> f(state)
inspect(args...) = (display(args); args)

schema = Dict(
    "query" => "Query",
    "Query" => "Table",
    "Table" => fields(
        "migration" => "Migration"
    ),
    "Migration" => fields(
        "speed" => "Column"
    ),
    "Column" => fields(
        "name" => "String"
    )
)

display(schema)

struct TableDict2 <: AbstractDict{String,Any}
end

Base.getindex(td::TableDict2, key::String) = ((root, args, ctx, info) -> inspect((root,args,ctx)))
Base.haskey(td::TableDict2, key::String) = true

resolvers = Dict(
    "Table" => Dict(
        "migration" => returns(collect_state, state)
    ),
    "Migration" => Dict(
        "speed" => (args...) -> (display(args); Dict("name" => "speed"))
    ),
    "Column" => Dict(
        "name" => ((args...) -> (display(args); "hello"))
    )
)
# -



Schema(schema, resolvers).execute("""
    {
        migration{
            speed{
                name
            }
        }
    }
""")

Schema(schema, resolvers).execute(""" { migration } """)

# +
schema = Dict(
"query" => "Query"

,"Query"=> fields(
    "migration"=>"MigrationData"
   )

,"MigrationData" => fields(
   "speed"=>"Column"
  )
,"Column" => fields(
   "name"=>"String"
  )   
)



 resolvers=Dict(
    "Query"=>Dict(
        "migration" => (root,args,ctx,info)->(Dict("speed"=>"Diana"))
    )
    ,"MigrationData"=>Dict(
      "edad" => (root,args,ctx,info)->(root["edad"])
    )
    ,"Column"=>Dict(
      "name" => (root,args,ctx,info)-> (display(root); "str'ng")
    )
)
display(resolvers)
Schema(schema, resolvers).execute("""
{
    migration{
        speed{
            name
        }

    }
}
""") |> display
Schema(schema, resolvers).execute(""" { migration { speed { name } } } """)
# -

{
    table(
        name: "migration"
        filter: {
            speed: {
                op: "bin"
                args: [1,10]
            }
        }
    )
    {
        position_x: column(name: "centroid_x")
        position_y: column(name: "centroid_y")
    }
}

# + jupyter={"outputs_hidden": true} tags=[]
state.
# -


