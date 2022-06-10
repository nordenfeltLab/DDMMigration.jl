using Documenter
using DDMMigration

makedocs(
    sitename = "DDMMigration",
    format = Documenter.HTML(),
    modules = [DDMMigration],
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nordenfeltLab.github.io/DDMMigration.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Microscopes" => ["microscopes/nikon.md"]
    ]
)

deploydocs(;
    repo="github.com/nordenfeltLab/DDMMigration.jl",
    devbranch="main",
)
