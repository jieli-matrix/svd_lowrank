using lowranksvd
using Documenter

DocMeta.setdocmeta!(lowranksvd, :DocTestSetup, :(using lowranksvd); recursive=true)

makedocs(;
    modules=[lowranksvd],
    authors="jieli-matrix <li_j20@fudan.edu.cn> and contributors",
    repo="https://github.com/jieli-matrix/lowranksvd.jl/blob/{commit}{path}#{line}",
    sitename="lowranksvd.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jieli-matrix.github.io/lowranksvd.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jieli-matrix/lowranksvd.jl",
)
