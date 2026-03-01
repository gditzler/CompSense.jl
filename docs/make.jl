using Documenter
using CompSense

makedocs(
    sitename = "CompSense.jl",
    modules = [CompSense],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://gditzler.github.io/CompSense.jl",
    ),
    checkdocs = :exports,
    warnonly = [:missing_docs],
    pages = [
        "Home" => "index.md",
        "Algorithms" => "algorithms.md",
        "Sensing Matrices" => "sensing.md",
        "Basis / Dictionary" => "basis.md",
        "Utilities" => "utilities.md",
        "Metrics" => "metrics.md",
        "Analysis" => "analysis.md",
    ],
)

deploydocs(
    repo = "github.com/gditzler/CompSense.jl.git",
    devbranch = "main",
)
