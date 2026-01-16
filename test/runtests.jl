using Test
using CompSense
using LinearAlgebra
using Random

# Set seed for reproducibility
Random.seed!(42)

@testset "CompSense.jl" begin
    include("test_utils.jl")
    include("test_sl0.jl")
    include("test_l0em.jl")
    include("test_irwls.jl")
end
