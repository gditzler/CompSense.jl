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
    include("test_omp.jl")
    include("test_fista.jl")
    include("test_iht.jl")
    include("test_cosamp.jl")
    include("test_akron.jl")
    include("test_kron.jl")
end
