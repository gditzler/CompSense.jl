using Test
using CompSense
using LinearAlgebra
using Random

# Set seed for reproducibility
Random.seed!(42)

@testset "CompSense.jl" begin
    include("test_utils.jl")
    include("test_noise.jl")
    include("test_metrics.jl")
    include("test_analysis.jl")
    include("test_sl0.jl")
    include("test_l0em.jl")
    include("test_irwls.jl")
    include("test_omp.jl")
    include("test_fista.jl")
    include("test_iht.jl")
    include("test_cosamp.jl")
    include("test_akron.jl")
    include("test_kron.jl")
    include("test_reweightedl1.jl")
    include("test_basispursuit.jl")
    include("test_bpdn.jl")
    include("test_lasso.jl")
    include("test_sp.jl")
    include("test_niht.jl")
    include("test_admm.jl")
    include("test_amp.jl")
    include("test_basis.jl")
    include("test_biht.jl")
    include("test_somp.jl")
    include("test_grouplasso.jl")
    include("test_svt.jl")
end
