# Iterative Hard Thresholding (IHT) Example
#
# IHT solves the constrained problem:
#   min_x ||Ax - b||_2^2  subject to ||x||_0 ≤ k
#
# It alternates between gradient descent and hard thresholding.

using CompSense
using LinearAlgebra
using Random

Random.seed!(789)

println("=" ^ 60)
println("Iterative Hard Thresholding (IHT) Example")
println("=" ^ 60)

# ============================================================================
# Basic Usage
# ============================================================================
println("\n1. Basic Usage")
println("-" ^ 40)

# Create a sparse recovery problem
A, x_true, b = gaussian_sensing(50, 200, 8)

# Recover using IHT with known sparsity
x_iht = IHT(A, b; sparsity=8)

println("True signal:      $(count(!iszero, x_true)) non-zeros")
println("Recovered signal: $(count(!iszero, x_iht)) non-zeros")
println("Relative error:   $(round(norm(x_iht - x_true) / norm(x_true) * 100, digits=2))%")

# ============================================================================
# Exact Sparsity Guarantee
# ============================================================================
println("\n2. IHT Guarantees Exact Sparsity")
println("-" ^ 40)

# Unlike FISTA, IHT always returns exactly k non-zeros
for k in [4, 8, 12, 16, 20]
    x_rec = IHT(A, b; sparsity=k)
    nnz = count(!iszero, x_rec)
    @assert nnz == k "IHT should return exactly k non-zeros"
    err = norm(x_rec - x_true) / norm(x_true)
    println("  k=$(lpad(k, 2)): $(lpad(nnz, 2)) non-zeros, error = $(round(err * 100, digits=2))%")
end

# ============================================================================
# Step Size (μ) Effect
# ============================================================================
println("\n3. Effect of Step Size (μ)")
println("-" ^ 40)

# The default step size is 1/||A||_2^2
# Smaller step size = more stable but slower
# Larger step size = faster but may diverge

A_norm_sq = opnorm(A)^2
default_mu = 1.0 / A_norm_sq

println("Default μ = 1/||A||₂² ≈ $(round(default_mu, digits=4))")
println()

for scale in [0.5, 1.0, 1.5, 2.0]
    μ = scale * default_mu
    x_rec = IHT(A, b; sparsity=8, mu=μ, maxiter=500)
    err = norm(x_rec - x_true) / norm(x_true)
    converged = err < 0.5 ? "✓" : "✗"
    println("  μ=$(round(μ, digits=4)) ($(scale)×default): error = $(round(err * 100, digits=2))% $converged")
end

# ============================================================================
# Convergence Speed
# ============================================================================
println("\n4. Convergence Over Iterations")
println("-" ^ 40)

println("maxiter | Non-zeros | Rel. Error")
println("-" ^ 40)
for maxiter in [10, 25, 50, 100, 200, 500]
    x_rec = IHT(A, b; sparsity=8, maxiter=maxiter)
    nnz = count(!iszero, x_rec)
    err = norm(x_rec - x_true) / norm(x_true)
    println("$(lpad(maxiter, 6))  |    $(lpad(nnz, 2))     |   $(round(err * 100, digits=2))%")
end

# ============================================================================
# Comparison: IHT vs OMP vs CoSaMP
# ============================================================================
println("\n5. IHT vs Other Sparsity-Constrained Methods")
println("-" ^ 40)

A2, x_true2, b2 = gaussian_sensing(60, 200, 10)

x_omp = OMP(A2, b2; sparsity=10)
x_iht = IHT(A2, b2; sparsity=10)
x_cosamp = CoSaMP(A2, b2; sparsity=10)

algorithms = [
    ("OMP", x_omp),
    ("IHT", x_iht),
    ("CoSaMP", x_cosamp),
]

println("Algorithm | Non-zeros | Rel. Error | Residual")
println("-" ^ 55)
for (name, x_rec) in algorithms
    nnz = count(!iszero, x_rec)
    err = norm(x_rec - x_true2) / norm(x_true2)
    res = norm(A2 * x_rec - b2)
    println("$(rpad(name, 9)) |    $(lpad(nnz, 2))     |   $(lpad(round(err * 100, digits=2), 6))%  |   $(round(res, digits=4))")
end

# ============================================================================
# Noisy Measurements
# ============================================================================
println("\n6. Recovery from Noisy Measurements")
println("-" ^ 40)

A_clean, x_clean, b_clean = gaussian_sensing(70, 200, 8)

for σ in [0.0, 0.01, 0.05, 0.1]
    b_noisy = b_clean + σ * randn(length(b_clean))
    x_rec = IHT(A_clean, b_noisy; sparsity=8, maxiter=500)
    err = norm(x_rec - x_clean) / norm(x_clean)
    println("  σ=$(lpad(σ, 4)): error = $(round(err * 100, digits=2))%")
end

println("\n✓ IHT example complete!")
println("\nKey takeaways:")
println("  • IHT enforces exact sparsity (always k non-zeros)")
println("  • Simple: gradient step + keep k largest entries")
println("  • Step size μ affects convergence (default: 1/||A||₂²)")
println("  • May need more iterations than OMP for same accuracy")
println("  • Works well when columns of A are nearly orthogonal")
