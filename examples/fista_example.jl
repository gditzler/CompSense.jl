# FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) Example
#
# FISTA solves the LASSO problem:
#   min_x  (1/2)||Ax - b||_2^2 + λ||x||_1
#
# It's an accelerated proximal gradient method with O(1/k²) convergence.

using CompSense
using LinearAlgebra
using Random

Random.seed!(456)

println("=" ^ 60)
println("FISTA (Fast Iterative Shrinkage-Thresholding) Example")
println("=" ^ 60)

# ============================================================================
# Basic Usage
# ============================================================================
println("\n1. Basic Usage")
println("-" ^ 40)

# Create a sparse recovery problem
A, x_true, b = gaussian_sensing(60, 200, 10)

# Recover using FISTA
x_fista = FISTA(A, b; lambda=0.1)

println("True signal:      $(count(!iszero, x_true)) non-zeros")
println("Recovered signal: $(count(xi -> abs(xi) > 0.01, x_fista)) non-zeros (threshold=0.01)")
println("Relative error:   $(round(norm(x_fista - x_true) / norm(x_true) * 100, digits=2))%")

# ============================================================================
# Effect of Lambda (Regularization Parameter)
# ============================================================================
println("\n2. Effect of λ (Regularization Parameter)")
println("-" ^ 40)

# Lambda controls the sparsity-accuracy tradeoff
println("λ        | Non-zeros | Rel. Error")
println("-" ^ 40)
for λ in [0.01, 0.05, 0.1, 0.5, 1.0]
    x_rec = FISTA(A, b; lambda=λ, maxiter=500)
    nnz = count(xi -> abs(xi) > 0.01, x_rec)
    err = norm(x_rec - x_true) / norm(x_true)
    println("$(lpad(λ, 7)) |    $(lpad(nnz, 3))    |   $(round(err * 100, digits=2))%")
end

println("\n→ Smaller λ: denser solution, better fit")
println("→ Larger λ:  sparser solution, more regularization")

# ============================================================================
# Convergence Behavior
# ============================================================================
println("\n3. Effect of Maximum Iterations")
println("-" ^ 40)

for maxiter in [50, 100, 200, 500, 1000]
    x_rec = FISTA(A, b; lambda=0.1, maxiter=maxiter)
    err = norm(x_rec - x_true) / norm(x_true)
    println("  maxiter=$(lpad(maxiter, 4)): error = $(round(err * 100, digits=2))%")
end

# ============================================================================
# Comparison with Exact Sparsity Methods
# ============================================================================
println("\n4. FISTA vs Greedy Methods (sparsity=10)")
println("-" ^ 40)

# FISTA doesn't enforce exact sparsity, but we can compare
x_omp = OMP(A, b; sparsity=10)
x_iht = IHT(A, b; sparsity=10)
x_fista_tuned = FISTA(A, b; lambda=0.15, maxiter=1000)

algorithms = [
    ("OMP (k=10)", x_omp),
    ("IHT (k=10)", x_iht),
    ("FISTA (λ=0.15)", x_fista_tuned),
]

for (name, x_rec) in algorithms
    nnz = count(xi -> abs(xi) > 0.01, x_rec)
    err = norm(x_rec - x_true) / norm(x_true)
    println("$(rpad(name, 18)): $(lpad(nnz, 2)) non-zeros, error = $(round(err * 100, digits=2))%")
end

# ============================================================================
# Noisy Measurements
# ============================================================================
println("\n5. Recovery from Noisy Measurements")
println("-" ^ 40)

A_clean, x_clean, b_clean = gaussian_sensing(80, 200, 10)

println("Noise σ | Optimal λ | Rel. Error")
println("-" ^ 40)
for σ in [0.0, 0.01, 0.05, 0.1]
    b_noisy = b_clean + σ * randn(length(b_clean))

    # For noisy data, larger λ often works better
    # This is a simple heuristic; cross-validation is better in practice
    λ_optimal = max(0.1, σ * 2)

    x_rec = FISTA(A_clean, b_noisy; lambda=λ_optimal, maxiter=500)
    err = norm(x_rec - x_clean) / norm(x_clean)
    println("  $(lpad(σ, 4))   |   $(lpad(λ_optimal, 4))    |   $(round(err * 100, digits=2))%")
end

println("\n✓ FISTA example complete!")
println("\nKey takeaways:")
println("  • FISTA solves the LASSO (L1-regularized least squares)")
println("  • λ controls sparsity: larger λ → sparser solution")
println("  • O(1/k²) convergence rate (faster than ISTA)")
println("  • Soft thresholding produces approximate sparsity")
println("  • Robust to noise with proper λ tuning")
