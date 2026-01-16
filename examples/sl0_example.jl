# Smoothed L0 (SL0) Example
#
# SL0 approximates the L0 norm with a smooth Gaussian function and
# uses gradient ascent to maximize sparsity while satisfying Ax = b.

using CompSense
using LinearAlgebra
using Random

Random.seed!(202)

println("=" ^ 60)
println("Smoothed L0 (SL0) Example")
println("=" ^ 60)

# ============================================================================
# Basic Usage
# ============================================================================
println("\n1. Basic Usage")
println("-" ^ 40)

# Create a sparse recovery problem
A, x_true, b = gaussian_sensing(50, 200, 8)

# Recover using SL0
x_sl0 = SL0(A, b)

println("True signal:      $(count(!iszero, x_true)) non-zeros")
println("Recovered signal: $(count(xi -> abs(xi) > 0.001, x_sl0)) non-zeros (threshold=0.001)")
println("Relative error:   $(round(norm(x_sl0 - x_true) / norm(x_true) * 100, digits=2))%")

# ============================================================================
# Algorithm Overview
# ============================================================================
println("\n2. Understanding SL0")
println("-" ^ 40)
println("""
SL0 approximates ||x||_0 with a smooth function:
  F_σ(x) = Σ exp(-x_i²/σ²)

For small σ, F_σ(x) ≈ n - ||x||_0

The algorithm:
  1. Start with minimum norm solution: x = A⁺b
  2. Set initial σ = 2 * max|x|
  3. Repeat:
     a. Gradient ascent on F_σ (promotes sparsity)
     b. Project back to feasible set {x : Ax = b}
     c. Decrease σ (σ ← σ * decrease_factor)

Key: Gradual σ decrease avoids local minima!
""")

# ============================================================================
# Effect of Sigma Decrease Factor
# ============================================================================
println("3. Effect of Sigma Decrease Factor")
println("-" ^ 40)

println("Decrease Factor | Non-zeros | Rel. Error")
println("-" ^ 45)
for factor in [0.99, 0.95, 0.9, 0.85, 0.7, 0.5]
    x_rec = SL0(A, b; sigma_decrease_factor=factor, maxiter=200)
    nnz = count(xi -> abs(xi) > 0.001, x_rec)
    err = norm(x_rec - x_true) / norm(x_true)
    println("     $(lpad(factor, 4))        |    $(lpad(nnz, 2))     |   $(round(err * 100, digits=2))%")
end

println("\n→ Slower decrease (larger factor): better accuracy, more iterations")
println("→ Faster decrease (smaller factor): faster, but may miss optimum")

# ============================================================================
# Effect of Maximum Iterations
# ============================================================================
println("\n4. Effect of Maximum Iterations")
println("-" ^ 40)

println("maxiter | Non-zeros | Rel. Error")
println("-" ^ 40)
for maxiter in [10, 25, 50, 100, 150, 200, 300]
    x_rec = SL0(A, b; maxiter=maxiter)
    nnz = count(xi -> abs(xi) > 0.001, x_rec)
    err = norm(x_rec - x_true) / norm(x_true)
    println("$(lpad(maxiter, 6))  |    $(lpad(nnz, 2))     |   $(round(err * 100, digits=2))%")
end

# ============================================================================
# Epsilon Threshold
# ============================================================================
println("\n5. Effect of Epsilon (Zero Threshold)")
println("-" ^ 40)

x_base = SL0(A, b; maxiter=200)

println("Epsilon  | Non-zeros | Rel. Error")
println("-" ^ 40)
for eps in [1e-6, 1e-4, 1e-3, 1e-2, 0.1]
    x_rec = SL0(A, b; maxiter=200, epsilon=eps)
    nnz = count(xi -> abs(xi) > eps, x_rec)
    err = norm(x_rec - x_true) / norm(x_true)
    println("$(lpad(eps, 7))  |    $(lpad(nnz, 2))     |   $(round(err * 100, digits=2))%")
end

# ============================================================================
# Comparison with Optimization-Based Methods
# ============================================================================
println("\n6. SL0 vs L0EM vs FISTA")
println("-" ^ 40)

x_l0em = L0EM(A, b)
x_fista = FISTA(A, b; lambda=0.1)

algorithms = [
    ("SL0", x_sl0),
    ("L0EM", x_l0em),
    ("FISTA", x_fista),
]

println("Algorithm | Non-zeros | Rel. Error")
println("-" ^ 40)
for (name, x_rec) in algorithms
    nnz = count(xi -> abs(xi) > 0.01, x_rec)
    err = norm(x_rec - x_true) / norm(x_true)
    println("$(rpad(name, 9)) |    $(lpad(nnz, 2))     |   $(round(err * 100, digits=2))%")
end

# ============================================================================
# Different Sensing Matrices
# ============================================================================
println("\n7. SL0 with Different Sensing Matrices")
println("-" ^ 40)

Random.seed!(303)
n, p, k = 50, 200, 8

println("Matrix Type  | Non-zeros | Rel. Error | Feasibility ||Ax-b||")
println("-" ^ 60)

matrix_types = [
    ("Gaussian", gaussian_sensing(n, p, k)),
    ("Bernoulli", bernoulli_sensing(n, p, k)),
    ("DCT", dct_sensing(n, p, k)),
    ("Uniform", uniform_sensing(n, p, k)),
    ("Toeplitz", toeplitz_sensing(n, p, k)),
]

for (name, (A_mat, x_mat, b_mat)) in matrix_types
    x_rec = SL0(A_mat, b_mat; maxiter=200)
    nnz = count(xi -> abs(xi) > 0.01, x_rec)
    err = norm(x_rec - x_mat) / norm(x_mat)
    feas = norm(A_mat * x_rec - b_mat)
    println("$(rpad(name, 12)) |    $(lpad(nnz, 2))     |   $(lpad(round(err * 100, digits=2), 6))%  |   $(round(feas, digits=6))")
end

println("\n✓ SL0 example complete!")
println("\nKey takeaways:")
println("  • SL0 is a fast gradient-based method")
println("  • Approximates L0 with smooth Gaussian function")
println("  • sigma_decrease_factor controls speed vs accuracy")
println("  • Works well with various sensing matrix types")
println("  • Projects to feasible set to satisfy Ax = b")
