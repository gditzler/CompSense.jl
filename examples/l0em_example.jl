# L0EM (L0 Expectation-Maximization) Example
#
# L0EM uses an EM framework to solve the L0-regularized problem:
#   min_x ||Ax - b||_2^2 + λ ||x||_0

using CompSense
using LinearAlgebra
using Random

Random.seed!(404)

println("=" ^ 60)
println("L0EM (L0 Expectation-Maximization) Example")
println("=" ^ 60)

# ============================================================================
# Basic Usage
# ============================================================================
println("\n1. Basic Usage")
println("-" ^ 40)

# Create a sparse recovery problem
A, x_true, b = gaussian_sensing(50, 200, 8)

# Recover using L0EM
x_l0em = L0EM(A, b)

println("True signal:      $(count(!iszero, x_true)) non-zeros")
println("Recovered signal: $(count(xi -> abs(xi) > 0.001, x_l0em)) non-zeros (threshold=0.001)")
println("Relative error:   $(round(norm(x_l0em - x_true) / norm(x_true) * 100, digits=2))%")

# ============================================================================
# Algorithm Overview
# ============================================================================
println("\n2. Understanding L0EM")
println("-" ^ 40)
println("""
L0EM iteratively reweights the least squares problem:
  1. Initialize: θ = (A'A + λI)⁻¹ A'b
  2. E-step: η = θ (store current estimate)
  3. M-step: Update weights based on η²
  4. Solve weighted least squares for new θ
  5. Repeat until convergence

Key insight: Entries with small magnitude get down-weighted,
promoting sparsity through the reweighting scheme.
""")

# ============================================================================
# Effect of Lambda (Regularization)
# ============================================================================
println("3. Effect of λ (Regularization Parameter)")
println("-" ^ 40)

println("λ        | Non-zeros | Rel. Error")
println("-" ^ 40)
for λ in [1e-5, 1e-4, 1e-3, 1e-2, 0.1]
    x_rec = L0EM(A, b; lambda=λ)
    nnz = count(xi -> abs(xi) > 0.001, x_rec)
    err = norm(x_rec - x_true) / norm(x_true)
    println("$(lpad(λ, 7))  |    $(lpad(nnz, 2))     |   $(round(err * 100, digits=2))%")
end

println("\n→ Smaller λ: less regularization, potentially overfitting")
println("→ Larger λ: more regularization, sparser solutions")

# ============================================================================
# Convergence Behavior
# ============================================================================
println("\n4. Effect of Maximum Iterations")
println("-" ^ 40)

println("maxiter | Non-zeros | Rel. Error")
println("-" ^ 40)
for maxiter in [5, 10, 20, 50, 100]
    x_rec = L0EM(A, b; maxiter=maxiter)
    nnz = count(xi -> abs(xi) > 0.001, x_rec)
    err = norm(x_rec - x_true) / norm(x_true)
    println("$(lpad(maxiter, 6))  |    $(lpad(nnz, 2))     |   $(round(err * 100, digits=2))%")
end

println("\n→ L0EM often converges quickly!")

# ============================================================================
# Epsilon (Convergence Threshold)
# ============================================================================
println("\n5. Effect of Epsilon (Convergence Threshold)")
println("-" ^ 40)

println("epsilon | Non-zeros | Rel. Error")
println("-" ^ 40)
for eps in [1e-6, 1e-4, 1e-3, 1e-2, 0.1]
    x_rec = L0EM(A, b; epsilon=eps)
    nnz = count(xi -> abs(xi) > eps, x_rec)
    err = norm(x_rec - x_true) / norm(x_true)
    println("$(lpad(eps, 6))  |    $(lpad(nnz, 2))     |   $(round(err * 100, digits=2))%")
end

# ============================================================================
# L0EM vs SL0 vs FISTA
# ============================================================================
println("\n6. L0EM vs SL0 vs FISTA")
println("-" ^ 40)

x_sl0 = SL0(A, b)
x_fista = FISTA(A, b; lambda=0.1)

algorithms = [
    ("L0EM", x_l0em),
    ("SL0", x_sl0),
    ("FISTA", x_fista),
]

println("Algorithm | Non-zeros | Rel. Error | ||Ax - b||")
println("-" ^ 50)
for (name, x_rec) in algorithms
    nnz = count(xi -> abs(xi) > 0.01, x_rec)
    err = norm(x_rec - x_true) / norm(x_true)
    res = norm(A * x_rec - b)
    println("$(rpad(name, 9)) |    $(lpad(nnz, 2))     |   $(lpad(round(err * 100, digits=2), 6))%  |   $(round(res, digits=6))")
end

# ============================================================================
# Multiple Random Trials
# ============================================================================
println("\n7. Performance Across Random Problems")
println("-" ^ 40)

n_trials = 10
l0em_errors = Float64[]
sl0_errors = Float64[]

for _ in 1:n_trials
    A_trial, x_trial, b_trial = gaussian_sensing(50, 200, 8)

    x_l0em_trial = L0EM(A_trial, b_trial)
    x_sl0_trial = SL0(A_trial, b_trial)

    push!(l0em_errors, norm(x_l0em_trial - x_trial) / norm(x_trial))
    push!(sl0_errors, norm(x_sl0_trial - x_trial) / norm(x_trial))
end

println("Average relative error over $n_trials trials:")
println("  L0EM: $(round(mean(l0em_errors) * 100, digits=2))% ± $(round(std(l0em_errors) * 100, digits=2))%")
println("  SL0:  $(round(mean(sl0_errors) * 100, digits=2))% ± $(round(std(sl0_errors) * 100, digits=2))%")

# Helper function (using Statistics would be cleaner)
mean(x) = sum(x) / length(x)
std(x) = sqrt(sum((x .- mean(x)).^2) / length(x))

# ============================================================================
# Noisy Measurements
# ============================================================================
println("\n8. Recovery from Noisy Measurements")
println("-" ^ 40)

A_clean, x_clean, b_clean = gaussian_sensing(60, 200, 8)

for σ in [0.0, 0.01, 0.05, 0.1]
    b_noisy = b_clean + σ * randn(length(b_clean))

    # For noisy problems, slightly larger lambda can help
    λ = σ > 0 ? 0.01 : 0.001

    x_rec = L0EM(A_clean, b_noisy; lambda=λ)
    err = norm(x_rec - x_clean) / norm(x_clean)
    println("  σ=$(lpad(σ, 4)), λ=$(lpad(λ, 5)): error = $(round(err * 100, digits=2))%")
end

println("\n✓ L0EM example complete!")
println("\nKey takeaways:")
println("  • L0EM directly optimizes L0-regularized objective")
println("  • Uses EM framework with iterative reweighting")
println("  • λ controls sparsity vs data fidelity tradeoff")
println("  • Typically converges in few iterations")
println("  • Good balance between speed and accuracy")
