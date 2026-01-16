# CoSaMP (Compressive Sampling Matching Pursuit) Example
#
# CoSaMP combines greedy pursuit with subspace projection. It has
# provable recovery guarantees under the Restricted Isometry Property (RIP).

using CompSense
using LinearAlgebra
using Random

Random.seed!(101)

println("=" ^ 60)
println("CoSaMP (Compressive Sampling Matching Pursuit) Example")
println("=" ^ 60)

# ============================================================================
# Basic Usage
# ============================================================================
println("\n1. Basic Usage")
println("-" ^ 40)

# Create a sparse recovery problem
A, x_true, b = gaussian_sensing(60, 200, 10)

# Recover using CoSaMP with known sparsity
x_cosamp = CoSaMP(A, b; sparsity=10)

println("True signal:      $(count(!iszero, x_true)) non-zeros")
println("Recovered signal: $(count(!iszero, x_cosamp)) non-zeros")
println("Relative error:   $(round(norm(x_cosamp - x_true) / norm(x_true) * 100, digits=2))%")

# ============================================================================
# CoSaMP Algorithm Steps
# ============================================================================
println("\n2. Understanding CoSaMP")
println("-" ^ 40)
println("""
CoSaMP performs these steps each iteration:
  1. Form proxy: u = Aᵀr (correlate columns with residual)
  2. Identify Ω = indices of 2k largest entries in u
  3. Merge: T = Ω ∪ support(x)
  4. Least squares: x̃ = argmin ||A_T z - b||₂
  5. Prune: x = keep k largest entries of x̃
  6. Update residual: r = b - Ax

Key insight: Uses 2k candidates to avoid getting stuck!
""")

# ============================================================================
# Effect of Sparsity Parameter
# ============================================================================
println("3. Effect of Sparsity Parameter k")
println("-" ^ 40)

# True sparsity is 10
println("True sparsity: 10")
println()
println("k  | Non-zeros | Rel. Error | Residual ||Ax-b||")
println("-" ^ 50)
for k in [5, 8, 10, 12, 15, 20]
    x_rec = CoSaMP(A, b; sparsity=k)
    nnz = count(!iszero, x_rec)
    err = norm(x_rec - x_true) / norm(x_true)
    res = norm(A * x_rec - b)
    println("$(lpad(k, 2)) |    $(lpad(nnz, 2))     |   $(lpad(round(err * 100, digits=2), 6))%  |   $(round(res, digits=6))")
end

# ============================================================================
# Convergence Behavior
# ============================================================================
println("\n4. Convergence Over Iterations")
println("-" ^ 40)

println("maxiter | Rel. Error | Residual")
println("-" ^ 40)
for maxiter in [1, 2, 5, 10, 20, 50, 100]
    x_rec = CoSaMP(A, b; sparsity=10, maxiter=maxiter)
    err = norm(x_rec - x_true) / norm(x_true)
    res = norm(A * x_rec - b)
    println("$(lpad(maxiter, 6))  |   $(lpad(round(err * 100, digits=2), 6))%  |   $(round(res, digits=6))")
end

println("\n→ CoSaMP often converges in just a few iterations!")

# ============================================================================
# Comparison: CoSaMP vs OMP vs IHT
# ============================================================================
println("\n5. CoSaMP vs OMP vs IHT")
println("-" ^ 40)

# Test on multiple random problems
n_trials = 10
errors = Dict("OMP" => Float64[], "IHT" => Float64[], "CoSaMP" => Float64[])

for trial in 1:n_trials
    A_trial, x_trial, b_trial = gaussian_sensing(60, 200, 10)

    x_omp = OMP(A_trial, b_trial; sparsity=10)
    x_iht = IHT(A_trial, b_trial; sparsity=10)
    x_cosamp = CoSaMP(A_trial, b_trial; sparsity=10)

    push!(errors["OMP"], norm(x_omp - x_trial) / norm(x_trial))
    push!(errors["IHT"], norm(x_iht - x_trial) / norm(x_trial))
    push!(errors["CoSaMP"], norm(x_cosamp - x_trial) / norm(x_trial))
end

println("Average relative error over $n_trials trials:")
for alg in ["OMP", "IHT", "CoSaMP"]
    mean_err = sum(errors[alg]) / n_trials
    std_err = sqrt(sum((errors[alg] .- mean_err).^2) / n_trials)
    println("  $(rpad(alg, 7)): $(round(mean_err * 100, digits=2))% ± $(round(std_err * 100, digits=2))%")
end

# ============================================================================
# Support Recovery
# ============================================================================
println("\n6. Support Recovery Analysis")
println("-" ^ 40)

# Check exact support recovery
true_support = Set(findall(!iszero, x_true))
cosamp_support = Set(findall(!iszero, x_cosamp))

correct = length(intersect(true_support, cosamp_support))
missed = length(setdiff(true_support, cosamp_support))
false_pos = length(setdiff(cosamp_support, true_support))

println("True support:          $(sort(collect(true_support)))")
println("Recovered support:     $(sort(collect(cosamp_support)))")
println()
println("Correctly identified:  $correct / $(length(true_support))")
println("Missed:                $missed")
println("False positives:       $false_pos")

# ============================================================================
# Noisy Measurements
# ============================================================================
println("\n7. Recovery from Noisy Measurements")
println("-" ^ 40)

A_clean, x_clean, b_clean = gaussian_sensing(80, 200, 10)

for σ in [0.0, 0.01, 0.05, 0.1]
    b_noisy = b_clean + σ * randn(length(b_clean))
    x_rec = CoSaMP(A_clean, b_noisy; sparsity=10)
    err = norm(x_rec - x_clean) / norm(x_clean)
    println("  σ=$(lpad(σ, 4)): error = $(round(err * 100, digits=2))%")
end

println("\n✓ CoSaMP example complete!")
println("\nKey takeaways:")
println("  • CoSaMP has provable recovery guarantees under RIP")
println("  • Uses 2k candidates per iteration (more aggressive than OMP)")
println("  • Often converges in fewer iterations than IHT")
println("  • Robust to noise with proper problem scaling")
println("  • Best choice when theoretical guarantees matter")
