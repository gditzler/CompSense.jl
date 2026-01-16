# Orthogonal Matching Pursuit (OMP) Example
#
# OMP is a greedy algorithm that iteratively selects the column of A
# most correlated with the current residual, then solves least squares
# on the selected support.

using CompSense
using LinearAlgebra
using Random

Random.seed!(123)

println("=" ^ 60)
println("Orthogonal Matching Pursuit (OMP) Example")
println("=" ^ 60)

# ============================================================================
# Basic Usage
# ============================================================================
println("\n1. Basic Usage")
println("-" ^ 40)

# Create a sparse recovery problem
A, x_true, b = gaussian_sensing(50, 200, 8)

# Recover using OMP with known sparsity
x_omp = OMP(A, b; sparsity=8)

println("True signal:      $(count(!iszero, x_true)) non-zeros")
println("Recovered signal: $(count(!iszero, x_omp)) non-zeros")
println("Relative error:   $(round(norm(x_omp - x_true) / norm(x_true) * 100, digits=2))%")

# ============================================================================
# Effect of Sparsity Parameter
# ============================================================================
println("\n2. Effect of Sparsity Parameter")
println("-" ^ 40)

# True sparsity is 8, let's see what happens with different settings
for k in [4, 8, 12, 16]
    x_rec = OMP(A, b; sparsity=k)
    err = norm(x_rec - x_true) / norm(x_true)
    nnz = count(!iszero, x_rec)
    println("  k=$k: $(lpad(nnz, 2)) non-zeros, error = $(round(err * 100, digits=2))%")
end

# ============================================================================
# Early Stopping with Tolerance
# ============================================================================
println("\n3. Early Stopping with Residual Tolerance")
println("-" ^ 40)

# OMP can stop early if the residual is small enough
x_tight = OMP(A, b; sparsity=20, tol=1e-10)
x_loose = OMP(A, b; sparsity=20, tol=1e-2)

println("Tight tolerance (1e-10): $(count(!iszero, x_tight)) non-zeros")
println("Loose tolerance (1e-2):  $(count(!iszero, x_loose)) non-zeros")

# ============================================================================
# Support Recovery
# ============================================================================
println("\n4. Support Recovery Analysis")
println("-" ^ 40)

# Check if we recovered the correct support (locations of non-zeros)
true_support = Set(findall(!iszero, x_true))
recovered_support = Set(findall(!iszero, x_omp))

correct = length(intersect(true_support, recovered_support))
missed = length(setdiff(true_support, recovered_support))
false_pos = length(setdiff(recovered_support, true_support))

println("True support size:     $(length(true_support))")
println("Correctly identified:  $correct")
println("Missed:                $missed")
println("False positives:       $false_pos")

# ============================================================================
# Noisy Measurements
# ============================================================================
println("\n5. Recovery from Noisy Measurements")
println("-" ^ 40)

# Add noise to measurements
noise_levels = [0.0, 0.01, 0.05, 0.1]
A_clean, x_clean, b_clean = gaussian_sensing(60, 200, 8)

for σ in noise_levels
    b_noisy = b_clean + σ * randn(length(b_clean))
    x_rec = OMP(A_clean, b_noisy; sparsity=8)
    err = norm(x_rec - x_clean) / norm(x_clean)
    println("  σ=$(lpad(σ, 4)): error = $(round(err * 100, digits=2))%")
end

println("\n✓ OMP example complete!")
println("\nKey takeaways:")
println("  • OMP is fast and simple to implement")
println("  • Works best when sparsity k is known or slightly overestimated")
println("  • Sensitive to noise in measurements")
println("  • Guaranteed to recover sparse signals under RIP conditions")
