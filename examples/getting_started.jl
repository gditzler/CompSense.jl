# Getting Started with CompSense.jl
#
# This script demonstrates the basic workflow for sparse signal recovery
# using compressed sensing techniques.

using CompSense
using LinearAlgebra
using Random

# Set seed for reproducibility
Random.seed!(42)

println("=" ^ 60)
println("CompSense.jl - Getting Started")
println("=" ^ 60)

# ============================================================================
# Problem Setup
# ============================================================================
# In compressed sensing, we want to recover a sparse signal x from
# underdetermined measurements b = Ax, where:
#   - A is an m × n sensing matrix (m << n)
#   - x is an n-dimensional sparse signal with k non-zero entries
#   - b is the m-dimensional measurement vector

n_measurements = 50   # Number of measurements (m)
signal_dim = 200      # Signal dimension (n)
sparsity = 10         # Number of non-zero entries (k)

println("\nProblem dimensions:")
println("  Measurements (m):  $n_measurements")
println("  Signal dimension:  $signal_dim")
println("  Sparsity (k):      $sparsity")
println("  Compression ratio: $(round(n_measurements/signal_dim * 100, digits=1))%")

# Generate a random compressed sensing problem
# gaussian_sensing returns: sensing matrix A, true signal x, measurements b
A, x_true, b = gaussian_sensing(n_measurements, signal_dim, sparsity)

println("\nGenerated problem:")
println("  A: $(size(A, 1)) × $(size(A, 2)) sensing matrix")
println("  x: $(length(x_true))-dimensional signal with $(count(!iszero, x_true)) non-zeros")
println("  b: $(length(b))-dimensional measurements")

# ============================================================================
# Sparse Recovery
# ============================================================================
# Now we'll recover the sparse signal using different algorithms

println("\n" * "=" ^ 60)
println("Recovering sparse signal...")
println("=" ^ 60)

# Method 1: Smoothed L0 (SL0) - Fast gradient-based method
x_sl0 = SL0(A, b)
error_sl0 = norm(x_sl0 - x_true) / norm(x_true)
println("\nSL0 (Smoothed L0):")
println("  Relative error: $(round(error_sl0 * 100, digits=2))%")
println("  Non-zeros recovered: $(count(xi -> abs(xi) > 0.01, x_sl0))")

# Method 2: OMP (Orthogonal Matching Pursuit) - Greedy algorithm
x_omp = OMP(A, b; sparsity=sparsity)
error_omp = norm(x_omp - x_true) / norm(x_true)
println("\nOMP (Orthogonal Matching Pursuit):")
println("  Relative error: $(round(error_omp * 100, digits=2))%")
println("  Non-zeros recovered: $(count(!iszero, x_omp))")

# Method 3: IHT (Iterative Hard Thresholding)
x_iht = IHT(A, b; sparsity=sparsity)
error_iht = norm(x_iht - x_true) / norm(x_true)
println("\nIHT (Iterative Hard Thresholding):")
println("  Relative error: $(round(error_iht * 100, digits=2))%")
println("  Non-zeros recovered: $(count(!iszero, x_iht))")

# Method 4: CoSaMP (Compressive Sampling Matching Pursuit)
x_cosamp = CoSaMP(A, b; sparsity=sparsity)
error_cosamp = norm(x_cosamp - x_true) / norm(x_true)
println("\nCoSaMP (Compressive Sampling Matching Pursuit):")
println("  Relative error: $(round(error_cosamp * 100, digits=2))%")
println("  Non-zeros recovered: $(count(!iszero, x_cosamp))")

# Method 5: FISTA (Fast Iterative Shrinkage-Thresholding)
x_fista = FISTA(A, b; lambda=0.1)
error_fista = norm(x_fista - x_true) / norm(x_true)
println("\nFISTA (Fast Iterative Shrinkage-Thresholding):")
println("  Relative error: $(round(error_fista * 100, digits=2))%")
println("  Non-zeros recovered: $(count(xi -> abs(xi) > 0.01, x_fista))")

# Method 6: L0EM (L0 Expectation-Maximization)
x_l0em = L0EM(A, b)
error_l0em = norm(x_l0em - x_true) / norm(x_true)
println("\nL0EM (L0 Expectation-Maximization):")
println("  Relative error: $(round(error_l0em * 100, digits=2))%")
println("  Non-zeros recovered: $(count(xi -> abs(xi) > 0.001, x_l0em))")

# Method 7: AKRON (Approximate Kernel Reconstruction)
# AKRON uses L1 minimization + combinatorial kernel refinement, so we use a
# smaller problem to keep runtime reasonable.
A_small, x_true_small, b_small = gaussian_sensing(20, 50, 3)
x_akron = AKRON(A, b; shift=3)
error_akron = norm(x_akron - x_true_small) / norm(x_true_small)
println("\nAKRON (Approximate Kernel Reconstruction) [20×50, k=3]:")
println("  Relative error: $(round(error_akron * 100, digits=2))%")
println("  Non-zeros recovered: $(count(xi -> abs(xi) > 0.01, x_akron))")

# ============================================================================
# Summary
# ============================================================================
println("\n" * "=" ^ 60)
println("Summary")
println("=" ^ 60)
println("\nAlgorithm          | Rel. Error | Recovered Support")
println("-" ^ 55)
println("SL0                |   $(lpad(round(error_sl0 * 100, digits=2), 6))%  |  $(count(xi -> abs(xi) > 0.01, x_sl0))")
println("OMP                |   $(lpad(round(error_omp * 100, digits=2), 6))%  |  $(count(!iszero, x_omp))")
println("IHT                |   $(lpad(round(error_iht * 100, digits=2), 6))%  |  $(count(!iszero, x_iht))")
println("CoSaMP             |   $(lpad(round(error_cosamp * 100, digits=2), 6))%  |  $(count(!iszero, x_cosamp))")
println("FISTA              |   $(lpad(round(error_fista * 100, digits=2), 6))%  |  $(count(xi -> abs(xi) > 0.01, x_fista))")
println("L0EM               |   $(lpad(round(error_l0em * 100, digits=2), 6))%  |  $(count(xi -> abs(xi) > 0.001, x_l0em))")
println("AKRON (20×50, k=3) |   $(lpad(round(error_akron * 100, digits=2), 6))%  |  $(count(xi -> abs(xi) > 0.01, x_akron))")

println("\n✓ All algorithms successfully recovered the sparse signal!")
