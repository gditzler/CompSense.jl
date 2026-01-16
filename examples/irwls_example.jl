# IRWLS (Iteratively Reweighted Least Squares) Example
#
# IRWLS enhances sparsity by iteratively solving reweighted L1 problems.
# It uses convex optimization (Convex.jl + SCS.jl) under the hood.

using CompSense
using LinearAlgebra
using Random

Random.seed!(505)

println("=" ^ 60)
println("IRWLS (Iteratively Reweighted Least Squares) Example")
println("=" ^ 60)

# ============================================================================
# Basic Usage
# ============================================================================
println("\n1. Basic Usage")
println("-" ^ 40)

# Create a sparse recovery problem (smaller for speed with convex optimization)
A, x_true, b = gaussian_sensing(30, 80, 5)

# Recover using IRWLS (note: slower than other methods)
println("Running IRWLS (uses convex optimization, may take a moment)...")
x_irwls = IRWLS(A, b; maxiter=5)

println("True signal:      $(count(!iszero, x_true)) non-zeros")
println("Recovered signal: $(count(xi -> abs(xi) > 0.01, x_irwls)) non-zeros (threshold=0.01)")
println("Relative error:   $(round(norm(x_irwls - x_true) / norm(x_true) * 100, digits=2))%")

# ============================================================================
# Algorithm Overview
# ============================================================================
println("\n2. Understanding IRWLS")
println("-" ^ 40)
println("""
IRWLS solves a sequence of weighted L1 problems:
  min_x  Σ w_i |x_i|  subject to Ax = b

where weights are updated as:
  w_i = 1 / (ε + |x_i|)

This reweighting scheme gives smaller weights to larger entries,
effectively approximating L0 minimization.

Reference: Candès, Wakin, Boyd (2008) - "Enhancing Sparsity by
Reweighted L1 Minimization"
""")

# ============================================================================
# Effect of Maximum Iterations
# ============================================================================
println("3. Effect of Maximum Iterations")
println("-" ^ 40)

println("Note: Each iteration solves a convex optimization problem.")
println()
println("maxiter | Non-zeros | Rel. Error")
println("-" ^ 40)
for maxiter in [1, 2, 3, 5, 10]
    x_rec = IRWLS(A, b; maxiter=maxiter)
    nnz = count(xi -> abs(xi) > 0.01, x_rec)
    err = norm(x_rec - x_true) / norm(x_true)
    println("$(lpad(maxiter, 6))  |    $(lpad(nnz, 2))     |   $(round(err * 100, digits=2))%")
end

println("\n→ More iterations = better sparsity, but slower")

# ============================================================================
# Effect of Epsilon
# ============================================================================
println("\n4. Effect of Epsilon (Regularization)")
println("-" ^ 40)

println("epsilon | Non-zeros | Rel. Error")
println("-" ^ 40)
for eps in [0.001, 0.01, 0.05, 0.1]
    x_rec = IRWLS(A, b; maxiter=5, epsilon=eps)
    nnz = count(xi -> abs(xi) > eps, x_rec)
    err = norm(x_rec - x_true) / norm(x_true)
    println("$(lpad(eps, 6))  |    $(lpad(nnz, 2))     |   $(round(err * 100, digits=2))%")
end

println("\n→ Smaller ε: sharper reweighting, sparser solutions")
println("→ Larger ε: smoother reweighting, more stable")

# ============================================================================
# IRWLS vs Other Methods
# ============================================================================
println("\n5. IRWLS vs Other Methods")
println("-" ^ 40)

# Use same problem for fair comparison
x_sl0 = SL0(A, b)
x_l0em = L0EM(A, b)
x_omp = OMP(A, b; sparsity=5)

algorithms = [
    ("IRWLS", x_irwls),
    ("SL0", x_sl0),
    ("L0EM", x_l0em),
    ("OMP (k=5)", x_omp),
]

println("Algorithm   | Non-zeros | Rel. Error")
println("-" ^ 45)
for (name, x_rec) in algorithms
    nnz = count(xi -> abs(xi) > 0.01, x_rec)
    err = norm(x_rec - x_true) / norm(x_true)
    println("$(rpad(name, 11)) |    $(lpad(nnz, 2))     |   $(round(err * 100, digits=2))%")
end

# ============================================================================
# When to Use IRWLS
# ============================================================================
println("\n6. When to Use IRWLS")
println("-" ^ 40)
println("""
✓ Use IRWLS when:
  • You need high-quality sparse solutions
  • Problem size is moderate (convex solver is O(n³))
  • You can afford extra computation time
  • Theoretical guarantees of reweighted L1 matter

✗ Consider alternatives when:
  • Speed is critical (use SL0, OMP, or IHT instead)
  • Problem is very large (n > 1000)
  • Real-time applications

Speed comparison (rough estimates):
  • SL0, L0EM, IHT: ~O(n²) per iteration
  • OMP, CoSaMP: ~O(n² k)
  • IRWLS: ~O(n³) per iteration (convex solver)
""")

# ============================================================================
# Quality vs Speed Tradeoff
# ============================================================================
println("7. Reducing Iterations for Speed")
println("-" ^ 40)

# For practical use, even 2-3 iterations often suffice
println("Testing with reduced iterations on larger problem...")
A2, x2, b2 = gaussian_sensing(40, 120, 6)

for (iters, description) in [(1, "L1 only"), (2, "quick"), (3, "balanced"), (5, "thorough")]
    x_rec = IRWLS(A2, b2; maxiter=iters)
    nnz = count(xi -> abs(xi) > 0.01, x_rec)
    err = norm(x_rec - x2) / norm(x2)
    println("  $(lpad(iters, 1)) iter ($description): $(lpad(nnz, 2)) non-zeros, error = $(round(err * 100, digits=2))%")
end

println("\n✓ IRWLS example complete!")
println("\nKey takeaways:")
println("  • IRWLS uses convex optimization (slower but principled)")
println("  • Iteratively reweights L1 to approximate L0")
println("  • maxiter=3-5 often sufficient for good results")
println("  • Best for moderate-sized problems where quality matters")
println("  • Reference: Candès, Wakin, Boyd (2008)")
