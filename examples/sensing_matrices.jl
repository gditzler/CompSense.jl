# Sensing Matrix Types Example
#
# This script demonstrates different sensing matrix constructions
# and their properties for compressed sensing.

using CompSense
using LinearAlgebra
using Random

Random.seed!(606)

println("=" ^ 60)
println("Sensing Matrix Types in CompSense.jl")
println("=" ^ 60)

# ============================================================================
# Problem Setup
# ============================================================================
n = 50   # measurements
p = 200  # signal dimension
k = 10   # sparsity

println("\nProblem dimensions: m=$n measurements, n=$p signal dimension, k=$k sparsity")

# ============================================================================
# Gaussian Sensing Matrix
# ============================================================================
println("\n" * "=" ^ 60)
println("1. Gaussian Random Matrix")
println("=" ^ 60)
println("""
A(i,j) ~ N(0, 1)

Properties:
  • Satisfies RIP with high probability
  • O(k log(n/k)) measurements suffice
  • Gold standard for theoretical analysis
  • Dense matrix: O(mn) storage
""")

A_gauss, x_gauss, b_gauss = gaussian_sensing(n, p, k)
x_rec = SL0(A_gauss, b_gauss)
println("Recovery error: $(round(norm(x_rec - x_gauss) / norm(x_gauss) * 100, digits=2))%")

# ============================================================================
# Bernoulli Sensing Matrix
# ============================================================================
println("\n" * "=" ^ 60)
println("2. Bernoulli/Rademacher Matrix")
println("=" ^ 60)
println("""
A(i,j) = ±1/√m with equal probability

Properties:
  • Binary entries: computationally simple
  • Similar RIP properties to Gaussian
  • Useful for binary measurement systems
  • Faster matrix-vector products with bit operations
""")

A_bern, x_bern, b_bern = bernoulli_sensing(n, p, k)
x_rec = SL0(A_bern, b_bern)
println("Recovery error: $(round(norm(x_rec - x_bern) / norm(x_bern) * 100, digits=2))%")

# ============================================================================
# Partial Fourier Matrix
# ============================================================================
println("\n" * "=" ^ 60)
println("3. Partial Fourier Matrix")
println("=" ^ 60)
println("""
A = randomly selected rows of DFT matrix

Properties:
  • O(n log n) fast matrix-vector products (FFT)
  • Critical for MRI, radar, spectroscopy
  • Good RIP for sparse signals in time domain
  • Complex-valued (real version available)
""")

A_four, x_four, b_four = fourier_sensing(n, p, k)
x_rec = SL0(A_four, b_four)
println("Recovery error: $(round(norm(x_rec - x_four) / norm(x_four) * 100, digits=2))%")

# ============================================================================
# Partial DCT Matrix
# ============================================================================
println("\n" * "=" ^ 60)
println("4. Partial DCT Matrix")
println("=" ^ 60)
println("""
A = randomly selected rows of DCT matrix

Properties:
  • Real-valued alternative to Fourier
  • O(n log n) fast transforms (DCT)
  • Basis for JPEG, MPEG compression
  • Good for piecewise smooth signals
""")

A_dct, x_dct, b_dct = dct_sensing(n, p, k)
x_rec = SL0(A_dct, b_dct)
println("Recovery error: $(round(norm(x_rec - x_dct) / norm(x_dct) * 100, digits=2))%")

# ============================================================================
# Partial Hadamard Matrix
# ============================================================================
println("\n" * "=" ^ 60)
println("5. Partial Hadamard Matrix")
println("=" ^ 60)
println("""
A = randomly selected rows of Hadamard matrix

Properties:
  • ±1 entries only
  • O(n log n) fast Walsh-Hadamard transform
  • Used in CDMA, quantum computing
  • Requires n to be power of 2
""")

# Hadamard requires power of 2
p_had = 256
A_had, x_had, b_had = hadamard_sensing(n, p_had, k)
x_rec = SL0(A_had, b_had)
println("Recovery error: $(round(norm(x_rec - x_had) / norm(x_had) * 100, digits=2))%")

# ============================================================================
# Sparse Random Matrix
# ============================================================================
println("\n" * "=" ^ 60)
println("6. Sparse Random Matrix")
println("=" ^ 60)
println("""
A has only ~density fraction of entries non-zero

Properties:
  • Reduced storage: O(density × mn)
  • Faster matrix-vector products
  • Good for very large-scale problems
  • Weaker RIP constants than dense matrices
""")

A_sparse, x_sparse, b_sparse = sparse_sensing(n, p, k; density=0.1)
x_rec = SL0(A_sparse, b_sparse)
println("Recovery error: $(round(norm(x_rec - x_sparse) / norm(x_sparse) * 100, digits=2))%")
println("Matrix density: $(round(sum(A_sparse .!= 0) / length(A_sparse) * 100, digits=1))%")

# ============================================================================
# Uniform Random Matrix
# ============================================================================
println("\n" * "=" ^ 60)
println("7. Uniform Random Matrix")
println("=" ^ 60)
println("""
A(i,j) ~ Uniform(-1, 1)

Properties:
  • Bounded entries
  • Simple to generate
  • Less common in theory, but practical
""")

A_unif, x_unif, b_unif = uniform_sensing(n, p, k)
x_rec = SL0(A_unif, b_unif)
println("Recovery error: $(round(norm(x_rec - x_unif) / norm(x_unif) * 100, digits=2))%")

# ============================================================================
# Toeplitz Matrix
# ============================================================================
println("\n" * "=" ^ 60)
println("8. Toeplitz Matrix")
println("=" ^ 60)
println("""
A has constant diagonals: A(i,j) = a(i-j)

Properties:
  • Models convolution/LTI systems
  • O(n log n) matrix-vector products (FFT)
  • Only O(m+n) parameters to store
  • Natural for time-series, communications
""")

A_toep, x_toep, b_toep = toeplitz_sensing(n, p, k)
x_rec = SL0(A_toep, b_toep)
println("Recovery error: $(round(norm(x_rec - x_toep) / norm(x_toep) * 100, digits=2))%")

# ============================================================================
# Comparison Summary
# ============================================================================
println("\n" * "=" ^ 60)
println("Summary: Recovery Performance Comparison")
println("=" ^ 60)

matrices = [
    ("Gaussian", gaussian_sensing(n, p, k)),
    ("Bernoulli", bernoulli_sensing(n, p, k)),
    ("Fourier", fourier_sensing(n, p, k)),
    ("DCT", dct_sensing(n, p, k)),
    ("Hadamard", hadamard_sensing(n, 256, k)),
    ("Sparse (10%)", sparse_sensing(n, p, k; density=0.1)),
    ("Uniform", uniform_sensing(n, p, k)),
    ("Toeplitz", toeplitz_sensing(n, p, k)),
]

println("\nMatrix Type   | SL0 Error | OMP Error | IHT Error")
println("-" ^ 55)
for (name, (A, x, b)) in matrices
    err_sl0 = norm(SL0(A, b) - x) / norm(x)
    err_omp = norm(OMP(A, b; sparsity=k) - x) / norm(x)
    err_iht = norm(IHT(A, b; sparsity=k) - x) / norm(x)
    println("$(rpad(name, 13)) |  $(lpad(round(err_sl0*100, digits=1), 5))%   |  $(lpad(round(err_omp*100, digits=1), 5))%   |  $(lpad(round(err_iht*100, digits=1), 5))%")
end

# ============================================================================
# Practical Recommendations
# ============================================================================
println("\n" * "=" ^ 60)
println("Practical Recommendations")
println("=" ^ 60)
println("""
Choose your sensing matrix based on:

1. Hardware constraints:
   • Optical systems → Fourier/DCT
   • Binary sensors → Bernoulli/Hadamard
   • Convolution → Toeplitz

2. Computational requirements:
   • Need fast Ax and A'y → Fourier, DCT, Hadamard
   • Memory limited → Sparse or structured matrices
   • General purpose → Gaussian

3. Theoretical guarantees:
   • Strongest RIP → Gaussian, Bernoulli
   • Deterministic → Hadamard (specific constructions)

4. Signal structure:
   • Time-series → Toeplitz
   • Images → DCT, Fourier
   • General sparse → Gaussian
""")

println("✓ Sensing matrices example complete!")
