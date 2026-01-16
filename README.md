# CompSense.jl

A Julia package for **compressed sensing** and **sparse signal recovery**. CompSense.jl provides efficient implementations of several algorithms to solve the underdetermined system Ax = b, where x is known to be sparse.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/gditzler/CompSense.jl")
```

Or in the Julia REPL package mode (press `]`):

```
add https://github.com/gditzler/CompSense.jl
```

## Quick Start

```julia
using CompSense

# Generate a synthetic compressed sensing problem
# 50 measurements, 200-dimensional signal, 10 non-zero entries
A, x_true, b = gaussian_sensing(50, 200, 10)

# Recover the sparse signal using different algorithms
x_sl0 = SL0(A, b)                      # Smoothed L0
x_omp = OMP(A, b; sparsity=10)         # Orthogonal Matching Pursuit
x_iht = IHT(A, b; sparsity=10)         # Iterative Hard Thresholding
x_cosamp = CoSaMP(A, b; sparsity=10)   # Compressive Sampling Matching Pursuit
x_fista = FISTA(A, b; lambda=0.1)      # Fast Iterative Shrinkage-Thresholding

# Check recovery quality
using LinearAlgebra
println("SL0 error:    ", norm(x_sl0 - x_true) / norm(x_true))
println("OMP error:    ", norm(x_omp - x_true) / norm(x_true))
println("IHT error:    ", norm(x_iht - x_true) / norm(x_true))
println("CoSaMP error: ", norm(x_cosamp - x_true) / norm(x_true))
println("FISTA error:  ", norm(x_fista - x_true) / norm(x_true))
```

## Sparse Recovery Algorithms

CompSense provides **7 algorithms** for sparse signal recovery:

| Algorithm | Type | Sparsity | Speed | Best For |
|:----------|:-----|:---------|:------|:---------|
| **OMP** | Greedy | Exact (k) | Fast | Known sparsity, baseline |
| **CoSaMP** | Greedy | Exact (k) | Fast | Theoretical guarantees |
| **IHT** | Thresholding | Exact (k) | Very Fast | Simple, well-conditioned A |
| **FISTA** | Proximal | Approximate | Fast | Noisy data, LASSO |
| **SL0** | Smoothing | Approximate | Very Fast | General purpose |
| **L0EM** | EM | Approximate | Fast | Balance speed/accuracy |
| **IRWLS** | Convex | Approximate | Slow | High-quality solutions |

---

### OMP - Orthogonal Matching Pursuit

Greedy algorithm that iteratively selects atoms most correlated with the residual.

```julia
x = OMP(A, b; sparsity=10, tol=1e-6, maxiter=nothing)
```

**Parameters:**
- `sparsity`: Target number of non-zeros k (default: number of measurements)
- `tol`: Residual tolerance for early stopping (default: 1e-6)
- `maxiter`: Maximum iterations (default: sparsity)

**Reference:** Tropp & Gilbert, ["Signal Recovery From Random Measurements Via Orthogonal Matching Pursuit"](https://ieeexplore.ieee.org/document/4385788), IEEE Trans. Info. Theory, 2007.

---

### CoSaMP - Compressive Sampling Matching Pursuit

Greedy pursuit with subspace projection and provable recovery guarantees.

```julia
x = CoSaMP(A, b; sparsity=10, maxiter=100, tol=1e-6)
```

**Parameters:**
- `sparsity`: Target number of non-zeros k (required)
- `maxiter`: Maximum iterations (default: 100)
- `tol`: Residual tolerance (default: 1e-6)

**Reference:** Needell & Tropp, ["CoSaMP: Iterative signal recovery from incomplete and inaccurate samples"](https://arxiv.org/abs/0803.2392), Appl. Comput. Harmon. Anal., 2009.

---

### IHT - Iterative Hard Thresholding

Gradient descent with hard thresholding to enforce exact sparsity.

```julia
x = IHT(A, b; sparsity=10, maxiter=500, tol=1e-6, mu=nothing)
```

**Parameters:**
- `sparsity`: Target number of non-zeros k (required)
- `maxiter`: Maximum iterations (default: 500)
- `tol`: Convergence tolerance (default: 1e-6)
- `mu`: Step size (default: 1/‖A‖₂²)

**Reference:** Blumensath & Davies, ["Iterative Hard Thresholding for Compressed Sensing"](https://arxiv.org/abs/0805.0510), Appl. Comput. Harmon. Anal., 2009.

---

### FISTA - Fast Iterative Shrinkage-Thresholding Algorithm

Accelerated proximal gradient method for LASSO with O(1/k²) convergence.

```julia
x = FISTA(A, b; lambda=0.1, maxiter=500, tol=1e-6, L=nothing)
```

**Parameters:**
- `lambda`: L1 regularization parameter (default: 0.1)
- `maxiter`: Maximum iterations (default: 500)
- `tol`: Convergence tolerance (default: 1e-6)
- `L`: Lipschitz constant (default: computed as ‖A‖₂²)

**Reference:** Beck & Teboulle, ["A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems"](https://epubs.siam.org/doi/10.1137/080716542), SIAM J. Imaging Sci., 2009.

---

### SL0 - Smoothed L0

Approximates the L0 norm with a smooth Gaussian function and uses gradient ascent.

```julia
x = SL0(A, b; sigma_decrease_factor=0.85, maxiter=150, epsilon=0.001)
```

**Parameters:**
- `sigma_decrease_factor`: Annealing rate for smoothing (default: 0.85)
- `maxiter`: Maximum outer iterations (default: 150)
- `epsilon`: Threshold for zeroing coefficients (default: 0.001)

**Reference:** [Smoothed L0 (SL0)](http://ee.sharif.edu/~SLzero/)

---

### L0EM - L0 Expectation-Maximization

EM-based algorithm that directly solves the L0-regularized optimization problem.

```julia
x = L0EM(A, b; lambda=0.001, epsilon=0.001, maxiter=50)
```

**Parameters:**
- `lambda`: Regularization parameter (default: 0.001)
- `epsilon`: Convergence threshold (default: 0.001)
- `maxiter`: Maximum iterations (default: 50)

**Reference:** Liu & Li, ["L0-EM Algorithm for Sparse Recovery"](https://arxiv.org/pdf/1407.7508v1.pdf)

---

### IRWLS - Iteratively Reweighted Least Squares

Enhances sparsity by iteratively solving reweighted L1 minimization problems.

```julia
x = IRWLS(A, b; maxiter=100, epsilon=0.01)
```

**Parameters:**
- `maxiter`: Maximum iterations (default: 100)
- `epsilon`: Convergence threshold and regularization (default: 0.01)

**Reference:** Candès, Wakin & Boyd, ["Enhancing Sparsity by Reweighted L1 Minimization"](https://web.stanford.edu/~boyd/papers/pdf/rwl1.pdf), J. Fourier Anal. Appl., 2008.

> **Note:** IRWLS uses convex optimization (via [Convex.jl](https://github.com/jump-dev/Convex.jl) and [SCS.jl](https://github.com/jump-dev/SCS.jl)) and is slower than other methods.

## Sensing Matrix Generators

CompSense provides multiple sensing matrix types for benchmarking:

| Matrix Type | Function | Fast Transform | Notes |
|:------------|:---------|:--------------:|:------|
| Gaussian | `gaussian_sensing` | ✗ | Gold standard, satisfies RIP |
| Bernoulli | `bernoulli_sensing` | ✗ | ±1 entries, simple |
| Fourier | `fourier_sensing` | ✓ O(n log n) | MRI, radar, spectroscopy |
| DCT | `dct_sensing` | ✓ O(n log n) | JPEG, MPEG |
| Hadamard | `hadamard_sensing` | ✓ O(n log n) | Requires n = 2^k |
| Sparse | `sparse_sensing` | ✗ | Large-scale problems |
| Uniform | `uniform_sensing` | ✗ | Bounded entries |
| Toeplitz | `toeplitz_sensing` | ✓ O(n log n) | Convolution/LTI systems |

### Example Usage

```julia
# Gaussian random matrix (most common)
A, x, b = gaussian_sensing(n, p, k)

# Partial Fourier matrix (for MRI, radar)
A, x, b = fourier_sensing(n, p, k; real_valued=true)

# Hadamard matrix (requires p = power of 2)
A, x, b = hadamard_sensing(n, 256, k)

# Generate just a sparse signal
x = generate_sparse_signal(p, k; min_magnitude=1.0)
```

## Examples

The `examples/` directory contains scripts demonstrating each algorithm:

| File | Description |
|:-----|:------------|
| `getting_started.jl` | Basic workflow with all algorithms |
| `omp_example.jl` | OMP usage and parameter tuning |
| `cosamp_example.jl` | CoSaMP with support recovery analysis |
| `iht_example.jl` | IHT step size and convergence |
| `fista_example.jl` | FISTA lambda tuning for LASSO |
| `sl0_example.jl` | SL0 annealing schedule |
| `l0em_example.jl` | L0EM regularization effects |
| `irwls_example.jl` | IRWLS quality vs speed |
| `sensing_matrices.jl` | Comparison of matrix types |
| `algorithm_comparison.jl` | **Interactive Pluto notebook** |

### Running Examples

```julia
# Run a single example
include("examples/getting_started.jl")

# Or run any specific algorithm example
include("examples/omp_example.jl")
```

### Interactive Pluto Notebook

The `algorithm_comparison.jl` file is a [Pluto.jl](https://github.com/fonsp/Pluto.jl) notebook that provides an interactive comparison of all algorithms with sliders to adjust problem parameters.

**To run the Pluto notebook:**

```julia
# Install Pluto if you haven't already
using Pkg
Pkg.add("Pluto")

# Launch Pluto
using Pluto
Pluto.run()
```

Then open `examples/algorithm_comparison.jl` in the Pluto browser interface.

**What you can explore:**
- Adjust measurements (m), signal dimension (n), and sparsity (k)
- Compare recovery error across all algorithms
- Analyze support recovery accuracy
- Study noise sensitivity
- View measurement sweep phase transitions

## Performance Tips

1. **Fastest algorithms:** SL0 and IHT for most problems
2. **Known sparsity:** Use OMP, IHT, or CoSaMP
3. **Noisy measurements:** Use FISTA with tuned lambda
4. **Theoretical guarantees:** Use CoSaMP
5. **Highest quality:** Use IRWLS (but slower)
6. **Very large problems:** Reduce `maxiter`, use sparse matrices

## Example: Comparing All Algorithms

```julia
using CompSense
using LinearAlgebra

# Create test problem
A, x_true, b = gaussian_sensing(60, 200, 10)

# Run all algorithms
results = [
    ("OMP",    OMP(A, b; sparsity=10)),
    ("CoSaMP", CoSaMP(A, b; sparsity=10)),
    ("IHT",    IHT(A, b; sparsity=10)),
    ("FISTA",  FISTA(A, b; lambda=0.1)),
    ("SL0",    SL0(A, b)),
    ("L0EM",   L0EM(A, b)),
]

# Compare errors
for (name, x_rec) in results
    err = norm(x_rec - x_true) / norm(x_true)
    println("$name: $(round(err * 100, digits=2))% error")
end
```

## Dependencies

- [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) (stdlib)
- [Random](https://docs.julialang.org/en/v1/stdlib/Random/) (stdlib)
- [Convex.jl](https://github.com/jump-dev/Convex.jl) - Disciplined convex programming
- [SCS.jl](https://github.com/jump-dev/SCS.jl) - Splitting Conic Solver

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
