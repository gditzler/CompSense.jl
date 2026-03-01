# CompSense.jl

A Julia package for **compressed sensing** and **sparse signal recovery**. CompSense.jl provides efficient implementations of **21 algorithms** to solve the underdetermined system Ax = b, where x is known to be sparse — including support for 1-bit compressed sensing, multiple measurement vectors (MMV), matrix completion, and recovery in arbitrary bases/dictionaries.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/gditzler/CompSense.jl")
```

Or in the Julia REPL package mode (press `]`):

```
add https://github.com/gditzler/CompSense.jl
```

## Documentation

Full API documentation: [https://gditzler.github.io/CompSense.jl](https://gditzler.github.io/CompSense.jl)

## Quick Start

```julia
using CompSense

# Generate a synthetic compressed sensing problem
# 50 measurements, 200-dimensional signal, 10 non-zero entries
A, x_true, b = gaussian_sensing(50, 200, 10)

# Recover the sparse signal using different algorithms
x_omp = OMP(A, b; sparsity=10)         # Orthogonal Matching Pursuit
x_iht = IHT(A, b; sparsity=10)         # Iterative Hard Thresholding
x_fista = FISTA(A, b; lambda=0.1)      # Fast Iterative Shrinkage-Thresholding
x_admm = ADMM(A, b; lambda=0.1)       # Alternating Direction Method of Multipliers
x_sl0 = SL0(A, b)                      # Smoothed L0

# Check recovery quality with built-in metrics
println("OMP error:   ", recovery_error(x_omp, x_true))
println("FISTA error: ", recovery_error(x_fista, x_true))
println("ADMM error:  ", recovery_error(x_admm, x_true))

# Support recovery analysis
prec, rec, f1 = support_recovery(x_omp, x_true)
println("OMP support — precision: $prec, recall: $rec, F1: $f1")
```

## Sparse Recovery Algorithms

CompSense provides **21 algorithms** for sparse signal recovery:

### Standard Algorithms (17)

| Algorithm | Type | Sparsity | Best For |
|:----------|:-----|:---------|:---------|
| **OMP** | Greedy | Exact (k) | Known sparsity, baseline |
| **CoSaMP** | Greedy | Exact (k) | Theoretical guarantees |
| **SP** | Greedy | Exact (k) | Fast greedy with subspace steps |
| **IHT** | Thresholding | Exact (k) | Simple, well-conditioned A |
| **NIHT** | Thresholding | Exact (k) | Adaptive step size |
| **FISTA** | Proximal | Approximate | Noisy data, LASSO |
| **ADMM** | Proximal | Approximate | Scalable L1 regularization |
| **AMP** | Message Passing | Approximate | Large i.i.d. Gaussian A |
| **LASSO** | Convex | Approximate | Variable selection |
| **BPDN** | Convex | Approximate | Noisy L1 minimization |
| **BasisPursuit** | Convex | Approximate | Exact L1 minimization |
| **IRWLS** | Convex | Approximate | High-quality solutions |
| **ReweightedL1** | Convex | Approximate | Enhanced sparsity via reweighting |
| **SL0** | Smoothing | Approximate | General purpose, fast |
| **L0EM** | EM | Approximate | Balance speed/accuracy |
| **AKRON** | Combinatorial | Exact | Data-driven subspace learning |
| **KRON** | Combinatorial | Exact | Exact combinatorial recovery |

### Extended Algorithms (4)

| Algorithm | Type | Problem | Best For |
|:----------|:-----|:--------|:---------|
| **BIHT** | Thresholding | 1-bit CS | Binary/sign measurements |
| **SOMP** | Greedy | MMV | Joint sparse support recovery |
| **GroupLASSO** | Regularization | Group sparsity | Structured sparsity patterns |
| **SVT** | Nuclear norm | Matrix completion | Low-rank matrix recovery |

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

### SP - Subspace Pursuit

Greedy algorithm similar to CoSaMP with subspace refinement steps.

```julia
x = SP(A, b; sparsity=10, maxiter=100, tol=1e-6)
```

**Reference:** Dai & Milenkovic, "Subspace Pursuit for Compressive Sensing Signal Reconstruction," IEEE Trans. Info. Theory, 2009.

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

### NIHT - Normalized Iterative Hard Thresholding

IHT variant with adaptive step size normalization for improved convergence.

```julia
x = NIHT(A, b; sparsity=10, maxiter=500, tol=1e-6)
```

**Reference:** Blumensath & Davies, "Normalized Iterative Hard Thresholding: Guaranteed Stability and Performance," IEEE J. Selected Topics in Signal Processing, 2010.

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

### ADMM - Alternating Direction Method of Multipliers

Scalable proximal method for L1-regularized least squares via variable splitting.

```julia
x = ADMM(A, b; lambda=0.1, rho=1.0, maxiter=500, tol=1e-6)
```

**Reference:** Boyd et al., "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers," Foundations and Trends in Machine Learning, 2011.

---

### AMP - Approximate Message Passing

Message passing algorithm with Onsager correction, optimal for i.i.d. Gaussian matrices.

```julia
x = AMP(A, b; maxiter=500, tol=1e-6)
```

**Reference:** Donoho, Maleki & Montanari, "Message Passing Algorithms for Compressed Sensing," PNAS, 2009.

---

### LASSO

Least Absolute Shrinkage and Selection Operator via convex optimization.

```julia
x = LASSO(A, b; lambda=0.1, epsilon=1e-4)
```

**Reference:** Tibshirani, "Regression Shrinkage and Selection via the Lasso," J. Royal Statistical Society, Series B, 1996.

---

### BPDN - Basis Pursuit Denoising

L1 minimization subject to a noise-bounded data fidelity constraint.

```julia
x = BPDN(A, b; sigma=0.1, epsilon=1e-4)
```

**Reference:** Chen, Donoho & Saunders, "Atomic Decomposition by Basis Pursuit," SIAM Review, 2001.

---

### BasisPursuit

Exact L1 minimization subject to Ax = b.

```julia
x = BasisPursuit(A, b; epsilon=1e-4)
```

**Reference:** Chen, Donoho & Saunders, "Atomic Decomposition by Basis Pursuit," SIAM Review, 2001.

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

---

### ReweightedL1 - Reweighted L1 Minimization

Iterative reweighted L1 approach for enhanced sparsity recovery.

```julia
x = ReweightedL1(A, b; maxiter=100, epsilon=0.01)
```

**Reference:** Candès, Wakin & Boyd, "Enhancing Sparsity by Reweighted L1 Minimization," J. Fourier Anal. Appl., 2008.

---

### AKRON - Approximate Kernel RecOnstructioN

Data-driven combinatorial approach using kernel approximations.

```julia
x = AKRON(A, b; shift=3, sparsity_threshold=1e-3)
```

**Reference:** Ditzler & Bouaynaya, "Approximate Kernel Reconstruction for Data-Driven Subspace Learning," 2019.

---

### KRON - Kernel RecOnstructioN

Exact combinatorial recovery via kernel enumeration.

```julia
x = KRON(A, b; epsilon=1e-6)
```

**Reference:** Bayar, Bouaynaya & Shterenberg, ["Kernel Reconstruction: An Exact Greedy Algorithm for Compressive Sensing"](https://ieeexplore.ieee.org/abstract/document/7032355/), IEEE GlobalSIP, 2014.

---

### BIHT - Binary Iterative Hard Thresholding

Recovery from 1-bit (sign) compressed sensing measurements.

```julia
x = BIHT(A, y; sparsity=10, maxiter=1000, tol=1e-6, tau=nothing)
```

**Reference:** Jacques et al., "Robust 1-Bit Compressive Sensing via Binary Stable Embeddings of Sparse Vectors," IEEE Trans. Info. Theory, 2013.

---

### SOMP - Simultaneous Orthogonal Matching Pursuit

Greedy joint sparse recovery for Multiple Measurement Vectors (MMV).

```julia
X = SOMP(A, B; sparsity=10, tol=1e-6, maxiter=nothing)
```

**Reference:** Tropp, Gilbert & Strauss, "Algorithms for Simultaneous Sparse Approximation," Signal Processing, 2006.

---

### GroupLASSO - Group LASSO

ADMM-based group sparsity regularization for structured recovery.

```julia
x = GroupLASSO(A, b, groups; lambda=0.1, rho=1.0, maxiter=500, tol=1e-6)
```

**Reference:** Yuan & Lin, "Model selection and estimation in regression with grouped variables," J. Royal Statistical Society, 2006.

---

### SVT - Singular Value Thresholding

Low-rank matrix recovery from partially observed entries.

```julia
M = SVT(Omega, values, m, n; tau=nothing, delta=nothing, maxiter=500, tol=1e-4)
```

**Reference:** Cai, Candès & Shen, "A Singular Value Thresholding Algorithm for Matrix Completion," SIAM J. Optimization, 2010.

---

## Basis / Dictionary Support

When a signal is sparse in a known basis Ψ (e.g., DCT), use `recover_in_basis` to recover it transparently:

```julia
using CompSense

A, x_true, b = gaussian_sensing(50, 200, 10)
Psi = dct_matrix(200)

# Recover signal sparse in DCT basis
x_hat = recover_in_basis(A, b, Psi, OMP; sparsity=10)
println("Error: ", recovery_error(x_hat, x_true))
```

## 1-Bit Compressed Sensing

Recover sparse signals from sign measurements using `BIHT`:

```julia
using CompSense

A, x_true, y = onebit_sensing(80, 200, 10)
x_hat = BIHT(A, y; sparsity=10)
```

## Multiple Measurement Vectors (MMV)

Jointly recover signals sharing a common sparse support using `SOMP`:

```julia
using CompSense

A, X_true, B = generate_mmv_problem(60, 200, 8, 5)  # 5 measurement vectors
X_hat = SOMP(A, B; sparsity=8)
```

## Matrix Completion

Recover a low-rank matrix from partially observed entries using `SVT`:

```julia
using CompSense

Omega, values, M_true, m, n = generate_matrix_completion_problem(50, 50, 5, 0.5)
M_hat = SVT(Omega, values, m, n)
```

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

## Recovery Metrics

| Function | Description |
|:---------|:------------|
| `recovery_error(x_hat, x_true)` | Relative L2 error: ‖x̂ - x‖₂ / ‖x‖₂ |
| `support_recovery(x_hat, x_true)` | Precision, recall, F1 for support set |
| `snr(x_hat, x_true)` | Signal-to-noise ratio in dB |
| `nmse(x_hat, x_true)` | Normalized mean squared error |
| `phase_transition(alg, n_range, k_range, p)` | Automated phase transition sweep |

## Sensing Matrix Analysis

| Function | Description |
|:---------|:------------|
| `mutual_coherence(A)` | Maximum absolute inner product of normalized columns |
| `babel_function(A, k)` | Cumulative coherence μ₁(k) |
| `spark(A)` | Smallest number of linearly dependent columns |
| `column_coherence_matrix(A)` | Full Gram matrix of normalized columns |

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

## Performance Tips

1. **Fastest algorithms:** SL0 and IHT for most problems
2. **Known sparsity:** Use OMP, IHT, CoSaMP, or SP
3. **Noisy measurements:** Use FISTA or ADMM with tuned lambda
4. **Theoretical guarantees:** Use CoSaMP
5. **Highest quality:** Use IRWLS or ReweightedL1 (but slower)
6. **Very large problems:** Reduce `maxiter`, use sparse matrices
7. **Adaptive step size:** Use NIHT over IHT
8. **1-bit measurements:** Use BIHT
9. **Joint sparse recovery:** Use SOMP for MMV problems

## Example: Comparing All Algorithms

```julia
using CompSense
using LinearAlgebra

# Create test problem
A, x_true, b = gaussian_sensing(60, 200, 10)

# Run algorithms
results = [
    ("OMP",    OMP(A, b; sparsity=10)),
    ("CoSaMP", CoSaMP(A, b; sparsity=10)),
    ("SP",     SP(A, b; sparsity=10)),
    ("IHT",    IHT(A, b; sparsity=10)),
    ("NIHT",   NIHT(A, b; sparsity=10)),
    ("FISTA",  FISTA(A, b; lambda=0.1)),
    ("ADMM",   ADMM(A, b; lambda=0.1)),
    ("AMP",    AMP(A, b)),
    ("SL0",    SL0(A, b)),
    ("L0EM",   L0EM(A, b)),
]

# Compare errors
for (name, x_rec) in results
    err = recovery_error(x_rec, x_true)
    println("$name: $(round(err * 100, digits=2))% error")
end
```

## Dependencies

- [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) (stdlib)
- [Random](https://docs.julialang.org/en/v1/stdlib/Random/) (stdlib)
- [Convex.jl](https://github.com/jump-dev/Convex.jl) - Disciplined convex programming
- [SCS.jl](https://github.com/jump-dev/SCS.jl) - Splitting Conic Solver
- [Combinatorics.jl](https://github.com/JuliaMath/Combinatorics.jl) - Combinatorial algorithms (used by AKRON/KRON)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
