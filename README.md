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
A, x_true, b = cs_model(50, 200, 10)

# Recover the sparse signal using Smoothed L0
x_recovered = SL0(A, b)

# Check recovery quality
using LinearAlgebra
println("Recovery error: ", norm(x_recovered - x_true))
```

## Algorithms

### SL0 - Smoothed L0

The Smoothed L0 algorithm approximates the L0 norm with a smooth Gaussian function and uses gradient ascent to maximize sparsity.

```julia
x = SL0(A, b; sigma_decrease_factor=0.85, maxiter=150, epsilon=0.001)
```

**Parameters:**
- `sigma_decrease_factor`: Annealing rate for the smoothing parameter (default: 0.85)
- `maxiter`: Maximum number of outer iterations (default: 150)
- `epsilon`: Threshold for zeroing small coefficients (default: 0.001)

**Reference:** [Smoothed L0 (SL0)](http://ee.sharif.edu/~SLzero/)

---

### L0EM - L0 Expectation-Maximization

An EM-based algorithm that directly solves the L0-regularized optimization problem.

```julia
x = L0EM(A, b; lambda=0.001, epsilon=0.001, maxiter=50)
```

**Parameters:**
- `lambda`: Regularization parameter (default: 0.001)
- `epsilon`: Convergence threshold (default: 0.001)
- `maxiter`: Maximum iterations (default: 50)

**Reference:** Liu and Li, ["L0-EM Algorithm for Sparse Recovery"](https://arxiv.org/pdf/1407.7508v1.pdf)

---

### IRWLS - Iteratively Reweighted Least Squares

Enhances sparsity by iteratively solving reweighted L1 minimization problems.

```julia
x = IRWLS(A, b; maxiter=100, epsilon=0.01)
```

**Parameters:**
- `maxiter`: Maximum number of iterations (default: 100)
- `epsilon`: Convergence threshold and regularization (default: 0.01)

**Reference:** Candès, Wakin, and Boyd, ["Enhancing Sparsity by Reweighted L1 Minimization"](https://web.stanford.edu/~boyd/papers/pdf/rwl1.pdf), J Fourier Anal Appl (2008) 14: 877–905.

> **Note:** IRWLS uses convex optimization (via [Convex.jl](https://github.com/jump-dev/Convex.jl) and [SCS.jl](https://github.com/jump-dev/SCS.jl)) and is slower than SL0 and L0EM for large problems.

## Utilities

### cs_model - Generate Test Problems

Create synthetic compressed sensing problems for testing:

```julia
A, x_true, b = cs_model(n, p, k; type="Gaussian")
```

**Parameters:**
- `n`: Number of measurements (rows)
- `p`: Signal dimension (columns)
- `k`: Number of non-zero entries in the true signal
- `type`: Sensing matrix type (currently only "Gaussian" supported)

**Returns:**
- `A`: n × p sensing matrix with full row rank
- `x_true`: Sparse signal with exactly k non-zeros
- `b`: Measurement vector b = A * x_true

## Performance Tips

1. **SL0 is fastest** for most problems due to its gradient-based approach
2. **L0EM** provides a good balance of speed and accuracy
3. **IRWLS** gives high-quality solutions but is slower due to convex optimization
4. For very large problems, consider reducing `maxiter` and tuning `epsilon`

## Example: Comparing Algorithms

```julia
using CompSense
using LinearAlgebra

# Create test problem
A, x_true, b = cs_model(100, 500, 20)

# Compare algorithms
x_sl0 = SL0(A, b)
x_l0em = L0EM(A, b)
x_irwls = IRWLS(A, b)

println("SL0 error:   ", norm(x_sl0 - x_true))
println("L0EM error:  ", norm(x_l0em - x_true))
println("IRWLS error: ", norm(x_irwls - x_true))
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
