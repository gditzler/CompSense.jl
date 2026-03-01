# CompSense.jl

A Julia package for **compressed sensing** and **sparse signal recovery**.

## Features

- **21 sparse recovery algorithms** — greedy pursuit, thresholding, proximal, convex optimization, smoothing, combinatorial, and matrix recovery
- **8 sensing matrix generators** — Gaussian, Bernoulli, Fourier, DCT, Hadamard, Sparse, Uniform, Toeplitz
- **Basis / dictionary support** — recover signals sparse in an arbitrary basis via `recover_in_basis`
- **1-bit compressed sensing** — `onebit_sensing` + `BIHT`
- **Multiple Measurement Vectors (MMV)** — `generate_mmv_problem` + `SOMP`
- **Matrix completion** — `generate_matrix_completion_problem` + `SVT`
- **Recovery metrics** — `recovery_error`, `support_recovery`, `snr`, `nmse`, `phase_transition`
- **Sensing matrix analysis** — `mutual_coherence`, `babel_function`, `spark`, `column_coherence_matrix`

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/gditzler/CompSense.jl")
```

## Quick Start

```julia
using CompSense

# Generate a synthetic compressed sensing problem
# 50 measurements, 200-dimensional signal, 10 non-zero entries
A, x_true, b = gaussian_sensing(50, 200, 10)

# Recover the sparse signal
x_omp = OMP(A, b; sparsity=10)
x_fista = FISTA(A, b; lambda=0.1)
x_admm = ADMM(A, b; lambda=0.1)

# Evaluate recovery quality
println("OMP error:   ", recovery_error(x_omp, x_true))
println("FISTA error: ", recovery_error(x_fista, x_true))
println("ADMM error:  ", recovery_error(x_admm, x_true))

prec, rec, f1 = support_recovery(x_omp, x_true)
println("OMP support F1: ", f1)
```

## Contents

```@contents
Pages = [
    "algorithms.md",
    "sensing.md",
    "basis.md",
    "utilities.md",
    "metrics.md",
    "analysis.md",
]
Depth = 2
```
