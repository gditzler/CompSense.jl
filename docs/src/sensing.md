# Sensing Matrices

CompSense provides **8 sensing matrix generators** for benchmarking sparse recovery algorithms.

| Matrix Type | Function | Fast Transform | Notes |
|:------------|:---------|:--------------:|:------|
| Gaussian | [`gaussian_sensing`](@ref) | no | Gold standard, satisfies RIP |
| Bernoulli | [`bernoulli_sensing`](@ref) | no | ±1 entries, simple |
| Fourier | [`fourier_sensing`](@ref) | O(n log n) | MRI, radar, spectroscopy |
| DCT | [`dct_sensing`](@ref) | O(n log n) | JPEG, MPEG |
| Hadamard | [`hadamard_sensing`](@ref) | O(n log n) | Requires n = 2^k |
| Sparse | [`sparse_sensing`](@ref) | no | Large-scale problems |
| Uniform | [`uniform_sensing`](@ref) | no | Bounded entries |
| Toeplitz | [`toeplitz_sensing`](@ref) | O(n log n) | Convolution/LTI systems |

## API Reference

```@docs
gaussian_sensing
bernoulli_sensing
fourier_sensing
dct_sensing
hadamard_sensing
sparse_sensing
uniform_sensing
toeplitz_sensing
```
