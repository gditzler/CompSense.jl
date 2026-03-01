# Copyright (c) 2026
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Note: LinearAlgebra is imported by the parent module

"""
    recovery_error(x_hat, x_true)

Compute the relative L2 recovery error: ‖x̂ - x‖₂ / ‖x‖₂.

Returns `Inf` if `x_true` is the zero vector.

# Arguments
- `x_hat::AbstractVector`: Recovered signal
- `x_true::AbstractVector`: True signal

# Returns
- `Float64`: Relative L2 error

# Example
```julia
x_true = [1.0, 0.0, 0.0, 2.0]
x_hat  = [0.99, 0.01, 0.0, 1.98]
recovery_error(x_hat, x_true)  # ≈ 0.013
```
"""
function recovery_error(x_hat::AbstractVector, x_true::AbstractVector)
    x_norm = norm(x_true)
    if x_norm == 0
        return Inf
    end
    return norm(x_hat - x_true) / x_norm
end

"""
    support_recovery(x_hat, x_true; tol=1e-6)

Compute precision, recall, and F1 score for support set recovery.

The support of a vector is the set of indices with magnitude above `tol`.

# Arguments
- `x_hat::AbstractVector`: Recovered signal
- `x_true::AbstractVector`: True signal
- `tol::Real`: Threshold for determining non-zero entries (default: 1e-6)

# Returns
- `NamedTuple{(:precision, :recall, :f1)}`: Support recovery metrics
  - `precision`: fraction of recovered support that is correct
  - `recall`: fraction of true support that is recovered
  - `f1`: harmonic mean of precision and recall

# Example
```julia
x_true = [1.0, 0.0, 0.0, 2.0, 0.0]
x_hat  = [0.99, 0.0, 0.0, 1.98, 0.01]
support_recovery(x_hat, x_true)  # (precision=0.667, recall=1.0, f1=0.8)
```
"""
function support_recovery(x_hat::AbstractVector, x_true::AbstractVector; tol::Real=1e-6)
    supp_hat = Set(findall(xi -> abs(xi) > tol, x_hat))
    supp_true = Set(findall(xi -> abs(xi) > tol, x_true))

    if isempty(supp_true) && isempty(supp_hat)
        return (precision=1.0, recall=1.0, f1=1.0)
    end

    tp = length(intersect(supp_hat, supp_true))

    precision = isempty(supp_hat) ? 0.0 : tp / length(supp_hat)
    recall = isempty(supp_true) ? 0.0 : tp / length(supp_true)

    if precision + recall == 0
        f1 = 0.0
    else
        f1 = 2 * precision * recall / (precision + recall)
    end

    return (precision=precision, recall=recall, f1=f1)
end

"""
    snr(x_hat, x_true)

Compute signal-to-noise ratio in dB: 10 * log10(‖x‖₂² / ‖x̂ - x‖₂²).

Returns `Inf` if recovery is exact, `-Inf` if `x_true` is the zero vector.

# Arguments
- `x_hat::AbstractVector`: Recovered signal
- `x_true::AbstractVector`: True signal

# Returns
- `Float64`: SNR in dB

# Example
```julia
x_true = [1.0, 0.0, 0.0, 2.0]
x_hat  = [0.99, 0.01, 0.0, 1.98]
snr(x_hat, x_true)  # ≈ 37.4 dB
```
"""
function snr(x_hat::AbstractVector, x_true::AbstractVector)
    signal_power = norm(x_true)^2
    noise_power = norm(x_hat - x_true)^2

    if signal_power == 0
        return -Inf
    end
    if noise_power == 0
        return Inf
    end

    return 10 * log10(signal_power / noise_power)
end

"""
    nmse(x_hat, x_true)

Compute normalized mean squared error: ‖x̂ - x‖₂² / ‖x‖₂².

Returns `Inf` if `x_true` is the zero vector.

# Arguments
- `x_hat::AbstractVector`: Recovered signal
- `x_true::AbstractVector`: True signal

# Returns
- `Float64`: Normalized MSE

# Example
```julia
x_true = [1.0, 0.0, 0.0, 2.0]
x_hat  = [0.99, 0.01, 0.0, 1.98]
nmse(x_hat, x_true)  # ≈ 0.00018
```
"""
function nmse(x_hat::AbstractVector, x_true::AbstractVector)
    x_norm_sq = norm(x_true)^2
    if x_norm_sq == 0
        return Inf
    end
    return norm(x_hat - x_true)^2 / x_norm_sq
end

"""
    phase_transition(algorithm, n_range, k_range, p; trials=10, kwargs...)

Run an automated phase transition sweep for a sparse recovery algorithm.

For each (n, k) pair, runs `trials` random experiments and records the
fraction of successful recoveries (relative error < `success_tol`).

# Arguments
- `algorithm::Function`: Recovery algorithm with signature `algorithm(A, b; kwargs...)`
- `n_range`: Range or vector of measurement counts
- `k_range`: Range or vector of sparsity levels
- `p::Integer`: Signal dimension
- `trials::Int`: Number of random trials per (n, k) pair (default: 10)
- `success_tol::Real`: Relative error threshold for success (default: 1e-2)
- `kwargs...`: Additional keyword arguments passed to `algorithm`

# Returns
- `Matrix{Float64}`: Success probability matrix of size `(length(n_range), length(k_range))`
  where entry (i, j) is the fraction of successful recoveries for (n_range[i], k_range[j])

# Example
```julia
result = phase_transition(OMP, 20:10:80, 1:5:30, 200; trials=20, sparsity_from_k=true)
```

# Notes
- When `sparsity_from_k=true`, passes `sparsity=k` to the algorithm for each trial.
  This is useful for algorithms like OMP, IHT, CoSaMP that require a sparsity parameter.
- When `sparsity_from_k=false` (default), does not pass sparsity.
"""
function phase_transition(algorithm::Function,
                          n_range,
                          k_range,
                          p::Integer;
                          trials::Int=10,
                          success_tol::Real=1e-2,
                          sparsity_from_k::Bool=false,
                          kwargs...)
    n_vals = collect(n_range)
    k_vals = collect(k_range)
    result = zeros(length(n_vals), length(k_vals))

    for (i, n) in enumerate(n_vals)
        for (j, k) in enumerate(k_vals)
            if k > n || k <= 0 || n > p
                result[i, j] = 0.0
                continue
            end

            successes = 0
            for _ in 1:trials
                A, x_true, b = gaussian_sensing(n, p, k)

                try
                    x_hat = if sparsity_from_k
                        algorithm(A, b; sparsity=k, kwargs...)
                    else
                        algorithm(A, b; kwargs...)
                    end

                    if recovery_error(x_hat, x_true) < success_tol
                        successes += 1
                    end
                catch
                    # Algorithm failure counts as unsuccessful
                end
            end

            result[i, j] = successes / trials
        end
    end

    return result
end
