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
    hard_threshold!(x, k)

In-place hard thresholding: keep only the k largest magnitude entries,
set the rest to zero.

# Arguments
- `x::AbstractVector`: Vector to threshold (modified in-place)
- `k::Int`: Number of entries to keep

# Returns
- `Vector{Int}`: Indices of the k largest entries (support set)
"""
function hard_threshold!(x::AbstractVector{T}, k::Int) where {T<:Real}
    n = length(x)
    k = min(k, n)

    # Find indices of k largest entries by magnitude
    perm = sortperm(abs.(x), rev=true)
    support = perm[1:k]

    # Zero out entries not in support
    @inbounds for i in (k+1):n
        x[perm[i]] = zero(T)
    end

    return sort(support)
end

"""
    IHT(A, b; sparsity, maxiter=500, tol=1e-6, mu=nothing)

Find the solution to Ax=b using Iterative Hard Thresholding (IHT).

Solves: min_x ‖Ax - b‖₂²  subject to ‖x‖₀ ≤ k

# Arguments
- `A::AbstractMatrix`: Sensing matrix in Ax=b (m × n)
- `b::AbstractVector`: Measurement vector in Ax=b (m × 1)
- `sparsity::Int`: Target sparsity level k (required)
- `maxiter::Int`: Maximum number of iterations (default: 500)
- `tol::Real`: Convergence tolerance on relative change (default: 1e-6)
- `mu::Union{Real,Nothing}`: Step size; if nothing, uses 1/‖A‖₂² (default: nothing)

# Returns
- `Vector`: k-sparse solution x to Ax=b

# Example
```julia
A = randn(50, 200)
x_true = zeros(200)
x_true[rand(1:200, 10)] = randn(10)
b = A * x_true
x = IHT(A, b; sparsity=10)
```

# Algorithm
Implements the Iterative Hard Thresholding algorithm:

1. Gradient step: z = x + μ * Aᵀ(b - Ax)
2. Hard thresholding: x = H_k(z) (keep k largest entries)
3. Repeat until convergence

Reference:
> Blumensath, T. and Davies, M.E., "Iterative Hard Thresholding for
> Compressed Sensing," Applied and Computational Harmonic Analysis, 2009.
"""
function IHT(A::AbstractMatrix{T},
             b::AbstractVector{T};
             sparsity::Int,
             maxiter::Int=500,
             tol::Real=1e-6,
             mu::Union{Real,Nothing}=nothing) where {T<:Real}
    _, n = size(A)

    # Compute step size if not provided
    # Use normalized IHT step size: μ = 1/‖A‖₂²
    if isnothing(mu)
        mu_val = one(T) / opnorm(A)^2
    else
        mu_val = T(mu)
    end

    # Precompute A'b
    Atb = A' * b

    # Initialize
    x = zeros(T, n)
    x_old = similar(x)

    # Pre-allocate working arrays
    residual = similar(b)
    gradient = similar(x)

    for _ in 1:maxiter
        copyto!(x_old, x)

        # Compute residual: r = b - Ax
        mul!(residual, A, x)
        residual .= b .- residual

        # Gradient step: z = x + μ * A'r
        mul!(gradient, A', residual)
        @. x = x + mu_val * gradient

        # Hard thresholding: keep k largest entries
        hard_threshold!(x, sparsity)

        # Check convergence
        x_norm = norm(x)
        if x_norm > eps(T)
            rel_change = norm(x - x_old) / x_norm
            if rel_change < tol
                break
            end
        elseif norm(x - x_old) < tol
            break
        end
    end

    return x
end

# Convenience method for mixed numeric types
function IHT(A::AbstractMatrix, b::AbstractVector; kwargs...)
    T = promote_type(eltype(A), eltype(b))
    return IHT(convert(Matrix{T}, A), convert(Vector{T}, b); kwargs...)
end
