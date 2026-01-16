# Copyright (c) 2021
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
    CoSaMP(A, b; sparsity, maxiter=100, tol=1e-6)

Find the solution to Ax=b using Compressive Sampling Matching Pursuit (CoSaMP).

Solves: min_x ‖Ax - b‖₂²  subject to ‖x‖₀ ≤ k

# Arguments
- `A::AbstractMatrix`: Sensing matrix in Ax=b (m × n)
- `b::AbstractVector`: Measurement vector in Ax=b (m × 1)
- `sparsity::Int`: Target sparsity level k (required)
- `maxiter::Int`: Maximum number of iterations (default: 100)
- `tol::Real`: Residual norm tolerance for early stopping (default: 1e-6)

# Returns
- `Vector`: k-sparse solution x to Ax=b

# Example
```julia
A = randn(50, 200)
x_true = zeros(200)
x_true[rand(1:200, 10)] = randn(10)
b = A * x_true
x = CoSaMP(A, b; sparsity=10)
```

# Algorithm
Implements the CoSaMP algorithm:

1. Form signal proxy: u = Aᵀr (correlation with residual)
2. Identify Ω = 2k largest components of u
3. Merge with current support: T = Ω ∪ supp(x)
4. Estimate signal via least squares: x̃ = A_T† b
5. Prune: x = H_k(x̃) (keep k largest entries)
6. Update residual: r = b - Ax
7. Repeat until convergence

Reference:
> Needell, D. and Tropp, J.A., "CoSaMP: Iterative signal recovery from
> incomplete and inaccurate samples," Applied and Computational Harmonic
> Analysis, 2009.
"""
function CoSaMP(A::AbstractMatrix{T},
                b::AbstractVector{T};
                sparsity::Int,
                maxiter::Int=100,
                tol::Real=1e-6) where {T<:Real}
    m, n = size(A)
    k = min(sparsity, m ÷ 2)  # Ensure 2k doesn't exceed m

    # Initialize
    x = zeros(T, n)
    residual = copy(b)

    for _ in 1:maxiter
        # Form signal proxy: correlation with residual
        proxy = A' * residual

        # Identify 2k largest components
        perm = sortperm(abs.(proxy), rev=true)
        omega = perm[1:min(2k, n)]

        # Merge with current support
        current_support = findall(!iszero, x)
        merged_support = sort(union(omega, current_support))

        # Least squares on merged support
        if isempty(merged_support)
            break
        end

        A_merged = A[:, merged_support]
        x_merged = A_merged \ b

        # Create full vector from merged solution
        x_full = zeros(T, n)
        @inbounds for (i, j) in enumerate(merged_support)
            x_full[j] = x_merged[i]
        end

        # Prune to k largest entries
        perm_full = sortperm(abs.(x_full), rev=true)
        final_support = perm_full[1:k]

        # Update x: keep only k largest
        fill!(x, zero(T))
        @inbounds for j in final_support
            x[j] = x_full[j]
        end

        # Update residual
        residual = b - A * x

        # Check convergence
        if norm(residual) < tol
            break
        end
    end

    return x
end

# Convenience method for mixed numeric types
function CoSaMP(A::AbstractMatrix, b::AbstractVector; kwargs...)
    T = promote_type(eltype(A), eltype(b))
    return CoSaMP(convert(Matrix{T}, A), convert(Vector{T}, b); kwargs...)
end
