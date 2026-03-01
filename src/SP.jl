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
    SP(A, b; sparsity, maxiter=100, tol=1e-6)

Find the solution to Ax=b using Subspace Pursuit.

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
x = SP(A, b; sparsity=10)
```

# Algorithm
Implements the Subspace Pursuit algorithm, a close relative of CoSaMP
with different selection and pruning steps:

1. Initialize: select k columns most correlated with b
2. Expand: add k columns most correlated with residual
3. Estimate: least squares on merged 2k support
4. Prune: keep k largest entries
5. Re-estimate: least squares on final k support
6. Repeat until convergence

The key difference from CoSaMP is the re-estimation step after pruning,
which often improves practical performance.

Reference:
> Dai, W. and Milenkovic, O., "Subspace Pursuit for Compressive Sensing
> Signal Reconstruction," IEEE Trans. Information Theory, 2009.
"""
function SP(A::AbstractMatrix{T},
            b::AbstractVector{T};
            sparsity::Int,
            maxiter::Int=100,
            tol::Real=1e-6) where {T<:Real}
    m, n = size(A)
    k = min(sparsity, m)

    # Initialize: select k columns most correlated with b
    proxy = A' * b
    abs_proxy = abs.(proxy)
    perm = sortperm(abs_proxy, rev=true)
    support = sort(perm[1:k])

    # Initial least squares estimate
    x = zeros(T, n)
    x_sub = A[:, support] \ b
    @inbounds for (i, j) in enumerate(support)
        x[j] = x_sub[i]
    end

    # Pre-allocate working arrays
    residual = similar(b)
    x_full = zeros(T, n)
    abs_buf = zeros(T, n)

    for _ in 1:maxiter
        # Compute residual
        mul!(residual, A, x)
        @. residual = b - residual

        # Check convergence
        if norm(residual) < tol
            break
        end

        # Identify k new candidates from residual correlation
        mul!(proxy, A', residual)
        @. abs_buf = abs(proxy)
        perm = sortperm(abs_buf, rev=true)
        new_candidates = perm[1:k]

        # Merge with current support
        merged_support = sort(union(support, new_candidates))

        # Least squares on merged support
        A_merged = A[:, merged_support]
        x_merged = A_merged \ b

        # Full vector from merged solution
        fill!(x_full, zero(T))
        @inbounds for (i, j) in enumerate(merged_support)
            x_full[j] = x_merged[i]
        end

        # Prune to k largest entries
        @. abs_buf = abs(x_full)
        perm_full = sortperm(abs_buf, rev=true)
        support = sort(perm_full[1:k])

        # Re-estimate on pruned support (key difference from CoSaMP)
        A_pruned = A[:, support]
        x_pruned = A_pruned \ b

        fill!(x, zero(T))
        @inbounds for (i, j) in enumerate(support)
            x[j] = x_pruned[i]
        end
    end

    return x
end

# Convenience method for mixed numeric types
function SP(A::AbstractMatrix, b::AbstractVector; kwargs...)
    T = promote_type(eltype(A), eltype(b))
    return SP(convert(Matrix{T}, A), convert(Vector{T}, b); kwargs...)
end
