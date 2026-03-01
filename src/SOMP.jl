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
    SOMP(A, B; sparsity=nothing, tol=1e-6, maxiter=nothing)

Find the solution to AX=B using Simultaneous Orthogonal Matching Pursuit (SOMP)
for Multiple Measurement Vectors (MMV).

Recovers a jointly sparse matrix X where all columns share the same support
(non-zero row indices).

# Arguments
- `A::AbstractMatrix`: Sensing matrix (n x p)
- `B::AbstractMatrix`: Measurement matrix (n x L), each column is a measurement vector
- `sparsity::Union{Int,Nothing}`: Target row-sparsity level k (default: n)
- `tol::Real`: Residual Frobenius norm tolerance for early stopping (default: 1e-6)
- `maxiter::Union{Int,Nothing}`: Maximum iterations (default: sparsity)

# Returns
- `Matrix{Float64}`: p x L solution matrix X with at most k non-zero rows

# Example
```julia
n, p, k, L = 50, 200, 5, 3
A = randn(n, p)
support = sort(randperm(p)[1:k])
X_true = zeros(p, L)
X_true[support, :] = randn(k, L)
B = A * X_true
X_recovered = SOMP(A, B; sparsity=k)
```

# Algorithm
Implements Simultaneous OMP (SOMP / MMV-OMP):

1. Initialize residual R = B, support set Omega = {}
2. Compute correlations C = A'R, select atom maximizing sum of squared correlations
   across all L columns: j* = argmax_j sum_l |C[j,l]|^2
3. Add j* to support set Omega
4. Solve joint least squares: X_Omega = A_Omega \\ B
5. Update residual: R = B - A_Omega * X_Omega
6. Repeat until sparsity reached or ||R||_F < tol

Reference:
> Tropp, J.A., Gilbert, A.C., and Strauss, M.J., "Algorithms for
> Simultaneous Sparse Approximation," Signal Processing, 2006.
"""
function SOMP(A::AbstractMatrix{T},
              B::AbstractMatrix{T};
              sparsity::Union{Int,Nothing}=nothing,
              tol::Real=1e-6,
              maxiter::Union{Int,Nothing}=nothing) where {T<:Real}
    n, p = size(A)
    L = size(B, 2)

    # Default sparsity to number of measurements
    k = isnothing(sparsity) ? n : min(sparsity, n)
    max_iters = isnothing(maxiter) ? k : maxiter

    # Initialize
    X = zeros(T, p, L)
    R = copy(B)
    support = Int[]

    # Pre-allocate correlation buffer
    C = zeros(T, p, L)

    # Precompute column norms for correlation normalization
    col_norms = [norm(@view A[:, j]) for j in 1:p]

    for _ in 1:max_iters
        # Compute correlations: C = A'R
        mul!(C, A', R)

        # Normalize by column norms
        @inbounds for j in 1:p
            if col_norms[j] > eps(T)
                for l in 1:L
                    C[j, l] /= col_norms[j]
                end
            end
        end

        # Exclude already selected indices
        @inbounds for j in support
            for l in 1:L
                C[j, l] = zero(T)
            end
        end

        # Select atom maximizing total squared correlation across all columns
        best_val = zero(T)
        idx = 1
        @inbounds for j in 1:p
            total_sq = zero(T)
            for l in 1:L
                total_sq += C[j, l]^2
            end
            if total_sq > best_val
                best_val = total_sq
                idx = j
            end
        end
        push!(support, idx)

        # Solve joint least squares on current support
        A_support = A[:, support]
        X_support = A_support \ B

        # Update solution
        fill!(X, zero(T))
        @inbounds for (i, j) in enumerate(support)
            for l in 1:L
                X[j, l] = X_support[i, l]
            end
        end

        # Update residual: R = B - A * X
        mul!(R, A, X)
        @. R = B - R

        # Check convergence (Frobenius norm)
        if norm(R) < tol
            break
        end
    end

    return X
end

# Convenience method for mixed numeric types
function SOMP(A::AbstractMatrix, B::AbstractMatrix; kwargs...)
    T = promote_type(eltype(A), eltype(B))
    return SOMP(convert(Matrix{T}, A), convert(Matrix{T}, B); kwargs...)
end
