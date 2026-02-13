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
    OMP(A, b; sparsity=nothing, tol=1e-6, maxiter=nothing)

Find the solution to Ax=b using Orthogonal Matching Pursuit (OMP).

# Arguments
- `A::AbstractMatrix`: Sensing matrix in Ax=b (m × n)
- `b::AbstractVector`: Measurement vector in Ax=b (m × 1)
- `sparsity::Union{Int,Nothing}`: Target sparsity level k (default: m)
- `tol::Real`: Residual norm tolerance for early stopping (default: 1e-6)
- `maxiter::Union{Int,Nothing}`: Maximum iterations (default: sparsity)

# Returns
- `Vector`: Sparse solution x to Ax=b

# Example
```julia
A = randn(50, 200)
x_true = zeros(200)
x_true[rand(1:200, 10)] = randn(10)
b = A * x_true
x = OMP(A, b; sparsity=10)
```

# Algorithm
Implements the standard Orthogonal Matching Pursuit algorithm:

1. Initialize residual r = b, support set Ω = ∅
2. Find column of A most correlated with residual
3. Add index to support set Ω
4. Solve least squares: x_Ω = argmin ‖A_Ω x - b‖₂
5. Update residual: r = b - A_Ω x_Ω
6. Repeat until sparsity reached or ‖r‖₂ < tol

Reference:
> Tropp, J.A. and Gilbert, A.C., "Signal Recovery From Random Measurements
> Via Orthogonal Matching Pursuit," IEEE Trans. Info. Theory, 2007.
"""
function OMP(A::AbstractMatrix{T},
             b::AbstractVector{T};
             sparsity::Union{Int,Nothing}=nothing,
             tol::Real=1e-6,
             maxiter::Union{Int,Nothing}=nothing) where {T<:Real}
    m, n = size(A)

    # Default sparsity to number of measurements
    k = isnothing(sparsity) ? m : min(sparsity, m)
    max_iters = isnothing(maxiter) ? k : maxiter

    # Initialize
    x = zeros(T, n)
    residual = copy(b)
    support = Int[]

    # Precompute column norms for correlation normalization
    col_norms = [norm(A[:, j]) for j in 1:n]

    for _ in 1:max_iters
        # Find column most correlated with residual
        correlations = A' * residual

        # Normalize by column norms to get proper correlation
        @inbounds for j in 1:n
            if col_norms[j] > eps(T)
                correlations[j] /= col_norms[j]
            end
        end

        # Exclude already selected indices
        @inbounds for j in support
            correlations[j] = zero(T)
        end

        # Select index with maximum absolute correlation
        _, idx = findmax(abs.(correlations))
        push!(support, idx)

        # Solve least squares on current support
        A_support = A[:, support]
        x_support = A_support \ b

        # Update solution
        fill!(x, zero(T))
        @inbounds for (i, j) in enumerate(support)
            x[j] = x_support[i]
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
function OMP(A::AbstractMatrix, b::AbstractVector; kwargs...)
    T = promote_type(eltype(A), eltype(b))
    return OMP(convert(Matrix{T}, A), convert(Vector{T}, b); kwargs...)
end
