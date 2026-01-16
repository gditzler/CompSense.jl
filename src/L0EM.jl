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
    L0EM(A, b; lambda=0.001, epsilon=0.001, maxiter=50)

Find the solution to Ax=b using an efficient EM algorithm that directly
solves the L0 optimization problem.

# Arguments
- `A::AbstractMatrix`: Sensing matrix in Ax=b
- `b::AbstractVector`: Measurement vector in Ax=b
- `lambda::Real`: Regularization parameter (default: 0.001)
- `epsilon::Real`: Convergence threshold (default: 0.001)
- `maxiter::Int`: Maximum number of iterations (default: 50)

# Returns
- `Vector`: Sparse solution x to Ax=b

# Example
```julia
A = randn(10, 100)
b = randn(10)
x = L0EM(A, b; maxiter=50, epsilon=0.001)
```

# Algorithm
Implements Liu and Li's L0-EM algorithm from:
> "L0-EM Algorithm for Sparse Recovery" (https://arxiv.org/pdf/1407.7508v1.pdf)

The algorithm uses an EM framework to iteratively reweight the problem,
effectively solving the L0-regularized least squares problem.
"""
function L0EM(A::AbstractMatrix{T},
              b::AbstractVector{T};
              lambda::Real=0.001,
              epsilon::Real=0.001,
              maxiter::Int=50) where {T<:Real}
    n, p = size(A)

    # Precompute A'*A and A'*b for efficiency
    AtA = A' * A
    Atb = A' * b

    # Get initial solution using regularized least squares
    # Use \ operator instead of inv() for numerical stability and performance
    theta = (AtA + lambda * I) \ Atb

    # Pre-allocate working arrays
    eta = similar(theta)
    eta_sq = similar(theta)
    A_weighted = similar(A)

    for _ in 1:maxiter
        copyto!(eta, theta)

        # Compute eta squared element-wise
        eta_sq .= eta .^ 2

        # Weight each column of A by the corresponding eta^2 value
        # This is more efficient than repeat() + Hadamard product
        @inbounds for j in 1:p
            @views A_weighted[:, j] .= A[:, j] .* eta_sq[j]
        end

        # Update theta using backslash (numerically stable)
        # Solves: (A_weighted' * A + λI) * theta = A_weighted' * b
        theta = (A_weighted' * A + lambda * I) \ (A_weighted' * b)

        # Check convergence
        if norm(theta - eta, 2) <= epsilon
            break
        end
    end

    # Threshold small values to zero
    x = copy(theta)
    x[abs.(x) .< epsilon] .= zero(T)

    return x
end

# Convenience method for mixed numeric types
function L0EM(A::AbstractMatrix, b::AbstractVector; kwargs...)
    T = promote_type(eltype(A), eltype(b))
    return L0EM(convert(Matrix{T}, A), convert(Vector{T}, b); kwargs...)
end
