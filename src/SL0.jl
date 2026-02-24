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
    SL0(A, b; sigma_decrease_factor=0.85, maxiter=150, epsilon=0.001)

Find the solution to Ax=b using the Smoothed L0 (SL0) algorithm.

# Arguments
- `A::AbstractMatrix`: Sensing matrix in Ax=b
- `b::AbstractVector`: Measurement vector in Ax=b
- `sigma_decrease_factor::Real`: Factor by which sigma decreases each iteration (default: 0.85)
- `maxiter::Int`: Maximum number of outer iterations (default: 150)
- `epsilon::Real`: Threshold for zeroing small coefficients (default: 0.001)

# Returns
- `Vector`: Sparse solution x to Ax=b

# Example
```julia
A = randn(10, 100)
b = randn(10)
x = SL0(A, b; sigma_decrease_factor=0.85, maxiter=150)
```

# Algorithm
Implements the Smoothed L0 algorithm from:
> http://ee.sharif.edu/~SLzero/

The algorithm approximates the L0 norm with a smooth Gaussian function
and uses gradient ascent to maximize sparsity while maintaining feasibility.
"""
function SL0(A::AbstractMatrix{T},
             b::AbstractVector{T};
             sigma_decrease_factor::Real=0.85,
             maxiter::Int=150,
             epsilon::Real=0.001) where {T<:Real}

    # Algorithm constants
    mu_0 = T(2)  # Scales the gradient step in steepest ascent
    L = 3        # Number of internal steepest ascent iterations
    eps_T = convert(T, epsilon)

    # Compute pseudo-inverse once (expensive but needed for projection)
    A_pinv = pinv(A)

    # Initialize with minimum norm solution
    s = A_pinv * b
    sigma = 2 * maximum(abs, s)

    # Pre-allocate working arrays for in-place operations
    delta = similar(s)
    residual = similar(b)

    sigma_sq = sigma^2

    for _ in 1:maxiter
        for _ in 1:L
            # Compute gradient of smoothed L0 approximation
            # δ = s * exp(-|s|²/σ²)
            @. delta = s * exp(-abs(s)^2 / sigma_sq)

            # Steepest ascent step
            @. s -= mu_0 * delta

            # Project back to feasible set: s = s - A⁺(As - b)
            mul!(residual, A, s)
            residual .-= b
            s .-= A_pinv * residual
        end

        # Decrease sigma (annealing schedule)
        sigma *= sigma_decrease_factor
        sigma_sq = sigma^2
    end

    # Threshold small values to zero
    x = copy(s)
    @inbounds for i in eachindex(x)
        if abs(x[i]) < eps_T
            x[i] = zero(T)
        end
    end

    return x
end

# Convenience method for mixed numeric types
function SL0(A::AbstractMatrix, b::AbstractVector; kwargs...)
    T = promote_type(eltype(A), eltype(b))
    return SL0(convert(Matrix{T}, A), convert(Vector{T}, b); kwargs...)
end
