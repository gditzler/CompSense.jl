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
    soft_threshold!(out, z, threshold)

In-place soft thresholding (proximal operator for L1 norm) element-wise.

Computes: out[i] = sign(z[i]) * max(|z[i]| - threshold, 0)
"""
function soft_threshold!(out::AbstractVector{T}, z::AbstractVector{T}, threshold::Real) where {T<:Real}
    @inbounds for i in eachindex(out, z)
        zi = z[i]
        azi = abs(zi)
        out[i] = azi > threshold ? sign(zi) * (azi - threshold) : zero(T)
    end
    return out
end

"""
    FISTA(A, b; lambda=0.1, maxiter=500, tol=1e-6, L=nothing)

Find the solution to Ax=b using the Fast Iterative Shrinkage-Thresholding
Algorithm (FISTA) for LASSO/L1-regularized least squares.

Solves: min_x  (1/2)‖Ax - b‖₂² + λ‖x‖₁

# Arguments
- `A::AbstractMatrix`: Sensing matrix in Ax=b (m × n)
- `b::AbstractVector`: Measurement vector in Ax=b (m × 1)
- `lambda::Real`: L1 regularization parameter (default: 0.1)
- `maxiter::Int`: Maximum number of iterations (default: 500)
- `tol::Real`: Convergence tolerance on relative change (default: 1e-6)
- `L::Union{Real,Nothing}`: Lipschitz constant of ∇f; if nothing, computed as ‖AᵀA‖₂ (default: nothing)

# Returns
- `Vector`: Sparse solution x

# Example
```julia
A = randn(50, 200)
x_true = zeros(200)
x_true[rand(1:200, 10)] = randn(10)
b = A * x_true + 0.01 * randn(50)
x = FISTA(A, b; lambda=0.1)
```

# Algorithm
Implements the accelerated proximal gradient method from:

> Beck, A. and Teboulle, M., "A Fast Iterative Shrinkage-Thresholding
> Algorithm for Linear Inverse Problems," SIAM J. Imaging Sciences, 2009.

The algorithm uses Nesterov momentum to achieve O(1/k²) convergence rate,
compared to O(1/k) for standard ISTA.
"""
function FISTA(A::AbstractMatrix{T},
               b::AbstractVector{T};
               lambda::Real=0.1,
               maxiter::Int=500,
               tol::Real=1e-6,
               L::Union{Real,Nothing}=nothing) where {T<:Real}
    _, n = size(A)

    # Compute Lipschitz constant if not provided
    # L = largest eigenvalue of A'A = ‖A‖₂²
    if isnothing(L)
        # Use power iteration for largest singular value
        L_const = opnorm(A)^2
    else
        L_const = T(L)
    end

    # Precompute A'A and A'b for efficiency
    AtA = A' * A
    Atb = A' * b

    lambda_T = convert(T, lambda)

    # Initialize
    x = zeros(T, n)
    x_old = zeros(T, n)
    y = zeros(T, n)
    t = one(T)

    # Step size
    step = one(T) / L_const

    # Pre-allocate working arrays
    grad = similar(x)
    z = similar(x)

    for iter in 1:maxiter
        # Compute gradient: ∇f(y) = A'(Ay - b) = A'Ay - A'b
        mul!(grad, AtA, y)
        grad .-= Atb

        # Gradient step (fused broadcast, no allocation)
        @. z = y - step * grad

        # Proximal step (in-place soft thresholding)
        copyto!(x_old, x)
        soft_threshold!(x, z, lambda_T * step)

        # Nesterov momentum update
        t_old = t
        t = (one(T) + sqrt(one(T) + 4 * t^2)) / 2

        # Update y with momentum
        momentum = (t_old - one(T)) / t
        @. y = x + momentum * (x - x_old)

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
function FISTA(A::AbstractMatrix, b::AbstractVector; kwargs...)
    T = promote_type(eltype(A), eltype(b))
    return FISTA(convert(Matrix{T}, A), convert(Vector{T}, b); kwargs...)
end
