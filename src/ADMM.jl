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
    ADMM(A, b; lambda=0.1, rho=1.0, maxiter=500, tol=1e-6)

Find the solution to Ax≈b using the Alternating Direction Method of Multipliers
(ADMM) for LASSO / L1-regularized least squares.

Solves: min  ½‖Ax - b‖₂² + λ‖x‖₁

# Arguments
- `A::AbstractMatrix`: Sensing matrix in Ax=b (m × n)
- `b::AbstractVector`: Measurement vector in Ax=b (m × 1)
- `lambda::Real`: L1 regularization parameter (default: 0.1)
- `rho::Real`: ADMM penalty parameter (default: 1.0)
- `maxiter::Int`: Maximum number of iterations (default: 500)
- `tol::Real`: Convergence tolerance on primal/dual residuals (default: 1e-6)

# Returns
- `Vector`: Sparse solution x

# Example
```julia
A = randn(50, 200)
x_true = zeros(200)
x_true[rand(1:200, 10)] = randn(10)
b = A * x_true
x = ADMM(A, b; lambda=0.1)
```

# Algorithm
Reformulates the LASSO as:

    min  ½‖Ax - b‖₂² + λ‖z‖₁  subject to x = z

and applies ADMM splitting:

1. x-update: x = (AᵀA + ρI)⁻¹(Aᵀb + ρ(z - u))
2. z-update: z = S_{λ/ρ}(x + u)  (soft thresholding)
3. u-update: u = u + x - z  (dual variable / scaled residual)

The factorization of (AᵀA + ρI) is computed once and reused.

Reference:
> Boyd, S. et al., "Distributed Optimization and Statistical Learning via
> the Alternating Direction Method of Multipliers," Foundations and Trends
> in Machine Learning, 2011.
"""
function ADMM(A::AbstractMatrix{T},
              b::AbstractVector{T};
              lambda::Real=0.1,
              rho::Real=1.0,
              maxiter::Int=500,
              tol::Real=1e-6) where {T<:Real}
    _, n = size(A)
    rho_T = convert(T, rho)
    lambda_T = convert(T, lambda)
    threshold = lambda_T / rho_T

    # Precompute factorization: (A'A + ρI)
    AtA = A' * A
    @inbounds for i in 1:n
        AtA[i, i] += rho_T
    end
    F = cholesky(Symmetric(AtA))
    Atb = A' * b

    # Initialize
    x = zeros(T, n)
    z = zeros(T, n)
    u = zeros(T, n)

    # Pre-allocate working arrays
    rhs = similar(x)
    z_old = similar(z)

    for _ in 1:maxiter
        copyto!(z_old, z)

        # x-update: x = (A'A + ρI) \ (A'b + ρ(z - u))
        @. rhs = Atb + rho_T * (z - u)
        x .= F \ rhs

        # z-update: soft thresholding of (x + u)
        @inbounds for i in eachindex(z)
            v = x[i] + u[i]
            av = abs(v)
            z[i] = av > threshold ? sign(v) * (av - threshold) : zero(T)
        end

        # u-update: dual variable
        @. u = u + x - z

        # Check convergence via primal and dual residuals
        primal_res = norm(x - z)
        dual_res = rho_T * norm(z - z_old)

        if primal_res < tol && dual_res < tol
            break
        end
    end

    return z
end

# Convenience method for mixed numeric types
function ADMM(A::AbstractMatrix, b::AbstractVector; kwargs...)
    T = promote_type(eltype(A), eltype(b))
    return ADMM(convert(Matrix{T}, A), convert(Vector{T}, b); kwargs...)
end
