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
    block_soft_threshold!(z, v, groups, threshold)

In-place block (group) soft thresholding.

For each group g in groups, applies:
    z_g = max(1 - threshold/‖v_g‖₂, 0) * v_g

# Arguments
- `z::AbstractVector`: Output vector (modified in-place)
- `v::AbstractVector`: Input vector
- `groups::Vector{Vector{Int}}`: List of index groups
- `threshold::Real`: Thresholding parameter
"""
function block_soft_threshold!(z::AbstractVector{T}, v::AbstractVector{T},
                               groups::Vector{Vector{Int}},
                               threshold::Real) where {T<:Real}
    fill!(z, zero(T))
    @inbounds for g in groups
        # Compute group norm
        group_norm = zero(T)
        for i in g
            group_norm += v[i]^2
        end
        group_norm = sqrt(group_norm)

        # Block soft threshold
        if group_norm > threshold
            scale = one(T) - T(threshold) / group_norm
            for i in g
                z[i] = scale * v[i]
            end
        end
    end
end

"""
    GroupLASSO(A, b, groups; lambda=0.1, rho=1.0, maxiter=500, tol=1e-6)

Solve the Group LASSO problem using ADMM.

Solves: min  1/2 ‖Ax - b‖₂² + lambda * sum_g ‖x_g‖₂

where x_g denotes the subvector of x corresponding to group g.

# Arguments
- `A::AbstractMatrix`: Sensing matrix (m x n)
- `b::AbstractVector`: Measurement vector (m x 1)
- `groups::Vector{Vector{Int}}`: List of index groups, each a vector of column indices
- `lambda::Real`: Group regularization parameter (default: 0.1)
- `rho::Real`: ADMM penalty parameter (default: 1.0)
- `maxiter::Int`: Maximum number of iterations (default: 500)
- `tol::Real`: Convergence tolerance on primal/dual residuals (default: 1e-6)

# Returns
- `Vector`: Group-sparse solution x

# Example
```julia
A = randn(50, 200)
b = randn(50)
groups = [collect(i:i+3) for i in 1:4:200]  # groups of 4
x = GroupLASSO(A, b, groups; lambda=0.1)
```

# Algorithm
Reformulates the Group LASSO as:

    min  1/2 ‖Ax - b‖₂² + lambda * sum_g ‖z_g‖₂  subject to x = z

and applies ADMM splitting:

1. x-update: x = (A'A + rho*I)^{-1} (A'b + rho*(z - u))
2. z-update: z_g = block_soft_threshold(x_g + u_g, lambda/rho)
3. u-update: u = u + x - z

Reference:
> Yuan, M. and Lin, Y., "Model selection and estimation in regression with
> grouped variables," Journal of the Royal Statistical Society, 2006.
"""
function GroupLASSO(A::AbstractMatrix{T},
                    b::AbstractVector{T},
                    groups::Vector{Vector{Int}};
                    lambda::Real=0.1,
                    rho::Real=1.0,
                    maxiter::Int=500,
                    tol::Real=1e-6) where {T<:Real}
    _, n = size(A)
    rho_T = convert(T, rho)
    lambda_T = convert(T, lambda)
    threshold = lambda_T / rho_T

    # Precompute factorization: (A'A + rho*I)
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
    v = similar(x)

    for _ in 1:maxiter
        copyto!(z_old, z)

        # x-update: x = (A'A + rho*I) \ (A'b + rho*(z - u))
        @. rhs = Atb + rho_T * (z - u)
        x .= F \ rhs

        # z-update: block soft thresholding of (x + u)
        @. v = x + u
        block_soft_threshold!(z, v, groups, threshold)

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
function GroupLASSO(A::AbstractMatrix, b::AbstractVector,
                    groups::Vector{Vector{Int}}; kwargs...)
    T = promote_type(eltype(A), eltype(b))
    return GroupLASSO(convert(Matrix{T}, A), convert(Vector{T}, b), groups; kwargs...)
end
