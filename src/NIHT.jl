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
# Note: hard_threshold! is defined in IHT.jl

"""
    NIHT(A, b; sparsity, maxiter=500, tol=1e-6)

Find the solution to Ax=b using Normalized Iterative Hard Thresholding (NIHT).

Solves: min_x ‖Ax - b‖₂²  subject to ‖x‖₀ ≤ k

# Arguments
- `A::AbstractMatrix`: Sensing matrix in Ax=b (m × n)
- `b::AbstractVector`: Measurement vector in Ax=b (m × 1)
- `sparsity::Int`: Target sparsity level k (required)
- `maxiter::Int`: Maximum number of iterations (default: 500)
- `tol::Real`: Convergence tolerance on relative change (default: 1e-6)

# Returns
- `Vector`: k-sparse solution x to Ax=b

# Example
```julia
A = randn(50, 200)
x_true = zeros(200)
x_true[rand(1:200, 10)] = randn(10)
b = A * x_true
x = NIHT(A, b; sparsity=10)
```

# Algorithm
Implements Normalized IHT with adaptive step size selection:

1. Gradient step: g = Aᵀ(b - Ax)
2. Compute support of top-k entries of (x + g): Ω = supp(H_k(x + g))
3. Adaptive step size: μ = ‖g_Ω‖₂² / ‖A g_Ω‖₂² (restricted to support)
4. Update: x = H_k(x + μ * g)
5. Repeat until convergence

The normalized step size adapts to the local geometry, providing better
convergence than fixed-step IHT, especially for ill-conditioned problems.

Reference:
> Blumensath, T. and Davies, M.E., "Normalized Iterative Hard Thresholding:
> Guaranteed Stability and Performance," IEEE J. Selected Topics in Signal
> Processing, 2010.
"""
function NIHT(A::AbstractMatrix{T},
              b::AbstractVector{T};
              sparsity::Int,
              maxiter::Int=500,
              tol::Real=1e-6) where {T<:Real}
    _, n = size(A)

    # Initialize
    x = zeros(T, n)
    x_old = similar(x)

    # Pre-allocate working arrays
    residual = similar(b)
    gradient = similar(x)
    g_omega = similar(x)
    Ag_omega = similar(b)

    for _ in 1:maxiter
        copyto!(x_old, x)

        # Compute residual: r = b - Ax
        mul!(residual, A, x)
        @. residual = b - residual

        # Gradient: g = A'r
        mul!(gradient, A', residual)

        # Identify support: top-k indices of |x + g|
        temp = x + gradient
        abs_temp = abs.(temp)
        perm = sortperm(abs_temp, rev=true)
        omega = perm[1:min(sparsity, n)]

        # Compute restricted gradient g_Ω
        fill!(g_omega, zero(T))
        @inbounds for j in omega
            g_omega[j] = gradient[j]
        end

        # Adaptive step size: μ = ‖g_Ω‖² / ‖A g_Ω‖²
        mul!(Ag_omega, A, g_omega)
        g_norm_sq = dot(g_omega, g_omega)
        Ag_norm_sq = dot(Ag_omega, Ag_omega)

        if Ag_norm_sq > eps(T)
            mu = g_norm_sq / Ag_norm_sq
        else
            mu = one(T)
        end

        # Update: x = H_k(x + μ * g)
        @. x = x + mu * gradient
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
function NIHT(A::AbstractMatrix, b::AbstractVector; kwargs...)
    T = promote_type(eltype(A), eltype(b))
    return NIHT(convert(Matrix{T}, A), convert(Vector{T}, b); kwargs...)
end
