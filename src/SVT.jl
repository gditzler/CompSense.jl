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
    SVT(Omega, values, m, n; tau=nothing, delta=nothing, maxiter=500, tol=1e-4)

Recover a low-rank matrix from partially observed entries using Singular Value
Thresholding (SVT).

# Arguments
- `Omega::Vector{Tuple{Int,Int}}`: Observed entry indices (i, j)
- `values::Vector`: Observed entry values
- `m::Int`: Number of rows of the target matrix
- `n::Int`: Number of columns of the target matrix
- `tau::Union{Real,Nothing}`: Singular value threshold; default 5*sqrt(m*n)
- `delta::Union{Real,Nothing}`: Step size; default 1.2*m*n/length(Omega)
- `maxiter::Int`: Maximum number of iterations (default: 500)
- `tol::Real`: Convergence tolerance on relative change (default: 1e-4)

# Returns
- `Matrix{Float64}`: Recovered m x n low-rank matrix

# Example
```julia
# Generate a rank-3 matrix and observe 50% of entries
M_true = randn(20, 20) * randn(20, 20)'  # not exactly low-rank
Omega = [(i, j) for i in 1:20 for j in 1:20 if rand() < 0.5]
values = [M_true[i, j] for (i, j) in Omega]
M_recovered = SVT(Omega, values, 20, 20)
```

# Algorithm
Implements the Singular Value Thresholding algorithm:

1. Initialize dual variable Y = 0
2. Compute SVD of Y: Y = U * diag(sigma) * V'
3. Soft-threshold singular values: X = U * diag(max(sigma - tau, 0)) * V'
4. Update Y on observed entries: Y = Y + delta * P_Omega(M - X)
5. Repeat until convergence

Reference:
> Cai, J.F., Candes, E.J., and Shen, Z., "A Singular Value Thresholding
> Algorithm for Matrix Completion," SIAM J. Optimization, 2010.
"""
function SVT(Omega::Vector{Tuple{Int,Int}},
             values::Vector{T},
             m::Int, n::Int;
             tau::Union{Real,Nothing}=nothing,
             delta::Union{Real,Nothing}=nothing,
             maxiter::Int=500,
             tol::Real=1e-4) where {T<:Real}
    num_observed = length(Omega)

    # Default parameters
    tau_val = isnothing(tau) ? 5.0 * sqrt(m * n) : Float64(tau)
    delta_val = isnothing(delta) ? 1.2 * m * n / num_observed : Float64(delta)

    # Initialize dual variable Y from observed entries
    Y = zeros(m, n)
    @inbounds for (idx, (i, j)) in enumerate(Omega)
        Y[i, j] = delta_val * values[idx]
    end

    X = zeros(m, n)

    for _ in 1:maxiter
        X_old = copy(X)

        # SVD of Y
        F = svd(Y)

        # Soft-threshold singular values
        sigma_thresh = max.(F.S .- tau_val, 0.0)

        # Reconstruct X = U * diag(sigma_thresh) * V'
        # Only use non-zero singular values for efficiency
        r = count(s -> s > 0, sigma_thresh)
        if r == 0
            fill!(X, 0.0)
        else
            X .= F.U[:, 1:r] * Diagonal(sigma_thresh[1:r]) * F.Vt[1:r, :]
        end

        # Update Y on observed entries: Y = Y + delta * (observed - X_observed)
        @inbounds for (idx, (i, j)) in enumerate(Omega)
            Y[i, j] += delta_val * (values[idx] - X[i, j])
        end

        # Check convergence
        x_norm = norm(X)
        if x_norm > eps()
            rel_change = norm(X - X_old) / x_norm
            if rel_change < tol
                break
            end
        elseif norm(X - X_old) < tol
            break
        end
    end

    return X
end

# Convenience method for non-Float64 values
function SVT(Omega::Vector{Tuple{Int,Int}}, values::Vector, m::Int, n::Int; kwargs...)
    return SVT(Omega, convert(Vector{Float64}, values), m, n; kwargs...)
end
