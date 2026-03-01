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
    AMP(A, b; maxiter=500, tol=1e-6)

Find the solution to Ax=b using Approximate Message Passing (AMP).

Solves sparse recovery with i.i.d. Gaussian sensing matrices using the
AMP algorithm with soft-thresholding denoiser.

# Arguments
- `A::AbstractMatrix`: Sensing matrix in Ax=b (m × n), assumed i.i.d. Gaussian
- `b::AbstractVector`: Measurement vector in Ax=b (m × 1)
- `maxiter::Int`: Maximum number of iterations (default: 500)
- `tol::Real`: Convergence tolerance on relative change (default: 1e-6)

# Returns
- `Vector`: Sparse solution x

# Example
```julia
A, x_true, b = gaussian_sensing(50, 200, 10)
x = AMP(A, b)
```

# Algorithm
Implements AMP with soft-thresholding denoiser and Onsager correction:

1. Residual: z = b - Ax + (z_prev/m) * ⟨η'(Aᵀz_prev + x; τ)⟩
2. Effective noise: τ² = ‖z‖₂² / m
3. Threshold: λ = τ (universal threshold)
4. Update: x = η_λ(Aᵀz + x)  (soft thresholding)

The Onsager correction term `(z/m) * ⟨η'⟩` is what distinguishes AMP from
standard iterative thresholding and enables sharp phase transitions matching
the Donoho-Tanner bound for Gaussian matrices.

Reference:
> Donoho, D.L., Maleki, A., and Montanari, A., "Message Passing Algorithms
> for Compressed Sensing," PNAS, 2009.

# Notes
- AMP is specifically designed for i.i.d. Gaussian matrices. Performance may
  degrade significantly for structured or non-Gaussian sensing matrices.
"""
function AMP(A::AbstractMatrix{T},
             b::AbstractVector{T};
             maxiter::Int=500,
             tol::Real=1e-6) where {T<:Real}
    m, n = size(A)
    delta = m / n  # measurement ratio

    # Initialize
    x = zeros(T, n)
    x_old = similar(x)
    z = copy(b)  # residual

    # Pre-allocate working arrays
    pseudo = similar(x)

    for _ in 1:maxiter
        copyto!(x_old, x)

        # Effective noise level: τ = ‖z‖₂ / √m
        tau = norm(z) / sqrt(m)

        # Threshold level
        threshold = tau

        # Pseudo-data: w = A'z + x
        mul!(pseudo, A', z)
        @. pseudo = pseudo + x

        # Soft thresholding denoiser
        nnz_count = 0
        @inbounds for i in eachindex(x)
            wi = pseudo[i]
            awi = abs(wi)
            if awi > threshold
                x[i] = sign(wi) * (awi - threshold)
                nnz_count += 1
            else
                x[i] = zero(T)
            end
        end

        # Onsager correction term: (z/m) * number_of_nonzeros_in_denoiser_output
        # For soft thresholding, ⟨η'⟩ = (1/n) * |{i : |w_i| > τ}| = nnz/n
        onsager_coeff = T(nnz_count) / T(m)

        # Residual update with Onsager correction
        # z = b - Ax + onsager * z_prev
        z_prev = copy(z)
        mul!(z, A, x)
        @. z = b - z + onsager_coeff * z_prev

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
function AMP(A::AbstractMatrix, b::AbstractVector; kwargs...)
    T = promote_type(eltype(A), eltype(b))
    return AMP(convert(Matrix{T}, A), convert(Vector{T}, b); kwargs...)
end
