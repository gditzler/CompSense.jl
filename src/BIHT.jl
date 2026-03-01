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
# Depends on hard_threshold! from IHT.jl

"""
    BIHT(A, y; sparsity, maxiter=1000, tol=1e-6, tau=nothing)

Find the solution to 1-bit compressed sensing using Binary Iterative Hard
Thresholding (BIHT).

Solves: find x such that y = sign(Ax), ‖x‖₀ ≤ k, ‖x‖₂ = 1

# Arguments
- `A::AbstractMatrix`: Sensing matrix (n x p)
- `y::AbstractVector`: Binary measurements in {-1, +1} (n x 1)
- `sparsity::Int`: Target sparsity level k (required)
- `maxiter::Int`: Maximum number of iterations (default: 1000)
- `tol::Real`: Convergence tolerance on relative change (default: 1e-6)
- `tau::Union{Real,Nothing}`: Step size; if nothing, uses 1/n (default: nothing)

# Returns
- `Vector`: k-sparse unit-norm solution x

# Example
```julia
A = randn(200, 100)
x_true = zeros(100); x_true[1:5] = randn(5); x_true ./= norm(x_true)
y = sign.(A * x_true)
x = BIHT(A, y; sparsity=5)
```

# Algorithm
Implements Binary Iterative Hard Thresholding:

1. Gradient step on sign-consistency loss: z = x + tau * A'(y - sign(Ax))
2. Hard thresholding: x = H_k(z) (keep k largest entries)
3. Normalize to unit sphere: x = x / ‖x‖₂

Reference:
> Jacques, L. et al., "Robust 1-Bit Compressive Sensing via Binary Stable
> Embeddings of Sparse Vectors," IEEE Trans. Info. Theory, 2013.
"""
function BIHT(A::AbstractMatrix{T},
              y::AbstractVector{T};
              sparsity::Int,
              maxiter::Int=1000,
              tol::Real=1e-6,
              tau::Union{Real,Nothing}=nothing) where {T<:Real}
    n, p = size(A)

    # Default step size
    tau_val = isnothing(tau) ? one(T) / n : T(tau)

    # Initialize
    x = zeros(T, p)
    x_old = similar(x)

    # Pre-allocate working arrays
    Ax = similar(y)
    gradient = similar(x)

    for _ in 1:maxiter
        copyto!(x_old, x)

        # Compute Ax and sign consistency gradient
        mul!(Ax, A, x)
        @. Ax = y - sign(Ax)

        # Gradient step: z = x + tau * A'(y - sign(Ax))
        mul!(gradient, A', Ax)
        @. x = x + tau_val * gradient

        # Hard thresholding: keep k largest entries
        hard_threshold!(x, sparsity)

        # Normalize to unit sphere (sign is scale-invariant)
        x_norm = norm(x)
        if x_norm > eps(T)
            x ./= x_norm
        end

        # Check convergence
        if norm(x - x_old) < tol
            break
        end
    end

    return x
end

# Convenience method for mixed numeric types
function BIHT(A::AbstractMatrix, y::AbstractVector; kwargs...)
    T = promote_type(eltype(A), eltype(y))
    return BIHT(convert(Matrix{T}, A), convert(Vector{T}, y); kwargs...)
end
