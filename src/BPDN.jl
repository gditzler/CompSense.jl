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

# Note: LinearAlgebra, Convex, SCS are imported by the parent module

"""
    BPDN(A, b; sigma=0.1, epsilon=1e-4)

Find the solution to Ax≈b using Basis Pursuit Denoising.

Solves: min ‖x‖₁  subject to ‖Ax - b‖₂ ≤ σ

This is the noise-aware variant of Basis Pursuit, essential for real-world
compressed sensing where measurements are corrupted by noise.

# Arguments
- `A::AbstractMatrix`: Sensing matrix in Ax=b (m × n)
- `b::AbstractVector`: Noisy measurement vector (m × 1)
- `sigma::Real`: Noise level bound on ‖Ax - b‖₂ (default: 0.1)
- `epsilon::Real`: Threshold for zeroing small coefficients (default: 1e-4)

# Returns
- `Vector{Float64}`: Sparse solution x

# Example
```julia
A, x_true, b = gaussian_sensing(50, 200, 10; snr=20.0)
x_recovered = BPDN(A, b; sigma=0.5)
```

# Algorithm
Solves the second-order cone program:

    min ‖x‖₁  subject to ‖Ax - b‖₂ ≤ σ

via Convex.jl with the SCS solver.

# References
> Chen, S.S., Donoho, D.L., and Saunders, M.A., "Atomic Decomposition
> by Basis Pursuit," SIAM Review, 2001.
"""
function BPDN(A::AbstractMatrix{T},
              b::AbstractVector{T};
              sigma::Real=0.1,
              epsilon::Real=1e-4) where {T<:Real}
    _, p = size(A)
    eps_T = convert(T, epsilon)
    sigma_T = convert(T, sigma)

    x_var = Variable(p)

    silent_solver = Convex.MOI.OptimizerWithAttributes(
        SCS.Optimizer,
        Convex.MOI.Silent() => true
    )

    prob = minimize(norm(x_var, 1), norm(A * x_var - b, 2) <= sigma_T)
    solve!(prob, silent_solver)

    xhat = evaluate(x_var)

    # Threshold small values for clean sparse output
    x = Vector{T}(xhat)
    @inbounds for i in eachindex(x)
        if abs(x[i]) < eps_T
            x[i] = zero(T)
        end
    end

    return x
end

# Convenience method for mixed numeric types
function BPDN(A::AbstractMatrix, b::AbstractVector; kwargs...)
    T = promote_type(eltype(A), eltype(b))
    return BPDN(convert(Matrix{T}, A), convert(Vector{T}, b); kwargs...)
end
