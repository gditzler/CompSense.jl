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
    BasisPursuit(A, b; epsilon=1e-4)

Find the solution to Ax=b using Basis Pursuit (L1 minimization).

Solves: min ‖x‖₁  subject to Ax = b

This is the foundational algorithm of compressed sensing. Under RIP or
incoherence conditions, the L1 minimization recovers the sparsest solution.

# Arguments
- `A::AbstractMatrix`: Sensing matrix in Ax=b (m × n)
- `b::AbstractVector`: Measurement vector in Ax=b (m × 1)
- `epsilon::Real`: Threshold for zeroing small coefficients (default: 1e-4)

# Returns
- `Vector{Float64}`: Sparse solution x

# Example
```julia
A, x_true, b = gaussian_sensing(50, 200, 10)
x_recovered = BasisPursuit(A, b)
```

# Algorithm
Solves the linear program:

    min ‖x‖₁  subject to Ax = b

via Convex.jl with the SCS solver.

# References
> Chen, S.S., Donoho, D.L., and Saunders, M.A., "Atomic Decomposition
> by Basis Pursuit," SIAM Review, 2001.
"""
function BasisPursuit(A::AbstractMatrix{T},
                      b::AbstractVector{T};
                      epsilon::Real=1e-4) where {T<:Real}
    _, p = size(A)
    eps_T = convert(T, epsilon)

    x_var = Variable(p)

    silent_solver = Convex.MOI.OptimizerWithAttributes(
        SCS.Optimizer,
        Convex.MOI.Silent() => true
    )

    prob = minimize(norm(x_var, 1), A * x_var == b)
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
function BasisPursuit(A::AbstractMatrix, b::AbstractVector; kwargs...)
    T = promote_type(eltype(A), eltype(b))
    return BasisPursuit(convert(Matrix{T}, A), convert(Vector{T}, b); kwargs...)
end
