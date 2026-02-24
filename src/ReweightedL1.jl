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
    ReweightedL1(A, b; maxiter=100, epsilon=0.01)

Find the solution to Ax=b using Reweighted L1 Minimization.

# Arguments
- `A::AbstractMatrix`: Sensing matrix in Ax=b
- `b::AbstractVector`: Measurement vector in Ax=b
- `maxiter::Int`: Maximum number of optimization iterations (default: 100)
- `epsilon::Real`: Convergence threshold and small value regularization (default: 0.01)

# Returns
- `Vector{Float64}`: Sparse solution x to Ax=b

# Example
```julia
A = randn(10, 100)
b = randn(10)
x = ReweightedL1(A, b; maxiter=5, epsilon=0.01)
```

# Algorithm
Implements the reweighted L1 minimization algorithm. At each iteration,
a weighted L1 minimization problem is solved, and the weights are updated
as `w = 1 / (ε + |x|)` to promote sparsity. The algorithm converges when
the change in weights falls below a tolerance.

# Reference
> Emmanuel J. Candès, Michael B. Wakin, and Stephen P. Boyd, "Enhancing
> Sparsity by Reweighted L1 Minimization," J Fourier Anal Appl (2008)
> 14: 877–905.
"""
function ReweightedL1(A::AbstractMatrix{T},
                      b::AbstractVector{T};
                      maxiter::Int=100,
                      epsilon::Real=0.01) where {T<:Real}
    _, p = size(A)
    eps_T = convert(T, epsilon)

    # Initialize weights uniformly
    w = ones(T, p)
    w_old = ones(T, p)
    xhat = zeros(T, p)

    # Create variable once and reuse across iterations
    x_var = Variable(p)

    # Create silent solver using Convex's re-exported MOI (Convex.jl 0.16+ API)
    silent_solver = Convex.MOI.OptimizerWithAttributes(
        SCS.Optimizer,
        Convex.MOI.Silent() => true
    )

    for _ in 1:maxiter
        # Use Diagonal instead of diagm for efficiency (O(n) vs O(n²) storage)
        W = Diagonal(w)

        # Solve the weighted L1 minimization problem
        prob = minimize(norm(W * x_var, 1), A * x_var == b)
        solve!(prob, silent_solver)
        xhat .= evaluate(x_var)

        # Update weights: fused broadcast avoids intermediate allocations
        @. w = 1 / (eps_T + abs(xhat))

        # Check convergence
        if norm(w - w_old, 2) <= 1e-6
            break
        end
        copyto!(w_old, w)
    end

    # Threshold small values to zero for clean sparse output
    x = copy(xhat)
    @inbounds for i in eachindex(x)
        if abs(x[i]) < eps_T
            x[i] = zero(T)
        end
    end

    return x
end

# Convenience method for mixed numeric types
function ReweightedL1(A::AbstractMatrix, b::AbstractVector; kwargs...)
    T = promote_type(eltype(A), eltype(b))
    return ReweightedL1(convert(Matrix{T}, A), convert(Vector{T}, b); kwargs...)
end
