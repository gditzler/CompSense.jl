# Copyright (c) 2021
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
    IRWLS(A, b; maxiter=100, epsilon=0.01)

Find the solution to Ax=b using Iteratively Reweighted Least Squares (IRWLS).

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
x = IRWLS(A, b; maxiter=5, epsilon=0.01)
```

# Algorithm
Implements the reweighted L1 minimization algorithm from:

> Emmanuel J. Candès, Michael B. Wakin, and Stephen P. Boyd, "Enhancing
> Sparsity by Reweighted L1 Minimization," J Fourier Anal Appl (2008)
> 14: 877–905.

The algorithm iteratively solves weighted L1 minimization problems,
updating weights based on the current solution to promote sparsity.
"""
function IRWLS(A::AbstractMatrix{T},
               b::AbstractVector{T};
               maxiter::Int=100,
               epsilon::Real=0.01) where {T<:Real}
    _, p = size(A)

    # Initialize weights uniformly
    w = ones(T, p)
    w_old = copy(w)
    xhat = zeros(T, p)

    # Create solver factory (SCS 2.x API uses silent keyword in solve!)
    solver = SCS.Optimizer

    for _ in 1:maxiter
        # Use Diagonal instead of diagm for efficiency (O(n) vs O(n²) storage)
        W = Diagonal(w)

        # Solve the weighted L1 minimization problem
        x_var = Variable(p)
        prob = minimize(norm(W * x_var, 1), A * x_var == b)
        solve!(prob, solver; silent=true)
        xhat = evaluate(x_var)

        # Update weights: smaller coefficients get larger weights
        w = 1 ./ (epsilon .+ abs.(xhat))

        # Check convergence
        if norm(w - w_old, 2) <= 1e-6
            break
        end
        w_old = copy(w)
    end

    # Threshold small values to zero for clean sparse output
    x = copy(xhat)
    x[abs.(x) .< epsilon] .= zero(T)

    return x
end

# Convenience method for mixed numeric types
function IRWLS(A::AbstractMatrix, b::AbstractVector; kwargs...)
    T = promote_type(eltype(A), eltype(b))
    return IRWLS(convert(Matrix{T}, A), convert(Vector{T}, b); kwargs...)
end
