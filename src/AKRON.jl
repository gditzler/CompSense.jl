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
# TODO: This algorithm is work in progress and not yet exported

"""
    AKRON(A, b; epsilon=1e-3)

Find the solution to Ax=b using the AKRON algorithm.

# Arguments
- `A::AbstractMatrix`: Sensing matrix in Ax=b
- `b::AbstractVector`: Measurement vector in Ax=b
- `epsilon::Real`: Tolerance parameter (default: 1e-3)

# Returns
- `Vector`: Sparse solution x to Ax=b

# Status
⚠️ This function is currently under development and not yet exported.
"""
function AKRON(A::AbstractMatrix{T},
               b::AbstractVector{T};
               epsilon::Real=1e-3) where {T<:Real}

    # Create silent solver using Convex's re-exported MOI (Convex.jl 0.16+ API)
    silent_solver = Convex.MOI.OptimizerWithAttributes(
        SCS.Optimizer,
        Convex.MOI.Silent() => true
    )

    # Find the null space and dimensions
    X = nullspace(A)
    s = size(X, 2)
    n = size(A, 2)

    # Adjust epsilon
    epsilon2 = epsilon / 2

    # Get initial L1 minimization solution
    xhat_var = Variable(n)
    prob = minimize(norm(xhat_var, 1), norm(A * xhat_var - b, 2) <= epsilon2)
    solve!(prob, silent_solver)
    xhat = evaluate(xhat_var)

    # Compute signs and sort by magnitude
    x_tmp = copy(xhat)
    sgn = sign.(xhat)

    sorted_idx = sortperm(abs.(xhat))
    smallest = sorted_idx[1:s]
    largest = setdiff(1:n, smallest)

    x_tmp = x_tmp[largest]
    lgst_smallest = maximum(abs.(xhat[smallest]))
    sgn_largest = sign.(x_tmp)

    # TODO: Complete the AKRON algorithm implementation

    return xhat
end
