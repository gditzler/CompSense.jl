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

using LinearAlgebra, Convex, SCS;

"""
    IRLS(A::Matrix{Float64}, b::Vector{Float64}; maxiter=100, epsilon=0.01)

Find the solution to Ax=b using Iterative Recursive Least Squares.

### Input

- `A`       -- Matrix: Ax=b
- `b`       -- Vector: Ax=b
- `maxiter` -- (optional) number of optmization iterations
- `epsilon` -- (optional) threshold to stop optimizing

### Output

Solution to Ax=b (Vector{Float64})

### Example

julia> A = randn(10, 100);
julia> b = randn(10);
julia> x = IRLS(A, b, maxiter=5, epsilon=.01);
julia> println(x)

### Algorithm

This function implements Andrew's monotone chain convex hull algorithm to
Enhancing Sparsity by Reweighted L1 Minimization

    Emmanuel J. Candes, Michael B. Wakin, and Stephen P. Boyd, "Enhancing
        Sparsity by Reweighted L1 Minimization," J Fourier Anal Appl (2008)
        14: 877–905.

"""
function IRWLS(A::Matrix{Float64},
               b::Vector{Float64};
               maxiter=100,
               epsilon=0.01)
    # get the second column dim of A
    local p, x, xhat, w;
    _, p = size(A);
    w = ones(p);
    w_old = w;
    o = ones(p);
    x = zeros(p);

    solver = () -> SCS.Optimizer(verbose=0)

    for i = 1:maxiter
        # create a diagonal matrix from the vector w
        W = diagm(w);

        # solve the convex optimization task
        xhat = Variable(p);
        prob = minimize(norm(W*xhat, 1), A*xhat==b)
        solve!(prob, solver)
        xhat = evaluate(xhat);

        # check the stop condition then move on
        w = 1.0./(epsilon*o + abs.(xhat))
        if norm(w - w_old, 2) <= 1e-6
            break;
        end
        w = w_old;
    end
    x = xhat;
    i = abs.(x) .< epsilon;
    x[i] = zeros(sum(i));
    return x;
end
