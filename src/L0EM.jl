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

using LinearAlgebra;

"""
    L0EM(A::Matrix{Float64}, b::Vector{Float64}; maxiter=100, epsilon=0.01)

Find the solution to Ax=b using an efficient EM algoritm that directly solves
the L0 optimization problem.


### Input

- `A`       -- Matrix: Ax=b
- `b`       -- Vector: Ax=b
- `maxiter` -- (optional) number of optmization iterations
- `epsilon` -- (optional) threshold to stop optimizing
- `lambda`  -- (optional) regularization

### Output

Solution to Ax=b (Vector{Float64})

### Example

julia> A = randn(10, 100);
julia> b = randn(10);
julia> x = L0EM(A, b, maxiter=5, epsilon=.01);
julia> println(x)

### Algorithm

This function implements Liu and Li's L0EM algorithms in https://arxiv.org/pdf/1407.7508v1.pdf.

"""
function L0EM(A::Matrix{Float64},
              b::Vector{Float64};
              lambda=.001,
              epsilon=.001,
              maxiter=50)
    local n, p, theta, A_eta, eta
    eps_stop = .01
    eps_zero = .01
    n, p = size(A)
    eye = Matrix{Float64}(I, p, p)

    # get the initial solution
    theta = inv(A'*A + lambda*eye)*A'*b

    # continue to optimize theta
    for i = 1:maxiter
        eta = theta

        # copy the squared eta terms into a matrix so we can use the hammard product.
        A_eta = repeat(eta.^2, 1, n)'
        A_eta = A_eta.*A

        # update theta
        theta = inv(A_eta'*A + lambda*eye)*A_eta'*b

        # check to break the loop due to a small difference in norm
        if norm(theta-eta, 2) <= epsilon
            break
        end
    end
    x = theta
    i = abs.(x) .< epsilon
    x[i] = zeros(sum(i))
    return x
end


# A = randn(10, 50); b = randn(10);
# x = L0EM(A, b);
# println(x)
