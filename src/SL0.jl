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

using LinearAlgebra

"""
    SL0(A::Matrix{Float64}, b::Vector{Float64}; maxiter=100, epsilon=0.01, sigma_decrease_factor=0.85)

Find the solution to Ax=b using Iterative Recursive Least Squares.

### Input

- `A`       -- Matrix: Ax=b
- `b`       -- Vector: Ax=b
- `sigma_decrease_factor` -- (optional) number of optmization iterations
- `epsilon` -- (optional) threshold to stop optimizing
- `maxiter` -- (optional) max number of iterations

### Output

Solution to Ax=b (Vector{Float64})

### Algorithm

Smoothed L0 (http://ee.sharif.edu/~SLzero/)

"""
function SL0(A::Matrix{Float64},
             b::Vector{Float64};
             sigma_decrease_factor=.85,
             maxiter=150,
             epsilon=.001)

    # set up the locals
    local mu_0, L, A_pinv, s, x;

    # assign constants
    mu_0 = 2          # The  value  of  mu_0  scales  the sequence of mu
    L = 3             # number  of  iterations of the internal (steepest ascent) loop
    A_pinv = pinv(A)  # pseudo-inverse of matrix A defined by A_pinv=A'*inv(A*A')

    # initialize the solution
    s = A_pinv*b
    sigma = 2*maximum(abs.(s))

    for j = 1:maxiter

        for i = 1:L
            delta = s.*exp.(-abs.(s).^2/sigma^2)
            s -= mu_0*delta
            s -= A_pinv*(A*s - b)
        end
        sigma *= sigma_decrease_factor
    end
    x = s
    i = abs.(x) .< epsilon
    x[i] = zeros(sum(i))
    return x
end
