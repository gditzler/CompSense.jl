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

function AKRON(A::Matrix{Float64}, b::Vector{Float64}, epsilon::Float64=1e-3)

    # set the optimizer 
    solver = () -> SCS.Optimizer(verbose=0)

    # find the null space and the size 
    X = nullspace(A);
    s = size(X, 2);
    n = size(A, 2);

    # adjust the value of epsilon 
    epsilon2 = epsilon/2;

    # get the intiial solution
    xhat = Variable(n);
    prob = minimize(norm(xhat, 1), norm(A*xhat-b, 2) <= epsilon2)
    solve!(prob, solver)
    xhat = evaluate(xhat);

    x_tmp = xhat;
    x_l1 = xhat;
    sgn = sign(xhat);

    i = permsort(abs.(xhat));
    smallest = i[1:s];
    largest = setdiff(1:n, smallest);
    x_tmp = x_tmp[largest];
    lgst_smallest = max(abs.(x[smallest]));
    sgn_largest = sign.(x_tmp);



end
