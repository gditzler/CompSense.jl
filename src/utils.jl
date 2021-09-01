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

using LinearAlgebra, Random; 


"""
    SL0(n::Int64, p::Int64, k::Int64, type::String="Gaussian")

Generate a dataset for compressed sensing. 

### Input

- `n       -- Int64: number of rows 
- `p`      -- Int64: number of columns
- `k`      -- Int64: number true non-zeros
- `type`   -- String: type to generate ["Gaussian"]

### Output

(A, x, y) for Ax=y
"""
function cs_model(n::Int64, p::Int64, k::Int64, type::String="Gaussian")
    local A, x; 
    if type == "Gaussian"
        A = randn(n, p);   
        while rank(A) != n
            A = randn(n, p);
        end 
        x = zeros(p);
        pp = sign.(randn(k)).*(ones(k)+abs.(randn(k)));
        rp = randperm(n);
        x[rp[1:k]] = pp;
        b = A*x;
    else
        error("Uknown type in cs_model(n,m,type).")
    end

    return A, x, b;

end