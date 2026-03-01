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

# Note: LinearAlgebra is imported by the parent module

"""
    dct_matrix(p)

Construct a p x p orthonormal DCT-II matrix.

# Arguments
- `p::Integer`: Matrix dimension

# Returns
- `Matrix{Float64}`: p x p orthonormal DCT-II matrix where D'D = I

# Example
```julia
D = dct_matrix(64)
@assert D' * D ≈ I  # orthonormal
```
"""
function dct_matrix(p::Integer)
    D = zeros(p, p)
    @inbounds for j in 1:p
        for l in 1:p
            if j == 1
                D[j, l] = sqrt(1 / p)
            else
                D[j, l] = sqrt(2 / p) * cos(π * (j - 1) * (2 * l - 1) / (2 * p))
            end
        end
    end
    return D
end

"""
    identity_matrix(p)

Construct a p x p Float64 identity matrix.

# Arguments
- `p::Integer`: Matrix dimension

# Returns
- `Matrix{Float64}`: p x p identity matrix
"""
function identity_matrix(p::Integer)
    return Matrix{Float64}(I, p, p)
end

"""
    recover_in_basis(A, b, Psi, algorithm; kwargs...)

Recover a signal that is sparse in the basis (dictionary) Psi.

Given measurements `b = A * x` where `x = Psi * s` for some sparse `s`,
this function forms `Phi = A * Psi`, solves for the sparse coefficients `s`
using the specified algorithm, and returns `x = Psi * s`.

# Arguments
- `A::AbstractMatrix`: Sensing matrix (n x p)
- `b::AbstractVector`: Measurement vector (n x 1)
- `Psi::AbstractMatrix`: Sparsity basis / dictionary (p x p)
- `algorithm::Function`: Any CompSense algorithm (e.g., OMP, ADMM, IHT)
- `kwargs...`: Keyword arguments passed to the algorithm

# Returns
- `Vector{Float64}`: Recovered signal x = Psi * s

# Example
```julia
A, x_true, b = gaussian_sensing(50, 200, 10)
Psi = dct_matrix(200)
x_recovered = recover_in_basis(A, b, Psi, OMP; sparsity=10)
```
"""
function recover_in_basis(A::AbstractMatrix, b::AbstractVector,
                          Psi::AbstractMatrix, algorithm::Function; kwargs...)
    Phi = A * Psi
    s = algorithm(Phi, b; kwargs...)
    return Psi * s
end
