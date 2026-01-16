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

# Note: LinearAlgebra is imported by the parent module
# TODO: This algorithm is work in progress and not yet exported

"""
    KRON(A, b)

Find the solution to Ax=b using the KRON algorithm.

# Arguments
- `A::AbstractMatrix`: Sensing matrix in Ax=b
- `b::AbstractVector`: Measurement vector in Ax=b

# Returns
- `Vector`: Sparse solution x to Ax=b

# Status
⚠️ This function is currently under development and not yet exported.
"""
function KRON(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T<:Real}
    # TODO: Implement the KRON algorithm
    error("KRON is not yet implemented")
end
