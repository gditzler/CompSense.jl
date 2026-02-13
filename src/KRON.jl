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

# Note: LinearAlgebra and Combinatorics are imported by the parent module

"""
    KRON(A, b; epsilon=1e-6)

Find the sparsest exact solution to Ax=b using the KRON combinatorial algorithm.

KRON enumerates all possible support patterns by exploiting the null space of A.
For each `(n choose s)` combination (where `s = dim(null(A))`), it constructs a
candidate solution and selects the one with the fewest nonzero entries.

# Arguments
- `A::AbstractMatrix`: Sensing matrix in Ax=b (m × n, m < n)
- `b::AbstractVector`: Measurement vector in Ax=b (m × 1)
- `epsilon::Real`: Threshold below which values are set to zero (default: 1e-6)

# Returns
- `Vector`: Sparsest exact solution x to Ax=b

# Example
```julia
A = randn(8, 20)
x_true = zeros(20)
x_true[rand(1:20, 3)] = randn(3)
b = A * x_true
x = KRON(A, b)
```

# Algorithm
1. Compute null space X = nullspace(A) with dimension s
2. Compute particular solution x₀ = A' * ((A A')⁻¹ b)
3. For each combination of s indices from {1, …, n}:
   - Extract submatrix BX = X[combo, :] and subvector Bx₀ = x₀[combo]
   - Compute candidate: x = x₀ - X * (BX \\ Bx₀)
   - Threshold small values to zero
4. Return the candidate with the fewest nonzero entries

Note: The combinatorial enumeration has complexity O(n choose s), which grows
rapidly. This algorithm is best suited for small problem sizes.

Reference:
> Ditzler, G. and Assaleh, K., "KRON: An Approach for the Integration of
> the Kernel Trick within the KRON Compressed Sensing Framework."
"""
function KRON(A::AbstractMatrix{T},
              b::AbstractVector{T};
              epsilon::Real=1e-6) where {T<:Real}
    m, n = size(A)

    # Compute null space basis: X is n × s where s = n - rank(A)
    X = nullspace(A)
    s = size(X, 2)

    if s == 0
        # Fully determined or overdetermined — unique solution
        return A \ b
    end

    # Particular solution in the row space of A
    x₀ = A' * ((A * A') \ b)

    # Pre-allocate candidate buffer
    x_candidate = similar(x₀)

    # Track the sparsest solution found
    best_x = copy(x₀)
    # Threshold x₀ for initial sparsity count
    @inbounds for i in eachindex(best_x)
        if abs(best_x[i]) < epsilon
            best_x[i] = zero(T)
        end
    end
    best_sparsity = count(!iszero, best_x)

    for combo in combinations(1:n, s)
        # Extract rows selected by this combination — avoids building full n×n B matrix
        BX = X[combo, :]         # s × s
        Bx₀ = x₀[combo]         # s-vector

        # Skip singular submatrices
        F = lu(BX; check=false)
        issuccess(F) || continue

        # Solve the s × s system and compute candidate
        # x = x₀ - X * (BX \ Bx₀)
        alpha = F \ Bx₀
        mul!(x_candidate, X, alpha)
        @. x_candidate = x₀ - x_candidate

        # Threshold small values
        @inbounds for i in eachindex(x_candidate)
            if abs(x_candidate[i]) < epsilon
                x_candidate[i] = zero(T)
            end
        end

        # Check if this is the sparsest solution so far
        sp = count(!iszero, x_candidate)
        if sp < best_sparsity
            best_sparsity = sp
            copyto!(best_x, x_candidate)
        end
    end

    return best_x
end

# Convenience method for mixed numeric types
function KRON(A::AbstractMatrix, b::AbstractVector; kwargs...)
    T = promote_type(eltype(A), eltype(b))
    return KRON(convert(Matrix{T}, A), convert(Vector{T}, b); kwargs...)
end
