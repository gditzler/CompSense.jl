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
    mutual_coherence(A)

Compute the mutual coherence of matrix A:

    μ(A) = max_{i≠j} |⟨aᵢ, aⱼ⟩| / (‖aᵢ‖₂ ‖aⱼ‖₂)

The mutual coherence is the maximum absolute normalized inner product between
distinct columns. It bounds the spark and is key for recovery guarantees:
if ‖x‖₀ < (1 + 1/μ(A)) / 2, then x is the unique sparsest solution.

# Arguments
- `A::AbstractMatrix`: Sensing matrix (m × n)

# Returns
- `Float64`: Mutual coherence in [0, 1]

# Example
```julia
A = randn(50, 200)
mu = mutual_coherence(A)
```

# References
- Donoho & Elad (2003), "Optimally sparse representation"
"""
function mutual_coherence(A::AbstractMatrix)
    G = column_coherence_matrix(A)
    n = size(G, 1)

    mu = 0.0
    @inbounds for j in 1:n
        for i in 1:(j-1)
            val = abs(G[i, j])
            if val > mu
                mu = val
            end
        end
    end

    return mu
end

"""
    babel_function(A, k)

Compute the cumulative coherence (Babel function) μ₁(k) for matrix A.

    μ₁(k) = max_i  max_{|Λ|=k, i∉Λ}  Σ_{j∈Λ} |⟨aᵢ, aⱼ⟩| / (‖aᵢ‖₂ ‖aⱼ‖₂)

The Babel function provides tighter recovery guarantees than mutual coherence.
If μ₁(k-1) + μ₁(k) < 1, greedy algorithms like OMP exactly recover any
k-sparse signal.

# Arguments
- `A::AbstractMatrix`: Sensing matrix (m × n)
- `k::Int`: Sparsity level

# Returns
- `Float64`: Babel function value μ₁(k)

# Example
```julia
A = randn(50, 200)
mu1 = babel_function(A, 10)
```

# References
- Tropp (2004), "Greed is Good: Algorithmic Results for Sparse Approximation"
"""
function babel_function(A::AbstractMatrix, k::Int)
    G = column_coherence_matrix(A)
    n = size(G, 1)

    if k >= n
        throw(ArgumentError("k must be less than number of columns, got k=$k, n=$n"))
    end

    mu1 = 0.0
    row_buf = zeros(n - 1)

    @inbounds for i in 1:n
        # Collect |G[i,j]| for j != i
        idx = 0
        for j in 1:n
            if j != i
                idx += 1
                row_buf[idx] = abs(G[i, j])
            end
        end

        # Sort descending and sum the k largest
        sort!(view(row_buf, 1:idx), rev=true)
        s = sum(view(row_buf, 1:min(k, idx)))

        if s > mu1
            mu1 = s
        end
    end

    return mu1
end

"""
    spark(A)

Compute the spark of matrix A: the smallest number of linearly dependent columns.

    spark(A) = min { ‖x‖₀ : Ax = 0, x ≠ 0 }

The spark provides an exact (but NP-hard in general) sparsity threshold:
if ‖x‖₀ < spark(A)/2, then x is the unique sparsest solution to Ax = b.

# Arguments
- `A::AbstractMatrix`: Sensing matrix (m × n), must be small (n ≤ 20 recommended)

# Returns
- `Int`: Spark of A

# Example
```julia
A = randn(5, 8)
s = spark(A)
```

# Notes
- Computing the spark is NP-hard in general. This function enumerates all
  subsets and is only practical for small matrices (n ≤ ~20).
- Throws `ArgumentError` if n > 30 to prevent accidental combinatorial explosion.
"""
function spark(A::AbstractMatrix)
    _, n = size(A)

    if n > 30
        throw(ArgumentError("spark computation is combinatorial (NP-hard); n=$n is too large. Use n ≤ 30."))
    end

    for s in 2:n
        for cols in combinations(1:n, s)
            A_sub = A[:, cols]
            if rank(A_sub) < s
                return s
            end
        end
    end

    return n + 1
end

"""
    column_coherence_matrix(A)

Compute the full Gram matrix of normalized columns of A.

    G[i,j] = ⟨aᵢ, aⱼ⟩ / (‖aᵢ‖₂ ‖aⱼ‖₂)

Diagonal entries are 1. Off-diagonal entries measure pairwise column correlation.
Useful for visualizing coherence structure (e.g., as a heatmap).

# Arguments
- `A::AbstractMatrix`: Sensing matrix (m × n)

# Returns
- `Matrix{Float64}`: n × n coherence Gram matrix

# Example
```julia
A = randn(50, 200)
G = column_coherence_matrix(A)
# heatmap(abs.(G)) for visualization
```
"""
function column_coherence_matrix(A::AbstractMatrix)
    n = size(A, 2)

    # Normalize columns
    A_norm = similar(A, Float64)
    @inbounds for j in 1:n
        col_norm = norm(view(A, :, j))
        if col_norm > 0
            A_norm[:, j] = view(A, :, j) / col_norm
        else
            A_norm[:, j] .= 0.0
        end
    end

    return A_norm' * A_norm
end
