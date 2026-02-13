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

# Note: LinearAlgebra, Convex, SCS, Combinatorics are imported by the parent module

"""
    AKRON(A, b; shift=3, sparsity_threshold=1e-3)

Find the solution to Ax=b using Approximate Kernel RecOnstructioN (AKRON).

AKRON combines L1 minimization with a kernel-based refinement step that
enumerates candidate zero-entry positions to find a sparser solution.

# Arguments
- `A::AbstractMatrix`: Sensing matrix (m×n) in Ax=b
- `b::AbstractVector`: Measurement vector (m-dimensional) in Ax=b
- `shift::Int`: Number of extra candidate indices beyond kernel dimension (default: 3)
- `sparsity_threshold::Real`: Threshold below which entries are considered zero (default: 1e-3)

# Returns
- `Vector{T}`: Sparse solution x to Ax=b

# Example
```julia
A = randn(10, 30)
x_true = zeros(30); x_true[1] = 1.0; x_true[5] = -2.0;
b = A * x_true
x = AKRON(A, b; shift=3)
```

# Algorithm
1. Solve basis pursuit: min ||x||₁ subject to Ax = b
2. Sort entries by magnitude and identify the s+shift smallest as candidate zeros,
   where s = n - m is the kernel dimension
3. Enumerate all C(s+shift, s) combinations of candidate zero-index sets
4. For each combination, solve least squares on the complementary columns
5. Select the combination yielding the sparsest solution (ties broken by L1 norm)

# References
> Gregory Ditzler and Nidhal Bouaynaya, "Approximate Kernel Reconstruction
> for Data-Driven Subspace Learning," 2019.
"""
function AKRON(A::AbstractMatrix{T}, b::AbstractVector{T};
               shift::Int=3,
               sparsity_threshold::Real=1e-3) where {T<:Real}
    m, n = size(A)
    s = n - m  # kernel dimension

    # Phase 1: L1 minimization (basis pursuit)
    silent_solver = Convex.MOI.OptimizerWithAttributes(
        SCS.Optimizer,
        Convex.MOI.Silent() => true
    )
    x_var = Variable(n)
    prob = minimize(norm(x_var, 1), A * x_var == b)
    solve!(prob, silent_solver)
    x_l1 = vec(evaluate(x_var))

    # Sort entries by absolute value (ascending) and take s+shift smallest
    sorted_indices = sortperm(abs.(x_l1))
    num_candidates = min(s + shift, n)
    smallest = sorted_indices[1:num_candidates]

    # Collect combinations for indexed parallel access
    ncols = n - s  # number of complement columns (= m)
    combrows = collect(combinations(smallest, s))
    num_combos = length(combrows)

    # Per-combination result arrays (written independently, no races)
    sp_arr  = Vector{Int}(undef, num_combos)
    err_arr = Vector{T}(undef, num_combos)
    xhat_arr = [Vector{T}(undef, ncols) for _ in 1:num_combos]
    j_arr    = [Vector{Int}(undef, ncols) for _ in 1:num_combos]

    # Phase 2: Kernel refinement — parallel over combinations (like MATLAB parfor)
    Threads.@threads for r in 1:num_combos
        # Task-local temporaries (allocated per task, no sharing)
        Aj_local = Matrix{T}(undef, m, ncols)
        candidate_mask = falses(n)

        # Build complement index set using mask
        combo = combrows[r]
        @inbounds for idx in combo
            candidate_mask[idx] = true
        end
        ci = 0
        @inbounds for i in 1:n
            if !candidate_mask[i]
                ci += 1
                j_arr[r][ci] = i
            end
        end

        # Copy selected columns into task-local buffer
        @inbounds for c in 1:ncols
            col = j_arr[r][c]
            for row in 1:m
                Aj_local[row, c] = A[row, col]
            end
        end

        # Solve least squares via QR (fast), fall back to pinv if singular
        try
            xhat_arr[r] .= Aj_local \ b
        catch e
            e isa SingularException || rethrow()
            xhat_arr[r] .= pinv(Aj_local) * b
        end

        cur_sp = count(xi -> abs(xi) > sparsity_threshold, xhat_arr[r])

        # Handle sparsity == 0 edge case: treat as worst (skip as candidate)
        if cur_sp == 0
            cur_sp = typemax(Int) - 1
        end

        sp_arr[r]  = cur_sp
        err_arr[r] = norm(xhat_arr[r], 1)
    end

    # Reduce to find global best: prefer fewer nonzeros, then smaller L1 norm
    best_r = 1
    for r in 2:num_combos
        if sp_arr[r] < sp_arr[best_r] || (sp_arr[r] == sp_arr[best_r] && err_arr[r] < err_arr[best_r])
            best_r = r
        end
    end

    # Reconstruct final solution
    x_kr = zeros(T, n)
    @inbounds for i in 1:ncols
        x_kr[j_arr[best_r][i]] = xhat_arr[best_r][i]
    end

    return x_kr
end

# Convenience method for mixed numeric types
function AKRON(A::AbstractMatrix, b::AbstractVector; kwargs...)
    T = promote_type(eltype(A), eltype(b))
    return AKRON(convert(Matrix{T}, A), convert(Vector{T}, b); kwargs...)
end
