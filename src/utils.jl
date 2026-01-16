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

# Note: LinearAlgebra and Random are imported by the parent module

"""
    cs_model(n, p, k; type="Gaussian")

Generate a synthetic compressed sensing problem for testing sparse recovery algorithms.

# Arguments
- `n::Integer`: Number of measurements (rows of sensing matrix)
- `p::Integer`: Signal dimension (columns of sensing matrix, n < p for underdetermined system)
- `k::Integer`: Sparsity level (number of non-zero entries in true signal)
- `type::String`: Type of sensing matrix to generate (default: "Gaussian")

# Returns
- `A::Matrix{Float64}`: Sensing matrix of size n × p with full row rank
- `x::Vector{Float64}`: True sparse signal with exactly k non-zeros
- `b::Vector{Float64}`: Measurement vector b = Ax

# Example
```julia
# Create a compressed sensing problem: 50 measurements, 200-dimensional signal, 10 non-zeros
A, x_true, b = cs_model(50, 200, 10)

# Recover the signal
x_recovered = SL0(A, b)

# Check recovery quality
println("Recovery error: ", norm(x_recovered - x_true))
```

# Notes
- The function ensures A has full row rank (rank = n)
- Non-zero entries in x have magnitude ≥ 1 (bounded away from zero)
- For reliable recovery, typically need n ≥ O(k log(p/k)) measurements
"""
function cs_model(n::Integer, p::Integer, k::Integer; type::String="Gaussian")
    if type == "Gaussian"
        # Generate random Gaussian sensing matrix with full row rank
        A = randn(n, p)
        while rank(A) != n
            A = randn(n, p)
        end

        # Generate sparse signal with k non-zeros
        # Magnitudes are 1 + |noise| to ensure entries are bounded away from zero
        x = zeros(p)
        nonzero_values = sign.(randn(k)) .* (ones(k) .+ abs.(randn(k)))

        # Randomly select k positions for non-zeros
        nonzero_positions = randperm(p)[1:k]
        x[nonzero_positions] = nonzero_values

        # Generate measurements
        b = A * x
    else
        throw(ArgumentError("Unknown sensing matrix type: '$type'. Supported types: \"Gaussian\""))
    end

    return A, x, b
end
