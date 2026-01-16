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

# Note: LinearAlgebra, Random, FFTW are imported by the parent module

#==============================================================================#
# Helper Functions
#==============================================================================#

"""
    generate_sparse_signal(p, k; min_magnitude=1.0)

Generate a sparse signal vector with exactly k non-zero entries.

# Arguments
- `p::Integer`: Signal dimension
- `k::Integer`: Number of non-zero entries
- `min_magnitude::Real`: Minimum absolute value of non-zero entries (default: 1.0)

# Returns
- `x::Vector{Float64}`: Sparse signal with k non-zeros at random positions
"""
function generate_sparse_signal(p::Integer, k::Integer; min_magnitude::Real=1.0)
    x = zeros(p)
    # Generate non-zero values with magnitude >= min_magnitude
    nonzero_values = sign.(randn(k)) .* (min_magnitude .+ abs.(randn(k)))
    # Randomly select k positions
    nonzero_positions = randperm(p)[1:k]
    x[nonzero_positions] = nonzero_values
    return x
end

"""
    ensure_full_row_rank(A, n, generator)

Regenerate matrix until it has full row rank.
"""
function ensure_full_row_rank(A::AbstractMatrix, n::Integer, generator::Function)
    while rank(A) != n
        A = generator()
    end
    return A
end

#==============================================================================#
# Sensing Matrix Generators
#==============================================================================#

"""
    gaussian_sensing(n, p, k; normalize=false)

Generate a compressed sensing problem with Gaussian sensing matrix.

The sensing matrix A has i.i.d. entries drawn from N(0, 1). Gaussian matrices
satisfy the Restricted Isometry Property (RIP) with high probability when
n ≥ O(k log(p/k)), making them ideal for theoretical analysis.

# Arguments
- `n::Integer`: Number of measurements (rows)
- `p::Integer`: Signal dimension (columns), typically n < p
- `k::Integer`: Sparsity level (number of non-zeros in true signal)
- `normalize::Bool`: If true, normalize columns to unit norm (default: false)

# Returns
- `A::Matrix{Float64}`: n × p Gaussian sensing matrix with full row rank
- `x::Vector{Float64}`: Sparse signal with exactly k non-zeros
- `b::Vector{Float64}`: Measurement vector b = Ax

# Example
```julia
A, x, b = gaussian_sensing(50, 200, 10)
x_recovered = SL0(A, b)
```

# References
- Candès & Tao (2005), "Decoding by Linear Programming"
"""
function gaussian_sensing(n::Integer, p::Integer, k::Integer; normalize::Bool=false)
    # Generate Gaussian matrix with full row rank
    A = randn(n, p)
    A = ensure_full_row_rank(A, n, () -> randn(n, p))

    if normalize
        # Normalize columns to unit norm
        for j in 1:p
            A[:, j] ./= norm(A[:, j])
        end
    end

    x = generate_sparse_signal(p, k)
    b = A * x

    return A, x, b
end

"""
    bernoulli_sensing(n, p, k; scaled=true)

Generate a compressed sensing problem with Bernoulli (Rademacher) sensing matrix.

The sensing matrix A has i.i.d. entries that are ±1 with equal probability.
Like Gaussian matrices, Bernoulli matrices satisfy RIP with high probability.
They are computationally simpler and useful for hardware implementations.

# Arguments
- `n::Integer`: Number of measurements (rows)
- `p::Integer`: Signal dimension (columns)
- `k::Integer`: Sparsity level
- `scaled::Bool`: If true, scale by 1/√n for proper normalization (default: true)

# Returns
- `A::Matrix{Float64}`: n × p Bernoulli sensing matrix
- `x::Vector{Float64}`: Sparse signal with exactly k non-zeros
- `b::Vector{Float64}`: Measurement vector b = Ax

# Example
```julia
A, x, b = bernoulli_sensing(50, 200, 10)
x_recovered = L0EM(A, b)
```

# References
- Achlioptas (2003), "Database-friendly random projections"
"""
function bernoulli_sensing(n::Integer, p::Integer, k::Integer; scaled::Bool=true)
    # Generate ±1 entries with equal probability
    A = Float64.(2 .* (rand(n, p) .> 0.5) .- 1)
    A = ensure_full_row_rank(A, n, () -> Float64.(2 .* (rand(n, p) .> 0.5) .- 1))

    if scaled
        A ./= sqrt(n)
    end

    x = generate_sparse_signal(p, k)
    b = A * x

    return A, x, b
end

"""
    fourier_sensing(n, p, k; real_valued=true)

Generate a compressed sensing problem with partial Fourier sensing matrix.

The sensing matrix consists of n randomly selected rows from a p × p DFT matrix.
This is highly relevant for applications like MRI, radar, and spectroscopy where
measurements are taken in the frequency domain.

# Arguments
- `n::Integer`: Number of measurements (rows to select)
- `p::Integer`: Signal dimension (DFT size)
- `k::Integer`: Sparsity level
- `real_valued::Bool`: If true, return real-valued matrix (default: true)

# Returns
- `A::Matrix{Float64}`: n × p partial Fourier matrix
- `x::Vector{Float64}`: Sparse signal with exactly k non-zeros
- `b::Vector{Float64}`: Measurement vector b = Ax

# Example
```julia
A, x, b = fourier_sensing(50, 200, 10)
x_recovered = SL0(A, b)
```

# Notes
- When real_valued=true, uses real part of DFT rows (still satisfies RIP)
- Fourier matrices allow O(p log p) fast matrix-vector products via FFT

# References
- Candès, Romberg & Tao (2006), "Robust uncertainty principles"
"""
function fourier_sensing(n::Integer, p::Integer, k::Integer; real_valued::Bool=true)
    # Construct full DFT matrix (normalized)
    # F[j,k] = exp(-2πi(j-1)(k-1)/p) / √p
    F = zeros(ComplexF64, p, p)
    for j in 1:p
        for l in 1:p
            F[j, l] = exp(-2π * im * (j - 1) * (l - 1) / p) / sqrt(p)
        end
    end

    # Randomly select n rows
    selected_rows = randperm(p)[1:n]
    A_complex = F[selected_rows, :]

    if real_valued
        # Use real and imaginary parts stacked, then select n rows
        A_real = real.(A_complex)
        A_imag = imag.(A_complex)
        # Combine and ensure we have n rows with good properties
        A = vcat(A_real, A_imag)
        A = A[1:n, :]  # Take first n rows
    else
        A = real.(A_complex)
    end

    # Ensure full row rank
    max_attempts = 100
    attempts = 0
    while rank(A) != n && attempts < max_attempts
        selected_rows = randperm(p)[1:n]
        A_complex = F[selected_rows, :]
        if real_valued
            A = real.(A_complex)
        else
            A = real.(A_complex)
        end
        attempts += 1
    end

    x = generate_sparse_signal(p, k)
    b = A * x

    return A, x, b
end

"""
    dct_sensing(n, p, k)

Generate a compressed sensing problem with partial DCT sensing matrix.

The sensing matrix consists of n randomly selected rows from a p × p
Discrete Cosine Transform (DCT-II) matrix. DCT is widely used in image
and video compression (JPEG, MPEG) and provides good incoherence with
canonical (sparse) bases.

# Arguments
- `n::Integer`: Number of measurements (rows to select)
- `p::Integer`: Signal dimension (DCT size)
- `k::Integer`: Sparsity level

# Returns
- `A::Matrix{Float64}`: n × p partial DCT matrix
- `x::Vector{Float64}`: Sparse signal with exactly k non-zeros
- `b::Vector{Float64}`: Measurement vector b = Ax

# Example
```julia
A, x, b = dct_sensing(50, 200, 10)
x_recovered = IRWLS(A, b)
```

# Notes
- DCT matrices are real-valued (no complex arithmetic needed)
- Allow O(p log p) fast transforms

# References
- Candès & Romberg (2007), "Sparsity and incoherence in compressive sampling"
"""
function dct_sensing(n::Integer, p::Integer, k::Integer)
    # Construct DCT-II matrix (orthonormal)
    # D[j,k] = √(2/p) * cos(π(j-1)(2k-1)/(2p)) for j>1
    # D[1,k] = √(1/p)
    D = zeros(p, p)

    for j in 1:p
        for l in 1:p
            if j == 1
                D[j, l] = sqrt(1 / p)
            else
                D[j, l] = sqrt(2 / p) * cos(π * (j - 1) * (2 * l - 1) / (2 * p))
            end
        end
    end

    # Randomly select n rows
    selected_rows = randperm(p)[1:n]
    A = D[selected_rows, :]

    # Ensure full row rank
    A = ensure_full_row_rank(A, n, () -> begin
        rows = randperm(p)[1:n]
        D[rows, :]
    end)

    x = generate_sparse_signal(p, k)
    b = A * x

    return A, x, b
end

"""
    hadamard_sensing(n, p, k; normalized=true)

Generate a compressed sensing problem with partial Hadamard sensing matrix.

The sensing matrix consists of n randomly selected rows from a Hadamard matrix.
Hadamard matrices have entries ±1 and allow O(p log p) fast transforms.
Requires p to be a power of 2.

# Arguments
- `n::Integer`: Number of measurements (rows to select)
- `p::Integer`: Signal dimension (must be power of 2)
- `k::Integer`: Sparsity level
- `normalized::Bool`: If true, normalize by 1/√p (default: true)

# Returns
- `A::Matrix{Float64}`: n × p partial Hadamard matrix
- `x::Vector{Float64}`: Sparse signal with exactly k non-zeros
- `b::Vector{Float64}`: Measurement vector b = Ax

# Example
```julia
A, x, b = hadamard_sensing(32, 128, 5)  # p must be power of 2
x_recovered = SL0(A, b)
```

# Notes
- Throws error if p is not a power of 2
- Very fast matrix-vector products via Walsh-Hadamard transform
"""
function hadamard_sensing(n::Integer, p::Integer, k::Integer; normalized::Bool=true)
    # Check that p is a power of 2
    if !ispow2(p)
        throw(ArgumentError("p must be a power of 2 for Hadamard matrix, got p=$p"))
    end

    # Construct Hadamard matrix using Sylvester's construction
    H = hadamard_matrix(p)

    if normalized
        H = H ./ sqrt(p)
    end

    # Randomly select n rows
    selected_rows = randperm(p)[1:n]
    A = Float64.(H[selected_rows, :])

    x = generate_sparse_signal(p, k)
    b = A * x

    return A, x, b
end

"""
    hadamard_matrix(n)

Construct an n × n Hadamard matrix using Sylvester's recursive construction.
n must be a power of 2.
"""
function hadamard_matrix(n::Integer)
    if n == 1
        return [1]
    end
    H_half = hadamard_matrix(n ÷ 2)
    return [H_half H_half; H_half -H_half]
end

"""
    sparse_sensing(n, p, k; density=0.1, normalize_cols=true)

Generate a compressed sensing problem with sparse random sensing matrix.

The sensing matrix A is itself sparse, with approximately density × n × p
non-zero entries drawn from N(0,1). Sparse sensing matrices are useful for
large-scale problems where dense matrices are impractical.

# Arguments
- `n::Integer`: Number of measurements (rows)
- `p::Integer`: Signal dimension (columns)
- `k::Integer`: Sparsity level of true signal
- `density::Real`: Fraction of non-zero entries in A (default: 0.1)
- `normalize_cols::Bool`: If true, normalize columns to unit norm (default: true)

# Returns
- `A::Matrix{Float64}`: n × p sparse sensing matrix (returned as dense)
- `x::Vector{Float64}`: Sparse signal with exactly k non-zeros
- `b::Vector{Float64}`: Measurement vector b = Ax

# Example
```julia
A, x, b = sparse_sensing(100, 1000, 20; density=0.05)
x_recovered = L0EM(A, b)
```

# Notes
- Low density (< 0.1) may cause rank deficiency; function retries if needed
- For very large problems, consider using SparseArrays

# References
- Gilbert et al. (2010), "Sparse Recovery Using Sparse Matrices"
"""
function sparse_sensing(n::Integer, p::Integer, k::Integer;
                        density::Real=0.1,
                        normalize_cols::Bool=true)
    if density <= 0 || density > 1
        throw(ArgumentError("density must be in (0, 1], got $density"))
    end

    # Generate sparse matrix: each entry is non-zero with probability 'density'
    function generate_sparse()
        mask = rand(n, p) .< density
        A = randn(n, p) .* mask
        return A
    end

    A = generate_sparse()
    A = ensure_full_row_rank(A, n, generate_sparse)

    if normalize_cols
        for j in 1:p
            col_norm = norm(A[:, j])
            if col_norm > 0
                A[:, j] ./= col_norm
            end
        end
    end

    x = generate_sparse_signal(p, k)
    b = A * x

    return A, x, b
end

"""
    uniform_sensing(n, p, k; low=-1.0, high=1.0)

Generate a compressed sensing problem with uniform random sensing matrix.

The sensing matrix A has i.i.d. entries drawn from U(low, high).

# Arguments
- `n::Integer`: Number of measurements (rows)
- `p::Integer`: Signal dimension (columns)
- `k::Integer`: Sparsity level
- `low::Real`: Lower bound of uniform distribution (default: -1.0)
- `high::Real`: Upper bound of uniform distribution (default: 1.0)

# Returns
- `A::Matrix{Float64}`: n × p uniform random matrix
- `x::Vector{Float64}`: Sparse signal with exactly k non-zeros
- `b::Vector{Float64}`: Measurement vector b = Ax

# Example
```julia
A, x, b = uniform_sensing(50, 200, 10)
x_recovered = SL0(A, b)
```
"""
function uniform_sensing(n::Integer, p::Integer, k::Integer;
                         low::Real=-1.0,
                         high::Real=1.0)
    range = high - low
    A = low .+ range .* rand(n, p)
    A = ensure_full_row_rank(A, n, () -> low .+ range .* rand(n, p))

    x = generate_sparse_signal(p, k)
    b = A * x

    return A, x, b
end

"""
    toeplitz_sensing(n, p, k; normalize=true)

Generate a compressed sensing problem with random Toeplitz sensing matrix.

A Toeplitz matrix has constant diagonals, making it suitable for modeling
convolution operations. Only 2p-1 random values are needed to define an
n × p Toeplitz matrix, enabling efficient storage and fast O(p log p)
matrix-vector products via FFT.

# Arguments
- `n::Integer`: Number of measurements (rows)
- `p::Integer`: Signal dimension (columns)
- `k::Integer`: Sparsity level
- `normalize::Bool`: If true, normalize by 1/√n (default: true)

# Returns
- `A::Matrix{Float64}`: n × p Toeplitz sensing matrix
- `x::Vector{Float64}`: Sparse signal with exactly k non-zeros
- `b::Vector{Float64}`: Measurement vector b = Ax

# Example
```julia
A, x, b = toeplitz_sensing(50, 200, 10)
x_recovered = SL0(A, b)
```

# Notes
- Toeplitz matrices model linear time-invariant (LTI) systems
- Useful for signal processing applications

# References
- Haupt et al. (2010), "Toeplitz Compressed Sensing Matrices"
"""
function toeplitz_sensing(n::Integer, p::Integer, k::Integer; normalize::Bool=true)
    # Generate the first column and first row values
    # For an n×p Toeplitz matrix, we need n + p - 1 values
    values = randn(n + p - 1)

    # Construct Toeplitz matrix
    # A[i,j] = values[i - j + p] (with appropriate indexing)
    A = zeros(n, p)
    for i in 1:n
        for j in 1:p
            A[i, j] = values[i - j + p]
        end
    end

    A = ensure_full_row_rank(A, n, () -> begin
        v = randn(n + p - 1)
        T = zeros(n, p)
        for i in 1:n
            for j in 1:p
                T[i, j] = v[i - j + p]
            end
        end
        T
    end)

    if normalize
        A ./= sqrt(n)
    end

    x = generate_sparse_signal(p, k)
    b = A * x

    return A, x, b
end

#==============================================================================#
# Deprecated: Original cs_model for backward compatibility
#==============================================================================#

"""
    cs_model(n, p, k; type="Gaussian")

!!! warning "Deprecated"
    This function is deprecated. Use the specific sensing functions instead:
    - `gaussian_sensing(n, p, k)`
    - `bernoulli_sensing(n, p, k)`
    - `fourier_sensing(n, p, k)`
    - `dct_sensing(n, p, k)`
    - `hadamard_sensing(n, p, k)`
    - `sparse_sensing(n, p, k)`
    - `uniform_sensing(n, p, k)`
    - `toeplitz_sensing(n, p, k)`
"""
function cs_model(n::Integer, p::Integer, k::Integer; type::String="Gaussian")
    Base.depwarn(
        "cs_model is deprecated, use gaussian_sensing, bernoulli_sensing, etc. instead",
        :cs_model
    )

    type_lower = lowercase(type)

    if type_lower == "gaussian"
        return gaussian_sensing(n, p, k)
    elseif type_lower == "bernoulli" || type_lower == "rademacher"
        return bernoulli_sensing(n, p, k)
    elseif type_lower == "fourier" || type_lower == "dft"
        return fourier_sensing(n, p, k)
    elseif type_lower == "dct"
        return dct_sensing(n, p, k)
    elseif type_lower == "hadamard"
        return hadamard_sensing(n, p, k)
    elseif type_lower == "sparse"
        return sparse_sensing(n, p, k)
    elseif type_lower == "uniform"
        return uniform_sensing(n, p, k)
    elseif type_lower == "toeplitz"
        return toeplitz_sensing(n, p, k)
    else
        throw(ArgumentError("Unknown sensing matrix type: '$type'"))
    end
end
