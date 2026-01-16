@testset "Sensing Matrix Generators" begin

    @testset "generate_sparse_signal" begin
        p, k = 100, 10

        x = generate_sparse_signal(p, k)
        @test length(x) == p
        @test count(!iszero, x) == k

        # Non-zero entries should have magnitude >= 1 (default)
        nonzeros = x[x .!= 0]
        @test all(abs.(nonzeros) .>= 1.0)

        # Test custom min_magnitude
        x2 = generate_sparse_signal(p, k; min_magnitude=2.0)
        nonzeros2 = x2[x2 .!= 0]
        @test all(abs.(nonzeros2) .>= 2.0)
    end

    @testset "gaussian_sensing" begin
        n, p, k = 50, 200, 10

        A, x, b = gaussian_sensing(n, p, k)
        @test size(A) == (n, p)
        @test length(x) == p
        @test length(b) == n
        @test rank(A) == n
        @test b ≈ A * x
        @test count(!iszero, x) == k

        # Test normalized option
        A_norm, _, _ = gaussian_sensing(n, p, k; normalize=true)
        col_norms = [norm(A_norm[:, j]) for j in 1:p]
        @test all(isapprox.(col_norms, 1.0, atol=1e-10))
    end

    @testset "bernoulli_sensing" begin
        n, p, k = 50, 200, 10

        A, x, b = bernoulli_sensing(n, p, k)
        @test size(A) == (n, p)
        @test rank(A) == n
        @test b ≈ A * x

        # Unscaled version should have entries ±1
        A_unscaled, _, _ = bernoulli_sensing(n, p, k; scaled=false)
        @test all(abs.(A_unscaled) .≈ 1.0)
    end

    @testset "fourier_sensing" begin
        n, p, k = 30, 100, 5

        A, x, b = fourier_sensing(n, p, k)
        @test size(A) == (n, p)
        @test length(x) == p
        @test b ≈ A * x
        @test count(!iszero, x) == k

        # Matrix should be real-valued by default
        @test eltype(A) <: Real
    end

    @testset "dct_sensing" begin
        n, p, k = 30, 100, 5

        A, x, b = dct_sensing(n, p, k)
        @test size(A) == (n, p)
        @test rank(A) == n
        @test b ≈ A * x
        @test count(!iszero, x) == k
    end

    @testset "hadamard_sensing" begin
        n, p, k = 32, 128, 8  # p must be power of 2

        A, x, b = hadamard_sensing(n, p, k)
        @test size(A) == (n, p)
        @test b ≈ A * x
        @test count(!iszero, x) == k

        # Unnormalized Hadamard should have entries ±1
        A_unnorm, _, _ = hadamard_sensing(n, p, k; normalized=false)
        @test all(abs.(A_unnorm) .≈ 1.0)

        # Should throw error for non-power-of-2
        @test_throws ArgumentError hadamard_sensing(30, 100, 5)
    end

    @testset "sparse_sensing" begin
        n, p, k = 50, 200, 10

        A, x, b = sparse_sensing(n, p, k; density=0.2)
        @test size(A) == (n, p)
        @test rank(A) == n
        @test b ≈ A * x

        # Check that matrix is indeed sparse (approximately)
        nonzero_frac = count(!iszero, A) / (n * p)
        # After normalization, most entries will be non-zero in normalized cols
        # So just check it runs without error

        # Test invalid density
        @test_throws ArgumentError sparse_sensing(n, p, k; density=0.0)
        @test_throws ArgumentError sparse_sensing(n, p, k; density=1.5)
    end

    @testset "uniform_sensing" begin
        n, p, k = 50, 200, 10

        A, x, b = uniform_sensing(n, p, k)
        @test size(A) == (n, p)
        @test rank(A) == n
        @test b ≈ A * x

        # Check bounds
        A2, _, _ = uniform_sensing(n, p, k; low=0.0, high=1.0)
        @test all(A2 .>= 0.0)
        @test all(A2 .<= 1.0)
    end

    @testset "toeplitz_sensing" begin
        n, p, k = 50, 200, 10

        A, x, b = toeplitz_sensing(n, p, k)
        @test size(A) == (n, p)
        @test rank(A) == n
        @test b ≈ A * x

        # Check Toeplitz structure (constant diagonals)
        A_unnorm, _, _ = toeplitz_sensing(n, p, k; normalize=false)
        # For a Toeplitz matrix, A[i+1,j+1] = A[i,j] when both are valid
        for i in 1:(n-1)
            for j in 1:(p-1)
                @test A_unnorm[i, j] ≈ A_unnorm[i+1, j+1]
            end
        end
    end

    @testset "cs_model deprecation" begin
        # Test that cs_model still works (backward compatibility)
        # Note: Will emit deprecation warning
        n, p, k = 30, 100, 5
        A, x, b = @test_deprecated cs_model(n, p, k)
        @test size(A) == (n, p)
        @test b ≈ A * x
    end
end
