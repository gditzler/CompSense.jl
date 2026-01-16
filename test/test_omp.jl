@testset "OMP" begin
    @testset "Basic sparse recovery" begin
        Random.seed!(123)
        n, p, k = 50, 200, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = OMP(A, b; sparsity=k)

        # Check that we recover the support (non-zero locations)
        true_support = findall(!iszero, x_true)
        recovered_support = findall(!iszero, x_recovered)

        # Should recover most of the support
        support_overlap = length(intersect(true_support, recovered_support))
        @test support_overlap >= k - 2
    end

    @testset "Recovery accuracy" begin
        Random.seed!(456)
        n, p, k = 60, 150, 8
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = OMP(A, b; sparsity=k)

        # Relative error should be reasonable
        rel_error = norm(x_recovered - x_true) / norm(x_true)
        @test rel_error < 0.5
    end

    @testset "Output properties" begin
        Random.seed!(789)
        n, p, k = 40, 100, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = OMP(A, b; sparsity=k)

        @test length(x_recovered) == p
        @test eltype(x_recovered) <: Real

        # Solution should approximately satisfy Ax = b
        residual = norm(A * x_recovered - b) / norm(b)
        @test residual < 0.1
    end

    @testset "Sparsity constraint" begin
        Random.seed!(101)
        n, p, k = 30, 80, 4
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = OMP(A, b; sparsity=k)

        # Number of non-zeros should be at most k
        @test sum(!iszero, x_recovered) <= k
    end

    @testset "Early stopping with tolerance" begin
        Random.seed!(202)
        n, p, k = 40, 100, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = OMP(A, b; sparsity=k, tol=1e-8)
        @test length(x_recovered) == p
    end

    @testset "Different sensing matrices" begin
        Random.seed!(303)

        # Test with Bernoulli sensing
        A, x_true, b = bernoulli_sensing(40, 100, 5)
        x_recovered = OMP(A, b; sparsity=5)
        @test length(x_recovered) == 100

        # Test with uniform sensing
        A2, x_true2, b2 = uniform_sensing(40, 100, 5)
        x_recovered2 = OMP(A2, b2; sparsity=5)
        @test length(x_recovered2) == 100
    end
end
