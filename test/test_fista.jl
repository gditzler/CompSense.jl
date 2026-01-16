@testset "FISTA" begin
    @testset "Basic sparse recovery" begin
        Random.seed!(123)
        n, p, k = 50, 200, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = FISTA(A, b; lambda=0.1, maxiter=500)

        # Check that we recover the support (non-zero locations)
        true_support = findall(!iszero, x_true)
        recovered_support = findall(xi -> abs(xi) > 0.1, x_recovered)

        # Should recover most of the support
        support_overlap = length(intersect(true_support, recovered_support))
        @test support_overlap >= k - 2
    end

    @testset "Recovery accuracy" begin
        Random.seed!(456)
        n, p, k = 60, 150, 8
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = FISTA(A, b; lambda=0.05, maxiter=1000)

        # Relative error should be reasonable
        rel_error = norm(x_recovered - x_true) / norm(x_true)
        @test rel_error < 0.5
    end

    @testset "Output properties" begin
        Random.seed!(789)
        n, p, k = 40, 100, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = FISTA(A, b; lambda=0.1)

        @test length(x_recovered) == p
        @test eltype(x_recovered) <: Real
    end

    @testset "Lambda regularization effect" begin
        Random.seed!(101)
        n, p, k = 30, 80, 4
        A, x_true, b = gaussian_sensing(n, p, k)

        # Higher lambda should give sparser solution
        x_high_lambda = FISTA(A, b; lambda=1.0)
        x_low_lambda = FISTA(A, b; lambda=0.01)

        sparsity_high = sum(abs.(x_high_lambda) .< 0.01)
        sparsity_low = sum(abs.(x_low_lambda) .< 0.01)

        @test sparsity_high >= sparsity_low
    end

    @testset "Convergence tolerance" begin
        Random.seed!(202)
        n, p, k = 40, 100, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        x_tight = FISTA(A, b; lambda=0.1, tol=1e-8, maxiter=1000)
        x_loose = FISTA(A, b; lambda=0.1, tol=1e-3, maxiter=1000)

        @test length(x_tight) == p
        @test length(x_loose) == p
    end

    @testset "Different sensing matrices" begin
        Random.seed!(303)

        # Test with Bernoulli sensing
        A, x_true, b = bernoulli_sensing(40, 100, 5)
        x_recovered = FISTA(A, b; lambda=0.1)
        @test length(x_recovered) == 100

        # Test with uniform sensing
        A2, x_true2, b2 = uniform_sensing(40, 100, 5)
        x_recovered2 = FISTA(A2, b2; lambda=0.1)
        @test length(x_recovered2) == 100
    end
end
