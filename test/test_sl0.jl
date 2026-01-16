@testset "SL0" begin
    @testset "Basic sparse recovery" begin
        # Create a well-conditioned sparse recovery problem
        Random.seed!(123)
        n, p, k = 50, 200, 5
        A, x_true, b = cs_model(n, p, k)

        x_recovered = SL0(A, b; maxiter=200, epsilon=0.01)

        # Check that we recover the support (non-zero locations)
        true_support = findall(!iszero, x_true)
        recovered_support = findall(xi -> abs(xi) > 0.1, x_recovered)

        # Should recover most of the support
        support_overlap = length(intersect(true_support, recovered_support))
        @test support_overlap >= k - 2  # Allow some tolerance
    end

    @testset "Recovery accuracy" begin
        Random.seed!(456)
        n, p, k = 60, 150, 8
        A, x_true, b = cs_model(n, p, k)

        x_recovered = SL0(A, b; maxiter=200)

        # Relative error should be reasonable
        rel_error = norm(x_recovered - x_true) / norm(x_true)
        @test rel_error < 0.5  # Within 50% relative error
    end

    @testset "Output properties" begin
        Random.seed!(789)
        n, p, k = 40, 100, 5
        A, x_true, b = cs_model(n, p, k)

        x_recovered = SL0(A, b)

        # Output should be a vector of correct length
        @test length(x_recovered) == p
        @test eltype(x_recovered) <: Real

        # Solution should approximately satisfy Ax = b
        residual = norm(A * x_recovered - b) / norm(b)
        @test residual < 0.1
    end

    @testset "Parameter variations" begin
        Random.seed!(101)
        n, p, k = 30, 80, 4
        A, x_true, b = cs_model(n, p, k)

        # Different sigma_decrease_factor
        x1 = SL0(A, b; sigma_decrease_factor=0.9)
        x2 = SL0(A, b; sigma_decrease_factor=0.7)

        @test length(x1) == p
        @test length(x2) == p
    end

    @testset "Type flexibility" begin
        Random.seed!(202)
        A = randn(20, 50)
        b = randn(20)

        # Should work with default Float64
        x = SL0(A, b; maxiter=50)
        @test length(x) == 50
    end
end
