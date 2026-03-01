@testset "ADMM" begin
    @testset "Basic sparse recovery" begin
        Random.seed!(123)
        n, p, k = 50, 200, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = ADMM(A, b; lambda=0.01, maxiter=1000)

        true_support = findall(!iszero, x_true)
        recovered_support = findall(xi -> abs(xi) > 0.1, x_recovered)
        support_overlap = length(intersect(true_support, recovered_support))
        @test support_overlap >= k - 2
    end

    @testset "Recovery accuracy" begin
        Random.seed!(456)
        n, p, k = 60, 150, 8
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = ADMM(A, b; lambda=0.01, maxiter=1000)

        rel_error = norm(x_recovered - x_true) / norm(x_true)
        @test rel_error < 0.5
    end

    @testset "Output properties" begin
        Random.seed!(789)
        n, p, k = 40, 100, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = ADMM(A, b; lambda=0.1)

        @test length(x_recovered) == p
        @test eltype(x_recovered) <: Real
    end

    @testset "Rho parameter" begin
        Random.seed!(101)
        n, p, k = 40, 100, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        # Different rho values should all produce valid results
        x1 = ADMM(A, b; lambda=0.01, rho=0.1)
        x2 = ADMM(A, b; lambda=0.01, rho=1.0)
        x3 = ADMM(A, b; lambda=0.01, rho=10.0)

        @test length(x1) == p
        @test length(x2) == p
        @test length(x3) == p
    end

    @testset "Different sensing matrices" begin
        Random.seed!(303)

        A, x_true, b = bernoulli_sensing(40, 100, 5)
        x_recovered = ADMM(A, b; lambda=0.01)
        @test length(x_recovered) == 100

        A2, x_true2, b2 = uniform_sensing(40, 100, 5)
        x_recovered2 = ADMM(A2, b2; lambda=0.01)
        @test length(x_recovered2) == 100
    end
end
