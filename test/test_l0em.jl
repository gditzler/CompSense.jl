@testset "L0EM" begin
    @testset "Basic sparse recovery" begin
        Random.seed!(111)
        n, p, k = 50, 200, 5
        A, x_true, b = cs_model(n, p, k)

        x_recovered = L0EM(A, b; maxiter=100, epsilon=0.01)

        # Check support recovery
        true_support = findall(!iszero, x_true)
        recovered_support = findall(xi -> abs(xi) > 0.1, x_recovered)

        support_overlap = length(intersect(true_support, recovered_support))
        @test support_overlap >= k - 2
    end

    @testset "Recovery accuracy" begin
        Random.seed!(222)
        n, p, k = 60, 150, 8
        A, x_true, b = cs_model(n, p, k)

        x_recovered = L0EM(A, b; maxiter=100)

        # Check relative error
        rel_error = norm(x_recovered - x_true) / norm(x_true)
        @test rel_error < 0.5
    end

    @testset "Output properties" begin
        Random.seed!(333)
        n, p, k = 40, 100, 5
        A, x_true, b = cs_model(n, p, k)

        x_recovered = L0EM(A, b)

        @test length(x_recovered) == p
        @test eltype(x_recovered) <: Real

        # Residual check
        residual = norm(A * x_recovered - b) / norm(b)
        @test residual < 0.2
    end

    @testset "Regularization parameter" begin
        Random.seed!(444)
        n, p, k = 30, 80, 4
        A, x_true, b = cs_model(n, p, k)

        # Different lambda values
        x1 = L0EM(A, b; lambda=0.01)
        x2 = L0EM(A, b; lambda=0.0001)

        @test length(x1) == p
        @test length(x2) == p
    end

    @testset "Convergence" begin
        Random.seed!(555)
        n, p, k = 30, 80, 4
        A, x_true, b = cs_model(n, p, k)

        # Should converge with few iterations for simple problems
        x_few = L0EM(A, b; maxiter=10)
        x_many = L0EM(A, b; maxiter=100)

        @test length(x_few) == p
        @test length(x_many) == p
    end

    @testset "Type flexibility" begin
        Random.seed!(666)
        A = randn(20, 50)
        b = randn(20)

        x = L0EM(A, b; maxiter=30)
        @test length(x) == 50
    end
end
