@testset "BIHT" begin
    @testset "onebit_sensing generator" begin
        Random.seed!(100)
        n, p, k = 200, 100, 5

        A, x, y = onebit_sensing(n, p, k)

        @test size(A) == (n, p)
        @test length(x) == p
        @test length(y) == n

        # x should be unit norm
        @test norm(x) ≈ 1.0 atol=1e-10

        # x should be k-sparse
        @test sum(!iszero, x) == k

        # y should be binary {-1, +1}
        @test all(yi -> yi == 1.0 || yi == -1.0, y)

        # y should equal sign(Ax)
        @test y == sign.(A * x)
    end

    @testset "Support overlap" begin
        Random.seed!(123)
        n, p, k = 300, 100, 5

        A, x_true, y = onebit_sensing(n, p, k)
        x_recovered = BIHT(A, y; sparsity=k, maxiter=2000)

        true_support = findall(!iszero, x_true)
        recovered_support = findall(xi -> abs(xi) > 0.01, x_recovered)

        support_overlap = length(intersect(true_support, recovered_support))
        @test support_overlap >= k - 2
    end

    @testset "Unit norm output" begin
        Random.seed!(456)
        n, p, k = 300, 100, 5

        A, x_true, y = onebit_sensing(n, p, k)
        x_recovered = BIHT(A, y; sparsity=k)

        # Output should be unit norm
        @test norm(x_recovered) ≈ 1.0 atol=1e-6
    end

    @testset "Sign consistency" begin
        Random.seed!(789)
        n, p, k = 400, 100, 5

        A, x_true, y = onebit_sensing(n, p, k)
        x_recovered = BIHT(A, y; sparsity=k, maxiter=2000)

        # Check sign consistency: sign(A*x_recovered) should match y
        y_recovered = sign.(A * x_recovered)
        consistency = sum(y_recovered .== y) / n
        @test consistency > 0.70
    end

    @testset "Angular error" begin
        Random.seed!(101)
        n, p, k = 400, 100, 5

        A, x_true, y = onebit_sensing(n, p, k)
        x_recovered = BIHT(A, y; sparsity=k, maxiter=2000)

        # Angular error: angle between x_true and x_recovered
        # Both are unit norm, so cos(angle) = dot(x_true, x_recovered)
        cos_angle = abs(dot(x_true, x_recovered))
        @test cos_angle > 0.5  # Should have some angular agreement
    end

    @testset "Output properties" begin
        Random.seed!(202)
        n, p, k = 200, 100, 5

        A, x_true, y = onebit_sensing(n, p, k)
        x_recovered = BIHT(A, y; sparsity=k)

        @test length(x_recovered) == p
        @test eltype(x_recovered) <: Real

        # Should be k-sparse
        @test sum(!iszero, x_recovered) <= k
    end

    @testset "Custom tau parameter" begin
        Random.seed!(303)
        n, p, k = 200, 100, 5

        A, x_true, y = onebit_sensing(n, p, k)

        # Should run without error with custom tau
        x1 = BIHT(A, y; sparsity=k, tau=0.01)
        x2 = BIHT(A, y; sparsity=k, tau=0.001)

        @test length(x1) == p
        @test length(x2) == p
    end
end
