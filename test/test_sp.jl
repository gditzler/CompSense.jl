@testset "SP (Subspace Pursuit)" begin
    @testset "Basic sparse recovery" begin
        Random.seed!(123)
        n, p, k = 50, 200, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = SP(A, b; sparsity=k)

        true_support = findall(!iszero, x_true)
        recovered_support = findall(!iszero, x_recovered)
        support_overlap = length(intersect(true_support, recovered_support))
        @test support_overlap >= k - 2
    end

    @testset "Recovery accuracy" begin
        Random.seed!(456)
        n, p, k = 60, 150, 8
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = SP(A, b; sparsity=k)

        rel_error = norm(x_recovered - x_true) / norm(x_true)
        @test rel_error < 0.5
    end

    @testset "Output properties" begin
        Random.seed!(789)
        n, p, k = 40, 100, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = SP(A, b; sparsity=k)

        @test length(x_recovered) == p
        @test eltype(x_recovered) <: Real

        # Solution should be at most k-sparse
        @test sum(!iszero, x_recovered) <= k
    end

    @testset "Sparsity constraint" begin
        Random.seed!(101)
        n, p, k = 30, 80, 4
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = SP(A, b; sparsity=k)
        @test sum(!iszero, x_recovered) <= k
    end

    @testset "Different sensing matrices" begin
        Random.seed!(303)

        A, x_true, b = bernoulli_sensing(40, 100, 5)
        x_recovered = SP(A, b; sparsity=5)
        @test length(x_recovered) == 100

        A2, x_true2, b2 = uniform_sensing(40, 100, 5)
        x_recovered2 = SP(A2, b2; sparsity=5)
        @test length(x_recovered2) == 100
    end
end
