@testset "LASSO" begin
    @testset "Basic sparse recovery" begin
        Random.seed!(123)
        n, p, k = 50, 200, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = LASSO(A, b; lambda=0.01)

        true_support = findall(!iszero, x_true)
        recovered_support = findall(!iszero, x_recovered)
        support_overlap = length(intersect(true_support, recovered_support))
        @test support_overlap >= k - 2
    end

    @testset "Recovery accuracy" begin
        Random.seed!(456)
        n, p, k = 60, 150, 8
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = LASSO(A, b; lambda=0.01)

        rel_error = norm(x_recovered - x_true) / norm(x_true)
        @test rel_error < 0.5
    end

    @testset "Output properties" begin
        Random.seed!(789)
        n, p, k = 40, 100, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = LASSO(A, b; lambda=0.1)

        @test length(x_recovered) == p
        @test eltype(x_recovered) <: Real
    end

    @testset "Lambda effect on sparsity" begin
        Random.seed!(101)
        n, p, k = 50, 100, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        # Larger lambda -> sparser solution
        x_small_lambda = LASSO(A, b; lambda=0.1)
        x_large_lambda = LASSO(A, b; lambda=10.0)

        # Count entries above a meaningful threshold
        nnz_small = sum(xi -> abs(xi) > 0.01, x_small_lambda)
        nnz_large = sum(xi -> abs(xi) > 0.01, x_large_lambda)
        @test nnz_large <= nnz_small
    end
end
