@testset "AMP" begin
    @testset "Basic sparse recovery" begin
        Random.seed!(123)
        n, p, k = 80, 200, 5
        # AMP works best with Gaussian matrices scaled by 1/sqrt(m)
        A, x_true, b = gaussian_sensing(n, p, k)
        A ./= sqrt(n)
        b = A * x_true

        x_recovered = AMP(A, b; maxiter=1000)

        true_support = findall(!iszero, x_true)
        recovered_support = findall(xi -> abs(xi) > 0.1, x_recovered)
        support_overlap = length(intersect(true_support, recovered_support))
        @test support_overlap >= k - 3
    end

    @testset "Output properties" begin
        Random.seed!(789)
        n, p, k = 80, 200, 5
        A, x_true, b = gaussian_sensing(n, p, k)
        A ./= sqrt(n)
        b = A * x_true

        x_recovered = AMP(A, b)

        @test length(x_recovered) == p
        @test eltype(x_recovered) <: Real
    end

    @testset "Convergence" begin
        Random.seed!(456)
        n, p, k = 100, 200, 5
        A, x_true, b = gaussian_sensing(n, p, k)
        A ./= sqrt(n)
        b = A * x_true

        x_recovered = AMP(A, b; maxiter=500)

        # Should achieve reasonable residual
        residual = norm(A * x_recovered - b)
        @test residual < 1.0
    end
end
