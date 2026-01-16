@testset "IRWLS" begin
    @testset "Basic sparse recovery" begin
        Random.seed!(1111)
        # Use smaller problem size since IRWLS uses convex optimization
        n, p, k = 20, 50, 3
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = IRWLS(A, b; maxiter=5, epsilon=0.01)

        # Check support recovery
        true_support = findall(!iszero, x_true)
        recovered_support = findall(xi -> abs(xi) > 0.1, x_recovered)

        support_overlap = length(intersect(true_support, recovered_support))
        @test support_overlap >= k - 2
    end

    @testset "Output properties" begin
        Random.seed!(2222)
        n, p, k = 15, 40, 3
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = IRWLS(A, b; maxiter=3)

        @test length(x_recovered) == p
        @test eltype(x_recovered) <: Real

        # Residual check - IRWLS enforces equality constraint
        residual = norm(A * x_recovered - b) / norm(b)
        @test residual < 0.1
    end

    @testset "Sparsity promotion" begin
        Random.seed!(3333)
        n, p, k = 15, 40, 3
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = IRWLS(A, b; maxiter=5, epsilon=0.05)

        # Solution should be sparse (many near-zero entries)
        num_small = count(xi -> abs(xi) < 0.1, x_recovered)
        @test num_small >= p - 2 * k
    end

    @testset "Epsilon thresholding" begin
        Random.seed!(4444)
        n, p, k = 15, 40, 3
        A, x_true, b = gaussian_sensing(n, p, k)

        epsilon = 0.1
        x_recovered = IRWLS(A, b; maxiter=3, epsilon=epsilon)

        # Entries smaller than epsilon should be exactly zero
        small_entries = x_recovered[abs.(x_recovered) .< epsilon]
        @test all(small_entries .== 0.0)
    end

    @testset "Different sensing matrices" begin
        Random.seed!(5555)

        # Test with Bernoulli sensing
        A, x_true, b = bernoulli_sensing(15, 40, 3)
        x_recovered = IRWLS(A, b; maxiter=3)
        @test length(x_recovered) == 40
    end
end
