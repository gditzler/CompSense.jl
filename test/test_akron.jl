@testset "AKRON" begin
    @testset "Basic sparse recovery" begin
        Random.seed!(1111)
        # Use smaller problem size since AKRON uses convex optimization + combinatorics
        n, p, k = 20, 50, 3
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = AKRON(A, b; shift=3)

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

        x_recovered = AKRON(A, b)

        @test length(x_recovered) == p
        @test eltype(x_recovered) <: Real

        # Residual check — AKRON solves via least squares on selected columns
        residual = norm(A * x_recovered - b) / norm(b)
        @test residual < 0.1
    end

    @testset "Sparsity promotion" begin
        Random.seed!(3333)
        n, p, k = 15, 40, 3
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = AKRON(A, b; shift=3, sparsity_threshold=1e-3)

        # Solution should be sparse (many near-zero entries)
        num_small = count(xi -> abs(xi) < 0.1, x_recovered)
        @test num_small >= p - 2 * k
    end

    @testset "Shift parameter" begin
        Random.seed!(4444)
        n, p, k = 15, 40, 3
        A, x_true, b = gaussian_sensing(n, p, k)

        # Different shift values should all produce valid solutions
        x1 = AKRON(A, b; shift=2)
        x2 = AKRON(A, b; shift=5)

        @test length(x1) == p
        @test length(x2) == p
        @test norm(A * x1 - b) / norm(b) < 0.5
        @test norm(A * x2 - b) / norm(b) < 0.5
    end

    @testset "Different sensing matrices" begin
        Random.seed!(5555)

        # Test with Bernoulli sensing
        A, x_true, b = bernoulli_sensing(15, 40, 3)
        x_recovered = AKRON(A, b)
        @test length(x_recovered) == 40
    end
end
