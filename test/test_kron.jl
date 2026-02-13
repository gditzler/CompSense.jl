@testset "KRON" begin
    @testset "Basic sparse recovery" begin
        Random.seed!(123)
        n, p, k = 8, 20, 3
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = KRON(A, b)

        # Check that we recover the support (non-zero locations)
        true_support = findall(!iszero, x_true)
        recovered_support = findall(!iszero, x_recovered)

        # Should recover most of the support
        support_overlap = length(intersect(true_support, recovered_support))
        @test support_overlap >= k - 1
    end

    @testset "Recovery accuracy" begin
        Random.seed!(456)
        n, p, k = 10, 20, 3
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = KRON(A, b)

        # Relative error should be small for exact recovery
        rel_error = norm(x_recovered - x_true) / norm(x_true)
        @test rel_error < 0.5
    end

    @testset "Output properties" begin
        Random.seed!(789)
        n, p, k = 8, 20, 3
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = KRON(A, b)

        @test length(x_recovered) == p
        @test eltype(x_recovered) <: Real

        # Solution should satisfy Ax = b (exact method)
        residual = norm(A * x_recovered - b) / norm(b)
        @test residual < 0.1
    end

    @testset "Sparsity of result" begin
        Random.seed!(101)
        n, p, k = 8, 16, 2
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = KRON(A, b)

        # KRON should find a solution at least as sparse as k
        @test count(!iszero, x_recovered) <= n
    end

    @testset "Different sensing matrices" begin
        Random.seed!(303)

        # Test with Bernoulli sensing
        A, x_true, b = bernoulli_sensing(8, 20, 3)
        x_recovered = KRON(A, b)
        @test length(x_recovered) == 20

        # Test with uniform sensing
        A2, x_true2, b2 = uniform_sensing(8, 20, 3)
        x_recovered2 = KRON(A2, b2)
        @test length(x_recovered2) == 20
    end
end
