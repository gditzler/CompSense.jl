@testset "BasisPursuit" begin
    @testset "Basic sparse recovery" begin
        Random.seed!(123)
        n, p, k = 50, 200, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = BasisPursuit(A, b)

        # Check support recovery
        true_support = findall(!iszero, x_true)
        recovered_support = findall(!iszero, x_recovered)
        support_overlap = length(intersect(true_support, recovered_support))
        @test support_overlap >= k - 2
    end

    @testset "Recovery accuracy" begin
        Random.seed!(456)
        n, p, k = 60, 150, 8
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = BasisPursuit(A, b)

        rel_error = norm(x_recovered - x_true) / norm(x_true)
        @test rel_error < 0.1
    end

    @testset "Output properties" begin
        Random.seed!(789)
        n, p, k = 40, 100, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = BasisPursuit(A, b)

        @test length(x_recovered) == p
        @test eltype(x_recovered) <: Real
    end

    @testset "Different sensing matrices" begin
        Random.seed!(303)

        A, x_true, b = bernoulli_sensing(40, 100, 5)
        x_recovered = BasisPursuit(A, b)
        @test length(x_recovered) == 100

        A2, x_true2, b2 = uniform_sensing(40, 100, 5)
        x_recovered2 = BasisPursuit(A2, b2)
        @test length(x_recovered2) == 100
    end
end
