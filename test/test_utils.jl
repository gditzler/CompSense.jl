@testset "cs_model" begin
    @testset "Basic functionality" begin
        n, p, k = 50, 200, 10
        A, x, b = cs_model(n, p, k)

        @test size(A) == (n, p)
        @test length(x) == p
        @test length(b) == n
    end

    @testset "Matrix properties" begin
        n, p, k = 30, 100, 5
        A, x, b = cs_model(n, p, k)

        # A should have full row rank
        @test rank(A) == n

        # b should equal A*x
        @test b ≈ A * x
    end

    @testset "Sparsity" begin
        n, p, k = 50, 200, 15
        A, x, b = cs_model(n, p, k)

        # x should have exactly k non-zeros
        @test count(!iszero, x) == k

        # Non-zero entries should have magnitude >= 1
        nonzeros = x[x .!= 0]
        @test all(abs.(nonzeros) .>= 1.0)
    end

    @testset "Different problem sizes" begin
        # Small problem
        A1, x1, b1 = cs_model(10, 50, 3)
        @test size(A1) == (10, 50)
        @test count(!iszero, x1) == 3

        # Larger problem
        A2, x2, b2 = cs_model(100, 500, 20)
        @test size(A2) == (100, 500)
        @test count(!iszero, x2) == 20
    end

    @testset "Invalid type throws error" begin
        @test_throws ArgumentError cs_model(10, 50, 3; type="InvalidType")
    end
end
