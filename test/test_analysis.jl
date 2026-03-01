@testset "Sensing Matrix Analysis" begin
    @testset "column_coherence_matrix" begin
        Random.seed!(42)
        A = randn(20, 10)
        G = column_coherence_matrix(A)

        @test size(G) == (10, 10)

        # Diagonal should be 1
        for i in 1:10
            @test G[i, i] ≈ 1.0 atol=1e-10
        end

        # Should be symmetric
        @test G ≈ G'

        # Off-diagonal entries in [-1, 1]
        for i in 1:10
            for j in 1:10
                @test -1.0 - 1e-10 <= G[i, j] <= 1.0 + 1e-10
            end
        end
    end

    @testset "mutual_coherence" begin
        Random.seed!(42)

        # Identity matrix should have coherence 0
        I_mat = Matrix{Float64}(I, 10, 10)
        @test mutual_coherence(I_mat) ≈ 0.0 atol=1e-10

        # Random matrix should have coherence in (0, 1)
        A = randn(50, 100)
        mu = mutual_coherence(A)
        @test mu > 0
        @test mu <= 1.0

        # Highly correlated columns -> high coherence
        A2 = randn(20, 5)
        A2[:, 2] = A2[:, 1] + 1e-6 * randn(20)  # Nearly identical columns
        @test mutual_coherence(A2) > 0.99
    end

    @testset "babel_function" begin
        Random.seed!(42)
        A = randn(50, 100)

        # Babel function should be non-negative and non-decreasing in k
        mu1_5 = babel_function(A, 5)
        mu1_10 = babel_function(A, 10)
        @test mu1_5 >= 0
        @test mu1_10 >= mu1_5

        # For k=1, babel function equals mutual coherence
        @test babel_function(A, 1) ≈ mutual_coherence(A) atol=1e-10

        # k too large
        @test_throws ArgumentError babel_function(A, 100)
    end

    @testset "spark" begin
        # Identity matrix: no linearly dependent columns -> spark = n+1
        I_mat = Matrix{Float64}(I, 5, 5)
        @test spark(I_mat) == 6

        # Matrix with duplicate column: spark = 2
        A = randn(5, 4)
        A_dup = hcat(A, A[:, 1])  # 5x5 with col 5 = col 1
        @test spark(A_dup) == 2

        # Too large matrix
        @test_throws ArgumentError spark(randn(10, 31))
    end
end
