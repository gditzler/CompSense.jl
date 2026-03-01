@testset "SVT" begin
    @testset "generate_matrix_completion_problem" begin
        Random.seed!(100)
        m, n, r = 30, 30, 3
        fraction = 0.5

        Omega, values, M_true = generate_matrix_completion_problem(m, n, r, fraction)

        @test size(M_true) == (m, n)

        # M_true should be rank r
        @test rank(M_true) == r

        # Number of observed entries should match
        expected_num = round(Int, fraction * m * n)
        @test length(Omega) == expected_num
        @test length(values) == expected_num

        # Values should match M_true at observed locations
        for (idx, (i, j)) in enumerate(Omega)
            @test values[idx] ≈ M_true[i, j]
        end

        # Indices should be valid
        for (i, j) in Omega
            @test 1 <= i <= m
            @test 1 <= j <= n
        end
    end

    @testset "generate_matrix_completion_problem validation" begin
        # Invalid fraction
        @test_throws ArgumentError generate_matrix_completion_problem(10, 10, 2, 0.0)
        @test_throws ArgumentError generate_matrix_completion_problem(10, 10, 2, 1.5)

        # Invalid rank
        @test_throws ArgumentError generate_matrix_completion_problem(10, 10, 15, 0.5)
    end

    @testset "Recovery error" begin
        Random.seed!(123)
        m, n, r = 20, 20, 2
        fraction = 0.8

        Omega, values, M_true = generate_matrix_completion_problem(m, n, r, fraction)
        # Use smaller tau appropriate for this matrix size
        M_recovered = SVT(Omega, values, m, n; tau=5.0 * sqrt(r * m), maxiter=1000, tol=1e-5)

        # Recovery error should be reasonable
        rel_error = norm(M_recovered - M_true) / norm(M_true)
        @test rel_error < 0.5
    end

    @testset "Output dimensions" begin
        Random.seed!(456)
        m, n, r = 15, 20, 2
        fraction = 0.7

        Omega, values, M_true = generate_matrix_completion_problem(m, n, r, fraction)
        M_recovered = SVT(Omega, values, m, n)

        @test size(M_recovered) == (m, n)
        @test eltype(M_recovered) <: Real
    end

    @testset "Approximate low-rank" begin
        Random.seed!(789)
        m, n, r = 20, 20, 2
        fraction = 0.8

        Omega, values, M_true = generate_matrix_completion_problem(m, n, r, fraction)
        M_recovered = SVT(Omega, values, m, n; tau=5.0 * sqrt(r * m), maxiter=1000)

        # Recovered matrix should be approximately low-rank
        sv = svdvals(M_recovered)
        # Most energy should be in the first r singular values
        total_energy = sum(sv.^2)
        top_r_energy = sum(sv[1:r].^2)
        if total_energy > 0
            @test top_r_energy / total_energy > 0.5
        end
    end

    @testset "Observed entry consistency" begin
        Random.seed!(101)
        m, n, r = 20, 20, 2
        fraction = 0.8

        Omega, values, M_true = generate_matrix_completion_problem(m, n, r, fraction)
        M_recovered = SVT(Omega, values, m, n; tau=5.0 * sqrt(r * m), maxiter=1000, tol=1e-5)

        # Recovered matrix should approximately match observed entries
        obs_error = 0.0
        for (idx, (i, j)) in enumerate(Omega)
            obs_error += (M_recovered[i, j] - values[idx])^2
        end
        obs_rmse = sqrt(obs_error / length(Omega))
        @test obs_rmse < norm(M_true) * 0.3
    end

    @testset "Custom parameters" begin
        Random.seed!(202)
        m, n, r = 15, 15, 2
        fraction = 0.7

        Omega, values, M_true = generate_matrix_completion_problem(m, n, r, fraction)

        # Should run without error with custom parameters
        M1 = SVT(Omega, values, m, n; tau=10.0)
        M2 = SVT(Omega, values, m, n; delta=0.5)

        @test size(M1) == (m, n)
        @test size(M2) == (m, n)
    end
end
