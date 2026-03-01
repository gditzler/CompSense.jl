@testset "SOMP" begin
    @testset "generate_mmv_problem" begin
        Random.seed!(100)
        n, p, k, L = 50, 200, 5, 3

        A, X, B = generate_mmv_problem(n, p, k, L)

        @test size(A) == (n, p)
        @test size(X) == (p, L)
        @test size(B) == (n, L)

        # X should have exactly k non-zero rows
        nonzero_rows = findall(i -> any(!iszero, X[i, :]), 1:p)
        @test length(nonzero_rows) == k

        # All columns of X should share the same support
        for l in 1:L
            support_l = findall(!iszero, X[:, l])
            @test Set(support_l) == Set(nonzero_rows)
        end

        # B should equal A*X
        @test B ≈ A * X atol=1e-10
    end

    @testset "generate_mmv_problem with noise" begin
        Random.seed!(101)
        n, p, k, L = 50, 200, 5, 3

        A, X, B = generate_mmv_problem(n, p, k, L; snr=20.0)

        @test size(B) == (n, L)
        # Noisy B should not exactly equal A*X
        @test norm(B - A * X) > 0
    end

    @testset "Joint support recovery" begin
        Random.seed!(123)
        n, p, k, L = 60, 150, 5, 4

        A, X_true, B = generate_mmv_problem(n, p, k, L)
        X_recovered = SOMP(A, B; sparsity=k)

        # Check row-support recovery
        true_row_support = findall(i -> any(!iszero, X_true[i, :]), 1:p)
        recovered_row_support = findall(i -> any(xi -> abs(xi) > 1e-10, X_recovered[i, :]), 1:p)

        support_overlap = length(intersect(true_row_support, recovered_row_support))
        @test support_overlap >= k - 2
    end

    @testset "Recovery accuracy" begin
        Random.seed!(456)
        n, p, k, L = 60, 150, 5, 3

        A, X_true, B = generate_mmv_problem(n, p, k, L)
        X_recovered = SOMP(A, B; sparsity=k)

        rel_error = norm(X_recovered - X_true) / norm(X_true)
        @test rel_error < 0.5
    end

    @testset "Output properties" begin
        Random.seed!(789)
        n, p, k, L = 40, 100, 5, 3

        A, X_true, B = generate_mmv_problem(n, p, k, L)
        X_recovered = SOMP(A, B; sparsity=k)

        @test size(X_recovered) == (p, L)
        @test eltype(X_recovered) <: Real

        # Residual should be small
        residual = norm(A * X_recovered - B) / norm(B)
        @test residual < 0.1
    end

    @testset "Row-sparsity constraint" begin
        Random.seed!(101)
        n, p, k, L = 40, 100, 4, 3

        A, X_true, B = generate_mmv_problem(n, p, k, L)
        X_recovered = SOMP(A, B; sparsity=k)

        # Number of non-zero rows should be at most k
        nonzero_rows = count(i -> any(!iszero, X_recovered[i, :]), 1:p)
        @test nonzero_rows <= k
    end

    @testset "L=1 reduces to OMP-like behavior" begin
        Random.seed!(202)
        n, p, k = 50, 100, 5
        L = 1

        A, X_true, B = generate_mmv_problem(n, p, k, L)
        X_recovered = SOMP(A, B; sparsity=k)

        @test size(X_recovered) == (p, 1)

        # Should recover most of the support
        true_support = findall(!iszero, X_true[:, 1])
        recovered_support = findall(xi -> abs(xi) > 1e-10, X_recovered[:, 1])
        support_overlap = length(intersect(true_support, recovered_support))
        @test support_overlap >= k - 2
    end
end
