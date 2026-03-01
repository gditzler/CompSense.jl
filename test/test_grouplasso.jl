@testset "GroupLASSO" begin
    @testset "Group-sparse recovery" begin
        Random.seed!(123)
        n, p = 60, 100
        group_size = 5
        num_groups = p ÷ group_size
        groups = [collect((g-1)*group_size+1 : g*group_size) for g in 1:num_groups]

        # Create group-sparse signal: 2 active groups
        x_true = zeros(p)
        active_groups = [1, 5]
        for g in active_groups
            x_true[groups[g]] = randn(group_size)
        end

        A = randn(n, p)
        b = A * x_true

        x_recovered = GroupLASSO(A, b, groups; lambda=0.01, maxiter=1000)

        # Check that active groups are recovered
        for g in active_groups
            @test norm(x_recovered[groups[g]]) > 0.1
        end
    end

    @testset "Output properties" begin
        Random.seed!(456)
        n, p = 40, 80
        group_size = 4
        num_groups = p ÷ group_size
        groups = [collect((g-1)*group_size+1 : g*group_size) for g in 1:num_groups]

        x_true = zeros(p)
        x_true[groups[1]] = randn(group_size)
        A = randn(n, p)
        b = A * x_true

        x_recovered = GroupLASSO(A, b, groups; lambda=0.1)

        @test length(x_recovered) == p
        @test eltype(x_recovered) <: Real
    end

    @testset "Rho parameter" begin
        Random.seed!(789)
        n, p = 40, 80
        group_size = 4
        num_groups = p ÷ group_size
        groups = [collect((g-1)*group_size+1 : g*group_size) for g in 1:num_groups]

        x_true = zeros(p)
        x_true[groups[2]] = randn(group_size)
        A = randn(n, p)
        b = A * x_true

        # Different rho values should all produce valid results
        x1 = GroupLASSO(A, b, groups; lambda=0.01, rho=0.1)
        x2 = GroupLASSO(A, b, groups; lambda=0.01, rho=1.0)
        x3 = GroupLASSO(A, b, groups; lambda=0.01, rho=10.0)

        @test length(x1) == p
        @test length(x2) == p
        @test length(x3) == p
    end

    @testset "Singleton groups reduce to ADMM-like behavior" begin
        Random.seed!(101)
        n, p, k = 50, 100, 5

        # Singleton groups: each group has one element
        groups = [[i] for i in 1:p]

        A, x_true, b = gaussian_sensing(n, p, k)

        x_group = GroupLASSO(A, b, groups; lambda=0.01, rho=1.0, maxiter=1000)
        x_admm = ADMM(A, b; lambda=0.01, rho=1.0, maxiter=1000)

        # With singleton groups, GroupLASSO block soft threshold reduces to
        # element-wise soft threshold, which is the same as ADMM
        @test x_group ≈ x_admm atol=1e-6
    end

    @testset "Group sparsity pattern" begin
        Random.seed!(202)
        n, p = 80, 120
        group_size = 4
        num_groups = p ÷ group_size
        groups = [collect((g-1)*group_size+1 : g*group_size) for g in 1:num_groups]

        # Create signal with 2 active groups
        x_true = zeros(p)
        x_true[groups[3]] = randn(group_size)
        x_true[groups[7]] = randn(group_size)

        A = randn(n, p)
        b = A * x_true

        x_recovered = GroupLASSO(A, b, groups; lambda=0.05, maxiter=1000)

        # Inactive groups should have small norms
        for g in 1:num_groups
            if !(g in [3, 7])
                @test norm(x_recovered[groups[g]]) < norm(x_true) * 0.3
            end
        end
    end
end
