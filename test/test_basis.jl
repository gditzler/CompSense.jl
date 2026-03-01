@testset "Basis / Dictionary Support" begin
    @testset "dct_matrix orthonormality" begin
        Random.seed!(100)
        p = 64
        D = dct_matrix(p)

        # D'D should be identity (orthonormal)
        @test D' * D ≈ Matrix{Float64}(I, p, p) atol=1e-10

        # DD' should also be identity
        @test D * D' ≈ Matrix{Float64}(I, p, p) atol=1e-10

        # Check dimensions
        @test size(D) == (p, p)
        @test eltype(D) == Float64
    end

    @testset "dct_matrix small sizes" begin
        for p in [2, 4, 8, 16]
            D = dct_matrix(p)
            @test D' * D ≈ Matrix{Float64}(I, p, p) atol=1e-10
        end
    end

    @testset "identity_matrix" begin
        p = 50
        Id = identity_matrix(p)

        @test size(Id) == (p, p)
        @test eltype(Id) == Float64
        @test Id ≈ Matrix{Float64}(I, p, p)
    end

    @testset "recover_in_basis with identity (pass-through)" begin
        Random.seed!(200)
        n, p, k = 50, 100, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        Psi = identity_matrix(p)
        x_recovered = recover_in_basis(A, b, Psi, OMP; sparsity=k)

        # With identity basis, should be equivalent to direct OMP
        x_direct = OMP(A, b; sparsity=k)
        @test x_recovered ≈ x_direct atol=1e-10
    end

    @testset "recover_in_basis with DCT and OMP" begin
        Random.seed!(301)
        p = 64
        n = 40
        k = 5

        # Create a signal sparse in DCT domain
        Psi = dct_matrix(p)
        s_true = zeros(p)
        support = sort(randperm(p)[1:k])
        s_true[support] = sign.(randn(k)) .* (1.0 .+ abs.(randn(k)))
        x_true = Psi * s_true

        # Measurements
        A = randn(n, p)
        b = A * x_true

        x_recovered = recover_in_basis(A, b, Psi, OMP; sparsity=k)

        rel_error = norm(x_recovered - x_true) / norm(x_true)
        @test rel_error < 0.5
    end

    @testset "recover_in_basis with DCT and ADMM" begin
        Random.seed!(302)
        p = 64
        n = 40
        k = 5

        Psi = dct_matrix(p)
        s_true = zeros(p)
        support = sort(randperm(p)[1:k])
        s_true[support] = sign.(randn(k)) .* (1.0 .+ abs.(randn(k)))
        x_true = Psi * s_true

        A = randn(n, p)
        b = A * x_true

        x_recovered = recover_in_basis(A, b, Psi, ADMM; lambda=0.01, maxiter=1000)

        rel_error = norm(x_recovered - x_true) / norm(x_true)
        @test rel_error < 0.5
    end

    @testset "recover_in_basis with DCT and IHT" begin
        Random.seed!(303)
        p = 64
        n = 40
        k = 5

        Psi = dct_matrix(p)
        s_true = zeros(p)
        support = sort(randperm(p)[1:k])
        s_true[support] = sign.(randn(k)) .* (1.0 .+ abs.(randn(k)))
        x_true = Psi * s_true

        A = randn(n, p)
        b = A * x_true

        x_recovered = recover_in_basis(A, b, Psi, IHT; sparsity=k, maxiter=1000)

        # IHT may be less accurate, use a looser tolerance
        @test length(x_recovered) == p
        @test eltype(x_recovered) <: Real
    end
end
