@testset "BPDN" begin
    @testset "Noiseless recovery" begin
        Random.seed!(123)
        n, p, k = 50, 200, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        # With very small sigma, should behave like Basis Pursuit
        x_recovered = BPDN(A, b; sigma=1e-6)

        rel_error = norm(x_recovered - x_true) / norm(x_true)
        @test rel_error < 0.2
    end

    @testset "Noisy recovery" begin
        Random.seed!(456)
        n, p, k = 60, 150, 8
        A, x_true, b = gaussian_sensing(n, p, k)

        # Add noise
        noise = 0.01 * randn(n)
        b_noisy = b + noise

        x_recovered = BPDN(A, b_noisy; sigma=norm(noise) * 1.1)

        # Should still get reasonable recovery with appropriate sigma
        rel_error = norm(x_recovered - x_true) / norm(x_true)
        @test rel_error < 0.5
    end

    @testset "Output properties" begin
        Random.seed!(789)
        n, p, k = 40, 100, 5
        A, x_true, b = gaussian_sensing(n, p, k)

        x_recovered = BPDN(A, b; sigma=0.1)

        @test length(x_recovered) == p
        @test eltype(x_recovered) <: Real
    end
end
