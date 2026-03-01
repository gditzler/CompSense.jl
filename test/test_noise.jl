@testset "Noise Utilities" begin
    @testset "add_noise with SNR" begin
        Random.seed!(42)
        b = randn(100)

        b_noisy = add_noise(b; snr=20.0)
        @test length(b_noisy) == length(b)
        @test b_noisy != b

        # Verify approximate SNR (allow some variance)
        actual_snr = 10 * log10(norm(b)^2 / norm(b_noisy - b)^2)
        @test abs(actual_snr - 20.0) < 5.0
    end

    @testset "add_noise with sigma" begin
        Random.seed!(42)
        b = ones(1000)

        b_noisy = add_noise(b; sigma=0.1)
        noise = b_noisy - b
        @test length(b_noisy) == length(b)

        # Noise standard deviation should be approximately sigma
        noise_std = sqrt(sum(noise .^ 2) / length(noise) - (sum(noise) / length(noise))^2)
        @test abs(noise_std - 0.1) < 0.05
    end

    @testset "add_noise argument validation" begin
        b = randn(10)
        @test_throws ArgumentError add_noise(b)
        @test_throws ArgumentError add_noise(b; snr=20.0, sigma=0.1)
    end

    @testset "Sensing generators with noise" begin
        Random.seed!(42)
        n, p, k = 50, 200, 10

        # Gaussian with noise
        A, x, b = gaussian_sensing(n, p, k; snr=20.0)
        @test norm(b - A * x) > 0  # b != Ax

        # Bernoulli with noise
        A2, x2, b2 = bernoulli_sensing(n, p, k; snr=20.0)
        @test norm(b2 - A2 * x2) > 0

        # Uniform with noise
        A3, x3, b3 = uniform_sensing(n, p, k; snr=20.0)
        @test norm(b3 - A3 * x3) > 0

        # Without noise (default)
        A4, x4, b4 = gaussian_sensing(n, p, k)
        @test norm(b4 - A4 * x4) < 1e-10
    end
end
