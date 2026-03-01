@testset "Metrics" begin
    @testset "recovery_error" begin
        # Perfect recovery
        x_true = [1.0, 0.0, 0.0, 2.0, 0.0]
        @test recovery_error(x_true, x_true) == 0.0

        # Small error
        x_hat = [1.01, 0.0, 0.0, 1.99, 0.0]
        err = recovery_error(x_hat, x_true)
        @test err > 0
        @test err < 0.01

        # Zero true signal returns Inf
        @test recovery_error([1.0, 0.0], [0.0, 0.0]) == Inf
    end

    @testset "support_recovery" begin
        x_true = [1.0, 0.0, 0.0, 2.0, 0.0]

        # Perfect support recovery
        x_hat = [0.99, 0.0, 0.0, 2.01, 0.0]
        sr = support_recovery(x_hat, x_true)
        @test sr.precision == 1.0
        @test sr.recall == 1.0
        @test sr.f1 == 1.0

        # Partial recovery: recovered {1,3,4}, true is {1,4}
        x_hat2 = [1.0, 0.0, 0.5, 2.0, 0.0]
        sr2 = support_recovery(x_hat2, x_true)
        @test sr2.precision ≈ 2 / 3
        @test sr2.recall == 1.0

        # Missed entry: recovered {1}, true is {1,4}
        x_hat3 = [1.0, 0.0, 0.0, 0.0, 0.0]
        sr3 = support_recovery(x_hat3, x_true)
        @test sr3.precision == 1.0
        @test sr3.recall == 0.5

        # Both empty
        sr4 = support_recovery(zeros(5), zeros(5))
        @test sr4.precision == 1.0
        @test sr4.recall == 1.0
        @test sr4.f1 == 1.0
    end

    @testset "snr" begin
        x_true = [1.0, 0.0, 0.0, 2.0, 0.0]

        # Perfect recovery gives Inf SNR
        @test snr(x_true, x_true) == Inf

        # Known error
        x_hat = [1.0, 0.0, 0.0, 2.0, 0.1]
        snr_val = snr(x_hat, x_true)
        # signal_power = 5, noise_power = 0.01 -> SNR = 10*log10(500) ≈ 26.99 dB
        @test snr_val ≈ 10 * log10(5.0 / 0.01)

        # Zero true signal
        @test snr([1.0], [0.0]) == -Inf
    end

    @testset "nmse" begin
        x_true = [1.0, 0.0, 0.0, 2.0, 0.0]

        # Perfect recovery
        @test nmse(x_true, x_true) == 0.0

        # Known error
        x_hat = [1.1, 0.0, 0.0, 2.0, 0.0]
        @test nmse(x_hat, x_true) ≈ 0.01 / 5.0

        # Zero true signal
        @test nmse([1.0], [0.0]) == Inf
    end

    @testset "phase_transition" begin
        Random.seed!(42)

        # Small phase transition test with OMP (fast algorithm)
        result = phase_transition(
            OMP, [30, 50], [3, 5], 100;
            trials=3, success_tol=0.1, sparsity_from_k=true
        )

        @test size(result) == (2, 2)
        @test all(0.0 .<= result .<= 1.0)

        # More measurements + lower sparsity should have higher success
        # (result[2,1] = n=50,k=3 should be >= result[1,2] = n=30,k=5)
        @test result[2, 1] >= result[1, 2] - 0.5  # Allow some randomness

        # Skip invalid combos: k > n
        result2 = phase_transition(
            OMP, [5], [10], 100;
            trials=2, sparsity_from_k=true
        )
        @test result2[1, 1] == 0.0
    end
end
