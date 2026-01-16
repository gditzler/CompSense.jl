# Sparsity Sweep Benchmark
#
# This script benchmarks sparse recovery algorithms across different
# sparsity levels and generates a plot of recovery error vs. sparsity.
#
# Run with: julia --project=. examples/sparsity_sweep.jl

using CompSense
using LinearAlgebra
using Random
using Statistics
using Printf

# Check if Plots is available, install if needed
try
    using Plots
catch
    using Pkg
    Pkg.add("Plots")
    using Plots
end

println("=" ^ 70)
println("CompSense.jl - Sparsity Sweep Benchmark")
println("=" ^ 70)

# ============================================================================
# Experiment Parameters
# ============================================================================

const SIGNAL_DIM = 200           # Signal dimension (p)
const N_MEASUREMENTS = 80        # Number of measurements (n), ~40% of p
const SPARSITY_RANGE = 1:50      # k from 1 to 50 (0.5% to 25% of p)
const N_TRIALS = 100             # Number of trials per sparsity level
const SEED = 12345               # Random seed for reproducibility

# Algorithm-specific parameters (tuned for good performance)
const OMP_MARGIN = 0             # Extra iterations for OMP
const IHT_MAXITER = 500          # Max iterations for IHT
const COSAMP_MAXITER = 50        # Max iterations for CoSaMP
const FISTA_LAMBDA = 0.05        # Regularization for FISTA
const FISTA_MAXITER = 200        # Max iterations for FISTA
const SL0_MAXITER = 200          # Max iterations for SL0
const L0EM_MAXITER = 100         # Max iterations for L0EM

println("\nExperiment Configuration:")
println("  Signal dimension (p):    $SIGNAL_DIM")
println("  Measurements (n):        $N_MEASUREMENTS")
println("  Compression ratio:       $(round(N_MEASUREMENTS/SIGNAL_DIM * 100, digits=1))%")
println("  Sparsity range (k):      $(first(SPARSITY_RANGE)) to $(last(SPARSITY_RANGE))")
println("  Sparsity % range:        $(round(first(SPARSITY_RANGE)/SIGNAL_DIM*100, digits=1))% to $(round(last(SPARSITY_RANGE)/SIGNAL_DIM*100, digits=1))%")
println("  Trials per sparsity:     $N_TRIALS")

# ============================================================================
# Define Algorithms to Test
# ============================================================================

# Each algorithm is a tuple: (name, function, requires_sparsity)
algorithms = [
    ("SL0", (A, b, k) -> SL0(A, b; maxiter=SL0_MAXITER, epsilon=0.001), false),
    ("OMP", (A, b, k) -> OMP(A, b; sparsity=k+OMP_MARGIN), true),
    ("IHT", (A, b, k) -> IHT(A, b; sparsity=k, maxiter=IHT_MAXITER), true),
    ("CoSaMP", (A, b, k) -> CoSaMP(A, b; sparsity=k, maxiter=COSAMP_MAXITER), true),
    ("FISTA", (A, b, k) -> FISTA(A, b; lambda=FISTA_LAMBDA, maxiter=FISTA_MAXITER), false),
    ("L0EM", (A, b, k) -> L0EM(A, b; maxiter=L0EM_MAXITER, epsilon=0.001), false),
]

n_algorithms = length(algorithms)
n_sparsities = length(SPARSITY_RANGE)

# Pre-allocate results arrays
# Store mean and std of relative error for each (algorithm, sparsity) pair
errors_mean = zeros(n_algorithms, n_sparsities)
errors_std = zeros(n_algorithms, n_sparsities)

# ============================================================================
# Run Experiments
# ============================================================================

println("\n" * "=" ^ 70)
println("Running experiments...")
println("=" ^ 70)

Random.seed!(SEED)

for (i, k) in enumerate(SPARSITY_RANGE)
    sparsity_pct = round(k / SIGNAL_DIM * 100, digits=1)
    @printf("\rSparsity: k = %3d (%5.1f%%) ", k, sparsity_pct)

    # Run multiple trials for this sparsity level
    trial_errors = zeros(n_algorithms, N_TRIALS)

    for trial in 1:N_TRIALS
        # Generate a new random problem for each trial
        A, x_true, b = gaussian_sensing(N_MEASUREMENTS, SIGNAL_DIM, k)
        x_norm = norm(x_true)

        # Test each algorithm
        for (j, (name, alg_fn, _)) in enumerate(algorithms)
            try
                x_recovered = alg_fn(A, b, k)
                rel_error = norm(x_recovered - x_true) / x_norm
                trial_errors[j, trial] = rel_error
            catch e
                # If algorithm fails, record NaN
                trial_errors[j, trial] = NaN
            end
        end
    end

    # Compute mean and std across trials (ignoring NaN)
    for j in 1:n_algorithms
        valid_errors = filter(!isnan, trial_errors[j, :])
        if !isempty(valid_errors)
            errors_mean[j, i] = mean(valid_errors)
            errors_std[j, i] = std(valid_errors)
        else
            errors_mean[j, i] = NaN
            errors_std[j, i] = NaN
        end
    end
end

println("\n✓ Experiments complete!")

# ============================================================================
# Print Results Table
# ============================================================================

println("\n" * "=" ^ 70)
println("Results Summary (Mean Relative Error %)")
println("=" ^ 70)

# Header
@printf("\n%6s |", "k")
for (name, _, _) in algorithms
    @printf(" %8s", name)
end
println()
println("-" ^ (8 + 9 * n_algorithms))

# Select subset of sparsity levels to display
display_indices = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
display_indices = filter(i -> i <= n_sparsities, display_indices)

for i in display_indices
    k = SPARSITY_RANGE[i]
    @printf("%6d |", k)
    for j in 1:n_algorithms
        if isnan(errors_mean[j, i])
            @printf(" %8s", "N/A")
        else
            @printf(" %7.1f%%", errors_mean[j, i] * 100)
        end
    end
    println()
end

# ============================================================================
# Generate Plots
# ============================================================================

println("\n" * "=" ^ 70)
println("Generating plots...")
println("=" ^ 70)

# Sparsity as percentage of signal dimension
sparsity_pct = collect(SPARSITY_RANGE) ./ SIGNAL_DIM .* 100

# Define colors for each algorithm (colorblind-friendly palette)
colors = [:royalblue, :crimson, :forestgreen, :darkorange, :purple, :brown]

# Theoretical limit annotation
n_over_p = N_MEASUREMENTS / SIGNAL_DIM
theoretical_limit_pct = n_over_p / (2 * log(SIGNAL_DIM / N_MEASUREMENTS)) * 100

# ============================================================================
# Plot 1: Linear Scale
# ============================================================================

p_linear = plot(
    title = "Sparse Recovery Error vs. Sparsity Level (Linear Scale)",
    xlabel = "Sparsity (% of signal dimension)",
    ylabel = "Relative Error",
    legend = :topleft,
    legendfontsize = 8,
    titlefontsize = 11,
    guidefontsize = 10,
    size = (800, 500),
    margin = 5Plots.mm,
    grid = true,
    gridstyle = :dash,
    gridalpha = 0.3,
    ylims = (0, 1.5),
    xlims = (0, maximum(sparsity_pct) + 1),
)

# Plot each algorithm
for (j, (name, _, _)) in enumerate(algorithms)
    valid_mask = .!isnan.(errors_mean[j, :])
    x_vals = sparsity_pct[valid_mask]
    y_vals = errors_mean[j, valid_mask]

    plot!(p_linear, x_vals, y_vals,
        label = name,
        color = colors[j],
        linewidth = 2,
        marker = :circle,
        markersize = 3,
        markerstrokewidth = 0,
    )
end

# Add reference lines
hline!(p_linear, [0.1], color=:gray, linestyle=:dash, linewidth=1, label="10% error")
hline!(p_linear, [0.5], color=:gray, linestyle=:dot, linewidth=1, label="50% error")
hline!(p_linear, [1.0], color=:black, linestyle=:solid, linewidth=1, alpha=0.3, label="100% error")

# Add theoretical limit annotation
if theoretical_limit_pct < maximum(sparsity_pct)
    vline!(p_linear, [theoretical_limit_pct], color=:red, linestyle=:dash,
           linewidth=1, alpha=0.5, label="")
    annotate!(p_linear, theoretical_limit_pct + 0.5, 1.3,
              text("≈ theoretical\nlimit", 7, :left, :red))
end

# Save linear scale plot
output_path_linear = joinpath(@__DIR__, "sparsity_sweep_linear.png")
savefig(p_linear, output_path_linear)
println("\n✓ Linear scale plot saved to: $output_path_linear")

# ============================================================================
# Plot 2: Logarithmic Scale
# ============================================================================

p_log = plot(
    title = "Sparse Recovery Error vs. Sparsity Level (Log Scale)",
    xlabel = "Sparsity (% of signal dimension)",
    ylabel = "Relative Error (log scale)",
    legend = :topleft,
    legendfontsize = 8,
    titlefontsize = 11,
    guidefontsize = 10,
    size = (800, 500),
    margin = 5Plots.mm,
    grid = true,
    gridstyle = :dash,
    gridalpha = 0.3,
    yscale = :log10,
    ylims = (1e-4, 10),
    xlims = (0, maximum(sparsity_pct) + 1),
)

# Plot each algorithm
for (j, (name, _, _)) in enumerate(algorithms)
    valid_mask = .!isnan.(errors_mean[j, :])
    x_vals = sparsity_pct[valid_mask]
    y_vals = errors_mean[j, valid_mask]

    plot!(p_log, x_vals, y_vals,
        label = name,
        color = colors[j],
        linewidth = 2,
        marker = :circle,
        markersize = 3,
        markerstrokewidth = 0,
    )
end

# Add reference lines
hline!(p_log, [0.001], color=:gray, linestyle=:dash, linewidth=1, label="0.1% error")
hline!(p_log, [0.01], color=:gray, linestyle=:dashdot, linewidth=1, label="1% error")
hline!(p_log, [0.1], color=:gray, linestyle=:dot, linewidth=1, label="10% error")

# Add theoretical limit annotation
if theoretical_limit_pct < maximum(sparsity_pct)
    vline!(p_log, [theoretical_limit_pct], color=:red, linestyle=:dash,
           linewidth=1, alpha=0.5, label="")
    annotate!(p_log, theoretical_limit_pct + 0.5, 5,
              text("≈ theoretical\nlimit", 7, :left, :red))
end

# Save log scale plot
output_path_log = joinpath(@__DIR__, "sparsity_sweep_log.png")
savefig(p_log, output_path_log)
println("✓ Log scale plot saved to: $output_path_log")

# Display both plots
display(p_linear)
display(p_log)

# ============================================================================
# Additional Analysis
# ============================================================================

println("\n" * "=" ^ 70)
println("Analysis")
println("=" ^ 70)

# Find the sparsity level where each algorithm's error exceeds 10%
println("\nSparsity level (k) where error exceeds 10%:")
for (j, (name, _, _)) in enumerate(algorithms)
    threshold_idx = findfirst(x -> x > 0.1, errors_mean[j, :])
    if threshold_idx === nothing
        println("  $name: > $(last(SPARSITY_RANGE)) (never exceeded)")
    else
        k_threshold = SPARSITY_RANGE[threshold_idx]
        pct = round(k_threshold / SIGNAL_DIM * 100, digits=1)
        println("  $name: k = $k_threshold ($pct% sparsity)")
    end
end

# Find best algorithm at different sparsity levels
println("\nBest algorithm at each sparsity level:")
checkpoints = [5, 10, 20, 30, 40, 50]
for k in filter(x -> x <= last(SPARSITY_RANGE), checkpoints)
    i = k - first(SPARSITY_RANGE) + 1
    errors_at_k = errors_mean[:, i]
    # Find minimum ignoring NaN values
    valid_mask = .!isnan.(errors_at_k)
    if any(valid_mask)
        valid_errors = errors_at_k[valid_mask]
        valid_names = [algorithms[j][1] for j in 1:n_algorithms if valid_mask[j]]
        best_idx = argmin(valid_errors)
        best_name = valid_names[best_idx]
        best_error = valid_errors[best_idx]
        @printf("  k = %2d: %s (%.2f%% error)\n", k, best_name, best_error * 100)
    else
        @printf("  k = %2d: No valid results\n", k)
    end
end

println("\n" * "=" ^ 70)
println("Benchmark complete!")
println("=" ^ 70)
