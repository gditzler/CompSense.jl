### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ 8a7b3c10-b234-11ee-1234-0123456789ab
begin
    using Pkg
    Pkg.activate(Base.current_project())
    using CompSense
    using LinearAlgebra
    using Random
    using Statistics
end

# ╔═╡ 1a2b3c4d-5678-90ab-cdef-1234567890ab
md"""
# CompSense.jl Algorithm Comparison

This interactive notebook compares all sparse recovery algorithms in CompSense.jl.

**Algorithms covered:**
- **Greedy**: OMP, CoSaMP
- **Thresholding**: IHT, FISTA
- **Smoothing**: SL0, L0EM
- **Convex**: IRWLS

Use the sliders below to explore how different problem parameters affect recovery performance.
"""

# ╔═╡ 2a3b4c5d-6789-01ab-cdef-234567890123
md"## 1. Problem Setup"

# ╔═╡ 3a4b5c6d-7890-12ab-cdef-345678901234
md"""
**Number of measurements (m):** $(@bind n_measurements html"<input type='range' min='30' max='100' value='50' step='10'>")

**Signal dimension (n):** $(@bind signal_dim html"<input type='range' min='100' max='500' value='200' step='50'>")

**Sparsity (k):** $(@bind sparsity html"<input type='range' min='5' max='30' value='10' step='5'>")

**Random seed:** $(@bind seed html"<input type='range' min='1' max='100' value='42' step='1'>")
"""

# ╔═╡ 4a5b6c7d-8901-23ab-cdef-456789012345
begin
    Random.seed!(seed)
    A, x_true, b = gaussian_sensing(n_measurements, signal_dim, sparsity)
    md"""
    **Generated problem:**
    - Sensing matrix A: $(n_measurements) × $(signal_dim)
    - True signal: $(sparsity) non-zeros
    - Compression ratio: $(round(n_measurements/signal_dim * 100, digits=1))%
    """
end

# ╔═╡ 5a6b7c8d-9012-34ab-cdef-567890123456
md"## 2. Run All Algorithms"

# ╔═╡ 6a7b8c9d-0123-45ab-cdef-678901234567
begin
    # Run all algorithms
    results = Dict{String, NamedTuple}()

    # OMP
    t_omp = @elapsed x_omp = OMP(A, b; sparsity=sparsity)
    results["OMP"] = (
        x = x_omp,
        error = norm(x_omp - x_true) / norm(x_true),
        nnz = count(!iszero, x_omp),
        time = t_omp
    )

    # CoSaMP
    t_cosamp = @elapsed x_cosamp = CoSaMP(A, b; sparsity=sparsity)
    results["CoSaMP"] = (
        x = x_cosamp,
        error = norm(x_cosamp - x_true) / norm(x_true),
        nnz = count(!iszero, x_cosamp),
        time = t_cosamp
    )

    # IHT
    t_iht = @elapsed x_iht = IHT(A, b; sparsity=sparsity)
    results["IHT"] = (
        x = x_iht,
        error = norm(x_iht - x_true) / norm(x_true),
        nnz = count(!iszero, x_iht),
        time = t_iht
    )

    # FISTA
    t_fista = @elapsed x_fista = FISTA(A, b; lambda=0.1)
    results["FISTA"] = (
        x = x_fista,
        error = norm(x_fista - x_true) / norm(x_true),
        nnz = count(xi -> abs(xi) > 0.01, x_fista),
        time = t_fista
    )

    # SL0
    t_sl0 = @elapsed x_sl0 = SL0(A, b)
    results["SL0"] = (
        x = x_sl0,
        error = norm(x_sl0 - x_true) / norm(x_true),
        nnz = count(xi -> abs(xi) > 0.001, x_sl0),
        time = t_sl0
    )

    # L0EM
    t_l0em = @elapsed x_l0em = L0EM(A, b)
    results["L0EM"] = (
        x = x_l0em,
        error = norm(x_l0em - x_true) / norm(x_true),
        nnz = count(xi -> abs(xi) > 0.001, x_l0em),
        time = t_l0em
    )

    # IRWLS (fewer iterations for speed)
    t_irwls = @elapsed x_irwls = IRWLS(A, b; maxiter=3)
    results["IRWLS"] = (
        x = x_irwls,
        error = norm(x_irwls - x_true) / norm(x_true),
        nnz = count(xi -> abs(xi) > 0.01, x_irwls),
        time = t_irwls
    )

    nothing
end

# ╔═╡ 7a8b9c0d-1234-56ab-cdef-789012345678
md"## 3. Results Summary"

# ╔═╡ 8a9b0c1d-2345-67ab-cdef-890123456789
begin
    # Sort by error
    sorted_algs = sort(collect(keys(results)), by=k -> results[k].error)

    md"""
    ### Recovery Performance

    | Rank | Algorithm | Relative Error | Non-zeros | Time (ms) |
    |:----:|:----------|---------------:|----------:|----------:|
    $(join([
        "| $(i) | **$(alg)** | $(round(results[alg].error * 100, digits=2))% | $(results[alg].nnz) | $(round(results[alg].time * 1000, digits=2)) |"
        for (i, alg) in enumerate(sorted_algs)
    ], "\n"))

    **Best algorithm:** $(sorted_algs[1]) with $(round(results[sorted_algs[1]].error * 100, digits=2))% error
    """
end

# ╔═╡ 9a0b1c2d-3456-78ab-cdef-901234567890
md"## 4. Algorithm Categories"

# ╔═╡ 0a1b2c3d-4567-89ab-cdef-012345678901
md"""
### Greedy Pursuit (OMP, CoSaMP)
- Select atoms iteratively based on correlation with residual
- OMP: one atom per iteration
- CoSaMP: multiple atoms with pruning

### Hard Thresholding (IHT)
- Gradient descent + keep k largest entries
- Exact sparsity constraint

### Soft Thresholding (FISTA)
- Proximal gradient for LASSO
- Approximate sparsity via L1 regularization

### Smoothed Optimization (SL0, L0EM)
- Approximate L0 with smooth surrogates
- Gradient-based optimization

### Convex Relaxation (IRWLS)
- Reweighted L1 minimization
- Uses convex solver (slower but principled)
"""

# ╔═╡ 1b2c3d4e-5678-90ab-cdef-123456789012
md"## 5. Support Recovery Analysis"

# ╔═╡ 2c3d4e5f-6789-01ab-cdef-234567890123
begin
    true_support = Set(findall(!iszero, x_true))

    support_analysis = Dict{String, NamedTuple}()
    for (alg, res) in results
        threshold = alg in ["FISTA", "IRWLS"] ? 0.01 : (alg in ["SL0", "L0EM"] ? 0.001 : 0.0)
        rec_support = Set(findall(xi -> abs(xi) > threshold, res.x))
        correct = length(intersect(true_support, rec_support))
        missed = length(setdiff(true_support, rec_support))
        false_pos = length(setdiff(rec_support, true_support))
        support_analysis[alg] = (correct=correct, missed=missed, false_pos=false_pos)
    end

    md"""
    ### Support Recovery (True support size: $(length(true_support)))

    | Algorithm | Correct | Missed | False Positives |
    |:----------|--------:|-------:|----------------:|
    $(join([
        "| $(alg) | $(support_analysis[alg].correct) | $(support_analysis[alg].missed) | $(support_analysis[alg].false_pos) |"
        for alg in sorted_algs
    ], "\n"))
    """
end

# ╔═╡ 3d4e5f6a-7890-12ab-cdef-345678901234
md"## 6. Noise Sensitivity Analysis"

# ╔═╡ 4e5f6a7b-8901-23ab-cdef-456789012345
md"""
**Noise level (σ):** $(@bind noise_level html"<input type='range' min='0' max='0.2' value='0.05' step='0.01'>")
"""

# ╔═╡ 5f6a7b8c-9012-34ab-cdef-567890123456
begin
    Random.seed!(seed)
    A_noise, x_noise, b_clean = gaussian_sensing(n_measurements, signal_dim, sparsity)
    b_noisy = b_clean + noise_level * randn(length(b_clean))

    noise_results = Dict{String, Float64}()
    noise_results["OMP"] = norm(OMP(A_noise, b_noisy; sparsity=sparsity) - x_noise) / norm(x_noise)
    noise_results["CoSaMP"] = norm(CoSaMP(A_noise, b_noisy; sparsity=sparsity) - x_noise) / norm(x_noise)
    noise_results["IHT"] = norm(IHT(A_noise, b_noisy; sparsity=sparsity) - x_noise) / norm(x_noise)
    noise_results["FISTA"] = norm(FISTA(A_noise, b_noisy; lambda=max(0.1, noise_level*2)) - x_noise) / norm(x_noise)
    noise_results["SL0"] = norm(SL0(A_noise, b_noisy) - x_noise) / norm(x_noise)
    noise_results["L0EM"] = norm(L0EM(A_noise, b_noisy) - x_noise) / norm(x_noise)

    sorted_noise = sort(collect(keys(noise_results)), by=k -> noise_results[k])

    md"""
    ### Recovery with Noise Level σ = $(noise_level)

    | Algorithm | Relative Error |
    |:----------|---------------:|
    $(join([
        "| $(alg) | $(round(noise_results[alg] * 100, digits=2))% |"
        for alg in sorted_noise
    ], "\n"))

    *(IRWLS omitted for speed)*
    """
end

# ╔═╡ 6a7b8c9d-0123-45ab-cdef-678901234567
md"## 7. Measurement Sweep"

# ╔═╡ 7b8c9d0e-1234-56ab-cdef-789012345678
begin
    measurement_range = 30:10:100
    sweep_errors = Dict{String, Vector{Float64}}()

    for alg in ["OMP", "IHT", "SL0", "FISTA"]
        sweep_errors[alg] = Float64[]
    end

    Random.seed!(seed)
    for m in measurement_range
        A_sweep, x_sweep, b_sweep = gaussian_sensing(m, signal_dim, sparsity)
        push!(sweep_errors["OMP"], norm(OMP(A_sweep, b_sweep; sparsity=sparsity) - x_sweep) / norm(x_sweep))
        push!(sweep_errors["IHT"], norm(IHT(A_sweep, b_sweep; sparsity=sparsity) - x_sweep) / norm(x_sweep))
        push!(sweep_errors["SL0"], norm(SL0(A_sweep, b_sweep) - x_sweep) / norm(x_sweep))
        push!(sweep_errors["FISTA"], norm(FISTA(A_sweep, b_sweep; lambda=0.1) - x_sweep) / norm(x_sweep))
    end

    md"""
    ### Error vs Number of Measurements (n=$(signal_dim), k=$(sparsity))

    | m | OMP | IHT | SL0 | FISTA |
    |--:|----:|----:|----:|------:|
    $(join([
        "| $(m) | $(round(sweep_errors["OMP"][i]*100, digits=1))% | $(round(sweep_errors["IHT"][i]*100, digits=1))% | $(round(sweep_errors["SL0"][i]*100, digits=1))% | $(round(sweep_errors["FISTA"][i]*100, digits=1))% |"
        for (i, m) in enumerate(measurement_range)
    ], "\n"))

    **Observation:** More measurements → better recovery (phase transition around m ≈ 2k log(n/k))
    """
end

# ╔═╡ 8c9d0e1f-2345-67ab-cdef-890123456789
md"## 8. Algorithm Selection Guide"

# ╔═╡ 9d0e1f2a-3456-78ab-cdef-901234567890
md"""
### When to Use Each Algorithm

| Algorithm | Best For | Avoid When |
|:----------|:---------|:-----------|
| **OMP** | Known sparsity, fast baseline | Very large problems |
| **CoSaMP** | Theoretical guarantees needed | Memory constrained |
| **IHT** | Exact sparsity, simple implementation | Poorly conditioned A |
| **FISTA** | Noisy measurements, LASSO problems | Exact sparsity required |
| **SL0** | Fast approximate recovery | Need exact sparsity |
| **L0EM** | Balance of speed and accuracy | Very sparse signals |
| **IRWLS** | High-quality solutions | Large problems, speed critical |

### Speed Ranking (typically)
1. 🥇 SL0, IHT (fastest)
2. 🥈 L0EM, FISTA
3. 🥉 OMP, CoSaMP
4. 🐢 IRWLS (slowest - uses convex solver)
"""

# ╔═╡ 0e1f2a3b-4567-89ab-cdef-012345678901
md"""
---
## Summary

This notebook demonstrated CompSense.jl's sparse recovery algorithms on a synthetic compressed sensing problem.

**Key findings for this problem (m=$(n_measurements), n=$(signal_dim), k=$(sparsity)):**
- Best accuracy: **$(sorted_algs[1])**
- Fastest: depends on problem size and algorithm parameters

Try adjusting the sliders above to explore how algorithms perform under different conditions!
"""

# ╔═╡ Cell order:
# ╟─8a7b3c10-b234-11ee-1234-0123456789ab
# ╟─1a2b3c4d-5678-90ab-cdef-1234567890ab
# ╟─2a3b4c5d-6789-01ab-cdef-234567890123
# ╟─3a4b5c6d-7890-12ab-cdef-345678901234
# ╟─4a5b6c7d-8901-23ab-cdef-456789012345
# ╟─5a6b7c8d-9012-34ab-cdef-567890123456
# ╟─6a7b8c9d-0123-45ab-cdef-678901234567
# ╟─7a8b9c0d-1234-56ab-cdef-789012345678
# ╟─8a9b0c1d-2345-67ab-cdef-890123456789
# ╟─9a0b1c2d-3456-78ab-cdef-901234567890
# ╟─0a1b2c3d-4567-89ab-cdef-012345678901
# ╟─1b2c3d4e-5678-90ab-cdef-123456789012
# ╟─2c3d4e5f-6789-01ab-cdef-234567890123
# ╟─3d4e5f6a-7890-12ab-cdef-345678901234
# ╟─4e5f6a7b-8901-23ab-cdef-456789012345
# ╟─5f6a7b8c-9012-34ab-cdef-567890123456
# ╟─6a7b8c9d-0123-45ab-cdef-678901234567
# ╟─7b8c9d0e-1234-56ab-cdef-789012345678
# ╟─8c9d0e1f-2345-67ab-cdef-890123456789
# ╟─9d0e1f2a-3456-78ab-cdef-901234567890
# ╟─0e1f2a3b-4567-89ab-cdef-012345678901
