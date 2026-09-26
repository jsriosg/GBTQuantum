# Compress M Markov-chain configurations into unique states and integer multiplicities.
# UInt64 encoding is used when possible; a tuple fallback keeps the implementation general.
@inline function spin_key(x::AbstractVector{Int8})
    N = length(x)
    if N <= 64
        key = UInt64(0)
        @inbounds for i in 1:N
            x[i] > 0 && (key |= UInt64(1) << (i-1))
        end
        return key
    end
    return Tuple(x)
end

function compress_samples(samples::Matrix{Int8})
    M, N = size(samples)
    key_to_row = Dict{Any,Int}()
    states = Matrix{Int8}(undef, M, N)
    counts = Vector{Int}(undef, M)
    K = 0

    @inbounds for r in 1:M
        x = @view samples[r,:]
        key = spin_key(x)
        j = get(key_to_row, key, 0)
        if j == 0
            K += 1
            key_to_row[key] = K
            copyto!(@view(states[K,:]), x)
            counts[K] = 1
        else
            counts[j] += 1
        end
    end

    return states[1:K,:], counts[1:K]
end

struct VMCBatch
    states::Matrix{Int8}
    counts::Vector{Int}
    local_energy::Vector{ComplexF64}
    energy::ComplexF64
    variance::Float64
end

function vmc_batch(H::TFIMHamiltonian,
                   m::LogGBState,
                   samples::Matrix{Int8})
    states, counts = compress_samples(samples)
    K = size(states,1)
    eloc = Vector{ComplexF64}(undef, K)
    M = sum(counts)

    Esum = 0.0 + 0.0im
    @inbounds for j in 1:K
        e = local_energy!(H,m,@view states[j,:])
        eloc[j] = e
        Esum += counts[j] * e
    end
    E = Esum / M

    # For Hermitian H, Im(E) should converge statistically to zero.
    # Use Re(E) as the physical energy and retain Im(E) as a diagnostic.
    Ephys = real(E)
    varsum = 0.0
    @inbounds @simd for j in 1:K
        varsum += counts[j] * abs2(eloc[j] - Ephys)
    end
    variance = varsum / M

    return VMCBatch(states, counts, eloc, E, variance)
end

function make_targets(batch::VMCBatch)
    K = length(batch.counts)
    yA = Vector{Float64}(undef, K)
    yΦ = Vector{Float64}(undef, K)

    # Center by the empirical complex mean. For exact Hermitian expectation Im(E)=0;
    # empirical centering removes finite-sample global phase drift.
    E = batch.energy
    @inbounds @simd for j in 1:K
        Δ = batch.local_energy[j] - E
        yA[j] = -real(Δ)
        yΦ[j] = -imag(Δ)
    end
    return yA, yΦ
end

@inline function weighted_mean(y::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    s = 0.0
    W = 0.0
    @inbounds @simd for i in eachindex(y,w)
        wi = Float64(w[i])
        W += wi
        s += wi * Float64(y[i])
    end
    return s/W
end

function weighted_mse(tree::RegressionTree,
                      X::AbstractMatrix{<:Real},
                      y::AbstractVector{<:Real},
                      w::AbstractVector{<:Real})
    s = 0.0
    W = 0.0
    @inbounds for i in axes(X,1)
        wi = Float64(w[i])
        e = predict(tree,@view X[i,:]) - Float64(y[i])
        s += wi * e*e
        W += wi
    end
    return s/W
end

"""
    integrated_autocorrelation_time(x; maxlag=nothing)

Estimate the integrated autocorrelation time

    τ_int = 1/2 + Σₖ ρ(k)

using the initial-positive-sequence idea: accumulation stops when
the estimated autocorrelation becomes non-positive.

For independent samples τ_int ≈ 0.5, giving ESS ≈ length(x).
"""
function integrated_autocorrelation_time(
    x::AbstractVector{<:Real};
    maxlag::Union{Nothing,Int}=nothing,
)
    n = length(x)

    n < 2 && return 0.5

    μ = sum(x) / n

    # Population variance for autocorrelation normalization
    c0 = 0.0
    @inbounds @simd for i in eachindex(x)
        δ = x[i] - μ
        c0 += δ * δ
    end
    c0 /= n

    # Constant series
    c0 <= eps(Float64) && return 0.5

    K = isnothing(maxlag) ? min(n ÷ 2, 1000) : min(maxlag, n - 1)

    τ = 0.5

    @inbounds for lag in 1:K
        c = 0.0

        @simd for i in 1:(n - lag)
            c += (x[i] - μ) * (x[i + lag] - μ)
        end

        c /= (n - lag)

        ρ = c / c0

        # Simple positive-sequence truncation.
        if ρ <= 0.0
            break
        end

        τ += ρ
    end

    return max(τ, 0.5)
end


"""
    effective_sample_size(x)

Effective number of independent samples represented by an
autocorrelated scalar series.
"""
function effective_sample_size(x::AbstractVector{<:Real})
    n = length(x)

    n == 0 && return 0.0

    τ = integrated_autocorrelation_time(x)

    return min(Float64(n), n / (2τ))
end


"""
    energy_standard_error(local_energies)

Monte Carlo standard error of the mean local energy, corrected
using the estimated effective sample size.
"""
function energy_standard_error(local_energies::AbstractVector{<:Real})
    n = length(local_energies)

    n < 2 && return 0.0

    μ = sum(local_energies) / n

    var = 0.0
    @inbounds @simd for i in eachindex(local_energies)
        δ = local_energies[i] - μ
        var += δ * δ
    end

    # Sample variance
    var /= (n - 1)

    neff = effective_sample_size(local_energies)

    return sqrt(var / neff)
end

"""
    validation_vmc(model, H;
                   nsamples=2048,
                   burn_in_sweeps=100,
                   thinning_sweeps=1,
                   seed=12345)

Run an independent sequential Metropolis chain for the current model
and estimate:

    E
    Var(E_loc)
    τ_int
    ESS
    SE(E)

The samples generated here are used only for validation and never
for tree fitting.
"""
function validation_vmc(
    model::LogGBState,
    H::TFIMHamiltonian;
    nsamples::Int = 2048,
    burn_in_sweeps::Int = 100,
    thinning_sweeps::Int = 1,
    seed::Int = 12345,
)
    rng = MersenneTwister(seed)

    N = H.N

    # One independent sequential Markov chain
    state = Vector{Int8}(undef, N)

    @inbounds for j in 1:N
        state[j] = rand(rng, Bool) ? Int8(1) : Int8(-1)
    end

    currentA = logamplitude(model, state)

    # ---------------------------------------------------------
    # Lazy Metropolis transition
    #
    # With probability 1/2 remain in the current configuration.
    # Otherwise attempt one ordinary Metropolis spin flip.
    #
    # This removes the parity/periodicity problem that occurs
    # when measurements are made only after exactly N accepted
    # moves in the uniform-wavefunction limit.
    # ---------------------------------------------------------

    function lazy_step!(currentA)
        if rand(rng) < 0.5
            return currentA
        end

        _, newA = metropolis_step!(
            rng,
            model,
            state,
            currentA,
        )

        return newA
    end

    # Burn-in.
    #
    # Preserve approximately the old interpretation:
    # one "sweep" corresponds to N transition opportunities.
    for _ in 1:burn_in_sweeps
        for _ in 1:N
            currentA = lazy_step!(currentA)
        end
    end

    Eloc_series = Vector{Float64}(undef, nsamples)

    # ---------------------------------------------------------
    # Measurements
    #
    # thinning_sweeps = 1 means N lazy transition opportunities
    # between consecutive measurements.
    # ---------------------------------------------------------

    for k in 1:nsamples

        for _ in 1:thinning_sweeps
            for _ in 1:N
                currentA = lazy_step!(currentA)
            end
        end

        Eloc_series[k] =
            real(local_energy!(H, model, state))
    end

    E = sum(Eloc_series) / nsamples

    variance = 0.0

    @inbounds @simd for k in eachindex(Eloc_series)
        δ = Eloc_series[k] - E
        variance += δ * δ
    end

    variance /= nsamples

    τ = integrated_autocorrelation_time(Eloc_series)
    neff = effective_sample_size(Eloc_series)
    se = energy_standard_error(Eloc_series)

    return (
        energy = E,
        variance = variance,
        tau_int = τ,
        ess = neff,
        standard_error = se,
        samples = Eloc_series,
    )
end

"""
    validation_observables(model, H;
                           nsamples=2048,
                           burn_in_sweeps=100,
                           thinning_sweeps=1,
                           seed=54321)

Run an independent sequential Metropolis chain and estimate

    <m_z>
    <|m_z|>
    <m_z^2>
    <m_x>

together with autocorrelation and effective-sample-size diagnostics
for the diagonal observables.

The chain is independent from the samples used during training.
"""
function validation_observables(
    model::LogGBState,
    H::TFIMHamiltonian;
    nsamples::Int = 2048,
    burn_in_sweeps::Int = 100,
    thinning_sweeps::Int = 1,
    seed::Int = 54321,
)
    nsamples > 0 ||
        throw(ArgumentError("nsamples must be positive"))

    burn_in_sweeps >= 0 ||
        throw(ArgumentError("burn_in_sweeps must be non-negative"))

    thinning_sweeps >= 1 ||
        throw(ArgumentError("thinning_sweeps must be at least 1"))

    rng = MersenneTwister(seed)

    N = H.N

    # ---------------------------------------------------------
    # Initial state
    # ---------------------------------------------------------

    state = Vector{Int8}(undef, N)

    @inbounds for j in 1:N
        state[j] =
            rand(rng, Bool) ? Int8(1) : Int8(-1)
    end

    currentA = logamplitude(model, state)

    # ---------------------------------------------------------
    # Lazy Metropolis transition
    #
    # Same transition used by validation_vmc.
    # ---------------------------------------------------------

    function lazy_step!(currentA)
        if rand(rng) < 0.5
            return currentA
        end

        _, newA = metropolis_step!(
            rng,
            model,
            state,
            currentA,
        )

        return newA
    end

    # ---------------------------------------------------------
    # Burn-in
    # ---------------------------------------------------------

    for _ in 1:burn_in_sweeps
        for _ in 1:N
            currentA = lazy_step!(currentA)
        end
    end

    # ---------------------------------------------------------
    # Observable time series
    # ---------------------------------------------------------

    mz_series     = Vector{Float64}(undef, nsamples)
    abs_mz_series = Vector{Float64}(undef, nsamples)
    mz2_series    = Vector{Float64}(undef, nsamples)
    mx_series     = Vector{ComplexF64}(undef, nsamples)

    # ---------------------------------------------------------
    # Measurements
    # ---------------------------------------------------------

    @inbounds for k in 1:nsamples

        for _ in 1:thinning_sweeps
            for _ in 1:N
                currentA = lazy_step!(currentA)
            end
        end

        mz = magnetization_z(state)

        mz_series[k]     = mz
        abs_mz_series[k] = abs(mz)
        mz2_series[k]    = mz * mz

        mx_series[k] =
            local_magnetization_x(model, state)
    end

    # ---------------------------------------------------------
    # Means
    # ---------------------------------------------------------

    mz = sum(mz_series) / nsamples
    abs_mz = sum(abs_mz_series) / nsamples
    mz2 = sum(mz2_series) / nsamples
    mx = sum(mx_series) / nsamples

    # ---------------------------------------------------------
    # Autocorrelation diagnostics
    # ---------------------------------------------------------

    tau_mz =
        integrated_autocorrelation_time(mz_series)

    tau_abs_mz =
        integrated_autocorrelation_time(abs_mz_series)

    tau_mz2 =
        integrated_autocorrelation_time(mz2_series)

    ess_mz =
        effective_sample_size(mz_series)

    ess_abs_mz =
        effective_sample_size(abs_mz_series)

    ess_mz2 =
        effective_sample_size(mz2_series)

    # ---------------------------------------------------------
    # Standard errors
    #
    # energy_standard_error is mathematically just an
    # autocorrelation-corrected SE of a real scalar series,
    # so it can also be used here.
    # ---------------------------------------------------------

    se_mz =
        energy_standard_error(mz_series)

    se_abs_mz =
        energy_standard_error(abs_mz_series)

    se_mz2 =
        energy_standard_error(mz2_series)

    return (
        mz = mz,
        abs_mz = abs_mz,
        mz2 = mz2,
        mx = mx,

        tau_mz = tau_mz,
        tau_abs_mz = tau_abs_mz,
        tau_mz2 = tau_mz2,

        ess_mz = ess_mz,
        ess_abs_mz = ess_abs_mz,
        ess_mz2 = ess_mz2,

        se_mz = se_mz,
        se_abs_mz = se_abs_mz,
        se_mz2 = se_mz2,

        mz_samples = mz_series,
        abs_mz_samples = abs_mz_series,
        mz2_samples = mz2_series,
        mx_samples = mx_series,
    )
end

"""
    validation_distribution(model, H;
                            nsamples=100_000,
                            burn_in_sweeps=500,
                            thinning_sweeps=1,
                            seed=13579)

For small systems, compare the exact probability distribution represented
by `model`,

    p(x) = |ψ(x)|² / Σₓ |ψ(x)|²,

against the empirical distribution produced by an independent sequential
Metropolis chain.

Returns state-resolved probabilities, magnetization-resolved probabilities,
and distribution-distance diagnostics.
"""
function validation_distribution(
    model::LogGBState,
    H::TFIMHamiltonian;
    nsamples::Int = 100_000,
    burn_in_sweeps::Int = 500,
    thinning_sweeps::Int = 1,
    seed::Int = 13579,
)
    nsamples > 0 ||
        throw(ArgumentError("nsamples must be positive"))

    burn_in_sweeps >= 0 ||
        throw(ArgumentError("burn_in_sweeps must be non-negative"))

    thinning_sweeps >= 1 ||
        throw(ArgumentError("thinning_sweeps must be at least 1"))

    N = H.N

    # This is explicitly a small-system diagnostic.
    N <= 20 ||
        throw(ArgumentError(
            "validation_distribution is restricted to N <= 20"
        ))

    nstates = 1 << N

    # =========================================================
    # Exact GBT probability distribution
    # =========================================================

    logweights = Vector{Float64}(undef, nstates)
    state = Vector{Int8}(undef, N)

    @inbounds for s in 0:(nstates - 1)

        for j in 1:N
            state[j] =
                ((s >> (j - 1)) & 1) == 1 ?
                Int8(1) : Int8(-1)
        end

        logweights[s + 1] =
            2.0 * logamplitude(model, state)
    end

    maxlogweight = maximum(logweights)

    exact_prob = Vector{Float64}(undef, nstates)

    Z = 0.0

    @inbounds @simd for i in eachindex(logweights)
        p = exp(logweights[i] - maxlogweight)
        exact_prob[i] = p
        Z += p
    end

    exact_prob ./= Z

    # =========================================================
    # Independent Markov chain
    # =========================================================

    rng = MersenneTwister(seed)

    @inbounds for j in 1:N
        state[j] =
            rand(rng, Bool) ? Int8(1) : Int8(-1)
    end

    currentA = logamplitude(model, state)

    function lazy_step!(currentA)

        if rand(rng) < 0.5
            return currentA
        end

        _, newA = metropolis_step!(
            rng,
            model,
            state,
            currentA,
        )

        return newA
    end

    # Burn-in
    for _ in 1:burn_in_sweeps
        for _ in 1:N
            currentA = lazy_step!(currentA)
        end
    end

    counts = zeros(Int, nstates)

    # =========================================================
    # Sampling
    # =========================================================

    @inbounds for _ in 1:nsamples

        for _ in 1:thinning_sweeps
            for _ in 1:N
                currentA = lazy_step!(currentA)
            end
        end

        # Convert {-1,+1} spin configuration to binary index.
        key = 0

        for j in 1:N
            if state[j] > 0
                key |= 1 << (j - 1)
            end
        end

        counts[key + 1] += 1
    end

    sampled_prob = counts ./ nsamples

    # =========================================================
    # State-resolved distances
    # =========================================================

    total_variation =
        0.5 * sum(abs.(sampled_prob .- exact_prob))

    l1_distance =
        sum(abs.(sampled_prob .- exact_prob))

    max_abs_error =
        maximum(abs.(sampled_prob .- exact_prob))

    # =========================================================
    # Magnetization-resolved distribution
    #
    # For N spins:
    #
    # n_up = 0,...,N
    # mz   = (2*n_up - N)/N
    #
    # Therefore there are N+1 magnetization sectors.
    # =========================================================

    exact_mag_prob = zeros(Float64, N + 1)
    sampled_mag_prob = zeros(Float64, N + 1)

    @inbounds for s in 0:(nstates - 1)

        # Number of +1 spins.
        nup = count_ones(UInt(s))

        exact_mag_prob[nup + 1] +=
            exact_prob[s + 1]

        sampled_mag_prob[nup + 1] +=
            sampled_prob[s + 1]
    end

    magnetizations =
        [(2nup - N) / N for nup in 0:N]

    magnetization_tv =
        0.5 *
        sum(abs.(sampled_mag_prob .- exact_mag_prob))

    return (
        exact_prob = exact_prob,
        sampled_prob = sampled_prob,
        counts = counts,

        total_variation = total_variation,
        l1_distance = l1_distance,
        max_abs_error = max_abs_error,

        magnetizations = magnetizations,
        exact_magnetization_prob = exact_mag_prob,
        sampled_magnetization_prob = sampled_mag_prob,
        magnetization_tv = magnetization_tv,

        nsamples = nsamples,
        nstates = nstates,
    )
end

"""
    exact_model_energy(model, H)

Compute the exact Rayleigh quotient of `model` by enumerating the complete
spin Hilbert space.

This is intended only as a validation tool for small systems, since the
number of configurations grows as 2^N.

Returns a named tuple containing:
- `energy`: exact variational energy of the model
- `variance`: exact local-energy variance
- `nstates`: number of enumerated basis states
"""
function exact_model_energy(
    model::LogGBState,
    H::TFIMHamiltonian,
)
    N = H.N

    # UInt64 enumeration below requires N < 64.
    # In practice this function becomes exponentially expensive much earlier.
    N < 64 || throw(
        ArgumentError("exact_model_energy requires N < 64")
    )

    nstates = 1 << N

    states = Matrix{Int8}(undef, nstates, N)
    logweights = Vector{Float64}(undef, nstates)
    local_energies = Vector{ComplexF64}(undef, nstates)

    # ---------------------------------------------------------
    # Enumerate every spin configuration.
    #
    # bit = 0 -> spin -1
    # bit = 1 -> spin +1
    # ---------------------------------------------------------

    @inbounds for k in 0:(nstates - 1)

        row = k + 1

        for j in 1:N
            bit = (k >> (j - 1)) & 1
            states[row, j] = bit == 1 ? Int8(1) : Int8(-1)
        end

        x = @view states[row, :]

        # |ψ(x)|² = exp(2A(x))
        logweights[row] = 2.0 * logamplitude(model, x)

        local_energies[row] =
            local_energy!(H, model, x)
    end

    # ---------------------------------------------------------
    # Stable normalization.
    #
    # Directly calculating exp(2A) can overflow or underflow.
    # Subtracting max(logweight) leaves normalized probabilities
    # unchanged.
    # ---------------------------------------------------------

    maxlogweight = maximum(logweights)

    weights = Vector{Float64}(undef, nstates)

    Z = 0.0

    @inbounds @simd for k in 1:nstates
        w = exp(logweights[k] - maxlogweight)
        weights[k] = w
        Z += w
    end

    # ---------------------------------------------------------
    # Exact variational energy
    #
    # E = Σ p(x) E_loc(x)
    # ---------------------------------------------------------

    E = 0.0 + 0.0im

    @inbounds for k in 1:nstates
        E += weights[k] * local_energies[k]
    end

    E /= Z

    # ---------------------------------------------------------
    # Exact local-energy variance
    #
    # Var(E_loc) = < |E_loc - E|² >
    # ---------------------------------------------------------

    variance = 0.0

    @inbounds for k in 1:nstates
        δ = local_energies[k] - E
        variance += weights[k] * abs2(δ)
    end

    variance /= Z

    return (
        energy = E,
        variance = variance,
        nstates = nstates,
    )
end