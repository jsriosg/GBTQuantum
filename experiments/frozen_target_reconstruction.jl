using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Random
using Statistics
using Printf


# ============================================================
# FROZEN-TARGET RECONSTRUCTION EXPERIMENT
#
# Purpose
# -------
# Separate finite-sample regression error from the coupled
# evolution of the VMC wavefunction.
#
# Normal training is performed up to selected checkpoints.
# At each checkpoint the current model is frozen.
#
# From that frozen model we compute exactly over the complete
# Hilbert space:
#
#     p_t(x)
#     E_L,t(x)
#     E_t
#     y_t(x) = -Re[E_L,t(x) - E_t]
#
# Then, WITHOUT modifying the frozen model, we repeatedly draw
# IID samples from p_t(x), train fresh regression trees, and
# evaluate their reconstruction of the exact target.
#
# Two target constructions are compared:
#
#   VMC target:
#       yhat(x_i) = -Re[E_L(x_i) - Ehat]
#
#   Exact-energy control:
#       y*(x_i) = -Re[E_L(x_i) - E_exact]
#
# This isolates:
#
#   1. finite configuration-sampling error,
#   2. energy-estimation error,
#   3. weak-learner representation limits.
# ============================================================


# ============================================================
# Configuration
# ============================================================

const N = 12
const h = 1.0

# Start with the two extreme regimes.
const J_values = [
    0.05,
    2.00,
]

# Population used by the NORMAL training trajectory.
#
# Keep this fixed so that differences between checkpoints come
# from training time / physics rather than a different trajectory.
const training_nsamples = 256

const nepochs = 64

const checkpoint_epochs = Set([
    1,
    4,
    8,
    16,
    32,
    64,
])

# IID diagnostic sample sizes.
const diagnostic_sample_sizes = [
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
]

# Number of independent IID resamples at every frozen checkpoint.
const nresamples = 30

# Number of independent NORMAL training trajectories.
#
# Start with 3. This already gives:
#
#   2 J values
# × 3 trajectories
# × 6 checkpoints
# × 7 sample sizes
# × 30 resamples
# × 2 target controls
#
# = 15,120 diagnostic tree fits.
#
# Increase later for final statistics if necessary.
const ntraining_runs = 3

const max_depth = 4
const eta = 0.05

const burn_in_sweeps = 100
const sweeps_per_epoch = 1

const min_leaf_weight = 1.0
const min_gain = 0.0

const base_seed = 910_000


# ============================================================
# Hilbert-space enumeration
# ============================================================

function enumerate_states(N::Int)

    d = 1 << N

    states =
        Matrix{Int8}(undef, d, N)

    @inbounds for s in 0:(d - 1)

        for i in 1:N

            states[s + 1, i] =
                ((s >> (i - 1)) & 1) == 1 ?
                Int8(1) :
                Int8(-1)

        end
    end

    return states
end


# ============================================================
# Weighted statistics
# ============================================================

function weighted_mean_local(y, w)

    total = 0.0
    W = 0.0

    @inbounds for i in eachindex(y, w)

        wi = Float64(w[i])

        total += wi * Float64(y[i])
        W += wi

    end

    return total / W
end


function weighted_rms(y, w)

    total = 0.0
    W = 0.0

    @inbounds for i in eachindex(y, w)

        wi = Float64(w[i])

        total += wi * Float64(y[i])^2
        W += wi

    end

    return sqrt(total / W)
end


function weighted_r2(ytrue, ypred, w)

    μ =
        weighted_mean_local(
            ytrue,
            w,
        )

    ss_res = 0.0
    ss_tot = 0.0

    @inbounds for i in eachindex(ytrue, ypred, w)

        wi = Float64(w[i])

        δres =
            Float64(ytrue[i]) -
            Float64(ypred[i])

        δtot =
            Float64(ytrue[i]) -
            μ

        ss_res += wi * δres^2
        ss_tot += wi * δtot^2

    end

    ss_tot <= eps(Float64) &&
        return NaN

    return 1.0 - ss_res / ss_tot
end


function ordinary_r2(ytrue, ypred)

    μ = mean(ytrue)

    ss_res =
        sum(
            (ytrue .- ypred).^2
        )

    ss_tot =
        sum(
            (ytrue .- μ).^2
        )

    ss_tot <= eps(Float64) &&
        return NaN

    return 1.0 - ss_res / ss_tot
end


function probability_weighted_r2(
    ytrue,
    ypred,
    p,
)

    μ =
        sum(
            p .* ytrue
        )

    ss_res =
        sum(
            p .* (ytrue .- ypred).^2
        )

    ss_tot =
        sum(
            p .* (ytrue .- μ).^2
        )

    ss_tot <= eps(Float64) &&
        return NaN

    return 1.0 - ss_res / ss_tot
end


function safe_correlation(x, y)

    std(x) <= eps(Float64) &&
        return NaN

    std(y) <= eps(Float64) &&
        return NaN

    return cor(x, y)
end


function probability_weighted_correlation(
    x,
    y,
    p,
)

    μx =
        sum(
            p .* x
        )

    μy =
        sum(
            p .* y
        )

    dx =
        x .- μx

    dy =
        y .- μy

    varx =
        sum(
            p .* dx.^2
        )

    vary =
        sum(
            p .* dy.^2
        )

    if varx <= eps(Float64) ||
       vary <= eps(Float64)

        return NaN
    end

    covxy =
        sum(
            p .* dx .* dy
        )

    return covxy /
           sqrt(varx * vary)
end


# ============================================================
# Tree helpers
# ============================================================

function predict_all(
    tree::RegressionTree,
    X::AbstractMatrix{<:Real},
)

    y =
        Vector{Float64}(
            undef,
            size(X, 1),
        )

    @inbounds for i in axes(X, 1)

        y[i] =
            predict(
                tree,
                @view(X[i, :]),
            )

    end

    return y
end


function shift_tree_leaves(
    tree::RegressionTree,
    shift::Float64,
)

    nodes =
        copy(tree.nodes)

    @inbounds for i in eachindex(nodes)

        n = nodes[i]

        if n.isleaf

            nodes[i] =
                Node(
                    n.feature,
                    n.value - shift,
                    n.left,
                    n.right,
                    true,
                )

        end
    end

    return RegressionTree(nodes)
end


function scale_tree(
    tree::RegressionTree,
    scale::Float64,
)

    nodes =
        [
            n.isleaf ?
            Node(
                n.feature,
                scale * n.value,
                n.left,
                n.right,
                true,
            ) :
            n
            for n in tree.nodes
        ]

    return RegressionTree(nodes)
end


# ============================================================
# Exact frozen-model quantities
# ============================================================

function exact_probabilities(
    model::LogGBState,
    states::Matrix{Int8},
)

    d =
        size(states, 1)

    logweights =
        Vector{Float64}(undef, d)

    @inbounds for s in 1:d

        A =
            logamplitude(
                model,
                @view(states[s, :]),
            )

        logweights[s] =
            2.0 * A

    end

    # Stable normalization.
    m =
        maximum(logweights)

    weights =
        exp.(logweights .- m)

    return weights ./ sum(weights)
end


function exact_frozen_problem(
    H::TFIMHamiltonian,
    model::LogGBState,
    states::Matrix{Int8},
)

    p =
        exact_probabilities(
            model,
            states,
        )

    d =
        size(states, 1)

    eloc =
        Vector{ComplexF64}(undef, d)

    @inbounds for s in 1:d

        eloc[s] =
            local_energy!(
                H,
                model,
                @view(states[s, :]),
            )

    end

    E =
        sum(
            p .* eloc
        )

    target =
        Vector{Float64}(undef, d)

    @inbounds for s in 1:d

        target[s] =
            -real(
                eloc[s] - E
            )

    end

    target_mean =
        sum(
            p .* target
        )

    target_rms =
        sqrt(
            sum(
                p .* target.^2
            )
        )

    variance =
        sum(
            p .* abs2.(eloc .- E)
        )

    participation =
        1.0 /
        sum(abs2, p)

    return (
        probabilities = p,
        local_energy = eloc,
        energy = E,
        target = target,
        target_mean = target_mean,
        target_rms = target_rms,
        variance = variance,
        participation_ratio = participation,
        participation_fraction =
            participation / d,
    )
end


# ============================================================
# IID categorical sampling
# ============================================================

function cumulative_probabilities(
    p::Vector{Float64},
)

    cdf =
        cumsum(p)

    # Protect against floating-point normalization error.
    cdf[end] = 1.0

    return cdf
end


function iid_sample_indices!(
    rng::AbstractRNG,
    destination::Vector{Int},
    cdf::Vector{Float64},
)

    @inbounds for i in eachindex(destination)

        u =
            rand(rng)

        destination[i] =
            searchsortedfirst(
                cdf,
                u,
            )

    end

    return destination
end


# ============================================================
# Compress IID sample indices
#
# Returns unique Hilbert-state indices and multiplicities.
# ============================================================

function compress_indices(
    indices::Vector{Int},
)

    counts =
        Dict{Int, Int}()

    @inbounds for idx in indices

        counts[idx] =
            get(counts, idx, 0) + 1

    end

    unique_indices =
        sort!(
            collect(keys(counts))
        )

    weights =
        Vector{Int}(undef, length(unique_indices))

    @inbounds for i in eachindex(unique_indices)

        weights[i] =
            counts[unique_indices[i]]

    end

    return unique_indices, weights
end


# ============================================================
# Construct training matrix from Hilbert indices
# ============================================================

function states_from_indices(
    full_states::Matrix{Int8},
    indices::Vector{Int},
)

    X =
        Matrix{Int8}(
            undef,
            length(indices),
            size(full_states, 2),
        )

    @inbounds for i in eachindex(indices)

        X[i, :] .=
            @view full_states[indices[i], :]

    end

    return X
end


# ============================================================
# Sample probability-mass coverage
# ============================================================

function sample_probability_mass(
    unique_indices::Vector{Int},
    p::Vector{Float64},
)

    total = 0.0

    @inbounds for idx in unique_indices

        total += p[idx]

    end

    return total
end


# ============================================================
# Fit one diagnostic tree
# ============================================================

function fit_diagnostic_tree(
    X::Matrix{Int8},
    y::Vector{Float64},
    weights::Vector{Int},
)

    tree =
        GBTQuantum.grow_tree(
            X,
            y,
            weights;
            max_depth = max_depth,
            min_weight = min_leaf_weight,
            min_gain = min_gain,
        )

    prediction =
        predict_all(
            tree,
            X,
        )

    # Same gauge-centering procedure used by Optimizer.jl.
    μtree =
        weighted_mean_local(
            prediction,
            weights,
        )

    if μtree != 0.0

        tree =
            shift_tree_leaves(
                tree,
                μtree,
            )

        prediction =
            predict_all(
                tree,
                X,
            )

    end

    return tree, prediction
end


# ============================================================
# Evaluate one tree against the exact frozen problem
# ============================================================

function evaluate_tree(
    tree::RegressionTree,
    train_y::Vector{Float64},
    train_prediction::Vector{Float64},
    weights::Vector{Int},
    frozen,
    full_states::Matrix{Int8},
)

    full_prediction =
        predict_all(
            tree,
            full_states,
        )

    train_r2 =
        weighted_r2(
            train_y,
            train_prediction,
            weights,
        )

    probability_r2 =
        probability_weighted_r2(
            frozen.target,
            full_prediction,
            frozen.probabilities,
        )

    hilbert_r2 =
        ordinary_r2(
            frozen.target,
            full_prediction,
        )

    probability_corr =
        probability_weighted_correlation(
            frozen.target,
            full_prediction,
            frozen.probabilities,
        )

    hilbert_corr =
        safe_correlation(
            frozen.target,
            full_prediction,
        )

    return (
        train_r2 = train_r2,
        probability_r2 = probability_r2,
        hilbert_r2 = hilbert_r2,
        probability_corr = probability_corr,
        hilbert_corr = hilbert_corr,
    )
end


# ============================================================
# Frozen checkpoint resampling experiment
# ============================================================

function diagnose_frozen_checkpoint(
    rng::AbstractRNG,
    H::TFIMHamiltonian,
    model::LogGBState,
    full_states::Matrix{Int8},
    J::Float64,
    training_run::Int,
    epoch::Int,
)

    frozen =
        exact_frozen_problem(
            H,
            model,
            full_states,
        )

    p =
        frozen.probabilities

    cdf =
        cumulative_probabilities(p)

    rows =
        NamedTuple[]

    d =
        size(full_states, 1)

    @printf(
        "\n  Frozen epoch %2d: E=% .8f  target RMS=%.4e  PR/H=%.6f\n",
        epoch,
        real(frozen.energy),
        frozen.target_rms,
        frozen.participation_fraction,
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    abs(frozen.target_mean) > 1e-10 &&
        @warn(
            "Exact frozen target does not have zero p-weighted mean",
            epoch = epoch,
            target_mean = frozen.target_mean,
        )

    # For a real wavefunction this should agree closely:
    rms_variance_difference =
        abs(
            frozen.target_rms^2 -
            real(frozen.variance)
        )

    if rms_variance_difference > 1e-8

        @warn(
            "target RMS² and local-energy variance differ",
            epoch = epoch,
            difference = rms_variance_difference,
        )

    end

    # ========================================================
    # Diagnostic sample-size loop
    # ========================================================

    for M in diagnostic_sample_sizes

        probability_r2_vmc =
            Float64[]

        probability_r2_exactE =
            Float64[]

        for resample in 1:nresamples

            # ------------------------------------------------
            # IID sample from frozen exact p_t
            # ------------------------------------------------

            indices =
                Vector{Int}(undef, M)

            iid_sample_indices!(
                rng,
                indices,
                cdf,
            )

            unique_indices,
            weights =
                compress_indices(indices)

            X =
                states_from_indices(
                    full_states,
                    unique_indices,
                )

            nunique =
                length(unique_indices)

            unique_fraction =
                nunique / M

            hilbert_fraction =
                nunique / d

            probability_mass =
                sample_probability_mass(
                    unique_indices,
                    p,
                )

            # ------------------------------------------------
            # Sample local energies
            # ------------------------------------------------

            sample_eloc =
                frozen.local_energy[
                    unique_indices
                ]

            total_weight =
                sum(weights)

            Ehat =
                sum(
                    weights .* sample_eloc
                ) / total_weight

            energy_error =
                real(Ehat - frozen.energy)

            # ------------------------------------------------
            # Target A:
            #
            # Actual VMC target using sample-estimated energy.
            # ------------------------------------------------

            y_vmc =
                Vector{Float64}(
                    undef,
                    nunique,
                )

            @inbounds for i in 1:nunique

                y_vmc[i] =
                    -real(
                        sample_eloc[i] -
                        Ehat
                    )

            end

            # ------------------------------------------------
            # Target B:
            #
            # Control using exact frozen energy.
            # ------------------------------------------------

            y_exactE =
                Vector{Float64}(
                    undef,
                    nunique,
                )

            @inbounds for i in 1:nunique

                y_exactE[i] =
                    -real(
                        sample_eloc[i] -
                        frozen.energy
                    )

            end

            # ------------------------------------------------
            # Fit VMC-target tree
            # ------------------------------------------------

            tree_vmc,
            train_prediction_vmc =
                fit_diagnostic_tree(
                    X,
                    y_vmc,
                    weights,
                )

            eval_vmc =
                evaluate_tree(
                    tree_vmc,
                    y_vmc,
                    train_prediction_vmc,
                    weights,
                    frozen,
                    full_states,
                )

            # ------------------------------------------------
            # Fit exact-energy-control tree
            # ------------------------------------------------

            tree_exactE,
            train_prediction_exactE =
                fit_diagnostic_tree(
                    X,
                    y_exactE,
                    weights,
                )

            eval_exactE =
                evaluate_tree(
                    tree_exactE,
                    y_exactE,
                    train_prediction_exactE,
                    weights,
                    frozen,
                    full_states,
                )

            push!(
                probability_r2_vmc,
                eval_vmc.probability_r2,
            )

            push!(
                probability_r2_exactE,
                eval_exactE.probability_r2,
            )

            # ------------------------------------------------
            # Save one row for each target type
            # ------------------------------------------------

            common =
                (
                    J = J,
                    h = h,
                    J_over_h = J / h,

                    training_run =
                        training_run,

                    epoch =
                        epoch,

                    diagnostic_nsamples =
                        M,

                    resample =
                        resample,

                    frozen_energy =
                        real(frozen.energy),

                    frozen_variance =
                        real(frozen.variance),

                    frozen_target_rms =
                        frozen.target_rms,

                    participation_ratio =
                        frozen.participation_ratio,

                    participation_fraction =
                        frozen.participation_fraction,

                    unique_states =
                        nunique,

                    unique_fraction =
                        unique_fraction,

                    hilbert_fraction =
                        hilbert_fraction,

                    sample_probability_mass =
                        probability_mass,

                    sampled_energy =
                        real(Ehat),

                    energy_error =
                        energy_error,
                )

            push!(
                rows,
                merge(
                    common,
                    (
                        target_type =
                            "vmc_energy",

                        train_r2 =
                            eval_vmc.train_r2,

                        probability_r2 =
                            eval_vmc.probability_r2,

                        hilbert_r2 =
                            eval_vmc.hilbert_r2,

                        probability_corr =
                            eval_vmc.probability_corr,

                        hilbert_corr =
                            eval_vmc.hilbert_corr,
                    ),
                ),
            )

            push!(
                rows,
                merge(
                    common,
                    (
                        target_type =
                            "exact_energy",

                        train_r2 =
                            eval_exactE.train_r2,

                        probability_r2 =
                            eval_exactE.probability_r2,

                        hilbert_r2 =
                            eval_exactE.hilbert_r2,

                        probability_corr =
                            eval_exactE.probability_corr,

                        hilbert_corr =
                            eval_exactE.hilbert_corr,
                    ),
                ),
            )

        end

        # ----------------------------------------------------
        # Compact checkpoint output
        # ----------------------------------------------------

        finite_vmc =
            filter(
                isfinite,
                probability_r2_vmc,
            )

        finite_exact =
            filter(
                isfinite,
                probability_r2_exactE,
            )

        mean_vmc =
            isempty(finite_vmc) ?
            NaN :
            mean(finite_vmc)

        mean_exact =
            isempty(finite_exact) ?
            NaN :
            mean(finite_exact)

        @printf(
            "    M=%4d   mean R²p(VMC)=%.4f   mean R²p(exact-E)=%.4f\n",
            M,
            mean_vmc,
            mean_exact,
        )

    end

    return rows
end


# ============================================================
# One normal training trajectory
# ============================================================

function run_training_trajectory(
    J::Float64,
    training_run::Int,
    full_states::Matrix{Int8},
)

    H =
        TFIMHamiltonian(
            N;
            J = J,
            h = h,
            periodic = true,
        )

    rng_training =
        MersenneTwister(
            base_seed +
            round(Int, 100_000 * J) +
            10_000 * training_run,
        )

    rng_diagnostics =
        MersenneTwister(
            base_seed +
            5_000_000 +
            round(Int, 100_000 * J) +
            10_000 * training_run,
        )

    # --------------------------------------------------------
    # Initial VMC population
    # --------------------------------------------------------

    samples =
        Matrix{Int8}(
            undef,
            training_nsamples,
            N,
        )

    @inbounds for i in eachindex(samples)

        samples[i] =
            rand(rng_training, Bool) ?
            Int8(1) :
            Int8(-1)

    end

    # --------------------------------------------------------
    # Uniform initial model
    # --------------------------------------------------------

    model =
        LogGBState(
            logamp_bias = 0.0,
            phase_bias = 0.0,
            use_phase = false,
        )

    logamps =
        zeros(
            Float64,
            training_nsamples,
        )

    # --------------------------------------------------------
    # Burn-in
    # --------------------------------------------------------

    for _ in 1:burn_in_sweeps

        GBTQuantum.sweep!(
            rng_training,
            model,
            samples,
            logamps,
        )

    end

    rows =
        NamedTuple[]

    # ========================================================
    # Training epochs
    # ========================================================

    for epoch in 1:nepochs

        # ----------------------------------------------------
        # Current PRE-UPDATE batch
        # ----------------------------------------------------

        batch =
            vmc_batch(
                H,
                model,
                samples,
            )

        yA, _ =
            make_targets(batch)

        weights =
            batch.counts

        # ----------------------------------------------------
        # Freeze and diagnose BEFORE applying this epoch's tree
        # ----------------------------------------------------

        if epoch in checkpoint_epochs

            checkpoint_rows =
                diagnose_frozen_checkpoint(
                    rng_diagnostics,
                    H,
                    model,
                    full_states,
                    J,
                    training_run,
                    epoch,
                )

            append!(
                rows,
                checkpoint_rows,
            )

        end

        # ----------------------------------------------------
        # Actual optimizer tree
        # ----------------------------------------------------

        tree =
            GBTQuantum.grow_tree(
                batch.states,
                yA,
                weights;
                max_depth = max_depth,
                min_weight = min_leaf_weight,
                min_gain = min_gain,
            )

        train_prediction =
            predict_all(
                tree,
                batch.states,
            )

        μtree =
            weighted_mean_local(
                train_prediction,
                weights,
            )

        if μtree != 0.0

            tree =
                shift_tree_leaves(
                    tree,
                    μtree,
                )

        end

        # ----------------------------------------------------
        # Apply normal boosting update
        # ----------------------------------------------------

        push!(
            model.logamp.trees,
            scale_tree(
                tree,
                eta,
            ),
        )

        GBTQuantum.refresh_logamps!(
            logamps,
            model,
            samples,
        )

        for _ in 1:sweeps_per_epoch

            GBTQuantum.sweep!(
                rng_training,
                model,
                samples,
                logamps,
            )

        end
    end

    return rows
end


# ============================================================
# CSV writer
# ============================================================

function write_csv(
    path,
    rows,
)

    isempty(rows) &&
        error(
            "No diagnostic rows generated."
        )

    names =
        propertynames(
            first(rows)
        )

    open(path, "w") do io

        println(
            io,
            join(
                string.(names),
                ",",
            ),
        )

        for row in rows

            values =
                [
                    getproperty(row, name)
                    for name in names
                ]

            println(
                io,
                join(
                    values,
                    ",",
                ),
            )

        end
    end
end


# ============================================================
# Finite summary statistics
# ============================================================

function finite_mean_std(values)

    x =
        Float64[
            v
            for v in values
            if isfinite(v)
        ]

    isempty(x) &&
        return NaN, NaN

    length(x) == 1 &&
        return x[1], 0.0

    return mean(x), std(x)
end


# ============================================================
# Final summary
# ============================================================

function print_summary(rows)

    println()
    println()
    println("============================================================")
    println("FROZEN-TARGET RECONSTRUCTION SUMMARY")
    println("============================================================")
    println()

    println(
        " J/h  epoch     M       R²p VMC          R²p exact-E       ΔR²p"
    )

    println(
        "--------------------------------------------------------------------------"
    )

    for J in J_values

        for epoch in sort(
            collect(checkpoint_epochs)
        )

            for M in diagnostic_sample_sizes

                vmc_rows =
                    [
                        r
                        for r in rows
                        if r.J == J &&
                           r.epoch == epoch &&
                           r.diagnostic_nsamples == M &&
                           r.target_type == "vmc_energy"
                    ]

                exact_rows =
                    [
                        r
                        for r in rows
                        if r.J == J &&
                           r.epoch == epoch &&
                           r.diagnostic_nsamples == M &&
                           r.target_type == "exact_energy"
                    ]

                μvmc, σvmc =
                    finite_mean_std(
                        [
                            r.probability_r2
                            for r in vmc_rows
                        ]
                    )

                μexact, σexact =
                    finite_mean_std(
                        [
                            r.probability_r2
                            for r in exact_rows
                        ]
                    )

                Δ =
                    μexact - μvmc

                @printf(
                    "%4.2f   %3d   %4d    %7.3f ± %-7.3f   %7.3f ± %-7.3f   %+7.4f\n",
                    J / h,
                    epoch,
                    M,
                    μvmc,
                    σvmc,
                    μexact,
                    σexact,
                    Δ,
                )

            end

            println()

        end
    end
end


# ============================================================
# Main
# ============================================================

function main()

    d =
        1 << N

    println()
    println("============================================================")
    println("FROZEN-TARGET RECONSTRUCTION")
    println("N                     = ", N)
    println("Hilbert dimension     = ", d)
    println("h                     = ", h)
    println("Training population   = ", training_nsamples)
    println("Training epochs       = ", nepochs)
    println("Training runs / J     = ", ntraining_runs)
    println("Resamples/checkpoint  = ", nresamples)
    println(
        "Checkpoint epochs     = ",
        sort(
            collect(checkpoint_epochs)
        ),
    )
    println(
        "Diagnostic M          = ",
        diagnostic_sample_sizes,
    )
    println("============================================================")

    full_states =
        enumerate_states(N)

    all_rows =
        NamedTuple[]

    for J in J_values

        println()
        println()
        println("============================================================")

        @printf(
            "J/h = %.4f\n",
            J / h,
        )

        println("============================================================")

        for training_run in 1:ntraining_runs

            println()
            @printf(
                "Training trajectory %d/%d\n",
                training_run,
                ntraining_runs,
            )

            trajectory_rows =
                run_training_trajectory(
                    J,
                    training_run,
                    full_states,
                )

            append!(
                all_rows,
                trajectory_rows,
            )

        end
    end

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output_dir =
        joinpath(
            @__DIR__,
            "results",
        )

    mkpath(output_dir)

    csv_path =
        joinpath(
            output_dir,
            "frozen_target_reconstruction.csv",
        )

    write_csv(
        csv_path,
        all_rows,
    )

    print_summary(
        all_rows,
    )

    println()
    println("============================================================")
    println("RESULTS WRITTEN TO")
    println(csv_path)
    println("============================================================")
end


main()