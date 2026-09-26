using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Random
using Statistics
using Printf

# ============================================================
# TRAINING-TARGET DYNAMICS
#
# Follow the VMC regression problem through boosting time.
#
# At selected epochs measure:
#
#   - exact model energy
#   - sampled VMC energy
#   - exact target RMS
#   - sampled-target RMS
#   - tree R² on training population
#   - tree R² over complete Hilbert space
#   - target correlation over complete Hilbert space
#   - instantaneous state coverage
#   - cumulative state coverage
#   - cumulative probability-mass coverage
#   - participation ratio of |ψ|²
#
# The training loop reproduces Optimizer.jl but exposes the
# PRE-UPDATE model at diagnostic epochs.
# ============================================================


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

const N = 12
const h = 1.0

const J_values = [
    0.05,
    0.50,
    1.00,
    2.00,
]

const sample_sizes = [
    64,
    256,
]

const nepochs = 64

const diagnostic_epochs = Set([
    1, 2, 4, 8, 16, 32, 64
])

const nruns = 10

const max_depth = 4
const eta = 0.05

const burn_in_sweeps = 100
const sweeps_per_epoch = 1

const min_leaf_weight = 1.0
const min_gain = 0.0

const base_seed = 730_000


# ============================================================
# Hilbert-space helpers
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


# ------------------------------------------------------------
# Weighted statistics
# ------------------------------------------------------------

function weighted_mean_local(y, w)

    s = 0.0
    W = 0.0

    @inbounds for i in eachindex(y, w)

        wi = Float64(w[i])

        s += wi * Float64(y[i])
        W += wi
    end

    return s / W
end


function weighted_rms(y, w)

    s = 0.0
    W = 0.0

    @inbounds for i in eachindex(y, w)

        wi = Float64(w[i])

        s += wi * Float64(y[i])^2
        W += wi
    end

    return sqrt(s / W)
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

function probability_weighted_r2(
    ytrue::AbstractVector{<:Real},
    ypred::AbstractVector{<:Real},
    p::AbstractVector{<:Real},
)
    length(ytrue) == length(ypred) == length(p) ||
        throw(DimensionMismatch("Inputs must have equal length"))

    μ = sum(p .* ytrue)

    ss_res = sum(p .* (ytrue .- ypred).^2)
    ss_tot = sum(p .* (ytrue .- μ).^2)

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


function safe_correlation(x, y)

    std(x) <= eps(Float64) &&
        return NaN

    std(y) <= eps(Float64) &&
        return NaN

    return cor(x, y)
end

function probability_weighted_correlation(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    p::AbstractVector{<:Real},
)
    length(x) == length(y) == length(p) ||
        throw(DimensionMismatch("Inputs must have equal length"))

    μx = sum(p .* x)
    μy = sum(p .* y)

    dx = x .- μx
    dy = y .- μy

    varx = sum(p .* dx.^2)
    vary = sum(p .* dy.^2)

    if varx <= eps(Float64) || vary <= eps(Float64)
        return NaN
    end

    covxy = sum(p .* dx .* dy)

    return covxy / sqrt(varx * vary)
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

    nodes = copy(tree.nodes)

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
# Exact current-model distribution
# ============================================================

function exact_probabilities(
    model::LogGBState,
    states::Matrix{Int8},
)

    d = size(states, 1)

    logweights =
        Vector{Float64}(undef, d)

    @inbounds for s in 1:d

        A =
            logamplitude(
                model,
                @view(states[s, :]),
            )

        logweights[s] = 2.0 * A
    end

    # log-sum-exp stabilization
    m = maximum(logweights)

    weights =
        exp.(logweights .- m)

    Z = sum(weights)

    return weights ./ Z
end


# ============================================================
# Exact target over the complete Hilbert space
# ============================================================

function exact_target(
    H::TFIMHamiltonian,
    model::LogGBState,
    states::Matrix{Int8},
    probabilities::Vector{Float64},
)

    d = size(states, 1)

    eloc =
        Vector{ComplexF64}(undef, d)

    @inbounds for s in 1:d

        # local_energy! flips and restores the supplied state.
        eloc[s] =
            local_energy!(
                H,
                model,
                @view(states[s, :]),
            )
    end

    E =
        sum(
            probabilities .* eloc
        )

    target =
        Vector{Float64}(undef, d)

    @inbounds for s in 1:d

        target[s] =
            -real(
                eloc[s] - E
            )
    end

    variance =
        sum(
            probabilities .*
            abs2.(eloc .- real(E))
        )

    return (
        target = target,
        local_energy = eloc,
        energy = E,
        variance = variance,
    )
end


# ============================================================
# Probability mass of visited configurations
# ============================================================

function visited_probability_mass(
    probabilities::Vector{Float64},
    visited::Set{UInt64},
)

    mass = 0.0

    @inbounds for key in visited

        # spin_key encodes state s as the binary integer s.
        #
        # Hilbert enumeration uses:
        #
        #     row = s + 1
        #
        row =
            Int(key) + 1

        mass += probabilities[row]
    end

    return mass
end


# ============================================================
# Participation ratio
# ============================================================

function participation_ratio(
    probabilities::Vector{Float64},
)

    return 1.0 /
           sum(abs2, probabilities)
end


# ============================================================
# One complete training run
# ============================================================

function run_training_diagnostics(
    J::Float64,
    M::Int,
    run::Int,
    full_states::Matrix{Int8},
)

    d = size(full_states, 1)

    H =
        TFIMHamiltonian(
            N;
            J = J,
            h = h,
            periodic = true,
        )

    rng =
        MersenneTwister(
            base_seed +
            round(Int, 100_000 * J) +
            1_000 * M +
            run,
        )

    # --------------------------------------------------------
    # Initial population
    # --------------------------------------------------------

    samples =
        Matrix{Int8}(undef, M, N)

    @inbounds for i in eachindex(samples)

        samples[i] =
            rand(rng, Bool) ?
            Int8(1) :
            Int8(-1)
    end

    # --------------------------------------------------------
    # Initial model = uniform positive state
    # --------------------------------------------------------

    model =
        LogGBState(
            logamp_bias = 0.0,
            phase_bias = 0.0,
            use_phase = false,
        )

    logamps =
        zeros(Float64, M)

    # --------------------------------------------------------
    # Burn-in
    # --------------------------------------------------------

    for _ in 1:burn_in_sweeps

        GBTQuantum.sweep!(
            rng,
            model,
            samples,
            logamps,
        )
    end

    # --------------------------------------------------------
    # Cumulative state set
    # --------------------------------------------------------

    visited =
        Set{UInt64}()

    # --------------------------------------------------------
    # Diagnostic rows
    # --------------------------------------------------------

    rows =
        NamedTuple[]

    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in 1:nepochs

        # ----------------------------------------------------
        # PRE-UPDATE VMC batch
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
        # Record current states as visited BEFORE diagnostics
        # ----------------------------------------------------

        @inbounds for j in axes(batch.states, 1)

            push!(
                visited,
                GBTQuantum.spin_key(
                    @view(batch.states[j, :])
                ),
            )
        end

        # ----------------------------------------------------
        # Fit the same tree used by Optimizer.jl
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

        # Same gauge fixing as train()
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

            train_prediction =
                predict_all(
                    tree,
                    batch.states,
                )
        end

        # ====================================================
        # Exact diagnostics at selected epochs
        # ====================================================

        if epoch in diagnostic_epochs

            probabilities =
                exact_probabilities(
                    model,
                    full_states,
                )

            exact =
                exact_target(
                    H,
                    model,
                    full_states,
                    probabilities,
                )

            exact_y =
                exact.target

            # -----------------------------------------------
            # Tree prediction over entire Hilbert space
            # -----------------------------------------------

            hilbert_prediction =
                predict_all(
                    tree,
                    full_states,
                )

            # -----------------------------------------------
            # Target magnitudes
            # -----------------------------------------------

            sampled_target_rms =
                weighted_rms(
                    yA,
                    weights,
                )

            exact_target_rms =
                sqrt(
                    sum(
                        probabilities .*
                        exact_y.^2
                    )
                )

            # -----------------------------------------------
            # Reconstruction diagnostics
            # -----------------------------------------------

            train_r2 =
                weighted_r2(
                    yA,
                    train_prediction,
                    weights,
                )

            # Uniform Hilbert-space R²:
            #
            # Measures reconstruction of the target FUNCTION,
            # independent of current |ψ|² weighting.
            hilbert_r2 =
                ordinary_r2(
                    exact_y,
                    hilbert_prediction,
                )

            hilbert_corr =
                safe_correlation(
                    exact_y,
                    hilbert_prediction,
                )

            probability_r2 =
                probability_weighted_r2(
                    exact_y,
                    hilbert_prediction,
                    probabilities,
                )

            probability_corr =
                probability_weighted_correlation(
                    exact_y,
                    hilbert_prediction,
                    probabilities,
                )

            # -----------------------------------------------
            # Coverage
            # -----------------------------------------------

            nunique =
                size(batch.states, 1)

            instantaneous_coverage =
                nunique / d

            cumulative_coverage =
                length(visited) / d

            probability_coverage =
                visited_probability_mass(
                    probabilities,
                    visited,
                )

            # -----------------------------------------------
            # Participation ratio
            # -----------------------------------------------

            pr =
                participation_ratio(
                    probabilities,
                )

            pr_fraction =
                pr / d

            # -----------------------------------------------
            # Energies
            # -----------------------------------------------

            sampled_energy =
                real(batch.energy)

            exact_model_energy =
                real(exact.energy)

            # -----------------------------------------------
            # Save diagnostic row
            # -----------------------------------------------

            push!(
                rows,
                (
                    J = J,
                    h = h,
                    J_over_h = J / h,

                    nsamples = M,
                    run = run,
                    epoch = epoch,

                    sampled_energy =
                        sampled_energy,

                    exact_model_energy =
                        exact_model_energy,

                    exact_model_variance =
                        exact.variance,

                    sampled_target_rms =
                        sampled_target_rms,

                    exact_target_rms =
                        exact_target_rms,

                    target_rms_ratio =
                        sampled_target_rms /
                        max(
                            exact_target_rms,
                            eps(Float64),
                        ),

                    train_r2 =
                        train_r2,

                    probability_r2 =
                        probability_r2,

                    hilbert_r2 =
                        hilbert_r2,

                    probability_corr =
                        probability_corr,

                    hilbert_corr =
                        hilbert_corr,

                    generalization_gap =
                        train_r2 -
                        hilbert_r2,

                    sampling_gap =
                        train_r2 -
                        probability_r2,

                    relevance_gap =
                        probability_r2 -
                        hilbert_r2,

                    instantaneous_state_coverage =
                        instantaneous_coverage,

                    cumulative_state_coverage =
                        cumulative_coverage,

                    cumulative_probability_coverage =
                        probability_coverage,

                    participation_ratio =
                        pr,

                    participation_fraction =
                        pr_fraction,

                    unique_fraction =
                        nunique / M,
                ),
            )

            @printf(
                "J/h=%4.2f  M=%4d  run=%2d  epoch=%2d  R²tr=% .3f  R²p=% .3f  R²H=% .3f  ρp=% .3f  ρH=% .3f  Cprob=%.3f  PR/H=%.3f\n",
                J / h,
                M,
                run,
                epoch,
                train_r2,
                probability_r2,
                hilbert_r2,
                probability_corr,
                hilbert_corr,
                probability_coverage,
                pr_fraction,
            )
        end

        # ====================================================
        # Apply actual optimizer update AFTER diagnostics
        # ====================================================

        push!(
            model.logamp.trees,
            scale_tree(
                tree,
                eta,
            ),
        )

        # Model changed => cached amplitudes stale
        GBTQuantum.refresh_logamps!(
            logamps,
            model,
            samples,
        )

        # Same population evolution as train()
        for _ in 1:sweeps_per_epoch

            GBTQuantum.sweep!(
                rng,
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

function write_csv(path, rows)

    isempty(rows) &&
        error("No diagnostic rows generated.")

    names =
        propertynames(first(rows))

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
# Summary helper
# ============================================================

function finite_mean(x)

    y =
        [
            Float64(v)
            for v in x
            if isfinite(v)
        ]

    isempty(y) &&
        return NaN

    return mean(y)
end


# ============================================================
# Main
# ============================================================

function main()

    d = 1 << N

    println()
    println("================================================")
    println("TRAINING-TARGET DYNAMICS")
    println("N                  = ", N)
    println("Hilbert dimension  = ", d)
    println("h                  = ", h)
    println("Epochs             = ", nepochs)
    println("Independent runs   = ", nruns)
    println("Diagnostic epochs  = ", sort(collect(diagnostic_epochs)))
    println("================================================")
    println()

    full_states =
        enumerate_states(N)

    all_rows =
        NamedTuple[]

    for J in J_values

        println()
        println("================================================")
        @printf("FIELD RATIO J/h = %.4f\n", J / h)
        println("================================================")

        for M in sample_sizes

            println()
            println("-----------------------------------------------")
            @printf(
                "Samples = %d  nominal Hilbert fraction = %.4f\n",
                M,
                M / d,
            )
            println("-----------------------------------------------")

            for run in 1:nruns

                rows =
                    run_training_diagnostics(
                        J,
                        M,
                        run,
                        full_states,
                    )

                append!(
                    all_rows,
                    rows,
                )
            end
        end
    end

    # --------------------------------------------------------
    # Save results
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
            "training_target_dynamics.csv",
        )

    write_csv(
        csv_path,
        all_rows,
    )

    println()
    println("================================================")
    println("RESULTS WRITTEN TO")
    println(csv_path)
    println("================================================")
    println()

    # --------------------------------------------------------
    # Compact final summary at epoch 64
    # --------------------------------------------------------

    println()
    println("============== FINAL-EPOCH SUMMARY ==============")
    println()
    println(
    " J/h     M      R²tr      R²p       R²H       ρp        ρH      Cprob     PR/H"
    )

    println(
        "--------------------------------------------------------------------------------"
    )

    for J in J_values

        for M in sample_sizes

            selected =
                [
                    r
                    for r in all_rows
                    if r.J == J &&
                       r.nsamples == M &&
                       r.epoch == nepochs
                ]

            r2 =
                finite_mean(
                    [r.hilbert_r2 for r in selected]
                )

            corr =
                finite_mean(
                    [r.hilbert_corr for r in selected]
                )

            cs =
                finite_mean(
                    [
                        r.cumulative_state_coverage
                        for r in selected
                    ]
                )

            cp =
                finite_mean(
                    [
                        r.cumulative_probability_coverage
                        for r in selected
                    ]
                )

            pr =
                finite_mean(
                    [
                        r.participation_fraction
                        for r in selected
                    ]
                )

            rtrain =
                finite_mean(
                    [r.train_r2 for r in selected]
                )

            rp =
                finite_mean(
                    [r.probability_r2 for r in selected]
                )

            rhop =
                finite_mean(
                    [r.probability_corr for r in selected]
                )

            @printf(
                "%5.2f  %4d   %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f\n",
                J / h,
                M,
                rtrain,
                rp,
                r2,
                rhop,
                corr,
                cp,
                pr,
            )
        end
    end

    println()
    println("================================================")
end


main()