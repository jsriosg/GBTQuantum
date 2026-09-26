module ExactTreeCapacityExperiment

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Random
using Statistics
using Printf


# ============================================================
# EXACT FULL-HILBERT TREE-CAPACITY EXPERIMENT
#
# Purpose
# -------
# Measure the reconstruction capacity of a single greedy
# regression tree when finite configuration-sampling error is
# removed completely.
#
# We reproduce the same normal VMC training trajectories used
# in frozen_target_reconstruction.jl.
#
# At selected PRE-UPDATE checkpoints we freeze the current
# wavefunction and enumerate the complete Hilbert space.
#
# For every configuration x we calculate exactly:
#
#     p_t(x)
#     E_L,t(x)
#     E_t
#     y_t(x) = -Re[E_L,t(x) - E_t]
#
# We then fit ONE regression tree using ALL Hilbert states,
# with p_t(x) as the regression weight.
#
# Therefore:
#
#     no MCMC error,
#     no finite-sample error,
#     no energy-estimation error.
#
# The remaining reconstruction error measures the combined
# effect of:
#
#     1. finite tree depth,
#     2. greedy splitting,
#     3. axis-aligned spin representation.
#
# Tree depth is swept to determine whether increasing weak-
# learner capacity removes the reconstruction bottleneck.
# ============================================================


# ============================================================
# Configuration
# ============================================================

const N = 12
const h = 1.0

const J_values = [
    0.05,
    2.00,
]

# Same normal-training trajectory as
# frozen_target_reconstruction.jl.
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

const ntraining_runs = 3

# Actual optimizer tree depth.
const optimizer_max_depth = 4

# Diagnostic exact-capacity depth sweep.
const diagnostic_depths = [
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12,
]

const eta = 0.05

const burn_in_sweeps = 100
const sweeps_per_epoch = 1

# Normal optimizer settings.
const optimizer_min_leaf_weight = 1.0
const optimizer_min_gain = 0.0

# IMPORTANT:
#
# Exact diagnostic weights are probabilities and therefore
# sum to 1.0. The normal min_weight=1.0 would prohibit every
# split.
#
# We therefore use a numerically negligible minimum weight.
const exact_min_leaf_weight = 1e-14
const exact_min_gain = 0.0

# Keep identical to frozen_target_reconstruction.jl so that
# training trajectories are directly comparable.
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

        total +=
            wi * Float64(y[i])

        W += wi
    end

    return total / W
end


function weighted_r2(
    ytrue,
    ypred,
    w,
)

    μ =
        weighted_mean_local(
            ytrue,
            w,
        )

    ss_res = 0.0
    ss_tot = 0.0

    @inbounds for i in eachindex(
        ytrue,
        ypred,
        w,
    )

        wi =
            Float64(w[i])

        δres =
            Float64(ytrue[i]) -
            Float64(ypred[i])

        δtot =
            Float64(ytrue[i]) -
            μ

        ss_res +=
            wi * δres^2

        ss_tot +=
            wi * δtot^2
    end

    ss_tot <= eps(Float64) &&
        return NaN

    return 1.0 -
           ss_res / ss_tot
end


function ordinary_r2(
    ytrue,
    ypred,
)

    μ =
        mean(ytrue)

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

    return 1.0 -
           ss_res / ss_tot
end


function safe_correlation(
    x,
    y,
)

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
# Tree prediction
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


# ============================================================
# Tree transformations
# ============================================================

function shift_tree_leaves(
    tree::RegressionTree,
    shift::Float64,
)

    nodes =
        copy(tree.nodes)

    @inbounds for i in eachindex(nodes)

        n =
            nodes[i]

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
# Tree structural diagnostics
# ============================================================

function tree_leaf_count(
    tree::RegressionTree,
)

    count(
        n -> n.isleaf,
        tree.nodes,
    )
end


function node_depth(
    tree::RegressionTree,
    node_index::Integer,
    depth::Int,
)

    node =
        tree.nodes[node_index]

    node.isleaf &&
        return depth

    left_depth =
        node_depth(
            tree,
            node.left,
            depth + 1,
        )

    right_depth =
        node_depth(
            tree,
            node.right,
            depth + 1,
        )

    return max(
        left_depth,
        right_depth,
    )
end


function tree_actual_depth(
    tree::RegressionTree,
)

    isempty(tree.nodes) &&
        return 0

    return node_depth(
        tree,
        1,
        0,
    )
end


# ============================================================
# Exact model probability distribution
# ============================================================

function exact_probabilities(
    model::LogGBState,
    states::Matrix{Int8},
)

    d =
        size(states, 1)

    logweights =
        Vector{Float64}(
            undef,
            d,
        )

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
        exp.(
            logweights .- m
        )

    return weights /
           sum(weights)
end


# ============================================================
# Exact frozen problem
# ============================================================

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
        Vector{ComplexF64}(
            undef,
            d,
        )

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
        Vector{Float64}(
            undef,
            d,
        )

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

    participation_ratio =
        1.0 /
        sum(abs2, p)

    participation_fraction =
        participation_ratio / d

    return (
        probabilities = p,
        local_energy = eloc,
        energy = E,
        target = target,
        target_mean = target_mean,
        target_rms = target_rms,
        variance = variance,
        participation_ratio =
            participation_ratio,
        participation_fraction =
            participation_fraction,
    )
end


# ============================================================
# Exact full-Hilbert tree fit
# ============================================================

function fit_exact_tree(
    full_states::Matrix{Int8},
    frozen,
    depth::Int,
)

    # Every Hilbert state is used.
    #
    # The exact model probabilities are the regression weights.
    tree =
        GBTQuantum.grow_tree(
            full_states,
            frozen.target,
            frozen.probabilities;
            max_depth = depth,
            min_weight =
                exact_min_leaf_weight,
            min_gain =
                exact_min_gain,
        )

    prediction =
        predict_all(
            tree,
            full_states,
        )

    # Same gauge-centering logic as the actual optimizer.
    μtree =
        weighted_mean_local(
            prediction,
            frozen.probabilities,
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
                full_states,
            )
    end

    return tree, prediction
end


# ============================================================
# Diagnose one frozen checkpoint
# ============================================================

function diagnose_exact_capacity(
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

    rows =
        NamedTuple[]

    @printf(
        "\n  Frozen epoch %2d: E=% .8f  target RMS=%.4e  PR/H=%.6f\n",
        epoch,
        real(frozen.energy),
        frozen.target_rms,
        frozen.participation_fraction,
    )

    # --------------------------------------------------------
    # Exact-target sanity checks
    # --------------------------------------------------------

    if abs(frozen.target_mean) > 1e-10

        @warn(
            "Exact frozen target does not have zero p-weighted mean",
            epoch = epoch,
            target_mean =
                frozen.target_mean,
        )
    end

    rms_variance_difference =
        abs(
            frozen.target_rms^2 -
            real(frozen.variance)
        )

    if rms_variance_difference > 1e-8

        @warn(
            "target RMS² and local-energy variance differ",
            epoch = epoch,
            difference =
                rms_variance_difference,
        )
    end

    # ========================================================
    # Tree-depth sweep
    # ========================================================

    for depth in diagnostic_depths

        tree,
        prediction =
            fit_exact_tree(
                full_states,
                frozen,
                depth,
            )

        probability_r2 =
            weighted_r2(
                frozen.target,
                prediction,
                frozen.probabilities,
            )

        hilbert_r2 =
            ordinary_r2(
                frozen.target,
                prediction,
            )

        probability_corr =
            probability_weighted_correlation(
                frozen.target,
                prediction,
                frozen.probabilities,
            )

        hilbert_corr =
            safe_correlation(
                frozen.target,
                prediction,
            )

        leaves =
            tree_leaf_count(tree)

        actual_depth =
            tree_actual_depth(tree)

        residual_rms =
            sqrt(
                sum(
                    frozen.probabilities .*
                    (
                        frozen.target .-
                        prediction
                    ).^2
                )
            )

        relative_residual_rms =
            frozen.target_rms >
            eps(Float64) ?
            residual_rms /
            frozen.target_rms :
            NaN

        @printf(
            "    depth=%2d   actual=%2d   leaves=%4d   R²p=% .6f   R²H=% .6f   corrp=% .6f\n",
            depth,
            actual_depth,
            leaves,
            probability_r2,
            hilbert_r2,
            probability_corr,
        )

        push!(
            rows,
            (
                J = J,
                h = h,
                J_over_h = J / h,

                training_run =
                    training_run,

                epoch =
                    epoch,

                requested_depth =
                    depth,

                actual_depth =
                    actual_depth,

                leaf_count =
                    leaves,

                frozen_energy =
                    real(frozen.energy),

                frozen_variance =
                    real(frozen.variance),

                frozen_target_mean =
                    frozen.target_mean,

                frozen_target_rms =
                    frozen.target_rms,

                participation_ratio =
                    frozen.participation_ratio,

                participation_fraction =
                    frozen.participation_fraction,

                probability_r2 =
                    probability_r2,

                hilbert_r2 =
                    hilbert_r2,

                probability_corr =
                    probability_corr,

                hilbert_corr =
                    hilbert_corr,

                residual_rms =
                    residual_rms,

                relative_residual_rms =
                    relative_residual_rms,
            ),
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

    # EXACTLY the same training seed construction as
    # frozen_target_reconstruction.jl.
    rng_training =
        MersenneTwister(
            base_seed +
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
    # Normal training
    # ========================================================

    for epoch in 1:nepochs

        # ----------------------------------------------------
        # Current PRE-UPDATE VMC batch
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
        # Freeze BEFORE applying this epoch's tree.
        #
        # This matches frozen_target_reconstruction.jl exactly.
        # ----------------------------------------------------

        if epoch in checkpoint_epochs

            checkpoint_rows =
                diagnose_exact_capacity(
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
                max_depth =
                    optimizer_max_depth,
                min_weight =
                    optimizer_min_leaf_weight,
                min_gain =
                    optimizer_min_gain,
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
# Final depth-capacity summary
# ============================================================

function print_summary(rows)

    println()
    println()
    println("============================================================")
    println("EXACT FULL-HILBERT TREE-CAPACITY SUMMARY")
    println("============================================================")
    println()

    println(
        " J/h  epoch  depth      R²p mean ± std        R²H mean ± std"
    )

    println(
        "--------------------------------------------------------------------------"
    )

    for J in J_values

        for epoch in sort(
            collect(checkpoint_epochs)
        )

            for depth in diagnostic_depths

                selected =
                    [
                        r
                        for r in rows
                        if r.J == J &&
                           r.epoch == epoch &&
                           r.requested_depth == depth
                    ]

                μp, σp =
                    finite_mean_std(
                        [
                            r.probability_r2
                            for r in selected
                        ]
                    )

                μH, σH =
                    finite_mean_std(
                        [
                            r.hilbert_r2
                            for r in selected
                        ]
                    )

                @printf(
                    "%4.2f   %3d    %2d      %7.4f ± %-7.4f    %7.4f ± %-7.4f\n",
                    J / h,
                    epoch,
                    depth,
                    μp,
                    σp,
                    μH,
                    σH,
                )
            end

            println()
        end
    end
end


# ============================================================
# Key checkpoint summary
# ============================================================

function print_key_checkpoints(rows)

    key_checkpoints = [
        (0.05, 1),
        (0.05, 64),
        (2.00, 16),
        (2.00, 64),
    ]

    println()
    println()
    println("============================================================")
    println("KEY CAPACITY CURVES")
    println("============================================================")

    for (J, epoch) in key_checkpoints

        println()

        @printf(
            "J/h = %.2f, epoch = %d\n",
            J / h,
            epoch,
        )

        println(
            "depth      mean R²p      std R²p      mean leaves"
        )

        println(
            "------------------------------------------------"
        )

        for depth in diagnostic_depths

            selected =
                [
                    r
                    for r in rows
                    if r.J == J &&
                       r.epoch == epoch &&
                       r.requested_depth == depth
                ]

            μp, σp =
                finite_mean_std(
                    [
                        r.probability_r2
                        for r in selected
                    ]
                )

            mean_leaves =
                isempty(selected) ?
                NaN :
                mean(
                    [
                        r.leaf_count
                        for r in selected
                    ]
                )

            @printf(
                "%3d        %8.4f      %8.4f      %8.2f\n",
                depth,
                μp,
                σp,
                mean_leaves,
            )
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
    println("EXACT FULL-HILBERT TREE CAPACITY")
    println("N                     = ", N)
    println("Hilbert dimension     = ", d)
    println("h                     = ", h)
    println("Training population   = ", training_nsamples)
    println("Training epochs       = ", nepochs)
    println("Training runs / J     = ", ntraining_runs)

    println(
        "Checkpoint epochs     = ",
        sort(
            collect(checkpoint_epochs)
        ),
    )

    println(
        "Diagnostic depths     = ",
        diagnostic_depths,
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
            "exact_tree_capacity.csv",
        )

    write_csv(
        csv_path,
        all_rows,
    )

    print_summary(
        all_rows,
    )

    print_key_checkpoints(
        all_rows,
    )

    println()
    println()
    println("============================================================")
    println("RESULTS WRITTEN TO")
    println(csv_path)
    println("============================================================")
end



end # module ExactTreeCapacityExperiment

if abspath(PROGRAM_FILE) == @__FILE__
    ExactTreeCapacityExperiment.main()
end
