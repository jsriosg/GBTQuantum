using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Random
using Statistics
using Printf

# ============================================================
# Initial-target signal / finite-sample reconstruction experiment
#
# Purpose:
#
# At epoch 0 the GBT wavefunction is uniform:
#
#     A(x) = 0
#     Φ(x) = 0
#
# For the TFIM
#
#     H = -J Σ_i σᶻ_i σᶻ_{i+1} - h Σ_i σˣ_i
#
# this gives
#
#     E_loc(x) = -J Σ_i x_i x_{i+1} - Nh.
#
# Under the exact uniform distribution:
#
#     E = -Nh
#
# and therefore the exact magnitude target is
#
#     y_A(x) = -[E_loc(x)-E]
#            = J Σ_i x_i x_{i+1}.
#
# This experiment asks:
#
#   1. How large is that physical target signal?
#   2. How accurately does a finite Monte Carlo population
#      represent it?
#   3. Can a regression tree fitted to that finite sample
#      reconstruct the target over the COMPLETE Hilbert space?
#   4. Is there evidence of finite-sample overfitting?
#
# No boosting update is applied. We study only the first
# regression problem seen by the optimizer.
# ============================================================


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

const N = 12
const h = 1.0

# J/h values. Since h = 1, these are numerically also J.
const J_values = [
    0.02,
    0.05,
    0.10,
    0.20,
    0.50,
    1.00,
    2.00,
]

const sample_sizes = [
    32,
    64,
    128,
    256,
    512,
    1024,
]

# This experiment is cheap compared with full training,
# so use many independent realizations.
const nruns = 50

const max_depth = 4
const min_leaf_weight = 1.0
const min_gain = 0.0

const base_seed = 91_000


# ============================================================
# Helpers
# ============================================================

"""
    enumerate_states(N)

Return all 2^N spin configurations as a Matrix{Int8}.

Each row is one z-basis state with spins ±1.
"""
function enumerate_states(N::Int)

    d = 1 << N
    states = Matrix{Int8}(undef, d, N)

    @inbounds for s in 0:(d - 1)
        for i in 1:N
            states[s + 1, i] =
                ((s >> (i - 1)) & 1) == 1 ?
                Int8(1) : Int8(-1)
        end
    end

    return states
end


"""
    bond_sum(x; periodic=true)

Compute

    Σ_i x_i x_{i+1}

for one spin configuration.
"""
@inline function bond_sum(
    x::AbstractVector{<:Real};
    periodic::Bool = true,
)

    Nloc = length(x)

    s = 0.0

    @inbounds for i in 1:(Nloc - 1)
        s += x[i] * x[i + 1]
    end

    if periodic
        s += x[Nloc] * x[1]
    end

    return s
end


"""
    exact_initial_targets(states, J)

Exact epoch-0 magnitude target for the uniform wavefunction:

    y_A(x) = J Σ_i x_i x_{i+1}.
"""
function exact_initial_targets(
    states::Matrix{Int8},
    J::Float64,
)

    d = size(states, 1)

    y = Vector{Float64}(undef, d)

    @inbounds for s in 1:d
        y[s] = J * bond_sum(
            @view(states[s, :])
        )
    end

    return y
end


"""
Weighted mean.
"""
function weighted_mean_local(
    y::AbstractVector{<:Real},
    w::AbstractVector{<:Real},
)

    length(y) == length(w) ||
        throw(DimensionMismatch("y and w must have the same length"))

    total = 0.0
    W = 0.0

    @inbounds for i in eachindex(y, w)
        wi = Float64(w[i])
        total += wi * Float64(y[i])
        W += wi
    end

    return total / W
end


"""
Weighted MSE.
"""
function weighted_mse_local(
    ytrue::AbstractVector{<:Real},
    ypred::AbstractVector{<:Real},
    w::AbstractVector{<:Real},
)

    total = 0.0
    W = 0.0

    @inbounds for i in eachindex(ytrue, ypred, w)

        wi = Float64(w[i])

        δ =
            Float64(ypred[i]) -
            Float64(ytrue[i])

        total += wi * δ * δ
        W += wi
    end

    return total / W
end


"""
Weighted R².

Returns NaN if the target has zero weighted variance.
"""
function weighted_r2(
    ytrue::AbstractVector{<:Real},
    ypred::AbstractVector{<:Real},
    w::AbstractVector{<:Real},
)

    μ = weighted_mean_local(ytrue, w)

    ss_res = 0.0
    ss_tot = 0.0

    @inbounds for i in eachindex(ytrue, ypred, w)

        wi = Float64(w[i])

        δres =
            Float64(ytrue[i]) -
            Float64(ypred[i])

        δtot =
            Float64(ytrue[i]) - μ

        ss_res += wi * δres * δres
        ss_tot += wi * δtot * δtot
    end

    if ss_tot <= eps(Float64)
        return NaN
    end

    return 1.0 - ss_res / ss_tot
end


"""
Ordinary full-space R².
"""
function ordinary_r2(
    ytrue::AbstractVector{<:Real},
    ypred::AbstractVector{<:Real},
)

    μ = mean(ytrue)

    ss_res = 0.0
    ss_tot = 0.0

    @inbounds for i in eachindex(ytrue, ypred)

        δres =
            Float64(ytrue[i]) -
            Float64(ypred[i])

        δtot =
            Float64(ytrue[i]) - μ

        ss_res += δres * δres
        ss_tot += δtot * δtot
    end

    if ss_tot <= eps(Float64)
        return NaN
    end

    return 1.0 - ss_res / ss_tot
end


"""
Pearson correlation.

Returns NaN for a constant vector.
"""
function safe_correlation(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)

    sx = std(x)
    sy = std(y)

    if sx <= eps(Float64) ||
       sy <= eps(Float64)

        return NaN
    end

    return cor(x, y)
end


"""
Shift all leaves of a regression tree by a constant.

This reproduces the gauge-fixing operation used in train().
"""
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


"""
Predict one tree over all rows of X.
"""
function predict_all(
    tree::RegressionTree,
    X::AbstractMatrix{<:Real},
)

    y = Vector{Float64}(undef, size(X, 1))

    @inbounds for i in axes(X, 1)
        y[i] =
            predict(
                tree,
                @view(X[i, :]),
            )
    end

    return y
end


"""
Mean and standard deviation ignoring NaN/Inf.
"""
function finite_mean_std(x)

    values =
        Float64[
            v for v in x
            if isfinite(v)
        ]

    if isempty(values)
        return NaN, NaN
    end

    if length(values) == 1
        return values[1], 0.0
    end

    return mean(values), std(values)
end


# ============================================================
# Main experiment
# ============================================================

function main()

    d = 1 << N

    println()
    println("================================================")
    println("INITIAL TARGET SIGNAL / RECONSTRUCTION EXPERIMENT")
    println("N                  = ", N)
    println("Hilbert dimension  = ", d)
    println("h                  = ", h)
    println("Independent runs   = ", nruns)
    println("Tree max depth     = ", max_depth)
    println("================================================")
    println()

    # --------------------------------------------------------
    # Enumerate complete Hilbert space once.
    # --------------------------------------------------------

    full_states = enumerate_states(N)

    nJ = length(J_values)
    nM = length(sample_sizes)

    # --------------------------------------------------------
    # Storage
    #
    # Dimensions:
    #
    #     [J index, sample-size index, run]
    # --------------------------------------------------------

    target_rms =
        fill(NaN, nJ, nM, nruns)

    target_std =
        fill(NaN, nJ, nM, nruns)

    target_exact_rms =
        fill(NaN, nJ, nM, nruns)

    target_exact_std =
        fill(NaN, nJ, nM, nruns)

    target_rms_ratio =
        fill(NaN, nJ, nM, nruns)

    tree_train_mse =
        fill(NaN, nJ, nM, nruns)

    tree_hilbert_mse =
        fill(NaN, nJ, nM, nruns)

    tree_train_r2 =
        fill(NaN, nJ, nM, nruns)

    tree_hilbert_r2 =
        fill(NaN, nJ, nM, nruns)

    generalization_gap =
        fill(NaN, nJ, nM, nruns)

    target_correlation =
        fill(NaN, nJ, nM, nruns)

    unique_fraction =
        fill(NaN, nJ, nM, nruns)

    hilbert_coverage =
        fill(NaN, nJ, nM, nruns)

    empirical_energy_error =
        fill(NaN, nJ, nM, nruns)

    # --------------------------------------------------------
    # Loop over J/h
    # --------------------------------------------------------

    for (iJ, J) in enumerate(J_values)

        H =
            TFIMHamiltonian(
                N;
                J = J,
                h = h,
                periodic = true,
            )

        # Exact uniform-state target over entire Hilbert space.
        yexact =
            exact_initial_targets(
                full_states,
                J,
            )

        exact_rms =
            sqrt(mean(abs2, yexact))

        exact_std =
            std(yexact; corrected = false)

        # Analytical prediction:
        #
        #     RMS[y] = J sqrt(N)
        #
        analytic_rms =
            J * sqrt(N)

        println("================================================")
        @printf("J/h = %.4f\n", J / h)
        @printf(
            "Exact target RMS      = %.8e\n",
            exact_rms,
        )
        @printf(
            "Analytic J*sqrt(N)    = %.8e\n",
            analytic_rms,
        )
        @printf(
            "Relative difference   = %.3e\n",
            abs(exact_rms - analytic_rms) /
            max(abs(analytic_rms), eps(Float64)),
        )
        println("================================================")
        println()

        # ----------------------------------------------------
        # Loop over sample sizes
        # ----------------------------------------------------

        for (iM, M) in enumerate(sample_sizes)

            @printf(
                "Samples = %d  (nominal Hilbert fraction %.4f)\n",
                M,
                M / d,
            )

            for run in 1:nruns

                rng =
                    MersenneTwister(
                        base_seed +
                        100_000 * iJ +
                        1_000 * iM +
                        run,
                    )

                # --------------------------------------------
                # Exact epoch-0 model used by train():
                #
                # A = 0
                # Φ = 0
                # --------------------------------------------

                model =
                    LogGBState(
                        logamp_bias = 0.0,
                        phase_bias = 0.0,
                        use_phase = false,
                    )

                # --------------------------------------------
                # At epoch zero the target distribution is
                # exactly uniform, so independent uniform
                # samples are exact draws from |ψ|².
                #
                # This intentionally removes Markov-chain
                # autocorrelation from the present experiment.
                # We are isolating finite-population target
                # reconstruction.
                # --------------------------------------------

                samples =
                    Matrix{Int8}(undef, M, N)

                @inbounds for i in eachindex(samples)
                    samples[i] =
                        rand(rng, Bool) ?
                        Int8(1) :
                        Int8(-1)
                end

                # --------------------------------------------
                # Use the actual package VMC machinery.
                # --------------------------------------------

                batch =
                    vmc_batch(
                        H,
                        model,
                        samples,
                    )

                yA, _ =
                    make_targets(batch)

                weights = batch.counts

                K = length(weights)

                # --------------------------------------------
                # Finite-sample target statistics.
                #
                # Need multiplicity weights because yA contains
                # one value per UNIQUE sampled configuration.
                # --------------------------------------------

                μsample =
                    weighted_mean_local(
                        yA,
                        weights,
                    )

                second_moment = 0.0
                variance_sample = 0.0

                @inbounds for j in eachindex(yA, weights)

                    wj = Float64(weights[j])
                    yj = yA[j]

                    second_moment +=
                        wj * yj * yj

                    δ = yj - μsample

                    variance_sample +=
                        wj * δ * δ
                end

                second_moment /= M
                variance_sample /= M

                sample_rms =
                    sqrt(second_moment)

                sample_std =
                    sqrt(max(variance_sample, 0.0))

                target_rms[iJ, iM, run] =
                    sample_rms

                target_std[iJ, iM, run] =
                    sample_std

                target_exact_rms[iJ, iM, run] =
                    exact_rms

                target_exact_std[iJ, iM, run] =
                    exact_std

                target_rms_ratio[iJ, iM, run] =
                    sample_rms / exact_rms

                # --------------------------------------------
                # Exact uniform-state energy:
                #
                #     E = -Nh
                #
                # Compare against empirical batch mean.
                # --------------------------------------------

                Eexact_uniform =
                    -N * h

                empirical_energy_error[iJ, iM, run] =
                    real(batch.energy) -
                    Eexact_uniform

                # --------------------------------------------
                # Fit exactly the same type of magnitude tree
                # used by train().
                # --------------------------------------------

                tree =
                    GBTQuantum.grow_tree(
                        batch.states,
                        yA,
                        weights;
                        max_depth = max_depth,
                        min_weight = min_leaf_weight,
                        min_gain = min_gain,
                    )

                # --------------------------------------------
                # Reproduce train() gauge fixing.
                # --------------------------------------------

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

                    train_prediction =
                        predict_all(
                            tree,
                            batch.states,
                        )
                end

                # --------------------------------------------
                # Training-set reconstruction.
                # --------------------------------------------

                train_mse =
                    weighted_mse_local(
                        yA,
                        train_prediction,
                        weights,
                    )

                train_r2 =
                    weighted_r2(
                        yA,
                        train_prediction,
                        weights,
                    )

                tree_train_mse[iJ, iM, run] =
                    train_mse

                tree_train_r2[iJ, iM, run] =
                    train_r2

                # --------------------------------------------
                # Evaluate same tree over COMPLETE Hilbert
                # space.
                # --------------------------------------------

                hilbert_prediction =
                    predict_all(
                        tree,
                        full_states,
                    )

                hilbert_mse =
                    mean(
                        (
                            hilbert_prediction .-
                            yexact
                        ) .^ 2
                    )

                hilbert_r2 =
                    ordinary_r2(
                        yexact,
                        hilbert_prediction,
                    )

                tree_hilbert_mse[iJ, iM, run] =
                    hilbert_mse

                tree_hilbert_r2[iJ, iM, run] =
                    hilbert_r2

                generalization_gap[iJ, iM, run] =
                    train_r2 - hilbert_r2

                # --------------------------------------------
                # Correlation between learned update and exact
                # physical update across Hilbert space.
                # --------------------------------------------

                target_correlation[iJ, iM, run] =
                    safe_correlation(
                        yexact,
                        hilbert_prediction,
                    )

                # --------------------------------------------
                # Coverage diagnostics.
                # --------------------------------------------

                unique_fraction[iJ, iM, run] =
                    K / M

                hilbert_coverage[iJ, iM, run] =
                    K / d

                if run == 1 ||
                   run == nruns ||
                   run % 10 == 0

                    @printf(
                        "  run %2d/%d   R²train=% .4f   R²Hilbert=% .4f   corr=% .4f   RMS ratio=%.4f   cov=%.4f\n",
                        run,
                        nruns,
                        train_r2,
                        hilbert_r2,
                        target_correlation[iJ, iM, run],
                        target_rms_ratio[iJ, iM, run],
                        hilbert_coverage[iJ, iM, run],
                    )
                end
            end

            println()
        end
    end


    # ========================================================
    # Summary
    # ========================================================

    println()
    println("==============================================================")
    println("INITIAL TARGET RECONSTRUCTION SUMMARY")
    println("==============================================================")
    println()

    for (iJ, J) in enumerate(J_values)

        @printf("J/h = %.4f\n", J / h)

        println(
            " M      target RMS   RMS/exact   " *
            "R² train    R² Hilbert   gap        corr"
        )

        println(
            "---------------------------------------------------------------"
        )

        for (iM, M) in enumerate(sample_sizes)

            rms_mean, _ =
                finite_mean_std(
                    @view(target_rms[iJ, iM, :])
                )

            ratio_mean, _ =
                finite_mean_std(
                    @view(target_rms_ratio[iJ, iM, :])
                )

            train_mean, _ =
                finite_mean_std(
                    @view(tree_train_r2[iJ, iM, :])
                )

            hilbert_mean, _ =
                finite_mean_std(
                    @view(tree_hilbert_r2[iJ, iM, :])
                )

            gap_mean, _ =
                finite_mean_std(
                    @view(generalization_gap[iJ, iM, :])
                )

            corr_mean, _ =
                finite_mean_std(
                    @view(target_correlation[iJ, iM, :])
                )

            @printf(
                "%4d   %10.4e   %8.4f   %9.4f   %10.4f   %8.4f   %8.4f\n",
                M,
                rms_mean,
                ratio_mean,
                train_mean,
                hilbert_mean,
                gap_mean,
                corr_mean,
            )
        end

        println()
    end


    # ========================================================
    # CSV output
    # ========================================================

    output_dir =
        joinpath(@__DIR__, "results")

    mkpath(output_dir)

    csv_path =
        joinpath(
            output_dir,
            "initial_target_snr.csv",
        )

    open(csv_path, "w") do io

        println(
            io,
            "J,h,J_over_h,M,run," *
            "target_rms,target_std," *
            "target_exact_rms,target_exact_std," *
            "target_rms_ratio," *
            "tree_train_mse,tree_hilbert_mse," *
            "tree_train_r2,tree_hilbert_r2," *
            "generalization_gap,target_correlation," *
            "unique_fraction,hilbert_coverage," *
            "empirical_energy_error"
        )

        for (iJ, J) in enumerate(J_values)
            for (iM, M) in enumerate(sample_sizes)
                for run in 1:nruns

                    println(
                        io,
                        join(
                            (
                                J,
                                h,
                                J / h,
                                M,
                                run,

                                target_rms[iJ, iM, run],
                                target_std[iJ, iM, run],

                                target_exact_rms[iJ, iM, run],
                                target_exact_std[iJ, iM, run],

                                target_rms_ratio[iJ, iM, run],

                                tree_train_mse[iJ, iM, run],
                                tree_hilbert_mse[iJ, iM, run],

                                tree_train_r2[iJ, iM, run],
                                tree_hilbert_r2[iJ, iM, run],

                                generalization_gap[iJ, iM, run],
                                target_correlation[iJ, iM, run],

                                unique_fraction[iJ, iM, run],
                                hilbert_coverage[iJ, iM, run],

                                empirical_energy_error[iJ, iM, run],
                            ),
                            ",",
                        )
                    )
                end
            end
        end
    end

    println()
    println("Results written to:")
    println(csv_path)
    println()


    # ========================================================
    # Interpretation reminder
    # ========================================================

    println("==============================================================")
    println("INTERPRETATION")
    println("==============================================================")
    println(
        "R² train    : how well the tree fits the sampled VMC target."
    )
    println(
        "R² Hilbert  : how well that tree reconstructs the exact target"
    )
    println(
        "              over all 2^N configurations."
    )
    println(
        "gap         : R² train - R² Hilbert."
    )
    println(
        "corr        : correlation between tree prediction and exact"
    )
    println(
        "              target over the complete Hilbert space."
    )
    println()
    println(
        "Large train R² + poor Hilbert R² indicates finite-sample"
    )
    println(
        "overfitting rather than learning of the physical target."
    )
    println("==============================================================")
end


main()