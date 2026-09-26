using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module SplitRecoveryExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

# ============================================================
# ORACLE SPLIT-RECOVERY EXPERIMENT
# ============================================================
# Question:
#   Does alternative sampling improve reconstruction because it recovers
#   the splits selected by a full-Hilbert-space (oracle) regression tree?
#
# At frozen training checkpoints we build ONE oracle depth-4 tree using the
# exact p-weighted target.  For every internal oracle node we then keep the
# oracle region fixed and ask what split would be selected from an
# importance-weighted sample restricted to that same region.
#
# This prevents errors at an earlier sampled split from changing the region
# tested at a deeper node.  We therefore measure split-estimation quality,
# not propagation of topology errors.
# ============================================================

const N = 12
const h = 1.0
const J = 2.0

const training_nsamples = 256
const nepochs = 64
const checkpoint_epochs = Set([8, 32, 64])
const ntraining_runs = 3

const optimizer_max_depth = 4
const eta = 0.05
const burn_in_sweeps = 100
const sweeps_per_epoch = 1
const optimizer_min_leaf_weight = 1.0
const optimizer_min_gain = 0.0

const oracle_depth = 4
const exact_min_leaf_weight = 1e-14
const exact_min_gain = 0.0
const sampled_min_leaf_weight = 1e-14

const diagnostic_sample_sizes = [256, 1024]
const nsampling_runs = 100
const mixture_values = [
    (0.00, 0.00),
    (0.50, 0.00),
    (0.25, 0.25),
    (0.50, 0.25),
]

const base_seed = 1_020_000
const diagnostic_seed_base = 9_020_000

function exact_frozen_problem(H, model, states)
    p = exact_probabilities(model, states)
    eloc = ComplexF64[
        local_energy!(H, model, @view(states[i, :])) for i in axes(states, 1)
    ]
    E = sum(p .* eloc)
    target = -real.(eloc .- E)
    mu = sum(p .* target)
    centered = target .- mu
    signal_density = p .* centered.^2
    total_signal = sum(signal_density)
    q_signal = total_signal > eps(Float64) ? signal_density ./ total_signal : copy(p)
    q_uniform = fill(1.0 / length(p), length(p))
    return (
        probabilities=p,
        target=target,
        energy=E,
        target_rms=sqrt(sum(p .* target.^2)),
        q_signal=q_signal,
        q_uniform=q_uniform,
    )
end

function proposal_distribution(frozen, alpha, beta)
    alpha < 0 && error("alpha must be nonnegative")
    beta < 0 && error("beta must be nonnegative")
    alpha + beta > 1 + 1e-12 && error("alpha + beta must be <= 1")
    q = (1-alpha-beta) .* frozen.probabilities .+
        alpha .* frozen.q_signal .+
        beta .* frozen.q_uniform
    q ./= sum(q)
    return q
end

# Exact gain landscape for a fixed node region.  This reproduces the split
# criterion in src/Trees.jl without modifying production tree code.
function gain_landscape(X, y, w, idx; min_weight=0.0)
    nfeatures = size(X, 2)
    gains = fill(-Inf, nfeatures)
    W = sum(w[i] for i in idx)
    W > 0 || return gains
    S = sum(w[i] * y[i] for i in idx)
    parent_score = S*S/W

    for f in 1:nfeatures
        WL = 0.0
        SL = 0.0
        @inbounds for i in idx
            if X[i, f] < 0
                wi = w[i]
                WL += wi
                SL += wi * y[i]
            end
        end
        WR = W - WL
        (WL < min_weight || WR < min_weight || WL <= 0 || WR <= 0) && continue
        SR = S - SL
        gains[f] = SL*SL/WL + SR*SR/WR - parent_score
    end
    return gains
end

function best_feature_from_gains(gains)
    best_feature = 0
    best_gain = 0.0
    for f in eachindex(gains)
        g = gains[f]
        if isfinite(g) && g > best_gain
            best_gain = g
            best_feature = f
        end
    end
    return best_feature, best_gain
end

function oracle_tree(full_states, frozen)
    return GBTQuantum.grow_tree(
        full_states, frozen.target, frozen.probabilities;
        max_depth=oracle_depth,
        min_weight=exact_min_leaf_weight,
        min_gain=exact_min_gain,
    )
end

# Enumerate internal oracle nodes together with the exact Hilbert-space
# region that reaches each node.  Depth(root)=0.
function oracle_node_regions(tree, X)
    rows = NamedTuple[]
    function walk(node_idx::Int, depth::Int, idx::Vector{Int}, path::String)
        node = tree.nodes[node_idx]
        node.isleaf && return
        push!(rows, (
            node_index=node_idx,
            depth=depth,
            path=path,
            oracle_feature=Int(node.feature),
            state_indices=copy(idx),
        ))
        left_idx = Int[]
        right_idx = Int[]
        f = Int(node.feature)
        for i in idx
            if X[i, f] < 0
                push!(left_idx, i)
            else
                push!(right_idx, i)
            end
        end
        walk(Int(node.left), depth+1, left_idx, path * "L")
        walk(Int(node.right), depth+1, right_idx, path * "R")
    end
    walk(1, 0, collect(axes(X, 1)), "")
    return rows
end

function local_sample_indices(global_draws, region_mask)
    return [i for i in global_draws if region_mask[i]]
end

function importance_ess(draws, q, p)
    isempty(draws) && return 0.0
    w = [p[i] / q[i] for i in draws]
    sw = sum(w); sw2 = sum(abs2, w)
    return sw2 > 0 ? sw^2 / sw2 : 0.0
end

function sampled_gain_landscape(X, y, p, q, draws, region_mask)
    local_draws = local_sample_indices(draws, region_mask)
    isempty(local_draws) && return fill(-Inf, size(X,2)), 0, 0, 0.0

    unique_idx, counts = compress_indices(local_draws)
    weights = counts .* p[unique_idx] ./ q[unique_idx]
    gains = gain_landscape(
        X, y, weights, collect(eachindex(unique_idx));
        min_weight=sampled_min_leaf_weight,
    )
    # gain_landscape above needs a compact X/y matching the compact weights.
    Xlocal = X[unique_idx, :]
    ylocal = y[unique_idx]
    gains = gain_landscape(
        Xlocal, ylocal, weights, collect(eachindex(unique_idx));
        min_weight=sampled_min_leaf_weight,
    )
    ess = importance_ess(local_draws, q, p)
    return gains, length(local_draws), length(unique_idx), ess
end

function diagnostic_seed(training_run, epoch, M, sampling_run, alpha, beta)
    return diagnostic_seed_base +
           100_000*training_run + 1_000*epoch + M + 10*sampling_run +
           round(Int, 10_000*alpha) + round(Int, 100_000*beta)
end

function diagnose_checkpoint(H, model, full_states, training_run, epoch)
    frozen = exact_frozen_problem(H, model, full_states)
    tree = oracle_tree(full_states, frozen)
    nodes = oracle_node_regions(tree, full_states)
    dH = size(full_states, 1)

    # Precompute oracle gain landscape at every oracle node.
    oracle_nodes = NamedTuple[]
    for node in nodes
        idx = node.state_indices
        gains = gain_landscape(
            full_states, frozen.target, frozen.probabilities, idx;
            min_weight=exact_min_leaf_weight,
        )
        bestf, bestg = best_feature_from_gains(gains)
        push!(oracle_nodes, merge(node, (
            oracle_best_feature=bestf,
            oracle_best_gain=bestg,
            oracle_probability_mass=sum(frozen.probabilities[idx]),
            oracle_state_fraction=length(idx)/dH,
        )))
    end

    @printf("\nEpoch %d: E=% .8f target RMS=%.4e oracle internal nodes=%d\n",
            epoch, real(frozen.energy), frozen.target_rms, length(oracle_nodes))

    rows = NamedTuple[]
    for M in diagnostic_sample_sizes, (alpha,beta) in mixture_values
        q = proposal_distribution(frozen, alpha, beta)
        for sampling_run in 1:nsampling_runs
            rng = MersenneTwister(diagnostic_seed(
                training_run, epoch, M, sampling_run, alpha, beta,
            ))
            draws = draw_categorical_indices(rng, q, M)

            for node in oracle_nodes
                region_mask = falses(dH)
                region_mask[node.state_indices] .= true
                sgains, nlocal, nunique, local_ess = sampled_gain_landscape(
                    full_states, frozen.target, frozen.probabilities, q,
                    draws, region_mask,
                )
                sf, sg = best_feature_from_gains(sgains)

                exact_match = sf != 0 && sf == node.oracle_best_feature
                chosen_oracle_gain = sf == 0 ? 0.0 : begin
                    ogains = gain_landscape(
                        full_states, frozen.target, frozen.probabilities,
                        node.state_indices;
                        min_weight=exact_min_leaf_weight,
                    )
                    g = ogains[sf]
                    isfinite(g) ? max(g, 0.0) : 0.0
                end
                relative_oracle_gain = node.oracle_best_gain > eps(Float64) ?
                    chosen_oracle_gain / node.oracle_best_gain : NaN

                # Correlation of gain landscapes over features valid in both.
                ogains = gain_landscape(
                    full_states, frozen.target, frozen.probabilities,
                    node.state_indices;
                    min_weight=exact_min_leaf_weight,
                )
                valid = [f for f in 1:N if isfinite(ogains[f]) && isfinite(sgains[f])]
                gain_corr = length(valid) >= 2 && std(ogains[valid]) > eps(Float64) &&
                            std(sgains[valid]) > eps(Float64) ?
                            cor(ogains[valid], sgains[valid]) : NaN

                push!(rows, (
                    J=J, h=h, J_over_h=J/h,
                    training_run=training_run, epoch=epoch,
                    M=M, sampling_run=sampling_run,
                    alpha=alpha, beta=beta,
                    born_fraction=1-alpha-beta,
                    node_index=node.node_index,
                    node_depth=node.depth,
                    node_path=node.path,
                    oracle_feature=node.oracle_best_feature,
                    sampled_feature=sf,
                    exact_split_match=exact_match,
                    oracle_best_gain=node.oracle_best_gain,
                    sampled_best_gain=sg,
                    chosen_oracle_gain=chosen_oracle_gain,
                    relative_oracle_gain=relative_oracle_gain,
                    gain_landscape_correlation=gain_corr,
                    oracle_probability_mass=node.oracle_probability_mass,
                    oracle_state_fraction=node.oracle_state_fraction,
                    local_draws=nlocal,
                    local_unique_states=nunique,
                    local_ess=local_ess,
                    local_ess_fraction=M > 0 ? local_ess/M : 0.0,
                    conditional_ess_fraction=nlocal > 0 ? local_ess/nlocal : 0.0,
                ))
            end
        end

        S = [r for r in rows if r.M==M && r.alpha==alpha && r.beta==beta]
        root = [r for r in S if r.node_depth==0]
        @printf("  M=%4d a=%.2f b=%.2f  root-match=%.3f root-RG=%.3f\n",
                M, alpha, beta,
                mean(Float64(r.exact_split_match) for r in root),
                finite_mean(r.relative_oracle_gain for r in root))
    end
    return rows
end

function run_training_trajectory(training_run, full_states)
    H = TFIMHamiltonian(N; J=J, h=h, periodic=true)
    rng = MersenneTwister(base_seed + 10_000*training_run)
    samples = Matrix{Int8}(undef, training_nsamples, N)
    @inbounds for i in eachindex(samples)
        samples[i] = rand(rng, Bool) ? Int8(1) : Int8(-1)
    end
    model = LogGBState(logamp_bias=0.0, phase_bias=0.0, use_phase=false)
    logamps = zeros(Float64, training_nsamples)
    for _ in 1:burn_in_sweeps
        GBTQuantum.sweep!(rng, model, samples, logamps)
    end

    rows = NamedTuple[]
    for epoch in 1:nepochs
        batch = vmc_batch(H, model, samples)
        yA, _ = make_targets(batch)
        weights = batch.counts

        if epoch in checkpoint_epochs
            append!(rows, diagnose_checkpoint(
                H, model, full_states, training_run, epoch,
            ))
        end

        tree = GBTQuantum.grow_tree(
            batch.states, yA, weights;
            max_depth=optimizer_max_depth,
            min_weight=optimizer_min_leaf_weight,
            min_gain=optimizer_min_gain,
        )
        pred = predict_all(tree, batch.states)
        mu = weighted_mean(pred, weights)
        if isfinite(mu) && mu != 0.0
            tree = shift_tree_leaves(tree, mu)
        end
        push!(model.logamp.trees, scale_tree(tree, eta))
        GBTQuantum.refresh_logamps!(logamps, model, samples)
        for _ in 1:sweeps_per_epoch
            GBTQuantum.sweep!(rng, model, samples, logamps)
        end
    end
    return rows
end

function print_summary(rows)
    println("\n================ SPLIT RECOVERY SUMMARY ================")
    println("epoch    M     a     b depth   split-match      <RG>   <gain-corr>  <local ESS>  <ESS/local>")
    println("------------------------------------------------------------------------------------------------")
    for epoch in sort(collect(checkpoint_epochs)), M in diagnostic_sample_sizes,
        (alpha,beta) in mixture_values, depth in 0:(oracle_depth-1)
        S = [r for r in rows if r.epoch==epoch && r.M==M &&
             r.alpha==alpha && r.beta==beta && r.node_depth==depth]
        isempty(S) && continue
        @printf("%5d %4d %5.2f %5.2f %5d      %7.3f   %7.3f      %7.3f      %8.2f      %7.3f\n",
                epoch, M, alpha, beta, depth,
                mean(Float64(r.exact_split_match) for r in S),
                finite_mean(r.relative_oracle_gain for r in S),
                finite_mean(r.gain_landscape_correlation for r in S),
                finite_mean(r.local_ess for r in S),
                finite_mean(r.conditional_ess_fraction for r in S))
    end
end

function main()
    println("\n============================================================")
    println("ORACLE SPLIT-RECOVERY EXPERIMENT")
    println("N                     = ", N)
    println("J/h                   = ", J/h)
    println("Training trajectories = ", ntraining_runs)
    println("Checkpoints           = ", sort(collect(checkpoint_epochs)))
    println("Diagnostic M          = ", diagnostic_sample_sizes)
    println("Oracle depth          = ", oracle_depth)
    println("Mixtures              = ", mixture_values)
    println("Sampling runs         = ", nsampling_runs)
    println("============================================================")

    full_states = enumerate_states(N)
    rows = NamedTuple[]
    for training_run in 1:ntraining_runs
        @printf("\nTraining trajectory %d/%d\n", training_run, ntraining_runs)
        append!(rows, run_training_trajectory(training_run, full_states))
    end

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    path = joinpath(outdir, "split_recovery.csv")
    write_namedtuple_csv(path, rows)
    print_summary(rows)
    println("\nRESULTS WRITTEN TO\n", path)
    return rows
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    SplitRecoveryExperiment.main()
end
