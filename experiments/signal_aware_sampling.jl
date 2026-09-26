using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module SignalAwareSamplingExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

# ============================================================
# SIGNAL-AWARE SAMPLING EXPERIMENT
# ============================================================
# Frozen pre-update checkpoints are generated with the real optimizer.
# At each checkpoint we compare diagnostic trees trained from M IID draws
# from
#
#   r_alpha(x) = (1-alpha) p(x) + alpha q(x),
#
# where
#   p(x) = |psi(x)|^2,
#   q(x) ∝ p(x) [u(x)-<u>_p]^2,
#   u(x) = -Re[E_L(x)-E].
#
# Because r_alpha != p for alpha>0, sampled regression uses importance
# weights p(x)/r_alpha(x), including multiplicities. All methods therefore
# target the same p-weighted regression objective.
# ============================================================

const N = 12
const h = 1.0
const J_values = [0.05, 2.00]

const training_nsamples = 256
const nepochs = 64
const checkpoint_epochs = Set([1, 8, 32, 64])
const ntraining_runs = 3

const optimizer_max_depth = 4
const eta = 0.05
const burn_in_sweeps = 100
const sweeps_per_epoch = 1
const optimizer_min_leaf_weight = 1.0
const optimizer_min_gain = 0.0

const diagnostic_sample_sizes = [64, 256, 1024]
const diagnostic_depths = [4, 8]
const nsampling_runs = 20
const alpha_values = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]

const exact_min_leaf_weight = 1e-14
const exact_min_gain = 0.0
const sampled_min_leaf_weight = 1e-14
const sampled_min_gain = 0.0

const base_seed = 910_000
const diagnostic_seed_base = 8_410_000

function exact_frozen_problem(H::TFIMHamiltonian, model::LogGBState, states::Matrix{Int8})
    p = exact_probabilities(model, states)
    eloc = ComplexF64[local_energy!(H, model, @view(states[s, :])) for s in axes(states, 1)]
    E = sum(p .* eloc)
    target = -real.(eloc .- E)
    target_mean = sum(p .* target)
    centered = target .- target_mean
    signal_density = p .* centered.^2
    signal_total = sum(signal_density)
    q = signal_total > eps(Float64) ? signal_density / signal_total : copy(p)
    target_rms = sqrt(sum(p .* target.^2))
    PR = 1.0 / sum(abs2, p)
    return (probabilities=p, local_energy=eloc, energy=E, target=target,
            target_mean=target_mean, target_rms=target_rms, q=q,
            signal_total=signal_total, participation_ratio=PR,
            participation_fraction=PR/length(p))
end

function fit_exact_tree(full_states, frozen, depth)
    tree = GBTQuantum.grow_tree(full_states, frozen.target, frozen.probabilities;
                                max_depth=depth, min_weight=exact_min_leaf_weight,
                                min_gain=exact_min_gain)
    pred = predict_all(tree, full_states)
    mu = weighted_mean(pred, frozen.probabilities)
    if isfinite(mu) && mu != 0.0
        tree = shift_tree_leaves(tree, mu)
        pred = predict_all(tree, full_states)
    end
    return tree, pred
end

function proposal_distribution(frozen, alpha::Float64)
    r = (1.0-alpha) .* frozen.probabilities .+ alpha .* frozen.q
    s = sum(r)
    s <= 0 && error("Proposal distribution has zero mass")
    r ./= s
    return r
end

function coverage_metrics(unique_idx, frozen, dH)
    return length(unique_idx)/dH,
           sum(frozen.probabilities[unique_idx]),
           sum(frozen.q[unique_idx])
end

function importance_diagnostics(indices, r, p)
    w = Float64[p[i]/r[i] for i in indices]
    sw = sum(w)
    sw2 = sum(abs2, w)
    ess = sw2 > 0 ? sw^2/sw2 : 0.0
    return ess, ess/length(indices), maximum(w), std(w)/max(mean(w), eps(Float64))
end

function fit_importance_tree(full_states, frozen, indices, r, depth)
    unique_idx, counts = compress_indices(indices)
    X = full_states[unique_idx, :]
    y = frozen.target[unique_idx]
    weights = counts .* frozen.probabilities[unique_idx] ./ r[unique_idx]
    tree = GBTQuantum.grow_tree(X, y, weights;
                                max_depth=depth,
                                min_weight=sampled_min_leaf_weight,
                                min_gain=sampled_min_gain)
    sample_pred = predict_all(tree, X)
    mu = weighted_mean(sample_pred, weights)
    isfinite(mu) && mu != 0.0 && (tree = shift_tree_leaves(tree, mu))
    return tree, predict_all(tree, full_states), unique_idx
end

function evaluate_tree(pred, frozen)
    return weighted_r2(frozen.target, pred, frozen.probabilities),
           ordinary_r2(frozen.target, pred)
end

function diagnose(H, model, full_states, J, training_run, epoch)
    frozen = exact_frozen_problem(H, model, full_states)
    dH = size(full_states, 1)
    @printf("\n  Frozen epoch %2d: E=% .8f  target RMS=%.4e  PR/H=%.6f\n",
            epoch, real(frozen.energy), frozen.target_rms, frozen.participation_fraction)

    oracle = Dict{Int,NamedTuple}()
    for depth in diagnostic_depths
        tree, pred = fit_exact_tree(full_states, frozen, depth)
        Rp, RH = evaluate_tree(pred, frozen)
        oracle[depth] = (Rp=Rp, RH=RH, leaves=tree_leaf_count(tree))
        @printf("    oracle depth=%d: R²p=% .6f  R²H=% .6f\n", depth, Rp, RH)
    end

    rows = NamedTuple[]
    for M in diagnostic_sample_sizes, alpha in alpha_values
        r = proposal_distribution(frozen, alpha)
        for sampling_run in 1:nsampling_runs
            seed = diagnostic_seed_base + round(Int, 100_000*J) + 100_000*training_run +
                   1_000*epoch + 10*sampling_run + M + round(Int, 10_000*alpha)
            idx = draw_categorical_indices(MersenneTwister(seed), r, M)
            unique_idx, _ = compress_indices(idx)
            CH, Cp, Cq = coverage_metrics(unique_idx, frozen, dH)
            iess, iess_fraction, wmax, wcv = importance_diagnostics(idx, r, frozen.probabilities)
            for depth in diagnostic_depths
                tree, pred, _ = fit_importance_tree(full_states, frozen, idx, r, depth)
                Rp, RH = evaluate_tree(pred, frozen)
                push!(rows, (
                    J=J, h=h, J_over_h=J/h, training_run=training_run, epoch=epoch,
                    M=M, sampling_run=sampling_run, alpha=alpha, tree_depth=depth,
                    frozen_energy=real(frozen.energy), frozen_target_rms=frozen.target_rms,
                    participation_fraction=frozen.participation_fraction,
                    unique_states=length(unique_idx), hilbert_coverage=CH,
                    probability_coverage=Cp, signal_coverage=Cq,
                    importance_ess=iess, importance_ess_fraction=iess_fraction,
                    importance_weight_max=wmax, importance_weight_cv=wcv,
                    exact_Rp=oracle[depth].Rp, sampled_Rp=Rp,
                    sampling_penalty_Rp=oracle[depth].Rp-Rp,
                    exact_RH=oracle[depth].RH, sampled_RH=RH,
                    sampling_penalty_RH=oracle[depth].RH-RH,
                    exact_leaf_count=oracle[depth].leaves,
                    sampled_leaf_count=tree_leaf_count(tree)))
            end
        end
        S = [r0 for r0 in rows if r0.M == M && r0.alpha == alpha && r0.tree_depth == 4]
        @printf("    M=%4d alpha=%4.2f d=4: <Cp>=%.3f <Cq>=%.3f <IESS/M>=%.3f <R²p>=%.3f <penalty>=%.3f\n",
                M, alpha,
                mean(x.probability_coverage for x in S),
                mean(x.signal_coverage for x in S),
                mean(x.importance_ess_fraction for x in S),
                mean(x.sampled_Rp for x in S),
                mean(x.sampling_penalty_Rp for x in S))
    end
    return rows
end

function run_training_trajectory(J, training_run, full_states)
    H = TFIMHamiltonian(N; J=J, h=h, periodic=true)
    rng = MersenneTwister(base_seed + round(Int, 100_000*J) + 10_000*training_run)
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
            append!(rows, diagnose(H, model, full_states, J, training_run, epoch))
        end

        tree = GBTQuantum.grow_tree(batch.states, yA, weights;
                                    max_depth=optimizer_max_depth,
                                    min_weight=optimizer_min_leaf_weight,
                                    min_gain=optimizer_min_gain)
        train_pred = predict_all(tree, batch.states)
        mu = weighted_mean(train_pred, weights)
        isfinite(mu) && mu != 0.0 && (tree = shift_tree_leaves(tree, mu))
        push!(model.logamp.trees, scale_tree(tree, eta))
        GBTQuantum.refresh_logamps!(logamps, model, samples)
        for _ in 1:sweeps_per_epoch
            GBTQuantum.sweep!(rng, model, samples, logamps)
        end
    end
    return rows
end

function print_summary(rows)
    println("\n\n================ SIGNAL-AWARE SAMPLING SUMMARY ================")
    println(" J/h epoch    M alpha  d      <Cp>      <Cq>   <IESS/M>     <R²p>    penalty")
    println("--------------------------------------------------------------------------------")
    for J in J_values, epoch in sort(collect(checkpoint_epochs)), M in diagnostic_sample_sizes,
        alpha in alpha_values, depth in diagnostic_depths
        S = [r for r in rows if r.J == J && r.epoch == epoch && r.M == M &&
             r.alpha == alpha && r.tree_depth == depth]
        isempty(S) && continue
        @printf("%4.2f %4d %4d %5.2f  %d   %7.3f   %7.3f    %7.3f   %8.3f   %8.3f\n",
                J/h, epoch, M, alpha, depth,
                finite_mean(r.probability_coverage for r in S),
                finite_mean(r.signal_coverage for r in S),
                finite_mean(r.importance_ess_fraction for r in S),
                finite_mean(r.sampled_Rp for r in S),
                finite_mean(r.sampling_penalty_Rp for r in S))
    end
end

function main()
    dH = 1 << N
    println("\n============================================================")
    println("SIGNAL-AWARE SAMPLING EXPERIMENT")
    println("N                     = ", N)
    println("Hilbert dimension     = ", dH)
    println("J/h values            = ", J_values)
    println("Training trajectories = ", ntraining_runs)
    println("Checkpoints           = ", sort(collect(checkpoint_epochs)))
    println("Diagnostic M          = ", diagnostic_sample_sizes)
    println("Diagnostic depths     = ", diagnostic_depths)
    println("alpha values          = ", alpha_values)
    println("Sampling runs         = ", nsampling_runs)
    println("============================================================")

    full_states = enumerate_states(N)
    all_rows = NamedTuple[]
    for J in J_values
        @printf("\n================ J/h = %.4f ================\n", J/h)
        for training_run in 1:ntraining_runs
            @printf("\nTraining trajectory %d/%d\n", training_run, ntraining_runs)
            append!(all_rows, run_training_trajectory(J, training_run, full_states))
        end
    end

    output_dir = joinpath(@__DIR__, "results")
    mkpath(output_dir)
    csv_path = joinpath(output_dir, "signal_aware_sampling.csv")
    write_namedtuple_csv(csv_path, all_rows)
    print_summary(all_rows)
    println("\nRESULTS WRITTEN TO\n", csv_path)
    return all_rows
end

end # module SignalAwareSamplingExperiment

if abspath(PROGRAM_FILE) == @__FILE__
    SignalAwareSamplingExperiment.main()
end
