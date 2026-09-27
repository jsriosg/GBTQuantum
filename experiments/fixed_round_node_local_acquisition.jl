using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module FixedRoundNodeLocalAcquisitionExperiment

# Reuse the corrected node-local machinery (including ancestor->descendant
# conditional importance weights) from the sequential experiment.
include(joinpath(@__DIR__, "sequential_node_local_acquisition.jl"))
using .SequentialNodeLocalAcquisitionExperiment
const S = SequentialNodeLocalAcquisitionExperiment

using GBTQuantum
using Random
using Statistics
using Printf

const rounds_sweep = [0, 1, 2, 3, 4]
const repetitions_fixed = 20
const seed_offset = 83_000_000

# Force exactly R IF acquisition rounds at every splittable node. This removes
# the Z stopping rule and directly measures the marginal value/cost of each
# additional acquisition round.
function fixed_round_trial(rng, X, fr, R)
    obs = S.Obs[]
    qroot = copy(fr.probabilities)
    S.draw_obs!(rng, obs, qroot, S.pilot_root, "")
    cost = Ref(S.pilot_root)
    decisions = Dict{String,Int}()
    S.grow_fixed_rounds!(rng, X, fr.target, fr.probabilities,
                         collect(axes(X,1)), "", 0, obs, decisions, cost, R)
    return decisions, cost[]
end

function diagnose_fixed(H, model, X, run, epoch)
    fr = S.exact_frozen_problem(H, model, X)
    _, oracle = S.exact_oracle(fr, X)
    rows = NamedTuple[]

    @printf("\nEpoch %d: E=% .8f target RMS=%.4e oracle internal nodes=%d\n",
            epoch, real(fr.energy), fr.target_rms, length(oracle))

    for R in rounds_sweep
        correct = zeros(Int, S.oracle_depth)
        eligible = zeros(Int, S.oracle_depth)
        allok = 0
        costs = Float64[]

        for rep in 1:repetitions_fixed
            rng = MersenneTwister(seed_offset + 10_000_000*run + 100_000*epoch + 10_000*R + rep)
            dec, cost = fixed_round_trial(rng, X, fr, R)
            score = S.score_conditional_by_depth(dec, oracle)
            correct .+= score.correct
            eligible .+= score.eligible
            allok += score.all
            push!(costs, cost)
        end

        rec = [eligible[d] > 0 ? correct[d]/eligible[d] : NaN for d in 1:S.oracle_depth]
        full = allok/repetitions_fixed
        mc = mean(costs)
        @printf("  rounds=%d: conditional depth recovery=%s full-tree=%.3f cost=%.1f\n",
                R, string(round.(rec,digits=3)), full, mc)

        for d in 1:S.oracle_depth
            push!(rows, (training_run=run, epoch=epoch, energy=real(fr.energy),
                target_rms=fr.target_rms, rounds=R, depth=d-1,
                correct=correct[d], eligible=eligible[d],
                conditional_recovery=rec[d], full_tree_recovery=full,
                mean_evaluations=mc, repetitions=repetitions_fixed,
                root_pilot=S.pilot_root, acquisition_batch=S.acquisition_batch,
                epsilon=S.if_born_epsilon))
        end
    end
    return rows
end

function run_training_fixed(run, X)
    H = TFIMHamiltonian(S.N; J=S.J, h=S.h, periodic=true)
    rng = MersenneTwister(S.base_seed + 10_000*run)
    samples = Matrix{Int8}(undef, S.training_nsamples, S.N)
    @inbounds for i in eachindex(samples)
        samples[i] = rand(rng, Bool) ? Int8(1) : Int8(-1)
    end
    model = LogGBState(logamp_bias=0.0, phase_bias=0.0, use_phase=false)
    logamps = zeros(S.training_nsamples)
    for _ in 1:S.burn_in_sweeps
        GBTQuantum.sweep!(rng, model, samples, logamps)
    end
    rows = NamedTuple[]
    for epoch in 1:S.nepochs
        batch = vmc_batch(H, model, samples)
        yA, _ = make_targets(batch); w = batch.counts
        if epoch in S.checkpoint_epochs
            append!(rows, diagnose_fixed(H, model, X, run, epoch))
        end
        tree = GBTQuantum.grow_tree(batch.states, yA, w;
            max_depth=S.optimizer_max_depth,
            min_weight=S.optimizer_min_leaf_weight,
            min_gain=S.optimizer_min_gain)
        pred = predict_all(tree, batch.states); mu = weighted_mean(pred, w)
        if isfinite(mu) && mu != 0
            tree = shift_tree_leaves(tree, mu)
        end
        push!(model.logamp.trees, scale_tree(tree, S.eta))
        GBTQuantum.refresh_logamps!(logamps, model, samples)
        for _ in 1:S.sweeps_per_epoch
            GBTQuantum.sweep!(rng, model, samples, logamps)
        end
    end
    return rows
end

function main()
    println("\n============================================================")
    println("FIXED-ROUND NODE-LOCAL IF ACQUISITION")
    println("N=$(S.N) J/h=$(S.J/S.h) runs=$(S.ntraining_runs) checkpoints=$(sort(collect(S.checkpoint_epochs)))")
    println("root pilot=$(S.pilot_root), batch=$(S.acquisition_batch), rounds sweep=$rounds_sweep")
    println("repetitions=$repetitions_fixed epsilon=$(S.if_born_epsilon)")
    println("purpose: empirical recovery-vs-cost curve, without Z stopping")
    println("============================================================")
    X = enumerate_states(S.N)
    rows = NamedTuple[]
    for run in 1:S.ntraining_runs
        @printf("\nTraining trajectory %d/%d\n", run, S.ntraining_runs)
        append!(rows, run_training_fixed(run, X))
    end
    outdir = joinpath(@__DIR__, "results"); mkpath(outdir)
    path = joinpath(outdir, "fixed_round_node_local_acquisition.csv")
    write_namedtuple_csv(path, rows)
    println("\nResults written to experiments/results/fixed_round_node_local_acquisition.csv")
end

export main

end

if abspath(PROGRAM_FILE) == @__FILE__
    FixedRoundNodeLocalAcquisitionExperiment.main()
end
