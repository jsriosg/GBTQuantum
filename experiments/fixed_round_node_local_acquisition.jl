using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module FixedRoundNodeLocalAcquisitionExperiment

include(joinpath(@__DIR__, "sequential_node_local_acquisition.jl"))
const S = SequentialNodeLocalAcquisitionExperiment
const G = S.G
const U = S.ExperimentUtils

using GBTQuantum
using Random
using Statistics
using Printf

const rounds_sweep = [0, 1, 2, 3, 4]
const repetitions_fixed = 20
const seed_offset = 83_000_000

function grow_fixed_rounds!(rng,X,y,p,node_idx,path,depth,obs,decisions,cost,R)
    depth >= G.oracle_depth && return
    length(node_idx) <= 1 && return

    ninherited = count(o -> (o.x in node_idx), obs)
    if ninherited < S.min_pilot_in_node
        pv = G.conditional_p(p,node_idx)
        add = max(S.local_pilot_target - ninherited, 0)
        S.draw_obs!(rng,obs,pv,add,path)
        cost[] += add
    end

    est = S.estimated_landscape(X,y,p,node_idx,obs)
    f1,f2,_,_ = G.top_two(est.g)
    f1 == 0 && return

    for _ in 1:R
        f2 == 0 && break
        q = S.acquisition_q(X,y,p,node_idx,obs,f1,f2)
        S.draw_obs!(rng,obs,q,S.acquisition_batch,path)
        cost[] += S.acquisition_batch
        est = S.estimated_landscape(X,y,p,node_idx,obs)
        f1,f2,_,_ = G.top_two(est.g)
        f1 == 0 && return
    end

    decisions[path] = f1
    L,Ridx = G.split_indices(X,node_idx,f1)
    grow_fixed_rounds!(rng,X,y,p,L,path*"L",depth+1,obs,decisions,cost,R)
    grow_fixed_rounds!(rng,X,y,p,Ridx,path*"R",depth+1,obs,decisions,cost,R)
end

function fixed_round_trial(rng,X,fr,R)
    obs = G.Obs[]
    qroot = copy(fr.probabilities)
    S.draw_obs!(rng,obs,qroot,G.pilot_root,"")
    cost = Ref(G.pilot_root)
    decisions = Dict{String,Int}()
    grow_fixed_rounds!(rng,X,fr.target,fr.probabilities,
                       collect(axes(X,1)),"",0,obs,decisions,cost,R)
    return decisions,cost[]
end

function diagnose_fixed(H,model,X,run,epoch)
    fr = G.exact_frozen_problem(H,model,X)
    _,oracle = G.exact_oracle(fr,X)
    rows = NamedTuple[]
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e oracle internal nodes=%d\n",
            epoch,real(fr.energy),fr.target_rms,length(oracle))

    for R in rounds_sweep
        correct = zeros(Int,G.oracle_depth)
        eligible = zeros(Int,G.oracle_depth)
        allok = 0
        costs = Float64[]
        for rep in 1:repetitions_fixed
            rng = MersenneTwister(seed_offset + 10_000_000*run + 100_000*epoch + 10_000*R + rep)
            dec,cost = fixed_round_trial(rng,X,fr,R)
            e,c = S.conditional_depth_score(dec,oracle)
            eligible .+= e; correct .+= c
            allok += S.full_tree_score(dec,oracle)
            push!(costs,cost)
        end
        rec = [eligible[d]>0 ? correct[d]/eligible[d] : NaN for d in 1:G.oracle_depth]
        full = allok/repetitions_fixed
        mc = mean(costs)
        @printf("  rounds=%d: conditional depth recovery=%s full-tree=%.3f cost=%.1f\n",
                R,string(round.(rec,digits=3)),full,mc)
        for d in 1:G.oracle_depth
            push!(rows,(training_run=run,epoch=epoch,energy=real(fr.energy),target_rms=fr.target_rms,
                rounds=R,depth=d-1,correct=correct[d],eligible=eligible[d],
                conditional_recovery=rec[d],full_tree_recovery=full,
                mean_evaluations=mc,repetitions=repetitions_fixed,
                root_pilot=G.pilot_root,acquisition_batch=S.acquisition_batch,
                epsilon=G.if_born_epsilon))
        end
    end
    return rows
end

function run_training_fixed(run,X)
    H = TFIMHamiltonian(G.N;J=G.J,h=G.h,periodic=true)
    rng = MersenneTwister(G.base_seed + 10_000*run)
    samples = Matrix{Int8}(undef,G.training_nsamples,G.N)
    @inbounds for i in eachindex(samples)
        samples[i] = rand(rng,Bool) ? Int8(1) : Int8(-1)
    end
    model = LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    logamps = zeros(G.training_nsamples)
    for _ in 1:G.burn_in_sweeps
        GBTQuantum.sweep!(rng,model,samples,logamps)
    end
    rows = NamedTuple[]
    for epoch in 1:G.nepochs
        batch = vmc_batch(H,model,samples)
        yA,_ = make_targets(batch); w=batch.counts
        if epoch in G.checkpoint_epochs
            append!(rows,diagnose_fixed(H,model,X,run,epoch))
        end
        tree = GBTQuantum.grow_tree(batch.states,yA,w;
            max_depth=G.optimizer_max_depth,min_weight=G.optimizer_min_leaf_weight,
            min_gain=G.optimizer_min_gain)
        pred = U.predict_all(tree,batch.states); mu=U.weighted_mean(pred,w)
        if isfinite(mu) && mu != 0
            tree=U.shift_tree_leaves(tree,mu)
        end
        push!(model.logamp.trees,U.scale_tree(tree,G.eta))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:G.sweeps_per_epoch
            GBTQuantum.sweep!(rng,model,samples,logamps)
        end
    end
    return rows
end

function main()
    println("\n============================================================")
    println("FIXED-ROUND NODE-LOCAL IF ACQUISITION")
    println("N=$(G.N) J/h=$(G.J/G.h) runs=$(G.ntraining_runs) checkpoints=$(sort(collect(G.checkpoint_epochs)))")
    println("root pilot=$(G.pilot_root), batch=$(S.acquisition_batch), rounds sweep=$rounds_sweep")
    println("repetitions=$repetitions_fixed epsilon=$(G.if_born_epsilon)")
    println("purpose: empirical recovery-vs-cost curve, without Z stopping")
    println("============================================================")
    X = U.enumerate_states(G.N)
    rows = NamedTuple[]
    for run in 1:G.ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,G.ntraining_runs)
        append!(rows,run_training_fixed(run,X))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"fixed_round_node_local_acquisition.csv")
    U.write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/fixed_round_node_local_acquisition.csv")
end

export main

end

if abspath(PROGRAM_FILE) == @__FILE__
    FixedRoundNodeLocalAcquisitionExperiment.main()
end
