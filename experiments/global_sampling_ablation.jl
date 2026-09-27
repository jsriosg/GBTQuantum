using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module GlobalSamplingAblationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

# Isolate sampling from tree representation:
# every sampled method trains the SAME ordinary weighted-MSE regression tree.
# Only the proposal q(x) changes. Importance weights p(x)/q(x) preserve the
# desired Born-weighted training objective.

const N = 12
const h = 1.0
const J = 2.0
const training_nsamples = 256
const nepochs = 64
const checkpoint_epochs = Set([8,32,64])
const ntraining_runs = 3
const optimizer_max_depth = 4
const eta = 0.05
const burn_in_sweeps = 100
const sweeps_per_epoch = 1
const optimizer_min_leaf_weight = 1.0
const optimizer_min_gain = 0.0

const diagnostic_depth = 4
const diagnostic_min_weight = 1e-14
const sample_sizes = [256,512,1024]
const repetitions = 50
const exploration_eps = [0.05,0.10,0.25]
const base_seed = 1_920_000
const diagnostic_seed_base = 94_000_000
const support_floor = 1e-15

function exact_frozen_problem(H,model,X)
    p = exact_probabilities(model,X)
    eloc = ComplexF64[local_energy!(H,model,@view(X[i,:])) for i in axes(X,1)]
    E = sum(p .* eloc)
    y = -real.(eloc .- E)
    return (probabilities=p,target=y,energy=E,target_rms=sqrt(sum(p.*y.^2)))
end

function normalize_positive(v)
    q=max.(Float64.(v),0.0)
    s=sum(q)
    if !(s>eps()) || !isfinite(s)
        fill!(q,1/length(q)); return q
    end
    q./=s
    return q
end

function proposals(fr)
    p=fr.probabilities; y=fr.target; d=length(p)
    u=fill(1/d,d)
    py1=normalize_positive(p .* abs.(y))
    py2=normalize_positive(p .* y.^2)
    out=NamedTuple[]
    push!(out,(name="born",epsilon=0.0,q=copy(p)))
    for e in exploration_eps
        push!(out,(name="born_uniform",epsilon=e,q=normalize_positive((1-e).*p .+ e.*u)))
        push!(out,(name="born_abs_target",epsilon=e,q=normalize_positive((1-e).*p .+ e.*py1)))
        push!(out,(name="born_sq_target",epsilon=e,q=normalize_positive((1-e).*p .+ e.*py2)))
    end
    return out
end

function train_from_proposal(rng,X,fr,q,M)
    draws=draw_categorical_indices(rng,q,M)
    # Sum importance mass for repeated configurations. The irrelevant global
    # 1/M factor is omitted because tree gains are homogeneous in weights.
    mass=Dict{Int,Float64}()
    for x in draws
        mass[x]=get(mass,x,0.0)+fr.probabilities[x]/max(q[x],support_floor)
    end
    idx=sort!(collect(keys(mass)))
    w=Float64[mass[x] for x in idx]
    tree=GBTQuantum.grow_tree(X[idx,:],fr.target[idx],w;
        max_depth=diagnostic_depth,min_weight=diagnostic_min_weight,min_gain=0.0)
    return tree,length(idx)
end

function exact_tree(X,fr)
    GBTQuantum.grow_tree(X,fr.target,fr.probabilities;
        max_depth=diagnostic_depth,min_weight=diagnostic_min_weight,min_gain=0.0)
end

function tree_metrics(tree,X,fr,oracle)
    pred=predict_all(tree,X)
    r2p=weighted_r2(fr.target,pred,fr.probabilities)
    r2u=ordinary_r2(fr.target,pred)
    rmsep=sqrt(sum(fr.probabilities .* (fr.target.-pred).^2))
    # Root agreement is a clean split diagnostic that does not mix descendant
    # regions after an ancestor mistake.
    f=Int(tree.nodes[1].feature); fo=Int(oracle.nodes[1].feature)
    root=(f==fo)
    return (r2_born=r2p,r2_uniform=r2u,rmse_born=rmsep,root_match=root,
            nodes=length(tree.nodes),leaves=tree_leaf_count(tree))
end

function diagnose(H,model,X,run,epoch)
    fr=exact_frozen_problem(H,model,X)
    oracle=exact_tree(X,fr)
    om=tree_metrics(oracle,X,fr,oracle)
    rows=NamedTuple[]
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e\n",epoch,real(fr.energy),fr.target_rms)
    @printf("  FULL HILBERT: Born-R2=%.4f uniform-R2=%.4f RMSE=%.4e nodes=%d\n",
            om.r2_born,om.r2_uniform,om.rmse_born,om.nodes)
    push!(rows,(training_run=run,epoch=epoch,method="full_hilbert",epsilon=0.0,M=length(fr.probabilities),
        energy=real(fr.energy),target_rms=fr.target_rms,r2_born=om.r2_born,r2_uniform=om.r2_uniform,
        rmse_born=om.rmse_born,root_recovery=1.0,mean_unique=Float64(length(fr.probabilities)),
        nodes=Float64(om.nodes),leaves=Float64(om.leaves),repetitions=1))

    for M in sample_sizes
        for pr in proposals(fr)
            vals=NamedTuple[]
            for rep in 1:repetitions
                # common deterministic seed family for fair method comparisons
                methodhash=sum(Int(c) for c in codeunits(pr.name)) + round(Int,1000*pr.epsilon)
                rng=MersenneTwister(diagnostic_seed_base+10_000_000*run+100_000*epoch+10_000*M+100*methodhash+rep)
                tree,nu=train_from_proposal(rng,X,fr,pr.q,M)
                m=tree_metrics(tree,X,fr,oracle)
                push!(vals,(r2_born=m.r2_born,r2_uniform=m.r2_uniform,rmse_born=m.rmse_born,
                    root=Float64(m.root_match),unique=Float64(nu),nodes=Float64(m.nodes),leaves=Float64(m.leaves)))
            end
            mr2p=mean(v.r2_born for v in vals); mr2u=mean(v.r2_uniform for v in vals)
            mrmse=mean(v.rmse_born for v in vals); rr=mean(v.root for v in vals)
            mu=mean(v.unique for v in vals); mn=mean(v.nodes for v in vals); ml=mean(v.leaves for v in vals)
            @printf("  M=%4d %-17s eps=%4.2f Born-R2=% .4f uniform-R2=% .4f root=%.3f unique=%.1f\n",
                    M,pr.name,pr.epsilon,mr2p,mr2u,rr,mu)
            push!(rows,(training_run=run,epoch=epoch,method=pr.name,epsilon=pr.epsilon,M=M,
                energy=real(fr.energy),target_rms=fr.target_rms,r2_born=mr2p,r2_uniform=mr2u,
                rmse_born=mrmse,root_recovery=rr,mean_unique=mu,nodes=mn,leaves=ml,repetitions=repetitions))
        end
    end
    return rows
end

function run_training(run,X)
    H=TFIMHamiltonian(N;J=J,h=h,periodic=true)
    rng=MersenneTwister(base_seed+10_000*run)
    samples=Matrix{Int8}(undef,training_nsamples,N)
    @inbounds for i in eachindex(samples)
        samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1)
    end
    model=LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    logamps=zeros(training_nsamples)
    for _ in 1:burn_in_sweeps
        GBTQuantum.sweep!(rng,model,samples,logamps)
    end
    rows=NamedTuple[]
    for epoch in 1:nepochs
        batch=vmc_batch(H,model,samples)
        yA,_=make_targets(batch); w=batch.counts
        if epoch in checkpoint_epochs
            append!(rows,diagnose(H,model,X,run,epoch))
        end
        tree=GBTQuantum.grow_tree(batch.states,yA,w;
            max_depth=optimizer_max_depth,min_weight=optimizer_min_leaf_weight,min_gain=optimizer_min_gain)
        pred=predict_all(tree,batch.states); mu=weighted_mean(pred,w)
        if isfinite(mu)&&mu!=0 tree=shift_tree_leaves(tree,mu) end
        push!(model.logamp.trees,scale_tree(tree,eta))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:sweeps_per_epoch
            GBTQuantum.sweep!(rng,model,samples,logamps)
        end
    end
    return rows
end

function main()
    println("\n============================================================")
    println("GLOBAL TRAINING-SAMPLE ABLATION")
    println("N=$N J/h=$(J/h) runs=$ntraining_runs checkpoints=$(sort(collect(checkpoint_epochs)))")
    println("M=$sample_sizes repetitions=$repetitions exploration eps=$exploration_eps")
    println("Same ordinary weighted-MSE tree for every method")
    println("Proposals: Born, Born+uniform, Born+p|y|, Born+p*y^2, plus full Hilbert ceiling")
    println("============================================================")
    X=enumerate_states(N); rows=NamedTuple[]
    for run in 1:ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,ntraining_runs)
        append!(rows,run_training(run,X))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"global_sampling_ablation.csv")
    write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/global_sampling_ablation.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    GlobalSamplingAblationExperiment.main()
end
