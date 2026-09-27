using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module BoostingStepSamplingAblationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

const N=12; const h=1.0; const J=2.0
const training_nsamples=256; const nepochs=64
const checkpoint_epochs=Set([8,32,64]); const ntraining_runs=3
const optimizer_max_depth=4; const eta_training=0.05
const burn_in_sweeps=100; const sweeps_per_epoch=1
const optimizer_min_leaf_weight=1.0; const optimizer_min_gain=0.0
const diagnostic_depth=4; const diagnostic_min_weight=1e-14
const M=1024; const repetitions=50
const exploration_eps=[0.10,0.25]
const eta_grid=[0.01,0.025,0.05,0.10,0.20]
const base_seed=1_920_000; const diagnostic_seed_base=97_000_000
const support_floor=1e-15

function exact_frozen_problem(H,model,X)
    p=exact_probabilities(model,X)
    eloc=ComplexF64[local_energy!(H,model,@view(X[i,:])) for i in axes(X,1)]
    E=sum(p.*eloc); y=-real.(eloc.-E)
    return (probabilities=p,target=y,energy=real(E),target_rms=sqrt(sum(p.*y.^2)))
end

function normalize_positive(v)
    q=max.(Float64.(v),0.0); s=sum(q)
    if !(s>eps()) || !isfinite(s) fill!(q,1/length(q)); return q end
    q./=s; q
end

function proposals(fr)
    p=fr.probabilities; y=fr.target; d=length(p); u=fill(1/d,d)
    py1=normalize_positive(p.*abs.(y)); py2=normalize_positive(p.*y.^2)
    out=[(name="born",epsilon=0.0,q=copy(p))]
    for e in exploration_eps
        push!(out,(name="born_uniform",epsilon=e,q=normalize_positive((1-e).*p.+e.*u)))
        push!(out,(name="born_abs_target",epsilon=e,q=normalize_positive((1-e).*p.+e.*py1)))
        push!(out,(name="born_sq_target",epsilon=e,q=normalize_positive((1-e).*p.+e.*py2)))
    end
    out
end

function sampled_tree(rng,X,fr,q)
    draws=draw_categorical_indices(rng,q,M); mass=Dict{Int,Float64}()
    for x in draws mass[x]=get(mass,x,0.0)+fr.probabilities[x]/max(q[x],support_floor) end
    idx=sort!(collect(keys(mass))); w=Float64[mass[x] for x in idx]
    tree=GBTQuantum.grow_tree(X[idx,:],fr.target[idx],w;
        max_depth=diagnostic_depth,min_weight=diagnostic_min_weight,min_gain=0.0)
    return tree,length(idx)
end

function full_tree(X,fr)
    GBTQuantum.grow_tree(X,fr.target,fr.probabilities;
        max_depth=diagnostic_depth,min_weight=diagnostic_min_weight,min_gain=0.0)
end

# Exact variational energy after adding eta*f(x) to the log amplitude. For TFIM
# with real positive wavefunctions, evaluate amplitudes on all basis states and
# apply H exactly through the known local connectivity.
function exact_energy_after_tree(H,model,tree,X,eta)
    n=size(X,1)
    loga=Float64[logamplitude(model,@view(X[i,:])) + eta*predict(tree,@view(X[i,:])) for i in 1:n]
    amax=maximum(loga); amp=exp.(loga.-amax)
    norm2=sum(abs2,amp)
    # TFIM diagonal term plus one-spin-flip off-diagonal -h.
    num=0.0
    # map spin bitstring to enumeration index; enumerate_states ordering is
    # binary with -1/1 spins, so build a robust dictionary rather than assume it.
    key(x)=Tuple(x)
    index=Dict{Tuple{Vararg{Int8}},Int}()
    for i in 1:n index[key(@view X[i,:])]=i end
    for i in 1:n
        x=@view X[i,:]
        diag=0.0
        for j in 1:N
            jp=(j==N ? 1 : j+1)
            diag += -J*Float64(x[j])*Float64(x[jp])
        end
        num += diag*amp[i]^2
        for j in 1:N
            xf=collect(x); xf[j] = -xf[j]
            k=index[key(xf)]
            num += (-h)*amp[i]*amp[k]
        end
    end
    return num/norm2
end

function evaluate_tree(H,model,tree,X,fr)
    deltas=Float64[]
    for e in eta_grid push!(deltas,exact_energy_after_tree(H,model,tree,X,e)-fr.energy) end
    i=argmin(deltas)
    i05=findfirst(==(0.05),eta_grid)
    return (delta_eta05=deltas[i05],best_delta=deltas[i],best_eta=eta_grid[i],all=deltas)
end

function diagnose(H,model,X,run,epoch)
    fr=exact_frozen_problem(H,model,X); rows=NamedTuple[]
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e\n",epoch,fr.energy,fr.target_rms)
    ft=full_tree(X,fr); fe=evaluate_tree(H,model,ft,X,fr)
    @printf("  FULL HILBERT: dE(eta=.05)=% .6e best dE=% .6e @ eta=%.3f\n",fe.delta_eta05,fe.best_delta,fe.best_eta)
    push!(rows,(training_run=run,epoch=epoch,method="full_hilbert",epsilon=0.0,M=length(fr.probabilities),
        energy=fr.energy,target_rms=fr.target_rms,mean_unique=Float64(length(fr.probabilities)),
        delta_eta05=fe.delta_eta05,best_delta=fe.best_delta,best_eta=fe.best_eta,repetitions=1))

    for pr in proposals(fr)
        vals=NamedTuple[]
        for rep in 1:repetitions
            methodhash=sum(Int(c) for c in codeunits(pr.name))+round(Int,1000*pr.epsilon)
            rng=MersenneTwister(diagnostic_seed_base+10_000_000*run+100_000*epoch+100*methodhash+rep)
            tree,nu=sampled_tree(rng,X,fr,pr.q); ev=evaluate_tree(H,model,tree,X,fr)
            push!(vals,(d05=ev.delta_eta05,bd=ev.best_delta,be=ev.best_eta,nu=Float64(nu)))
        end
        d05=mean(v.d05 for v in vals); bd=mean(v.bd for v in vals); be=mean(v.be for v in vals); nu=mean(v.nu for v in vals)
        improve05=mean(v.d05<0 for v in vals); improvebest=mean(v.bd<0 for v in vals)
        @printf("  %-17s eps=%4.2f dE(.05)=% .6e best dE=% .6e eta*=%.3f improve(.05)=%.2f unique=%.1f\n",
            pr.name,pr.epsilon,d05,bd,be,improve05,nu)
        push!(rows,(training_run=run,epoch=epoch,method=pr.name,epsilon=pr.epsilon,M=M,
            energy=fr.energy,target_rms=fr.target_rms,mean_unique=nu,delta_eta05=d05,
            best_delta=bd,best_eta=be,improve_eta05=improve05,improve_best=improvebest,repetitions=repetitions))
    end
    rows
end

function run_training(run,X)
    H=TFIMHamiltonian(N;J=J,h=h,periodic=true)
    rng=MersenneTwister(base_seed+10_000*run)
    samples=Matrix{Int8}(undef,training_nsamples,N)
    @inbounds for i in eachindex(samples) samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1) end
    model=LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); logamps=zeros(training_nsamples)
    for _ in 1:burn_in_sweeps GBTQuantum.sweep!(rng,model,samples,logamps) end
    rows=NamedTuple[]
    for epoch in 1:nepochs
        batch=vmc_batch(H,model,samples); yA,_=make_targets(batch); w=batch.counts
        if epoch in checkpoint_epochs append!(rows,diagnose(H,model,X,run,epoch)) end
        tree=GBTQuantum.grow_tree(batch.states,yA,w;max_depth=optimizer_max_depth,
            min_weight=optimizer_min_leaf_weight,min_gain=optimizer_min_gain)
        pred=predict_all(tree,batch.states); mu=weighted_mean(pred,w)
        if isfinite(mu)&&mu!=0 tree=shift_tree_leaves(tree,mu) end
        push!(model.logamp.trees,scale_tree(tree,eta_training))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:sweeps_per_epoch GBTQuantum.sweep!(rng,model,samples,logamps) end
    end
    rows
end

function main()
    println("\n============================================================")
    println("DIRECT BOOSTING-STEP SAMPLING ABLATION")
    println("N=$N J/h=$(J/h) runs=$ntraining_runs checkpoints=$(sort(collect(checkpoint_epochs)))")
    println("M=$M repetitions=$repetitions eps=$exploration_eps eta grid=$eta_grid")
    println("metric: exact full-Hilbert energy change after adding eta*f(x)")
    println("============================================================")
    X=enumerate_states(N); rows=NamedTuple[]
    for run in 1:ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,ntraining_runs)
        append!(rows,run_training(run,X))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"boosting_step_sampling_ablation.csv")
    write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/boosting_step_sampling_ablation.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    BoostingStepSamplingAblationExperiment.main()
end
