using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module ReweightedArmijoValidationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "newton_step_validation.jl"))
const NV = NewtonStepValidationExperiment

const armijo_alpha=0.10
const backtrack_beta=0.50
const max_backtracks=12
const eta_cap=0.40
const observable_M=1024
const observable_repetitions=50

# Draw an independent Born observable sample. Repetitions are represented by
# categorical draws from the exact frozen p only for this N=12 validation.
function observable_draw(rng,fr)
    NV.draw_categorical_indices(rng,fr.probabilities,observable_M)
end

# Self-normalized reweighting from p0 to p_eta. For each sampled x,
# E_loc^eta(x)=sum_x' H_xx' psi(x')/psi(x) exp(eta[f(x')-f(x)]).
function reweighted_energy(fr,X,flips,f,idx,eta)
    vals=Float64[]; logw=Float64[]
    for i in idx
        diag=0.0
        for j in 1:NV.N
            jp=(j==NV.N ? 1 : j+1)
            diag += -NV.J*Float64(X[i,j])*Float64(X[i,jp])
        end
        el=diag
        for j in 1:NV.N
            k=flips[i,j]
            ratio=exp(fr.logamp[k]-fr.logamp[i])
            el += -NV.h*ratio*exp(eta*(f[k]-f[i]))
        end
        push!(vals,el); push!(logw,2eta*f[i])
    end
    m=maximum(logw); w=exp.(logw .- m)
    return sum(w.*vals)/sum(w)
end

function mc_armijo(fr,X,flips,f,idx)
    g,c=NV.analytic_derivatives(fr,X,flips,f)
    if !(isfinite(g)&&g<0)
        return (eta=0.0,evals=0,accepted=false)
    end
    eta=(isfinite(c)&&c>0) ? min(-g/c,eta_cap) : min(NV.eta_training,eta_cap)
    # Use the same observable sample for E0 and every candidate: common random
    # numbers reduce noise in the energy difference and Armijo decision.
    E0hat=reweighted_energy(fr,X,flips,f,idx,0.0)
    for k in 0:max_backtracks
        Ehat=reweighted_energy(fr,X,flips,f,idx,eta)
        if isfinite(Ehat) && Ehat <= E0hat + armijo_alpha*eta*g
            return (eta=eta,evals=k+1,accepted=true)
        end
        eta*=backtrack_beta
    end
    return (eta=0.0,evals=max_backtracks+1,accepted=false)
end

function exact_armijo(fr,X,flips,f)
    g,c=NV.analytic_derivatives(fr,X,flips,f)
    if !(isfinite(g)&&g<0) return (eta=0.0,evals=0,accepted=false) end
    eta=(isfinite(c)&&c>0) ? min(-g/c,eta_cap) : min(NV.eta_training,eta_cap)
    for k in 0:max_backtracks
        E=NV.exact_energy_eta(fr,X,flips,f,eta)
        if isfinite(E) && E <= fr.energy + armijo_alpha*eta*g
            return (eta=eta,evals=k+1,accepted=true)
        end
        eta*=backtrack_beta
    end
    return (eta=0.0,evals=max_backtracks+1,accepted=false)
end

function evaluate_method(fr,X,flips,f,rng)
    oracle=exact_armijo(fr,X,flips,f)
    etaopt,dopt=NV.exact_line_search(fr,X,flips,f)
    vals=NamedTuple[]
    for r in 1:observable_repetitions
        idx=observable_draw(rng,fr)
        mc=mc_armijo(fr,X,flips,f,idx)
        dtrue=NV.exact_energy_eta(fr,X,flips,f,mc.eta)-fr.energy
        push!(vals,(eta=mc.eta,evals=Float64(mc.evals),accepted=Float64(mc.accepted),
                    dtrue=dtrue,same=Float64(isapprox(mc.eta,oracle.eta;atol=1e-12))))
    end
    return (eta_oracle=oracle.eta,delta_oracle=NV.exact_energy_eta(fr,X,flips,f,oracle.eta)-fr.energy,
            eta_exact=etaopt,delta_exact=dopt,eta_mc=mean(v.eta for v in vals),
            delta_mc_true=mean(v.dtrue for v in vals),mc_evals=mean(v.evals for v in vals),
            mc_accept=mean(v.accepted for v in vals),same_eta=mean(v.same for v in vals),
            safe_fraction=mean(v.dtrue<=0 for v in vals),regret=mean(v.dtrue-dopt for v in vals))
end

function diagnose(H,model,X,flips,run,epoch)
    fr=NV.exact_frozen_problem(H,model,X); rows=NamedTuple[]
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e\n",epoch,fr.energy,fr.target_rms)
    methods=Any[("full_hilbert",0.0,NV.full_tree(X,fr),4096.0)]
    for pr in NV.proposals(fr)
        rngtree=MersenneTwister(NV.diagnostic_seed_base+10_000_000*run+100_000*epoch+round(Int,1000*pr.epsilon)+sum(codeunits(pr.name)))
        tree,nu=NV.sampled_tree(rngtree,X,fr,pr.q)
        push!(methods,(pr.name,pr.epsilon,tree,Float64(nu)))
    end
    for (name,eps,tree,nu) in methods
        f=NV.centered_predictions(tree,X,fr.probabilities)
        rng=MersenneTwister(301_000_000+10_000_000*run+100_000*epoch+sum(codeunits(name)))
        ev=evaluate_method(fr,X,flips,f,rng)
        @printf("  %-15s oracle eta=%.4f | MC eta=%.4f same=%.2f safe=%.2f evals=%.2f | true dE=% .3e exact*=% .3e regret=%.2e\n",
            name,ev.eta_oracle,ev.eta_mc,ev.same_eta,ev.safe_fraction,ev.mc_evals,ev.delta_mc_true,ev.delta_exact,ev.regret)
        push!(rows,(training_run=run,epoch=epoch,method=name,epsilon=eps,energy=fr.energy,
            target_rms=fr.target_rms,eta_oracle_armijo=ev.eta_oracle,delta_oracle_armijo=ev.delta_oracle,
            eta_mc_armijo=ev.eta_mc,delta_mc_true=ev.delta_mc_true,mc_evals=ev.mc_evals,
            mc_acceptance=ev.mc_accept,same_eta_fraction=ev.same_eta,safe_fraction=ev.safe_fraction,
            eta_exact=ev.eta_exact,delta_exact=ev.delta_exact,regret_to_exact=ev.regret,
            mean_unique=nu,observable_M=observable_M,observable_repetitions=observable_repetitions))
    end
    rows
end

function run_training(run,X,flips)
    H=GBTQuantum.TFIMHamiltonian(NV.N;J=NV.J,h=NV.h,periodic=true)
    rng=MersenneTwister(NV.base_seed+10_000*run)
    samples=Matrix{Int8}(undef,NV.training_nsamples,NV.N)
    @inbounds for i in eachindex(samples) samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1) end
    model=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); logamps=zeros(NV.training_nsamples)
    for _ in 1:NV.burn_in_sweeps GBTQuantum.sweep!(rng,model,samples,logamps) end
    rows=NamedTuple[]
    for epoch in 1:NV.nepochs
        batch=GBTQuantum.vmc_batch(H,model,samples); yA,_=GBTQuantum.make_targets(batch); w=batch.counts
        if epoch in NV.checkpoint_epochs append!(rows,diagnose(H,model,X,flips,run,epoch)) end
        tree=GBTQuantum.grow_tree(batch.states,yA,w;max_depth=NV.optimizer_max_depth,
            min_weight=NV.optimizer_min_leaf_weight,min_gain=NV.optimizer_min_gain)
        pred=NV.predict_all(tree,batch.states); mu=NV.weighted_mean(pred,w)
        if isfinite(mu)&&mu!=0 tree=NV.shift_tree_leaves(tree,mu) end
        push!(model.logamp.trees,NV.scale_tree(tree,NV.eta_training))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:NV.sweeps_per_epoch GBTQuantum.sweep!(rng,model,samples,logamps) end
    end
    rows
end

function main()
    println("\n============================================================")
    println("REWEIGHTED MONTE CARLO ARMIJO VALIDATION")
    println("N=$(NV.N) J/h=$(NV.J/NV.h) runs=$(NV.ntraining_runs) checkpoints=$(sort(collect(NV.checkpoint_epochs)))")
    println("training M=$(NV.M); observable Born M=$observable_M repetitions=$observable_repetitions")
    println("Armijo alpha=$armijo_alpha beta=$backtrack_beta max_backtracks=$max_backtracks")
    println("MC decisions use one common Born sample reweighted across candidate eta values")
    println("============================================================")
    X=NV.enumerate_states(NV.N); flips=NV.build_flip_index(X); rows=NamedTuple[]
    for run in 1:NV.ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,NV.ntraining_runs)
        append!(rows,run_training(run,X,flips))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"reweighted_armijo_validation.csv")
    NV.write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/reweighted_armijo_validation.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    ReweightedArmijoValidationExperiment.main()
end
