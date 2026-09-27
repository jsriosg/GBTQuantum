using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module AutocorrelationBatchArmijoValidationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "reused_baseline_armijo_validation.jl"))
const RB = ReusedBaselineArmijoValidationExperiment
const SC = RB.SC
const NV = RB.NV

const zcrit = 1.0
const repetitions = 50
const policies = ["batch_iid", "batch_tau"]

common_seed(run,epoch,name,rep) =
    960_000_000 + 10_000_000*run + 100_000*epoch + rep + sum(codeunits(name))

function tau_int_ips(x::AbstractVector{<:Real})
    n=length(x)
    n < 4 && return 0.5
    y=Float64.(x) .- mean(x)
    gamma0=sum(abs2,y)/n
    gamma0 <= eps() && return 0.5
    maxlag=min(n-1, max(2, n÷4))
    rho=Vector{Float64}(undef,maxlag)
    @inbounds for lag in 1:maxlag
        rho[lag]=dot(@view(y[1:n-lag]),@view(y[1+lag:n]))/((n-lag)*gamma0)
    end
    s=0.0; k=1
    while k <= maxlag
        pair=rho[k]+(k+1<=maxlag ? rho[k+1] : 0.0)
        pair <= 0 && break
        s += pair; k += 2
    end
    max(0.5,0.5+s)
end

function chronological_local_energies(H,model,samples)
    n=size(samples,1)
    e=Vector{Float64}(undef,n)
    @inbounds for i in 1:n
        # VMC.jl defines local_energy! (not local_energy).  Keep the original
        # walker ordering so the autocorrelation calculation is meaningful.
        e[i]=real(GBTQuantum.local_energy!(H,model,@view(samples[i,:])))
    end
    e
end

function baseline_stats(H,model,samples)
    e=chronological_local_energies(H,model,samples)
    n=length(e); E0=mean(e)
    v=n>1 ? var(e;corrected=true) : 0.0
    tau=tau_int_ips(e)
    neff=min(Float64(n),max(1.0,n/(2tau)))
    se_iid=sqrt(max(v,0.0)/n)
    se_tau=sqrt(max(v,0.0)/neff)
    (E0=E0,se_iid=se_iid,se_tau=se_tau,tau=tau,neff=neff,n=n)
end

function armijo(rng,fr,X,flips,f,g,c,base,policy)
    eta=SC.candidate_eta(g,c)
    eta==0 && return (eta=0.0,accepted=false,total=0,refine=0,back=0)
    se0=policy=="batch_iid" ? base.se_iid : base.se_tau
    total=0; refine=0; back=0
    for _ in 0:SC.max_backtracks
        veta=Float64[]; previous=0; rejected=false
        for budget in (256,512)
            add=budget-previous
            SC.draw_energies!(rng,veta,fr,X,flips,f,eta,add)
            total += add; previous=budget
            Ee,see=SC.stats(veta)
            Dhat=Ee-base.E0-SC.armijo_alpha*eta*g
            seD=sqrt(se0^2+see^2)
            hi=Dhat+zcrit*seD; lo=Dhat-zcrit*seD
            if hi < 0
                return (eta=eta,accepted=true,total=total,refine=refine,back=back)
            elseif lo > 0
                rejected=true; break
            elseif budget==256
                refine += 1
            end
        end
        back += 1
        eta *= SC.beta
    end
    (eta=0.0,accepted=false,total=total,refine=refine,back=back)
end

function evaluate_tree(fr,X,flips,f,base,run,epoch,name,eps)
    g,c=NV.analytic_derivatives(fr,X,flips,f)
    etaoracle=SC.exact_armijo(fr,X,flips,f,g,c)
    etaopt,dopt=NV.exact_line_search(fr,X,flips,f)
    rows=NamedTuple[]
    for policy in policies
        vals=NamedTuple[]
        for rep in 1:repetitions
            rng=MersenneTwister(common_seed(run,epoch,name,rep))
            mc=armijo(rng,fr,X,flips,f,g,c,base,policy)
            d=NV.exact_energy_eta(fr,X,flips,f,mc.eta)-fr.energy
            imp=max(-d,0.0)
            push!(vals,(eta=mc.eta,safe=Float64(d<=1e-12),d=d,
                same=Float64(isapprox(mc.eta,etaoracle;atol=1e-12)),total=Float64(mc.total),
                refine=Float64(mc.refine),back=Float64(mc.back),eff=mc.total>0 ? imp/mc.total : NaN))
        end
        mv(s)=mean(getproperty(v,s) for v in vals)
        @printf("    %-10s eta=%.4f safe=%.2f same=%.2f dE=% .3e new=%4.0f back=%.2f refine=%.2f eff=%.3e\n",
            policy,mv(:eta),mv(:safe),mv(:same),mv(:d),mv(:total),mv(:back),mv(:refine),mv(:eff))
        push!(rows,(training_run=run,epoch=epoch,method=name,training_epsilon=eps,policy=policy,
            energy=fr.energy,target_rms=fr.target_rms,batch_E0=base.E0,batch_n=base.n,
            tau_int=base.tau,batch_neff=base.neff,se_iid=base.se_iid,se_tau=base.se_tau,
            g=g,curvature=c,eta_oracle_armijo=etaoracle,eta_exact=etaopt,delta_exact=dopt,
            eta_mc=mv(:eta),delta_mc_true=mv(:d),safe_fraction=mv(:safe),same_eta_fraction=mv(:same),
            mean_new_samples=mv(:total),mean_backtracks=mv(:back),mean_refinements=mv(:refine),
            mean_improvement_per_new_sample=mv(:eff),regret_to_exact=mv(:d)-dopt))
    end
    rows
end

function diagnose(H,model,samples,X,flips,run,epoch)
    fr=NV.exact_frozen_problem(H,model,X)
    base=baseline_stats(H,model,samples)
    @printf("\nEpoch %d E=% .8f RMS=%.3e | batch E0=% .8f tau=%.3f Neff=%.1f/%d SE iid=%.3e tau=%.3e\n",
        epoch,fr.energy,fr.target_rms,base.E0,base.tau,base.neff,base.n,base.se_iid,base.se_tau)
    rows=NamedTuple[]
    methods=Any[("full_hilbert",0.0,NV.full_tree(X,fr))]
    for pr in NV.proposals(fr)
        rng=MersenneTwister(NV.diagnostic_seed_base+10_000_000*run+100_000*epoch+sum(codeunits(pr.name))+round(Int,1000pr.epsilon))
        tree,_=NV.sampled_tree(rng,X,fr,pr.q)
        push!(methods,(pr.name,pr.epsilon,tree))
    end
    for (name,eps,tree) in methods
        f=NV.centered_predictions(tree,X,fr.probabilities)
        @printf("  %s\n",name)
        append!(rows,evaluate_tree(fr,X,flips,f,base,run,epoch,name,eps))
    end
    rows
end

function run_training(run,X,flips)
    H=GBTQuantum.TFIMHamiltonian(NV.N;J=NV.J,h=NV.h,periodic=true)
    rng=MersenneTwister(NV.base_seed+10_000*run)
    samples=Matrix{Int8}(undef,NV.training_nsamples,NV.N)
    @inbounds for i in eachindex(samples) samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1) end
    model=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    logamps=zeros(NV.training_nsamples)
    for _ in 1:NV.burn_in_sweeps GBTQuantum.sweep!(rng,model,samples,logamps) end
    rows=NamedTuple[]
    for epoch in 1:NV.nepochs
        batch=GBTQuantum.vmc_batch(H,model,samples); yA,_=GBTQuantum.make_targets(batch); w=batch.counts
        if epoch in NV.checkpoint_epochs append!(rows,diagnose(H,model,samples,X,flips,run,epoch)) end
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
    println("AUTOCORRELATION-AWARE BATCH E(0) ARMIJO VALIDATION")
    println("Compare iid batch SE against integrated-autocorrelation corrected SE")
    println("Armijo candidate budget 256 -> 512, z=$zcrit, repetitions=$repetitions")
    println("Also logs tau_int and effective batch size at every checkpoint")
    println("Keep proposal families separate: this diagnoses optimizer uncertainty; it does not assume p*y^2 is desirable")
    println("============================================================")
    X=NV.enumerate_states(NV.N); flips=NV.build_flip_index(X); rows=NamedTuple[]
    for run in 1:NV.ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,NV.ntraining_runs)
        append!(rows,run_training(run,X,flips))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"autocorrelation_batch_armijo_validation.csv")
    NV.write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/autocorrelation_batch_armijo_validation.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    AutocorrelationBatchArmijoValidationExperiment.main()
end
