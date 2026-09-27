using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module ReweightingOverlapDiagnostics

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "newton_step_validation.jl"))
const NV = NewtonStepValidationExperiment

const armijo_alpha=0.10
const beta=0.50
const max_backtracks=12
const eta_cap=0.40
const Mval=1024
const repetitions=50
const validation_eps=[0.0,0.10,0.25]
const ess_thresholds=[0.0,0.10,0.20,0.30,0.50]
const tiny=1e-300

function candidate_eta(g,c)
    !(isfinite(g)&&g<0) && return 0.0
    (isfinite(c)&&c>0) ? min(-g/c,eta_cap) : min(NV.eta_training,eta_cap)
end

function exact_trial_distribution(fr,f,eta)
    lw=2eta .* f
    m=maximum(lw)
    a=fr.probabilities .* exp.(lw .- m)
    Z=sum(a)
    peta=a./Z
    peta,lw
end

function exact_overlap(fr,f,eta)
    peta,lw=exact_trial_distribution(fr,f,eta)
    # Exact population ESS fraction for reweighting p0 -> peta:
    # (E_p w)^2/E_p[w^2]. Scale weights for stability.
    m=maximum(lw); w=exp.(lw .- m); p=fr.probabilities
    essfrac=(sum(p.*w)^2)/sum(p.*w.^2)
    kl=sum(peta .* log.(max.(peta,tiny)./max.(p,tiny)))
    return essfrac,kl,maximum(peta)
end

function validation_proposal(fr,eps)
    d=length(fr.probabilities)
    eps==0 && return copy(fr.probabilities)
    (1-eps).*fr.probabilities .+ eps/d
end

function draw_validation(rng,q)
    NV.draw_categorical_indices(rng,q,Mval)
end

# Importance estimate under arbitrary q. Target numerator is expectation under p0.
# For candidate eta, unnormalized target factor is p0/q * exp(2 eta f).
function estimated_energy(fr,X,flips,f,idx,q,eta)
    logw=Vector{Float64}(undef,length(idx)); el=Vector{Float64}(undef,length(idx))
    for (a,i) in pairs(idx)
        diag=0.0
        for j in 1:NV.N
            jp=(j==NV.N ? 1 : j+1)
            diag += -NV.J*Float64(X[i,j])*Float64(X[i,jp])
        end
        loc=diag
        for j in 1:NV.N
            k=flips[i,j]
            loc += -NV.h*exp(fr.logamp[k]-fr.logamp[i]) * exp(eta*(f[k]-f[i]))
        end
        el[a]=loc
        logw[a]=log(max(fr.probabilities[i],tiny))-log(max(q[i],tiny))+2eta*f[i]
    end
    m=maximum(logw); w=exp.(logw .- m); sw=sum(w)
    Ehat=sum(w.*el)/sw
    ess=(sw^2)/sum(w.^2)
    maxshare=maximum(w)/sw
    return Ehat,ess/length(w),maxshare
end

function exact_armijo(fr,X,flips,f,g,c)
    eta=candidate_eta(g,c)
    eta==0 && return 0.0
    for _ in 0:max_backtracks
        E=NV.exact_energy_eta(fr,X,flips,f,eta)
        E <= fr.energy + armijo_alpha*eta*g && return eta
        eta*=beta
    end
    0.0
end

function mc_armijo(fr,X,flips,f,g,c,idx,q,essmin)
    eta=candidate_eta(g,c)
    eta==0 && return (eta=0.0,evals=0,accepted=false,miness=1.0,maxshare=0.0)
    E0,_,_=estimated_energy(fr,X,flips,f,idx,q,0.0)
    miness=1.0; maxshare=0.0
    for k in 0:max_backtracks
        Ehat,ess,share=estimated_energy(fr,X,flips,f,idx,q,eta)
        miness=min(miness,ess); maxshare=max(maxshare,share)
        # Reliability gate comes before the Armijo decision.
        if ess >= essmin && isfinite(Ehat) && Ehat <= E0 + armijo_alpha*eta*g
            return (eta=eta,evals=k+1,accepted=true,miness=miness,maxshare=maxshare)
        end
        eta*=beta
    end
    (eta=0.0,evals=max_backtracks+1,accepted=false,miness=miness,maxshare=maxshare)
end

function evaluate_tree(fr,X,flips,f,run,epoch,name,eps_train)
    g,c=NV.analytic_derivatives(fr,X,flips,f)
    eta0=candidate_eta(g,c); etaoracle=exact_armijo(fr,X,flips,f,g,c)
    etaopt,dopt=NV.exact_line_search(fr,X,flips,f)
    essx,klx,pmax=exact_overlap(fr,f,eta0)
    rows=NamedTuple[]
    for epsval in validation_eps
        q=validation_proposal(fr,epsval)
        for essmin in ess_thresholds
            vals=NamedTuple[]
            for rep in 1:repetitions
                seed=510_000_000+10_000_000*run+100_000*epoch+10_000*round(Int,100epsval)+100*round(Int,100essmin)+rep+sum(codeunits(name))
                rng=MersenneTwister(seed); idx=draw_validation(rng,q)
                mc=mc_armijo(fr,X,flips,f,g,c,idx,q,essmin)
                dtrue=NV.exact_energy_eta(fr,X,flips,f,mc.eta)-fr.energy
                push!(vals,(eta=mc.eta,evals=Float64(mc.evals),accepted=Float64(mc.accepted),
                    safe=Float64(dtrue<=1e-12),same=Float64(isapprox(mc.eta,etaoracle;atol=1e-12)),
                    dtrue=dtrue,miness=mc.miness,maxshare=mc.maxshare))
            end
            mv(s)=mean(getproperty(v,s) for v in vals)
            push!(rows,(training_run=run,epoch=epoch,method=name,training_epsilon=eps_train,
                validation_epsilon=epsval,ess_threshold=essmin,energy=fr.energy,target_rms=fr.target_rms,
                g=g,curvature=c,eta_initial=eta0,eta_oracle_armijo=etaoracle,eta_exact=etaopt,
                delta_exact=dopt,exact_ess_fraction_initial=essx,exact_kl_initial=klx,
                exact_trial_max_probability=pmax,eta_mc=mv(:eta),delta_mc_true=mv(:dtrue),
                safe_fraction=mv(:safe),same_eta_fraction=mv(:same),acceptance_fraction=mv(:accepted),
                mean_evals=mv(:evals),mean_min_sample_ess_fraction=mv(:miness),
                mean_max_weight_share=mv(:maxshare),regret_to_exact=mv(:dtrue)-dopt,repetitions=repetitions))
        end
    end
    rows
end

function diagnose(H,model,X,flips,run,epoch)
    fr=NV.exact_frozen_problem(H,model,X); rows=NamedTuple[]
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e\n",epoch,fr.energy,fr.target_rms)
    methods=Any[("full_hilbert",0.0,NV.full_tree(X,fr))]
    for pr in NV.proposals(fr)
        rng=MersenneTwister(NV.diagnostic_seed_base+10_000_000*run+100_000*epoch+sum(codeunits(pr.name))+round(Int,1000pr.epsilon))
        tree,_=NV.sampled_tree(rng,X,fr,pr.q); push!(methods,(pr.name,pr.epsilon,tree))
    end
    for (name,eps,tree) in methods
        f=NV.centered_predictions(tree,X,fr.probabilities)
        g,c=NV.analytic_derivatives(fr,X,flips,f); eta0=candidate_eta(g,c)
        exess,kl,_=exact_overlap(fr,f,eta0)
        @printf("  %-15s eta0=%.4f exact ESS=%.3f KL=%.3f\n",name,eta0,exess,kl)
        append!(rows,evaluate_tree(fr,X,flips,f,run,epoch,name,eps))
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
    println("REWEIGHTING OVERLAP + VALIDATION PROPOSAL DIAGNOSTICS")
    println("N=$(NV.N) J/h=$(NV.J/NV.h) runs=$(NV.ntraining_runs) checkpoints=$(sort(collect(NV.checkpoint_epochs)))")
    println("Mval=$Mval repetitions=$repetitions validation eps=$validation_eps")
    println("ESS thresholds=$ess_thresholds")
    println("Measures exact overlap, sampled ESS, weight concentration, safety and regret")
    println("============================================================")
    X=NV.enumerate_states(NV.N); flips=NV.build_flip_index(X); rows=NamedTuple[]
    for run in 1:NV.ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,NV.ntraining_runs)
        append!(rows,run_training(run,X,flips))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"reweighting_overlap_diagnostics.csv")
    NV.write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/reweighting_overlap_diagnostics.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    ReweightingOverlapDiagnostics.main()
end
