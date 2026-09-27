using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module ArmijoHardBudgetValidationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "autocorrelation_batch_armijo_validation.jl"))
const AC = AutocorrelationBatchArmijoValidationExperiment
const SC = AC.SC
const NV = AC.NV

const repetitions = 50
const zcrit = 1.0
const total_budgets = [256, 512, 1024]
const first_batch = 256

common_seed(run,epoch,name,rep) =
    970_000_000 + 10_000_000*run + 100_000*epoch + rep + sum(codeunits(name))

# Production-oriented policy: E(0) and its uncertainty are reused from the
# training VMC batch. Only candidate-energy evaluations count against budget.
# A candidate is first tested with up to 256 draws. If ambiguous and budget
# remains, it may receive one additional block up to 512 total draws. A
# rejected candidate is backtracked, but the whole search stops once the hard
# global budget is exhausted. If no step is certified, the weak learner is
# rejected (eta=0).
function budgeted_armijo(rng,fr,X,flips,f,g,c,base,total_budget)
    eta=SC.candidate_eta(g,c)
    eta==0 && return (eta=0.0,accepted=false,used=0,back=0,refine=0,reason="zero_candidate")

    used=0; back=0; refine=0
    while back <= SC.max_backtracks && used < total_budget
        veta=Float64[]
        candidate_used=0
        while candidate_used < 512 && used < total_budget
            block=min(first_batch,512-candidate_used,total_budget-used)
            block <= 0 && break
            SC.draw_energies!(rng,veta,fr,X,flips,f,eta,block)
            used += block; candidate_used += block

            Ee,see=SC.stats(veta)
            Dhat=Ee-base.E0-SC.armijo_alpha*eta*g
            seD=sqrt(base.se_iid^2+see^2)
            hi=Dhat+zcrit*seD
            lo=Dhat-zcrit*seD

            if hi < 0
                return (eta=eta,accepted=true,used=used,back=back,refine=refine,reason="accepted")
            elseif lo > 0
                break
            elseif candidate_used < 512 && used < total_budget
                refine += 1
            else
                # Ambiguous at the per-candidate cap. Backtrack if global
                # budget remains; otherwise reject the weak learner.
                break
            end
        end
        back += 1
        eta *= SC.beta
    end

    reason = used >= total_budget ? "budget_exhausted" : "max_backtracks"
    (eta=0.0,accepted=false,used=used,back=back,refine=refine,reason=reason)
end

function evaluate_tree(fr,X,flips,f,base,run,epoch,name,eps)
    g,c=NV.analytic_derivatives(fr,X,flips,f)
    etaoracle=SC.exact_armijo(fr,X,flips,f,g,c)
    etaopt,dopt=NV.exact_line_search(fr,X,flips,f)
    rows=NamedTuple[]

    for budget in total_budgets
        vals=NamedTuple[]
        for rep in 1:repetitions
            rng=MersenneTwister(common_seed(run,epoch,name,rep))
            mc=budgeted_armijo(rng,fr,X,flips,f,g,c,base,budget)
            d=NV.exact_energy_eta(fr,X,flips,f,mc.eta)-fr.energy
            imp=max(-d,0.0)
            push!(vals,(eta=mc.eta,d=d,safe=Float64(d<=1e-12),
                accepted=Float64(mc.accepted),same=Float64(isapprox(mc.eta,etaoracle;atol=1e-12)),
                used=Float64(mc.used),back=Float64(mc.back),refine=Float64(mc.refine),
                exhausted=Float64(mc.reason=="budget_exhausted"),
                eff=mc.used>0 ? imp/mc.used : NaN))
        end
        mv(s)=mean(getproperty(v,s) for v in vals)
        @printf("    B=%4d eta=%.4f accept=%.2f safe=%.2f dE=% .3e used=%5.1f exhaust=%.2f back=%.2f eff=%.3e\n",
            budget,mv(:eta),mv(:accepted),mv(:safe),mv(:d),mv(:used),mv(:exhausted),mv(:back),mv(:eff))
        push!(rows,(training_run=run,epoch=epoch,method=name,training_epsilon=eps,
            hard_budget=budget,energy=fr.energy,target_rms=fr.target_rms,
            batch_E0=base.E0,batch_n=base.n,se0=base.se_iid,g=g,curvature=c,
            eta_oracle_armijo=etaoracle,eta_exact=etaopt,delta_exact=dopt,
            eta_mc=mv(:eta),delta_mc_true=mv(:d),accept_fraction=mv(:accepted),
            safe_fraction=mv(:safe),same_eta_fraction=mv(:same),mean_new_samples=mv(:used),
            budget_exhausted_fraction=mv(:exhausted),mean_backtracks=mv(:back),
            mean_refinements=mv(:refine),mean_improvement_per_new_sample=mv(:eff),
            regret_to_exact=mv(:d)-dopt))
    end
    rows
end

function diagnose(H,model,samples,X,flips,run,epoch)
    fr=NV.exact_frozen_problem(H,model,X)
    base=AC.baseline_stats(H,model,samples)
    @printf("\nEpoch %d E=% .8f RMS=%.3e | reused batch E0=% .8f SE=%.3e\n",
        epoch,fr.energy,fr.target_rms,base.E0,base.se_iid)
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
    @inbounds for i in eachindex(samples)
        samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1)
    end
    model=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    logamps=zeros(NV.training_nsamples)
    for _ in 1:NV.burn_in_sweeps
        GBTQuantum.sweep!(rng,model,samples,logamps)
    end

    rows=NamedTuple[]
    for epoch in 1:NV.nepochs
        batch=GBTQuantum.vmc_batch(H,model,samples)
        yA,_=GBTQuantum.make_targets(batch)
        w=batch.counts

        if epoch in NV.checkpoint_epochs
            append!(rows,diagnose(H,model,samples,X,flips,run,epoch))
        end

        tree=GBTQuantum.grow_tree(batch.states,yA,w;
            max_depth=NV.optimizer_max_depth,
            min_weight=NV.optimizer_min_leaf_weight,
            min_gain=NV.optimizer_min_gain)
        pred=NV.predict_all(tree,batch.states)
        mu=NV.weighted_mean(pred,w)
        if isfinite(mu) && mu != 0
            tree=NV.shift_tree_leaves(tree,mu)
        end
        push!(model.logamp.trees,NV.scale_tree(tree,NV.eta_training))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:NV.sweeps_per_epoch
            GBTQuantum.sweep!(rng,model,samples,logamps)
        end
    end
    rows
end

function main()
    println("\n============================================================")
    println("HARD-BUDGET ARMIJO VALIDATION")
    println("N=$(NV.N) J/h=$(NV.J/NV.h) runs=$(NV.ntraining_runs) checkpoints=$(NV.checkpoint_epochs)")
    println("repetitions=$repetitions hard new-sample budgets=$total_budgets")
    println("E(0) reused from training VMC batch; IID baseline SE")
    println("candidate evaluation: 256, optionally refine to 512; global budget is strict")
    println("if no descent step is certified inside budget, reject weak learner (eta=0)")
    println("metrics: safety, acceptance, exact dE, regret, cost, exhaustion, efficiency")
    println("============================================================")

    X=NV.enumerate_states(NV.N)
    flips=NV.build_flip_index(X)
    rows=NamedTuple[]
    for run in 1:NV.ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,NV.ntraining_runs)
        append!(rows,run_training(run,X,flips))
    end

    outdir=joinpath(@__DIR__,"results")
    mkpath(outdir)
    path=joinpath(outdir,"armijo_hard_budget_validation.csv")
    NV.write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/armijo_hard_budget_validation.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    ArmijoHardBudgetValidationExperiment.main()
end
