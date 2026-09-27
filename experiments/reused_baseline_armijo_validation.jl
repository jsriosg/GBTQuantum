using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module ReusedBaselineArmijoValidationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "minimal_budget_armijo_validation.jl"))
const MB = MinimalBudgetArmijoValidationExperiment
const SC = MB.SC
const NV = MB.NV

const zcrit = 1.0
const repetitions = 50
const policies = ["fresh_E0_256_512", "reuse_E0_256_512", "batch_E0_256_512"]

common_seed(run,epoch,name,rep) =
    950_000_000 + 10_000_000*run + 100_000*epoch + rep + sum(codeunits(name))

function batch_baseline_stats(batch)
    w = Float64.(batch.counts)
    e = Float64.(batch.local_energy)
    n = sum(w)
    E0 = sum(w .* e) / n
    # Approximate iid standard error of the original VMC observations, using
    # compressed multiplicities. This intentionally mirrors the simple MC
    # confidence model used by the validation experiment.
    var0 = n > 1 ? sum(w .* (e .- E0).^2) / (n - 1) : 0.0
    se0 = sqrt(max(var0,0.0) / n)
    return E0,se0,Int(round(n))
end

function reused_baseline_armijo(rng,fr,X,flips,f,g,c,policy,batchstats)
    eta = SC.candidate_eta(g,c)
    eta == 0 && return (eta=0.0,accepted=false,candidate_evals=0,total_new_samples=0,
        baseline_new_samples=0,baseline_source_samples=0,refinements=0,ambiguous=0,backtracks=0)

    budgets=(256,512)
    candidate_evals=0; total_new_samples=0; baseline_new_samples=0
    refinements=0; ambiguous=0; backtracks=0

    # Reused-MC baseline: estimate E(0) once at the maximum baseline precision
    # and keep exactly the same estimate for eta0, eta0/2, ... .
    v0 = Float64[]
    if policy == "reuse_E0_256_512"
        SC.draw_energies!(rng,v0,fr,X,flips,f,0.0,512)
        total_new_samples += 512
        baseline_new_samples = 512
        E0,se0 = SC.stats(v0)
        baseline_source_samples = 512
    elseif policy == "batch_E0_256_512"
        E0,se0,baseline_source_samples = batchstats
    elseif policy == "fresh_E0_256_512"
        E0=NaN; se0=NaN; baseline_source_samples=0
    else
        error("unknown policy: $policy")
    end

    for _ in 0:SC.max_backtracks
        veta=Float64[]
        fresh0=Float64[]
        previous=0
        rejected=false

        for budget in budgets
            add=budget-previous
            if policy == "fresh_E0_256_512"
                SC.draw_energies!(rng,fresh0,fr,X,flips,f,0.0,add)
                total_new_samples += add
                baseline_new_samples += add
                E0,se0 = SC.stats(fresh0)
            end
            SC.draw_energies!(rng,veta,fr,X,flips,f,eta,add)
            total_new_samples += add
            candidate_evals += add
            previous=budget

            Ee,see=SC.stats(veta)
            Dhat=Ee-E0-SC.armijo_alpha*eta*g
            seD=sqrt(se0^2+see^2)
            hi=Dhat+zcrit*seD
            lo=Dhat-zcrit*seD

            if hi < 0
                return (eta=eta,accepted=true,candidate_evals=candidate_evals,
                    total_new_samples=total_new_samples,baseline_new_samples=baseline_new_samples,
                    baseline_source_samples=baseline_source_samples,refinements=refinements,
                    ambiguous=ambiguous,backtracks=backtracks)
            elseif lo > 0
                rejected=true
                break
            elseif budget != budgets[end]
                ambiguous += 1; refinements += 1
            end
        end
        if !rejected ambiguous += 1 end
        backtracks += 1
        eta *= SC.beta
    end

    (eta=0.0,accepted=false,candidate_evals=candidate_evals,total_new_samples=total_new_samples,
     baseline_new_samples=baseline_new_samples,baseline_source_samples=baseline_source_samples,
     refinements=refinements,ambiguous=ambiguous,backtracks=backtracks)
end

function evaluate_tree(fr,X,flips,f,batchstats,run,epoch,name,train_eps)
    g,c=NV.analytic_derivatives(fr,X,flips,f)
    eta0=SC.candidate_eta(g,c)
    etaoracle=SC.exact_armijo(fr,X,flips,f,g,c)
    etaopt,dopt=NV.exact_line_search(fr,X,flips,f)
    rows=NamedTuple[]

    for policy in policies
        vals=NamedTuple[]
        for rep in 1:repetitions
            rng=MersenneTwister(common_seed(run,epoch,name,rep))
            mc=reused_baseline_armijo(rng,fr,X,flips,f,g,c,policy,batchstats)
            dtrue=NV.exact_energy_eta(fr,X,flips,f,mc.eta)-fr.energy
            improvement=max(-dtrue,0.0)
            efficiency=mc.total_new_samples>0 ? improvement/mc.total_new_samples : NaN
            push!(vals,(eta=mc.eta,safe=Float64(dtrue<=1e-12),
                same=Float64(isapprox(mc.eta,etaoracle;atol=1e-12)),dtrue=dtrue,
                accepted=Float64(mc.accepted),candidate_evals=Float64(mc.candidate_evals),
                total_new_samples=Float64(mc.total_new_samples),
                baseline_new_samples=Float64(mc.baseline_new_samples),
                baseline_source_samples=Float64(mc.baseline_source_samples),
                refinements=Float64(mc.refinements),backtracks=Float64(mc.backtracks),
                improvement=improvement,efficiency=efficiency))
        end
        mv(s)=mean(getproperty(v,s) for v in vals)
        @printf("    %-20s oracle=%.4f MC=%.4f same=%.2f safe=%.2f dE=% .3e new=%4.0f base-new=%3.0f back=%.2f eff=%.3e regret=%.2e\n",
            policy,etaoracle,mv(:eta),mv(:same),mv(:safe),mv(:dtrue),mv(:total_new_samples),
            mv(:baseline_new_samples),mv(:backtracks),mv(:efficiency),mv(:dtrue)-dopt)
        push!(rows,(training_run=run,epoch=epoch,method=name,training_epsilon=train_eps,
            policy=policy,z=zcrit,repetitions=repetitions,energy=fr.energy,target_rms=fr.target_rms,
            g=g,curvature=c,eta_initial=eta0,eta_oracle_armijo=etaoracle,eta_exact=etaopt,
            delta_exact=dopt,eta_mc=mv(:eta),delta_mc_true=mv(:dtrue),safe_fraction=mv(:safe),
            same_eta_fraction=mv(:same),acceptance_fraction=mv(:accepted),
            mean_candidate_samples=mv(:candidate_evals),mean_new_samples=mv(:total_new_samples),
            mean_baseline_new_samples=mv(:baseline_new_samples),
            mean_baseline_source_samples=mv(:baseline_source_samples),
            mean_backtracks=mv(:backtracks),mean_refinements=mv(:refinements),
            mean_energy_improvement=mv(:improvement),mean_improvement_per_new_sample=mv(:efficiency),
            regret_to_exact=mv(:dtrue)-dopt))
    end
    rows
end

function diagnose(H,model,batch,X,flips,run,epoch)
    fr=NV.exact_frozen_problem(H,model,X)
    batchstats=batch_baseline_stats(batch)
    rows=NamedTuple[]
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e | batch E0=% .8f se=%.3e n=%d\n",
        epoch,fr.energy,fr.target_rms,batchstats...)
    methods=Any[("full_hilbert",0.0,NV.full_tree(X,fr))]
    for pr in NV.proposals(fr)
        rng=MersenneTwister(NV.diagnostic_seed_base+10_000_000*run+100_000*epoch+sum(codeunits(pr.name))+round(Int,1000pr.epsilon))
        tree,_=NV.sampled_tree(rng,X,fr,pr.q)
        push!(methods,(pr.name,pr.epsilon,tree))
    end
    for (name,eps,tree) in methods
        f=NV.centered_predictions(tree,X,fr.probabilities)
        append!(rows,evaluate_tree(fr,X,flips,f,batchstats,run,epoch,name,eps))
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
        if epoch in NV.checkpoint_epochs append!(rows,diagnose(H,model,batch,X,flips,run,epoch)) end
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
    println("REUSED BASELINE E(0) ARMIJO VALIDATION")
    println("N=$(NV.N) J/h=$(NV.J/NV.h) runs=$(NV.ntraining_runs) checkpoints=$(sort(collect(NV.checkpoint_epochs)))")
    println("candidate policy: 256 -> 512 only if ambiguous; z=$zcrit repetitions=$repetitions")
    println("compare: fresh E0 each candidate; one reused 512-sample E0; existing training-batch E0")
    println("batch baseline costs ZERO new validation samples")
    println("unresolved at 512 conservatively backtracks")
    println("IMPORTANT: batch SE uses the same iid approximation as the current validation machinery")
    println("metrics count NEW validation samples separately from baseline source observations")
    println("============================================================")

    X=NV.enumerate_states(NV.N); flips=NV.build_flip_index(X); rows=NamedTuple[]
    for run in 1:NV.ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,NV.ntraining_runs)
        append!(rows,run_training(run,X,flips))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"reused_baseline_armijo_validation.csv")
    NV.write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/reused_baseline_armijo_validation.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    ReusedBaselineArmijoValidationExperiment.main()
end
