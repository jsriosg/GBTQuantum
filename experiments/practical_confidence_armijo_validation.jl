using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module PracticalConfidenceArmijoValidationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "sequential_confidence_armijo_validation.jl"))
const SC = SequentialConfidenceArmijoValidationExperiment
const NV = SC.NV

const zcrit = 1.0
const gamma_values = [0.0, 0.01, 0.025, 0.05, 0.10]
const repetitions = 50

# Practical indifference band around the Armijo boundary.  The scale |eta*g|
# is the first-order predicted energy decrease, so gamma is dimensionless.
# If the confidence interval cannot resolve the sign of D but lies inside this
# band, we stop refining and accept the candidate as practically equivalent to
# satisfying Armijo.  Outside the band we retain the ordinary confidence rule.
function practical_sequential_armijo(rng,fr,X,flips,f,g,c,gamma)
    eta=SC.candidate_eta(g,c)
    eta==0 && return (eta=0.0,accepted=false,candidate_evals=0,total_samples=0,
        refinements=0,ambiguous=0,practical_accepts=0)

    v0=Float64[]
    SC.draw_energies!(rng,v0,fr,X,flips,f,0.0,SC.initial_M)
    total_samples=SC.initial_M
    candidate_evals=0
    refinements=0
    ambiguous=0
    practical_accepts=0

    for _ in 0:SC.max_backtracks
        veta=Float64[]
        SC.draw_energies!(rng,veta,fr,X,flips,f,eta,SC.initial_M)
        total_samples += SC.initial_M
        candidate_evals += SC.initial_M
        M=SC.initial_M

        while true
            E0,se0=SC.stats(v0); Ee,see=SC.stats(veta)
            Dhat=Ee-E0-SC.armijo_alpha*eta*g
            seD=sqrt(se0^2+see^2)
            delta=gamma*abs(eta*g)
            lo=Dhat-zcrit*seD
            hi=Dhat+zcrit*seD

            if hi < 0
                return (eta=eta,accepted=true,candidate_evals=candidate_evals,
                    total_samples=total_samples,refinements=refinements,
                    ambiguous=ambiguous,practical_accepts=practical_accepts)
            elseif lo > 0
                break
            elseif abs(Dhat) + zcrit*seD <= delta
                practical_accepts += 1
                return (eta=eta,accepted=true,candidate_evals=candidate_evals,
                    total_samples=total_samples,refinements=refinements,
                    ambiguous=ambiguous,practical_accepts=practical_accepts)
            elseif M >= SC.max_M
                ambiguous += 1
                break
            else
                ambiguous += 1
                newM=min(2M,SC.max_M); add=newM-M
                SC.draw_energies!(rng,v0,fr,X,flips,f,0.0,add)
                SC.draw_energies!(rng,veta,fr,X,flips,f,eta,add)
                total_samples += 2add
                candidate_evals += add
                refinements += 1
                M=newM
            end
        end
        eta*=SC.beta
    end
    (eta=0.0,accepted=false,candidate_evals=candidate_evals,total_samples=total_samples,
     refinements=refinements,ambiguous=ambiguous,practical_accepts=practical_accepts)
end

function evaluate_tree(fr,X,flips,f,run,epoch,name,train_eps)
    g,c=NV.analytic_derivatives(fr,X,flips,f)
    eta0=SC.candidate_eta(g,c)
    etaoracle=SC.exact_armijo(fr,X,flips,f,g,c)
    etaopt,dopt=NV.exact_line_search(fr,X,flips,f)
    rows=NamedTuple[]
    for gamma in gamma_values
        vals=NamedTuple[]
        for rep in 1:repetitions
            seed=920_000_000+10_000_000*run+100_000*epoch+10_000*round(Int,1000gamma)+rep+sum(codeunits(name))
            rng=MersenneTwister(seed)
            mc=practical_sequential_armijo(rng,fr,X,flips,f,g,c,gamma)
            dtrue=NV.exact_energy_eta(fr,X,flips,f,mc.eta)-fr.energy
            push!(vals,(eta=mc.eta,safe=Float64(dtrue<=1e-12),
                same=Float64(isapprox(mc.eta,etaoracle;atol=1e-12)),
                accepted=Float64(mc.accepted),dtrue=dtrue,
                candidate_evals=Float64(mc.candidate_evals),total_samples=Float64(mc.total_samples),
                refinements=Float64(mc.refinements),ambiguous=Float64(mc.ambiguous),
                practical=Float64(mc.practical_accepts>0)))
        end
        mv(s)=mean(getproperty(v,s) for v in vals)
        @printf("    gamma=%.3f oracle=%.4f MC=%.4f same=%.2f safe=%.2f dE=% .3e cand.samples=%.0f total=%.0f refine=%.2f practical=%.2f regret=%.2e\n",
            gamma,etaoracle,mv(:eta),mv(:same),mv(:safe),mv(:dtrue),mv(:candidate_evals),
            mv(:total_samples),mv(:refinements),mv(:practical),mv(:dtrue)-dopt)
        push!(rows,(training_run=run,epoch=epoch,method=name,training_epsilon=train_eps,
            z=zcrit,gamma=gamma,initial_M=SC.initial_M,max_M=SC.max_M,repetitions=repetitions,
            energy=fr.energy,target_rms=fr.target_rms,g=g,curvature=c,eta_initial=eta0,
            eta_oracle_armijo=etaoracle,eta_exact=etaopt,delta_exact=dopt,
            eta_mc=mv(:eta),delta_mc_true=mv(:dtrue),safe_fraction=mv(:safe),
            same_eta_fraction=mv(:same),acceptance_fraction=mv(:accepted),
            practical_accept_fraction=mv(:practical),mean_candidate_samples=mv(:candidate_evals),
            mean_total_samples=mv(:total_samples),mean_refinements=mv(:refinements),
            mean_ambiguous_events=mv(:ambiguous),regret_to_exact=mv(:dtrue)-dopt))
    end
    rows
end

function diagnose(H,model,X,flips,run,epoch)
    fr=NV.exact_frozen_problem(H,model,X); rows=NamedTuple[]
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e\n",epoch,fr.energy,fr.target_rms)
    methods=Any[("full_hilbert",0.0,NV.full_tree(X,fr))]
    for pr in NV.proposals(fr)
        rng=MersenneTwister(NV.diagnostic_seed_base+10_000_000*run+100_000*epoch+sum(codeunits(pr.name))+round(Int,1000pr.epsilon))
        tree,_=NV.sampled_tree(rng,X,fr,pr.q)
        push!(methods,(pr.name,pr.epsilon,tree))
    end
    for (name,eps,tree) in methods
        f=NV.centered_predictions(tree,X,fr.probabilities)
        g,c=NV.analytic_derivatives(fr,X,flips,f)
        @printf("  %-15s eta0=%.4f\n",name,SC.candidate_eta(g,c))
        append!(rows,evaluate_tree(fr,X,flips,f,run,epoch,name,eps))
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
    println("PRACTICAL-CONFIDENCE DIRECT-SAMPLING ARMIJO")
    println("N=$(NV.N) J/h=$(NV.J/NV.h) runs=$(NV.ntraining_runs) checkpoints=$(sort(collect(NV.checkpoint_epochs)))")
    println("z=$zcrit gamma=$gamma_values initial M=$(SC.initial_M) max M=$(SC.max_M) repetitions=$repetitions")
    println("delta = gamma*abs(eta*g); practical acceptance requires the whole z-CI inside [-delta,+delta]")
    println("Candidate samples remain exact draws from p_eta to isolate optimizer statistics")
    println("============================================================")
    X=NV.enumerate_states(NV.N); flips=NV.build_flip_index(X); rows=NamedTuple[]
    for run in 1:NV.ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,NV.ntraining_runs)
        append!(rows,run_training(run,X,flips))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"practical_confidence_armijo_validation.csv")
    NV.write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/practical_confidence_armijo_validation.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    PracticalConfidenceArmijoValidationExperiment.main()
end
