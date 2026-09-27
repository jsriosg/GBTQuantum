using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module BoundedCostArmijoValidationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "sequential_confidence_armijo_validation.jl"))
const SC = SequentialConfidenceArmijoValidationExperiment
const NV = SC.NV

const zcrit = 1.0
const repetitions = 50
const policies = ["sequential_4096", "fixed_512", "fixed_1024", "two_stage_256_1024"]

# Common-random-number seed: deliberately independent of policy so every policy
# in a repetition starts from the same RNG stream. This removes most of the
# Monte Carlo noise from policy-to-policy comparisons.
common_seed(run, epoch, name, rep) =
    930_000_000 + 10_000_000*run + 100_000*epoch + rep + sum(codeunits(name))

function bounded_armijo(rng, fr, X, flips, f, g, c, policy)
    eta = SC.candidate_eta(g,c)
    eta == 0 && return (eta=0.0, accepted=false, candidate_evals=0,
        total_samples=0, refinements=0, ambiguous=0)

    if policy == "sequential_4096"
        return SC.sequential_armijo(rng,fr,X,flips,f,g,c,zcrit)
    end

    budgets = policy == "fixed_512" ? (512,) :
              policy == "fixed_1024" ? (1024,) :
              policy == "two_stage_256_1024" ? (256,1024) :
              error("unknown policy: $policy")

    # Draw a fresh E0 sample for every Armijo candidate, as well as a fresh
    # candidate sample. This makes each decision bounded by the advertised
    # budget and keeps the comparison simple. Within a decision, the two-stage
    # policy extends the same samples from 256 to 1024 rather than restarting.
    candidate_evals = 0
    total_samples = 0
    refinements = 0
    ambiguous = 0

    for _ in 0:SC.max_backtracks
        v0 = Float64[]
        veta = Float64[]
        previous = 0
        decided = false

        for budget in budgets
            add = budget - previous
            SC.draw_energies!(rng,v0,fr,X,flips,f,0.0,add)
            SC.draw_energies!(rng,veta,fr,X,flips,f,eta,add)
            total_samples += 2add
            candidate_evals += add
            previous = budget

            E0,se0 = SC.stats(v0)
            Ee,see = SC.stats(veta)
            Dhat = Ee-E0-SC.armijo_alpha*eta*g
            seD = sqrt(se0^2+see^2)
            hi = Dhat + zcrit*seD
            lo = Dhat - zcrit*seD

            if hi < 0
                return (eta=eta, accepted=true, candidate_evals=candidate_evals,
                    total_samples=total_samples, refinements=refinements,
                    ambiguous=ambiguous)
            elseif lo > 0
                decided = true
                break
            elseif budget != budgets[end]
                ambiguous += 1
                refinements += 1
            end
        end

        # At the fixed budget, an unresolved decision is deliberately treated
        # as insufficient evidence to accept. Backtrack instead of spending
        # more samples. This is the key bounded-cost rule being tested.
        if !decided
            ambiguous += 1
        end
        eta *= SC.beta
    end

    (eta=0.0, accepted=false, candidate_evals=candidate_evals,
     total_samples=total_samples, refinements=refinements, ambiguous=ambiguous)
end

function evaluate_tree(fr,X,flips,f,run,epoch,name,train_eps)
    g,c = NV.analytic_derivatives(fr,X,flips,f)
    eta0 = SC.candidate_eta(g,c)
    etaoracle = SC.exact_armijo(fr,X,flips,f,g,c)
    etaopt,dopt = NV.exact_line_search(fr,X,flips,f)
    rows = NamedTuple[]

    for policy in policies
        vals = NamedTuple[]
        for rep in 1:repetitions
            # Same seed for all policies in this repetition = common random numbers.
            rng = MersenneTwister(common_seed(run,epoch,name,rep))
            mc = bounded_armijo(rng,fr,X,flips,f,g,c,policy)
            dtrue = NV.exact_energy_eta(fr,X,flips,f,mc.eta)-fr.energy
            improvement = max(-dtrue,0.0)
            efficiency = mc.total_samples > 0 ? improvement/mc.total_samples : 0.0
            push!(vals,(eta=mc.eta,
                safe=Float64(dtrue<=1e-12),
                same=Float64(isapprox(mc.eta,etaoracle;atol=1e-12)),
                accepted=Float64(mc.accepted),
                dtrue=dtrue,
                candidate_evals=Float64(mc.candidate_evals),
                total_samples=Float64(mc.total_samples),
                refinements=Float64(mc.refinements),
                ambiguous=Float64(mc.ambiguous),
                improvement=improvement,
                efficiency=efficiency))
        end

        mv(s)=mean(getproperty(v,s) for v in vals)
        @printf("    %-20s oracle=%.4f MC=%.4f same=%.2f safe=%.2f dE=% .3e total=%.0f refine=%.2f eff=%.3e regret=%.2e\n",
            policy,etaoracle,mv(:eta),mv(:same),mv(:safe),mv(:dtrue),
            mv(:total_samples),mv(:refinements),mv(:efficiency),mv(:dtrue)-dopt)

        push!(rows,(training_run=run,epoch=epoch,method=name,training_epsilon=train_eps,
            policy=policy,z=zcrit,repetitions=repetitions,energy=fr.energy,
            target_rms=fr.target_rms,g=g,curvature=c,eta_initial=eta0,
            eta_oracle_armijo=etaoracle,eta_exact=etaopt,delta_exact=dopt,
            eta_mc=mv(:eta),delta_mc_true=mv(:dtrue),safe_fraction=mv(:safe),
            same_eta_fraction=mv(:same),acceptance_fraction=mv(:accepted),
            mean_candidate_samples=mv(:candidate_evals),mean_total_samples=mv(:total_samples),
            mean_refinements=mv(:refinements),mean_ambiguous_events=mv(:ambiguous),
            mean_energy_improvement=mv(:improvement),
            mean_improvement_per_sample=mv(:efficiency),
            regret_to_exact=mv(:dtrue)-dopt))
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
    println("BOUNDED-COST CONFIDENCE ARMIJO VALIDATION")
    println("N=$(NV.N) J/h=$(NV.J/NV.h) runs=$(NV.ntraining_runs) checkpoints=$(sort(collect(NV.checkpoint_epochs)))")
    println("z=$zcrit repetitions=$repetitions")
    println("policies: sequential max M=4096; fixed M=512; fixed M=1024; two-stage 256->1024")
    println("unresolved bounded decisions backtrack; no bounded policy exceeds its per-candidate cap")
    println("common random-number seeds are shared across policies within each repetition")
    println("metrics include safety, oracle agreement, regret, cost, and -dE/total_samples")
    println("Candidate samples remain exact draws from p_eta to isolate optimizer statistics")
    println("============================================================")

    X=NV.enumerate_states(NV.N); flips=NV.build_flip_index(X); rows=NamedTuple[]
    for run in 1:NV.ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,NV.ntraining_runs)
        append!(rows,run_training(run,X,flips))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"bounded_cost_armijo_validation.csv")
    NV.write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/bounded_cost_armijo_validation.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    BoundedCostArmijoValidationExperiment.main()
end
