using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module NewtonBacktrackingValidationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

# Reuse the already validated frozen-state, proposal, derivative and exact-energy
# machinery. This experiment changes only the step-selection rule.
include(joinpath(@__DIR__, "newton_step_validation.jl"))
const NV = NewtonStepValidationExperiment

const armijo_alpha = 0.10
const backtrack_beta = 0.50
const max_backtracks = 12
const eta_cap = 0.40

function safeguarded_newton(fr, X, flips, f)
    g,c = NV.analytic_derivatives(fr,X,flips,f)
    # No positive step can be certified as a descent direction from the local slope.
    if !(isfinite(g) && g < 0)
        return (eta=0.0, delta=0.0, evals=0, backtracks=0, accepted=false,
                g=g, c=c, eta_initial=0.0)
    end
    # Newton when local curvature is usable; otherwise conservative capped trial.
    eta0 = (isfinite(c) && c > 0) ? min(-g/c,eta_cap) : min(NV.eta_training,eta_cap)
    eta = eta0
    for k in 0:max_backtracks
        Etrial = NV.exact_energy_eta(fr,X,flips,f,eta)
        # Armijo sufficient decrease: E(eta) <= E0 + alpha*eta*g.
        if isfinite(Etrial) && Etrial <= fr.energy + armijo_alpha*eta*g
            return (eta=eta, delta=Etrial-fr.energy, evals=k+1, backtracks=k,
                    accepted=true,g=g,c=c,eta_initial=eta0)
        end
        eta *= backtrack_beta
    end
    return (eta=0.0,delta=0.0,evals=max_backtracks+1,backtracks=max_backtracks+1,
            accepted=false,g=g,c=c,eta_initial=eta0)
end

function evaluate_all(fr,X,flips,f)
    g,c=NV.analytic_derivatives(fr,X,flips,f)
    etaN=(g<0 && isfinite(c) && c>0) ? clamp(-g/c,0.0,eta_cap) : 0.0
    dN=NV.exact_energy_eta(fr,X,flips,f,etaN)-fr.energy
    etaE,dE=NV.exact_line_search(fr,X,flips,f)
    d05=NV.exact_energy_eta(fr,X,flips,f,0.05)-fr.energy
    bt=safeguarded_newton(fr,X,flips,f)
    return (g=g,c=c,eta_newton=etaN,delta_newton=dN,eta_bt=bt.eta,delta_bt=bt.delta,
            bt_evals=bt.evals,bt_steps=bt.backtracks,bt_accepted=bt.accepted,
            eta_exact=etaE,delta_exact=dE,delta_fixed=d05,
            regret_bt=bt.delta-dE,regret_newton=dN-dE)
end

function diagnose(H,model,X,flips,run,epoch)
    fr=NV.exact_frozen_problem(H,model,X); rows=NamedTuple[]
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e\n",epoch,fr.energy,fr.target_rms)

    ft=NV.full_tree(X,fr); ff=NV.centered_predictions(ft,X,fr.probabilities)
    ev=evaluate_all(fr,X,flips,ff)
    @printf("  FULL: etaN=%.4f dEN=% .3e | etaBT=%.4f dEBT=% .3e evals=%d | eta*=%.4f dE*=% .3e\n",
        ev.eta_newton,ev.delta_newton,ev.eta_bt,ev.delta_bt,ev.bt_evals,ev.eta_exact,ev.delta_exact)
    push!(rows,(training_run=run,epoch=epoch,method="full_hilbert",epsilon=0.0,
        energy=fr.energy,target_rms=fr.target_rms,g=ev.g,curvature=ev.c,
        eta_newton=ev.eta_newton,delta_newton=ev.delta_newton,
        eta_backtracking=ev.eta_bt,delta_backtracking=ev.delta_bt,
        backtracking_evals=Float64(ev.bt_evals),backtracking_steps=Float64(ev.bt_steps),
        backtracking_acceptance=Float64(ev.bt_accepted),eta_exact=ev.eta_exact,
        delta_exact=ev.delta_exact,delta_fixed=ev.delta_fixed,
        regret_backtracking=ev.regret_bt,regret_newton=ev.regret_newton,
        mean_unique=4096.0,repetitions=1))

    for pr in NV.proposals(fr)
        vals=NamedTuple[]
        for rep in 1:NV.repetitions
            methodhash=sum(Int(c) for c in codeunits(pr.name))+round(Int,1000*pr.epsilon)
            rng=MersenneTwister(NV.diagnostic_seed_base+10_000_000*run+100_000*epoch+100*methodhash+rep)
            tree,nu=NV.sampled_tree(rng,X,fr,pr.q)
            f=NV.centered_predictions(tree,X,fr.probabilities)
            e=evaluate_all(fr,X,flips,f)
            push!(vals,merge(e,(unique=Float64(nu),)))
        end
        meanv(sym)=mean(getproperty(v,sym) for v in vals)
        @printf("  %-15s fixed=% .3e Newton=% .3e BT=% .3e exact=% .3e | etaBT=%.4f evals=%.2f accept=%.2f regret=%.2e\n",
            pr.name,meanv(:delta_fixed),meanv(:delta_newton),meanv(:delta_bt),meanv(:delta_exact),
            meanv(:eta_bt),meanv(:bt_evals),meanv(:bt_accepted),meanv(:regret_bt))
        push!(rows,(training_run=run,epoch=epoch,method=pr.name,epsilon=pr.epsilon,
            energy=fr.energy,target_rms=fr.target_rms,g=meanv(:g),curvature=meanv(:c),
            eta_newton=meanv(:eta_newton),delta_newton=meanv(:delta_newton),
            eta_backtracking=meanv(:eta_bt),delta_backtracking=meanv(:delta_bt),
            backtracking_evals=meanv(:bt_evals),backtracking_steps=meanv(:bt_steps),
            backtracking_acceptance=meanv(:bt_accepted),eta_exact=meanv(:eta_exact),
            delta_exact=meanv(:delta_exact),delta_fixed=meanv(:delta_fixed),
            regret_backtracking=meanv(:regret_bt),regret_newton=meanv(:regret_newton),
            mean_unique=meanv(:unique),repetitions=NV.repetitions))
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
    println("SAFEGUARDED NEWTON + ARMIJO BACKTRACKING VALIDATION")
    println("N=$(NV.N) J/h=$(NV.J/NV.h) runs=$(NV.ntraining_runs) checkpoints=$(sort(collect(NV.checkpoint_epochs)))")
    println("M=$(NV.M) repetitions=$(NV.repetitions)")
    println("Armijo alpha=$armijo_alpha beta=$backtrack_beta max_backtracks=$max_backtracks eta_cap=$eta_cap")
    println("Compare fixed eta=.05, raw Newton, safeguarded Newton, exact line search")
    println("============================================================")
    X=NV.enumerate_states(NV.N); flips=NV.build_flip_index(X); rows=NamedTuple[]
    for run in 1:NV.ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,NV.ntraining_runs)
        append!(rows,run_training(run,X,flips))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"newton_backtracking_validation.csv")
    NV.write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/newton_backtracking_validation.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    NewtonBacktrackingValidationExperiment.main()
end
