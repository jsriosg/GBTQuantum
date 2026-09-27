using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module BornVsDecayFinalValidationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "end_to_end_sampling_policy_ablation.jl"))
const EE = EndToEndSamplingPolicyAblationExperiment
const NV = EE.NV
const HB = EE.HB
const AC = EE.AC

const repetitions = 40
const tree_M = 256
const validation_budget = 512
const decay_epsilon0 = 0.25
const decay_tau = 24.0
const support_floor = 1e-15
const seed_base = 1_940_000_000

const policies = [
    (name="born", kind=:born),
    (name="uniform_decay", kind=:decay_uniform),
]

epsilon_for(policy, epoch) = policy.kind == :born ? 0.0 : decay_epsilon0 * exp(-(epoch-1)/decay_tau)

function proposal(fr, eps)
    p=fr.probabilities
    eps<=0 && return copy(p)
    u=fill(1.0/length(p),length(p))
    NV.normalize_positive((1-eps).*p .+ eps.*u)
end

function acquire_tree(rng,X,fr,q)
    draws=NV.draw_categorical_indices(rng,q,tree_M)
    mass=Dict{Int,Float64}()
    for i in draws
        mass[i]=get(mass,i,0.0)+fr.probabilities[i]/max(q[i],support_floor)
    end
    idx=sort!(collect(keys(mass)))
    w=Float64[mass[i] for i in idx]
    tree=GBTQuantum.grow_tree(X[idx,:],fr.target[idx],w;
        max_depth=NV.optimizer_max_depth,
        min_weight=NV.optimizer_min_leaf_weight,
        min_gain=NV.optimizer_min_gain)
    tree,length(idx),sum(fr.probabilities[i] for i in idx)
end

function initialize_chain(rng)
    samples=Matrix{Int8}(undef,NV.training_nsamples,NV.N)
    @inbounds for i in eachindex(samples)
        samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1)
    end
    samples
end

function run_one(policy,rep,X,flips)
    rng=MersenneTwister(seed_base+100_000*rep)
    H=GBTQuantum.TFIMHamiltonian(NV.N;J=NV.J,h=NV.h,periodic=true)
    samples=initialize_chain(rng)
    model=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    logamps=zeros(NV.training_nsamples)
    for _ in 1:NV.burn_in_sweeps
        GBTQuantum.sweep!(rng,model,samples,logamps)
    end

    rows=NamedTuple[]
    cumulative_new=0
    accepted_count=0
    false_accept_count=0

    for epoch in 1:NV.nepochs
        base=AC.baseline_stats(H,model,samples)
        fr=NV.exact_frozen_problem(H,model,X)
        eps=epsilon_for(policy,epoch)
        q=proposal(fr,eps)

        tag=sum(codeunits(policy.name))
        arng=MersenneTwister(seed_base+10_000_000*rep+10_000*epoch+tag)
        tree,nunique,born_mass=acquire_tree(arng,X,fr,q)
        f=NV.centered_predictions(tree,X,fr.probabilities)
        g,c=NV.analytic_derivatives(fr,X,flips,f)

        vrng=MersenneTwister(seed_base+20_000_000*rep+10_000*epoch+tag)
        mc=HB.budgeted_armijo(vrng,fr,X,flips,f,g,c,base,validation_budget)
        eta=mc.eta
        true_before=fr.energy
        true_after=NV.exact_energy_eta(fr,X,flips,f,eta)
        delta=true_after-true_before
        accepted=mc.accepted && eta>0
        false_accept=accepted && delta>1e-10

        cumulative_new += tree_M+mc.used
        accepted_count += accepted
        false_accept_count += false_accept

        if accepted
            push!(model.logamp.trees,NV.scale_tree(tree,eta))
            GBTQuantum.refresh_logamps!(logamps,model,samples)
        end
        for _ in 1:NV.sweeps_per_epoch
            GBTQuantum.sweep!(rng,model,samples,logamps)
        end

        alpha = 0.1
        exact_armijo_rhs = true_before + alpha*eta*g
        exact_armijo_slack = true_after - exact_armijo_rhs

        push!(rows,(
            repetition=rep,policy=policy.name,epoch=epoch,epsilon=eps,
            energy_before=true_before,energy_after=true_after,delta_energy=delta,
            eta=eta,g=g,curvature=c,accepted=Int(accepted),
            false_accept=Int(false_accept),exact_armijo_slack=exact_armijo_slack,
            safe=Int(delta<=1e-10),tree_samples=tree_M,
            validation_samples=mc.used,total_new_samples=tree_M+mc.used,
            cumulative_new_samples=cumulative_new,unique_states=nunique,
            represented_born_mass=born_mass,backtracks=mc.back,refinements=mc.refine,
            budget_exhausted=Int(mc.reason=="budget_exhausted"),
            cumulative_accepted=accepted_count,cumulative_false_accepts=false_accept_count,
            baseline_E0=base.E0,baseline_se_iid=base.se_iid,
            baseline_se_tau=base.se_tau,baseline_tau=base.tau,
            baseline_neff=base.neff,baseline_n=base.n,
            armijo_reason=mc.reason,
        ))

        if epoch==1 || epoch in NV.checkpoint_epochs || epoch==NV.nepochs || false_accept
            flag=false_accept ? " FALSE-ACCEPT" : ""
            @printf("  %-13s rep=%2d ep=%3d eps=%.3f E=% .8f dE=% .3e eta=%.4f acc=%d val=%d cum=%d%s\n",
                policy.name,rep,epoch,eps,true_after,delta,eta,Int(accepted),mc.used,cumulative_new,flag)
        end
    end
    rows
end

function summarize(rows)
    println("\n============================================================")
    println("FINAL VALIDATION SUMMARY")
    println("============================================================")
    for p in policies
        allp=[r for r in rows if r.policy==p.name]
        final=[r for r in allp if r.epoch==NV.nepochs]
        ef=[r.energy_after for r in final]
        costs=[r.cumulative_new_samples for r in final]
        acc=sum(r.accepted for r in allp)/length(allp)
        fa=sum(r.false_accept for r in allp)
        nacc=sum(r.accepted for r in allp)
        farate=nacc>0 ? fa/nacc : 0.0
        @printf("%-13s E64=% .8f ± %.3e | cost=%.1f | accept=%.3f | false accepts=%d/%d (%.3f)\n",
            p.name,mean(ef),std(ef),mean(costs),acc,fa,nacc,farate)
    end

    born=Dict(r.repetition=>r.energy_after for r in rows if r.policy=="born" && r.epoch==NV.nepochs)
    decay=Dict(r.repetition=>r.energy_after for r in rows if r.policy=="uniform_decay" && r.epoch==NV.nepochs)
    reps=sort!(collect(intersect(keys(born),keys(decay))))
    d=[decay[i]-born[i] for i in reps]
    @printf("\nPaired E64 difference decay-born = % .6e ± %.3e (SD), n=%d\n",mean(d),std(d),length(d))
    println("Negative means lower terminal energy for decay in that paired comparison.")
end

function main()
    println("\n============================================================")
    println("FINAL BORN VS DECAYING-UNIFORM VALIDATION")
    println("N=$(NV.N) J/h=$(NV.J/NV.h) epochs=$(NV.nepochs) repetitions=$repetitions")
    println("tree M=$tree_M; validation hard budget=$validation_budget")
    println("decay epsilon_t=$decay_epsilon0*exp(-(t-1)/$decay_tau)")
    println("primary endpoint: energy versus cumulative new MC samples")
    println("secondary: stochastic Armijo false-accept diagnostics")
    println("exact Hilbert energy is diagnostic only, never an acceptance oracle")
    println("============================================================")

    X=NV.enumerate_states(NV.N)
    flips=NV.build_flip_index(X)
    rows=NamedTuple[]
    for rep in 1:repetitions
        println("\nRepetition $rep/$repetitions")
        for p in policies
            append!(rows,run_one(p,rep,X,flips))
        end
    end
    summarize(rows)
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"born_vs_decay_final_validation.csv")
    NV.write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/born_vs_decay_final_validation.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    BornVsDecayFinalValidationExperiment.main()
end
