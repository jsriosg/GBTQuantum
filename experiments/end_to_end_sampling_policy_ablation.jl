using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module EndToEndSamplingPolicyAblationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "final_sampling_distribution_ablation.jl"))
const FS = FinalSamplingDistributionAblationExperiment
const HB = FS.HB
const AC = FS.AC
const NV = FS.NV

# End-to-end test: the sampling policy now determines every successive tree and
# therefore the entire optimization trajectory.
const tree_M = 256
const validation_budget = 512
const repetitions = 10
const epsilon_fixed_small = 0.10
const epsilon_fixed_large = 0.25
const decay_tau = 24.0
const decay_epsilon0 = 0.25
const support_floor = 1e-15
const seed_base = 1_730_000_000

const policies = [
    (name="born", kind=:born),
    (name="uniform_010", kind=:fixed_uniform_010),
    (name="uniform_025", kind=:fixed_uniform_025),
    (name="uniform_decay", kind=:decay_uniform),
]

function epsilon_for(policy, epoch)
    policy.kind == :born && return 0.0
    policy.kind == :fixed_uniform_010 && return epsilon_fixed_small
    policy.kind == :fixed_uniform_025 && return epsilon_fixed_large
    policy.kind == :decay_uniform && return decay_epsilon0 * exp(-(epoch-1)/decay_tau)
    error("unknown policy $(policy.kind)")
end

function proposal(fr, eps)
    p = fr.probabilities
    eps <= 0 && return copy(p)
    u = fill(1.0/length(p), length(p))
    NV.normalize_positive((1-eps).*p .+ eps.*u)
end

function acquire_tree(rng, X, fr, q)
    draws = NV.draw_categorical_indices(rng, q, tree_M)
    mass = Dict{Int,Float64}()
    for i in draws
        mass[i] = get(mass,i,0.0) + fr.probabilities[i]/max(q[i],support_floor)
    end
    idx = sort!(collect(keys(mass)))
    w = Float64[mass[i] for i in idx]
    tree = GBTQuantum.grow_tree(X[idx,:],fr.target[idx],w;
        max_depth=NV.optimizer_max_depth,
        min_weight=NV.optimizer_min_leaf_weight,
        min_gain=NV.optimizer_min_gain)
    born_mass = sum(fr.probabilities[i] for i in idx)
    return tree,length(idx),born_mass
end

function initialize_chain(rng)
    samples=Matrix{Int8}(undef,NV.training_nsamples,NV.N)
    @inbounds for i in eachindex(samples)
        samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1)
    end
    samples
end

function run_one(policy, rep, X, flips)
    # Same seed across policies for a given repetition: common initial state.
    rng=MersenneTwister(seed_base + 100_000*rep)
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

    for epoch in 1:NV.nepochs
        # Existing VMC chain gives the reusable baseline E(0).
        base=AC.baseline_stats(H,model,samples)
        fr=NV.exact_frozen_problem(H,model,X) # diagnostics only; not used as operational energy oracle
        eps=epsilon_for(policy,epoch)
        q=proposal(fr,eps)

        # Policy-dependent acquisition stream, deterministic by policy/rep/epoch.
        arng=MersenneTwister(seed_base + 10_000_000*rep + 10_000*epoch + sum(codeunits(policy.name)))
        tree,nunique,born_mass=acquire_tree(arng,X,fr,q)
        f=NV.centered_predictions(tree,X,fr.probabilities)
        g,c=NV.analytic_derivatives(fr,X,flips,f)

        vrng=MersenneTwister(seed_base + 20_000_000*rep + 10_000*epoch + sum(codeunits(policy.name)))
        mc=HB.budgeted_armijo(vrng,fr,X,flips,f,g,c,base,validation_budget)
        eta=mc.eta
        true_before=fr.energy
        true_after=NV.exact_energy_eta(fr,X,flips,f,eta)
        delta=true_after-true_before

        cumulative_new += tree_M + mc.used
        if mc.accepted && eta > 0
            accepted_count += 1
            # f was centered by subtracting a constant. Constants are gauge in
            # log-amplitude, so applying eta to the uncentered tree is physically
            # equivalent and preserves the tree representation.
            push!(model.logamp.trees,NV.scale_tree(tree,eta))
            GBTQuantum.refresh_logamps!(logamps,model,samples)
        end

        # Advance the chain under the updated model before the next boosting step.
        for _ in 1:NV.sweeps_per_epoch
            GBTQuantum.sweep!(rng,model,samples,logamps)
        end

        push!(rows,(
            repetition=rep, policy=policy.name, epoch=epoch, epsilon=eps,
            energy_before=true_before, energy_after=true_after,
            delta_energy=delta, eta=eta, g=g, curvature=c,
            accepted=Int(mc.accepted && eta>0), safe=Int(delta<=1e-12),
            tree_samples=tree_M, validation_samples=mc.used,
            total_new_samples=tree_M+mc.used,
            cumulative_new_samples=cumulative_new,
            unique_states=nunique, represented_born_mass=born_mass,
            backtracks=mc.back, refinements=mc.refine,
            budget_exhausted=Int(mc.reason=="budget_exhausted"),
            cumulative_accepted=accepted_count,
        ))

        if epoch == 1 || epoch in NV.checkpoint_epochs || epoch == NV.nepochs
            @printf("  %-15s rep=%2d epoch=%3d eps=%.3f E=% .8f dE=% .3e eta=%.4f acc=%d cost=%d cum=%d\n",
                policy.name,rep,epoch,eps,true_after,delta,eta,
                Int(mc.accepted && eta>0),tree_M+mc.used,cumulative_new)
        end
    end
    rows
end

function summarize(rows)
    println("\nFINAL SUMMARY")
    for policy in policies
        rs=[r for r in rows if r.policy==policy.name && r.epoch==NV.nepochs]
        energies=[r.energy_after for r in rs]
        costs=[r.cumulative_new_samples for r in rs]
        accepts=[r.cumulative_accepted for r in rs]
        @printf("  %-15s Efinal=% .8f ± %.2e | cumulative samples=%.1f | accepted trees=%.1f\n",
            policy.name,mean(energies),length(energies)>1 ? std(energies) : 0.0,
            mean(costs),mean(accepts))
    end
end

function main()
    println("\n============================================================")
    println("END-TO-END SAMPLING POLICY ABLATION")
    println("N=$(NV.N) J/h=$(NV.J/NV.h) epochs=$(NV.nepochs) repetitions=$repetitions")
    println("tree acquisition M=$tree_M; validation hard budget=$validation_budget")
    println("policies: Born; fixed uniform eps=.10; fixed uniform eps=.25; decaying uniform")
    println("decay: epsilon_t=$(decay_epsilon0)*exp(-(t-1)/$(decay_tau))")
    println("each policy determines every successive tree and its own trajectory")
    println("primary comparison: energy versus cumulative new Monte Carlo samples")
    println("============================================================")

    X=NV.enumerate_states(NV.N)
    flips=NV.build_flip_index(X)
    rows=NamedTuple[]
    for rep in 1:repetitions
        println("\nRepetition $rep/$repetitions")
        for policy in policies
            append!(rows,run_one(policy,rep,X,flips))
        end
    end

    summarize(rows)
    outdir=joinpath(@__DIR__,"results")
    mkpath(outdir)
    path=joinpath(outdir,"end_to_end_sampling_policy_ablation.csv")
    NV.write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/end_to_end_sampling_policy_ablation.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    EndToEndSamplingPolicyAblationExperiment.main()
end
