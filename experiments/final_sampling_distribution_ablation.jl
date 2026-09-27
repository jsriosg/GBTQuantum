using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module FinalSamplingDistributionAblationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "armijo_hard_budget_validation.jl"))
const HB = ArmijoHardBudgetValidationExperiment
const AC = HB.AC
const SC = HB.SC
const NV = HB.NV

# Final controlled sampling ablation.  The tree learner and the step-size
# machinery are held fixed; only the distribution used to acquire the tree
# training data is changed.
const training_sample_sizes = [256, 512, 1024]
const exploration_eps = [0.05, 0.10, 0.25]
const repetitions = 50
const validation_budget = 512
const support_floor = 1e-15
const diagnostic_seed_base = 1_310_000_000

normalize_positive(v) = NV.normalize_positive(v)

function proposal_set(fr)
    p = fr.probabilities
    y = fr.target
    d = length(p)
    u = fill(1.0/d, d)
    py2 = normalize_positive(p .* y.^2)

    out = NamedTuple[(name="born", epsilon=0.0, q=copy(p))]
    for eps in exploration_eps
        push!(out, (name="born_uniform", epsilon=eps,
                    q=normalize_positive((1-eps).*p .+ eps.*u)))
        push!(out, (name="born_sq_target", epsilon=eps,
                    q=normalize_positive((1-eps).*p .+ eps.*py2)))
    end
    out
end

# Draw M configurations from q, but fit the same Born-weighted MSE objective
# using importance weights p/q. Repeated configurations are compressed exactly.
function sampled_tree(rng, X, fr, q, M)
    draws = NV.draw_categorical_indices(rng, q, M)
    mass = Dict{Int,Float64}()
    for i in draws
        mass[i] = get(mass, i, 0.0) + fr.probabilities[i] / max(q[i], support_floor)
    end
    idx = sort!(collect(keys(mass)))
    w = Float64[mass[i] for i in idx]
    tree = GBTQuantum.grow_tree(X[idx,:], fr.target[idx], w;
        max_depth=NV.diagnostic_depth,
        min_weight=NV.diagnostic_min_weight,
        min_gain=0.0)
    return tree, length(idx)
end

function represented_born_mass(draws_unique, p)
    sum(p[i] for i in draws_unique)
end

# Same as sampled_tree, but also return coverage diagnostics without changing
# the fitted learner.
function sampled_tree_with_coverage(rng, X, fr, q, M)
    draws = NV.draw_categorical_indices(rng, q, M)
    mass = Dict{Int,Float64}()
    for i in draws
        mass[i] = get(mass, i, 0.0) + fr.probabilities[i] / max(q[i], support_floor)
    end
    idx = sort!(collect(keys(mass)))
    w = Float64[mass[i] for i in idx]
    tree = GBTQuantum.grow_tree(X[idx,:], fr.target[idx], w;
        max_depth=NV.diagnostic_depth,
        min_weight=NV.diagnostic_min_weight,
        min_gain=0.0)
    return tree, length(idx), represented_born_mass(idx, fr.probabilities)
end

seed(run, epoch, name, eps, M, rep) =
    diagnostic_seed_base + 10_000_000*run + 100_000*epoch +
    10_000*M + 100*round(Int,100eps) + rep + sum(codeunits(name))

function one_repetition(fr, X, flips, base, pr, M, run, epoch, rep)
    rng = MersenneTwister(seed(run,epoch,pr.name,pr.epsilon,M,rep))
    tree, nunique, bornmass = sampled_tree_with_coverage(rng,X,fr,pr.q,M)
    f = NV.centered_predictions(tree,X,fr.probabilities)

    g,c = NV.analytic_derivatives(fr,X,flips,f)
    descent = g < 0
    eta_exact, delta_exact = NV.exact_line_search(fr,X,flips,f)

    # Independent RNG stream for validation so acquisition randomness and
    # Armijo randomness do not accidentally share a stream.
    vrng = MersenneTwister(seed(run,epoch,pr.name,pr.epsilon,M,rep) + 4_000_000_000)
    mc = HB.budgeted_armijo(vrng,fr,X,flips,f,g,c,base,validation_budget)
    delta_mc = NV.exact_energy_eta(fr,X,flips,f,mc.eta) - fr.energy
    improvement = max(-delta_mc,0.0)
    total_new = M + mc.used

    return (g=g, curvature=c, descent=Float64(descent),
        eta_exact=eta_exact, delta_exact=delta_exact,
        eta_mc=mc.eta, delta_mc=delta_mc,
        accepted=Float64(mc.accepted), safe=Float64(delta_mc <= 1e-12),
        validation_samples=Float64(mc.used), total_new_samples=Float64(total_new),
        exhausted=Float64(mc.reason=="budget_exhausted"),
        backtracks=Float64(mc.back), refinements=Float64(mc.refine),
        unique=Float64(nunique), born_mass=bornmass,
        efficiency=total_new>0 ? improvement/total_new : NaN,
        regret=delta_mc-delta_exact)
end

function aggregate(fr, X, flips, base, pr, M, run, epoch)
    vals = [one_repetition(fr,X,flips,base,pr,M,run,epoch,rep)
            for rep in 1:repetitions]
    mv(s) = mean(getproperty(v,s) for v in vals)

    @printf("  %-16s eps=%4.2f M=%4d | descent=%.2f accept=%.2f safe=%.2f dE=% .3e exact=% .3e unique=%6.1f mass=%.3f val=%5.1f eff=%.3e\n",
        pr.name,pr.epsilon,M,mv(:descent),mv(:accepted),mv(:safe),
        mv(:delta_mc),mv(:delta_exact),mv(:unique),mv(:born_mass),
        mv(:validation_samples),mv(:efficiency))

    (training_run=run, epoch=epoch, method=pr.name, epsilon=pr.epsilon,
     tree_sample_size=M, repetitions=repetitions, energy=fr.energy,
     target_rms=fr.target_rms, mean_g=mv(:g), mean_curvature=mv(:curvature),
     descent_fraction=mv(:descent), accept_fraction=mv(:accepted),
     safe_fraction=mv(:safe), mean_eta_mc=mv(:eta_mc),
     mean_delta_mc_true=mv(:delta_mc), mean_eta_exact=mv(:eta_exact),
     mean_delta_exact=mv(:delta_exact), mean_regret=mv(:regret),
     mean_unique=mv(:unique), mean_represented_born_mass=mv(:born_mass),
     mean_validation_samples=mv(:validation_samples),
     mean_total_new_samples=mv(:total_new_samples),
     budget_exhausted_fraction=mv(:exhausted), mean_backtracks=mv(:backtracks),
     mean_refinements=mv(:refinements),
     mean_improvement_per_total_sample=mv(:efficiency))
end

function full_hilbert_reference(fr,X,flips,base,run,epoch)
    tree = NV.full_tree(X,fr)
    f = NV.centered_predictions(tree,X,fr.probabilities)
    g,c = NV.analytic_derivatives(fr,X,flips,f)
    eta_exact,delta_exact = NV.exact_line_search(fr,X,flips,f)
    # This is a direction ceiling, not a sampling-cost competitor.
    @printf("  FULL HILBERT reference | g=% .3e eta*=%.4f dE*=% .3e\n",g,eta_exact,delta_exact)
    (training_run=run,epoch=epoch,method="full_hilbert",epsilon=0.0,
     tree_sample_size=length(fr.probabilities),repetitions=1,energy=fr.energy,
     target_rms=fr.target_rms,mean_g=g,mean_curvature=c,
     descent_fraction=Float64(g<0),accept_fraction=NaN,safe_fraction=NaN,
     mean_eta_mc=NaN,mean_delta_mc_true=NaN,mean_eta_exact=eta_exact,
     mean_delta_exact=delta_exact,mean_regret=NaN,mean_unique=length(fr.probabilities),
     mean_represented_born_mass=1.0,mean_validation_samples=0.0,
     mean_total_new_samples=NaN,budget_exhausted_fraction=NaN,
     mean_backtracks=NaN,mean_refinements=NaN,
     mean_improvement_per_total_sample=NaN)
end

function diagnose(H,model,samples,X,flips,run,epoch)
    fr = NV.exact_frozen_problem(H,model,X)
    base = AC.baseline_stats(H,model,samples)
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e | reused E0=% .8f SE=%.2e\n",
        epoch,fr.energy,fr.target_rms,base.E0,base.se_iid)

    rows=NamedTuple[full_hilbert_reference(fr,X,flips,base,run,epoch)]
    for M in training_sample_sizes
        for pr in proposal_set(fr)
            push!(rows,aggregate(fr,X,flips,base,pr,M,run,epoch))
        end
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

        # Keep the trajectory identical to the established baseline optimizer.
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
    println("FINAL SAMPLING-DISTRIBUTION ABLATION")
    println("N=$(NV.N) J/h=$(NV.J/NV.h) runs=$(NV.ntraining_runs) checkpoints=$(sort(collect(NV.checkpoint_epochs)))")
    println("tree M=$training_sample_sizes repetitions=$repetitions")
    println("proposals: Born; Born+uniform; Born+p*y^2, eps=$exploration_eps")
    println("same weighted-MSE tree for all proposals; importance weights p/q")
    println("step: Newton proposal + stochastic Armijo, reused E(0), hard validation budget=$validation_budget")
    println("primary metric: true energy improvement per TOTAL new sample (tree acquisition + validation)")
    println("diagnostics: descent, acceptance, safety, exact directional optimum, unique states, represented Born mass")
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
    path=joinpath(outdir,"final_sampling_distribution_ablation.csv")
    NV.write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/final_sampling_distribution_ablation.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    FinalSamplingDistributionAblationExperiment.main()
end
