using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module DirectCandidateArmijoValidationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "newton_step_validation.jl"))
const NV = NewtonStepValidationExperiment

const armijo_alpha = 0.10
const beta = 0.50
const max_backtracks = 12
const eta_cap = 0.40
const candidate_Ms = [256, 512, 1024]
const repetitions = 50
const candidate_burn_sweeps = 8
const candidate_sweeps_between = 1

function candidate_eta(g,c)
    !(isfinite(g) && g < 0) && return 0.0
    (isfinite(c) && c > 0) ? min(-g/c,eta_cap) : min(NV.eta_training,eta_cap)
end

# Exact categorical draws from p_eta are used here deliberately. At N=12 this
# isolates the statistical question: if samples really come from the candidate
# state, is MC Armijo reliable? A later experiment can replace this oracle draw
# by an actual Metropolis chain targeting the same p_eta.
function candidate_probabilities(fr,f,eta)
    z = log.(max.(fr.probabilities,1e-300)) .+ 2eta .* f
    m = maximum(z)
    a = exp.(z .- m)
    a ./ sum(a)
end

function candidate_local_energy(fr,X,flips,f,i,eta)
    diag = 0.0
    for j in 1:NV.N
        jp = (j == NV.N ? 1 : j+1)
        diag += -NV.J * Float64(X[i,j]) * Float64(X[i,jp])
    end
    el = diag
    for j in 1:NV.N
        k = flips[i,j]
        ratio = exp(fr.logamp[k]-fr.logamp[i]) * exp(eta*(f[k]-f[i]))
        el += -NV.h * ratio
    end
    el
end

function direct_mc_energy(rng,fr,X,flips,f,eta,M)
    peta = candidate_probabilities(fr,f,eta)
    idx = NV.draw_categorical_indices(rng,peta,M)
    vals = [candidate_local_energy(fr,X,flips,f,i,eta) for i in idx]
    mean(vals), std(vals)/sqrt(M), length(unique(idx))
end

function exact_armijo(fr,X,flips,f,g,c)
    eta = candidate_eta(g,c)
    eta == 0 && return (eta=0.0,evals=0)
    for k in 0:max_backtracks
        E = NV.exact_energy_eta(fr,X,flips,f,eta)
        if E <= fr.energy + armijo_alpha*eta*g
            return (eta=eta,evals=k+1)
        end
        eta *= beta
    end
    (eta=0.0,evals=max_backtracks+1)
end

# E(0) is estimated independently from p0. This intentionally includes the
# observable Monte Carlo noise present in a scalable implementation.
function direct_mc_armijo(rng,fr,X,flips,f,g,c,M)
    eta = candidate_eta(g,c)
    eta == 0 && return (eta=0.0,evals=0,accepted=false,E0hat=fr.energy,se0=0.0,unique=0.0)
    E0hat,se0,u0 = direct_mc_energy(rng,fr,X,flips,f,0.0,M)
    unique_total = Float64(u0)
    for k in 0:max_backtracks
        Ehat,se,u = direct_mc_energy(rng,fr,X,flips,f,eta,M)
        unique_total += u
        if isfinite(Ehat) && Ehat <= E0hat + armijo_alpha*eta*g
            return (eta=eta,evals=k+1,accepted=true,E0hat=E0hat,se0=se0,
                    unique=unique_total/(k+2))
        end
        eta *= beta
    end
    (eta=0.0,evals=max_backtracks+1,accepted=false,E0hat=E0hat,se0=se0,
     unique=unique_total/(max_backtracks+2))
end

function evaluate_tree(fr,X,flips,f,run,epoch,name,train_eps)
    g,c = NV.analytic_derivatives(fr,X,flips,f)
    eta0 = candidate_eta(g,c)
    oracle = exact_armijo(fr,X,flips,f,g,c)
    etaopt,dopt = NV.exact_line_search(fr,X,flips,f)
    rows = NamedTuple[]
    for M in candidate_Ms
        vals = NamedTuple[]
        for rep in 1:repetitions
            seed = 730_000_000 + 10_000_000*run + 100_000*epoch + 10_000*M + rep + sum(codeunits(name))
            rng = MersenneTwister(seed)
            mc = direct_mc_armijo(rng,fr,X,flips,f,g,c,M)
            dtrue = NV.exact_energy_eta(fr,X,flips,f,mc.eta)-fr.energy
            push!(vals,(eta=mc.eta,evals=Float64(mc.evals),accepted=Float64(mc.accepted),
                safe=Float64(dtrue<=1e-12),same=Float64(isapprox(mc.eta,oracle.eta;atol=1e-12)),
                dtrue=dtrue,unique=mc.unique))
        end
        mv(s)=mean(getproperty(v,s) for v in vals)
        @printf("    M=%4d oracle=%.4f MC=%.4f same=%.2f safe=%.2f evals=%.2f dE=% .3e regret=%.2e\n",
            M,oracle.eta,mv(:eta),mv(:same),mv(:safe),mv(:evals),mv(:dtrue),mv(:dtrue)-dopt)
        push!(rows,(training_run=run,epoch=epoch,method=name,training_epsilon=train_eps,
            M=M,repetitions=repetitions,energy=fr.energy,target_rms=fr.target_rms,g=g,curvature=c,
            eta_initial=eta0,eta_oracle_armijo=oracle.eta,eta_exact=etaopt,delta_exact=dopt,
            eta_mc=mv(:eta),delta_mc_true=mv(:dtrue),safe_fraction=mv(:safe),
            same_eta_fraction=mv(:same),acceptance_fraction=mv(:accepted),mean_evals=mv(:evals),
            mean_unique_candidate=mv(:unique),regret_to_exact=mv(:dtrue)-dopt))
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
        @printf("  %-15s eta0=%.4f\n",name,candidate_eta(g,c))
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
    for _ in 1:NV.burn_in_sweeps
        GBTQuantum.sweep!(rng,model,samples,logamps)
    end
    rows=NamedTuple[]
    for epoch in 1:NV.nepochs
        batch=GBTQuantum.vmc_batch(H,model,samples)
        yA,_=GBTQuantum.make_targets(batch); w=batch.counts
        if epoch in NV.checkpoint_epochs
            append!(rows,diagnose(H,model,X,flips,run,epoch))
        end
        tree=GBTQuantum.grow_tree(batch.states,yA,w;max_depth=NV.optimizer_max_depth,
            min_weight=NV.optimizer_min_leaf_weight,min_gain=NV.optimizer_min_gain)
        pred=NV.predict_all(tree,batch.states); mu=NV.weighted_mean(pred,w)
        if isfinite(mu)&&mu!=0
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
    println("DIRECT CANDIDATE-STATE MONTE CARLO ARMIJO VALIDATION")
    println("N=$(NV.N) J/h=$(NV.J/NV.h) runs=$(NV.ntraining_runs) checkpoints=$(sort(collect(NV.checkpoint_epochs)))")
    println("candidate M sweep=$candidate_Ms repetitions=$repetitions")
    println("Armijo alpha=$armijo_alpha beta=$beta max_backtracks=$max_backtracks")
    println("Candidate samples are drawn directly from p_eta; no importance reweighting")
    println("Purpose: isolate whether direct candidate-state sampling removes overlap failures")
    println("============================================================")
    X=NV.enumerate_states(NV.N); flips=NV.build_flip_index(X); rows=NamedTuple[]
    for run in 1:NV.ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,NV.ntraining_runs)
        append!(rows,run_training(run,X,flips))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"direct_candidate_armijo_validation.csv")
    NV.write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/direct_candidate_armijo_validation.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    DirectCandidateArmijoValidationExperiment.main()
end
