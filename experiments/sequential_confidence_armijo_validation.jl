using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module SequentialConfidenceArmijoValidationExperiment

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
const initial_M = 256
const max_M = 4096
const z_values = [0.0, 1.0, 2.0, 3.0]
const repetitions = 50

function candidate_eta(g,c)
    !(isfinite(g) && g < 0) && return 0.0
    (isfinite(c) && c > 0) ? min(-g/c,eta_cap) : min(NV.eta_training,eta_cap)
end

function candidate_probabilities(fr,f,eta)
    z=log.(max.(fr.probabilities,1e-300)) .+ 2eta .* f
    m=maximum(z); a=exp.(z .- m); a./sum(a)
end

function local_energy(fr,X,flips,f,i,eta)
    diag=0.0
    for j in 1:NV.N
        jp=(j==NV.N ? 1 : j+1)
        diag += -NV.J*Float64(X[i,j])*Float64(X[i,jp])
    end
    el=diag
    for j in 1:NV.N
        k=flips[i,j]
        el += -NV.h*exp(fr.logamp[k]-fr.logamp[i] + eta*(f[k]-f[i]))
    end
    el
end

function draw_energies!(rng, vals, fr,X,flips,f,eta,n)
    n<=0 && return vals
    p=candidate_probabilities(fr,f,eta)
    idx=NV.draw_categorical_indices(rng,p,n)
    append!(vals,(local_energy(fr,X,flips,f,i,eta) for i in idx))
    vals
end

stats(v) = length(v)>1 ? (mean(v),std(v)/sqrt(length(v))) : (mean(v),Inf)

function exact_armijo(fr,X,flips,f,g,c)
    eta=candidate_eta(g,c)
    eta==0 && return 0.0
    for _ in 0:max_backtracks
        NV.exact_energy_eta(fr,X,flips,f,eta) <= fr.energy + armijo_alpha*eta*g && return eta
        eta*=beta
    end
    0.0
end

# E0 is sampled once and then reused for all candidate eta values. If a decision
# is ambiguous, both E0 and the current candidate are enlarged to the next
# doubling level. This keeps the confidence calculation symmetric and simple.
function sequential_armijo(rng,fr,X,flips,f,g,c,zcrit)
    eta=candidate_eta(g,c)
    eta==0 && return (eta=0.0,accepted=false,candidate_evals=0,total_samples=0,
        refinements=0,ambiguous=0)

    v0=Float64[]
    draw_energies!(rng,v0,fr,X,flips,f,0.0,initial_M)
    total_samples=initial_M
    candidate_evals=0
    refinements=0
    ambiguous=0

    for _ in 0:max_backtracks
        veta=Float64[]
        draw_energies!(rng,veta,fr,X,flips,f,eta,initial_M)
        total_samples += initial_M
        candidate_evals += initial_M
        M=initial_M

        while true
            E0,se0=stats(v0); Ee,see=stats(veta)
            Dhat=Ee-E0-armijo_alpha*eta*g
            seD=sqrt(se0^2+see^2)

            # z=0 recovers the ordinary plug-in stochastic Armijo rule.
            if Dhat + zcrit*seD < 0
                return (eta=eta,accepted=true,candidate_evals=candidate_evals,
                    total_samples=total_samples,refinements=refinements,ambiguous=ambiguous)
            elseif Dhat - zcrit*seD > 0
                break
            elseif M >= max_M
                # At the sample cap, unresolved means we do not have evidence
                # sufficient to accept: conservatively backtrack.
                ambiguous += 1
                break
            else
                ambiguous += 1
                newM=min(2M,max_M)
                add=newM-M
                draw_energies!(rng,v0,fr,X,flips,f,0.0,add)
                draw_energies!(rng,veta,fr,X,flips,f,eta,add)
                total_samples += 2add
                candidate_evals += add
                refinements += 1
                M=newM
            end
        end
        eta*=beta
    end
    (eta=0.0,accepted=false,candidate_evals=candidate_evals,total_samples=total_samples,
     refinements=refinements,ambiguous=ambiguous)
end

function evaluate_tree(fr,X,flips,f,run,epoch,name,train_eps)
    g,c=NV.analytic_derivatives(fr,X,flips,f)
    eta0=candidate_eta(g,c)
    etaoracle=exact_armijo(fr,X,flips,f,g,c)
    etaopt,dopt=NV.exact_line_search(fr,X,flips,f)
    rows=NamedTuple[]
    for zcrit in z_values
        vals=NamedTuple[]
        for rep in 1:repetitions
            seed=910_000_000+10_000_000*run+100_000*epoch+10_000*round(Int,10zcrit)+rep+sum(codeunits(name))
            rng=MersenneTwister(seed)
            mc=sequential_armijo(rng,fr,X,flips,f,g,c,zcrit)
            dtrue=NV.exact_energy_eta(fr,X,flips,f,mc.eta)-fr.energy
            push!(vals,(eta=mc.eta,safe=Float64(dtrue<=1e-12),
                same=Float64(isapprox(mc.eta,etaoracle;atol=1e-12)),
                accepted=Float64(mc.accepted),dtrue=dtrue,
                candidate_evals=Float64(mc.candidate_evals),total_samples=Float64(mc.total_samples),
                refinements=Float64(mc.refinements),ambiguous=Float64(mc.ambiguous)))
        end
        mv(s)=mean(getproperty(v,s) for v in vals)
        @printf("    z=%.1f oracle=%.4f MC=%.4f same=%.2f safe=%.2f dE=% .3e cand.samples=%.0f total=%.0f refine=%.2f regret=%.2e\n",
            zcrit,etaoracle,mv(:eta),mv(:same),mv(:safe),mv(:dtrue),mv(:candidate_evals),
            mv(:total_samples),mv(:refinements),mv(:dtrue)-dopt)
        push!(rows,(training_run=run,epoch=epoch,method=name,training_epsilon=train_eps,
            z=zcrit,initial_M=initial_M,max_M=max_M,repetitions=repetitions,energy=fr.energy,
            target_rms=fr.target_rms,g=g,curvature=c,eta_initial=eta0,
            eta_oracle_armijo=etaoracle,eta_exact=etaopt,delta_exact=dopt,
            eta_mc=mv(:eta),delta_mc_true=mv(:dtrue),safe_fraction=mv(:safe),
            same_eta_fraction=mv(:same),acceptance_fraction=mv(:accepted),
            mean_candidate_samples=mv(:candidate_evals),mean_total_samples=mv(:total_samples),
            mean_refinements=mv(:refinements),mean_ambiguous_events=mv(:ambiguous),
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
    println("SEQUENTIAL CONFIDENCE-AWARE DIRECT-SAMPLING ARMIJO")
    println("N=$(NV.N) J/h=$(NV.J/NV.h) runs=$(NV.ntraining_runs) checkpoints=$(sort(collect(NV.checkpoint_epochs)))")
    println("initial M=$initial_M max M=$max_M repetitions=$repetitions z=$z_values")
    println("Armijo alpha=$armijo_alpha beta=$beta max_backtracks=$max_backtracks")
    println("E0 sampled once and reused; ambiguous decisions double samples")
    println("At max M, unresolved candidates are conservatively rejected/backtracked")
    println("Candidate samples are exact draws from p_eta: this still isolates optimizer statistics from MCMC mixing")
    println("============================================================")
    X=NV.enumerate_states(NV.N); flips=NV.build_flip_index(X); rows=NamedTuple[]
    for run in 1:NV.ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,NV.ntraining_runs)
        append!(rows,run_training(run,X,flips))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"sequential_confidence_armijo_validation.csv")
    NV.write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/sequential_confidence_armijo_validation.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    SequentialConfidenceArmijoValidationExperiment.main()
end
