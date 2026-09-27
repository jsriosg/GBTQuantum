using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module CouplingExplorationSweepExperiment

using GBTQuantum
using Random
using Statistics
using Printf
using DelimitedFiles

include(joinpath(@__DIR__, "born_vs_decay_final_validation.jl"))
const FV = BornVsDecayFinalValidationExperiment
const NV = FV.NV
const HB = FV.HB
const AC = FV.AC

# Screening sweep. J/h=2 is included for a directly comparable internal control;
# the previous 40-repetition result remains the higher-precision estimate there.
const ratios = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
const repetitions = 10
const tree_M = 256
const validation_budget = 512
const decay_epsilon0 = 0.25
const decay_tau = 24.0
const support_floor = 1e-15
const seed_base = 2_160_000_000
const policies = [(name="born", kind=:born), (name="uniform_decay", kind=:decay)]

epsilon_for(p,t) = p.kind == :born ? 0.0 : decay_epsilon0*exp(-(t-1)/decay_tau)

function proposal(fr,eps)
    p=fr.probabilities
    eps<=0 && return copy(p)
    u=fill(1/length(p),length(p))
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
        max_depth=NV.optimizer_max_depth,min_weight=NV.optimizer_min_leaf_weight,
        min_gain=NV.optimizer_min_gain)
    return tree,length(idx),sum(fr.probabilities[i] for i in idx)
end

function initialize_chain(rng)
    s=Matrix{Int8}(undef,NV.training_nsamples,NV.N)
    @inbounds for i in eachindex(s); s[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
    s
end

# Exact ground-state energy for N=12, used ONLY as a benchmark diagnostic.
# We construct the dense TFIM Hamiltonian in the same z basis convention.
function exact_ground_energy(N,J,h)
    D=1<<N
    H=zeros(Float64,D,D)
    for b in 0:D-1
        diag=0.0
        for j in 0:N-1
            k=(j+1)%N
            sj=((b>>j)&1)==1 ? 1.0 : -1.0
            sk=((b>>k)&1)==1 ? 1.0 : -1.0
            diag += -J*sj*sk
            bp=b ⊻ (1<<j)
            H[b+1,bp+1] += -h
        end
        H[b+1,b+1] += diag
    end
    eigmin(Symmetric(H))
end

# Avoid requiring LinearAlgebra/Symmetric by power-shifting if unavailable in
# callers: LinearAlgebra is stdlib and imported locally here.
using LinearAlgebra

function run_one(ratio,policy,rep,X,flips,Egs)
    J=ratio; h=1.0
    ratio_tag=round(Int,1000ratio)
    rng=MersenneTwister(seed_base+100_000*rep+ratio_tag)
    H=GBTQuantum.TFIMHamiltonian(NV.N;J=J,h=h,periodic=true)
    samples=initialize_chain(rng)
    model=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    logamps=zeros(NV.training_nsamples)
    for _ in 1:NV.burn_in_sweeps; GBTQuantum.sweep!(rng,model,samples,logamps); end

    rows=NamedTuple[]; cumulative=0; nacc=0; nfalse=0
    for epoch in 1:NV.nepochs
        base=AC.baseline_stats(H,model,samples)
        fr=NV.exact_frozen_problem(H,model,X)
        eps=epsilon_for(policy,epoch); q=proposal(fr,eps)
        tag=sum(codeunits(policy.name))
        arng=MersenneTwister(seed_base+10_000_000*rep+100_000*ratio_tag+10_000*epoch+tag)
        tree,nunique,bmass=acquire_tree(arng,X,fr,q)
        f=NV.centered_predictions(tree,X,fr.probabilities)
        g,c=NV.analytic_derivatives(fr,X,flips,f)
        vrng=MersenneTwister(seed_base+20_000_000*rep+100_000*ratio_tag+10_000*epoch+tag)
        mc=HB.budgeted_armijo(vrng,fr,X,flips,f,g,c,base,validation_budget)
        eta=mc.eta; E0=fr.energy; E1=NV.exact_energy_eta(fr,X,flips,f,eta)
        dE=E1-E0; accepted=mc.accepted && eta>0; falseacc=accepted && dE>1e-10
        cumulative += tree_M+mc.used; nacc += accepted; nfalse += falseacc
        if accepted
            push!(model.logamp.trees,NV.scale_tree(tree,eta)); GBTQuantum.refresh_logamps!(logamps,model,samples)
        end
        for _ in 1:NV.sweeps_per_epoch; GBTQuantum.sweep!(rng,model,samples,logamps); end
        push!(rows,(ratio=ratio,repetition=rep,policy=policy.name,epoch=epoch,epsilon=eps,
            energy=E1,ground_energy=Egs,energy_error=E1-Egs,delta_energy=dE,eta=eta,
            accepted=Int(accepted),false_accept=Int(falseacc),validation_samples=mc.used,
            cumulative_samples=cumulative,unique_states=nunique,represented_born_mass=bmass,
            backtracks=mc.back,refinements=mc.refine,budget_exhausted=Int(mc.reason=="budget_exhausted"),
            cumulative_accepted=nacc,cumulative_false_accepts=nfalse,baseline_tau=base.tau,
            baseline_neff=base.neff))
    end
    rows
end

function write_summary(path,rows)
    open(path,"w") do io
        println(io,"ratio,policy,mean_final_energy,sd_final_energy,mean_energy_error,sd_energy_error,mean_cost,mean_unique,mean_born_mass,accept_rate,false_accept_rate")
        for r in ratios, p in policies
            a=[x for x in rows if x.ratio==r && x.policy==p.name]
            f=[x for x in a if x.epoch==NV.nepochs]
            acc=sum(x.accepted for x in a); fa=sum(x.false_accept for x in a)
            vals=(r,p.name,mean(x.energy for x in f),std(x.energy for x in f),mean(x.energy_error for x in f),std(x.energy_error for x in f),mean(x.cumulative_samples for x in f),mean(x.unique_states for x in a),mean(x.represented_born_mass for x in a),acc/length(a),acc>0 ? fa/acc : 0.0)
            println(io,join(vals,','))
        end
    end
end

# Plotting is optional so the numerical experiment still runs if Plots.jl is
# absent. Install once with: julia --project=. -e 'using Pkg; Pkg.add("Plots")'
function make_plots(summary_path,outdir)
    try
        @eval using Plots
    catch err
        println("\nPlots.jl not available; CSVs were produced normally.")
        println("To enable figures: julia --project=. -e 'using Pkg; Pkg.add(\"Plots\")'")
        return
    end
    dat=readdlm(summary_path,',',Any,'\n';header=true)[1]
    # fixed row order: for each ratio, Born then decay
    rs=Float64[dat[i,1] for i in 1:size(dat,1)]
    pol=String[string(dat[i,2]) for i in 1:size(dat,1)]
    function series(col,name)
        Float64[dat[i,col] for i in 1:size(dat,1) if pol[i]==name]
    end
    x=unique(rs)
    p1=Plots.plot(x,series(5,"born"),marker=:circle,label="Born",xlabel="J/h",ylabel="E - E_GS",title="Final energy error vs coupling")
    Plots.plot!(p1,x,series(5,"uniform_decay"),marker=:circle,label="Decaying uniform")
    Plots.savefig(p1,joinpath(outdir,"coupling_final_energy_error.png"))
    p2=Plots.plot(x,series(7,"born"),marker=:circle,label="Born",xlabel="J/h",ylabel="Cumulative new MC samples",title="Sampling cost vs coupling")
    Plots.plot!(p2,x,series(7,"uniform_decay"),marker=:circle,label="Decaying uniform")
    Plots.savefig(p2,joinpath(outdir,"coupling_sampling_cost.png"))
    p3=Plots.plot(x,series(8,"born"),marker=:circle,label="Born",xlabel="J/h",ylabel="Mean unique states / tree",title="Tree-sample diversity vs coupling")
    Plots.plot!(p3,x,series(8,"uniform_decay"),marker=:circle,label="Decaying uniform")
    Plots.savefig(p3,joinpath(outdir,"coupling_unique_states.png"))
    println("Plots written to experiments/results/coupling_*.png")
end

function main()
    println("\n============================================================")
    println("COUPLING SWEEP: BORN VS DECAYING EXPLORATION")
    println("ratios=$ratios repetitions=$repetitions epochs=$(NV.nepochs)")
    println("tree M=$tree_M validation budget=$validation_budget")
    println("decay epsilon_t=$decay_epsilon0*exp(-(t-1)/$decay_tau)")
    println("============================================================")
    X=NV.enumerate_states(NV.N); flips=NV.build_flip_index(X); rows=NamedTuple[]
    for ratio in ratios
        println("\nJ/h=$ratio: computing exact benchmark ground energy...")
        Egs=exact_ground_energy(NV.N,ratio,1.0)
        @printf("E_GS = %.10f\n",Egs)
        for rep in 1:repetitions
            println("  repetition $rep/$repetitions")
            for p in policies
                rr=run_one(ratio,p,rep,X,flips,Egs); append!(rows,rr)
                last=rr[end]
                @printf("    %-13s E64=% .8f error=% .3e cost=%d unique(avg)=%.1f\n",p.name,last.energy,last.energy_error,last.cumulative_samples,mean(x.unique_states for x in rr))
            end
        end
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    raw=joinpath(outdir,"coupling_exploration_sweep.csv"); NV.write_namedtuple_csv(raw,rows)
    summary=joinpath(outdir,"coupling_exploration_summary.csv"); write_summary(summary,rows)
    make_plots(summary,outdir)
    println("\nRaw results: experiments/results/coupling_exploration_sweep.csv")
    println("Summary:     experiments/results/coupling_exploration_summary.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    CouplingExplorationSweepExperiment.main()
end
