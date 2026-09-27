using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module NewtonStepValidationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

const N=12; const h=1.0; const J=2.0
const training_nsamples=256; const nepochs=64
const checkpoint_epochs=Set([8,32,64]); const ntraining_runs=3
const optimizer_max_depth=4; const eta_training=0.05
const burn_in_sweeps=100; const sweeps_per_epoch=1
const optimizer_min_leaf_weight=1.0; const optimizer_min_gain=0.0
const diagnostic_depth=4; const diagnostic_min_weight=1e-14
const M=1024; const repetitions=50
const base_seed=1_920_000; const diagnostic_seed_base=101_000_000
const support_floor=1e-15
const eta_max=0.40
const exact_grid_n=401

function exact_frozen_problem(H,model,X)
    p=exact_probabilities(model,X)
    loga=Float64[logamplitude(model,@view(X[i,:])) for i in axes(X,1)]
    eloc=ComplexF64[local_energy!(H,model,@view(X[i,:])) for i in axes(X,1)]
    E=real(sum(p.*eloc)); y=-real.(eloc .- E)
    return (probabilities=p,logamp=loga,eloc=real.(eloc),target=y,energy=E,
            target_rms=sqrt(sum(p.*y.^2)))
end

function normalize_positive(v)
    q=max.(Float64.(v),0.0); s=sum(q)
    if !(s>eps()) || !isfinite(s) fill!(q,1/length(q)); return q end
    q./=s; q
end

function proposals(fr)
    p=fr.probabilities; y=fr.target; d=length(p); u=fill(1/d,d)
    py2=normalize_positive(p.*y.^2)
    [(name="born",epsilon=0.0,q=copy(p)),
     (name="born_uniform",epsilon=0.25,q=normalize_positive(0.75.*p .+ 0.25.*u)),
     (name="born_sq_target",epsilon=0.25,q=normalize_positive(0.75.*p .+ 0.25.*py2))]
end

function sampled_tree(rng,X,fr,q)
    draws=draw_categorical_indices(rng,q,M); mass=Dict{Int,Float64}()
    for x in draws mass[x]=get(mass,x,0.0)+fr.probabilities[x]/max(q[x],support_floor) end
    idx=sort!(collect(keys(mass))); w=Float64[mass[x] for x in idx]
    tree=GBTQuantum.grow_tree(X[idx,:],fr.target[idx],w;
        max_depth=diagnostic_depth,min_weight=diagnostic_min_weight,min_gain=0.0)
    tree,length(idx)
end

function full_tree(X,fr)
    GBTQuantum.grow_tree(X,fr.target,fr.probabilities;
        max_depth=diagnostic_depth,min_weight=diagnostic_min_weight,min_gain=0.0)
end

# Center the weak learner in the exact Born gauge. This is the same translation
# freedom used by the training algorithm; it leaves all split decisions intact.
function centered_predictions(tree,X,p)
    f=predict_all(tree,X); f .-= sum(p.*f); f
end

# Exact TFIM energy for psi_eta(x)=psi(x)exp(eta*f(x)). Enumeration is binary
# and spin flips can be mapped once by dictionary, avoiding model reconstruction.
function build_flip_index(X)
    n=size(X,1); index=Dict{Tuple{Vararg{Int8}},Int}()
    for i in 1:n index[Tuple(@view X[i,:])]=i end
    flips=Matrix{Int}(undef,n,N)
    for i in 1:n, j in 1:N
        x=collect(@view X[i,:]); x[j]=-x[j]
        flips[i,j]=index[Tuple(x)]
    end
    flips
end

function exact_energy_eta(fr,X,flips,f,eta)
    z=fr.logamp .+ eta.*f; z .-= maximum(z); a=exp.(z)
    den=sum(abs2,a); num=0.0
    for i in axes(X,1)
        diag=0.0
        for j in 1:N
            jp=(j==N ? 1 : j+1)
            diag += -J*Float64(X[i,j])*Float64(X[i,jp])
        end
        num += diag*a[i]^2
        for j in 1:N num += -h*a[i]*a[flips[i,j]] end
    end
    num/den
end

# Analytic derivatives at eta=0. With <f>_p=0,
# E'(0)=2<f(E_loc-E)> = -2<f y>.
# E''(0)=2[<f^2(E_loc-E)> + <FHF> - E<f^2>].
function analytic_derivatives(fr,X,flips,f)
    p=fr.probabilities; E=fr.energy
    g=2*sum(p .* f .* (fr.eloc .- E))
    f2=sum(p.*f.^2)
    first=sum(p .* f.^2 .* (fr.eloc .- E))
    # <FHF> = sum_x p_x f_x sum_x' H_xx' psi_x'/psi_x f_x'.
    # Ratios follow directly from stored log amplitudes.
    fhf=0.0
    for i in axes(X,1)
        diag=0.0
        for j in 1:N
            jp=(j==N ? 1 : j+1)
            diag += -J*Float64(X[i,j])*Float64(X[i,jp])
        end
        inner=diag*f[i]
        for j in 1:N
            k=flips[i,j]
            inner += -h*exp(fr.logamp[k]-fr.logamp[i])*f[k]
        end
        fhf += p[i]*f[i]*inner
    end
    curvature=2*(first + fhf - E*f2)
    return g,curvature
end

function exact_line_search(fr,X,flips,f)
    etas=range(0.0,eta_max,length=exact_grid_n)
    energies=Float64[exact_energy_eta(fr,X,flips,f,e) for e in etas]
    i=argmin(energies)
    return Float64(etas[i]),energies[i]-fr.energy
end

function evaluate_direction(fr,X,flips,f)
    g,c=analytic_derivatives(fr,X,flips,f)
    descent=g<0
    etaN=(descent && isfinite(c) && c>0) ? clamp(-g/c,0.0,eta_max) : 0.0
    dN=exact_energy_eta(fr,X,flips,f,etaN)-fr.energy
    etaE,dE=exact_line_search(fr,X,flips,f)
    d05=exact_energy_eta(fr,X,flips,f,0.05)-fr.energy
    # Finite differences validate the analytic derivatives themselves.
    dh=1e-4
    Ep=exact_energy_eta(fr,X,flips,f,dh); Em=exact_energy_eta(fr,X,flips,f,-dh)
    gfd=(Ep-Em)/(2dh); cfd=(Ep-2fr.energy+Em)/(dh^2)
    return (g=g,c=c,eta_newton=etaN,delta_newton=dN,eta_exact=etaE,delta_exact=dE,
            delta_fixed=d05,descent=descent,g_fd=gfd,c_fd=cfd)
end

function diagnose(H,model,X,flips,run,epoch)
    fr=exact_frozen_problem(H,model,X); rows=NamedTuple[]
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e\n",epoch,fr.energy,fr.target_rms)

    ft=full_tree(X,fr); ff=centered_predictions(ft,X,fr.probabilities)
    ev=evaluate_direction(fr,X,flips,ff)
    @printf("  FULL: g=% .3e c=% .3e etaN=%.4f eta*=%.4f dEN=% .3e dE*=% .3e deriv.err=(%.1e,%.1e)\n",
        ev.g,ev.c,ev.eta_newton,ev.eta_exact,ev.delta_newton,ev.delta_exact,
        abs(ev.g-ev.g_fd),abs(ev.c-ev.c_fd))
    push!(rows,(training_run=run,epoch=epoch,method="full_hilbert",epsilon=0.0,
        energy=fr.energy,target_rms=fr.target_rms,g=ev.g,curvature=ev.c,eta_newton=ev.eta_newton,
        eta_exact=ev.eta_exact,delta_newton=ev.delta_newton,delta_exact=ev.delta_exact,
        delta_fixed=ev.delta_fixed,descent_fraction=Float64(ev.descent),mean_unique=4096.0,
        mean_abs_eta_error=abs(ev.eta_newton-ev.eta_exact),mean_energy_regret=ev.delta_newton-ev.delta_exact,
        grad_fd_error=abs(ev.g-ev.g_fd),curv_fd_error=abs(ev.c-ev.c_fd),repetitions=1))

    for pr in proposals(fr)
        vals=NamedTuple[]
        for rep in 1:repetitions
            methodhash=sum(Int(c) for c in codeunits(pr.name))+round(Int,1000*pr.epsilon)
            rng=MersenneTwister(diagnostic_seed_base+10_000_000*run+100_000*epoch+100*methodhash+rep)
            tree,nu=sampled_tree(rng,X,fr,pr.q)
            f=centered_predictions(tree,X,fr.probabilities); e=evaluate_direction(fr,X,flips,f)
            push!(vals,merge(e,(unique=Float64(nu),)))
        end
        mg=mean(v.g for v in vals); mc=mean(v.c for v in vals)
        en=mean(v.eta_newton for v in vals); ee=mean(v.eta_exact for v in vals)
        dn=mean(v.delta_newton for v in vals); de=mean(v.delta_exact for v in vals)
        df=mean(v.delta_fixed for v in vals); desc=mean(v.descent for v in vals)
        nu=mean(v.unique for v in vals); ae=mean(abs(v.eta_newton-v.eta_exact) for v in vals)
        regret=mean(v.delta_newton-v.delta_exact for v in vals)
        ge=mean(abs(v.g-v.g_fd) for v in vals); ce=mean(abs(v.c-v.c_fd) for v in vals)
        @printf("  %-15s etaN=%.4f eta*=%.4f dEN=% .3e dE*=% .3e regret=%.2e descent=%.2f unique=%.1f\n",
            pr.name,en,ee,dn,de,regret,desc,nu)
        push!(rows,(training_run=run,epoch=epoch,method=pr.name,epsilon=pr.epsilon,
            energy=fr.energy,target_rms=fr.target_rms,g=mg,curvature=mc,eta_newton=en,eta_exact=ee,
            delta_newton=dn,delta_exact=de,delta_fixed=df,descent_fraction=desc,mean_unique=nu,
            mean_abs_eta_error=ae,mean_energy_regret=regret,grad_fd_error=ge,curv_fd_error=ce,
            repetitions=repetitions))
    end
    rows
end

function run_training(run,X,flips)
    H=TFIMHamiltonian(N;J=J,h=h,periodic=true)
    rng=MersenneTwister(base_seed+10_000*run)
    samples=Matrix{Int8}(undef,training_nsamples,N)
    @inbounds for i in eachindex(samples) samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1) end
    model=LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); logamps=zeros(training_nsamples)
    for _ in 1:burn_in_sweeps GBTQuantum.sweep!(rng,model,samples,logamps) end
    rows=NamedTuple[]
    for epoch in 1:nepochs
        batch=vmc_batch(H,model,samples); yA,_=make_targets(batch); w=batch.counts
        if epoch in checkpoint_epochs append!(rows,diagnose(H,model,X,flips,run,epoch)) end
        tree=GBTQuantum.grow_tree(batch.states,yA,w;max_depth=optimizer_max_depth,
            min_weight=optimizer_min_leaf_weight,min_gain=optimizer_min_gain)
        pred=predict_all(tree,batch.states); mu=weighted_mean(pred,w)
        if isfinite(mu)&&mu!=0 tree=shift_tree_leaves(tree,mu) end
        push!(model.logamp.trees,scale_tree(tree,eta_training))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:sweeps_per_epoch GBTQuantum.sweep!(rng,model,samples,logamps) end
    end
    rows
end

function main()
    println("\n============================================================")
    println("ANALYTIC NEWTON STEP VALIDATION")
    println("N=$N J/h=$(J/h) runs=$ntraining_runs checkpoints=$(sort(collect(checkpoint_epochs)))")
    println("M=$M repetitions=$repetitions; proposals: Born, 25% uniform, 25% p*y^2")
    println("Newton eta=-E'(0)/E''(0) vs exact full-Hilbert line search [0,$eta_max]")
    println("============================================================")
    X=enumerate_states(N); flips=build_flip_index(X); rows=NamedTuple[]
    for run in 1:ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,ntraining_runs)
        append!(rows,run_training(run,X,flips))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"newton_step_validation.csv")
    write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/newton_step_validation.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    NewtonStepValidationExperiment.main()
end
