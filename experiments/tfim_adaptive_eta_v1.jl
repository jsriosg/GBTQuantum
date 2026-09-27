module TFIMAdaptiveEtaV1Experiment

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Random
using Statistics
using Printf
using LinearAlgebra

# V1 principle: preserve the original VMC -> target -> tree -> gauge-centering ->
# chain-update pipeline. The ONLY optimizer change is replacing fixed eta=0.05
# by a local quadratic/Newton proposal computed for the already-fitted tree.
# No Armijo/backtracking is used here.

const RATIOS = [0.05, 0.10, 0.25, 0.50, 1.0, 2.0]
const N = 8
const NSAMPLES = 512
const EPOCHS = 150
const MAX_DEPTH = 4
const ETA_FIXED = 0.05
const ETA_CAP = 0.40
const CURVATURE_FLOOR = 1e-12
const SEED = 1234

function center_tree(tree, X, weights)
    preds = [GBTQuantum.predict(tree, @view X[j,:]) for j in axes(X,1)]
    μ = GBTQuantum.weighted_mean(preds, weights)
    μ == 0.0 && return tree
    nodes = copy(tree.nodes)
    @inbounds for i in eachindex(nodes)
        n = nodes[i]
        n.isleaf && (nodes[i] = GBTQuantum.Node(n.feature,n.value-μ,n.left,n.right,true))
    end
    return GBTQuantum.RegressionTree(nodes)
end

function scaled_tree(tree, eta)
    GBTQuantum.RegressionTree([
        n.isleaf ? GBTQuantum.Node(n.feature,eta*n.value,n.left,n.right,true) : n
        for n in tree.nodes
    ])
end

# Estimate derivatives of E(eta) at eta=0 using the SAME empirical Born batch
# that produced the weak learner. VMCBatch stores one local energy per compressed
# unique state in `local_energy`, weighted by the corresponding `counts`.
function batch_reweighted_energy(batch, f, eta)
    w0 = Float64.(batch.counts)
    el = real.(batch.local_energy)
    z = 2.0 .* eta .* f
    zmax = maximum(z)
    rw = w0 .* exp.(z .- zmax)
    return sum(rw .* el) / sum(rw)
end

function adaptive_eta(batch, tree)
    f = [GBTQuantum.predict(tree, @view batch.states[j,:]) for j in axes(batch.states,1)]
    w = Float64.(batch.counts)
    el = real.(batch.local_energy)
    W = sum(w)
    Ef = sum(w .* f) / W
    Ee = sum(w .* el) / W
    g = 2.0 * sum(w .* (f .- Ef) .* (el .- Ee)) / W

    # Small local probe only for curvature estimation. This is NOT a line search.
    δ = 1e-3
    Ep = batch_reweighted_energy(batch,f, δ)
    E0 = batch_reweighted_energy(batch,f, 0.0)
    Em = batch_reweighted_energy(batch,f,-δ)
    c = (Ep - 2E0 + Em) / δ^2

    if !isfinite(g) || !isfinite(c) || g >= 0.0 || c <= CURVATURE_FLOOR
        return ETA_FIXED, g, c, :fallback
    end
    eta = clamp(-g/c, 0.0, ETA_CAP)
    if !isfinite(eta) || eta <= 0.0
        return ETA_FIXED, g, c, :fallback
    end
    return eta, g, c, :newton
end

function train_v1(H; adaptive::Bool)
    cfg = GBTQuantum.TrainingConfig(nsamples=NSAMPLES,epochs=EPOCHS,max_depth=MAX_DEPTH,
        eta=ETA_FIXED,burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=SEED,
        exact_diagnostics=false)
    rng = MersenneTwister(cfg.seed)
    samples = Matrix{Int8}(undef,cfg.nsamples,H.N)
    @inbounds for i in eachindex(samples); samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
    model = GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    logamps=zeros(Float64,cfg.nsamples)
    for _ in 1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,model,samples,logamps); end

    eta_hist=zeros(Float64,cfg.epochs); exact_hist=zeros(Float64,cfg.epochs)
    fallback=0
    for epoch in 1:cfg.epochs
        batch=GBTQuantum.vmc_batch(H,model,samples)
        yA,_=GBTQuantum.make_targets(batch); weights=batch.counts
        tree=GBTQuantum.grow_tree(batch.states,yA,weights;max_depth=cfg.max_depth,
            min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain)
        tree=center_tree(tree,batch.states,weights)
        eta,g,c,mode = adaptive ? adaptive_eta(batch,tree) : (ETA_FIXED,NaN,NaN,:fixed)
        fallback += mode == :fallback
        eta_hist[epoch]=eta
        push!(model.logamp.trees,scaled_tree(tree,eta))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,model,samples,logamps); end
        if epoch==1 || epoch%10==0 || epoch==cfg.epochs
            exact_hist[epoch]=real(GBTQuantum.exact_model_energy(model,H).energy)
        else
            exact_hist[epoch]=NaN
        end
    end
    stats=GBTQuantum.exact_model_energy(model,H)
    return (model=model,E=real(stats.energy),eta=eta_hist,exact=exact_hist,fallback=fallback)
end

function main()
    println("="^60)
    println("V1: ORIGINAL TFIM VMC PIPELINE + ADAPTIVE ETA ONLY")
    println("N=$N ratios=$RATIOS nsamples=$NSAMPLES epochs=$EPOCHS")
    println("control eta=$ETA_FIXED; adaptive eta cap=$ETA_CAP")
    println("NO Armijo, NO alternative sampling, NO reweighting decisions")
    println("="^60)
    rows=NamedTuple[]
    for ratio in RATIOS
        H=GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true)
        Egs=GBTQuantum.exact_ground_energy(H)
        println("\nJ/h = $ratio")
        control=train_v1(H;adaptive=false)
        adapt=train_v1(H;adaptive=true)
        ec=control.E-Egs; ea=adapt.E-Egs
        @printf("  E_GS             = %.12f\n",Egs)
        @printf("  fixed exact E    = %.12f  error=% .3e\n",control.E,ec)
        @printf("  adaptive exact E = %.12f  error=% .3e\n",adapt.E,ea)
        @printf("  adaptive eta: mean=%.5f min=%.5f max=%.5f final=%.5f fallbacks=%d/%d\n",
            mean(adapt.eta),minimum(adapt.eta),maximum(adapt.eta),adapt.eta[end],adapt.fallback,EPOCHS)
        @printf("  variational bound fixed/adaptive = %s / %s\n",string(ec>=-1e-10),string(ea>=-1e-10))
        push!(rows,(ratio=ratio,Egs=Egs,E_fixed=control.E,E_adaptive=adapt.E,
            error_fixed=ec,error_adaptive=ea,eta_mean=mean(adapt.eta),eta_min=minimum(adapt.eta),
            eta_max=maximum(adapt.eta),eta_final=adapt.eta[end],fallbacks=adapt.fallback))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"tfim_adaptive_eta_v1.csv")
    names=propertynames(rows[1])
    open(path,"w") do io
        println(io,join(string.(names),','))
        for r in rows; println(io,join((getproperty(r,n) for n in names),',')); end
    end
    println("\nResults written to $path")
end

end

if abspath(PROGRAM_FILE)==@__FILE__
    TFIMAdaptiveEtaV1Experiment.main()
end
