module TFIMAdaptiveEtaV2Experiment

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Random
using Statistics
using Printf

# V2 principle: preserve the original VMC -> target -> tree -> gauge-centering ->
# chain-update pipeline. The ONLY optimizer change relative to the fixed-eta
# control is replacing eta=0.05 by a local Newton proposal for the already-fitted
# tree. Relative to V1, only the curvature estimator is corrected: it now includes
# the eta-dependence of the off-diagonal local-energy ratios.
#
# No Armijo/backtracking, no alternative sampling, no ESS correction, and no
# reweighting-based accept/reject decision are used here.

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
    mu = GBTQuantum.weighted_mean(preds, weights)
    mu == 0.0 && return tree
    nodes = copy(tree.nodes)
    @inbounds for i in eachindex(nodes)
        n = nodes[i]
        n.isleaf && (nodes[i] = GBTQuantum.Node(n.feature,n.value-mu,n.left,n.right,true))
    end
    return GBTQuantum.RegressionTree(nodes)
end

function scaled_tree(tree, eta)
    GBTQuantum.RegressionTree([
        n.isleaf ? GBTQuantum.Node(n.feature,eta*n.value,n.left,n.right,true) : n
        for n in tree.nodes
    ])
end

# For the TFIM and a real log-amplitude direction f,
#
#   E_loc^(f)(x) = D(x) f(x)
#                  - h sum_i [psi(x^i)/psi(x)] f(x^i),
#
# where x^i differs from x by one spin flip. This is the extra term that V1's
# fixed-local-energy reweighting surrogate omitted.
function f_local_energy(H, model, tree, x)
    A0 = GBTQuantum.logamplitude(model, x)
    f0 = GBTQuantum.predict(tree, x)
    z = GBTQuantum.diagonal(H, x) * f0

    @inbounds for i in 1:H.N
        x[i] = -x[i]
        ratio = exp(GBTQuantum.logamplitude(model, x) - A0)
        fi = GBTQuantum.predict(tree, x)
        z -= H.h * ratio * fi
        x[i] = -x[i]
    end
    return z
end

# Estimate the true first and second derivatives of the variational energy along
# psi_eta(x) = psi(x) exp(eta f(x)) using the SAME empirical Born batch that
# produced the weak learner:
#
#   g = 2 Cov(f, E_loc)
#
#   c = 2 <f^2 E_loc> + 2 <f E_loc^(f)>
#       - 4 E <f^2> - 4 <f> g.
#
# The final term is retained even though the fitted tree is empirically
# gauge-centered, so the estimator remains correct up to numerical centering
# error.
function adaptive_eta(H, model, batch, tree)
    f = [GBTQuantum.predict(tree, @view batch.states[j,:]) for j in axes(batch.states,1)]
    w = Float64.(batch.counts)
    el = real.(batch.local_energy)
    W = sum(w)

    Ef = sum(w .* f) / W
    Ee = sum(w .* el) / W
    mean_f2 = sum(w .* (f .* f)) / W
    mean_f2_el = sum(w .* (f .* f) .* el) / W

    g = 2.0 * sum(w .* (f .- Ef) .* (el .- Ee)) / W

    qsum = 0.0
    @inbounds for j in axes(batch.states,1)
        x = @view batch.states[j,:]
        elf = f_local_energy(H, model, tree, x)
        qsum += w[j] * f[j] * elf
    end
    Qf = qsum / W

    c = 2.0 * mean_f2_el + 2.0 * Qf - 4.0 * Ee * mean_f2 - 4.0 * Ef * g

    if !isfinite(g) || !isfinite(c) || g >= 0.0 || c <= CURVATURE_FLOOR
        return ETA_FIXED, g, c, :fallback
    end

    eta = clamp(-g/c, 0.0, ETA_CAP)
    if !isfinite(eta) || eta <= 0.0
        return ETA_FIXED, g, c, :fallback
    end
    return eta, g, c, :newton
end

function train_v2(H; adaptive::Bool)
    cfg = GBTQuantum.TrainingConfig(nsamples=NSAMPLES,epochs=EPOCHS,max_depth=MAX_DEPTH,
        eta=ETA_FIXED,burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=SEED,
        exact_diagnostics=false)

    rng = MersenneTwister(cfg.seed)
    samples = Matrix{Int8}(undef,cfg.nsamples,H.N)
    @inbounds for i in eachindex(samples)
        samples[i] = rand(rng,Bool) ? Int8(1) : Int8(-1)
    end

    model = GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    logamps = zeros(Float64,cfg.nsamples)
    for _ in 1:cfg.burn_in_sweeps
        GBTQuantum.sweep!(rng,model,samples,logamps)
    end

    eta_hist = zeros(Float64,cfg.epochs)
    g_hist = fill(NaN,cfg.epochs)
    c_hist = fill(NaN,cfg.epochs)
    exact_hist = fill(NaN,cfg.epochs)
    fallback = 0

    for epoch in 1:cfg.epochs
        batch = GBTQuantum.vmc_batch(H,model,samples)
        yA,_ = GBTQuantum.make_targets(batch)
        weights = batch.counts

        tree = GBTQuantum.grow_tree(batch.states,yA,weights;
            max_depth=cfg.max_depth,min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain)
        tree = center_tree(tree,batch.states,weights)

        eta,g,c,mode = adaptive ? adaptive_eta(H,model,batch,tree) : (ETA_FIXED,NaN,NaN,:fixed)
        fallback += mode == :fallback
        eta_hist[epoch] = eta
        g_hist[epoch] = g
        c_hist[epoch] = c

        push!(model.logamp.trees,scaled_tree(tree,eta))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:cfg.sweeps_per_epoch
            GBTQuantum.sweep!(rng,model,samples,logamps)
        end

        if epoch==1 || epoch%10==0 || epoch==cfg.epochs
            exact_hist[epoch] = real(GBTQuantum.exact_model_energy(model,H).energy)
        end
    end

    stats = GBTQuantum.exact_model_energy(model,H)
    return (model=model,E=real(stats.energy),eta=eta_hist,g=g_hist,c=c_hist,
        exact=exact_hist,fallback=fallback)
end

function main()
    println("="^68)
    println("V2: ORIGINAL TFIM VMC PIPELINE + ADAPTIVE ETA WITH TRUE CURVATURE")
    println("N=$N ratios=$RATIOS nsamples=$NSAMPLES epochs=$EPOCHS")
    println("control eta=$ETA_FIXED; adaptive eta cap=$ETA_CAP")
    println("NO Armijo, NO alternative sampling, NO ESS correction")
    println("="^68)

    rows = NamedTuple[]
    for ratio in RATIOS
        H = GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true)
        Egs = GBTQuantum.exact_ground_energy(H)
        println("\nJ/h = $ratio")

        control = train_v2(H;adaptive=false)
        adapt = train_v2(H;adaptive=true)
        ec = control.E-Egs
        ea = adapt.E-Egs

        @printf("  E_GS             = %.12f\n",Egs)
        @printf("  fixed exact E    = %.12f  error=% .3e\n",control.E,ec)
        @printf("  adaptive exact E = %.12f  error=% .3e\n",adapt.E,ea)
        @printf("  adaptive eta: mean=%.5f min=%.5f max=%.5f final=%.5f fallbacks=%d/%d\n",
            mean(adapt.eta),minimum(adapt.eta),maximum(adapt.eta),adapt.eta[end],adapt.fallback,EPOCHS)
        @printf("  variational bound fixed/adaptive = %s / %s\n",string(ec>=-1e-10),string(ea>=-1e-10))

        valid_c = filter(isfinite, adapt.c)
        cmin = isempty(valid_c) ? NaN : minimum(valid_c)
        cmax = isempty(valid_c) ? NaN : maximum(valid_c)
        @printf("  adaptive curvature: min=% .4e max=% .4e\n",cmin,cmax)

        push!(rows,(
            ratio=ratio,Egs=Egs,E_fixed=control.E,E_adaptive=adapt.E,
            error_fixed=ec,error_adaptive=ea,
            eta_mean=mean(adapt.eta),eta_min=minimum(adapt.eta),
            eta_max=maximum(adapt.eta),eta_final=adapt.eta[end],
            fallbacks=adapt.fallback,c_min=cmin,c_max=cmax,
        ))
    end

    outdir = joinpath(@__DIR__,"results")
    mkpath(outdir)
    path = joinpath(outdir,"tfim_adaptive_eta_v2.csv")
    names = propertynames(rows[1])
    open(path,"w") do io
        println(io,join(string.(names),','))
        for r in rows
            println(io,join((getproperty(r,n) for n in names),','))
        end
    end
    println("\nResults written to $path")
end

end

if abspath(PROGRAM_FILE)==@__FILE__
    TFIMAdaptiveEtaV2Experiment.main()
end
