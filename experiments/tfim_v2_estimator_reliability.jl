module TFIMV2EstimatorReliability

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Random
using Statistics
using Printf

# Diagnostic: follow the ACTUAL adaptive-eta V2 trajectory and compare its
# Monte Carlo estimates (g_MC, c_MC, eta_MC) against full-Hilbert-space oracle
# values for the same fitted tree and pre-update model.
#
# No optimizer/sampler changes are introduced. This experiment is intended to
# locate when estimator reliability is lost and relate that loss to empirical
# sample concentration.

const RATIOS = [0.05, 0.5, 1.0, 2.0]
const N = 8
const NSAMPLES = 512
const EPOCHS = 150
const MAX_DEPTH = 4
const ETA_FIXED = 0.05
const ETA_CAP = 0.40
const CURVATURE_FLOOR = 1e-12
const SEED = 1234
const AUDIT_EPOCHS = Set(vcat(collect(1:10), [15,20,25,30,40,50,60,75,100,125,150]))

function center_tree(tree, X, weights)
    preds = [GBTQuantum.predict(tree, @view X[j,:]) for j in axes(X,1)]
    mu = GBTQuantum.weighted_mean(preds, weights)
    mu == 0.0 && return tree
    nodes = copy(tree.nodes)
    @inbounds for i in eachindex(nodes)
        n = nodes[i]
        if n.isleaf
            nodes[i] = GBTQuantum.Node(n.feature,n.value-mu,n.left,n.right,true)
        end
    end
    return GBTQuantum.RegressionTree(nodes)
end

function scaled_tree(tree, eta)
    GBTQuantum.RegressionTree([
        n.isleaf ? GBTQuantum.Node(n.feature,eta*n.value,n.left,n.right,true) : n
        for n in tree.nodes
    ])
end

function all_states(N)
    d = 1 << N
    X = Matrix{Int8}(undef,d,N)
    @inbounds for s in 0:(d-1), i in 1:N
        X[s+1,i] = ((s >> (i-1)) & 1) == 1 ? Int8(1) : Int8(-1)
    end
    return X
end

@inline flipped_row(j,i) = ((j-1) ⊻ (1 << (i-1))) + 1

function f_local_energy(H, model, tree, x)
    A0 = GBTQuantum.logamplitude(model,x)
    f0 = GBTQuantum.predict(tree,x)
    z = GBTQuantum.diagonal(H,x)*f0
    @inbounds for i in 1:H.N
        x[i] = -x[i]
        ratio = exp(GBTQuantum.logamplitude(model,x)-A0)
        fi = GBTQuantum.predict(tree,x)
        z -= H.h*ratio*fi
        x[i] = -x[i]
    end
    return z
end

# Same MC derivative estimator and decision rule used by adaptive-eta V2.
function mc_estimates(H, model, batch, tree)
    f = [GBTQuantum.predict(tree,@view batch.states[j,:]) for j in axes(batch.states,1)]
    w = Float64.(batch.counts)
    el = real.(batch.local_energy)
    W = sum(w)

    Ef = sum(w .* f)/W
    Ee = sum(w .* el)/W
    Ef2 = sum(w .* (f .* f))/W
    Ef2el = sum(w .* (f .* f) .* el)/W
    g = 2.0*sum(w .* (f .- Ef) .* (el .- Ee))/W

    qsum = 0.0
    @inbounds for j in axes(batch.states,1)
        x = @view batch.states[j,:]
        qsum += w[j]*f[j]*f_local_energy(H,model,tree,x)
    end
    Qf = qsum/W
    c = 2.0*Ef2el + 2.0*Qf - 4.0*Ee*Ef2 - 4.0*Ef*g

    if !isfinite(g) || !isfinite(c) || g >= 0.0 || c <= CURVATURE_FLOOR
        return g,c,ETA_FIXED,:fallback
    end
    eta_raw = -g/c
    eta = clamp(eta_raw,0.0,ETA_CAP)
    if !isfinite(eta) || eta <= 0.0
        return g,c,ETA_FIXED,:fallback
    end
    return g,c,eta,:newton
end

# Full-Hilbert-space oracle for the SAME model and fitted tree.
function exact_estimates(H, model, tree, states)
    d = size(states,1)
    A = Vector{Float64}(undef,d)
    f = Vector{Float64}(undef,d)
    @inbounds for j in 1:d
        x = @view states[j,:]
        A[j] = GBTQuantum.logamplitude(model,x)
        f[j] = GBTQuantum.predict(tree,x)
    end

    shift = maximum(2.0 .* A)
    w = exp.(2.0 .* A .- shift)
    W = sum(w)

    sf=0.0; sf2=0.0; se=0.0; sfe=0.0; sf2e=0.0; sq=0.0
    p2sum = 0.0
    pmax = 0.0

    @inbounds for j in 1:d
        x = @view states[j,:]
        fj = f[j]
        el = GBTQuantum.diagonal(H,x)
        elf = GBTQuantum.diagonal(H,x)*fj
        for i in 1:H.N
            k = flipped_row(j,i)
            ratio = exp(A[k]-A[j])
            el -= H.h*ratio
            elf -= H.h*ratio*f[k]
        end
        wj = w[j]
        sf += wj*fj
        sf2 += wj*fj*fj
        se += wj*el
        sfe += wj*fj*el
        sf2e += wj*fj*fj*el
        sq += wj*fj*elf
    end

    Ef=sf/W; Ef2=sf2/W; E=se/W; Efe=sfe/W; Ef2e=sf2e/W; Qf=sq/W
    g = 2.0*(Efe-Ef*E)
    c = 2.0*Ef2e + 2.0*Qf - 4.0*E*Ef2 - 4.0*Ef*g
    eta = isfinite(g) && isfinite(c) && g < 0.0 && c > CURVATURE_FLOOR ? -g/c : NaN

    # Exact-distribution concentration diagnostics.
    @inbounds for j in 1:d
        pj = w[j]/W
        p2sum += pj*pj
        pmax = max(pmax,pj)
    end
    participation = 1.0/p2sum
    return g,c,eta,E,participation,pmax
end

function empirical_diagnostics(batch)
    counts = Float64.(batch.counts)
    W = sum(counts)
    p = counts ./ W
    unique_states = length(counts)
    empirical_participation = 1.0/sum(p .* p)
    max_empirical_mass = maximum(p)
    return unique_states,empirical_participation,max_empirical_mass
end

relerr(a,b) = isfinite(a) && isfinite(b) ? abs(a-b)/max(abs(b),1e-14) : NaN

function train_and_audit(H,ratio,states)
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

    rows = NamedTuple[]
    failed = false

    for epoch in 1:cfg.epochs
        batch = GBTQuantum.vmc_batch(H,model,samples)
        yA,_ = GBTQuantum.make_targets(batch)
        weights = batch.counts
        tree = GBTQuantum.grow_tree(batch.states,yA,weights;
            max_depth=cfg.max_depth,min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain)
        tree = center_tree(tree,batch.states,weights)

        gmc,cmc,eta_used,mode = mc_estimates(H,model,batch,tree)

        if epoch in AUDIT_EPOCHS
            gex,cex,etaex,Eex,exact_part,pmax = exact_estimates(H,model,tree,states)
            nunique,emp_part,emp_max = empirical_diagnostics(batch)
            eta_mc_raw = isfinite(gmc) && isfinite(cmc) && cmc != 0.0 ? -gmc/cmc : NaN
            push!(rows,(
                ratio=ratio,epoch=epoch,mode=String(mode),
                unique_states=nunique,empirical_participation=emp_part,
                empirical_max_mass=emp_max,exact_participation=exact_part,
                exact_max_mass=pmax,E_exact_preupdate=Eex,
                g_mc=gmc,g_exact=gex,g_abs_error=gmc-gex,g_rel_error=relerr(gmc,gex),
                c_mc=cmc,c_exact=cex,c_abs_error=cmc-cex,c_rel_error=relerr(cmc,cex),
                eta_mc_raw=eta_mc_raw,eta_used=eta_used,eta_exact=etaex,
                eta_rel_error=relerr(eta_mc_raw,etaex),
            ))
            @printf("J/h=%4.2f ep=%3d %-8s unique=%3d part=%6.1f/%6.1f | g=% .3e/% .3e c=% .3e/% .3e eta=% .4f/% .4f\n",
                ratio,epoch,String(mode),nunique,emp_part,exact_part,gmc,gex,cmc,cex,eta_mc_raw,etaex)
        end

        push!(model.logamp.trees,scaled_tree(tree,eta_used))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:cfg.sweeps_per_epoch
            GBTQuantum.sweep!(rng,model,samples,logamps)
        end

        if any(!isfinite,logamps)
            @printf("J/h=%4.2f trajectory became non-finite after epoch %d; stopping this ratio.\n",ratio,epoch)
            failed = true
            break
        end
    end
    return rows,failed
end

function write_csv(path,rows)
    isempty(rows) && return
    names = propertynames(rows[1])
    open(path,"w") do io
        println(io,join(string.(names),','))
        for r in rows
            println(io,join((getproperty(r,n) for n in names),','))
        end
    end
end

function main()
    println("="^78)
    println("TFIM V2 ESTIMATOR RELIABILITY DIAGNOSTIC")
    println("Actual adaptive-eta V2 trajectory; exact oracle is diagnostic only")
    println("N=$N ratios=$RATIOS nsamples=$NSAMPLES epochs=$EPOCHS eta_cap=$ETA_CAP")
    println("audit epochs=$(sort!(collect(AUDIT_EPOCHS)))")
    println("="^78)

    states = all_states(N)
    rows = NamedTuple[]
    for ratio in RATIOS
        println("\nJ/h = $ratio")
        rr,failed = train_and_audit(GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true),ratio,states)
        append!(rows,rr)
        failed && println("  status: non-finite trajectory detected")
    end

    outdir = joinpath(@__DIR__,"results")
    mkpath(outdir)
    path = joinpath(outdir,"tfim_v2_estimator_reliability.csv")
    write_csv(path,rows)
    println("\nResults written to $path")
    println("Columns compare MC/oracle g, c, eta and empirical/exact concentration.")
end

end

if abspath(PROGRAM_FILE)==@__FILE__
    TFIMV2EstimatorReliability.main()
end
