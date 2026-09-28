module TFIMDerivativeSampleScaling

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Random
using Statistics
using Printf

# Frozen-checkpoint sample-scaling diagnostic.
#
# 1. Reproduce the actual adaptive-eta V2 trajectory with the canonical
#    NSAMPLES_TRAJ=512 sampler.
# 2. At selected PRE-UPDATE checkpoints, freeze both the model and the fitted,
#    gauge-centered tree.
# 3. Without changing the trajectory, draw independent Born samples of sizes
#    512, 1024, 2048, 4096, 8192 from the frozen model and re-estimate g and c.
# 4. Compare repeated MC estimates against the exact full-Hilbert-space oracle.
#
# This isolates finite-sample estimator statistics from optimizer dynamics.

const N = 8
const EPOCHS = 150
const MAX_DEPTH = 4
const ETA_FIXED = 0.05
const ETA_CAP = 0.40
const CURVATURE_FLOOR = 1e-12
const TRAJ_SEED = 1234
const NSAMPLES_TRAJ = 512
const SAMPLE_SIZES = [512, 1024, 2048, 4096, 8192]
const REPLICATES = 20
const REPL_BURN_IN = 100
const REPL_SWEEPS = 4

# Checkpoints motivated by the V2 reliability audit:
# J/h=1: epoch 8 = early estimator drift; epoch 50 = pre-failure oversized eta.
# J/h=2: epoch 4 = wrong curvature sign; epoch 10 = severe fallback;
#        epoch 50 = concentrated late regime; epoch 60 = wrong gradient sign.
const CHECKPOINTS = Dict(
    1.0 => Set([8, 50]),
    2.0 => Set([4, 10, 50, 60]),
)

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

function estimates_from_batch(H,model,batch,tree)
    f = [GBTQuantum.predict(tree,@view batch.states[j,:]) for j in axes(batch.states,1)]
    w = Float64.(batch.counts)
    el = real.(batch.local_energy)
    W = sum(w)
    Ef = sum(w .* f)/W
    Ee = sum(w .* el)/W
    Ef2 = sum(w .* f .* f)/W
    Ef2el = sum(w .* f .* f .* el)/W
    g = 2.0*sum(w .* (f .- Ef) .* (el .- Ee))/W
    qsum = 0.0
    @inbounds for j in axes(batch.states,1)
        x = @view batch.states[j,:]
        qsum += w[j]*f[j]*f_local_energy(H,model,tree,x)
    end
    Qf = qsum/W
    c = 2.0*Ef2el + 2.0*Qf - 4.0*Ee*Ef2 - 4.0*Ef*g
    eta = isfinite(g) && isfinite(c) && g < 0.0 && c > CURVATURE_FLOOR ? -g/c : NaN
    return g,c,eta
end

function v2_decision(H,model,batch,tree)
    g,c,eta_raw = estimates_from_batch(H,model,batch,tree)
    if !isfinite(g) || !isfinite(c) || g >= 0.0 || c <= CURVATURE_FLOOR || !isfinite(eta_raw) || eta_raw <= 0.0
        return ETA_FIXED,g,c,:fallback
    end
    return clamp(eta_raw,0.0,ETA_CAP),g,c,:newton
end

function exact_estimates(H,model,tree,states)
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
    sf=0.0; sf2=0.0; se=0.0; sfe=0.0; sf2e=0.0; sq=0.0; p2=0.0
    @inbounds for j in 1:d
        x = @view states[j,:]
        fj = f[j]
        el = GBTQuantum.diagonal(H,x)
        elf = GBTQuantum.diagonal(H,x)*fj
        for i in 1:H.N
            k = flipped_row(j,i)
            r = exp(A[k]-A[j])
            el -= H.h*r
            elf -= H.h*r*f[k]
        end
        wj=w[j]
        sf+=wj*fj; sf2+=wj*fj*fj; se+=wj*el; sfe+=wj*fj*el
        sf2e+=wj*fj*fj*el; sq+=wj*fj*elf
    end
    Ef=sf/W; Ef2=sf2/W; E=se/W; Efe=sfe/W; Ef2e=sf2e/W; Qf=sq/W
    g=2.0*(Efe-Ef*E)
    c=2.0*Ef2e+2.0*Qf-4.0*E*Ef2-4.0*Ef*g
    eta=isfinite(g) && isfinite(c) && g<0.0 && c>CURVATURE_FLOOR ? -g/c : NaN
    @inbounds for j in 1:d
        pj=w[j]/W
        p2 += pj*pj
    end
    return g,c,eta,1.0/p2
end

# Start each independent replicate from random configurations, equilibrate under
# the FROZEN model, then collect nsamples parallel-chain states after a few
# sweeps. This preserves the repository's ordinary Born/Metropolis sampler while
# making replicates independent of the training trajectory's current particles.
function independent_batch(H,model,nsamples,seed)
    rng = MersenneTwister(seed)
    samples = Matrix{Int8}(undef,nsamples,H.N)
    @inbounds for i in eachindex(samples)
        samples[i] = rand(rng,Bool) ? Int8(1) : Int8(-1)
    end
    logamps = zeros(Float64,nsamples)
    GBTQuantum.refresh_logamps!(logamps,model,samples)
    for _ in 1:REPL_BURN_IN
        GBTQuantum.sweep!(rng,model,samples,logamps)
    end
    for _ in 1:REPL_SWEEPS
        GBTQuantum.sweep!(rng,model,samples,logamps)
    end
    return GBTQuantum.vmc_batch(H,model,samples)
end

function empirical_stats(batch)
    p = Float64.(batch.counts)
    p ./= sum(p)
    return length(p), 1.0/sum(p .* p), maximum(p)
end

function audit_checkpoint(H,model,tree,states,ratio,epoch)
    gex,cex,etaex,exact_part = exact_estimates(H,model,tree,states)
    rows = NamedTuple[]
    @printf("\nFrozen checkpoint J/h=%.2f epoch=%d | exact g=% .4e c=% .4e eta=% .5f participation=%.2f\n",
        ratio,epoch,gex,cex,etaex,exact_part)

    for ns in SAMPLE_SIZES
        gs=Float64[]; cs=Float64[]; etas=Float64[]
        for rep in 1:REPLICATES
            # Deterministic but well-separated seed across checkpoint/size/replicate.
            seed = TRAJ_SEED + Int(round(1000ratio)) * 100000 + epoch * 1000 + ns + rep
            batch = independent_batch(H,model,ns,seed)
            g,c,eta = estimates_from_batch(H,model,batch,tree)
            nunique,part,pmax = empirical_stats(batch)
            push!(gs,g); push!(cs,c); push!(etas,eta)
            push!(rows,(
                ratio=ratio,epoch=epoch,nsamples=ns,replicate=rep,
                exact_participation=exact_part,unique_states=nunique,
                empirical_participation=part,empirical_max_mass=pmax,
                g_exact=gex,g_mc=g,g_error=g-gex,
                c_exact=cex,c_mc=c,c_error=c-cex,
                eta_exact=etaex,eta_mc=eta,
                g_sign_correct=signbit(g)==signbit(gex),
                c_sign_correct=signbit(c)==signbit(cex),
                eta_valid=isfinite(eta),
            ))
        end
        valideta = filter(isfinite,etas)
        g_rmse = sqrt(mean((gs .- gex).^2))
        c_rmse = sqrt(mean((cs .- cex).^2))
        eta_rmse = isempty(valideta) || !isfinite(etaex) ? NaN : sqrt(mean((valideta .- etaex).^2))
        @printf("  n=%5d | g mean=% .3e rmse=%.3e | c mean=% .3e rmse=%.3e | eta valid=%2d/%d mean=% .4f rmse=%.4f\n",
            ns,mean(gs),g_rmse,mean(cs),c_rmse,length(valideta),REPLICATES,
            isempty(valideta) ? NaN : mean(valideta),eta_rmse)
    end
    return rows
end

function train_to_checkpoints(H,ratio,states)
    cfg = GBTQuantum.TrainingConfig(nsamples=NSAMPLES_TRAJ,epochs=EPOCHS,max_depth=MAX_DEPTH,
        eta=ETA_FIXED,burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=TRAJ_SEED,
        exact_diagnostics=false)
    rng=MersenneTwister(cfg.seed)
    samples=Matrix{Int8}(undef,cfg.nsamples,H.N)
    @inbounds for i in eachindex(samples)
        samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1)
    end
    model=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    logamps=zeros(Float64,cfg.nsamples)
    for _ in 1:cfg.burn_in_sweeps
        GBTQuantum.sweep!(rng,model,samples,logamps)
    end
    rows=NamedTuple[]
    wanted=CHECKPOINTS[ratio]
    last_checkpoint=maximum(wanted)

    for epoch in 1:min(cfg.epochs,last_checkpoint)
        batch=GBTQuantum.vmc_batch(H,model,samples)
        yA,_=GBTQuantum.make_targets(batch)
        tree=GBTQuantum.grow_tree(batch.states,yA,batch.counts;
            max_depth=cfg.max_depth,min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain)
        tree=center_tree(tree,batch.states,batch.counts)

        eta_used,g,c,mode=v2_decision(H,model,batch,tree)

        if epoch in wanted
            @printf("Checkpoint reached on V2 trajectory: J/h=%.2f epoch=%d mode=%s eta_used=%.5f g_MC=% .3e c_MC=% .3e\n",
                ratio,epoch,String(mode),eta_used,g,c)
            append!(rows,audit_checkpoint(H,model,tree,states,ratio,epoch))
        end

        push!(model.logamp.trees,scaled_tree(tree,eta_used))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:cfg.sweeps_per_epoch
            GBTQuantum.sweep!(rng,model,samples,logamps)
        end
        if any(!isfinite,logamps)
            @printf("Trajectory became non-finite after epoch %d for J/h=%.2f.\n",epoch,ratio)
            break
        end
    end
    return rows
end

function write_csv(path,rows)
    isempty(rows) && return
    names=propertynames(rows[1])
    open(path,"w") do io
        println(io,join(string.(names),','))
        for r in rows
            println(io,join((getproperty(r,n) for n in names),','))
        end
    end
end

function main()
    println("="^82)
    println("TFIM FROZEN-CHECKPOINT DERIVATIVE SAMPLE-SCALING DIAGNOSTIC")
    println("sample sizes=$SAMPLE_SIZES replicates=$REPLICATES")
    println("trajectory sampler=$NSAMPLES_TRAJ; frozen replicate burn-in=$REPL_BURN_IN sweeps=$REPL_SWEEPS")
    println("checkpoints=$CHECKPOINTS")
    println("="^82)
    states=all_states(N)
    rows=NamedTuple[]
    for ratio in sort(collect(keys(CHECKPOINTS)))
        H=GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true)
        append!(rows,train_to_checkpoints(H,ratio,states))
    end
    outdir=joinpath(@__DIR__,"results")
    mkpath(outdir)
    path=joinpath(outdir,"tfim_derivative_sample_scaling.csv")
    write_csv(path,rows)
    println("\nResults written to $path")
    println("Use replicate-level rows to test RMSE scaling, sign-error rates, and eta reliability versus sample size.")
end

end

if abspath(PROGRAM_FILE)==@__FILE__
    TFIMDerivativeSampleScaling.main()
end
