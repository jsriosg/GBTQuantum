module TFIMEtaEnergyProfile

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Random
using Statistics
using Printf

# Diagnostic only: along selected fitted tree directions, compare the exact
# one-dimensional variational energy E(eta) with the local quadratic model
# E(0) + g*eta + 0.5*c*eta^2. The training trajectory itself remains the
# canonical fixed-eta=0.05 trajectory.
#
# IMPORTANT: g and c are evaluated analytically from the full Hilbert-space
# distribution. We deliberately do not obtain the curvature by subtracting
# three nearly equal energies, which was numerically unstable in the first
# version of this diagnostic.

const RATIOS = [0.05, 0.5, 1.0, 2.0]
const N = 8
const NSAMPLES = 512
const EPOCHS = 150
const MAX_DEPTH = 4
const ETA_FIXED = 0.05
const SEED = 1234
const AUDIT_EPOCHS = Set([1, 10, 50, 100])
const ETA_GRID = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
const CURVATURE_FLOOR = 1e-12

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

# Precompute A(x) and f(x) on the complete Hilbert space. Besides avoiding
# repeated tree/model evaluation, this prevents accidental mutation of a state
# view while looking up single-spin-flip neighbours.
function exact_path_data(model, tree, states)
    d = size(states,1)
    A = Vector{Float64}(undef,d)
    f = Vector{Float64}(undef,d)
    @inbounds for j in 1:d
        x = @view states[j,:]
        A[j] = GBTQuantum.logamplitude(model,x)
        f[j] = GBTQuantum.predict(tree,x)
    end
    return A,f
end

# With the enumeration used by all_states, flipping spin i toggles bit i-1,
# hence the zero-based state index s maps to s xor 2^(i-1).
@inline flipped_row(j, i) = ((j-1) ⊻ (1 << (i-1))) + 1

# Exact energy along A_eta(x)=A(x)+eta*f(x), without modifying model/states.
function exact_path_energy(H, states, A, f, eta)
    d = size(states,1)
    logw = 2.0 .* (A .+ eta .* f)
    shift = maximum(logw)
    w = exp.(logw .- shift)
    Z = sum(w)

    Esum = 0.0
    @inbounds for j in 1:d
        x = @view states[j,:]
        Aeta0 = A[j] + eta*f[j]
        el = GBTQuantum.diagonal(H,x)
        for i in 1:H.N
            k = flipped_row(j,i)
            Aetai = A[k] + eta*f[k]
            el -= H.h * exp(Aetai-Aeta0)
        end
        Esum += w[j]*el
    end
    return Esum/Z
end

# Exact full-Hilbert-space derivatives for
#     psi_eta(x) = psi(x) exp(eta f(x)).
#
# g = 2 Cov(f,E_loc)
# c = 2 <f^2 E_loc> + 2 <f E_loc^(f)>
#     - 4 E <f^2> - 4 <f> g
#
# These are the same derivatives validated in tfim_eta_curvature_audit.jl, but
# here all expectations are evaluated over the exact Born distribution.
function exact_derivatives(H, states, A, f)
    d = size(states,1)
    shift = maximum(2.0 .* A)
    w = exp.(2.0 .* A .- shift)
    W = sum(w)

    sum_f = 0.0
    sum_f2 = 0.0
    sum_el = 0.0
    sum_fel = 0.0
    sum_f2el = 0.0
    sum_felf = 0.0

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
        sum_f += wj*fj
        sum_f2 += wj*fj*fj
        sum_el += wj*el
        sum_fel += wj*fj*el
        sum_f2el += wj*fj*fj*el
        sum_felf += wj*fj*elf
    end

    Ef = sum_f/W
    Ef2 = sum_f2/W
    E0 = sum_el/W
    Efel = sum_fel/W
    Ef2el = sum_f2el/W
    Qf = sum_felf/W

    g = 2.0*(Efel-Ef*E0)
    c = 2.0*Ef2el + 2.0*Qf - 4.0*E0*Ef2 - 4.0*Ef*g
    return E0,g,c
end

function profile_rows(H, model, tree, states, ratio, epoch)
    A,f = exact_path_data(model,tree,states)
    E0,g,c = exact_derivatives(H,states,A,f)
    etaN = isfinite(g) && isfinite(c) && c > CURVATURE_FLOOR ? -g/c : NaN

    # Evaluate the exact energy at the unconstrained Newton proposal when it is
    # a finite nonnegative step, even when it is outside ETA_GRID.
    EN = isfinite(etaN) && etaN >= 0.0 ? exact_path_energy(H,states,A,f,etaN) : NaN

    rows = NamedTuple[]
    for eta in ETA_GRID
        E = exact_path_energy(H,states,A,f,eta)
        Equad = E0 + g*eta + 0.5*c*eta^2
        push!(rows,(
            ratio=ratio, epoch=epoch, eta=eta,
            E0=E0, E_exact=E, deltaE_exact=E-E0,
            E_quad=Equad, deltaE_quad=Equad-E0,
            quad_error=E-Equad,
            g_exact=g, c_exact=c, eta_newton=etaN,
            E_at_newton=EN, deltaE_at_newton=EN-E0,
        ))
    end
    return rows
end

function train_and_profile(H, ratio, states)
    cfg = GBTQuantum.TrainingConfig(
        nsamples=NSAMPLES,epochs=EPOCHS,max_depth=MAX_DEPTH,eta=ETA_FIXED,
        burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=SEED,
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
    for epoch in 1:cfg.epochs
        batch = GBTQuantum.vmc_batch(H,model,samples)
        yA,_ = GBTQuantum.make_targets(batch)
        weights = batch.counts
        tree = GBTQuantum.grow_tree(batch.states,yA,weights;
            max_depth=cfg.max_depth,min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain)
        tree = center_tree(tree,batch.states,weights)

        if epoch in AUDIT_EPOCHS
            pr = profile_rows(H,model,tree,states,ratio,epoch)
            append!(rows,pr)
            firstrow = first(pr)
            grid_best = pr[argmin(getproperty.(pr,:E_exact))]
            @printf("J/h=%4.2f epoch=%3d  g=% .4e c=% .4e eta_N=% .5f  grid_best_eta=%.3f  dE_N=% .4e\n",
                ratio,epoch,firstrow.g_exact,firstrow.c_exact,firstrow.eta_newton,
                grid_best.eta,firstrow.deltaE_at_newton)
        end

        # Continue along the canonical fixed-eta trajectory; profiling does not
        # alter training or choose eta.
        push!(model.logamp.trees,scaled_tree(tree,cfg.eta))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:cfg.sweeps_per_epoch
            GBTQuantum.sweep!(rng,model,samples,logamps)
        end
    end
    return rows
end

function write_csv(path, rows)
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
    println("="^72)
    println("TFIM EXACT ETA ENERGY-PROFILE DIAGNOSTIC")
    println("N=$N ratios=$RATIOS epochs=$(sort!(collect(AUDIT_EPOCHS)))")
    println("eta grid=$ETA_GRID")
    println("Training remains canonical fixed eta=$ETA_FIXED")
    println("Derivative oracle: analytic full-Hilbert-space expectations")
    println("="^72)

    states = all_states(N)
    rows = NamedTuple[]
    for ratio in RATIOS
        H = GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true)
        append!(rows,train_and_profile(H,ratio,states))
    end

    outdir = joinpath(@__DIR__,"results")
    mkpath(outdir)
    path = joinpath(outdir,"tfim_eta_energy_profile.csv")
    write_csv(path,rows)
    println("\nResults written to $path")
end

end

if abspath(PROGRAM_FILE)==@__FILE__
    TFIMEtaEnergyProfile.main()
end
