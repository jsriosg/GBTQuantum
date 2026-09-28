module TFIMEtaCurvatureAudit

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Random
using Statistics
using Printf

# Diagnostic experiment only. It does not modify the production optimizer.
#
# For a fitted, gauge-centered tree f and the one-dimensional path
#
#     psi_eta(x) = psi(x) exp(eta f(x)),
#
# compare three curvature estimates at eta = 0:
#
#   c_old   : curvature of the old fixed-local-energy reweighting surrogate
#   c_vmc   : curvature of the true variational energy estimated on the VMC batch
#   c_exact : full-Hilbert-space finite-difference oracle (N=8 only here)
#
# The comparison separates estimator bias from Monte Carlo error.

const RATIOS = [0.05, 1.0, 2.0]
const N = 8
const NSAMPLES = 512
const EPOCHS = 150
const MAX_DEPTH = 4
const ETA_FIXED = 0.05
const SEED = 1234
const AUDIT_EPOCHS = Set([1, 10, 25, 50, 100, 150])
const FD_DELTA = 1e-3

function center_tree(tree, X, weights)
    preds = [GBTQuantum.predict(tree, @view X[j, :]) for j in axes(X, 1)]
    mu = GBTQuantum.weighted_mean(preds, weights)
    mu == 0.0 && return tree

    nodes = copy(tree.nodes)
    @inbounds for i in eachindex(nodes)
        n = nodes[i]
        if n.isleaf
            nodes[i] = GBTQuantum.Node(n.feature, n.value - mu, n.left, n.right, true)
        end
    end
    return GBTQuantum.RegressionTree(nodes)
end

function scaled_tree(tree, eta)
    return GBTQuantum.RegressionTree([
        n.isleaf ? GBTQuantum.Node(n.feature, eta * n.value, n.left, n.right, true) : n
        for n in tree.nodes
    ])
end

# -----------------------------------------------------------------------------
# Old V1 surrogate: reweight probabilities while holding E_loc fixed.
# -----------------------------------------------------------------------------
function old_reweighted_energy(batch, f, eta)
    w0 = Float64.(batch.counts)
    el = real.(batch.local_energy)
    z = 2.0 .* eta .* f
    zmax = maximum(z)
    rw = w0 .* exp.(z .- zmax)
    return sum(rw .* el) / sum(rw)
end

function old_curvature(batch, f; delta=FD_DELTA)
    Ep = old_reweighted_energy(batch, f, delta)
    E0 = old_reweighted_energy(batch, f, 0.0)
    Em = old_reweighted_energy(batch, f, -delta)
    return (Ep - 2.0 * E0 + Em) / delta^2
end

# -----------------------------------------------------------------------------
# True first derivative estimated from the empirical Born batch.
# -----------------------------------------------------------------------------
function batch_gradient(batch, f)
    w = Float64.(batch.counts)
    el = real.(batch.local_energy)
    W = sum(w)
    Ef = sum(w .* f) / W
    Ee = sum(w .* el) / W
    g = 2.0 * sum(w .* (f .- Ef) .* (el .- Ee)) / W
    return g, Ef, Ee
end

# -----------------------------------------------------------------------------
# For TFIM,
#
#   E_loc^(f)(x) = D(x) f(x)
#                  - h sum_i [psi(x^i)/psi(x)] f(x^i)
#
# where x^i is x with spin i flipped.
# -----------------------------------------------------------------------------
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

# True variational curvature estimated on the same empirical Born batch:
#
# c = 2 <f^2 E_loc> + 2 <f E_loc^(f)>
#     - 4 E <f^2> - 4 <f> g.
function vmc_curvature(H, model, batch, tree, f, g, Ef, Ee)
    w = Float64.(batch.counts)
    W = sum(w)
    el = real.(batch.local_energy)

    mean_f2 = sum(w .* (f .^ 2)) / W
    mean_f2_el = sum(w .* (f .^ 2) .* el) / W

    qsum = 0.0
    @inbounds for j in axes(batch.states, 1)
        x = @view batch.states[j, :]
        elf = f_local_energy(H, model, tree, x)
        qsum += w[j] * f[j] * elf
    end
    Qf = qsum / W

    c = 2.0 * mean_f2_el + 2.0 * Qf - 4.0 * Ee * mean_f2 - 4.0 * Ef * g
    return c, Qf, mean_f2, mean_f2_el
end

# -----------------------------------------------------------------------------
# Full Hilbert-space oracle.
# -----------------------------------------------------------------------------
function all_states(N)
    d = 1 << N
    X = Matrix{Int8}(undef, d, N)
    @inbounds for s in 0:(d - 1)
        for i in 1:N
            X[s + 1, i] = ((s >> (i - 1)) & 1) == 1 ? Int8(1) : Int8(-1)
        end
    end
    return X
end

# Exact energy of the temporary path A_eta = A + eta*f, evaluated without
# modifying the model. This independently includes both the Born reweighting
# and the eta-dependence of all off-diagonal wavefunction ratios.
function exact_path_energy(H, model, tree, states, eta)
    d = size(states, 1)
    A = Vector{Float64}(undef, d)
    f = Vector{Float64}(undef, d)

    @inbounds for j in 1:d
        x = @view states[j, :]
        A[j] = GBTQuantum.logamplitude(model, x)
        f[j] = GBTQuantum.predict(tree, x)
    end

    logw = 2.0 .* (A .+ eta .* f)
    shift = maximum(logw)
    w = exp.(logw .- shift)
    Z = sum(w)

    Esum = 0.0
    @inbounds for j in 1:d
        x = @view states[j, :]
        Aeta0 = A[j] + eta * f[j]
        el = GBTQuantum.diagonal(H, x)

        for i in 1:H.N
            x[i] = -x[i]
            Aetai = GBTQuantum.logamplitude(model, x) + eta * GBTQuantum.predict(tree, x)
            el -= H.h * exp(Aetai - Aeta0)
            x[i] = -x[i]
        end
        Esum += w[j] * el
    end

    return Esum / Z
end

function exact_fd_derivatives(H, model, tree, states; delta=FD_DELTA)
    Em = exact_path_energy(H, model, tree, states, -delta)
    E0 = exact_path_energy(H, model, tree, states, 0.0)
    Ep = exact_path_energy(H, model, tree, states, delta)
    g = (Ep - Em) / (2.0 * delta)
    c = (Ep - 2.0 * E0 + Em) / delta^2
    return g, c, E0
end

safe_newton(g, c) = isfinite(g) && isfinite(c) && c > 0.0 ? -g / c : NaN

function audit_row(H, model, batch, tree, epoch, ratio, states)
    f = [GBTQuantum.predict(tree, @view batch.states[j, :]) for j in axes(batch.states, 1)]

    g_mc, mean_f, E_mc = batch_gradient(batch, f)
    c_old = old_curvature(batch, f)
    c_vmc, Qf, mean_f2, mean_f2_el = vmc_curvature(H, model, batch, tree, f, g_mc, mean_f, E_mc)
    g_exact, c_exact, E_exact_model = exact_fd_derivatives(H, model, tree, states)

    return (
        ratio=ratio,
        epoch=epoch,
        E_mc=E_mc,
        E_exact_model=E_exact_model,
        mean_f=mean_f,
        g_mc=g_mc,
        g_exact=g_exact,
        g_error=g_mc-g_exact,
        c_old=c_old,
        c_vmc=c_vmc,
        c_exact=c_exact,
        old_error=c_old-c_exact,
        vmc_error=c_vmc-c_exact,
        Qf=Qf,
        mean_f2=mean_f2,
        mean_f2_el=mean_f2_el,
        eta_old=safe_newton(g_mc, c_old),
        eta_vmc=safe_newton(g_mc, c_vmc),
        eta_oracle=safe_newton(g_exact, c_exact),
    )
end

function train_and_audit(H, ratio, states)
    cfg = GBTQuantum.TrainingConfig(
        nsamples=NSAMPLES,
        epochs=EPOCHS,
        max_depth=MAX_DEPTH,
        eta=ETA_FIXED,
        burn_in_sweeps=50,
        sweeps_per_epoch=2,
        use_phase=false,
        seed=SEED,
        exact_diagnostics=false,
    )

    rng = MersenneTwister(cfg.seed)
    samples = Matrix{Int8}(undef, cfg.nsamples, H.N)
    @inbounds for i in eachindex(samples)
        samples[i] = rand(rng, Bool) ? Int8(1) : Int8(-1)
    end

    model = GBTQuantum.LogGBState(logamp_bias=0.0, phase_bias=0.0, use_phase=false)
    logamps = zeros(Float64, cfg.nsamples)

    for _ in 1:cfg.burn_in_sweeps
        GBTQuantum.sweep!(rng, model, samples, logamps)
    end

    rows = NamedTuple[]

    for epoch in 1:cfg.epochs
        batch = GBTQuantum.vmc_batch(H, model, samples)
        yA, _ = GBTQuantum.make_targets(batch)
        weights = batch.counts

        tree = GBTQuantum.grow_tree(
            batch.states, yA, weights;
            max_depth=cfg.max_depth,
            min_weight=cfg.min_leaf_weight,
            min_gain=cfg.min_gain,
        )
        tree = center_tree(tree, batch.states, weights)

        # Audit the local direction BEFORE applying the ordinary fixed-eta update.
        if epoch in AUDIT_EPOCHS
            row = audit_row(H, model, batch, tree, epoch, ratio, states)
            push!(rows, row)
            @printf(
                "  epoch=%3d  g_mc=% .4e g_exact=% .4e | c_old=% .4e c_vmc=% .4e c_exact=% .4e\n",
                epoch, row.g_mc, row.g_exact, row.c_old, row.c_vmc, row.c_exact,
            )
            @printf(
                "             eta_old=% .4e eta_vmc=% .4e eta_oracle=% .4e | dc_old=% .3e dc_vmc=% .3e\n",
                row.eta_old, row.eta_vmc, row.eta_oracle, row.old_error, row.vmc_error,
            )
        end

        # Preserve the canonical fixed-eta training trajectory.
        push!(model.logamp.trees, scaled_tree(tree, cfg.eta))
        GBTQuantum.refresh_logamps!(logamps, model, samples)
        for _ in 1:cfg.sweeps_per_epoch
            GBTQuantum.sweep!(rng, model, samples, logamps)
        end
    end

    return rows
end

function write_csv(path, rows)
    isempty(rows) && return
    names = propertynames(rows[1])
    open(path, "w") do io
        println(io, join(string.(names), ','))
        for row in rows
            println(io, join((getproperty(row, n) for n in names), ','))
        end
    end
end

function main()
    println("="^72)
    println("TFIM ETA CURVATURE AUDIT")
    println("N=$N ratios=$RATIOS nsamples=$NSAMPLES epochs=$EPOCHS fixed eta=$ETA_FIXED")
    println("audit epochs=$(sort!(collect(AUDIT_EPOCHS))) finite-difference delta=$FD_DELTA")
    println("Training trajectory remains the canonical fixed-eta trajectory.")
    println("="^72)

    states = all_states(N)
    rows = NamedTuple[]

    for ratio in RATIOS
        println("\nJ/h = $ratio")
        H = GBTQuantum.TFIMHamiltonian(N; J=ratio, h=1.0, periodic=true)
        append!(rows, train_and_audit(H, ratio, states))
    end

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    path = joinpath(outdir, "tfim_eta_curvature_audit.csv")
    write_csv(path, rows)

    println("\nResults written to $path")
    println("Interpretation:")
    println("  c_vmc - c_exact  -> primarily finite-sample / VMC error")
    println("  c_old - c_exact  -> finite-sample error + old-surrogate estimator bias")
    return rows
end

end # module TFIMEtaCurvatureAudit

if abspath(PROGRAM_FILE) == @__FILE__
    TFIMEtaCurvatureAudit.main()
end
