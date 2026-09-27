using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module InfluenceAsymptoticScalingExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

# Diagnose whether the split-margin influence-function variance formula is
# asymptotically correct for the actual self-normalized importance estimator.
#
# We freeze representative oracle nodes at training checkpoints and sweep M.
# For each node/sampler we compare
#     M * Var(Dhat)
# with
#     V(q) = sum_x p(x)^2 psi(x)^2 / q(x),
# while tracking bias and the standardized variable
#     Z = sqrt(M) * (Dhat - D) / sqrt(V(q)).

const N = 12
const h = 1.0
const J = 2.0
const training_nsamples = 256
const nepochs = 64
const checkpoint_epochs = Set([8, 32, 64])
const ntraining_runs = 3
const optimizer_max_depth = 4
const eta = 0.05
const burn_in_sweeps = 100
const sweeps_per_epoch = 1
const optimizer_min_leaf_weight = 1.0
const optimizer_min_gain = 0.0
const oracle_depth = 4
const exact_min_leaf_weight = 1e-14
const sampled_min_leaf_weight = 1e-14
const sample_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]
const nsampling_runs = 400
const if_born_epsilon = 0.05
const support_floor = 1e-14
const base_seed = 1_720_000
const diagnostic_seed_base = 41_720_000

function csv_escape(x)
    s = string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s) || occursin('\r', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_namedtuple_csv(path, rows)
    open(path, "w") do io
        isempty(rows) && return
        cols = propertynames(first(rows))
        println(io, join(string.(cols), ","))
        for r in rows
            println(io, join((csv_escape(getproperty(r, c)) for c in cols), ","))
        end
    end
end

function exact_frozen_problem(H, model, X)
    p = exact_probabilities(model, X)
    eloc = ComplexF64[local_energy!(H, model, @view(X[i, :])) for i in axes(X, 1)]
    E = sum(p .* eloc)
    y = -real.(eloc .- E)
    return (probabilities=p, target=y, energy=E,
            target_rms=sqrt(sum(p .* y.^2)))
end

function gain_landscape(X, y, w, idx; min_weight=0.0)
    g = fill(-Inf, size(X, 2))
    W = sum(w[i] for i in idx)
    W > 0 || return g
    S = sum(w[i] * y[i] for i in idx)
    parent = S * S / W
    for f in axes(X, 2)
        WL = 0.0
        SL = 0.0
        @inbounds for i in idx
            if X[i, f] < 0
                WL += w[i]
                SL += w[i] * y[i]
            end
        end
        WR = W - WL
        if WL < min_weight || WR < min_weight || WL <= 0 || WR <= 0
            continue
        end
        SR = S - SL
        g[f] = SL * SL / WL + SR * SR / WR - parent
    end
    return g
end

function top_two(g)
    v = [i for i in eachindex(g) if isfinite(g[i]) && g[i] >= 0]
    isempty(v) && return (0, 0, 0.0, 0.0)
    sort!(v, by=i -> g[i], rev=true)
    f1 = v[1]
    f2 = length(v) >= 2 ? v[2] : 0
    return (f1, f2, g[f1], f2 == 0 ? 0.0 : g[f2])
end

function oracle_regions(tree, X)
    out = NamedTuple[]
    function walk(k, d, idx, path)
        n = tree.nodes[k]
        n.isleaf && return
        push!(out, (node_index=k, node_depth=d, node_path=path,
                    state_indices=copy(idx)))
        L = Int[]
        R = Int[]
        f = Int(n.feature)
        for i in idx
            X[i, f] < 0 ? push!(L, i) : push!(R, i)
        end
        walk(Int(n.left), d + 1, L, path * "L")
        walk(Int(n.right), d + 1, R, path * "R")
    end
    walk(1, 0, collect(axes(X, 1)), "")
    return out
end

function gain_and_if(X, y, p, idx, f)
    P = sum(p[idx])
    (P > 0 && f != 0) || return (gain=-Inf, psi=zeros(length(p)))
    r = 0.0
    a = 0.0
    m = 0.0
    @inbounds for i in idx
        pi = p[i] / P
        L = X[i, f] < 0 ? 1.0 : 0.0
        r += pi * L
        a += pi * y[i] * L
        m += pi * y[i]
    end
    (r > eps() && 1-r > eps()) || return (gain=-Inf, psi=zeros(length(p)))
    gain = a*a/r + (m-a)^2/(1-r) - m*m
    cr = -a*a/(r*r) + (m-a)^2/((1-r)^2)
    ca = 2a/r - 2(m-a)/(1-r)
    cm = 2(m-a)/(1-r) - 2m
    psi = zeros(Float64, length(p))
    @inbounds for i in idx
        L = X[i, f] < 0 ? 1.0 : 0.0
        psi[i] = cr*(L-r) + ca*(y[i]*L-a) + cm*(y[i]-m)
    end
    return (gain=gain, psi=psi)
end

function build_oracle(H, model, X)
    fr = exact_frozen_problem(H, model, X)
    p = fr.probabilities
    tree = GBTQuantum.grow_tree(X, fr.target, p;
        max_depth=oracle_depth,
        min_weight=exact_min_leaf_weight,
        min_gain=0.0)
    nodes = NamedTuple[]
    for n in oracle_regions(tree, X)
        P = sum(p[n.state_indices])
        pc = p ./ P
        g = gain_landscape(X, fr.target, pc, n.state_indices;
                           min_weight=exact_min_leaf_weight)
        f1, f2, G1, G2 = top_two(g)
        f2 == 0 && continue
        a = gain_and_if(X, fr.target, p, n.state_indices, f1)
        b = gain_and_if(X, fr.target, p, n.state_indices, f2)
        push!(nodes, merge(n, (
            probability_mass=P,
            f1=f1,
            f2=f2,
            G1=G1,
            G2=G2,
            margin=G1-G2,
            psi=a.psi .- b.psi
        )))
    end
    return fr, nodes
end

function normalized_q(raw, p)
    q = max.(Float64.(raw), 0.0)
    Z = sum(q)
    if !(Z > eps()) || !isfinite(Z)
        return copy(p)
    end
    q ./= Z
    q = max.(q, support_floor .* p)
    q ./= sum(q)
    return q
end

q_born(fr, node) = copy(fr.probabilities)
q_if_pure(fr, node) = normalized_q(fr.probabilities .* abs.(node.psi), fr.probabilities)
function q_if_mix(fr, node)
    q0 = q_if_pure(fr, node)
    return (1-if_born_epsilon) .* q0 .+ if_born_epsilon .* fr.probabilities
end

function theoretical_V(p, psi, q)
    s = 0.0
    @inbounds for i in eachindex(p)
        if p[i] > 0 && psi[i] != 0
            q[i] > 0 || return Inf
            s += p[i]^2 * psi[i]^2 / q[i]
        end
    end
    return s
end

function sampled_margin(X, y, p, q, draws, node)
    in_node = falses(length(p))
    in_node[node.state_indices] .= true
    localdraw = [i for i in draws if in_node[i]]
    isempty(localdraw) && return NaN
    u, c = compress_indices(localdraw)
    wraw = c .* p[u] ./ q[u]
    Z = sum(wraw)
    Z > 0 || return NaN
    w = wraw ./ Z
    g = gain_landscape(X[u, :], y[u], w, collect(eachindex(u));
                       min_weight=sampled_min_leaf_weight)
    ga = node.f1 <= length(g) ? g[node.f1] : -Inf
    gb = node.f2 <= length(g) ? g[node.f2] : -Inf
    return (isfinite(ga) && isfinite(gb)) ? ga - gb : NaN
end

# Representative nodes: root, a high-mass non-root node, and a low-mass
# non-root node. This intentionally spans the regimes in which the earlier
# finite-M discrepancy was most likely to differ.
function representative_nodes(nodes)
    isempty(nodes) && return NamedTuple[]
    selected = NamedTuple[]
    roots = [n for n in nodes if n.node_depth == 0]
    !isempty(roots) && push!(selected, first(roots))
    nonroot = [n for n in nodes if n.node_depth > 0]
    if !isempty(nonroot)
        hi = nonroot[argmax([n.probability_mass for n in nonroot])]
        lo = nonroot[argmin([n.probability_mass for n in nonroot])]
        if all(n.node_index != hi.node_index for n in selected)
            push!(selected, hi)
        end
        if all(n.node_index != lo.node_index for n in selected)
            push!(selected, lo)
        end
    end
    return selected
end

diagnostic_seed(run, epoch, M, nodej, samplerj, rep) =
    diagnostic_seed_base + 10_000_000*run + 100_000*epoch +
    10_000*nodej + 1_000*samplerj + M + rep

function summarize_samples(Dh, D, M, V)
    n = length(Dh)
    if n == 0
        return (mean_hat=NaN, bias=NaN, variance=NaN, mvar=NaN,
                ratio=NaN, zmean=NaN, zvar=NaN, zrmse=NaN,
                q025=NaN, q50=NaN, q975=NaN)
    end
    mean_hat = mean(Dh)
    bias = mean_hat - D
    variance = n > 1 ? var(Dh; corrected=true) : NaN
    mvar = isfinite(variance) ? M * variance : NaN
    ratio = (isfinite(mvar) && V > 0) ? mvar / V : NaN
    if V > 0 && isfinite(V)
        z = sqrt(M) .* (Dh .- D) ./ sqrt(V)
        zmean = mean(z)
        zvar = n > 1 ? var(z; corrected=true) : NaN
        zrmse = sqrt(mean(z.^2))
        q025 = quantile(z, 0.025)
        q50 = quantile(z, 0.5)
        q975 = quantile(z, 0.975)
    else
        zmean=zvar=zrmse=q025=q50=q975=NaN
    end
    return (mean_hat=mean_hat, bias=bias, variance=variance, mvar=mvar,
            ratio=ratio, zmean=zmean, zvar=zvar, zrmse=zrmse,
            q025=q025, q50=q50, q975=q975)
end

function diagnose(H, model, X, run, epoch)
    fr, allnodes = build_oracle(H, model, X)
    nodes = representative_nodes(allnodes)
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e oracle nodes=%d selected=%d\n",
            epoch, real(fr.energy), fr.target_rms, length(allnodes), length(nodes))
    rows = NamedTuple[]
    for (nj, n) in enumerate(nodes)
        @printf("  node=%d depth=%d path=%s P=%.3e D=%.3e f=(%d,%d)\n",
                n.node_index, n.node_depth, isempty(n.node_path) ? "root" : n.node_path,
                n.probability_mass, n.margin, n.f1, n.f2)
        samplers = [("born", q_born(fr,n)),
                    ("if_pure", q_if_pure(fr,n)),
                    ("if_mix_005", q_if_mix(fr,n))]
        for (sj, (sname, q)) in enumerate(samplers)
            V = theoretical_V(fr.probabilities, n.psi, q)
            @printf("    %-10s V=%.4e\n", sname, V)
            for M in sample_sizes
                Dh = Float64[]
                for rep in 1:nsampling_runs
                    rng = MersenneTwister(diagnostic_seed(run, epoch, M, nj, sj, rep))
                    draws = draw_categorical_indices(rng, q, M)
                    Dhat = sampled_margin(X, fr.target, fr.probabilities, q, draws, n)
                    isfinite(Dhat) && push!(Dh, Dhat)
                end
                s = summarize_samples(Dh, n.margin, M, V)
                @printf("      M=%5d valid=%3d bias=% .3e MVar/V=%8.3f Zmean=% .3f Zvar=%8.3f\n",
                        M, length(Dh), s.bias, s.ratio, s.zmean, s.zvar)
                push!(rows, (
                    training_run=run,
                    epoch=epoch,
                    node_index=n.node_index,
                    node_depth=n.node_depth,
                    node_path=n.node_path,
                    probability_mass=n.probability_mass,
                    f1=n.f1,
                    f2=n.f2,
                    oracle_G1=n.G1,
                    oracle_G2=n.G2,
                    oracle_margin=n.margin,
                    sampler=sname,
                    M=M,
                    repetitions=nsampling_runs,
                    valid_repetitions=length(Dh),
                    theoretical_V=V,
                    mean_margin_hat=s.mean_hat,
                    margin_bias=s.bias,
                    margin_variance=s.variance,
                    empirical_M_variance=s.mvar,
                    variance_ratio_empirical_theory=s.ratio,
                    standardized_mean=s.zmean,
                    standardized_variance=s.zvar,
                    standardized_rmse=s.zrmse,
                    standardized_q025=s.q025,
                    standardized_median=s.q50,
                    standardized_q975=s.q975
                ))
            end
        end
    end
    return rows
end

function run_training(run, X)
    H = TFIMHamiltonian(N; J=J, h=h, periodic=true)
    rng = MersenneTwister(base_seed + 10_000*run)
    samples = Matrix{Int8}(undef, training_nsamples, N)
    @inbounds for i in eachindex(samples)
        samples[i] = rand(rng, Bool) ? Int8(1) : Int8(-1)
    end
    model = LogGBState(logamp_bias=0.0, phase_bias=0.0, use_phase=false)
    logamps = zeros(training_nsamples)
    for _ in 1:burn_in_sweeps
        GBTQuantum.sweep!(rng, model, samples, logamps)
    end
    rows = NamedTuple[]
    for epoch in 1:nepochs
        batch = vmc_batch(H, model, samples)
        yA, _ = make_targets(batch)
        w = batch.counts
        if epoch in checkpoint_epochs
            append!(rows, diagnose(H, model, X, run, epoch))
        end
        tree = GBTQuantum.grow_tree(batch.states, yA, w;
            max_depth=optimizer_max_depth,
            min_weight=optimizer_min_leaf_weight,
            min_gain=optimizer_min_gain)
        pred = predict_all(tree, batch.states)
        mu = weighted_mean(pred, w)
        if isfinite(mu) && mu != 0
            tree = shift_tree_leaves(tree, mu)
        end
        push!(model.logamp.trees, scale_tree(tree, eta))
        GBTQuantum.refresh_logamps!(logamps, model, samples)
        for _ in 1:sweeps_per_epoch
            GBTQuantum.sweep!(rng, model, samples, logamps)
        end
    end
    return rows
end

function main()
    println("\n============================================================")
    println("INFLUENCE-FUNCTION ASYMPTOTIC SCALING")
    println("N=$N J/h=$(J/h) runs=$ntraining_runs checkpoints=$(sort(collect(checkpoint_epochs)))")
    println("M=$sample_sizes repetitions=$nsampling_runs")
    println("representative nodes: root + high-mass non-root + low-mass non-root")
    println("samplers: Born, IF, IF+$(100*if_born_epsilon)% Born")
    println("============================================================")
    X = enumerate_states(N)
    rows = NamedTuple[]
    for run in 1:ntraining_runs
        @printf("\nTraining trajectory %d/%d\n", run, ntraining_runs)
        append!(rows, run_training(run, X))
    end
    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    path = joinpath(outdir, "influence_asymptotic_scaling.csv")
    write_namedtuple_csv(path, rows)
    println("\nResults written to experiments/results/influence_asymptotic_scaling.csv")
end

export main, run_training, diagnose

end

if abspath(PROGRAM_FILE) == @__FILE__
    InfluenceAsymptoticScalingExperiment.main()
end
