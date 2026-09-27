using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module OracleAcquisitionSamplingExperiment

using GBTQuantum
using Random
using Statistics
using Printf
include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

# Frozen-target oracle benchmark for the structural acquisition distribution
# derived from the split-margin influence function.
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
const diagnostic_sample_sizes = [256, 1024]
const nsampling_runs = 100
const base_seed = 1_420_000
const diagnostic_seed_base = 9_420_000

# Defensive oracle mixture.  The pure oracle distribution is also tested.
const defensive_delta_born = 0.05
const defensive_epsilon_uniform = 0.05
const support_floor = 1e-14

function exact_frozen_problem(H, model, X)
    p = exact_probabilities(model, X)
    eloc = ComplexF64[local_energy!(H, model, @view(X[i, :])) for i in axes(X, 1)]
    E = sum(p .* eloc)
    y = -real.(eloc .- E)
    yc = y .- sum(p .* y)
    sig = p .* yc.^2
    Z = sum(sig)
    qsig = Z > eps() ? sig ./ Z : copy(p)
    u = fill(1.0 / length(p), length(p))
    return (probabilities=p, target=y, energy=E,
            target_rms=sqrt(sum(p .* y.^2)), q_signal=qsig, q_uniform=u)
end

function gain_landscape(X, y, w, idx; min_weight=0.0)
    g = fill(-Inf, size(X, 2))
    W = sum(w[i] for i in idx)
    W > 0 || return g
    S = sum(w[i] * y[i] for i in idx)
    parent = S*S/W
    for f in axes(X, 2)
        WL = 0.0; SL = 0.0
        @inbounds for i in idx
            if X[i, f] < 0
                WL += w[i]
                SL += w[i] * y[i]
            end
        end
        WR = W - WL
        (WL < min_weight || WR < min_weight || WL <= 0 || WR <= 0) && continue
        SR = S - SL
        g[f] = SL*SL/WL + SR*SR/WR - parent
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
        L = Int[]; R = Int[]
        f = Int(n.feature)
        for i in idx
            X[i, f] < 0 ? push!(L, i) : push!(R, i)
        end
        walk(Int(n.left), d+1, L, path*"L")
        walk(Int(n.right), d+1, R, path*"R")
    end
    walk(1, 0, collect(axes(X, 1)), "")
    return out
end

function conditional_oracle_gains(X, y, p, idx)
    mass = sum(p[idx])
    mass > 0 || return fill(-Inf, size(X, 2)), mass
    pc = p ./ mass
    return gain_landscape(X, y, pc, idx; min_weight=exact_min_leaf_weight), mass
end

# Influence function of G_f = a^2/r + (m-a)^2/(1-r) - m^2,
# under the node-conditional distribution p(x|v).
function gain_influence(X, y, p, idx, f)
    mass = sum(p[idx])
    phi = zeros(Float64, size(X, 1))
    (mass > 0 && f != 0) || return phi

    r = 0.0; a = 0.0; m = 0.0
    @inbounds for i in idx
        pv = p[i] / mass
        Li = X[i, f] < 0 ? 1.0 : 0.0
        r += pv * Li
        a += pv * y[i] * Li
        m += pv * y[i]
    end
    (r > eps() && 1-r > eps()) || return phi

    gr = -a^2/r^2 + (m-a)^2/(1-r)^2
    ga = 2a/r - 2(m-a)/(1-r)
    gm = 2(m-a)/(1-r) - 2m

    @inbounds for i in idx
        Li = X[i, f] < 0 ? 1.0 : 0.0
        phi[i] = gr*(Li-r) + ga*(y[i]*Li-a) + gm*(y[i]-m)
    end
    return phi
end

function build_oracle(H, model, X)
    fr = exact_frozen_problem(H, model, X)
    p = fr.probabilities
    tree = GBTQuantum.grow_tree(X, fr.target, p;
        max_depth=oracle_depth, min_weight=exact_min_leaf_weight, min_gain=0.0)

    nodes = NamedTuple[]
    for n in oracle_regions(tree, X)
        og, mass = conditional_oracle_gains(X, fr.target, p, n.state_indices)
        f1, f2, G1, G2 = top_two(og)
        phi1 = gain_influence(X, fr.target, p, n.state_indices, f1)
        phi2 = gain_influence(X, fr.target, p, n.state_indices, f2)
        # Phi_v converts the node-conditional influence to the global p measure.
        Phi = (phi1 .- phi2) ./ mass
        importance = mass * max(G1, 0.0)
        push!(nodes, merge(n, (oracle_gains=og, probability_mass=mass,
            f1=f1, f2=f2, G1=G1, G2=G2, margin=G1-G2,
            importance=importance, Phi=Phi)))
    end
    return fr, tree, nodes
end

function oracle_acquisition_distribution(fr, nodes)
    p = fr.probabilities
    A2 = zeros(Float64, length(p))
    for n in nodes
        n.importance <= 0 && continue
        @. A2 += n.importance * n.Phi^2
    end
    A = sqrt.(A2)
    q = p .* A
    Z = sum(q)
    Z > eps() || return copy(p), A
    q ./= Z
    # Keep mathematically negligible but strictly positive support so that
    # all importance-weighted gain estimates remain defined.
    q .= max.(q, support_floor .* p)
    q ./= sum(q)
    return q, A
end

function proposal_set(fr, qoracle)
    p = fr.probabilities; s = fr.q_signal; u = fr.q_uniform
    d = defensive_delta_born; e = defensive_epsilon_uniform
    qdef = (1-d-e).*qoracle .+ d.*p .+ e.*u
    qdef ./= sum(qdef)
    return [
        (name="born", q=copy(p)),
        (name="signal50", q=0.50.*p .+ 0.50.*s),
        (name="signal25_uniform25", q=0.50.*p .+ 0.25.*s .+ 0.25.*u),
        (name="signal50_uniform25", q=0.25.*p .+ 0.50.*s .+ 0.25.*u),
        (name="uniform", q=copy(u)),
        (name="oracle_structural", q=copy(qoracle)),
        (name="oracle_defensive", q=qdef),
    ]
end

function sampled_gains(X, y, p, q, draws, idx)
    mask = falses(size(X, 1)); mask[idx] .= true
    localdraw = [i for i in draws if mask[i]]
    isempty(localdraw) && return fill(-Inf, size(X,2)), 0, 0, 0.0
    u, c = compress_indices(localdraw)
    wraw = c .* p[u] ./ q[u]
    Z = sum(wraw)
    Z > 0 || return fill(-Inf, size(X,2)), length(localdraw), length(u), 0.0
    w = wraw ./ Z
    g = gain_landscape(X[u, :], y[u], w, collect(eachindex(u));
                       min_weight=sampled_min_leaf_weight)
    iw = [p[i]/q[i] for i in localdraw]
    ess = sum(iw)^2 / sum(abs2, iw)
    return g, length(localdraw), length(u), ess
end

diagnostic_seed(run, epoch, M, sampler_index, repetition) =
    diagnostic_seed_base + 100_000*run + 1_000*epoch + 10_000*sampler_index + M + 10*repetition

function diagnose(H, model, X, run, epoch)
    fr, _, nodes = build_oracle(H, model, X)
    qoracle, A = oracle_acquisition_distribution(fr, nodes)
    proposals = proposal_set(fr, qoracle)

    @printf("\nEpoch %d: E=% .8f target RMS=%.4e nodes=%d  Amax=%.3e\n",
            epoch, real(fr.energy), fr.target_rms, length(nodes), maximum(A))

    raw = NamedTuple[]
    summary = NamedTuple[]

    for M in diagnostic_sample_sizes
        for (si, prop) in enumerate(proposals)
            q = prop.q ./ sum(prop.q)
            block = NamedTuple[]
            for srun in 1:nsampling_runs
                rng = MersenneTwister(diagnostic_seed(run, epoch, M, si, srun))
                draws = draw_categorical_indices(rng, q, M)
                for n in nodes
                    sg, nlocal, nuniq, ess = sampled_gains(X, fr.target, fr.probabilities,
                                                           q, draws, n.state_indices)
                    sf, _, _, _ = top_two(sg)
                    chosen = sf == 0 ? 0.0 :
                        (isfinite(n.oracle_gains[sf]) ? max(n.oracle_gains[sf], 0.0) : 0.0)
                    RG = n.G1 > eps() ? chosen/n.G1 : NaN
                    push!(block, (training_run=run, epoch=epoch, M=M,
                        sampler=prop.name, sampling_run=srun,
                        node_index=n.node_index, node_depth=n.node_depth,
                        node_path=n.node_path, oracle_best_feature=n.f1,
                        oracle_second_feature=n.f2, G1=n.G1, G2=n.G2,
                        oracle_margin=n.margin, node_importance=n.importance,
                        oracle_probability_mass=n.probability_mass,
                        sampled_feature=sf, exact_split_match=(sf != 0 && sf == n.f1),
                        relative_oracle_gain=RG, local_draws=nlocal,
                        local_unique_states=nuniq, local_ess=ess))
                end
            end
            append!(raw, block)

            match = mean(Float64(r.exact_split_match) for r in block)
            rg = finite_mean(r.relative_oracle_gain for r in block)
            # Importance-weighted structural recovery, averaged over repetitions.
            denom = sum(n.importance for n in nodes)
            impmatch = if denom > eps()
                mean(sum(n.importance * Float64(r.exact_split_match)
                         for n in nodes
                         for r in block
                         if r.sampling_run == srun && r.node_index == n.node_index) / denom
                     for srun in 1:nsampling_runs)
            else
                NaN
            end
            imprg = if denom > eps()
                mean(sum(n.importance * r.relative_oracle_gain
                         for n in nodes
                         for r in block
                         if r.sampling_run == srun && r.node_index == n.node_index && isfinite(r.relative_oracle_gain)) / denom
                     for srun in 1:nsampling_runs)
            else
                NaN
            end
            uniq = mean(r.local_unique_states for r in block)
            ess = mean(r.local_ess for r in block)
            push!(summary, (training_run=run, epoch=epoch, M=M, sampler=prop.name,
                split_match_rate=match, mean_relative_oracle_gain=rg,
                importance_weighted_match=impmatch,
                importance_weighted_relative_oracle_gain=imprg,
                mean_local_unique_states=uniq, mean_local_ess=ess))

            @printf("  M=%4d %-25s match=%.3f <RG>=%.3f Wmatch=%.3f WRG=%.3f uniq=%.1f ESS=%.1f\n",
                    M, prop.name, match, rg, impmatch, imprg, uniq, ess)
        end
    end
    return raw, summary
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

    raw = NamedTuple[]; summary = NamedTuple[]
    for epoch in 1:nepochs
        batch = vmc_batch(H, model, samples)
        yA, _ = make_targets(batch)
        w = batch.counts
        if epoch in checkpoint_epochs
            r, s = diagnose(H, model, X, run, epoch)
            append!(raw, r); append!(summary, s)
        end
        tree = GBTQuantum.grow_tree(batch.states, yA, w;
            max_depth=optimizer_max_depth,
            min_weight=optimizer_min_leaf_weight,
            min_gain=optimizer_min_gain)
        pred = predict_all(tree, batch.states)
        mu = weighted_mean(pred, w)
        isfinite(mu) && mu != 0 && (tree = shift_tree_leaves(tree, mu))
        push!(model.logamp.trees, scale_tree(tree, eta))
        GBTQuantum.refresh_logamps!(logamps, model, samples)
        for _ in 1:sweeps_per_epoch
            GBTQuantum.sweep!(rng, model, samples, logamps)
        end
    end
    return raw, summary
end

function main()
    println("\n============================================================")
    println("ORACLE STRUCTURAL ACQUISITION SAMPLING BENCHMARK")
    println("N=$N J/h=$(J/h) runs=$ntraining_runs checkpoints=$(sort(collect(checkpoint_epochs)))")
    println("M=$diagnostic_sample_sizes repetitions=$nsampling_runs")
    @printf("defensive oracle: %.2f structural + %.2f Born + %.2f uniform\n",
            1-defensive_delta_born-defensive_epsilon_uniform,
            defensive_delta_born, defensive_epsilon_uniform)
    println("============================================================")

    X = enumerate_states(N)
    raw = NamedTuple[]; summary = NamedTuple[]
    for run in 1:ntraining_runs
        @printf("\nTraining trajectory %d/%d\n", run, ntraining_runs)
        r, s = run_training(run, X)
        append!(raw, r); append!(summary, s)
    end

    out = joinpath(@__DIR__, "results"); mkpath(out)
    rawpath = joinpath(out, "oracle_acquisition_sampling_raw.csv")
    summarypath = joinpath(out, "oracle_acquisition_sampling_summary.csv")
    write_namedtuple_csv(rawpath, raw)
    write_namedtuple_csv(summarypath, summary)
    println("\nRESULTS WRITTEN TO\n", rawpath, "\n", summarypath)
    return (raw=raw, summary=summary)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    OracleAcquisitionSamplingExperiment.main()
end
