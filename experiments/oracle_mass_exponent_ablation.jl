using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module OracleMassExponentAblationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

# Ablate the node-mass exponent alpha in
#   A_alpha(x)^2 = sum_v P(v)^alpha G1_v Phi_v(x)^2
#                = sum_v P(v)^(alpha-2) G1_v phi_D,v(x)^2.
# alpha=1 reproduces the original oracle structural acquisition;
# alpha=2 cancels the explicit inverse-node-mass factor.

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
const mass_exponents = [1.0, 1.5, 2.0, 2.5, 3.0]
const base_seed = 1_420_000
const diagnostic_seed_base = 11_420_000
const support_floor = 1e-14

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
        for row in rows
            println(io, join((csv_escape(getproperty(row, c)) for c in cols), ","))
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
        phiD = phi1 .- phi2
        Phi = phiD ./ mass
        push!(nodes, merge(n, (oracle_gains=og, probability_mass=mass,
            f1=f1, f2=f2, G1=G1, G2=G2, margin=G1-G2,
            phiD=phiD, Phi=Phi)))
    end
    return fr, nodes
end

function acquisition_for_alpha(fr, nodes, alpha)
    p = fr.probabilities
    A2 = zeros(Float64, length(p))
    node_contributions = Vector{Vector{Float64}}(undef, length(nodes))
    for (j,n) in enumerate(nodes)
        importance = n.probability_mass^alpha * max(n.G1, 0.0)
        c = importance .* n.Phi.^2
        node_contributions[j] = c
        @. A2 += c
    end
    A = sqrt.(A2)
    q = p .* A
    Z = sum(q)
    if Z > eps()
        q ./= Z
        q .= max.(q, support_floor .* p)
        q ./= sum(q)
    else
        q = copy(p)
    end
    return q, A, A2, node_contributions
end

function concentration_metrics(q, A, A2, nodes, contributions)
    qmax = maximum(q)
    qpr = 1.0 / sum(abs2, q)
    Hq = -sum(qi > 0 ? qi*log(qi) : 0.0 for qi in q)
    ieff = exp(Hq)
    imax = argmax(A)
    c_at_max = [c[imax] for c in contributions]
    jdom = isempty(c_at_max) ? 0 : argmax(c_at_max)
    domfrac = (jdom == 0 || A2[imax] <= 0) ? 0.0 : c_at_max[jdom]/A2[imax]
    domnode = jdom == 0 ? 0 : nodes[jdom].node_index
    domdepth = jdom == 0 ? -1 : nodes[jdom].node_depth
    dompath = jdom == 0 ? "" : nodes[jdom].node_path
    dommass = jdom == 0 ? NaN : nodes[jdom].probability_mass

    # Acquisition-weighted mean of r_max(x), where r_max is the largest
    # fractional node contribution to A(x)^2 at state x.
    rmax = zeros(Float64, length(q))
    for i in eachindex(q)
        if A2[i] > 0
            rmax[i] = maximum(c[i] for c in contributions) / A2[i]
        end
    end
    mean_rmax_q = sum(q .* rmax)

    # Fraction of total unnormalised A^2 attributable to each depth.
    depth_totals = Dict{Int,Float64}()
    totalc = sum(A2)
    for (j,n) in enumerate(nodes)
        depth_totals[n.node_depth] = get(depth_totals, n.node_depth, 0.0) + sum(contributions[j])
    end
    d0 = totalc > 0 ? get(depth_totals,0,0.0)/totalc : 0.0
    d1 = totalc > 0 ? get(depth_totals,1,0.0)/totalc : 0.0
    d2 = totalc > 0 ? get(depth_totals,2,0.0)/totalc : 0.0
    d3 = totalc > 0 ? get(depth_totals,3,0.0)/totalc : 0.0

    return (Amax=maximum(A), Amax_state=imax, qmax=qmax,
        q_participation_ratio=qpr, q_entropy_effective_states=ieff,
        dominant_node_index=domnode, dominant_node_depth=domdepth,
        dominant_node_path=dompath, dominant_node_mass=dommass,
        dominant_fraction_at_Amax=domfrac, q_weighted_mean_rmax=mean_rmax_q,
        depth0_A2_fraction=d0, depth1_A2_fraction=d1,
        depth2_A2_fraction=d2, depth3_A2_fraction=d3)
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
    g = gain_landscape(X[u,:], y[u], w, collect(eachindex(u));
                       min_weight=sampled_min_leaf_weight)
    iw = [p[i]/q[i] for i in localdraw]
    ess = sum(iw)^2 / sum(abs2, iw)
    return g, length(localdraw), length(u), ess
end

diagnostic_seed(run, epoch, M, ai, repetition) =
    diagnostic_seed_base + 1_000_000*run + 10_000*epoch + 100_000*ai + M + repetition

function diagnose(H, model, X, run, epoch)
    fr, nodes = build_oracle(H, model, X)
    raw = NamedTuple[]
    summary = NamedTuple[]
    concentration = NamedTuple[]
    node_rows = NamedTuple[]

    @printf("\nEpoch %d: E=% .8f target RMS=%.4e nodes=%d\n",
            epoch, real(fr.energy), fr.target_rms, length(nodes))

    for (ai,alpha) in enumerate(mass_exponents)
        q, A, A2, contributions = acquisition_for_alpha(fr, nodes, alpha)
        cm = concentration_metrics(q, A, A2, nodes, contributions)
        push!(concentration, merge((training_run=run, epoch=epoch, alpha=alpha,
            energy=real(fr.energy), target_rms=fr.target_rms), cm))

        @printf("  alpha=%.1f Amax=%.3e qmax=%.3e PR=%.1f expH=%.1f domP=%.3e domfrac=%.3f <rmax>q=%.3f depth=[%.3f %.3f %.3f %.3f]\n",
            alpha, cm.Amax, cm.qmax, cm.q_participation_ratio,
            cm.q_entropy_effective_states, cm.dominant_node_mass,
            cm.dominant_fraction_at_Amax, cm.q_weighted_mean_rmax,
            cm.depth0_A2_fraction, cm.depth1_A2_fraction,
            cm.depth2_A2_fraction, cm.depth3_A2_fraction)

        for (j,n) in enumerate(nodes)
            c = contributions[j]
            importance = n.probability_mass^alpha * max(n.G1,0.0)
            maxc = isempty(n.state_indices) ? 0.0 : maximum(c[n.state_indices])
            totalc = sum(c)
            push!(node_rows, (training_run=run, epoch=epoch, alpha=alpha,
                node_index=n.node_index, node_depth=n.node_depth,
                node_path=n.node_path, probability_mass=n.probability_mass,
                G1=n.G1, G2=n.G2, margin=n.margin,
                alpha_importance=importance,
                mass_power_factor=n.probability_mass^(alpha-2),
                max_abs_phiD=maximum(abs.(n.phiD[n.state_indices])),
                max_abs_Phi=maximum(abs.(n.Phi[n.state_indices])),
                max_A2_contribution=maxc, total_A2_contribution=totalc))
        end

        for M in diagnostic_sample_sizes
            block = NamedTuple[]
            for srun in 1:nsampling_runs
                rng = MersenneTwister(diagnostic_seed(run, epoch, M, ai, srun))
                draws = draw_categorical_indices(rng, q, M)
                for n in nodes
                    sg, nlocal, nuniq, ess = sampled_gains(X, fr.target,
                        fr.probabilities, q, draws, n.state_indices)
                    sf, _, _, _ = top_two(sg)
                    chosen = sf == 0 ? 0.0 :
                        (isfinite(n.oracle_gains[sf]) ? max(n.oracle_gains[sf],0.0) : 0.0)
                    RG = n.G1 > eps() ? chosen/n.G1 : NaN
                    # Keep the original alpha=1 importance as a common evaluation
                    # metric across all alpha values, so changing alpha does not
                    # change the scoring rule itself.
                    eval_importance = n.probability_mass * max(n.G1,0.0)
                    push!(block, (training_run=run, epoch=epoch, alpha=alpha,
                        M=M, sampling_run=srun, node_index=n.node_index,
                        node_depth=n.node_depth, node_path=n.node_path,
                        oracle_best_feature=n.f1, oracle_second_feature=n.f2,
                        G1=n.G1, G2=n.G2, oracle_margin=n.margin,
                        evaluation_importance=eval_importance,
                        oracle_probability_mass=n.probability_mass,
                        sampled_feature=sf,
                        exact_split_match=(sf != 0 && sf == n.f1),
                        relative_oracle_gain=RG, local_draws=nlocal,
                        local_unique_states=nuniq, local_ess=ess))
                end
            end
            append!(raw, block)

            match = mean(Float64(r.exact_split_match) for r in block)
            rgvals = [r.relative_oracle_gain for r in block if isfinite(r.relative_oracle_gain)]
            rg = isempty(rgvals) ? NaN : mean(rgvals)
            denom = sum(n.probability_mass * max(n.G1,0.0) for n in nodes)
            wmatch = denom > eps() ? mean(
                sum((n.probability_mass*max(n.G1,0.0))*Float64(r.exact_split_match)
                    for n in nodes for r in block
                    if r.sampling_run==srun && r.node_index==n.node_index)/denom
                for srun in 1:nsampling_runs) : NaN
            wrg = denom > eps() ? mean(
                sum((n.probability_mass*max(n.G1,0.0))*r.relative_oracle_gain
                    for n in nodes for r in block
                    if r.sampling_run==srun && r.node_index==n.node_index && isfinite(r.relative_oracle_gain))/denom
                for srun in 1:nsampling_runs) : NaN
            uniq = mean(r.local_unique_states for r in block)
            ess = mean(r.local_ess for r in block)
            push!(summary, (training_run=run, epoch=epoch, alpha=alpha, M=M,
                split_match_rate=match, mean_relative_oracle_gain=rg,
                importance_weighted_match=wmatch,
                importance_weighted_relative_oracle_gain=wrg,
                mean_local_unique_states=uniq, mean_local_ess=ess,
                qmax=cm.qmax, q_participation_ratio=cm.q_participation_ratio,
                q_entropy_effective_states=cm.q_entropy_effective_states,
                dominant_fraction_at_Amax=cm.dominant_fraction_at_Amax,
                q_weighted_mean_rmax=cm.q_weighted_mean_rmax))
            @printf("      M=%4d match=%.3f RG=%.3f Wmatch=%.3f WRG=%.3f uniq=%.1f ESS=%.1f\n",
                    M, match, rg, wmatch, wrg, uniq, ess)
        end
    end
    return raw, summary, concentration, node_rows
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

    raw=NamedTuple[]; summary=NamedTuple[]; concentration=NamedTuple[]; node_rows=NamedTuple[]
    for epoch in 1:nepochs
        batch = vmc_batch(H, model, samples)
        yA, _ = make_targets(batch)
        w = batch.counts
        if epoch in checkpoint_epochs
            r,s,c,n = diagnose(H, model, X, run, epoch)
            append!(raw,r); append!(summary,s); append!(concentration,c); append!(node_rows,n)
        end
        tree = GBTQuantum.grow_tree(batch.states, yA, w;
            max_depth=optimizer_max_depth, min_weight=optimizer_min_leaf_weight,
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
    return raw, summary, concentration, node_rows
end

function main()
    println("\n============================================================")
    println("ORACLE NODE-MASS EXPONENT ABLATION")
    println("N=$N J/h=$(J/h) runs=$ntraining_runs checkpoints=$(sort(collect(checkpoint_epochs)))")
    println("alpha=$mass_exponents M=$diagnostic_sample_sizes repetitions=$nsampling_runs")
    println("A_alpha^2 = sum_v P(v)^alpha G1_v Phi_v^2")
    println("============================================================")

    X = enumerate_states(N)
    raw=NamedTuple[]; summary=NamedTuple[]; concentration=NamedTuple[]; nodes=NamedTuple[]
    for run in 1:ntraining_runs
        @printf("\nTraining trajectory %d/%d\n", run, ntraining_runs)
        r,s,c,n = run_training(run, X)
        append!(raw,r); append!(summary,s); append!(concentration,c); append!(nodes,n)
    end

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    write_namedtuple_csv(joinpath(outdir,"oracle_mass_exponent_ablation_raw.csv"), raw)
    write_namedtuple_csv(joinpath(outdir,"oracle_mass_exponent_ablation_summary.csv"), summary)
    write_namedtuple_csv(joinpath(outdir,"oracle_mass_exponent_ablation_concentration.csv"), concentration)
    write_namedtuple_csv(joinpath(outdir,"oracle_mass_exponent_ablation_nodes.csv"), nodes)
    println("\nResults written under experiments/results/oracle_mass_exponent_ablation_*.csv")
end

export main, run_training, diagnose

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    OracleMassExponentAblationExperiment.main()
end
