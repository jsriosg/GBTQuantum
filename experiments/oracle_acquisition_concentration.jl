using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module OracleAcquisitionConcentrationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

# Diagnose WHY the oracle structural acquisition can become extremely
# concentrated late in training. In particular, decompose
#
#   A(x)^2 = sum_v P(v) G1_v Phi_v(x)^2
#          = sum_v [G1_v/P(v)] I_v(x) phi_D,v(x)^2
#
# node by node and identify the nodes/states responsible for Amax.

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
const base_seed = 1_420_000
const top_states_to_print = 12
const top_nodes_to_print = 15

# Lightweight CSV output so experiments do not require CSV.jl/DataFrames.jl.
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
        importance = mass * max(G1, 0.0)
        contribution = importance .* Phi.^2
        push!(nodes, merge(n, (oracle_gains=og, probability_mass=mass,
            f1=f1, f2=f2, G1=G1, G2=G2, margin=G1-G2,
            importance=importance, phiD=phiD, Phi=Phi,
            contribution=contribution)))
    end
    return fr, tree, nodes
end

function state_string(x)
    join((s > 0 ? "+" : "-") for s in x)
end

function diagnose_concentration(H, model, X, run, epoch)
    fr, _, nodes = build_oracle(H, model, X)
    p = fr.probabilities
    A2 = zeros(Float64, length(p))
    for n in nodes
        @. A2 += n.contribution
    end
    A = sqrt.(A2)
    qraw = p .* A
    Z = sum(qraw)
    q = Z > eps() ? qraw ./ Z : copy(p)

    imax = argmax(A)
    qmax = maximum(q)
    q_pr = 1.0 / sum(abs2, q)
    q_entropy = -sum(qi > 0 ? qi*log(qi) : 0.0 for qi in q)
    q_entropy_eff = exp(q_entropy)

    @printf("\nRun %d epoch %d: E=% .8f targetRMS=%.4e Amax=%.3e at state %d (%s) p=%.3e q=%.3e\n",
            run, epoch, real(fr.energy), fr.target_rms, A[imax], imax,
            state_string(@view X[imax,:]), p[imax], q[imax])
    @printf("  acquisition concentration: qmax=%.3e PR(q)=%.2f exp(Hq)=%.2f\n",
            qmax, q_pr, q_entropy_eff)

    node_rows = NamedTuple[]
    for n in nodes
        idx = n.state_indices
        maxphi_local = isempty(idx) ? 0.0 : maximum(abs.(n.phiD[idx]))
        maxPhi_local = isempty(idx) ? 0.0 : maximum(abs.(n.Phi[idx]))
        maxcontrib = isempty(idx) ? 0.0 : maximum(n.contribution[idx])
        icontrib = isempty(idx) ? 0 : idx[argmax(n.contribution[idx])]
        totalcontrib = sum(n.contribution)
        invmass_factor = n.probability_mass > 0 ? n.G1/n.probability_mass : Inf
        push!(node_rows, (training_run=run, epoch=epoch,
            node_index=n.node_index, node_depth=n.node_depth, node_path=n.node_path,
            probability_mass=n.probability_mass, G1=n.G1, G2=n.G2,
            margin=n.margin, importance=n.importance,
            G1_over_probability_mass=invmass_factor,
            max_abs_phiD=maxphi_local, max_abs_Phi=maxPhi_local,
            max_A2_contribution=maxcontrib,
            total_A2_contribution=totalcontrib,
            max_contribution_state=icontrib,
            max_contribution_state_p=(icontrib == 0 ? NaN : p[icontrib]),
            max_contribution_state_A=(icontrib == 0 ? NaN : A[icontrib]),
            max_contribution_state_q=(icontrib == 0 ? NaN : q[icontrib])))
    end

    order_nodes = sortperm(node_rows, by=r -> r.max_A2_contribution, rev=true)
    println("  Top nodes by max contribution to A(x)^2:")
    for k in order_nodes[1:min(top_nodes_to_print, length(order_nodes))]
        r = node_rows[k]
        @printf("    d=%d path=%-4s P=%.3e G1=%.3e D=%.3e G1/P=%.3e max|phiD|=%.3e max|Phi|=%.3e maxC=%.3e state=%d\n",
                r.node_depth, isempty(r.node_path) ? "root" : r.node_path,
                r.probability_mass, r.G1, r.margin, r.G1_over_probability_mass,
                r.max_abs_phiD, r.max_abs_Phi, r.max_A2_contribution,
                r.max_contribution_state)
    end

    state_rows = NamedTuple[]
    order_states = sortperm(A, rev=true)
    println("  Top states by A(x):")
    for rank in 1:min(top_states_to_print, length(order_states))
        i = order_states[rank]
        contribs = [(j, nodes[j].contribution[i]) for j in eachindex(nodes)
                    if nodes[j].contribution[i] > 0]
        sort!(contribs, by=t -> t[2], rev=true)
        topj = isempty(contribs) ? 0 : contribs[1][1]
        topc = isempty(contribs) ? 0.0 : contribs[1][2]
        topnode = topj == 0 ? 0 : nodes[topj].node_index
        toppath = topj == 0 ? "" : nodes[topj].node_path
        frac = A2[i] > 0 ? topc/A2[i] : 0.0
        @printf("    #%2d i=%4d x=%s A=%.3e A2=%.3e p=%.3e q=%.3e topnode=%d path=%s frac=%.3f\n",
                rank, i, state_string(@view X[i,:]), A[i], A2[i], p[i], q[i],
                topnode, isempty(toppath) ? "root" : toppath, frac)
        push!(state_rows, (training_run=run, epoch=epoch, rank=rank,
            state_index=i, state=state_string(@view X[i,:]), probability=p[i],
            acquisition_A=A[i], acquisition_A2=A2[i], oracle_q=q[i],
            dominant_node_index=topnode, dominant_node_path=toppath,
            dominant_node_fraction=frac))
    end

    checkpoint_row = (training_run=run, epoch=epoch, energy=real(fr.energy),
        target_rms=fr.target_rms, Amax=maximum(A), Amax_state=imax,
        Amax_state_p=p[imax], Amax_state_q=q[imax], qmax=qmax,
        q_participation_ratio=q_pr, q_entropy_effective_states=q_entropy_eff)

    return node_rows, state_rows, checkpoint_row
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

    node_rows = NamedTuple[]
    state_rows = NamedTuple[]
    checkpoint_rows = NamedTuple[]

    for epoch in 1:nepochs
        batch = vmc_batch(H, model, samples)
        yA, _ = make_targets(batch)
        w = batch.counts
        if epoch in checkpoint_epochs
            nr, sr, cr = diagnose_concentration(H, model, X, run, epoch)
            append!(node_rows, nr)
            append!(state_rows, sr)
            push!(checkpoint_rows, cr)
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
    return node_rows, state_rows, checkpoint_rows
end

function main()
    println("\n============================================================")
    println("ORACLE ACQUISITION CONCENTRATION DIAGNOSTIC")
    println("N=$N J/h=$(J/h) runs=$ntraining_runs checkpoints=$(sort(collect(checkpoint_epochs)))")
    println("Decomposes A(x)^2 node by node and identifies concentration sources")
    println("============================================================")

    X = enumerate_states(N)
    all_nodes = NamedTuple[]
    all_states = NamedTuple[]
    all_checkpoints = NamedTuple[]

    for run in 1:ntraining_runs
        @printf("\nTraining trajectory %d/%d\n", run, ntraining_runs)
        nr, sr, cr = run_training(run, X)
        append!(all_nodes, nr)
        append!(all_states, sr)
        append!(all_checkpoints, cr)
    end

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    nodefile = joinpath(outdir, "oracle_acquisition_concentration_nodes.csv")
    statefile = joinpath(outdir, "oracle_acquisition_concentration_states.csv")
    checkpointfile = joinpath(outdir, "oracle_acquisition_concentration_checkpoints.csv")
    write_namedtuple_csv(nodefile, all_nodes)
    write_namedtuple_csv(statefile, all_states)
    write_namedtuple_csv(checkpointfile, all_checkpoints)

    println("\nRESULTS WRITTEN TO")
    println(nodefile)
    println(statefile)
    println(checkpointfile)
end

export main, run_training, diagnose_concentration

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    OracleAcquisitionConcentrationExperiment.main()
end
