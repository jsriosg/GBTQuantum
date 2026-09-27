using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module OraclePathSurvivalExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

# Test the hierarchical interpretation of the mass-exponent ablation.
# For each oracle node v, estimate
#   s_v = P(all oracle ancestor splits on the path to v are recovered)
# from repeated importance-sampled reconstructions, and compare s_v with
# the simple proxies 1, P_v, and P_v^2.
#
# We also compare the empirical joint path survival with the independence
# approximation prod_{u in Anc(v)} rho_u, where rho_u is the marginal
# probability of recovering ancestor u's oracle split.

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
const mass_exponents = [1.0, 2.0, 3.0]
const base_seed = 1_420_000
const diagnostic_seed_base = 21_420_000
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
    function walk(k, d, idx, path, ancestors)
        n = tree.nodes[k]
        n.isleaf && return
        push!(out, (node_index=k, node_depth=d, node_path=path,
                    state_indices=copy(idx), ancestor_indices=copy(ancestors)))
        L = Int[]; R = Int[]
        f = Int(n.feature)
        for i in idx
            X[i, f] < 0 ? push!(L, i) : push!(R, i)
        end
        nextanc = [ancestors; k]
        walk(Int(n.left), d+1, L, path*"L", nextanc)
        walk(Int(n.right), d+1, R, path*"R", nextanc)
    end
    walk(1, 0, collect(axes(X,1)), "", Int[])
    return out
end

function conditional_oracle_gains(X, y, p, idx)
    mass = sum(p[idx])
    mass > 0 || return fill(-Inf, size(X,2)), mass
    pc = p ./ mass
    return gain_landscape(X, y, pc, idx; min_weight=exact_min_leaf_weight), mass
end

function gain_influence(X, y, p, idx, f)
    mass = sum(p[idx])
    phi = zeros(Float64, size(X,1))
    (mass > 0 && f != 0) || return phi
    r=0.0; a=0.0; m=0.0
    @inbounds for i in idx
        pv = p[i]/mass
        Li = X[i,f] < 0 ? 1.0 : 0.0
        r += pv*Li; a += pv*y[i]*Li; m += pv*y[i]
    end
    (r > eps() && 1-r > eps()) || return phi
    gr = -a^2/r^2 + (m-a)^2/(1-r)^2
    ga = 2a/r - 2(m-a)/(1-r)
    gm = 2(m-a)/(1-r) - 2m
    @inbounds for i in idx
        Li = X[i,f] < 0 ? 1.0 : 0.0
        phi[i] = gr*(Li-r) + ga*(y[i]*Li-a) + gm*(y[i]-m)
    end
    return phi
end

function build_oracle(H, model, X)
    fr = exact_frozen_problem(H, model, X)
    p = fr.probabilities
    tree = GBTQuantum.grow_tree(X, fr.target, p;
        max_depth=oracle_depth, min_weight=exact_min_leaf_weight, min_gain=0.0)
    regions = oracle_regions(tree, X)
    nodes = NamedTuple[]
    for n in regions
        og, mass = conditional_oracle_gains(X, fr.target, p, n.state_indices)
        f1,f2,G1,G2 = top_two(og)
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
    for n in nodes
        importance = n.probability_mass^alpha * max(n.G1,0.0)
        @. A2 += importance * n.Phi^2
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
    return q
end

function sampled_best_feature(X, y, p, q, draws, idx)
    mask = falses(size(X,1)); mask[idx] .= true
    localdraw = [i for i in draws if mask[i]]
    isempty(localdraw) && return 0
    u,c = compress_indices(localdraw)
    wraw = c .* p[u] ./ q[u]
    Z = sum(wraw)
    Z > 0 || return 0
    w = wraw ./ Z
    g = gain_landscape(X[u,:], y[u], w, collect(eachindex(u));
                       min_weight=sampled_min_leaf_weight)
    f,_,_,_ = top_two(g)
    return f
end

diagnostic_seed(run, epoch, M, ai, repetition) =
    diagnostic_seed_base + 1_000_000*run + 10_000*epoch + 100_000*ai + M + repetition

function diagnose(H, model, X, run, epoch)
    fr, nodes = build_oracle(H, model, X)
    nodepos = Dict(n.node_index => j for (j,n) in enumerate(nodes))
    raw_rows = NamedTuple[]
    summary_rows = NamedTuple[]

    @printf("\nEpoch %d: E=% .8f target RMS=%.4e nodes=%d\n",
            epoch, real(fr.energy), fr.target_rms, length(nodes))

    for (ai,alpha) in enumerate(mass_exponents)
        q = acquisition_for_alpha(fr, nodes, alpha)
        for M in diagnostic_sample_sizes
            correct = falses(nsampling_runs, length(nodes))
            path_survives = falses(nsampling_runs, length(nodes))

            for srun in 1:nsampling_runs
                rng = MersenneTwister(diagnostic_seed(run, epoch, M, ai, srun))
                draws = draw_categorical_indices(rng, q, M)
                for (j,n) in enumerate(nodes)
                    sf = sampled_best_feature(X, fr.target, fr.probabilities,
                                              q, draws, n.state_indices)
                    correct[srun,j] = sf != 0 && sf == n.f1
                end
                for (j,n) in enumerate(nodes)
                    ok = true
                    for aidx in n.ancestor_indices
                        ok &= correct[srun, nodepos[aidx]]
                    end
                    path_survives[srun,j] = ok
                end
            end

            rho = vec(mean(correct, dims=1))
            survival = vec(mean(path_survives, dims=1))

            for (j,n) in enumerate(nodes)
                indep = 1.0
                for aidx in n.ancestor_indices
                    indep *= rho[nodepos[aidx]]
                end
                Pv = n.probability_mass
                s = survival[j]
                push!(summary_rows, (training_run=run, epoch=epoch, alpha=alpha,
                    M=M, node_index=n.node_index, node_depth=n.node_depth,
                    node_path=n.node_path, probability_mass=Pv,
                    oracle_best_feature=n.f1, oracle_margin=n.margin,
                    marginal_split_recovery=rho[j], empirical_path_survival=s,
                    independent_path_survival=indep,
                    proxy_one=1.0, proxy_P=Pv, proxy_P2=Pv^2,
                    abs_error_proxy_one=abs(s-1.0),
                    abs_error_proxy_P=abs(s-Pv),
                    abs_error_proxy_P2=abs(s-Pv^2),
                    abs_error_independence=abs(s-indep)))
            end

            # Aggregate proxy quality over non-root nodes only: root path survival
            # is identically one and would trivially favour the constant proxy.
            nonroot = [j for (j,n) in enumerate(nodes) if n.node_depth > 0]
            mae1 = mean(abs(survival[j]-1.0) for j in nonroot)
            maeP = mean(abs(survival[j]-nodes[j].probability_mass) for j in nonroot)
            maeP2 = mean(abs(survival[j]-nodes[j].probability_mass^2) for j in nonroot)
            maeI = mean(begin
                prod_rho = prod(rho[nodepos[a]] for a in nodes[j].ancestor_indices)
                abs(survival[j]-prod_rho)
            end for j in nonroot)
            @printf("  alpha=%.1f M=%4d  path-proxy MAE: 1=%.3f  P=%.3f  P2=%.3f  indep=%.3f\n",
                    alpha, M, mae1, maeP, maeP2, maeI)
        end
    end
    return raw_rows, summary_rows
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

    raw=NamedTuple[]; summary=NamedTuple[]
    for epoch in 1:nepochs
        batch = vmc_batch(H, model, samples)
        yA,_ = make_targets(batch)
        w = batch.counts
        if epoch in checkpoint_epochs
            r,s = diagnose(H, model, X, run, epoch)
            append!(raw,r); append!(summary,s)
        end
        tree = GBTQuantum.grow_tree(batch.states, yA, w;
            max_depth=optimizer_max_depth,
            min_weight=optimizer_min_leaf_weight,
            min_gain=optimizer_min_gain)
        pred = predict_all(tree, batch.states)
        mu = weighted_mean(pred,w)
        isfinite(mu) && mu != 0 && (tree = shift_tree_leaves(tree,mu))
        push!(model.logamp.trees, scale_tree(tree,eta))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:sweeps_per_epoch
            GBTQuantum.sweep!(rng,model,samples,logamps)
        end
    end
    return raw,summary
end

function main()
    println("\n============================================================")
    println("ORACLE PATH-SURVIVAL DIAGNOSTIC")
    println("N=$N J/h=$(J/h) runs=$ntraining_runs checkpoints=$(sort(collect(checkpoint_epochs)))")
    println("alpha=$mass_exponents M=$diagnostic_sample_sizes repetitions=$nsampling_runs")
    println("Tests empirical s_v against 1, P_v, P_v^2, and product of ancestor recovery rates")
    println("============================================================")

    X = enumerate_states(N)
    raw=NamedTuple[]; summary=NamedTuple[]
    for run in 1:ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,ntraining_runs)
        r,s = run_training(run,X)
        append!(raw,r); append!(summary,s)
    end

    outdir = joinpath(@__DIR__,"results")
    mkpath(outdir)
    write_namedtuple_csv(joinpath(outdir,"oracle_path_survival_nodes.csv"),summary)
    println("\nResults written to experiments/results/oracle_path_survival_nodes.csv")
end

export main, run_training, diagnose

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    OraclePathSurvivalExperiment.main()
end
