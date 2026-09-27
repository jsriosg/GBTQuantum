using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module InfluenceAcquisitionValidationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

# Validate the first-order influence-function acquisition derived for recovery
# of the oracle tree split margin D_v = G_best - G_second.
#
# Stage 1 is deliberately node-local: for each oracle node v, compare Born,
# p*y^2, pure q*_v ∝ p*|psi_v|, and a support-safe mixture of q*_v with Born.
# This isolates the theorem from the separate problem of combining nodes into
# one global tree-growing acquisition distribution.

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
const nsampling_runs = 200
const epsilon_mix = 0.05
const support_floor = 1e-14
const base_seed = 1_730_000
const diagnostic_seed_base = 31_730_000
const gauge_shifts = [-7.25, -1.0, 0.5, 9.75]
const gauge_tolerance = 1e-10

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
    walk(1, 0, collect(axes(X,1)), "")
    return out
end

# Conditional gain in oracle region v. pc is normalized over that region.
function conditional_oracle_gains(X, y, p, idx)
    Pv = sum(p[idx])
    Pv > 0 || return fill(-Inf, size(X,2)), Pv
    pc = p ./ Pv
    return gain_landscape(X, y, pc, idx; min_weight=exact_min_leaf_weight), Pv
end

# Existing centered influence expression, retained independently as a cross-check.
# This is the IF for one candidate gain under the conditional distribution in v.
function gain_influence_centered(X, y, p, idx, f)
    Pv = sum(p[idx])
    phi = zeros(Float64, size(X,1))
    (Pv > 0 && f != 0) || return phi
    r=0.0; a=0.0; m=0.0
    @inbounds for i in idx
        pv = p[i]/Pv
        L = X[i,f] < 0 ? 1.0 : 0.0
        r += pv*L
        a += pv*y[i]*L
        m += pv*y[i]
    end
    (r > eps() && 1-r > eps()) || return phi
    gr = -a^2/r^2 + (m-a)^2/(1-r)^2
    ga = 2a/r - 2(m-a)/(1-r)
    gm = 2(m-a)/(1-r) - 2m
    @inbounds for i in idx
        L = X[i,f] < 0 ? 1.0 : 0.0
        phi[i] = gr*(L-r) + ga*(y[i]*L-a) + gm*(y[i]-m)
    end
    return phi
end

# Closed form of the same IF.  Let mu_L and mu_R be conditional target means.
# Since G = r(1-r)(mu_L-mu_R)^2, its contamination IF simplifies to
#   (y-mu)^2 - (y-mu_side)^2 - G
# for a point in the corresponding child.  The -G centering is essential.
function gain_influence_closed(X, y, p, idx, f)
    Pv = sum(p[idx])
    phi = zeros(Float64, size(X,1))
    (Pv > 0 && f != 0) || return phi
    pc = p ./ Pv
    r = sum(pc[i] for i in idx if X[i,f] < 0)
    (r > eps() && 1-r > eps()) || return phi
    mu = sum(pc[i]*y[i] for i in idx)
    muL = sum(pc[i]*y[i] for i in idx if X[i,f] < 0) / r
    muR = sum(pc[i]*y[i] for i in idx if X[i,f] >= 0) / (1-r)
    G = r*(1-r)*(muL-muR)^2
    @inbounds for i in idx
        mus = X[i,f] < 0 ? muL : muR
        phi[i] = (y[i]-mu)^2 - (y[i]-mus)^2 - G
    end
    return phi
end

function build_oracle(H, model, X)
    fr = exact_frozen_problem(H, model, X)
    p = fr.probabilities
    tree = GBTQuantum.grow_tree(X, fr.target, p;
        max_depth=oracle_depth, min_weight=exact_min_leaf_weight, min_gain=0.0)
    nodes = NamedTuple[]
    max_if_error = 0.0
    for n in oracle_regions(tree, X)
        og, Pv = conditional_oracle_gains(X, fr.target, p, n.state_indices)
        f1,f2,G1,G2 = top_two(og)
        f2 == 0 && continue
        old1 = gain_influence_centered(X, fr.target, p, n.state_indices, f1)
        old2 = gain_influence_centered(X, fr.target, p, n.state_indices, f2)
        new1 = gain_influence_closed(X, fr.target, p, n.state_indices, f1)
        new2 = gain_influence_closed(X, fr.target, p, n.state_indices, f2)
        psi_old = old1 .- old2
        psi = new1 .- new2
        err = maximum(abs.(psi_old .- psi))
        max_if_error = max(max_if_error, err)
        push!(nodes, merge(n, (probability_mass=Pv, oracle_gains=og,
            f1=f1, f2=f2, G1=G1, G2=G2, margin=G1-G2,
            psi=psi, if_crosscheck_error=err)))
    end
    max_if_error <= gauge_tolerance || error("closed-form IF cross-check failed: max error=$max_if_error")
    return fr, nodes, max_if_error
end

function gauge_check(X, fr, nodes)
    p = fr.probabilities
    max_gain_err = 0.0
    max_margin_err = 0.0
    max_psi_err = 0.0
    max_q_err = 0.0
    for c in gauge_shifts
        ys = fr.target .+ c
        for n in nodes
            gs, _ = conditional_oracle_gains(X, ys, p, n.state_indices)
            G1s = gs[n.f1]; G2s = gs[n.f2]
            psis = gain_influence_closed(X, ys, p, n.state_indices, n.f1) .-
                    gain_influence_closed(X, ys, p, n.state_indices, n.f2)
            q0 = node_if_distribution(p, n.psi)
            qs = node_if_distribution(p, psis)
            max_gain_err = max(max_gain_err, abs(G1s-n.G1), abs(G2s-n.G2))
            max_margin_err = max(max_margin_err, abs((G1s-G2s)-n.margin))
            max_psi_err = max(max_psi_err, maximum(abs.(psis .- n.psi)))
            max_q_err = max(max_q_err, maximum(abs.(qs .- q0)))
        end
    end
    maximum((max_gain_err,max_margin_err,max_psi_err,max_q_err)) <= gauge_tolerance ||
        error("gauge-invariance check failed")
    return (gain=max_gain_err, margin=max_margin_err, psi=max_psi_err, q=max_q_err)
end

function normalize_q(q, p)
    q = max.(q, support_floor .* p)
    Z = sum(q)
    return Z > 0 ? q ./ Z : copy(p)
end

node_if_distribution(p, psi) = normalize_q(p .* abs.(psi), p)

function node_if_mixture(p, psi)
    q0 = node_if_distribution(p, psi)
    return normalize_q((1-epsilon_mix).*q0 .+ epsilon_mix.*p, p)
end

function naive_distribution(p, y)
    return normalize_q(p .* y.^2, p)
end

function sampler_set(fr, n)
    p = fr.probabilities
    return [
        (name="born", q=copy(p)),
        (name="naive_py2", q=naive_distribution(p, fr.target)),
        (name="if_pure", q=node_if_distribution(p, n.psi)),
        (name="if_mix", q=node_if_mixture(p, n.psi)),
    ]
end

# First-order asymptotic variance of the conditional-node margin estimator.
# For global q, conditioning on node v gives the 1/Pv^2 factor.
function theoretical_variance_constant(p, q, psi, idx, Pv)
    Pv > 0 || return Inf
    s = 0.0
    @inbounds for i in idx
        q[i] > 0 || continue
        s += p[i]^2 * psi[i]^2 / q[i]
    end
    return s / Pv^2
end

function sampled_node_stats(X, y, p, q, draws, n)
    mask = falses(size(X,1)); mask[n.state_indices] .= true
    localdraw = [i for i in draws if mask[i]]
    isempty(localdraw) && return (best=0, D=NaN, relgain=0.0)
    u,c = compress_indices(localdraw)
    wraw = c .* p[u] ./ q[u]
    Z = sum(wraw)
    Z > 0 || return (best=0, D=NaN, relgain=0.0)
    w = wraw ./ Z
    g = gain_landscape(X[u,:], y[u], w, collect(eachindex(u));
                       min_weight=sampled_min_leaf_weight)
    best,_,bestgain,_ = top_two(g)
    D = (isfinite(g[n.f1]) && isfinite(g[n.f2])) ? g[n.f1]-g[n.f2] : NaN
    relgain = n.G1 > 0 && best != 0 && isfinite(g[best]) ? max(g[best],0.0)/n.G1 : 0.0
    return (best=best, D=D, relgain=relgain)
end

diagnostic_seed(run, epoch, M, nodej, samplerj, repetition) =
    diagnostic_seed_base + 10_000_000*run + 100_000*epoch +
    10_000*nodej + 1_000*samplerj + M + repetition

function diagnose(H, model, X, run, epoch)
    fr, nodes, iferr = build_oracle(H, model, X)
    gauge = gauge_check(X, fr, nodes)
    rows = NamedTuple[]

    @printf("\nEpoch %d: E=% .8f target RMS=%.4e nodes=%d\n",
            epoch, real(fr.energy), fr.target_rms, length(nodes))
    @printf("  IF cross-check max=%.3e | gauge max: G=%.3e D=%.3e psi=%.3e q=%.3e\n",
            iferr, gauge.gain, gauge.margin, gauge.psi, gauge.q)

    for (nodej,n) in enumerate(nodes)
        for (samplerj,sampler) in enumerate(sampler_set(fr,n))
            q = sampler.q
            Vtheory = theoretical_variance_constant(fr.probabilities, q, n.psi,
                                                     n.state_indices, n.probability_mass)
            Vopt = theoretical_variance_constant(fr.probabilities,
                node_if_distribution(fr.probabilities,n.psi), n.psi,
                n.state_indices, n.probability_mass)
            efficiency = isfinite(Vtheory) && Vtheory > 0 ? Vopt/Vtheory : NaN

            for M in diagnostic_sample_sizes
                Dhat = Float64[]
                correct = 0
                signcorrect = 0
                relgains = Float64[]
                for rep in 1:nsampling_runs
                    rng = MersenneTwister(diagnostic_seed(run,epoch,M,nodej,samplerj,rep))
                    draws = draw_categorical_indices(rng,q,M)
                    st = sampled_node_stats(X,fr.target,fr.probabilities,q,draws,n)
                    correct += st.best == n.f1
                    push!(relgains,st.relgain)
                    if isfinite(st.D)
                        push!(Dhat,st.D)
                        signcorrect += st.D > 0
                    end
                end
                nd = length(Dhat)
                meanD = nd > 0 ? mean(Dhat) : NaN
                biasD = nd > 0 ? meanD-n.margin : NaN
                varD = nd > 1 ? var(Dhat; corrected=true) : NaN
                Mvar = isfinite(varD) ? M*varD : NaN
                ratio = isfinite(Mvar) && Vtheory > 0 ? Mvar/Vtheory : NaN
                recovery = correct/nsampling_runs
                signrec = nd > 0 ? signcorrect/nd : NaN
                meanrg = mean(relgains)
                push!(rows,(training_run=run,epoch=epoch,node_index=n.node_index,
                    node_depth=n.node_depth,node_path=n.node_path,
                    probability_mass=n.probability_mass,f1=n.f1,f2=n.f2,
                    oracle_G1=n.G1,oracle_G2=n.G2,oracle_margin=n.margin,
                    sampler=sampler.name,M=M,repetitions=nsampling_runs,
                    finite_margin_repetitions=nd,split_recovery=recovery,
                    sign_recovery=signrec,mean_relative_gain=meanrg,
                    mean_margin_estimate=meanD,margin_bias=biasD,
                    margin_variance=varD,M_times_margin_variance=Mvar,
                    theoretical_variance_constant=Vtheory,
                    empirical_to_theory_variance_ratio=ratio,
                    theoretical_efficiency_vs_IF=efficiency,
                    if_crosscheck_error=n.if_crosscheck_error,
                    gauge_gain_error=gauge.gain,gauge_margin_error=gauge.margin,
                    gauge_psi_error=gauge.psi,gauge_q_error=gauge.q))
            end
        end
    end

    # Compact console summary over nodes for each sampler and M.
    for sampler in ("born","naive_py2","if_pure","if_mix")
        for M in diagnostic_sample_sizes
            r = [x for x in rows if x.sampler==sampler && x.M==M]
            @printf("  %-9s M=%4d  recovery=%.3f sign=%.3f RG=%.3f |biasD|=%.3e MVar/V=%.3f eff=%.3f\n",
                sampler,M,mean(x.split_recovery for x in r),
                mean(x.sign_recovery for x in r if isfinite(x.sign_recovery)),
                mean(x.mean_relative_gain for x in r),
                mean(abs(x.margin_bias) for x in r if isfinite(x.margin_bias)),
                mean(x.empirical_to_theory_variance_ratio for x in r if isfinite(x.empirical_to_theory_variance_ratio)),
                mean(x.theoretical_efficiency_vs_IF for x in r if isfinite(x.theoretical_efficiency_vs_IF)))
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

    rows=NamedTuple[]
    for epoch in 1:nepochs
        batch = vmc_batch(H, model, samples)
        yA,_ = make_targets(batch)
        w = batch.counts
        if epoch in checkpoint_epochs
            append!(rows,diagnose(H,model,X,run,epoch))
        end
        tree = GBTQuantum.grow_tree(batch.states,yA,w;
            max_depth=optimizer_max_depth,
            min_weight=optimizer_min_leaf_weight,
            min_gain=optimizer_min_gain)
        pred = predict_all(tree,batch.states)
        mu = weighted_mean(pred,w)
        isfinite(mu) && mu != 0 && (tree=shift_tree_leaves(tree,mu))
        push!(model.logamp.trees,scale_tree(tree,eta))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:sweeps_per_epoch
            GBTQuantum.sweep!(rng,model,samples,logamps)
        end
    end
    return rows
end

function main()
    println("\n============================================================")
    println("INFLUENCE-FUNCTION ACQUISITION VALIDATION")
    println("N=$N J/h=$(J/h) runs=$ntraining_runs checkpoints=$(sort(collect(checkpoint_epochs)))")
    println("M=$diagnostic_sample_sizes repetitions=$nsampling_runs epsilon=$epsilon_mix")
    println("Samplers: Born, p*y^2, p*|psi_v|, and epsilon-safe IF mixture")
    println("Tests gauge symmetry, IF identity, split recovery, and M Var(Dhat) -> V(q)")
    println("============================================================")

    X = enumerate_states(N)
    rows=NamedTuple[]
    for run in 1:ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,ntraining_runs)
        append!(rows,run_training(run,X))
    end
    outdir=joinpath(@__DIR__,"results")
    mkpath(outdir)
    outfile=joinpath(outdir,"influence_acquisition_validation.csv")
    write_namedtuple_csv(outfile,rows)
    println("\nResults written to experiments/results/influence_acquisition_validation.csv")
end

export main, run_training, diagnose

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    InfluenceAcquisitionValidationExperiment.main()
end
