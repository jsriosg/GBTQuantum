using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module SequentialNodeLocalAcquisitionExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

# Reuse the frozen-problem/oracle helpers from the first greedy experiment,
# but implement corrected inherited-sample weights and sequential acquisition.
include(joinpath(@__DIR__, "greedy_node_local_acquisition.jl"))
const G = GreedyNodeLocalAcquisitionExperiment

const z_thresholds = [0.0, 1.0, 2.0, 3.0, Inf]
const acquisition_batch = 128
const max_rounds_per_node = 4
const min_pilot_in_node = 16
const local_pilot_target = 64
const repetitions = 200
const output_name = "sequential_node_local_acquisition.csv"

# IMPORTANT correction relative to the first prototype:
# an observation drawn from an ancestor proposal q_u and reused in descendant v
# must use the CONDITIONAL ratio
#
#   p_v(x)/q_u(x|v) = [p(x)/P_v] / [q_u(x)/Q_u(v)]
#                    = p(x) Q_u(v) / [q_u(x) P_v].
#
# The factor Q_u(v)/P_v is common only among observations from the SAME origin
# proposal. It does NOT cancel when observations from several proposal rounds
# are pooled. The first prototype omitted it.
function observation_arrays(obs, p, node_idx)
    inset = falses(length(p)); inset[node_idx] .= true
    P = sum(p[node_idx])
    P > 0 || return (Int[], Float64[], G.Obs[])
    xs = Int[]; w = Float64[]; kept = G.Obs[]
    for o in obs
        inset[o.x] || continue
        Q = sum(o.proposal[node_idx])
        qi = o.proposal[o.x]
        if Q > 0 && qi > 0
            push!(xs, o.x)
            push!(w, (p[o.x] / P) / (qi / Q))
            push!(kept, o)
        end
    end
    return xs, w, kept
end

function estimated_landscape(X, y, p, node_idx, obs)
    xs, wraw, _ = observation_arrays(obs, p, node_idx)
    isempty(xs) && return (g=fill(-Inf, size(X,2)), xs=xs, w=Float64[])
    d = Dict{Int,Float64}()
    for (x,w) in zip(xs,wraw)
        d[x] = get(d,x,0.0) + w
    end
    u = sort!(collect(keys(d)))
    w = Float64[d[x] for x in u]
    Z = sum(w)
    Z > 0 || return (g=fill(-Inf,size(X,2)), xs=u, w=w)
    w ./= Z
    g = G.gain_landscape(X[u,:], y[u], w, collect(eachindex(u));
                         min_weight=G.sampled_min_leaf_weight)
    return (g=g, xs=u, w=w)
end

function estimated_if(X, y, p, node_idx, obs, f)
    est = estimated_landscape(X,y,p,node_idx,obs)
    u=est.xs; w=est.w
    psi=zeros(Float64,length(p))
    (f != 0 && !isempty(u)) || return psi
    r=0.0; a=0.0; m=0.0
    for (j,x) in enumerate(u)
        L = X[x,f] < 0 ? 1.0 : 0.0
        r += w[j]*L
        a += w[j]*y[x]*L
        m += w[j]*y[x]
    end
    (r > 1e-10 && 1-r > 1e-10) || return psi
    cr = -a*a/(r*r) + (m-a)^2/((1-r)^2)
    ca = 2a/r - 2(m-a)/(1-r)
    cm = 2(m-a)/(1-r) - 2m
    for x in node_idx
        L = X[x,f] < 0 ? 1.0 : 0.0
        psi[x] = cr*(L-r) + ca*(y[x]*L-a) + cm*(y[x]-m)
    end
    return psi
end

function acquisition_q(X,y,p,node_idx,obs,f1,f2)
    pv = G.conditional_p(p,node_idx)
    psi = estimated_if(X,y,p,node_idx,obs,f1) .-
          estimated_if(X,y,p,node_idx,obs,f2)
    qif = G.normalized_q(pv .* abs.(psi), pv)
    q = (1-G.if_born_epsilon).*qif .+ G.if_born_epsilon.*pv
    q ./= sum(q)
    return q
end

# Heuristic decision SNR used ONLY as an acquisition trigger. We do not assign
# it a fixed-sample Gaussian confidence interpretation. For heterogeneous
# proposal rounds, h_i = (p_v/q_i|v) psi(X_i) remains a valid first-order
# contribution for each retained observation.
function margin_diagnostic(X,y,p,node_idx,obs,f1,f2,G1,G2)
    f2 == 0 && return (D=Inf, se=0.0, z=Inf, n=0)
    psi = estimated_if(X,y,p,node_idx,obs,f1) .-
          estimated_if(X,y,p,node_idx,obs,f2)
    xs, ratios, _ = observation_arrays(obs,p,node_idx)
    n=length(xs)
    n < 2 && return (D=G1-G2,se=Inf,z=0.0,n=n)
    h = Float64[ratios[j]*psi[xs[j]] for j in eachindex(xs)]
    se = sqrt(var(h; corrected=true)/n)
    D = G1-G2
    z = (isfinite(se) && se>0) ? max(D,0.0)/se : (D>0 ? Inf : 0.0)
    return (D=D,se=se,z=z,n=n)
end

function draw_obs!(rng,obs,q,M,path)
    for x in draw_categorical_indices(rng,q,M)
        push!(obs,G.Obs(x,q,path))
    end
end

function grow_sequential!(rng,X,y,p,node_idx,path,depth,obs,decisions,cost,
                          threshold,round_log)
    depth >= G.oracle_depth && return
    length(node_idx) <= 1 && return

    ninherited = count(o -> (o.x in node_idx), obs)
    if ninherited < min_pilot_in_node
        pv=G.conditional_p(p,node_idx)
        add=max(local_pilot_target-ninherited,0)
        draw_obs!(rng,obs,pv,add,path)
        cost[] += add
    end

    rounds=0
    while true
        est=estimated_landscape(X,y,p,node_idx,obs)
        f1,f2,G1,G2=G.top_two(est.g)
        f1==0 && return
        md=margin_diagnostic(X,y,p,node_idx,obs,f1,f2,G1,G2)

        should_acquire = f2 != 0 && rounds < max_rounds_per_node && md.z < threshold
        if !should_acquire
            decisions[path]=f1
            push!(round_log,(path=path,depth=depth,rounds=rounds,z=md.z,n=md.n))
            L,R=G.split_indices(X,node_idx,f1)
            grow_sequential!(rng,X,y,p,L,path*"L",depth+1,obs,decisions,cost,threshold,round_log)
            grow_sequential!(rng,X,y,p,R,path*"R",depth+1,obs,decisions,cost,threshold,round_log)
            return
        end

        q=acquisition_q(X,y,p,node_idx,obs,f1,f2)
        draw_obs!(rng,obs,q,acquisition_batch,path)
        cost[] += acquisition_batch
        rounds += 1
    end
end

function adaptive_trial(rng,X,fr,threshold)
    obs=G.Obs[]
    qroot=copy(fr.probabilities)
    draw_obs!(rng,obs,qroot,G.pilot_root,"")
    cost=Ref(G.pilot_root)
    decisions=Dict{String,Int}()
    round_log=NamedTuple[]
    grow_sequential!(rng,X,fr.target,fr.probabilities,collect(axes(X,1)),"",0,
                     obs,decisions,cost,threshold,round_log)
    return decisions,cost[],round_log
end

# A node is eligible for conditional recovery only if every strict ancestor
# decision on its oracle path was reproduced. This avoids comparing different
# Hilbert-space regions that merely share the same L/R path label.
function ancestors_correct(decisions,oracle,path)
    d=length(path)
    d==0 && return true
    for k in 0:(d-1)
        prefix = k==0 ? "" : path[1:k]
        haskey(oracle,prefix) || return false
        get(decisions,prefix,0)==oracle[prefix].feature || return false
    end
    return true
end

function conditional_depth_score(decisions,oracle)
    eligible=zeros(Int,G.oracle_depth)
    correct=zeros(Int,G.oracle_depth)
    for (path,n) in oracle
        d=n.depth
        d < G.oracle_depth || continue
        if ancestors_correct(decisions,oracle,path)
            eligible[d+1]+=1
            correct[d+1]+=get(decisions,path,0)==n.feature
        end
    end
    return eligible,correct
end

function full_tree_score(decisions,oracle)
    all(get(decisions,path,0)==n.feature for (path,n) in oracle)
end

function diagnose(H,model,X,run,epoch)
    fr=G.exact_frozen_problem(H,model,X)
    _,oracle=G.exact_oracle(fr,X)
    rows=NamedTuple[]
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e oracle internal nodes=%d\n",
            epoch,real(fr.energy),fr.target_rms,length(oracle))

    # Same Born baseline as before, now with conditional-by-depth diagnostics.
    belig=zeros(Int,G.oracle_depth); bcorr=zeros(Int,G.oracle_depth); ball=0
    for rep in 1:repetitions
        rng=MersenneTwister(G.diagnostic_seed_base+90_000_000+10_000_000*run+100_000*epoch+rep)
        dec=G.baseline_decisions(rng,X,fr)
        e,c=conditional_depth_score(dec,oracle); belig .+= e; bcorr .+= c
        ball += full_tree_score(dec,oracle)
    end
    br=[belig[d]>0 ? bcorr[d]/belig[d] : NaN for d in 1:G.oracle_depth]
    @printf("  Born M=%d: conditional depth recovery=%s full-tree=%.3f cost=%d\n",
            G.baseline_M,string(round.(br,digits=3)),ball/repetitions,G.baseline_M)
    for d in 0:(G.oracle_depth-1)
        push!(rows,(training_run=run,epoch=epoch,method="global_born",threshold=NaN,
            depth=d,conditional_recovery=br[d+1],eligible_nodes=belig[d+1],
            full_tree_recovery=ball/repetitions,mean_evaluations=Float64(G.baseline_M),
            mean_acquisition_rounds=0.0,repetitions=repetitions))
    end

    for (ti,threshold) in enumerate(z_thresholds)
        elig=zeros(Int,G.oracle_depth); corr=zeros(Int,G.oracle_depth); allok=0
        costs=Float64[]; rounds=Float64[]
        for rep in 1:repetitions
            rng=MersenneTwister(G.diagnostic_seed_base+ti*20_000_000+
                10_000_000*run+100_000*epoch+rep)
            dec,cost,rlog=adaptive_trial(rng,X,fr,threshold)
            e,c=conditional_depth_score(dec,oracle); elig .+= e; corr .+= c
            allok += full_tree_score(dec,oracle)
            push!(costs,cost)
            push!(rounds,isempty(rlog) ? 0.0 : mean(r.rounds for r in rlog))
        end
        rr=[elig[d]>0 ? corr[d]/elig[d] : NaN for d in 1:G.oracle_depth]
        mc=mean(costs); mr=mean(rounds); fa=allok/repetitions
        label=isinf(threshold) ? "Inf" : string(threshold)
        @printf("  z<%-3s acquire: depth recovery=%s full-tree=%.3f cost=%.1f rounds/node=%.2f\n",
                label,string(round.(rr,digits=3)),fa,mc,mr)
        for d in 0:(G.oracle_depth-1)
            push!(rows,(training_run=run,epoch=epoch,method="sequential_if",threshold=threshold,
                depth=d,conditional_recovery=rr[d+1],eligible_nodes=elig[d+1],
                full_tree_recovery=fa,mean_evaluations=mc,
                mean_acquisition_rounds=mr,repetitions=repetitions))
        end
    end
    return rows
end

function run_training(run,X)
    H=TFIMHamiltonian(G.N;J=G.J,h=G.h,periodic=true)
    rng=MersenneTwister(G.base_seed+10_000*run)
    samples=Matrix{Int8}(undef,G.training_nsamples,G.N)
    @inbounds for i in eachindex(samples)
        samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1)
    end
    model=LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    logamps=zeros(G.training_nsamples)
    for _ in 1:G.burn_in_sweeps
        GBTQuantum.sweep!(rng,model,samples,logamps)
    end
    rows=NamedTuple[]
    for epoch in 1:G.nepochs
        batch=vmc_batch(H,model,samples)
        yA,_=make_targets(batch); w=batch.counts
        if epoch in G.checkpoint_epochs
            append!(rows,diagnose(H,model,X,run,epoch))
        end
        tree=GBTQuantum.grow_tree(batch.states,yA,w;
            max_depth=G.optimizer_max_depth,min_weight=G.optimizer_min_leaf_weight,
            min_gain=G.optimizer_min_gain)
        pred=predict_all(tree,batch.states); mu=weighted_mean(pred,w)
        if isfinite(mu) && mu != 0
            tree=shift_tree_leaves(tree,mu)
        end
        push!(model.logamp.trees,scale_tree(tree,G.eta))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:G.sweeps_per_epoch
            GBTQuantum.sweep!(rng,model,samples,logamps)
        end
    end
    return rows
end

function main()
    println("\n============================================================")
    println("SEQUENTIAL NODE-LOCAL IF ACQUISITION")
    println("N=$(G.N) J/h=$(G.J/G.h) runs=$(G.ntraining_runs) checkpoints=$(sort(collect(G.checkpoint_epochs)))")
    println("root pilot=$(G.pilot_root), batch=$acquisition_batch, max rounds/node=$max_rounds_per_node")
    println("threshold sweep=$z_thresholds, epsilon=$(G.if_born_epsilon)")
    println("IMPORTANT: corrected ancestor->descendant conditional importance weights")
    println("metrics: conditional recovery by depth, full-tree recovery, evaluation cost")
    println("============================================================")
    X=enumerate_states(G.N); rows=NamedTuple[]
    for run in 1:G.ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,G.ntraining_runs)
        append!(rows,run_training(run,X))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,output_name)
    write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/$output_name")
end

export main,run_training,diagnose

end

if abspath(PROGRAM_FILE) == @__FILE__
    SequentialNodeLocalAcquisitionExperiment.main()
end
