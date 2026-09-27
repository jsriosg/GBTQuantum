using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module GreedyNodeLocalAcquisitionExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

# First end-to-end test of greedy node-local acquisition.
#
# We freeze a VMC state, build the exact oracle tree, and compare:
#   (1) a fixed global Born sample used to grow the whole tree;
#   (2) a recursive node-local procedure:
#       inherited pilot -> estimate top two -> IF acquisition -> choose split -> recurse.
#
# This experiment deliberately uses the exact frozen p(x), y(x) to isolate the
# sampling/tree-growth question. The IF itself is estimated only from the
# node's currently available weighted pilot data.

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

const pilot_root = 256
const acquisition_batch = 256
const max_rounds_per_node = 1
const if_born_epsilon = 0.05
const min_pilot_in_node = 16
const repetitions = 200
const baseline_M = 1024
const support_floor = 1e-14
const base_seed = 1_920_000
const diagnostic_seed_base = 51_920_000

struct Obs
    x::Int
    proposal::Vector{Float64}   # conditional proposal on the node where drawn
    origin_path::String
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
            if X[i,f] < 0
                WL += w[i]; SL += w[i]*y[i]
            end
        end
        WR = W-WL
        (WL < min_weight || WR < min_weight || WL <= 0 || WR <= 0) && continue
        SR = S-SL
        g[f] = SL*SL/WL + SR*SR/WR - parent
    end
    return g
end

function top_two(g)
    v = [i for i in eachindex(g) if isfinite(g[i]) && g[i] >= 0]
    isempty(v) && return (0,0,0.0,0.0)
    sort!(v, by=i->g[i], rev=true)
    f1=v[1]; f2=length(v)>=2 ? v[2] : 0
    return (f1,f2,g[f1],f2==0 ? 0.0 : g[f2])
end

function oracle_regions(tree, X)
    out = Dict{String,NamedTuple}()
    function walk(k,d,idx,path)
        n=tree.nodes[k]
        n.isleaf && return
        out[path]=(feature=Int(n.feature), depth=d, state_indices=copy(idx))
        L=Int[]; R=Int[]; f=Int(n.feature)
        for i in idx
            X[i,f] < 0 ? push!(L,i) : push!(R,i)
        end
        walk(Int(n.left),d+1,L,path*"L")
        walk(Int(n.right),d+1,R,path*"R")
    end
    walk(1,0,collect(axes(X,1)),"")
    return out
end

function exact_oracle(fr,X)
    tree=GBTQuantum.grow_tree(X,fr.target,fr.probabilities;
        max_depth=oracle_depth,min_weight=exact_min_leaf_weight,min_gain=0.0)
    return tree,oracle_regions(tree,X)
end

function conditional_p(p, idx)
    q=zeros(Float64,length(p)); P=sum(p[idx]); P>0 || return q
    q[idx].=p[idx]./P
    return q
end

function normalized_q(raw, fallback)
    q=max.(Float64.(raw),0.0); Z=sum(q)
    if !(Z>eps()) || !isfinite(Z)
        return copy(fallback)
    end
    q./=Z
    q=max.(q,support_floor.*fallback)
    q./=sum(q)
    return q
end

# Reweight observations for the CURRENT node. An observation may have been
# drawn at an ancestor. Conditioning its ancestor proposal onto the current
# node only introduces a common constant, which cancels in self-normalized
# statistics. Hence p(x)/q_origin(x) is sufficient up to node-wise scale.
function observation_arrays(obs, p, node_idx)
    inset=falses(length(p)); inset[node_idx].=true
    kept=[o for o in obs if inset[o.x]]
    isempty(kept) && return (Int[],Float64[])
    xs=[o.x for o in kept]
    w=Float64[]
    for o in kept
        qi=o.proposal[o.x]
        push!(w, qi>0 ? p[o.x]/qi : 0.0)
    end
    return xs,w
end

function estimated_landscape(X,y,p,node_idx,obs)
    xs,wraw=observation_arrays(obs,p,node_idx)
    isempty(xs) && return (g=fill(-Inf,size(X,2)), xs=xs, w=Float64[])
    # compress repeated states while preserving summed importance mass
    d=Dict{Int,Float64}()
    for (x,w) in zip(xs,wraw)
        d[x]=get(d,x,0.0)+w
    end
    u=sort!(collect(keys(d))); w=Float64[d[x] for x in u]
    Z=sum(w); Z>0 || return (g=fill(-Inf,size(X,2)),xs=u,w=w)
    w./=Z
    g=gain_landscape(X[u,:],y[u],w,collect(eachindex(u));min_weight=sampled_min_leaf_weight)
    return (g=g,xs=u,w=w)
end

function estimated_if(X,y,p,node_idx,obs,f)
    est=estimated_landscape(X,y,p,node_idx,obs)
    u=est.xs; w=est.w
    psi=zeros(Float64,length(p))
    (f!=0 && !isempty(u)) || return psi
    r=0.0; a=0.0; m=0.0
    for (j,x) in enumerate(u)
        L=X[x,f]<0 ? 1.0 : 0.0
        r+=w[j]*L; a+=w[j]*y[x]*L; m+=w[j]*y[x]
    end
    (r>1e-10 && 1-r>1e-10) || return psi
    cr=-a*a/(r*r)+(m-a)^2/((1-r)^2)
    ca=2a/r-2(m-a)/(1-r)
    cm=2(m-a)/(1-r)-2m
    for x in node_idx
        L=X[x,f]<0 ? 1.0 : 0.0
        psi[x]=cr*(L-r)+ca*(y[x]*L-a)+cm*(y[x]-m)
    end
    return psi
end

function acquisition_q(X,y,p,node_idx,obs,f1,f2)
    pv=conditional_p(p,node_idx)
    psi1=estimated_if(X,y,p,node_idx,obs,f1)
    psi2=estimated_if(X,y,p,node_idx,obs,f2)
    psi=psi1.-psi2
    qif=normalized_q(pv.*abs.(psi),pv)
    q=(1-if_born_epsilon).*qif .+ if_born_epsilon.*pv
    q./=sum(q)
    return q
end

function draw_obs!(rng,obs,q,M,path)
    for x in draw_categorical_indices(rng,q,M)
        push!(obs,Obs(x,q,path))
    end
end

function split_indices(X,idx,f)
    L=Int[];R=Int[]
    for x in idx
        X[x,f]<0 ? push!(L,x) : push!(R,x)
    end
    return L,R
end

function grow_adaptive!(rng,X,y,p,node_idx,path,depth,obs,decisions,cost)
    depth>=oracle_depth && return
    length(node_idx)<=1 && return

    # If inherited data are too sparse, top up with a local Born pilot.
    ninherited=count(o->(o.x in node_idx),obs)
    if ninherited < min_pilot_in_node
        pv=conditional_p(p,node_idx)
        add=max(pilot_root-ninherited,0)
        draw_obs!(rng,obs,pv,add,path)
        cost[]+=add
    end

    est=estimated_landscape(X,y,p,node_idx,obs)
    f1,f2,_,_=top_two(est.g)
    f1==0 && return

    for _ in 1:max_rounds_per_node
        f2==0 && break
        q=acquisition_q(X,y,p,node_idx,obs,f1,f2)
        draw_obs!(rng,obs,q,acquisition_batch,path)
        cost[]+=acquisition_batch
        est=estimated_landscape(X,y,p,node_idx,obs)
        f1,f2,_,_=top_two(est.g)
        f1==0 && return
    end

    decisions[path]=f1
    L,R=split_indices(X,node_idx,f1)
    grow_adaptive!(rng,X,y,p,L,path*"L",depth+1,obs,decisions,cost)
    grow_adaptive!(rng,X,y,p,R,path*"R",depth+1,obs,decisions,cost)
end

function adaptive_trial(rng,X,fr)
    obs=Obs[]
    qroot=copy(fr.probabilities)
    draw_obs!(rng,obs,qroot,pilot_root,"")
    cost=Ref(pilot_root)
    decisions=Dict{String,Int}()
    grow_adaptive!(rng,X,fr.target,fr.probabilities,collect(axes(X,1)),"",0,obs,decisions,cost)
    return decisions,cost[]
end

function baseline_trial(rng,X,fr)
    draws=draw_categorical_indices(rng,fr.probabilities,baseline_M)
    u,c=compress_indices(draws)
    tree=GBTQuantum.grow_tree(X[u,:],fr.target[u],c;
        max_depth=oracle_depth,min_weight=sampled_min_leaf_weight,min_gain=0.0)
    return oracle_regions(tree,X[u,:])
end

function score_decisions(decisions,oracle)
    isempty(oracle) && return (correct=0,total=0,all=false)
    correct=0
    for (path,n) in oracle
        correct += get(decisions,path,0)==n.feature
    end
    total=length(oracle)
    return (correct=correct,total=total,all=(correct==total))
end

function baseline_decisions(rng,X,fr)
    draws=draw_categorical_indices(rng,fr.probabilities,baseline_M)
    u,c=compress_indices(draws)
    tree=GBTQuantum.grow_tree(X[u,:],fr.target[u],c;
        max_depth=oracle_depth,min_weight=sampled_min_leaf_weight,min_gain=0.0)
    # Paths must be evaluated on the full X, not the compressed matrix.
    regs=oracle_regions(tree,X)
    return Dict(path=>n.feature for (path,n) in regs)
end

function diagnose(H,model,X,run,epoch)
    fr=exact_frozen_problem(H,model,X)
    _,oracle=exact_oracle(fr,X)
    rows=NamedTuple[]
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e oracle internal nodes=%d\n",
            epoch,real(fr.energy),fr.target_rms,length(oracle))

    adaptive_correct=0; adaptive_total=0; adaptive_all=0; adaptive_cost=Float64[]
    base_correct=0; base_total=0; base_all=0

    for rep in 1:repetitions
        rng1=MersenneTwister(diagnostic_seed_base+10_000_000*run+100_000*epoch+rep)
        dec,cost=adaptive_trial(rng1,X,fr)
        s=score_decisions(dec,oracle)
        adaptive_correct+=s.correct; adaptive_total+=s.total; adaptive_all+=s.all
        push!(adaptive_cost,cost)

        rng2=MersenneTwister(diagnostic_seed_base+20_000_000+10_000_000*run+100_000*epoch+rep)
        bdec=baseline_decisions(rng2,X,fr)
        sb=score_decisions(bdec,oracle)
        base_correct+=sb.correct; base_total+=sb.total; base_all+=sb.all
    end

    ar=adaptive_total>0 ? adaptive_correct/adaptive_total : NaN
    br=base_total>0 ? base_correct/base_total : NaN
    aa=adaptive_all/repetitions; ba=base_all/repetitions
    mc=mean(adaptive_cost)
    @printf("  adaptive: node recovery=%.3f full-tree=%.3f mean new evaluations=%.1f\n",ar,aa,mc)
    @printf("  Born M=%d: node recovery=%.3f full-tree=%.3f evaluations=%d\n",baseline_M,br,ba,baseline_M)

    push!(rows,(training_run=run,epoch=epoch,energy=real(fr.energy),target_rms=fr.target_rms,
        method="adaptive_node_if",node_recovery=ar,full_tree_recovery=aa,
        mean_evaluations=mc,pilot_root=pilot_root,acquisition_batch=acquisition_batch,
        max_rounds=max_rounds_per_node,epsilon=if_born_epsilon,repetitions=repetitions))
    push!(rows,(training_run=run,epoch=epoch,energy=real(fr.energy),target_rms=fr.target_rms,
        method="global_born",node_recovery=br,full_tree_recovery=ba,
        mean_evaluations=Float64(baseline_M),pilot_root=pilot_root,acquisition_batch=acquisition_batch,
        max_rounds=max_rounds_per_node,epsilon=if_born_epsilon,repetitions=repetitions))
    return rows
end

function run_training(run,X)
    H=TFIMHamiltonian(N;J=J,h=h,periodic=true)
    rng=MersenneTwister(base_seed+10_000*run)
    samples=Matrix{Int8}(undef,training_nsamples,N)
    @inbounds for i in eachindex(samples)
        samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1)
    end
    model=LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    logamps=zeros(training_nsamples)
    for _ in 1:burn_in_sweeps
        GBTQuantum.sweep!(rng,model,samples,logamps)
    end
    rows=NamedTuple[]
    for epoch in 1:nepochs
        batch=vmc_batch(H,model,samples)
        yA,_=make_targets(batch); w=batch.counts
        if epoch in checkpoint_epochs
            append!(rows,diagnose(H,model,X,run,epoch))
        end
        tree=GBTQuantum.grow_tree(batch.states,yA,w;
            max_depth=optimizer_max_depth,min_weight=optimizer_min_leaf_weight,min_gain=optimizer_min_gain)
        pred=predict_all(tree,batch.states); mu=weighted_mean(pred,w)
        if isfinite(mu)&&mu!=0 tree=shift_tree_leaves(tree,mu) end
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
    println("GREEDY NODE-LOCAL IF ACQUISITION")
    println("N=$N J/h=$(J/h) runs=$ntraining_runs checkpoints=$(sort(collect(checkpoint_epochs)))")
    println("pilot=$pilot_root acquisition/node=$acquisition_batch rounds=$max_rounds_per_node epsilon=$if_born_epsilon")
    println("baseline: one global Born sample M=$baseline_M")
    println("metric: oracle node recovery, full-tree recovery, evaluation cost")
    println("============================================================")
    X=enumerate_states(N); rows=NamedTuple[]
    for run in 1:ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,ntraining_runs)
        append!(rows,run_training(run,X))
    end
    outdir=joinpath(@__DIR__,"results");mkpath(outdir)
    path=joinpath(outdir,"greedy_node_local_acquisition.csv")
    write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/greedy_node_local_acquisition.csv")
end

export main,run_training,diagnose

end

if abspath(PROGRAM_FILE) == @__FILE__
    GreedyNodeLocalAcquisitionExperiment.main()
end
