using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module SplitMarginSNRExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

# ============================================================
# SPLIT-MARGIN / SPLIT-SNR DIAGNOSTIC
# ============================================================
# For every internal node of an exact depth-4 oracle tree, compute the two
# largest exact split gains G1 >= G2 and the normalized oracle margin
#
#       mG = (G1 - G2) / G1.
#
# Repeated samples from each proposal q estimate the full gain landscape.
# Across sampling repetitions we measure the uncertainty in the estimated
# difference between the ORACLE-best and ORACLE-second-best features:
#
#       Dhat = Ghat(f1) - Ghat(f2)
#       SNRsplit = (G1 - G2) / std(Dhat).
#
# We then ask whether split recovery and relative oracle gain collapse onto
# this SNR more cleanly than onto M, local ESS, or raw state coverage.
# ============================================================

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
const mixture_values = [(0.00,0.00),(0.50,0.00),(0.25,0.25),(0.50,0.25)]
const base_seed = 1_120_000
const diagnostic_seed_base = 9_120_000

function exact_frozen_problem(H, model, states)
    p = exact_probabilities(model, states)
    eloc = ComplexF64[local_energy!(H, model, @view(states[i,:])) for i in axes(states,1)]
    E = sum(p .* eloc)
    y = -real.(eloc .- E)
    yc = y .- sum(p .* y)
    sig = p .* yc.^2
    Z = sum(sig)
    qsig = Z > eps(Float64) ? sig ./ Z : copy(p)
    return (probabilities=p, target=y, energy=E,
            target_rms=sqrt(sum(p .* y.^2)),
            q_signal=qsig, q_uniform=fill(1/length(p), length(p)))
end

function proposal_distribution(f, a, b)
    q = (1-a-b).*f.probabilities .+ a.*f.q_signal .+ b.*f.q_uniform
    q ./= sum(q)
    q
end

function gain_landscape(X,y,w,idx; min_weight=0.0)
    gains = fill(-Inf, size(X,2))
    W = sum(w[i] for i in idx); W > 0 || return gains
    S = sum(w[i]*y[i] for i in idx); parent = S*S/W
    for f in axes(X,2)
        WL=0.0; SL=0.0
        @inbounds for i in idx
            if X[i,f] < 0
                WL += w[i]; SL += w[i]*y[i]
            end
        end
        WR=W-WL
        (WL < min_weight || WR < min_weight || WL <= 0 || WR <= 0) && continue
        SR=S-SL
        gains[f]=SL*SL/WL + SR*SR/WR - parent
    end
    gains
end

function top_two(g)
    valid=[i for i in eachindex(g) if isfinite(g[i]) && g[i] >= 0]
    isempty(valid) && return (0,0,0.0,0.0)
    sort!(valid, by=i->g[i], rev=true)
    f1=valid[1]; f2=length(valid)>=2 ? valid[2] : 0
    return f1,f2,g[f1],f2==0 ? 0.0 : g[f2]
end

function oracle_regions(tree,X)
    out=NamedTuple[]
    function walk(k::Int,d::Int,idx::Vector{Int},path::String)
        n=tree.nodes[k]; n.isleaf && return
        push!(out,(node_index=k,node_depth=d,node_path=path,state_indices=copy(idx)))
        L=Int[]; R=Int[]; f=Int(n.feature)
        for i in idx
            (X[i,f] < 0 ? push!(L,i) : push!(R,i))
        end
        walk(Int(n.left),d+1,L,path*"L"); walk(Int(n.right),d+1,R,path*"R")
    end
    walk(1,0,collect(axes(X,1)),""); out
end

function sampled_gains(X,y,p,q,draws,mask)
    localdraw=[i for i in draws if mask[i]]
    isempty(localdraw) && return fill(-Inf,size(X,2)),0,0,0.0
    u,c=compress_indices(localdraw)
    w=c .* p[u] ./ q[u]
    Xl=X[u,:]; yl=y[u]
    g=gain_landscape(Xl,yl,w,collect(eachindex(u));min_weight=sampled_min_leaf_weight)
    iw=[p[i]/q[i] for i in localdraw]
    ess=sum(iw)^2/sum(abs2,iw)
    return g,length(localdraw),length(u),ess
end

function diagnostic_seed(run,epoch,M,s,a,b)
    diagnostic_seed_base + 100_000*run + 1_000*epoch + M + 10*s +
    round(Int,10_000*a) + round(Int,100_000*b)
end

function diagnose(H,model,X,run,epoch)
    fr=exact_frozen_problem(H,model,X); p=fr.probabilities; dH=size(X,1)
    tree=GBTQuantum.grow_tree(X,fr.target,p;max_depth=oracle_depth,
                              min_weight=exact_min_leaf_weight,min_gain=0.0)
    nodes=oracle_regions(tree,X)
    oracle=NamedTuple[]
    for n in nodes
        og=gain_landscape(X,fr.target,p,n.state_indices;min_weight=exact_min_leaf_weight)
        f1,f2,G1,G2=top_two(og)
        margin=G1-G2; rel=G1>eps() ? margin/G1 : NaN
        push!(oracle,merge(n,(oracle_gains=og,f1=f1,f2=f2,G1=G1,G2=G2,
                              margin=margin,relative_margin=rel,
                              probability_mass=sum(p[n.state_indices]),
                              state_fraction=length(n.state_indices)/dH)))
    end
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e nodes=%d\n",epoch,real(fr.energy),fr.target_rms,length(oracle))
    rows=NamedTuple[]
    group_rows=NamedTuple[]
    for M in diagnostic_sample_sizes,(a,b) in mixture_values
        q=proposal_distribution(fr,a,b)
        # Store one row per node x repetition, then aggregate Dhat uncertainty.
        block=NamedTuple[]
        for s in 1:nsampling_runs
            draws=draw_categorical_indices(MersenneTwister(diagnostic_seed(run,epoch,M,s,a,b)),q,M)
            for n in oracle
                mask=falses(dH); mask[n.state_indices].=true
                sg,nlocal,nuniq,ess=sampled_gains(X,fr.target,p,q,draws,mask)
                sf,_,_,_=top_two(sg)
                chosen = sf==0 ? 0.0 : (isfinite(n.oracle_gains[sf]) ? max(n.oracle_gains[sf],0.0) : 0.0)
                RG=n.G1>eps() ? chosen/n.G1 : NaN
                d̂=(n.f1!=0 && n.f2!=0 && isfinite(sg[n.f1]) && isfinite(sg[n.f2])) ? sg[n.f1]-sg[n.f2] : NaN
                push!(block,(training_run=run,epoch=epoch,M=M,alpha=a,beta=b,
                    sampling_run=s,node_index=n.node_index,node_depth=n.node_depth,node_path=n.node_path,
                    oracle_best_feature=n.f1,oracle_second_feature=n.f2,G1=n.G1,G2=n.G2,
                    oracle_margin=n.margin,relative_margin=n.relative_margin,
                    sampled_feature=sf,exact_split_match=(sf!=0 && sf==n.f1),relative_oracle_gain=RG,
                    sampled_oracle_pair_difference=d̂,local_draws=nlocal,local_unique_states=nuniq,
                    local_ess=ess,conditional_ess_fraction=nlocal>0 ? ess/nlocal : 0.0,
                    oracle_probability_mass=n.probability_mass,oracle_state_fraction=n.state_fraction))
            end
        end
        append!(rows,block)
        for n in oracle
            S=[r for r in block if r.node_index==n.node_index]
            D=[r.sampled_oracle_pair_difference for r in S if isfinite(r.sampled_oracle_pair_difference)]
            sigma=length(D)>=2 ? std(D) : NaN
            snr=isfinite(sigma) && sigma>eps() ? n.margin/sigma : (n.margin>0 && sigma==0 ? Inf : NaN)
            push!(group_rows,(training_run=run,epoch=epoch,M=M,alpha=a,beta=b,
                node_index=n.node_index,node_depth=n.node_depth,node_path=n.node_path,
                G1=n.G1,G2=n.G2,oracle_margin=n.margin,relative_margin=n.relative_margin,
                split_snr=snr,mean_Dhat=finite_mean(D),sigma_Dhat=sigma,
                split_match_rate=mean(Float64(r.exact_split_match) for r in S),
                mean_relative_oracle_gain=finite_mean(r.relative_oracle_gain for r in S),
                mean_local_draws=mean(r.local_draws for r in S),
                mean_local_unique_states=mean(r.local_unique_states for r in S),
                mean_local_ess=mean(r.local_ess for r in S)))
        end
        G=[r for r in group_rows if r.epoch==epoch && r.M==M && r.alpha==a && r.beta==b]
        @printf("  M=%4d a=%.2f b=%.2f  <SNR>=%.3f  match=%.3f  <RG>=%.3f\n",M,a,b,
                finite_mean(r.split_snr for r in G),mean(r.split_match_rate for r in G),
                finite_mean(r.mean_relative_oracle_gain for r in G))
    end
    return rows,group_rows
end

function run_training(run,X)
    H=TFIMHamiltonian(N;J=J,h=h,periodic=true)
    rng=MersenneTwister(base_seed+10_000*run)
    samples=Matrix{Int8}(undef,training_nsamples,N)
    @inbounds for i in eachindex(samples); samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
    model=LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); logamps=zeros(training_nsamples)
    for _ in 1:burn_in_sweeps; GBTQuantum.sweep!(rng,model,samples,logamps); end
    raw=NamedTuple[]; groups=NamedTuple[]
    for epoch in 1:nepochs
        batch=vmc_batch(H,model,samples); yA,_=make_targets(batch); w=batch.counts
        if epoch in checkpoint_epochs
            r,g=diagnose(H,model,X,run,epoch); append!(raw,r); append!(groups,g)
        end
        tree=GBTQuantum.grow_tree(batch.states,yA,w;max_depth=optimizer_max_depth,
                                  min_weight=optimizer_min_leaf_weight,min_gain=optimizer_min_gain)
        pred=predict_all(tree,batch.states); mu=weighted_mean(pred,w)
        if isfinite(mu) && mu!=0; tree=shift_tree_leaves(tree,mu); end
        push!(model.logamp.trees,scale_tree(tree,eta)); GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:sweeps_per_epoch; GBTQuantum.sweep!(rng,model,samples,logamps); end
    end
    raw,groups
end

function main()
    println("\n============================================================")
    println("SPLIT-MARGIN / SPLIT-SNR EXPERIMENT")
    println("N=$N  J/h=$(J/h)  runs=$ntraining_runs  checkpoints=$(sort(collect(checkpoint_epochs)))")
    println("M=$diagnostic_sample_sizes  mixtures=$mixture_values  repetitions=$nsampling_runs")
    println("============================================================")
    X=enumerate_states(N); raw=NamedTuple[]; groups=NamedTuple[]
    for run in 1:ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,ntraining_runs)
        r,g=run_training(run,X); append!(raw,r); append!(groups,g)
    end
    out=joinpath(@__DIR__,"results"); mkpath(out)
    rawpath=joinpath(out,"split_margin_snr_raw.csv")
    grouppath=joinpath(out,"split_margin_snr_nodes.csv")
    write_namedtuple_csv(rawpath,raw); write_namedtuple_csv(grouppath,groups)
    println("\nRESULTS WRITTEN TO\n",rawpath,"\n",grouppath)
    return (raw=raw,nodes=groups)
end

end

if abspath(PROGRAM_FILE)==@__FILE__
    SplitMarginSNRExperiment.main()
end
