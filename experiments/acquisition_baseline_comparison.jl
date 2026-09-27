using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module AcquisitionBaselineComparisonExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

# Head-to-head comparison on the SAME frozen problems:
#   p       : q(x) = p(x)
#   signal  : q(x) ∝ p(x) |y(x)|^2
#   influence(alpha): q(x) ∝ p(x) sqrt(sum_v P_v^alpha G_v Phi_v(x)^2)
#
# All sampled trees use importance weights p/q, so all methods target the
# same p-weighted regression problem.  We compare actual reconstruction
# outcomes, not an analytic prediction of them.

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
const diagnostic_depth = 4
const exact_min_leaf_weight = 1e-14
const sampled_min_leaf_weight = 1e-14
const diagnostic_sample_sizes = [256, 1024]
const nsampling_runs = 100
const mass_exponents = [1.0, 2.0, 3.0]
const base_seed = 1_630_000
const diagnostic_seed_base = 41_630_000
const support_floor = 1e-12

function csv_escape(x)
    s=string(x)
    if occursin(',',s) || occursin('"',s) || occursin('\n',s) || occursin('\r',s)
        return "\""*replace(s,"\""=>"\"\"")*"\""
    end
    s
end

function write_namedtuple_csv(path,rows)
    open(path,"w") do io
        isempty(rows) && return
        cols=propertynames(first(rows)); println(io,join(string.(cols),","))
        for r in rows
            println(io,join((csv_escape(getproperty(r,c)) for c in cols),","))
        end
    end
end

function exact_frozen_problem(H,model,X)
    p=exact_probabilities(model,X)
    eloc=ComplexF64[local_energy!(H,model,@view(X[i,:])) for i in axes(X,1)]
    E=sum(p.*eloc)
    y=-real.(eloc.-E)
    return (probabilities=p,target=y,energy=E,target_rms=sqrt(sum(p.*y.^2)))
end

function gain_landscape(X,y,w,idx;min_weight=0.0)
    g=fill(-Inf,size(X,2)); W=sum(w[i] for i in idx)
    W>0 || return g
    S=sum(w[i]*y[i] for i in idx); parent=S*S/W
    for f in axes(X,2)
        WL=0.0; SL=0.0
        @inbounds for i in idx
            if X[i,f]<0; WL+=w[i]; SL+=w[i]*y[i]; end
        end
        WR=W-WL
        (WL<min_weight || WR<min_weight || WL<=0 || WR<=0) && continue
        SR=S-SL
        g[f]=SL*SL/WL+SR*SR/WR-parent
    end
    g
end

function top_two(g)
    v=[i for i in eachindex(g) if isfinite(g[i]) && g[i]>=0]
    isempty(v) && return (0,0,0.0,0.0)
    sort!(v,by=i->g[i],rev=true)
    f1=v[1]; f2=length(v)>=2 ? v[2] : 0
    (f1,f2,g[f1],f2==0 ? 0.0 : g[f2])
end

function oracle_regions(tree,X)
    out=NamedTuple[]
    function walk(k,d,idx,path,ancestors)
        n=tree.nodes[k]; n.isleaf && return
        push!(out,(node_index=k,node_depth=d,node_path=path,state_indices=copy(idx),ancestor_indices=copy(ancestors)))
        L=Int[]; R=Int[]; f=Int(n.feature)
        for i in idx; X[i,f]<0 ? push!(L,i) : push!(R,i); end
        anc=[ancestors;k]
        walk(Int(n.left),d+1,L,path*"L",anc); walk(Int(n.right),d+1,R,path*"R",anc)
    end
    walk(1,0,collect(axes(X,1)),"",Int[]); out
end

function conditional_oracle_gains(X,y,p,idx)
    P=sum(p[idx]); P>0 || return fill(-Inf,size(X,2)),P
    gain_landscape(X,y,p./P,idx;min_weight=exact_min_leaf_weight),P
end

function gain_influence(X,y,p,idx,f)
    P=sum(p[idx]); phi=zeros(Float64,size(X,1)); (P>0 && f!=0) || return phi
    r=0.0; a=0.0; m=0.0
    @inbounds for i in idx
        pv=p[i]/P; L=X[i,f]<0 ? 1.0 : 0.0
        r+=pv*L; a+=pv*y[i]*L; m+=pv*y[i]
    end
    (r>eps() && 1-r>eps()) || return phi
    gr=-a^2/r^2+(m-a)^2/(1-r)^2
    ga=2a/r-2(m-a)/(1-r)
    gm=2(m-a)/(1-r)-2m
    @inbounds for i in idx
        L=X[i,f]<0 ? 1.0 : 0.0
        phi[i]=gr*(L-r)+ga*(y[i]*L-a)+gm*(y[i]-m)
    end
    phi
end

function build_oracle(fr,X)
    p=fr.probabilities
    tree=GBTQuantum.grow_tree(X,fr.target,p;max_depth=diagnostic_depth,min_weight=exact_min_leaf_weight,min_gain=0.0)
    pred=predict_all(tree,X); mu=weighted_mean(pred,p)
    isfinite(mu) && mu!=0 && (tree=shift_tree_leaves(tree,mu); pred=predict_all(tree,X))
    Rp=weighted_r2(fr.target,pred,p)
    nodes=NamedTuple[]
    for n in oracle_regions(tree,X)
        g,P=conditional_oracle_gains(X,fr.target,p,n.state_indices)
        f1,f2,G1,G2=top_two(g)
        phiD=gain_influence(X,fr.target,p,n.state_indices,f1).-gain_influence(X,fr.target,p,n.state_indices,f2)
        push!(nodes,merge(n,(probability_mass=P,f1=f1,f2=f2,G1=G1,G2=G2,margin=G1-G2,Phi=phiD./P)))
    end
    return tree,pred,Rp,nodes
end

function regularize_proposal(q,p)
    q=max.(q,support_floor.*p); s=sum(q)
    s>0 ? q./s : copy(p)
end

proposal_p(fr,nodes,alpha)=copy(fr.probabilities)

function proposal_signal(fr,nodes,alpha)
    q=fr.probabilities.*abs2.(fr.target)
    sum(q)>eps() ? regularize_proposal(q,fr.probabilities) : copy(fr.probabilities)
end

function proposal_influence(fr,nodes,alpha)
    p=fr.probabilities; A2=zeros(Float64,length(p))
    for n in nodes
        importance=n.probability_mass^alpha*max(n.G1,0.0)
        @. A2 += importance*n.Phi^2
    end
    q=p.*sqrt.(A2)
    sum(q)>eps() ? regularize_proposal(q,p) : copy(p)
end

function fit_sampled_tree(X,fr,draws,q)
    u,c=compress_indices(draws)
    w=c.*fr.probabilities[u]./q[u]
    tree=GBTQuantum.grow_tree(X[u,:],fr.target[u],w;max_depth=diagnostic_depth,min_weight=sampled_min_leaf_weight,min_gain=0.0)
    sp=predict_all(tree,X[u,:]); mu=weighted_mean(sp,w)
    isfinite(mu) && mu!=0 && (tree=shift_tree_leaves(tree,mu))
    pred=predict_all(tree,X)
    return tree,pred
end

function split_metrics(sampled_tree,oracle_nodes,X)
    # Evaluate sampled best split inside each fixed oracle region. This makes
    # node-wise split recovery comparable even if the sampled tree diverges.
    matches=Bool[]; relg=Float64[]
    # Tree fitting is evaluated separately; here recovery is reconstructed
    # from the sampled tree's actual node features only along surviving paths.
    snodes=Dict{String,Any}()
    function walk(k,path)
        n=sampled_tree.nodes[k]; n.isleaf && return
        snodes[path]=n
        walk(Int(n.left),path*"L"); walk(Int(n.right),path*"R")
    end
    walk(1,"")
    for n in oracle_nodes
        if haskey(snodes,n.node_path)
            s=snodes[n.node_path]; f=Int(s.feature)
            push!(matches,f==n.f1)
            # Relative gain is only meaningful through the oracle gain landscape;
            # use 1 for exact feature recovery and leave detailed gain to tree R2.
            push!(relg,f==n.f1 ? 1.0 : 0.0)
        else
            push!(matches,false); push!(relg,0.0)
        end
    end
    return mean(matches),mean(relg)
end

function path_survival(sampled_tree,oracle_nodes)
    snodes=Dict{String,Any}()
    function walk(k,path)
        n=sampled_tree.nodes[k]; n.isleaf && return
        snodes[path]=n; walk(Int(n.left),path*"L"); walk(Int(n.right),path*"R")
    end
    walk(1,"")
    surv=Bool[]
    for n in oracle_nodes
        ok=true; path=""
        for c in n.node_path
            if !haskey(snodes,path); ok=false; break; end
            on=first(x for x in oracle_nodes if x.node_path==path)
            if Int(snodes[path].feature)!=on.f1; ok=false; break; end
            path*=string(c)
        end
        push!(surv,ok)
    end
    mean(surv)
end

function diagnostic_seed(run,epoch,M,method_id,rep)
    diagnostic_seed_base+1_000_000*run+10_000*epoch+10_000*method_id+M+rep
end

function diagnose(H,model,X,run,epoch)
    fr=exact_frozen_problem(H,model,X)
    _,_,oracle_Rp,nodes=build_oracle(fr,X)
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e oracle R2p=%.4f nodes=%d\n",epoch,real(fr.energy),fr.target_rms,oracle_Rp,length(nodes))
    rows=NamedTuple[]
    methods=[("p",0.0,proposal_p), ("signal",0.0,proposal_signal)]
    for alpha in mass_exponents; push!(methods,("influence",alpha,proposal_influence)); end
    for (mid,(method,alpha,pfun)) in enumerate(methods), M in diagnostic_sample_sizes
        q=pfun(fr,nodes,alpha)
        for rep in 1:nsampling_runs
            rng=MersenneTwister(diagnostic_seed(run,epoch,M,mid,rep))
            draws=draw_categorical_indices(rng,q,M)
            tree,pred=fit_sampled_tree(X,fr,draws,q)
            Rp=weighted_r2(fr.target,pred,fr.probabilities)
            sm,_=split_metrics(tree,nodes,X)
            ps=path_survival(tree,nodes)
            u,_=compress_indices(draws)
            iw=fr.probabilities[draws]./q[draws]
            ess=sum(iw)^2/sum(abs2,iw)
            push!(rows,(training_run=run,epoch=epoch,M=M,sampling_run=rep,method=method,alpha=alpha,
                frozen_energy=real(fr.energy),frozen_target_rms=fr.target_rms,oracle_Rp=oracle_Rp,
                sampled_Rp=Rp,Rp_penalty=oracle_Rp-Rp,split_match=sm,path_survival=ps,
                unique_states=length(u),probability_coverage=sum(fr.probabilities[u]),
                importance_ess_fraction=ess/M))
        end
        S=[r for r in rows if r.M==M && r.method==method && r.alpha==alpha]
        @printf("  %-9s a=%3.1f M=%4d  <R2p>=%.3f penalty=%.3f split=%.3f path=%.3f ESS/M=%.3f\n",
            method,alpha,M,mean(r.sampled_Rp for r in S),mean(r.Rp_penalty for r in S),
            mean(r.split_match for r in S),mean(r.path_survival for r in S),mean(r.importance_ess_fraction for r in S))
    end
    rows
end

function run_training(run,X)
    H=TFIMHamiltonian(N;J=J,h=h,periodic=true)
    rng=MersenneTwister(base_seed+10_000*run)
    samples=Matrix{Int8}(undef,training_nsamples,N)
    @inbounds for i in eachindex(samples); samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
    model=LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    logamps=zeros(training_nsamples)
    for _ in 1:burn_in_sweeps; GBTQuantum.sweep!(rng,model,samples,logamps); end
    rows=NamedTuple[]
    for epoch in 1:nepochs
        batch=vmc_batch(H,model,samples); yA,_=make_targets(batch); w=batch.counts
        epoch in checkpoint_epochs && append!(rows,diagnose(H,model,X,run,epoch))
        tree=GBTQuantum.grow_tree(batch.states,yA,w;max_depth=optimizer_max_depth,min_weight=optimizer_min_leaf_weight,min_gain=optimizer_min_gain)
        pred=predict_all(tree,batch.states); mu=weighted_mean(pred,w)
        isfinite(mu) && mu!=0 && (tree=shift_tree_leaves(tree,mu))
        push!(model.logamp.trees,scale_tree(tree,eta)); GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:sweeps_per_epoch; GBTQuantum.sweep!(rng,model,samples,logamps); end
    end
    rows
end

function main()
    println("\n============================================================")
    println("ACQUISITION BASELINE COMPARISON")
    println("N=$N J/h=$(J/h) runs=$ntraining_runs checkpoints=$(sort(collect(checkpoint_epochs)))")
    println("M=$diagnostic_sample_sizes repetitions=$nsampling_runs influence alpha=$mass_exponents")
    println("Methods: p, p|y|^2, tree-influence acquisition")
    println("============================================================")
    X=enumerate_states(N); rows=NamedTuple[]
    for run in 1:ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,ntraining_runs)
        append!(rows,run_training(run,X))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"acquisition_baseline_comparison.csv")
    write_namedtuple_csv(path,rows)
    println("\nResults written to ",path)
    rows
end

export main,run_training,diagnose

end

if abspath(PROGRAM_FILE)==@__FILE__
    AcquisitionBaselineComparisonExperiment.main()
end
