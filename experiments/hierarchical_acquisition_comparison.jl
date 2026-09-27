using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module HierarchicalAcquisitionComparisonExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

# Tests the derived hierarchical acquisition against the strongest baselines:
#   p(x)
#   p(x)|y(x)|^2
#   influence acquisition with alpha=3
#   hierarchical acquisition q ∝ p sqrt(sum_v W_v R_v Phi_v^2)
#
# The hierarchical proposal is solved self-consistently because sigma_v[q]
# determines rho_v and R_v, while rho determines the downstream value W_v.

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
const influence_alpha = 3.0
const base_seed = 1_730_000
const diagnostic_seed_base = 51_730_000
const support_floor = 1e-12
const fixed_point_tol = 1e-8
const fixed_point_maxiter = 500
const fixed_point_damping = 0.5

normal_pdf(z) = exp(-0.5*z*z) / sqrt(2*pi)
normal_cdf(z) = 0.5 * erfc(-z/sqrt(2.0))

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
    (probabilities=p,target=y,energy=E,target_rms=sqrt(sum(p.*y.^2)))
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
    function walk(k,d,idx,path,parent_path)
        n=tree.nodes[k]; n.isleaf && return
        push!(out,(node_index=k,node_depth=d,node_path=path,parent_path=parent_path,state_indices=copy(idx)))
        L=Int[]; R=Int[]; f=Int(n.feature)
        for i in idx; X[i,f]<0 ? push!(L,i) : push!(R,i); end
        walk(Int(n.left),d+1,L,path*"L",path)
        walk(Int(n.right),d+1,R,path*"R",path)
    end
    walk(1,0,collect(axes(X,1)),"",nothing)
    out
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
        push!(nodes,merge(n,(probability_mass=P,f1=f1,f2=f2,G1=G1,G2=G2,margin=max(G1-G2,0.0),Phi=phiD./P)))
    end
    tree,pred,Rp,nodes
end

function regularize_proposal(q,p)
    q=max.(q,support_floor.*p); s=sum(q)
    s>0 ? q./s : copy(p)
end

function proposal_signal(fr)
    q=fr.probabilities.*abs2.(fr.target)
    sum(q)>eps() ? regularize_proposal(q,fr.probabilities) : copy(fr.probabilities)
end

function proposal_influence(fr,nodes)
    p=fr.probabilities; A2=zeros(Float64,length(p))
    for n in nodes
        importance=n.probability_mass^influence_alpha*max(n.G1,0.0)
        @. A2 += importance*n.Phi^2
    end
    q=p.*sqrt.(A2)
    sum(q)>eps() ? regularize_proposal(q,p) : copy(p)
end

function node_sigmas(fr, nodes, q, M)
    p = fr.probabilities

    return [
        sqrt(
            max(
                sum((p .^ 2) .* (n.Phi .^ 2) ./ q) / M,
                0.0
            )
        )
        for n in nodes
    ]
end

function recovery_probabilities(nodes,sigma)
    rho=zeros(Float64,length(nodes))
    for i in eachindex(nodes)
        D=nodes[i].margin; s=sigma[i]
        rho[i] = s<=eps() ? (D>0 ? 1.0 : 0.5) : normal_cdf(D/s)
    end
    rho
end

function hierarchical_values(nodes,rho)
    # V_v = G1_v.  C_v = V_v + rho_L C_L + rho_R C_R.
    # W_v = S_v^- C_v, where S_v^- is correct ancestor survival.
    path_to_i=Dict(n.node_path=>i for (i,n) in enumerate(nodes))
    C=zeros(Float64,length(nodes))
    order=sortperm(eachindex(nodes),by=i->nodes[i].node_depth,rev=true)
    for i in order
        n=nodes[i]; c=max(n.G1,0.0)
        for cp in (n.node_path*"L",n.node_path*"R")
            if haskey(path_to_i,cp)
                j=path_to_i[cp]; c += rho[j]*C[j]
            end
        end
        C[i]=c
    end
    W=zeros(Float64,length(nodes))
    for (i,n) in enumerate(nodes)
        survival=1.0; path=""
        for ch in n.node_path
            j=path_to_i[path]; survival*=rho[j]; path*=string(ch)
        end
        W[i]=survival*C[i]
    end
    W,C
end

function recoverability_sensitivity(nodes,sigma)
    R=zeros(Float64,length(nodes))
    for i in eachindex(nodes)
        D=nodes[i].margin; s=sigma[i]
        if D>0 && s>eps()
            z=D/s
            R[i]=D/(2*s^3)*normal_pdf(z)
        end
    end
    R
end

function proposal_hierarchical(fr,nodes,M)
    p=fr.probabilities; q=copy(p)
    converged=false; delta=Inf; iterations=0
    sigma=zeros(length(nodes)); rho=zeros(length(nodes)); W=zeros(length(nodes)); C=zeros(length(nodes)); R=zeros(length(nodes))
    for it in 1:fixed_point_maxiter
        iterations=it
        sigma=node_sigmas(fr,nodes,q,M)
        rho=recovery_probabilities(nodes,sigma)
        W,C=hierarchical_values(nodes,rho)
        R=recoverability_sensitivity(nodes,sigma)
        A2=zeros(Float64,length(p))
        for i in eachindex(nodes)
            @. A2 += (W[i]*R[i])*nodes[i].Phi^2
        end
        raw=p.*sqrt.(A2)
        qnew=sum(raw)>eps() ? regularize_proposal(raw,p) : copy(p)
        qnext=(1-fixed_point_damping).*q .+ fixed_point_damping.*qnew
        qnext ./= sum(qnext)
        delta=sum(abs.(qnext.-q))
        q=qnext
        if delta<fixed_point_tol
            converged=true; break
        end
    end
    # Recompute diagnostics at the returned q.
    sigma=node_sigmas(fr,nodes,q,M)
    rho=recovery_probabilities(nodes,sigma)
    W,C=hierarchical_values(nodes,rho)
    R=recoverability_sensitivity(nodes,sigma)
    return q,(iterations=iterations,converged=converged,delta=delta,sigma=sigma,rho=rho,W=W,C=C,R=R)
end

function fit_sampled_tree(X,fr,draws,q)
    u,c=compress_indices(draws)
    w=c.*fr.probabilities[u]./q[u]
    tree=GBTQuantum.grow_tree(X[u,:],fr.target[u],w;max_depth=diagnostic_depth,min_weight=sampled_min_leaf_weight,min_gain=0.0)
    sp=predict_all(tree,X[u,:]); mu=weighted_mean(sp,w)
    isfinite(mu) && mu!=0 && (tree=shift_tree_leaves(tree,mu))
    tree,predict_all(tree,X)
end

function sampled_nodes_by_path(tree)
    d=Dict{String,Any}()
    function walk(k,path)
        n=tree.nodes[k]; n.isleaf && return
        d[path]=n; walk(Int(n.left),path*"L"); walk(Int(n.right),path*"R")
    end
    walk(1,""); d
end

function split_match(tree,nodes)
    d=sampled_nodes_by_path(tree)
    mean(haskey(d,n.node_path) && Int(d[n.node_path].feature)==n.f1 for n in nodes)
end

function path_survival(tree,nodes)
    d=sampled_nodes_by_path(tree); path_to_node=Dict(n.node_path=>n for n in nodes)
    vals=Bool[]
    for n in nodes
        ok=true; path=""
        for ch in n.node_path
            if !haskey(d,path) || Int(d[path].feature)!=path_to_node[path].f1
                ok=false; break
            end
            path*=string(ch)
        end
        push!(vals,ok)
    end
    mean(vals)
end

function diagnostic_seed(run,epoch,M,method_id,rep)
    diagnostic_seed_base+1_000_000*run+10_000*epoch+10_000*method_id+M+rep
end

function diagnose(H,model,X,run,epoch)
    fr=exact_frozen_problem(H,model,X)
    _,_,oracle_Rp,nodes=build_oracle(fr,X)
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e oracle R2p=%.4f nodes=%d\n",epoch,real(fr.energy),fr.target_rms,oracle_Rp,length(nodes))
    rows=NamedTuple[]; node_rows=NamedTuple[]
    for M in diagnostic_sample_sizes
        qh,hd=proposal_hierarchical(fr,nodes,M)
        @printf("  hierarchical fixed point M=%4d: iter=%d converged=%s delta=%.3e\n",M,hd.iterations,string(hd.converged),hd.delta)
        for i in eachindex(nodes)
            n=nodes[i]
            push!(node_rows,(training_run=run,epoch=epoch,M=M,node_path=n.node_path,node_depth=n.node_depth,
                probability_mass=n.probability_mass,G1=n.G1,G2=n.G2,margin=n.margin,sigma=hd.sigma[i],rho=hd.rho[i],
                downstream_value=hd.C[i],hierarchical_value=hd.W[i],recoverability=hd.R[i],derived_weight=hd.W[i]*hd.R[i],
                fixed_point_iterations=hd.iterations,fixed_point_converged=hd.converged,fixed_point_delta=hd.delta))
        end
        methods=[("p",copy(fr.probabilities)),("signal",proposal_signal(fr)),("influence_a3",proposal_influence(fr,nodes)),("hierarchical",qh)]
        for (mid,(method,q)) in enumerate(methods)
            localrows=NamedTuple[]
            for rep in 1:nsampling_runs
                rng=MersenneTwister(diagnostic_seed(run,epoch,M,mid,rep))
                draws=draw_categorical_indices(rng,q,M)
                tree,pred=fit_sampled_tree(X,fr,draws,q)
                Rp=weighted_r2(fr.target,pred,fr.probabilities)
                sm=split_match(tree,nodes); ps=path_survival(tree,nodes)
                u,_=compress_indices(draws); iw=fr.probabilities[draws]./q[draws]
                ess=sum(iw)^2/sum(abs2,iw)
                r=(training_run=run,epoch=epoch,M=M,sampling_run=rep,method=method,
                    frozen_energy=real(fr.energy),frozen_target_rms=fr.target_rms,oracle_Rp=oracle_Rp,
                    sampled_Rp=Rp,Rp_penalty=oracle_Rp-Rp,split_match=sm,path_survival=ps,
                    unique_states=length(u),probability_coverage=sum(fr.probabilities[u]),importance_ess_fraction=ess/M)
                push!(rows,r); push!(localrows,r)
            end
            @printf("  %-12s M=%4d <R2p>=%.3f penalty=%.3f split=%.3f path=%.3f ESS/M=%.3f\n",
                method,M,mean(r.sampled_Rp for r in localrows),mean(r.Rp_penalty for r in localrows),
                mean(r.split_match for r in localrows),mean(r.path_survival for r in localrows),mean(r.importance_ess_fraction for r in localrows))
        end
    end
    rows,node_rows
end

function run_training(run,X)
    H=TFIMHamiltonian(N;J=J,h=h,periodic=true)
    rng=MersenneTwister(base_seed+10_000*run)
    samples=Matrix{Int8}(undef,training_nsamples,N)
    @inbounds for i in eachindex(samples); samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
    model=LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); logamps=zeros(training_nsamples)
    for _ in 1:burn_in_sweeps; GBTQuantum.sweep!(rng,model,samples,logamps); end
    rows=NamedTuple[]; node_rows=NamedTuple[]
    for epoch in 1:nepochs
        batch=vmc_batch(H,model,samples); yA,_=make_targets(batch); w=batch.counts
        if epoch in checkpoint_epochs
            r,nr=diagnose(H,model,X,run,epoch); append!(rows,r); append!(node_rows,nr)
        end
        tree=GBTQuantum.grow_tree(batch.states,yA,w;max_depth=optimizer_max_depth,min_weight=optimizer_min_leaf_weight,min_gain=optimizer_min_gain)
        pred=predict_all(tree,batch.states); mu=weighted_mean(pred,w)
        isfinite(mu) && mu!=0 && (tree=shift_tree_leaves(tree,mu))
        push!(model.logamp.trees,scale_tree(tree,eta)); GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:sweeps_per_epoch; GBTQuantum.sweep!(rng,model,samples,logamps); end
    end
    rows,node_rows
end

function main()
    println("\n============================================================")
    println("HIERARCHICAL ACQUISITION COMPARISON")
    println("N=$N J/h=$(J/h) runs=$ntraining_runs checkpoints=$(sort(collect(checkpoint_epochs)))")
    println("M=$diagnostic_sample_sizes repetitions=$nsampling_runs")
    println("Methods: p, p|y|^2, influence alpha=3, derived hierarchical")
    println("Hierarchical intrinsic value V_v = oracle G1_v")
    println("============================================================")
    X=enumerate_states(N); rows=NamedTuple[]; node_rows=NamedTuple[]
    for run in 1:ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,ntraining_runs)
        r,nr=run_training(run,X); append!(rows,r); append!(node_rows,nr)
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"hierarchical_acquisition_comparison.csv")
    nodepath=joinpath(outdir,"hierarchical_acquisition_nodes.csv")
    write_namedtuple_csv(path,rows); write_namedtuple_csv(nodepath,node_rows)
    println("\nResults written to ",path)
    println("Node diagnostics written to ",nodepath)
    rows,node_rows
end

export main,run_training,diagnose,proposal_hierarchical

end

if abspath(PROGRAM_FILE)==@__FILE__
    HierarchicalAcquisitionComparisonExperiment.main()
end
