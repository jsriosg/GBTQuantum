module TFIMEdgeImportanceMap

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Random
using Statistics
using Printf

# Exact N=8 diagnostic for the proposed Hamiltonian-edge sampling distribution.
# Reproduces the actual adaptive-eta V2 trajectory, freezes selected PRE-UPDATE
# checkpoints, and enumerates every undirected single-spin-flip TFIM edge.
#
# Gauge convention carried throughout: the fitted direction is centered so that
# <f>_p = 0. For exact checkpoint analysis we re-center f with the exact Born
# distribution (a constant shift is a physically irrelevant log-amplitude gauge).
# Thus g = 2<f E_loc> and
# c = N''/Z - 4 E <f^2>.

const N = 8
const NSAMPLES = 512
const EPOCHS = 150
const MAX_DEPTH = 4
const ETA_FIXED = 0.05
const ETA_CAP = 0.40
const CURVATURE_FLOOR = 1e-12
const SEED = 1234
const CHECKPOINTS = Dict(1.0 => Set([8,50]), 2.0 => Set([4,10]))
const MASS_THRESHOLDS = [0.50,0.90,0.95,0.99,0.999]

function center_tree(tree,X,weights)
    preds=[GBTQuantum.predict(tree,@view X[j,:]) for j in axes(X,1)]
    mu=GBTQuantum.weighted_mean(preds,weights)
    mu==0.0 && return tree
    nodes=copy(tree.nodes)
    @inbounds for i in eachindex(nodes)
        n=nodes[i]
        n.isleaf && (nodes[i]=GBTQuantum.Node(n.feature,n.value-mu,n.left,n.right,true))
    end
    GBTQuantum.RegressionTree(nodes)
end

scaled_tree(tree,eta)=GBTQuantum.RegressionTree([
    n.isleaf ? GBTQuantum.Node(n.feature,eta*n.value,n.left,n.right,true) : n for n in tree.nodes])

function all_states(N)
    d=1<<N
    X=Matrix{Int8}(undef,d,N)
    @inbounds for s in 0:d-1, i in 1:N
        X[s+1,i]=((s>>(i-1))&1)==1 ? Int8(1) : Int8(-1)
    end
    X
end
@inline flipped_row(j,i)=((j-1) ⊻ (1<<(i-1)))+1

function f_local_energy(H,model,tree,x)
    A0=GBTQuantum.logamplitude(model,x); f0=GBTQuantum.predict(tree,x)
    z=GBTQuantum.diagonal(H,x)*f0
    @inbounds for i in 1:H.N
        x[i]=-x[i]
        z -= H.h*exp(GBTQuantum.logamplitude(model,x)-A0)*GBTQuantum.predict(tree,x)
        x[i]=-x[i]
    end
    z
end

function mc_decision(H,model,batch,tree)
    f=[GBTQuantum.predict(tree,@view batch.states[j,:]) for j in axes(batch.states,1)]
    w=Float64.(batch.counts); el=real.(batch.local_energy); W=sum(w)
    Ef=sum(w.*f)/W; Ee=sum(w.*el)/W; Ef2=sum(w.*f.*f)/W
    Ef2el=sum(w.*f.*f.*el)/W
    g=2sum(w.*(f.-Ef).*(el.-Ee))/W
    q=0.0
    @inbounds for j in axes(batch.states,1)
        q += w[j]*f[j]*f_local_energy(H,model,tree,@view batch.states[j,:])
    end
    c=2Ef2el+2q/W-4Ee*Ef2-4Ef*g
    if !isfinite(g)||!isfinite(c)||g>=0||c<=CURVATURE_FLOOR
        return ETA_FIXED,g,c,:fallback
    end
    raw=-g/c
    (!isfinite(raw)||raw<=0) && return (ETA_FIXED,g,c,:fallback)
    return clamp(raw,0.0,ETA_CAP),g,c,:newton
end

function ranks(v)
    ord=sortperm(v); r=zeros(Float64,length(v)); i=1
    while i<=length(v)
        j=i
        while j<length(v) && v[ord[j+1]]==v[ord[i]]; j+=1; end
        rr=(i+j)/2
        for k in i:j; r[ord[k]]=rr; end
        i=j+1
    end
    r
end
spearman(a,b)=(std(a)==0||std(b)==0) ? NaN : cor(ranks(a),ranks(b))
state_bits(states,j)=join(states[j,i]==1 ? "1" : "0" for i in 1:size(states,2))

# Exact checkpoint arrays with exact gauge centering <f>_p=0.
function exact_checkpoint_data(H,model,tree,states)
    d=size(states,1); A=zeros(d); f=zeros(d)
    @inbounds for j in 1:d
        x=@view states[j,:]
        A[j]=GBTQuantum.logamplitude(model,x)
        f[j]=GBTQuantum.predict(tree,x)
    end
    m=maximum(2 .* A); w=exp.(2 .* A .- m); p=w/sum(w)
    f .-= sum(p.*f) # exact gauge projection: <f>_p = 0

    el=zeros(d)
    @inbounds for j in 1:d
        x=@view states[j,:]; e=GBTQuantum.diagonal(H,x)
        for i in 1:H.N
            k=flipped_row(j,i)
            e -= H.h*exp(A[k]-A[j])
        end
        el[j]=e
    end
    E=sum(p.*el); f2=sum(p.*f.*f)
    g=2sum(p.*f.*el) # <f>=0
    return A,p,f,el,E,f2,g
end

# Enumerate each undirected TFIM off-diagonal edge once (j<k).
# edge_base = |H_jk| sqrt(p_j p_k), the proposed q_edge unnormalized weight.
# The exact doubled off-diagonal contributions are:
#   G_edge = 2 H_jk sqrt(p_j p_k) (f_j+f_k)
#   C_edge = 2 H_jk sqrt(p_j p_k) (f_j+f_k)^2
function edge_table(H,A,p,f,states,ratio,epoch)
    rows=NamedTuple[]
    d=length(p)
    @inbounds for j in 1:d, i in 1:H.N
        k=flipped_row(j,i)
        j<k || continue
        hij=-H.h
        geometric=sqrt(p[j]*p[k])
        base=abs(hij)*geometric
        fs=f[j]+f[k]
        gedge=2hij*geometric*fs
        cedge=2hij*geometric*fs*fs
        push!(rows,(ratio=ratio,epoch=epoch,state_a=j-1,state_b=k-1,
            bits_a=state_bits(states,j),bits_b=state_bits(states,k),spin_flip=i,
            p_a=p[j],p_b=p[k],born_endpoint_mass=p[j]+p[k],
            edge_base=base,G_edge=gedge,absG_edge=abs(gedge),
            C_edge=cedge,absC_edge=abs(cedge),f_a=f[j],f_b=f[k]))
    end
    rows
end

function capture_summary(weights,importance,q)
    ord=sortperm(weights,rev=true); totalw=sum(weights); target=q*totalw
    acc=0.0; chosen=Int[]
    for j in ord
        push!(chosen,j); acc+=weights[j]; acc>=target && break
    end
    totalI=sum(importance)
    return length(chosen),acc/totalw,sum(importance[chosen])/max(totalI,eps())
end

function audit_checkpoint(H,model,tree,states,ratio,epoch,gmc,cmc,mode)
    A,p,f,el,E,f2,g=exact_checkpoint_data(H,model,tree,states)
    edges=edge_table(H,A,p,f,states,ratio,epoch)
    base=getproperty.(edges,:edge_base); absG=getproperty.(edges,:absG_edge); absC=getproperty.(edges,:absC_edge)
    Goff=sum(getproperty.(edges,:G_edge)); Coff=sum(getproperty.(edges,:C_edge))

    # Diagonal N''/Z contribution with <f>=0. Full curvature identity check.
    Cdiag=sum(p .* (4 .* [GBTQuantum.diagonal(H,@view states[j,:]) for j in axes(states,1)] .* f.*f))
    c=Cdiag+Coff-4E*f2
    eta=(g<0&&c>CURVATURE_FLOOR) ? -g/c : NaN

    println("\n", "-"^92)
    @printf("J/h=%.2f epoch=%d mode=%s | MC g=% .4e c=% .4e | exact-centered g=% .4e c=% .4e eta=% .5f\n",
        ratio,epoch,String(mode),gmc,cmc,g,c,eta)
    @printf("Gauge check <f>_p=% .3e | edges=%d | offdiag G=% .4e offdiag C=% .4e\n",
        sum(p.*f),length(edges),Goff,Coff)
    @printf("Spearman proposed edge_base vs |G_edge| = % .4f ; vs |C_edge| = % .4f\n",
        spearman(base,absG),spearman(base,absC))

    # Compare proposed edge distribution against a deliberately uniform edge baseline.
    uniform=ones(length(edges))
    for q in MASS_THRESHOLDS
        nb,mb,cg=capture_summary(base,absG,q)
        _,_,cc=capture_summary(base,absC,q)
        nu,_,ucg=capture_summary(uniform,absG,q)
        _,_,ucc=capture_summary(uniform,absC,q)
        @printf("edge-base mass %6.2f%%: %4d edges, captures |G_edge|=%7.3f%% |C_edge|=%7.3f%% | uniform same-mass baseline |G|=%7.3f%% |C|=%7.3f%%\n",
            100q,nb,100cg,100cc,100ucg,100ucc)
    end

    # Rank fields for CSV.
    rankb=zeros(Int,length(edges)); rankg=zeros(Int,length(edges)); rankc=zeros(Int,length(edges))
    for (r,j) in enumerate(sortperm(base,rev=true)); rankb[j]=r; end
    for (r,j) in enumerate(sortperm(absG,rev=true)); rankg[j]=r; end
    for (r,j) in enumerate(sortperm(absC,rev=true)); rankc[j]=r; end
    out=NamedTuple[]
    for j in eachindex(edges)
        e=edges[j]
        push!(out,merge(e,(rank_edge_base=rankb[j],rank_absG_edge=rankg[j],rank_absC_edge=rankc[j],
            E_exact=E,g_exact_centered=g,c_exact_centered=c,eta_exact_centered=eta,
            f_mean_exact=sum(p.*f),Cdiag=Cdiag,Coff=Coff)))
    end
    out
end

function train(H,ratio,states)
    cfg=GBTQuantum.TrainingConfig(nsamples=NSAMPLES,epochs=EPOCHS,max_depth=MAX_DEPTH,
        eta=ETA_FIXED,burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=SEED,
        exact_diagnostics=false)
    rng=MersenneTwister(cfg.seed); samples=Matrix{Int8}(undef,NSAMPLES,H.N)
    @inbounds for i in eachindex(samples); samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
    model=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    logamps=zeros(Float64,NSAMPLES)
    for _ in 1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,model,samples,logamps); end
    rows=NamedTuple[]; wanted=CHECKPOINTS[ratio]
    for epoch in 1:maximum(wanted)
        batch=GBTQuantum.vmc_batch(H,model,samples); yA,_=GBTQuantum.make_targets(batch)
        tree=GBTQuantum.grow_tree(batch.states,yA,batch.counts;max_depth=cfg.max_depth,
            min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain)
        tree=center_tree(tree,batch.states,batch.counts)
        eta,gmc,cmc,mode=mc_decision(H,model,batch,tree)
        epoch in wanted && append!(rows,audit_checkpoint(H,model,tree,states,ratio,epoch,gmc,cmc,mode))
        push!(model.logamp.trees,scaled_tree(tree,eta)); GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,model,samples,logamps); end
        if any(!isfinite,logamps)
            @printf("Trajectory non-finite after epoch %d for J/h=%.2f.\n",epoch,ratio); break
        end
    end
    rows
end

function write_csv(path,rows)
    isempty(rows)&&return; names=propertynames(rows[1])
    open(path,"w") do io
        println(io,join(string.(names),','))
        for r in rows; println(io,join((getproperty(r,n) for n in names),',')); end
    end
end

function main()
    println("="^92)
    println("TFIM EXACT HAMILTONIAN-EDGE IMPORTANCE MAP")
    println("N=$N Hilbert dimension=$(1<<N), undirected TFIM edges=$((1<<N)*N÷2)")
    println("Exact diagnostic gauge: <f>_p = 0")
    println("Proposed edge weight: |H_xx'| sqrt(p_x p_x')")
    println("checkpoints=$CHECKPOINTS")
    println("="^92)
    states=all_states(N); rows=NamedTuple[]
    for ratio in sort(collect(keys(CHECKPOINTS)))
        append!(rows,train(GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true),ratio,states))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"tfim_edge_importance_map.csv"); write_csv(path,rows)
    println("\nPer-edge results written to $path")
end
end

if abspath(PROGRAM_FILE)==@__FILE__; TFIMEdgeImportanceMap.main(); end
