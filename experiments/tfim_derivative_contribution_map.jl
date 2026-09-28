module TFIMDerivativeContributionMap

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Random
using Statistics
using Printf

# Exact N=8 microscope for the sample-efficiency problem.
# Reproduce the actual adaptive-eta V2 trajectory, freeze selected PRE-UPDATE
# model/tree checkpoints, enumerate all 2^N states, and ask whether Born mass
# p(x) is aligned with the states carrying the gradient/curvature information.

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

# Return exact per-state quantities. G_j and C_j sum algebraically to g and c.
function contribution_table(H,model,tree,states)
    d=size(states,1); A=zeros(d); f=zeros(d); el=zeros(d); elf=zeros(d)
    @inbounds for j in 1:d
        x=@view states[j,:]
        A[j]=GBTQuantum.logamplitude(model,x); f[j]=GBTQuantum.predict(tree,x)
    end
    lw=2 .* A; m=maximum(lw); w=exp.(lw.-m); p=w/sum(w)
    @inbounds for j in 1:d
        x=@view states[j,:]; fj=f[j]
        e=GBTQuantum.diagonal(H,x); ef=e*fj
        for i in 1:H.N
            k=flipped_row(j,i); r=exp(A[k]-A[j])
            e -= H.h*r; ef -= H.h*r*f[k]
        end
        el[j]=e; elf[j]=ef
    end
    E=sum(p.*el); Ef=sum(p.*f); Ef2=sum(p.*f.*f)
    G=2 .* p .* (f.-Ef) .* (el.-E)
    g=sum(G)
    C=p .* (2 .* f.*f.*el .+ 2 .* f.*elf .- 4E .* f.*f .- 4Ef*g)
    c=sum(C)
    return p,f,el,elf,G,C,E,g,c
end

# Spearman correlation without external dependencies; average ranks for ties.
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
function corr_safe(a,b)
    sa=std(a); sb=std(b); (sa==0||sb==0) ? NaN : cor(a,b)
end
spearman(a,b)=corr_safe(ranks(a),ranks(b))

function threshold_summary(p,G,C,threshold)
    ord=sortperm(p,rev=true); cp=0.0; chosen=Int[]
    for j in ord
        push!(chosen,j); cp+=p[j]; cp>=threshold && break
    end
    absG=sum(abs,G); absC=sum(abs,C)
    signed_g=sum(G); signed_c=sum(C)
    return length(chosen),cp,
        sum(abs.(G[chosen]))/max(absG,eps()),
        sum(abs.(C[chosen]))/max(absC,eps()),
        sum(G[chosen])/max(abs(signed_g),eps()),
        sum(C[chosen])/max(abs(signed_c),eps())
end

function state_bits(states,j)
    join(states[j,i]==1 ? "1" : "0" for i in 1:size(states,2))
end

function audit_checkpoint(H,model,tree,states,ratio,epoch,gmc,cmc,mode)
    p,f,el,elf,G,C,E,g,c=contribution_table(H,model,tree,states)
    absG=sum(abs,G); absC=sum(abs,C)
    eta=(g<0&&c>CURVATURE_FLOOR) ? -g/c : NaN
    println("\n", "-"^86)
    @printf("J/h=%.2f epoch=%d mode=%s | MC g=% .4e c=% .4e | exact g=% .4e c=% .4e eta=% .5f\n",
        ratio,epoch,String(mode),gmc,cmc,g,c,eta)
    @printf("Spearman: p vs |G| = % .4f ; p vs |C| = % .4f\n",spearman(p,abs.(G)),spearman(p,abs.(C)))
    @printf("Cancellation: |g|/sum|G|=%.4e ; |c|/sum|C|=%.4e\n",abs(g)/absG,abs(c)/absC)
    for q in MASS_THRESHOLDS
        n,mass,fg,fc,sg,sc=threshold_summary(p,G,C,q)
        @printf("Born mass %6.2f%%: %3d states, actual mass=%8.5f, captures |G|=%7.3f%% |C|=%7.3f%%, signed g=% .3f x exact, signed c=% .3f x exact\n",
            100q,n,mass,100fg,100fc,sg,sc)
    end

    # Per-state CSV rows; ranks are 1=largest importance.
    rankp=zeros(Int,length(p)); rankG=zeros(Int,length(p)); rankC=zeros(Int,length(p))
    for (r,j) in enumerate(sortperm(p,rev=true)); rankp[j]=r; end
    for (r,j) in enumerate(sortperm(abs.(G),rev=true)); rankG[j]=r; end
    for (r,j) in enumerate(sortperm(abs.(C),rev=true)); rankC[j]=r; end
    rows=NamedTuple[]
    for j in eachindex(p)
        push!(rows,(ratio=ratio,epoch=epoch,state_index=j-1,state_bits=state_bits(states,j),
            p=p[j],f=f[j],E_local=el[j],Ef_local=elf[j],G=G[j],absG=abs(G[j]),
            C=C[j],absC=abs(C[j]),rank_p=rankp[j],rank_absG=rankG[j],rank_absC=rankC[j],
            E_exact=E,g_exact=g,c_exact=c,eta_exact=eta))
    end
    return rows
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
            @printf("Trajectory non-finite after epoch %d for J/h=%.2f; later requested checkpoints unavailable.\n",epoch,ratio)
            break
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
    println("="^86)
    println("TFIM EXACT PER-STATE DERIVATIVE CONTRIBUTION MAP")
    println("N=$N, Hilbert dimension=$(1<<N), V2 trajectory samples=$NSAMPLES")
    println("checkpoints=$CHECKPOINTS")
    println("="^86)
    states=all_states(N); rows=NamedTuple[]
    for ratio in sort(collect(keys(CHECKPOINTS)))
        append!(rows,train(GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true),ratio,states))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"tfim_derivative_contribution_map.csv"); write_csv(path,rows)
    println("\nPer-state results written to $path")
end
end

if abspath(PROGRAM_FILE)==@__FILE__; TFIMDerivativeContributionMap.main(); end
