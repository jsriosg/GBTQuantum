module TFIMUpdateAwareEdgeOracle

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using GBTQuantum, Random, Statistics, Printf

# Exact N=8 oracle experiment for update-aware edge importance sampling.
# This does NOT yet claim a scalable estimator. It isolates the normalization issue.
# At frozen V2 checkpoints, enumerate all undirected TFIM edges and compare:
#   q0(e) ∝ |H| sqrt(p_x p_y)
#   qG(e) ∝ |H| sqrt(p_x p_y) |f_x+f_y|
#   qC(e) ∝ |H| sqrt(p_x p_y) (f_x+f_y)^2
# Then Monte Carlo sample edges from each exact oracle distribution and estimate the
# signed off-diagonal derivative sums using the exact oracle normalizer Q.
# Gauge: exact <f>_p = 0.

const N=8; const NSAMPLES=512; const MAX_DEPTH=4; const ETA_FIXED=0.05
const ETA_CAP=0.40; const CURVATURE_FLOOR=1e-12; const SEED=1234
const CHECKPOINTS=Dict(1.0=>Set([8,50]),2.0=>Set([4,10]))
const EDGE_SAMPLE_SIZES=[32,64,128,256,512]
const REPLICATES=100

function center_tree(tree,X,w)
    pr=[GBTQuantum.predict(tree,@view X[j,:]) for j in axes(X,1)]; mu=GBTQuantum.weighted_mean(pr,w)
    mu==0 && return tree; nodes=copy(tree.nodes)
    for i in eachindex(nodes); n=nodes[i]; n.isleaf && (nodes[i]=GBTQuantum.Node(n.feature,n.value-mu,n.left,n.right,true)); end
    GBTQuantum.RegressionTree(nodes)
end
scaled_tree(t,e)=GBTQuantum.RegressionTree([n.isleaf ? GBTQuantum.Node(n.feature,e*n.value,n.left,n.right,true) : n for n in t.nodes])
function states(N)
    X=Matrix{Int8}(undef,1<<N,N)
    for s in 0:(1<<N)-1, i in 1:N; X[s+1,i]=((s>>(i-1))&1)==1 ? Int8(1) : Int8(-1); end; X
end
@inline flip(j,i)=((j-1) ⊻ (1<<(i-1)))+1
function flocal(H,m,t,x)
    A=GBTQuantum.logamplitude(m,x); f=GBTQuantum.predict(t,x); z=GBTQuantum.diagonal(H,x)*f
    for i in 1:H.N; x[i]=-x[i]; z-=H.h*exp(GBTQuantum.logamplitude(m,x)-A)*GBTQuantum.predict(t,x); x[i]=-x[i]; end; z
end
function decision(H,m,b,t)
    f=[GBTQuantum.predict(t,@view b.states[j,:]) for j in axes(b.states,1)]; w=Float64.(b.counts); el=real.(b.local_energy); W=sum(w)
    Ef=sum(w.*f)/W; E=sum(w.*el)/W; Ef2=sum(w.*f.*f)/W; Ef2el=sum(w.*f.*f.*el)/W
    g=2sum(w.*(f.-Ef).*(el.-E))/W; q=0.0
    for j in axes(b.states,1); q+=w[j]*f[j]*flocal(H,m,t,@view b.states[j,:]); end
    c=2Ef2el+2q/W-4E*Ef2-4Ef*g
    if !isfinite(g)||!isfinite(c)||g>=0||c<=CURVATURE_FLOOR; return ETA_FIXED,g,c,:fallback; end
    raw=-g/c; (!isfinite(raw)||raw<=0) && return (ETA_FIXED,g,c,:fallback)
    clamp(raw,0.0,ETA_CAP),g,c,:newton
end

function exact_data(H,m,t,X)
    d=size(X,1); A=zeros(d); f=zeros(d)
    for j in 1:d; A[j]=GBTQuantum.logamplitude(m,@view X[j,:]); f[j]=GBTQuantum.predict(t,@view X[j,:]); end
    lw=2A; mm=maximum(lw); p=exp.(lw.-mm); p./=sum(p); f.-=sum(p.*f)
    el=zeros(d)
    for j in 1:d
        e=GBTQuantum.diagonal(H,@view X[j,:])
        for i in 1:H.N; k=flip(j,i); e-=H.h*exp(A[k]-A[j]); end; el[j]=e
    end
    E=sum(p.*el); g=2sum(p.*f.*el); f2=sum(p.*f.*f)
    A,p,f,E,g,f2
end

function edges(H,p,f)
    base=Float64[]; sg=Float64[]; sc=Float64[]
    for j in eachindex(p), i in 1:H.N
        k=flip(j,i); j<k || continue
        h=-H.h; b=abs(h)*sqrt(p[j]*p[k]); fs=f[j]+f[k]
        push!(base,b); push!(sg,2h*sqrt(p[j]*p[k])*fs); push!(sc,2h*sqrt(p[j]*p[k])*fs^2)
    end
    base,sg,sc
end

# iid categorical draw; exact enumeration supplies q and Q only for this oracle test.
function draw_index(rng,cdf)
    u=rand(rng); searchsortedfirst(cdf,u)
end
function oracle_estimates(rng,weights,signed,n,reps)
    Q=sum(weights); q=weights./Q; cdf=cumsum(q); vals=zeros(reps)
    for r in 1:reps
        s=0.0
        for _ in 1:n
            j=draw_index(rng,cdf)
            s += signed[j]/q[j]
        end
        vals[r]=s/n
    end
    vals,Q
end
function stats(v,truev)
    (mean=mean(v),bias=mean(v)-truev,rmse=sqrt(mean((v.-truev).^2)),sd=std(v))
end

function audit(H,m,t,X,ratio,epoch,gmc,cmc,mode)
    A,p,f,E,g,f2=exact_data(H,m,t,X); base,G,C=edges(H,p,f)
    Gtrue=sum(G); Ctrue=sum(C)
    Cdiag=sum(p .* (4 .* [GBTQuantum.diagonal(H,@view X[j,:]) for j in axes(X,1)] .* f.*f))
    c=Cdiag+Ctrue-4E*f2
    println("\n","-"^100)
    @printf("J/h=%.2f epoch=%d mode=%s | trajectory g=% .4e c=% .4e | exact g=% .4e c=% .4e\n",ratio,epoch,String(mode),gmc,cmc,g,c)
    @printf("<f>=% .2e | exact offdiag G=% .4e C=% .4e | Cdiag=% .4e | -4E<f2>=% .4e\n",sum(p.*f),Gtrue,Ctrue,Cdiag,-4E*f2)
    q0=base; qG=base.*abs.([G[j]/(2*(-H.h)*sqrt(p[1]*p[1])+eps()) for j in eachindex(G)]) # overwritten below cleanly
    # Since G=2 h sqrt(pp) fs and C=2 h sqrt(pp) fs^2, update-aware weights are simply |G|/2 and |C|/2 up to constants.
    qG=abs.(G); qC=abs.(C)
    @printf("Oracle normalizers: Q0=% .4e QG(abs signed G)=% .4e QC(abs signed C)=% .4e\n",sum(q0),sum(qG),sum(qC))
    rows=NamedTuple[]
    for n in EDGE_SAMPLE_SIZES
        for (name,w,signed,truev) in (("base->G",q0,G,Gtrue),("qG->G",qG,G,Gtrue),("base->C",q0,C,Ctrue),("qC->C",qC,C,Ctrue))
            rng=MersenneTwister(SEED + 100000*round(Int,ratio)+1000*epoch+17*n+sum(codeunits(name)))
            vals,Q=oracle_estimates(rng,w,signed,n,REPLICATES); st=stats(vals,truev)
            @printf("  n=%3d %-8s mean=% .4e bias=% .2e rmse=% .3e sd=% .3e\n",n,name,st.mean,st.bias,st.rmse,st.sd)
            push!(rows,(ratio=ratio,epoch=epoch,n=n,proposal=name,replicates=REPLICATES,true_value=truev,mean=st.mean,bias=st.bias,rmse=st.rmse,sd=st.sd,Q=Q))
        end
    end
    rows
end

function train(H,ratio,X)
    cfg=GBTQuantum.TrainingConfig(nsamples=NSAMPLES,epochs=150,max_depth=MAX_DEPTH,eta=ETA_FIXED,burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=SEED,exact_diagnostics=false)
    rng=MersenneTwister(SEED); S=Matrix{Int8}(undef,NSAMPLES,H.N); for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
    m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); la=zeros(NSAMPLES)
    for _ in 1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,m,S,la); end
    out=NamedTuple[]; wanted=CHECKPOINTS[ratio]
    for ep in 1:maximum(wanted)
        b=GBTQuantum.vmc_batch(H,m,S); y,_=GBTQuantum.make_targets(b); t=GBTQuantum.grow_tree(b.states,y,b.counts;max_depth=cfg.max_depth,min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain); t=center_tree(t,b.states,b.counts)
        eta,g,c,mode=decision(H,m,b,t); ep in wanted && append!(out,audit(H,m,t,X,ratio,ep,g,c,mode))
        push!(m.logamp.trees,scaled_tree(t,eta)); GBTQuantum.refresh_logamps!(la,m,S); for _ in 1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,m,S,la); end
        any(!isfinite,la) && break
    end; out
end
function csv(path,rows)
    isempty(rows)&&return; ns=propertynames(rows[1]); open(path,"w") do io; println(io,join(string.(ns),',')); for r in rows; println(io,join((getproperty(r,n) for n in ns),',')); end; end
end
function main()
    println("="^100); println("TFIM UPDATE-AWARE EDGE ORACLE SAMPLING DIAGNOSTIC"); println("N=$N, edge sample sizes=$EDGE_SAMPLE_SIZES, replicates=$REPLICATES"); println("Exact checkpoint gauge: <f>_p=0; qG/qC normalizers are ORACLE quantities in this experiment"); println("="^100)
    X=states(N); rows=NamedTuple[]
    for r in sort(collect(keys(CHECKPOINTS))); append!(rows,train(GBTQuantum.TFIMHamiltonian(N;J=r,h=1.0,periodic=true),r,X)); end
    dir=joinpath(@__DIR__,"results"); mkpath(dir); path=joinpath(dir,"tfim_update_aware_edge_oracle.csv"); csv(path,rows); println("\nResults written to $path")
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMUpdateAwareEdgeOracle.main(); end
