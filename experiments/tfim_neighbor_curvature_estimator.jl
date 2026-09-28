module TFIMNeighborCurvatureEstimator

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using GBTQuantum, Random, Statistics, Printf

# Diagnostic: can ordinary Born root samples estimate curvature reliably when each
# sampled root deterministically contributes its complete Hamiltonian neighborhood?
#
# Exact gauge: <f>_p = 0.
# For TFIM H_off=-h and y=x^i,
#   C_off = -h E_p[ sum_i exp(A(y)-A(x)) (f(x)+f(y))^2 ].
# Full curvature:
#   c = Cdiag + Coff - 4 E <f^2>,
#   Cdiag = 4 E_p[H_xx f(x)^2].
#
# IMPORTANT: algebraically this is an exact regrouping of the same curvature.
# The purpose is empirical: compare the finite-sample estimator, using the SAME
# frozen Born root samples, with the canonical V2 curvature estimator and exact c.

const N=8
const TRAJECTORY_SAMPLES=512
const SAMPLE_SIZES=[64,128,256,512]
const REPLICATES=100
const MAX_DEPTH=4
const ETA_FIXED=0.05
const ETA_CAP=0.40
const CURVATURE_FLOOR=1e-12
const SEED=1234
const CHECKPOINTS=Dict(1.0=>Set([8,50]),2.0=>Set([4,10]))

function center_tree(tree,X,w)
    pr=[GBTQuantum.predict(tree,@view X[j,:]) for j in axes(X,1)]
    mu=GBTQuantum.weighted_mean(pr,w); mu==0 && return tree
    nodes=copy(tree.nodes)
    for i in eachindex(nodes)
        n=nodes[i]
        n.isleaf && (nodes[i]=GBTQuantum.Node(n.feature,n.value-mu,n.left,n.right,true))
    end
    GBTQuantum.RegressionTree(nodes)
end
scaled_tree(t,e)=GBTQuantum.RegressionTree([n.isleaf ? GBTQuantum.Node(n.feature,e*n.value,n.left,n.right,true) : n for n in t.nodes])

function allstates(N)
    X=Matrix{Int8}(undef,1<<N,N)
    for s in 0:(1<<N)-1, i in 1:N
        X[s+1,i]=((s>>(i-1))&1)==1 ? Int8(1) : Int8(-1)
    end
    X
end
@inline fliprow(j,i)=((j-1) ⊻ (1<<(i-1)))+1

function flocal(H,m,t,x)
    A=GBTQuantum.logamplitude(m,x); f=GBTQuantum.predict(t,x)
    z=GBTQuantum.diagonal(H,x)*f
    for i in 1:H.N
        x[i]=-x[i]
        z-=H.h*exp(GBTQuantum.logamplitude(m,x)-A)*GBTQuantum.predict(t,x)
        x[i]=-x[i]
    end
    z
end

function canonical_mc(H,m,b,t)
    f=[GBTQuantum.predict(t,@view b.states[j,:]) for j in axes(b.states,1)]
    w=Float64.(b.counts); el=real.(b.local_energy); W=sum(w)
    Ef=sum(w.*f)/W; E=sum(w.*el)/W; Ef2=sum(w.*f.*f)/W
    Ef2el=sum(w.*f.*f.*el)/W
    g=2sum(w.*(f.-Ef).*(el.-E))/W
    q=0.0
    for j in axes(b.states,1)
        q+=w[j]*f[j]*flocal(H,m,t,@view b.states[j,:])
    end
    c=2Ef2el+2q/W-4E*Ef2-4Ef*g
    eta=(isfinite(g)&&isfinite(c)&&g<0&&c>CURVATURE_FLOOR) ? clamp(-g/c,0.0,ETA_CAP) : ETA_FIXED
    mode=(isfinite(g)&&isfinite(c)&&g<0&&c>CURVATURE_FLOOR) ? :newton : :fallback
    eta,g,c,mode
end

function exact_checkpoint(H,m,t,X)
    d=size(X,1); A=zeros(d); f=zeros(d)
    for j in 1:d
        A[j]=GBTQuantum.logamplitude(m,@view X[j,:]); f[j]=GBTQuantum.predict(t,@view X[j,:])
    end
    mm=maximum(2A); p=exp.(2A.-mm); p./=sum(p)
    f .-= sum(p.*f) # exact gauge
    el=zeros(d)
    for j in 1:d
        e=GBTQuantum.diagonal(H,@view X[j,:])
        for i in 1:H.N
            k=fliprow(j,i); e-=H.h*exp(A[k]-A[j])
        end
        el[j]=e
    end
    E=sum(p.*el); f2=sum(p.*f.*f); g=2sum(p.*f.*el)
    Cdiag=4sum(p .* [GBTQuantum.diagonal(H,@view X[j,:]) for j in 1:d] .* f.*f)
    Coff=0.0
    for j in 1:d, i in 1:H.N
        k=fliprow(j,i)
        Coff += p[j]*(-H.h)*exp(A[k]-A[j])*(f[j]+f[k])^2
    end
    c=Cdiag+Coff-4E*f2
    A,p,f,el,E,f2,g,Cdiag,Coff,c
end

# Draw iid roots exactly from frozen p. This removes MCMC autocorrelation and tests
# only estimator variance. A later test can feed the actual trajectory roots.
function categorical(rng,cdf)
    searchsortedfirst(cdf,rand(rng))
end

# Neighbor estimator from root indices. Uses exact-centered f only for this microscope.
function neighbor_from_roots(H,A,f,el,X,roots)
    M=length(roots); Es=0.0; f2s=0.0; Cds=0.0; Cos=0.0
    for j in roots
        fj=f[j]; Es+=el[j]; f2s+=fj^2
        Cds += 4*GBTQuantum.diagonal(H,@view X[j,:])*fj^2
        inner=0.0
        for i in 1:H.N
            k=fliprow(j,i)
            inner += exp(A[k]-A[j])*(fj+f[k])^2
        end
        Cos += -H.h*inner
    end
    Ehat=Es/M; f2hat=f2s/M; Cd=Cds/M; Co=Cos/M
    chat=Cd+Co-4Ehat*f2hat
    chat,Cd,Co,Ehat,f2hat
end

# Canonical curvature evaluated on the exact same iid roots (simple average, duplicates
# retained). This isolates whether regrouping changes variance rather than sample set.
function canonical_from_roots(H,A,f,el,X,roots)
    M=length(roots)
    Ehat=mean(el[roots]); Ef=mean(f[roots]); Ef2=mean(f[roots].^2)
    Ef2el=mean((f[roots].^2).*el[roots])
    ghat=2mean((f[roots].-Ef).*(el[roots].-Ehat))
    q=0.0
    for j in roots
        fj=f[j]; inner=GBTQuantum.diagonal(H,@view X[j,:])*fj
        for i in 1:H.N
            k=fliprow(j,i)
            inner += (-H.h)*exp(A[k]-A[j])*f[k]
        end
        q += fj*inner
    end
    2Ef2el+2q/M-4Ehat*Ef2-4Ef*ghat
end

function summarize(v,truev)
    (mean=mean(v),bias=mean(v)-truev,rmse=sqrt(mean((v.-truev).^2)),sd=std(v))
end

function audit(H,m,t,X,ratio,epoch,gmc,cmc,mode)
    A,p,f,el,E,f2,g,Cdiag,Coff,c=exact_checkpoint(H,m,t,X)
    @printf("\n%s\n", "-"^104)
    @printf("J/h=%.2f epoch=%d mode=%s | trajectory c=% .4e | exact centered g=% .4e c=% .4e eta=% .5f\n",
        ratio,epoch,String(mode),cmc,g,c,(g<0&&c>0 ? -g/c : NaN))
    @printf("<f>=% .3e | exact Cdiag=% .4e Coff=% .4e -4E<f2>=% .4e\n",sum(p.*f),Cdiag,Coff,-4E*f2)
    cdf=cumsum(p); rows=NamedTuple[]
    for M in SAMPLE_SIZES
        old=zeros(REPLICATES); nei=zeros(REPLICATES); co=zeros(REPLICATES)
        for r in 1:REPLICATES
            rng=MersenneTwister(SEED + round(Int,ratio*100000)+epoch*1000+M*17+r)
            roots=[categorical(rng,cdf) for _ in 1:M]
            old[r]=canonical_from_roots(H,A,f,el,X,roots)
            nei[r],_,co[r],_,_=neighbor_from_roots(H,A,f,el,X,roots)
        end
        so=summarize(old,c); sn=summarize(nei,c); sco=summarize(co,Coff)
        @printf("  M=%3d canonical c: mean=% .4e rmse=% .3e sd=% .3e | neighbor c: mean=% .4e rmse=% .3e sd=% .3e | neighbor Coff rmse=% .3e\n",
            M,so.mean,so.rmse,so.sd,sn.mean,sn.rmse,sn.sd,sco.rmse)
        push!(rows,(ratio=ratio,epoch=epoch,M=M,replicates=REPLICATES,c_exact=c,Coff_exact=Coff,
            canonical_mean=so.mean,canonical_bias=so.bias,canonical_rmse=so.rmse,canonical_sd=so.sd,
            neighbor_mean=sn.mean,neighbor_bias=sn.bias,neighbor_rmse=sn.rmse,neighbor_sd=sn.sd,
            neighbor_Coff_mean=sco.mean,neighbor_Coff_bias=sco.bias,neighbor_Coff_rmse=sco.rmse,neighbor_Coff_sd=sco.sd))
    end
    rows
end

function train(H,ratio,X)
    cfg=GBTQuantum.TrainingConfig(nsamples=TRAJECTORY_SAMPLES,epochs=150,max_depth=MAX_DEPTH,eta=ETA_FIXED,burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=SEED,exact_diagnostics=false)
    rng=MersenneTwister(SEED); S=Matrix{Int8}(undef,TRAJECTORY_SAMPLES,H.N)
    for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
    m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); la=zeros(TRAJECTORY_SAMPLES)
    for _ in 1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,m,S,la); end
    out=NamedTuple[]; wanted=CHECKPOINTS[ratio]
    for ep in 1:maximum(wanted)
        b=GBTQuantum.vmc_batch(H,m,S); y,_=GBTQuantum.make_targets(b)
        t=GBTQuantum.grow_tree(b.states,y,b.counts;max_depth=cfg.max_depth,min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain)
        t=center_tree(t,b.states,b.counts); eta,gmc,cmc,mode=canonical_mc(H,m,b,t)
        ep in wanted && append!(out,audit(H,m,t,X,ratio,ep,gmc,cmc,mode))
        push!(m.logamp.trees,scaled_tree(t,eta)); GBTQuantum.refresh_logamps!(la,m,S)
        for _ in 1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,m,S,la); end
        any(!isfinite,la) && break
    end
    out
end

function writecsv(path,rows)
    isempty(rows)&&return; ns=propertynames(rows[1])
    open(path,"w") do io
        println(io,join(string.(ns),','))
        for r in rows; println(io,join((getproperty(r,n) for n in ns),',')); end
    end
end

function main()
    println("="^104)
    println("TFIM HAMILTONIAN-NEIGHBORHOOD CURVATURE ESTIMATOR DIAGNOSTIC")
    println("N=$N; sample sizes=$SAMPLE_SIZES; replicates=$REPLICATES; V2 trajectory samples=$TRAJECTORY_SAMPLES")
    println("Frozen-root test uses iid exact Born roots; canonical and neighbor estimators receive identical roots")
    println("Exact diagnostic gauge: <f>_p=0")
    println("="^104)
    X=allstates(N); rows=NamedTuple[]
    for ratio in sort(collect(keys(CHECKPOINTS)))
        H=GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true)
        append!(rows,train(H,ratio,X))
    end
    dir=joinpath(@__DIR__,"results"); mkpath(dir)
    path=joinpath(dir,"tfim_neighbor_curvature_estimator.csv"); writecsv(path,rows)
    println("\nResults written to $path")
end
end

if abspath(PROGRAM_FILE)==@__FILE__; TFIMNeighborCurvatureEstimator.main(); end
