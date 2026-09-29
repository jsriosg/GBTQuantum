module TFIMLeafGradientTermDecomposition
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using GBTQuantum, Random, Statistics, Printf

# Decompose leafwise gradient error into leaf-mass and conditional-local-energy pieces.
# Same production-gauge tree is used for MC and exact oracle; exact never re-centers.
#
# g = 2(<f E_loc> - <f>E) = 2(A + B)
# A_L = f_L P_L Ebar_L
# B_L = -f_L P_L E
#
# Exact identity for A error:
# Ahat_L - A_L = f_L[(Phat_L-P_L) Ebar_L + Phat_L(Ebarhat_L-Ebar_L)].
# Thus we can distinguish mass-estimation error from conditional-energy-mean error.

const N=8
const NSAMPLES=512
const MAX_DEPTH=4
const ETA_FIXED=0.05
const SEED=1234
const CHECKPOINTS=Dict(1.0=>Set([8,50]),2.0=>Set([4,10]))

function allstates(N)
    X=Matrix{Int8}(undef,1<<N,N)
    for s=0:(1<<N)-1, i=1:N
        X[s+1,i]=((s>>(i-1))&1)==1 ? Int8(1) : Int8(-1)
    end
    X
end

function leaf_index(t,x)
    i=1
    while true
        n=t.nodes[i]; n.isleaf && return i
        i=x[n.feature]<=0 ? n.left : n.right
    end
end

function center_like_production(t,b)
    w=Float64.(b.counts)
    pred=[GBTQuantum.predict(t,@view b.states[j,:]) for j in axes(b.states,1)]
    mu=GBTQuantum.weighted_mean(pred,w); nd=copy(t.nodes)
    for i in eachindex(nd)
        n=nd[i]
        n.isleaf && (nd[i]=GBTQuantum.Node(n.feature,n.value-mu,n.left,n.right,true))
    end
    GBTQuantum.RegressionTree(nd),mu
end

scaled_tree(t,e)=GBTQuantum.RegressionTree([
    n.isleaf ? GBTQuantum.Node(n.feature,e*n.value,n.left,n.right,true) : n for n in t.nodes
])

function ranks(v)
    n=length(v); o=sortperm(v); r=zeros(Float64,n); k=1
    while k<=n
        q=k
        while q<n && v[o[q+1]]==v[o[k]]; q+=1; end
        rr=(k+q)/2
        for j=k:q; r[o[j]]=rr; end
        k=q+1
    end
    r
end
rankcorr(x,y)=(std(ranks(x))==0 || std(ranks(y))==0) ? NaN : cor(ranks(x),ranks(y))

function exact_arrays(H,m,t,X)
    d=size(X,1); A=zeros(d); f=zeros(d); leaf=zeros(Int,d); el=zeros(d)
    for j=1:d
        x=@view X[j,:]
        A[j]=GBTQuantum.logamplitude(m,x); f[j]=GBTQuantum.predict(t,x); leaf[j]=leaf_index(t,x)
    end
    p=exp.(2 .* A .- maximum(2 .* A)); p ./= sum(p)
    for j=1:d
        x=@view X[j,:]; el[j]=GBTQuantum.diagonal(H,x)
        for i=1:H.N
            k=((j-1) ⊻ (1<<(i-1)))+1
            el[j]-=H.h*exp(A[k]-A[j])
        end
    end
    Ef=sum(p .* f); E=sum(p .* el)
    g=2*sum(p .* (f .- Ef) .* (el .- E))
    (;p,f,leaf,el,Ef,E,g)
end

function sample_arrays(t,b)
    w=Float64.(b.counts); W=sum(w); el=real.(b.local_energy)
    f=[GBTQuantum.predict(t,@view b.states[j,:]) for j in axes(b.states,1)]
    leaf=[leaf_index(t,@view b.states[j,:]) for j in axes(b.states,1)]
    Ef=sum(w .* f)/W; E=sum(w .* el)/W
    g=2*sum(w .* (f .- Ef) .* (el .- E))/W
    (;w,W,el,f,leaf,Ef,E,g)
end

function audit(H,m,t,b,X,ratio,ep,shift)
    ex=exact_arrays(H,m,t,X); sm=sample_arrays(t,b)
    rows=NamedTuple[]; leaves=sort(unique(ex.leaf))
    @printf("\n%s\n","-"^148)
    @printf("J/h=%.2f epoch=%d | shift=% .3e | <f>MC=% .3e <f>exact=% .3e | E_MC=% .6e E_exact=% .6e | dg=% .3e\n",
        ratio,ep,shift,sm.Ef,ex.Ef,sm.E,ex.E,sm.g-ex.g)

    for L in leaves
        ix=findall(==(L),ex.leaf); im=findall(==(L),sm.leaf)
        P=sum(ex.p[ix]); Phat=sum(sm.w[im])/sm.W; dP=Phat-P; fL=ex.f[first(ix)]
        Ebar=sum(ex.p[ix] .* ex.el[ix])/P
        Ebarhat=isempty(im) ? NaN : sum(sm.w[im] .* sm.el[im])/sum(sm.w[im])
        dEbar=Ebarhat-Ebar

        Aex=fL*P*Ebar
        Amc=isempty(im) ? 0.0 : fL*Phat*Ebarhat
        dA=Amc-Aex
        dA_mass=fL*dP*Ebar
        dA_cond=isempty(im) ? -fL*P*Ebar : fL*Phat*dEbar
        identity_resid=dA-dA_mass-dA_cond

        Bex=-fL*P*ex.E
        Bmc=-fL*Phat*sm.E
        dB=Bmc-Bex
        dg=2*(dA+dB)

        # Split dB exactly into mass and global-factor errors:
        # Phat*Ehat - P*E = dP*E + Phat*dE.
        Qex=ex.E; Qmc=sm.E
        dB_mass=-fL*dP*Qex
        dB_global=-fL*Phat*(Qmc-Qex)

        sigP=sqrt(max(Phat*(1-Phat),0.0)/sm.W)
        nL=sum(sm.w[im])
        if isempty(im) || nL<=1
            se_Ebar=Inf
            sd_Ebar=NaN
        else
            mu=Ebarhat
            varL=sum(sm.w[im] .* (sm.el[im] .- mu).^2)/nL
            sd_Ebar=sqrt(max(varL,0.0))
            se_Ebar=sd_Ebar/sqrt(nL)
        end
        risk_mass=2*abs(fL)*sigP*abs(Ebar-ex.E)
        risk_cond=isfinite(se_Ebar) ? 2*abs(fL)*Phat*se_Ebar : Inf
        risk_quad=sqrt(risk_mass^2 + risk_cond^2)
        oracle_mass=2*abs(dA_mass)
        oracle_cond=2*abs(dA_cond)

        push!(rows,(ratio=ratio,epoch=ep,leaf=L,support=nL,f=fL,absf=abs(fL),
            born_mass=P,sample_mass=Phat,mass_error=dP,Ebar_exact=Ebar,Ebar_mc=Ebarhat,dEbar=dEbar,
            A_exact=Aex,A_mc=Amc,dA=dA,dA_mass=dA_mass,dA_conditional=dA_cond,A_identity_residual=identity_resid,
            B_exact=Bex,B_mc=Bmc,dB=dB,dB_mass=dB_mass,dB_global=dB_global,
            dg=dg,abs_dg=abs(dg),sigma_mass=sigP,sd_Ebar_mc=sd_Ebar,se_Ebar_mc=se_Ebar,
            risk_mass=risk_mass,risk_conditional=risk_cond,risk_quadrature=risk_quad,
            oracle_mass=oracle_mass,oracle_conditional=oracle_cond,
            exact_mean_f=ex.Ef,mc_mean_f=sm.Ef,E_exact=ex.E,E_mc=sm.E,g_exact=ex.g,g_mc=sm.g))
    end

    @printf("Sum checks: gExact=% .6e leaves=% .6e | gMC=% .6e leaves=% .6e | dg=% .3e leaves=% .3e\n",
        ex.g,2*sum(r.A_exact+r.B_exact for r in rows),sm.g,2*sum(r.A_mc+r.B_mc for r in rows),
        sm.g-ex.g,sum(r.dg for r in rows))
    @printf("Error components summed: 2*dA_mass=% .3e 2*dA_cond=% .3e 2*dB_mass=% .3e 2*dB_global=% .3e\n",
        2*sum(r.dA_mass for r in rows),2*sum(r.dA_conditional for r in rows),
        2*sum(r.dB_mass for r in rows),2*sum(r.dB_global for r in rows))

    y=[r.abs_dg for r in rows]
    @printf("Spearman with |dg_L|: oracle-mass=% .3f oracle-cond=% .3f risk-mass=% .3f risk-cond=% .3f risk-quadrature=% .3f\n",
        rankcorr([r.oracle_mass for r in rows],y),rankcorr([r.oracle_conditional for r in rows],y),
        rankcorr([r.risk_mass for r in rows],y),rankcorr([r.risk_conditional for r in rows],y),
        rankcorr([r.risk_quadrature for r in rows],y))

    ord=sortperm(rows,by=r->r.abs_dg,rev=true)
    println(" Top leaves by |dg_L|:")
    for k in ord[1:min(8,length(ord))]
        r=rows[k]
        @printf("  L=%3d dg=% .3e | 2dA_mass=% .2e 2dA_cond=% .2e 2dB=% .2e | W=%5.1f f=% .3e dP=% .2e dEbar=% .2e | Rm=% .2e Rc=% .2e\n",
            r.leaf,r.dg,2*r.dA_mass,2*r.dA_conditional,2*r.dB,r.support,r.f,r.mass_error,r.dEbar,r.risk_mass,r.risk_conditional)
    end
    rows
end

function train(H,ratio,X)
    cfg=GBTQuantum.TrainingConfig(nsamples=NSAMPLES,epochs=150,max_depth=MAX_DEPTH,eta=ETA_FIXED,
        burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=SEED,exact_diagnostics=false)
    rng=MersenneTwister(SEED); S=Matrix{Int8}(undef,NSAMPLES,H.N)
    for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
    m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); la=zeros(NSAMPLES)
    for _=1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,m,S,la); end
    out=NamedTuple[]; wanted=CHECKPOINTS[ratio]
    for ep=1:maximum(wanted)
        b=GBTQuantum.vmc_batch(H,m,S); y,_=GBTQuantum.make_targets(b)
        raw=GBTQuantum.grow_tree(b.states,y,b.counts;max_depth=cfg.max_depth,
            min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain)
        t,shift=center_like_production(raw,b)
        ep in wanted && append!(out,audit(H,m,t,b,X,ratio,ep,shift))
        push!(m.logamp.trees,scaled_tree(t,ETA_FIXED)); GBTQuantum.refresh_logamps!(la,m,S)
        for _=1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,m,S,la); end
        any(!isfinite,la) && break
    end
    out
end

function writecsv(path,rows)
    isempty(rows)&&return
    ns=propertynames(rows[1])
    open(path,"w") do io
        println(io,join(string.(ns),','))
        for r in rows; println(io,join((getproperty(r,n) for n in ns),',')); end
    end
end

function main()
    println("="^148)
    println("TFIM LEAF GRADIENT TERM-DECOMPOSITION DIAGNOSTIC")
    println("N=$N Hilbert=$(1<<N), M=$NSAMPLES, fixed eta=$ETA_FIXED")
    println("Decomposes gradient error into leaf-mass, conditional-local-energy, and global gauge/energy pieces.")
    println("Exact enumeration is oracle-only; production gauge is unchanged.")
    println("="^148)
    X=allstates(N); rows=NamedTuple[]
    for ratio in sort(collect(keys(CHECKPOINTS)))
        H=GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true)
        append!(rows,train(H,ratio,X))
    end
    dir=joinpath(@__DIR__,"results"); mkpath(dir)
    path=joinpath(dir,"tfim_leaf_gradient_term_decomposition.csv")
    writecsv(path,rows); println("\nResults written to $path")
end
end

if abspath(PROGRAM_FILE)==@__FILE__
    TFIMLeafGradientTermDecomposition.main()
end
