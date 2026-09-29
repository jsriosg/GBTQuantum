module TFIMLeafGradientErrorDiagnostic
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using GBTQuantum, Random, Statistics, Printf

# Frozen-checkpoint leafwise gradient-error diagnostic.
# Production gauge: the fitted tree is shifted so <f>_MC = 0.
# Exact enumeration evaluates that SAME tree and never re-centers it.
#
# g = 2(<f E_loc> - <f>E)
# and for leaf L (f=f_L inside the leaf):
# g_L = 2 f_L [ sum_{x in L} p_x E_loc(x) - P_L <f>E ].
#
# We compare exact-vs-MC leaf gradient errors with production-feasible uncertainty
# proxies. The simplest mass-only proxy is sigma(P_L)|f_L|. We also test an
# energy-aware iid standard-error proxy for the leaf random variable
# Z_L = 2 f_L 1_L (E_loc - E_MC), using the sampled second moment.
# This is a diagnostic, not yet a new training regularizer.

const N = 8
const NSAMPLES = 512
const MAX_DEPTH = 4
const ETA_FIXED = 0.05
const SEED = 1234
const CHECKPOINTS = Dict(1.0 => Set([8,50]), 2.0 => Set([4,10]))

function allstates(N)
    X = Matrix{Int8}(undef, 1 << N, N)
    for s = 0:(1 << N)-1, i = 1:N
        X[s+1,i] = ((s >> (i-1)) & 1) == 1 ? Int8(1) : Int8(-1)
    end
    X
end

function leaf_index(t,x)
    i=1
    while true
        n=t.nodes[i]; n.isleaf && return i
        i = x[n.feature] <= 0 ? n.left : n.right
    end
end

function center_like_production(t,b)
    w=Float64.(b.counts)
    pred=[GBTQuantum.predict(t,@view b.states[j,:]) for j in axes(b.states,1)]
    mu=GBTQuantum.weighted_mean(pred,w)
    nd=copy(t.nodes)
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
rankcorr(x,y) = (std(ranks(x))==0 || std(ranks(y))==0) ? NaN : cor(ranks(x),ranks(y))

function exact_arrays(H,m,t,X)
    d=size(X,1); A=zeros(d); f=zeros(d); leaf=zeros(Int,d); el=zeros(d)
    for j=1:d
        x=@view X[j,:]
        A[j]=GBTQuantum.logamplitude(m,x)
        f[j]=GBTQuantum.predict(t,x)
        leaf[j]=leaf_index(t,x)
    end
    p=exp.(2 .* A .- maximum(2 .* A)); p ./= sum(p)
    for j=1:d
        x=@view X[j,:]
        el[j]=GBTQuantum.diagonal(H,x)
        for i=1:H.N
            k=((j-1) ⊻ (1 << (i-1)))+1
            el[j] -= H.h * exp(A[k]-A[j])
        end
    end
    Ef=sum(p .* f); E=sum(p .* el)
    g=2 * sum(p .* (f .- Ef) .* (el .- E))
    (;p,f,leaf,el,Ef,E,g)
end

function sample_arrays(t,b)
    w=Float64.(b.counts); W=sum(w); el=real.(b.local_energy)
    f=[GBTQuantum.predict(t,@view b.states[j,:]) for j in axes(b.states,1)]
    leaf=[leaf_index(t,@view b.states[j,:]) for j in axes(b.states,1)]
    Ef=sum(w .* f)/W; E=sum(w .* el)/W
    g=2 * sum(w .* (f .- Ef) .* (el .- E))/W
    (;w,W,el,f,leaf,Ef,E,g)
end

function audit(H,m,t,b,X,ratio,ep,shift)
    ex=exact_arrays(H,m,t,X); sm=sample_arrays(t,b)
    leaves=sort(unique(ex.leaf)); rows=NamedTuple[]

    @printf("\n%s\n","-"^136)
    @printf("J/h=%.2f epoch=%d | shift=% .4e | <f>MC=% .3e <f>exact=% .3e | gMC=% .6e gExact=% .6e dg=% .3e\n",
        ratio,ep,shift,sm.Ef,ex.Ef,sm.g,ex.g,sm.g-ex.g)

    for L in leaves
        ix=findall(==(L),ex.leaf); im=findall(==(L),sm.leaf)
        P=sum(ex.p[ix]); Phat=sum(sm.w[im])/sm.W; dP=Phat-P
        fL=ex.f[first(ix)]

        # Exact additive decomposition of g using global exact Ef,E.
        gex=2 * sum(ex.p[ix] .* (ex.f[ix] .- ex.Ef) .* (ex.el[ix] .- ex.E))

        # MC additive decomposition using global sampled Ef,E.
        gmc=isempty(im) ? 0.0 : 2 * sum(sm.w[im] .* (sm.f[im] .- sm.Ef) .* (sm.el[im] .- sm.E))/sm.W
        dg=gmc-gex

        sigP=sqrt(max(Phat*(1-Phat),0.0)/sm.W)
        risk_mass=sigP*abs(fL)
        oracle_mass=abs(dP)*abs(fL)

        # Production-feasible energy-aware iid SE of leaf contribution:
        # Z=2(f_L-Ef_MC) 1_L (E_loc-E_MC), Var(mean Z)=Var(Z)/M.
        if isempty(im)
            zmean=0.0; z2mean=0.0
        else
            a=2 * (fL-sm.Ef)
            zmean=sum(sm.w[im] .* (a .* (sm.el[im] .- sm.E)))/sm.W
            z2mean=sum(sm.w[im] .* ((a .* (sm.el[im] .- sm.E)) .^ 2))/sm.W
        end
        risk_energy=sqrt(max(z2mean-zmean^2,0.0)/sm.W)

        # A simpler scale separating mass and within-leaf energy:
        if isempty(im)
            mean_abs_dE=0.0; rms_dE=0.0
        else
            mean_abs_dE=sum(sm.w[im] .* abs.(sm.el[im] .- sm.E))/sum(sm.w[im])
            rms_dE=sqrt(sum(sm.w[im] .* (sm.el[im] .- sm.E).^2)/sum(sm.w[im]))
        end
        risk_mass_energy=sigP*abs(fL-sm.Ef)*rms_dE

        push!(rows,(ratio=ratio,epoch=ep,leaf=L,support=sum(sm.w[im]),f=fL,absf=abs(fL),
            born_mass=P,sample_mass=Phat,mass_error=dP,abs_mass_error=abs(dP),
            sigma_mass=sigP,g_exact_leaf=gex,g_mc_leaf=gmc,dg=dg,abs_dg=abs(dg),
            oracle_absdP_absf=oracle_mass,risk_sigma_absf=risk_mass,
            risk_energy_iid=risk_energy,risk_sigma_absf_rmsdE=risk_mass_energy,
            mean_abs_dE_leaf=mean_abs_dE,rms_dE_leaf=rms_dE,
            exact_mean_f=ex.Ef,mc_mean_f=sm.Ef,E_exact=ex.E,E_mc=sm.E,
            g_exact=ex.g,g_mc=sm.g))
    end

    @printf("Leaf sum check: exact=% .6e MC=% .6e dg=% .3e\n",
        sum(r.g_exact_leaf for r in rows),sum(r.g_mc_leaf for r in rows),sum(r.dg for r in rows))

    y=[r.abs_dg for r in rows]
    @printf("Spearman with |dg_L|:\n")
    @printf("  |f|=% .3f support=% .3f |dP|=% .3f oracle |dP||f|=% .3f\n",
        rankcorr([r.absf for r in rows],y),rankcorr([r.support for r in rows],y),
        rankcorr([r.abs_mass_error for r in rows],y),rankcorr([r.oracle_absdP_absf for r in rows],y))
    @printf("  sigma(P)|f|=% .3f energy-aware-SE=% .3f sigma(P)|f-Ef|*rms(dE)=% .3f\n",
        rankcorr([r.risk_sigma_absf for r in rows],y),rankcorr([r.risk_energy_iid for r in rows],y),
        rankcorr([r.risk_sigma_absf_rmsdE for r in rows],y))

    ord=sortperm(rows,by=r->r.abs_dg,rev=true)
    println(" Top leaves by |dg_L|:")
    for k in ord[1:min(8,length(ord))]
        r=rows[k]
        @printf("  leaf=%3d dg=% .3e W=%5.1f f=% .3e dP=% .2e sigma|f|=% .2e energySE=% .2e rmsdE=% .2e\n",
            r.leaf,r.dg,r.support,r.f,r.mass_error,r.risk_sigma_absf,r.risk_energy_iid,r.rms_dE_leaf)
    end

    for key in (:risk_sigma_absf,:risk_energy_iid,:risk_sigma_absf_rmsdE)
        ordp=sortperm(rows,by=r->getproperty(r,key),rev=true)
        top=ordp[1:min(4,length(ordp))]
        capture=sum(rows[k].abs_dg for k in top)/sum(y)
        @printf("Top-4 %-28s captures %5.1f%% of sum |dg_L|\n",String(key),100*capture)
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
    isempty(rows) && return
    ns=propertynames(rows[1])
    open(path,"w") do io
        println(io,join(string.(ns),','))
        for r in rows; println(io,join((getproperty(r,n) for n in ns),',')); end
    end
end

function main()
    println("="^136)
    println("TFIM LEAF GRADIENT-ERROR / UNCERTAINTY DIAGNOSTIC")
    println("N=$N Hilbert=$(1<<N), trajectory samples=$NSAMPLES, fixed eta=$ETA_FIXED")
    println("Production gauge <f>_MC=0; exact oracle evaluates same tree without re-centering")
    println("Tests mass-only sigma(P)|f| and energy-aware iid leaf-gradient uncertainty")
    println("="^136)
    X=allstates(N); rows=NamedTuple[]
    for ratio in sort(collect(keys(CHECKPOINTS)))
        H=GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true)
        append!(rows,train(H,ratio,X))
    end
    dir=joinpath(@__DIR__,"results"); mkpath(dir)
    path=joinpath(dir,"tfim_leaf_gradient_error_diagnostic.csv")
    writecsv(path,rows)
    println("\nResults written to $path")
end
end

if abspath(PROGRAM_FILE)==@__FILE__
    TFIMLeafGradientErrorDiagnostic.main()
end
