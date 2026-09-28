module TFIMLeafMassUncertaintyDiagnostic
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using GBTQuantum, Random, Statistics, Printf

# Tests whether sampling uncertainty in leaf Born masses explains curvature fragility.
# The production tree is gauge-fixed exactly as in training: <f>_MC = 0.
# Exact enumeration is diagnostic only and NEVER re-centers the tree.
#
# For each leaf L we compare the oracle error
#   |Delta P_L| = |P_hat_L - P_L|
# against uncertainty estimates based only on the sampled leaf-indicator time series.
# We report both iid multinomial sigma and an autocorrelation-corrected sigma using
# an integrated autocorrelation time (IPS truncation) for I[x_t in L].
# The curvature-risk proxies are sigma(P_L) * f_L^2.

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

scaled_tree(t,e)=GBTQuantum.RegressionTree([n.isleaf ? GBTQuantum.Node(n.feature,e*n.value,n.left,n.right,true) : n for n in t.nodes])

# Reconstruct an ordered length-NSAMPLES leaf series from the compressed VMC batch.
# b.states/b.counts contain unique states and multiplicities, not Markov-chain order,
# so autocorrelation cannot be recovered from b alone. We therefore evaluate leaf IDs
# directly on the live walker matrix S, whose rows are the current ensemble walkers.
function leaf_series(t,S)
    [leaf_index(t,@view S[j,:]) for j in axes(S,1)]
end

# Integrated autocorrelation time using Geyer's initial-positive-sequence idea on
# adjacent autocorrelation pairs. For an ensemble of walkers at one epoch there is no
# literal temporal ordering across rows; hence this quantity is labelled a sensitivity
# diagnostic, not a rigorous MCMC ESS. iid sigma remains the primary production-feasible
# baseline in this experiment.
function tau_int_ips(z::Vector{Float64})
    n=length(z); n < 4 && return 1.0
    mu=mean(z); v=sum((z .- mu).^2)/n
    v <= eps(Float64) && return 1.0
    maxlag=min(n÷2,100)
    rho=zeros(maxlag)
    for lag=1:maxlag
        rho[lag]=sum((z[1:n-lag] .- mu).*(z[1+lag:n] .- mu))/((n-lag)*v)
    end
    s=0.0; k=1
    while k <= maxlag
        pair=rho[k] + (k+1 <= maxlag ? rho[k+1] : 0.0)
        pair <= 0 && break
        s += pair; k += 2
    end
    max(1.0,1 + 2s)
end

function ranks(v)
    # Average ranks for ties.
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

function exact_leaf_data(m,t,X)
    d=size(X,1); A=zeros(d); f=zeros(d); leaf=zeros(Int,d)
    for j=1:d
        x=@view X[j,:]
        A[j]=GBTQuantum.logamplitude(m,x); f[j]=GBTQuantum.predict(t,x); leaf[j]=leaf_index(t,x)
    end
    p=exp.(2 .* A .- maximum(2 .* A)); p ./= sum(p)
    (;p,f,leaf,meanf=sum(p .* f))
end

function audit(m,t,b,S,X,ratio,ep,shift)
    ex=exact_leaf_data(m,t,X)
    w=Float64.(b.counts); W=sum(w)
    sm_f=[GBTQuantum.predict(t,@view b.states[j,:]) for j in axes(b.states,1)]
    sm_leaf=[leaf_index(t,@view b.states[j,:]) for j in axes(b.states,1)]
    mc_meanf=sum(w .* sm_f)/W
    live_leaf=leaf_series(t,S)
    leaves=sort(unique(ex.leaf)); rows=NamedTuple[]

    for L in leaves
        ix=findall(==(L),ex.leaf); im=findall(==(L),sm_leaf)
        P=sum(ex.p[ix]); Phat=sum(w[im])/W; dP=Phat-P; fL=ex.f[first(ix)]
        # iid multinomial standard error using only sampled mass
        sig_iid=sqrt(max(Phat*(1-Phat),0.0)/W)
        # Sensitivity ESS from the current walker leaf-indicator ensemble.
        z=Float64.(live_leaf .== L); tau=tau_int_ips(z); Meff=W/tau
        sig_eff=sqrt(max(Phat*(1-Phat),0.0)/max(Meff,1.0))
        risk_oracle=abs(dP)*fL^2
        risk_iid=sig_iid*fL^2
        risk_eff=sig_eff*fL^2
        z_iid=sig_iid>0 ? abs(dP)/sig_iid : (abs(dP)==0 ? 0.0 : Inf)
        z_eff=sig_eff>0 ? abs(dP)/sig_eff : (abs(dP)==0 ? 0.0 : Inf)
        push!(rows,(ratio=ratio,epoch=ep,leaf=L,support=sum(w[im]),f=fL,absf=abs(fL),f2=fL^2,born_mass=P,sample_mass=Phat,mass_error=dP,abs_mass_error=abs(dP),sigma_iid=sig_iid,tau_sensitivity=tau,Meff_sensitivity=Meff,sigma_eff_sensitivity=sig_eff,z_iid=z_iid,z_eff_sensitivity=z_eff,oracle_risk_absdP_f2=risk_oracle,risk_iid_sigma_f2=risk_iid,risk_eff_sigma_f2=risk_eff,exact_mean_f=ex.meanf,mc_mean_f=mc_meanf))
    end

    absdp=[r.abs_mass_error for r in rows]; oracle=[r.oracle_risk_absdP_f2 for r in rows]
    @printf("\n%s\n","-"^128)
    @printf("J/h=%.2f epoch=%d | gauge shift=% .4e | <f>MC=% .3e <f>exact=% .3e\n",ratio,ep,shift,mc_meanf,ex.meanf)
    @printf("Mass uncertainty: mean |dP|=% .4e  RMS |dP|=% .4e  mean sigma_iid=% .4e  mean sigma_eff=% .4e\n",mean(absdp),sqrt(mean(absdp.^2)),mean(r.sigma_iid for r in rows),mean(r.sigma_eff_sensitivity for r in rows))
    @printf("Spearman |dP| vs sigma: iid=% .3f  eff-sensitivity=% .3f\n",rankcorr(absdp,[r.sigma_iid for r in rows]),rankcorr(absdp,[r.sigma_eff_sensitivity for r in rows]))
    @printf("Spearman oracle |dP|f^2 vs estimated risk: iid=% .3f  eff-sensitivity=% .3f\n",rankcorr(oracle,[r.risk_iid_sigma_f2 for r in rows]),rankcorr(oracle,[r.risk_eff_sigma_f2 for r in rows]))
    @printf("Coverage |dP| <= k*sigma_iid: 1sigma=%5.1f%% 2sigma=%5.1f%% 3sigma=%5.1f%%\n",100*mean(r.z_iid<=1 for r in rows),100*mean(r.z_iid<=2 for r in rows),100*mean(r.z_iid<=3 for r in rows))
    @printf("Gauge uncertainty proxies: exact |<f>|=% .4e ; iid RSS sigma(<f>)=% .4e ; eff RSS=% .4e\n",abs(ex.meanf),sqrt(sum((r.f*r.sigma_iid)^2 for r in rows)),sqrt(sum((r.f*r.sigma_eff_sensitivity)^2 for r in rows)))

    ord=sortperm(rows,by=r->r.oracle_risk_absdP_f2,rev=true)
    println(" Top leaves by oracle |dP| f^2:")
    for k in ord[1:min(8,length(ord))]
        r=rows[k]
        @printf("  leaf=%3d W=%5.1f P=% .4f Phat=% .4f |dP|=% .3e f=% .3e |dP|f2=% .3e sigma_iid*f2=% .3e z=%4.2f tau*=%.2f\n",r.leaf,r.support,r.born_mass,r.sample_mass,r.abs_mass_error,r.f,r.oracle_risk_absdP_f2,r.risk_iid_sigma_f2,r.z_iid,r.tau_sensitivity)
    end
    rows
end

function train(H,ratio,X)
    cfg=GBTQuantum.TrainingConfig(nsamples=NSAMPLES,epochs=150,max_depth=MAX_DEPTH,eta=ETA_FIXED,burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=SEED,exact_diagnostics=false)
    rng=MersenneTwister(SEED); S=Matrix{Int8}(undef,NSAMPLES,H.N)
    for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
    m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); la=zeros(NSAMPLES)
    for _=1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,m,S,la); end
    out=NamedTuple[]; wanted=CHECKPOINTS[ratio]
    for ep=1:maximum(wanted)
        b=GBTQuantum.vmc_batch(H,m,S); y,_=GBTQuantum.make_targets(b)
        raw=GBTQuantum.grow_tree(b.states,y,b.counts;max_depth=cfg.max_depth,min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain)
        t,shift=center_like_production(raw,b)
        ep in wanted && append!(out,audit(m,t,b,S,X,ratio,ep,shift))
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
    println("="^128)
    println("TFIM LEAF BORN-MASS UNCERTAINTY / CURVATURE-RISK DIAGNOSTIC")
    println("N=$N Hilbert=$(1<<N), trajectory samples=$NSAMPLES, fixed eta=$ETA_FIXED")
    println("Production gauge: sample-weighted <f>_MC=0; exact enumeration never re-centers f")
    println("Primary risk proxy: sigma_iid(P_L) f_L^2, tested against oracle |P_hat_L-P_L| f_L^2")
    println("tau*/ESS columns are sensitivity diagnostics only; walker rows are not a temporal chain")
    println("="^128)
    X=allstates(N); rows=NamedTuple[]
    for r in sort(collect(keys(CHECKPOINTS)))
        H=GBTQuantum.TFIMHamiltonian(N;J=r,h=1.0,periodic=true)
        append!(rows,train(H,r,X))
    end
    dir=joinpath(@__DIR__,"results"); mkpath(dir)
    path=joinpath(dir,"tfim_leaf_mass_uncertainty_diagnostic.csv")
    writecsv(path,rows)
    println("\nResults written to $path")
end
end

if abspath(PROGRAM_FILE)==@__FILE__
    TFIMLeafMassUncertaintyDiagnostic.main()
end
