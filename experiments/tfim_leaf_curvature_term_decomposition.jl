module TFIMLeafCurvatureTermDecomposition
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using GBTQuantum, Random, Statistics, Printf

# Decomposes curvature error leaf by leaf into the mathematical terms
#   C1 = 2 <f^2 E_loc>
#   C2 = 2 <f H f>
#   C3 = -4 E <f^2>
#   C4 = -4 <f> g
# using the SAME production-gauge-fixed tree for exact and MC evaluation.
# Production gauge: subtract sample-weighted mean from every leaf, so <f>_MC=0.
# Exact enumeration is an oracle only; it NEVER re-centers the tree.

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

@inline flip(j,i) = ((j-1) ⊻ (1 << (i-1))) + 1

function leaf_index(t,x)
    i = 1
    while true
        n = t.nodes[i]
        n.isleaf && return i
        i = x[n.feature] <= 0 ? n.left : n.right
    end
end

function center_like_production(t,b)
    w = Float64.(b.counts)
    pred = [GBTQuantum.predict(t, @view b.states[j,:]) for j in axes(b.states,1)]
    mu = GBTQuantum.weighted_mean(pred,w)
    nd = copy(t.nodes)
    for i in eachindex(nd)
        n = nd[i]
        n.isleaf && (nd[i] = GBTQuantum.Node(n.feature,n.value-mu,n.left,n.right,true))
    end
    GBTQuantum.RegressionTree(nd), mu
end

scaled_tree(t,e) = GBTQuantum.RegressionTree([
    n.isleaf ? GBTQuantum.Node(n.feature,e*n.value,n.left,n.right,true) : n for n in t.nodes
])

function exact_arrays(H,m,t,X)
    d = size(X,1)
    A = zeros(d); f = zeros(d); leaf = zeros(Int,d)
    for j = 1:d
        x = @view X[j,:]
        A[j] = GBTQuantum.logamplitude(m,x)
        f[j] = GBTQuantum.predict(t,x)
        leaf[j] = leaf_index(t,x)
    end
    p = exp.(2 .* A .- maximum(2 .* A)); p ./= sum(p)
    el = zeros(d); hf = zeros(d)
    for j = 1:d
        x = @view X[j,:]
        el[j] = GBTQuantum.diagonal(H,x)
        hf[j] = GBTQuantum.diagonal(H,x) * f[j]
        for i = 1:H.N
            k = flip(j,i)
            r = exp(A[k]-A[j])
            el[j] -= H.h * r
            hf[j] -= H.h * r * f[k]
        end
    end
    Ef = sum(p .* f); E = sum(p .* el); Ef2 = sum(p .* f.^2)
    g = 2 * sum(p .* (f .- Ef) .* (el .- E))
    C1 = 2 * sum(p .* f.^2 .* el)
    C2 = 2 * sum(p .* f .* hf)
    C3 = -4 * E * Ef2
    C4 = -4 * Ef * g
    c = C1 + C2 + C3 + C4
    (;A,p,f,leaf,el,hf,Ef,E,Ef2,g,C1,C2,C3,C4,c)
end

function sample_arrays(H,m,t,b)
    n = size(b.states,1); w = Float64.(b.counts); W = sum(w)
    f = zeros(n); leaf = zeros(Int,n); hf = zeros(n); el = real.(b.local_energy)
    for j = 1:n
        x = @view b.states[j,:]
        f[j] = GBTQuantum.predict(t,x); leaf[j] = leaf_index(t,x)
        A = GBTQuantum.logamplitude(m,x)
        z = GBTQuantum.diagonal(H,x) * f[j]
        for i = 1:H.N
            x[i] = -x[i]
            z -= H.h * exp(GBTQuantum.logamplitude(m,x)-A) * GBTQuantum.predict(t,x)
            x[i] = -x[i]
        end
        hf[j] = z
    end
    Ef = sum(w .* f)/W; E = sum(w .* el)/W; Ef2 = sum(w .* f.^2)/W
    g = 2 * sum(w .* (f .- Ef) .* (el .- E))/W
    C1 = 2 * sum(w .* f.^2 .* el)/W
    C2 = 2 * sum(w .* f .* hf)/W
    C3 = -4 * E * Ef2
    C4 = -4 * Ef * g
    c = C1 + C2 + C3 + C4
    (;w,W,f,leaf,hf,el,Ef,E,Ef2,g,C1,C2,C3,C4,c)
end

function neighbor_coverage(H,b,leaf_exact)
    represented = Set{Int}()
    for j in axes(b.states,1)
        s = 0
        for i = 1:H.N
            b.states[j,i] > 0 && (s |= 1 << (i-1))
        end
        push!(represented,s+1)
    end
    out = Dict{Int,Float64}()
    for L in unique(leaf_exact)
        roots = findall(==(L),leaf_exact); total = length(roots)*H.N; hit = 0
        for j in roots, i = 1:H.N
            flip(j,i) in represented && (hit += 1)
        end
        out[L] = total == 0 ? NaN : hit/total
    end
    out
end

function rankcorr(x,y)
    # simple Spearman via rank positions; sufficient here because leaf predictors are mostly distinct
    function ranks(v)
        o = sortperm(v); r = zeros(Float64,length(v))
        for (k,i) in enumerate(o); r[i] = k; end
        r
    end
    rx=ranks(x); ry=ranks(y); sx=std(rx); sy=std(ry)
    (sx==0 || sy==0) ? NaN : cor(rx,ry)
end

function audit(H,m,t,b,X,ratio,ep,shift)
    ex = exact_arrays(H,m,t,X); sm = sample_arrays(H,m,t,b); cov = neighbor_coverage(H,b,ex.leaf)
    @printf("\n%s\n", "-"^132)
    @printf("J/h=%.2f epoch=%d | gauge shift=% .4e | <f>MC=% .3e <f>exact=% .3e\n",ratio,ep,shift,sm.Ef,ex.Ef)
    @printf("exact: C1=% .5e C2=% .5e C3=% .5e C4=% .5e => c=% .5e\n",ex.C1,ex.C2,ex.C3,ex.C4,ex.c)
    @printf("MC:    C1=% .5e C2=% .5e C3=% .5e C4=% .5e => c=% .5e\n",sm.C1,sm.C2,sm.C3,sm.C4,sm.c)
    @printf("error: dC1=% .5e dC2=% .5e dC3=% .5e dC4=% .5e => dc=% .5e\n",sm.C1-ex.C1,sm.C2-ex.C2,sm.C3-ex.C3,sm.C4-ex.C4,sm.c-ex.c)

    rows = NamedTuple[]; leaves=sort(unique(ex.leaf))
    for L in leaves
        ix=findall(==(L),ex.leaf); im=findall(==(L),sm.leaf)
        P=sum(ex.p[ix]); PM=sum(sm.w[im])/sm.W; fL=ex.f[first(ix)]
        e1=2*sum(ex.p[ix] .* ex.f[ix].^2 .* ex.el[ix])
        e2=2*sum(ex.p[ix] .* ex.f[ix] .* ex.hf[ix])
        e3=-4*ex.E*sum(ex.p[ix] .* ex.f[ix].^2)
        e4=-4*ex.Ef*ex.g*P
        if isempty(im)
            m1=0.0; m2=0.0; m3=0.0; m4=0.0
        else
            m1=2*sum(sm.w[im] .* sm.f[im].^2 .* sm.el[im])/sm.W
            m2=2*sum(sm.w[im] .* sm.f[im] .* sm.hf[im])/sm.W
            m3=-4*sm.E*sum(sm.w[im] .* sm.f[im].^2)/sm.W
            m4=-4*sm.Ef*sm.g*PM
        end
        dc1=m1-e1; dc2=m2-e2; dc3=m3-e3; dc4=m4-e4; dc=dc1+dc2+dc3+dc4
        W=sum(sm.w[im]); dP=PM-P
        pred_mass=abs(dP)*fL^2
        pred_invW=W>0 ? fL^2/W : Inf
        pred_conf=W>0 ? abs(fL)/sqrt(W) : Inf
        push!(rows,(ratio=ratio,epoch=ep,leaf=L,support=W,f=fL,absf=abs(fL),born_mass=P,sample_mass=PM,mass_error=dP,abs_mass_error=abs(dP),neighbor_coverage=cov[L],C1_exact=e1,C1_mc=m1,dC1=dc1,C2_exact=e2,C2_mc=m2,dC2=dc2,C3_exact=e3,C3_mc=m3,dC3=dc3,C4_exact=e4,C4_mc=m4,dC4=dc4,c_exact_leaf=e1+e2+e3+e4,c_mc_leaf=m1+m2+m3+m4,dc=dc,abs_dc=abs(dc),predictor_absdP_f2=pred_mass,predictor_f2_over_W=pred_invW,predictor_absf_over_sqrtW=pred_conf,exact_mean_f=ex.Ef,mc_mean_f=sm.Ef,c_exact=ex.c,c_mc=sm.c))
    end

    @printf("leaf sums errors: dC1=% .5e dC2=% .5e dC3=% .5e dC4=% .5e dc=% .5e\n",sum(r.dC1 for r in rows),sum(r.dC2 for r in rows),sum(r.dC3 for r in rows),sum(r.dC4 for r in rows),sum(r.dc for r in rows))
    absdc=[r.abs_dc for r in rows]
    println("Spearman with |dc_L|:")
    @printf("  |f|=% .3f support=% .3f |dP|=% .3f neigh=% .3f |dP|f^2=% .3f f^2/W=% .3f |f|/sqrt(W)=% .3f\n",
        rankcorr([r.absf for r in rows],absdc),rankcorr([r.support for r in rows],absdc),rankcorr([r.abs_mass_error for r in rows],absdc),rankcorr([r.neighbor_coverage for r in rows],absdc),rankcorr([r.predictor_absdP_f2 for r in rows],absdc),rankcorr([r.predictor_f2_over_W for r in rows],absdc),rankcorr([r.predictor_absf_over_sqrtW for r in rows],absdc))

    ord=sortperm(rows,by=r->r.abs_dc,rev=true)
    println(" Top leaves by |dc_L| (term errors shown separately):")
    for k in ord[1:min(8,length(ord))]
        r=rows[k]
        @printf("  leaf=%3d dc=% .3e [dC1=% .2e dC2=% .2e dC3=% .2e dC4=% .2e] W=%5.1f f=% .3e dP=% .2e neigh=%5.1f%%\n",r.leaf,r.dc,r.dC1,r.dC2,r.dC3,r.dC4,r.support,r.f,r.mass_error,100*r.neighbor_coverage)
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
    println("="^132)
    println("TFIM LEAF CURVATURE TERM-DECOMPOSITION DIAGNOSTIC")
    println("N=$N Hilbert=$(1<<N), trajectory samples=$NSAMPLES, fixed eta=$ETA_FIXED")
    println("Production gauge <f>_MC=0; exact oracle does not re-center tree")
    println("c = C1 + C2 + C3 + C4 = 2<f^2 Eloc> + 2<f Hf> - 4E<f^2> - 4<f>g")
    println("Predictors: |dP|f^2, f^2/W, |f|/sqrt(W), plus support and Hamiltonian-neighbor coverage")
    println("="^132)
    X=allstates(N); rows=NamedTuple[]
    for r in sort(collect(keys(CHECKPOINTS)))
        H=GBTQuantum.TFIMHamiltonian(N;J=r,h=1.0,periodic=true)
        append!(rows,train(H,r,X))
    end
    dir=joinpath(@__DIR__,"results"); mkpath(dir)
    path=joinpath(dir,"tfim_leaf_curvature_term_decomposition.csv")
    writecsv(path,rows); println("\nResults written to $path")
end
end

if abspath(PROGRAM_FILE)==@__FILE__
    TFIMLeafCurvatureTermDecomposition.main()
end
