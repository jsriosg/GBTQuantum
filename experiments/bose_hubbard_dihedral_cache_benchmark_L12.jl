module BoseHubbardDihedralCacheBenchmark

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Random
using Statistics
using LinearAlgebra
using Printf

# Size-scaling study at unit filling. U/J=1 and 6 probe the two established
# Bose-Hubbard regimes without assigning a finite-size critical point.
const L = Ref(12)
const NBOS = Ref(12)
const J = 1.0
const U_VALUES = [1.0]
const LAMBDA = 4.0
const SEEDS = [1234]

# Frozen Bose-Hubbard symmetry-aware settings established by the L=8 studies.
const SAMPLE_SIZES = [1024]
const EPOCHS = 100
const CHECKPOINTS = Set([100])
const DEPTHS = [6]
const SYMMETRY_MODES = [:dihedral_full_cache, :dihedral_uncached, :dihedral_bounded_cache]
const ETA = 0.05
const BURN_IN_SWEEPS = 50
const SWEEPS_PER_EPOCH = 2

function canonical_translation(x)
    Lx=length(x)
    best=collect(x)
    cand=similar(best)
    for shift in 1:(Lx-1)
        @inbounds for i in 1:Lx
            cand[i]=x[mod1(i+shift,Lx)]
        end
        if Tuple(cand) < Tuple(best)
            copyto!(best,cand)
        end
    end
    return best
end

const DIHEDRAL_CACHE = Dict{Tuple{Vararg{Int16}},Vector{Int16}}()
const CACHE_HITS = Ref(0)
const CACHE_MISSES = Ref(0)
const CACHE_EVICTIONS = Ref(0)
const CANON_CALLS = Ref(0)
const BOUNDED_CACHE_MAX = 50_000

function canonical_dihedral_uncached(x)
    # D_L orbit = all cyclic translations of x and of one reflected copy.
    best=canonical_translation(x)
    reflected=reverse(collect(x))
    reflected_best=canonical_translation(reflected)
    return Tuple(reflected_best) < Tuple(best) ? reflected_best : best
end

function dihedral_orbit(x)
    Lx=length(x)
    orbit=Vector{Vector{Int16}}()
    for base in (Int16.(x), reverse(Int16.(x)))
        for shift in 0:(Lx-1)
            y=Vector{Int16}(undef,Lx)
            @inbounds for i in 1:Lx
                y[i]=base[mod1(i+shift,Lx)]
            end
            push!(orbit,y)
        end
    end
    return orbit
end

function canonical_dihedral_full_cache(x)
    CANON_CALLS[] += 1
    key=Tuple(Int16.(x))
    if haskey(DIHEDRAL_CACHE,key)
        CACHE_HITS[] += 1
        return DIHEDRAL_CACHE[key]
    end
    CACHE_MISSES[] += 1
    orbit=dihedral_orbit(x)
    best=orbit[1]
    for y in orbit[2:end]
        Tuple(y) < Tuple(best) && (best=y)
    end
    rep=copy(best)
    # A miss discovers the complete D_L orbit, so cache every equivalent state.
    for y in orbit
        DIHEDRAL_CACHE[Tuple(y)]=rep
    end
    return rep
end

function canonical_dihedral_uncached_counted(x)
    CANON_CALLS[] += 1
    CACHE_MISSES[] += 1
    return Int16.(canonical_dihedral_uncached(x))
end

function canonical_dihedral_bounded(x)
    CANON_CALLS[] += 1
    key=Tuple(Int16.(x))
    if haskey(DIHEDRAL_CACHE,key)
        CACHE_HITS[] += 1
        return DIHEDRAL_CACHE[key]
    end
    CACHE_MISSES[] += 1
    orbit=dihedral_orbit(x)
    best=orbit[1]
    for y in orbit[2:end]
        Tuple(y) < Tuple(best) && (best=y)
    end
    rep=copy(best)
    # Keep memory bounded. If inserting the complete orbit would exceed the
    # budget, clear the cache and begin a fresh working set.
    if length(DIHEDRAL_CACHE) + length(orbit) > BOUNDED_CACHE_MAX
        CACHE_EVICTIONS[] += length(DIHEDRAL_CACHE)
        empty!(DIHEDRAL_CACHE)
    end
    for y in orbit
        DIHEDRAL_CACHE[Tuple(y)]=rep
    end
    return rep
end

function reset_dihedral_cache!()
    empty!(DIHEDRAL_CACHE)
    CACHE_HITS[]=0; CACHE_MISSES[]=0; CACHE_EVICTIONS[]=0; CANON_CALLS[]=0
end

function feature_state(x,mode)
    mode === :raw && return x
    mode === :translation_canonical && return canonical_translation(x)
    mode === :dihedral_full_cache && return canonical_dihedral_full_cache(x)
    mode === :dihedral_uncached && return canonical_dihedral_uncached_counted(x)
    mode === :dihedral_bounded_cache && return canonical_dihedral_bounded(x)
    error("unknown symmetry mode $mode")
end

@inline function denom(W, M, lambda)
    p = clamp(W/M, 0.0, 1.0)
    sigma = sqrt(max(p*(1-p),0.0)/M)
    return W + 2lambda*M*sigma
end

# Same uncertainty-regularized weighted-LS objective used for TFIM, but with
# ordinary ordered numerical CART thresholds instead of the binary x_f < 0 split.
function grow_uncertainty_tree_numeric(X, y, w;
                                       max_depth=4,
                                       min_weight=1.0,
                                       min_gain=0.0,
                                       lambda=0.0)
    nobs,nfeatures = size(X)
    Mtot = sum(Float64.(w))
    nodes = GBTQuantum.Node[]
    idx0 = collect(1:nobs)
    S0 = sum(Float64(w[i])*Float64(y[i]) for i=1:nobs)

    function build(idx, depth, W, S)
        D = denom(W,Mtot,lambda)
        pos = Int32(length(nodes)+1)
        push!(nodes, GBTQuantum.Node(0,0.0,S/D,0,0,true))
        (depth>=max_depth || length(idx)<=1 || W<2min_weight) && return pos

        parent = S*S/D
        best = min_gain
        bf = 0
        bt = 0.0
        bWL = 0.0
        bSL = 0.0

        for f in 1:nfeatures
            vals = sort!(unique(Float64(X[i,f]) for i in idx))
            length(vals)<=1 && continue
            for q in 1:(length(vals)-1)
                threshold=(vals[q]+vals[q+1])/2
                WL=0.0; SL=0.0
                @inbounds for i in idx
                    if X[i,f] < threshold
                        wi=Float64(w[i])
                        WL += wi
                        SL += wi*Float64(y[i])
                    end
                end
                WR=W-WL
                (WL<min_weight || WR<min_weight) && continue
                SR=S-SL
                gain=SL*SL/denom(WL,Mtot,lambda)+
                     SR*SR/denom(WR,Mtot,lambda)-parent
                if gain>best
                    best=gain; bf=f; bt=threshold; bWL=WL; bSL=SL
                end
            end
        end

        bf==0 && return pos
        li=Int[]; ri=Int[]
        @inbounds for i in idx
            X[i,bf] < bt ? push!(li,i) : push!(ri,i)
        end
        left=build(li,depth+1,bWL,bSL)
        right=build(ri,depth+1,W-bWL,S-bSL)
        nodes[pos]=GBTQuantum.Node(Int32(bf),bt,0.0,left,right,false)
        return pos
    end

    build(idx0,0,Mtot,S0)
    return GBTQuantum.RegressionTree(nodes)
end

function center_tree(t, states, counts)
    w=Float64.(counts)
    p=[GBTQuantum.predict(t,@view states[j,:]) for j in axes(states,1)]
    mu=GBTQuantum.weighted_mean(p,w)
    nd=copy(t.nodes)
    for i in eachindex(nd)
        n=nd[i]
        if n.isleaf
            nd[i]=GBTQuantum.Node(n.feature,n.threshold,n.value-mu,n.left,n.right,true)
        end
    end
    return GBTQuantum.RegressionTree(nd),mu
end

function scaled_tree(t, eta)
    GBTQuantum.RegressionTree([
        n.isleaf ?
        GBTQuantum.Node(n.feature,n.threshold,eta*n.value,n.left,n.right,true) : n
        for n in t.nodes
    ])
end

function compress_bh(samples::Matrix{Int16})
    Mrows,Lx=size(samples)
    map=Dict{Tuple{Vararg{Int16}},Int}()
    states=Matrix{Int16}(undef,Mrows,Lx)
    counts=Vector{Int}(undef,Mrows)
    K=0
    @inbounds for r in 1:Mrows
        x=@view samples[r,:]
        key=Tuple(x)
        j=get(map,key,0)
        if j==0
            K+=1; map[key]=K
            copyto!(@view(states[K,:]),x)
            counts[K]=1
        else
            counts[j]+=1
        end
    end
    return states[1:K,:],counts[1:K]
end

function bh_batch(H,m,samples)
    states,counts=compress_bh(samples)
    K=size(states,1)
    eloc=Vector{ComplexF64}(undef,K)
    Mtot=sum(counts)
    Esum=0.0+0.0im
    @inbounds for j in 1:K
        e=GBTQuantum.local_energy!(H,m,@view states[j,:])
        eloc[j]=e
        Esum += counts[j]*e
    end
    E=Esum/Mtot
    y=Vector{Float64}(undef,K)
    @inbounds for j in 1:K
        y[j]=-real(eloc[j]-E)
    end
    return (states=states,counts=counts,local_energy=eloc,energy=E,targets=y)
end


@inline function model_logamp(m,x,mode)
    z=feature_state(x,mode)
    return m.logamp.bias + sum((GBTQuantum.predict(t,z) for t in m.logamp.trees); init=0.0)
end

function local_energy_mode(H,m,n,mode)
    mode === :raw && return GBTQuantum.local_energy!(H,m,n)
    A0=model_logamp(m,n,mode)
    z=ComplexF64(GBTQuantum.diagonal(H,n))
    @inbounds for (i,j) in GBTQuantum.bh_bonds(H)
        ni=Int(n[i]); nj=Int(n[j])
        if ni>0
            n[i]-=1; n[j]+=1
            dA=model_logamp(m,n,mode)-A0
            z -= H.J*sqrt(ni*(nj+1.0))*exp(dA)
            n[j]-=1; n[i]+=1
        end
        if nj>0
            n[j]-=1; n[i]+=1
            dA=model_logamp(m,n,mode)-A0
            z -= H.J*sqrt(nj*(ni+1.0))*exp(dA)
            n[i]-=1; n[j]+=1
        end
    end
    return z
end

function compress_mode(samples,mode)
    Mrows,Lx=size(samples)
    map=Dict{Tuple{Vararg{Int16}},Int}()
    states=Matrix{Int16}(undef,Mrows,Lx)
    counts=zeros(Int,Mrows); K=0
    for r in 1:Mrows
        z=Int16.(feature_state(@view(samples[r,:]),mode))
        key=Tuple(z); j=get(map,key,0)
        if j==0
            K+=1; map[key]=K; states[K,:].=z; counts[K]=1
        else
            counts[j]+=1
        end
    end
    return states[1:K,:],counts[1:K]
end

function bh_batch_mode(H,m,samples,mode)
    states,counts=compress_mode(samples,mode)
    K=size(states,1); eloc=Vector{ComplexF64}(undef,K)
    Esum=0.0+0.0im
    for j in 1:K
        x=copy(@view(states[j,:]))
        e=local_energy_mode(H,m,x,mode)
        eloc[j]=e; Esum += counts[j]*e
    end
    E=Esum/sum(counts)
    y=[-real(eloc[j]-E) for j in 1:K]
    return (states=states,counts=counts,local_energy=eloc,energy=E,targets=y)
end

function bh_sweep_mode!(rng,m,H,samples,logamps,mode)
    mode === :raw && return GBTQuantum.bh_sweep!(rng,m,H,samples,logamps)
    bonds=GBTQuantum.bh_bonds(H); accepted=0
    for r in axes(samples,1)
        n=@view samples[r,:]; A=logamps[r]
        for _ in 1:H.L
            i,j=bonds[rand(rng,eachindex(bonds))]
            src,dst=rand(rng,Bool) ? (i,j) : (j,i)
            n[src]==0 && continue
            n[src]-=1; n[dst]+=1
            Ap=model_logamp(m,n,mode)
            if log(rand(rng)) < min(0.0,2*(Ap-A))
                A=Ap; accepted+=1
            else
                n[dst]-=1; n[src]+=1
            end
        end
        logamps[r]=A
    end
    return accepted/(size(samples,1)*H.L)
end

function initial_samples(rng,M)
    S=Matrix{Int16}(undef,M,L[])
    # Random weak compositions generated by placing NBOS indistinguishable
    # particles independently; this is only a starting distribution and is
    # followed by burn-in.
    for r in 1:M
        fill!(@view(S[r,:]),0)
        for _ in 1:NBOS[]
            S[r,rand(rng,1:L[])] += 1
        end
    end
    return S
end

function exact_model_stats(m,H,basis)
    d=size(basis,1)
    A=[GBTQuantum.logamplitude(m,@view basis[r,:]) for r in 1:d]
    lw=2 .* A
    lw .-= maximum(lw)
    w=exp.(lw); Z=sum(w); p=w./Z
    el=[real(GBTQuantum.local_energy!(H,m,@view basis[r,:])) for r in 1:d]
    E=sum(p.*el)
    var=sum(p.*(el .- E).^2)

    nvar=0.0
    for r in 1:d
        x=@view basis[r,:]
        sitevar=sum((Float64(x[i])-NBOS[]/L[])^2 for i in 1:L)/L
        nvar += p[r]*sitevar
    end

    # Kinetic/hopping expectation follows from E = K + interaction - mu*N.
    interaction=0.0
    for r in 1:d
        x=@view basis[r,:]
        dint=0.5*H.U*sum(Float64(x[i])*(Float64(x[i])-1) for i in 1:L)
        interaction += p[r]*dint
    end
    kinetic=E-interaction+H.mu*NBOS
    return (energy=E,variance=var,number_variance=nvar,kinetic=kinetic)
end

function exact_model_stats_sym(m,H,basis,mode)
    d=size(basis,1)
    A=[sum(GBTQuantum.predict(t,feature_state(@view(basis[r,:]),mode)) for t in m.logamp.trees)+m.logamp.bias for r in 1:d]
    lw=2 .* A; lw .-= maximum(lw); w=exp.(lw); p=w./sum(w)
    # Evaluate local energy explicitly using the symmetry-aware amplitude ratio.
    el=zeros(Float64,d)
    bonds=GBTQuantum.bh_bonds(H)
    for r in 1:d
        x=collect(@view basis[r,:]); ax=A[r]
        e=GBTQuantum.diagonal(H,x)
        for (i,j) in bonds
            if x[i]>0
                y=copy(x); y[i]-=1; y[j]+=1
                ay=sum(GBTQuantum.predict(t,feature_state(y,mode)) for t in m.logamp.trees)+m.logamp.bias
                e -= H.J*sqrt(Float64(x[i])*(x[j]+1))*exp(ay-ax)
            end
            if x[j]>0
                y=copy(x); y[j]-=1; y[i]+=1
                ay=sum(GBTQuantum.predict(t,feature_state(y,mode)) for t in m.logamp.trees)+m.logamp.bias
                e -= H.J*sqrt(Float64(x[j])*(x[i]+1))*exp(ay-ax)
            end
        end
        el[r]=e
    end
    E=sum(p.*el); var=sum(p.*(el.-E).^2)
    nvar=sum(p[r]*sum((Float64(basis[r,i])-NBOS[]/L[])^2 for i in 1:L)/L for r in 1:d)
    interaction=sum(p[r]*(0.5*H.U*sum(Float64(basis[r,i])*(Float64(basis[r,i])-1) for i in 1:L)) for r in 1:d)
    kinetic=E-interaction+H.mu*NBOS
    return (energy=E,variance=var,number_variance=nvar,kinetic=kinetic)
end

function exact_ground_stats(H)
    gs=GBTQuantum.exact_ground_state(H,NBOS[])
    psi=gs.state
    p=abs2.(psi)
    basis=gs.basis

    nvar=0.0
    interaction=0.0
    for r in axes(basis,1)
        x=@view basis[r,:]
        nvar += p[r]*sum((Float64(x[i])-NBOS[]/L[])^2 for i in 1:L)/L
        interaction += p[r]*(0.5*H.U*sum(Float64(x[i])*(Float64(x[i])-1) for i in 1:L))
    end
    kinetic=gs.energy-interaction+H.mu*NBOS
    return (energy=gs.energy,number_variance=nvar,kinetic=kinetic,basis=basis)
end

function tree_update_stats(t)
    vals=[n.value for n in t.nodes if n.isleaf]
    return (maxabs=maximum(abs.(vals)), minval=minimum(vals), maxval=maximum(vals))
end

function train(H,lambda,eta,depth,seed,M,gs,rows,mode)
    rng=MersenneTwister(seed)
    S=initial_samples(rng,M)
    m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    la=[model_logamp(m,@view(S[r,:]),mode) for r in 1:M]

    for _ in 1:BURN_IN_SWEEPS
        bh_sweep_mode!(rng,m,H,S,la,mode)
    end

    peak_raw_leaf=0.0
    peak_scaled_leaf=0.0
    maxabs_logamp=maximum(abs.(la))

    for ep in 1:EPOCHS
        b=bh_batch_mode(H,m,S,mode)
        if !isfinite(real(b.energy)) || any(z->!isfinite(real(z)) || !isfinite(imag(z)),b.local_energy)
            return (model=m,finite=false,failure_epoch=ep,failure_stage="batch",
                    peak_raw_leaf=peak_raw_leaf,peak_scaled_leaf=peak_scaled_leaf,
                    maxabs_logamp=maxabs_logamp)
        end

        Xfit=b.states  # already canonical and orbit-compressed in symmetry mode
        raw=grow_uncertainty_tree_numeric(Xfit,b.targets,b.counts;
            max_depth=depth,min_weight=1.0,min_gain=0.0,lambda=lambda)
        t,_=center_tree(raw,Xfit,b.counts)
        st=tree_update_stats(t)
        peak_raw_leaf=max(peak_raw_leaf,st.maxabs)
        peak_scaled_leaf=max(peak_scaled_leaf,eta*st.maxabs)

        push!(m.logamp.trees,scaled_tree(t,eta))
        @inbounds for r in 1:M
            la[r]=model_logamp(m,@view(S[r,:]),mode)
        end
        finite_after_update=all(isfinite,la)
        finite_after_update && (maxabs_logamp=max(maxabs_logamp,maximum(abs.(la))))
        if !finite_after_update
            @printf("  FAILURE U/J=%.1f lambda=%.1f epoch=%d stage=tree_update raw|maxleaf|=%.3e scaled|maxleaf|=%.3e\n",
                    H.U/H.J,lambda,ep,st.maxabs,eta*st.maxabs)
            return (model=m,finite=false,failure_epoch=ep,failure_stage="tree_update",
                    peak_raw_leaf=peak_raw_leaf,peak_scaled_leaf=peak_scaled_leaf,
                    maxabs_logamp=maxabs_logamp)
        end

        for _ in 1:SWEEPS_PER_EPOCH
            bh_sweep_mode!(rng,m,H,S,la,mode)
        end
        if any(!isfinite,la)
            @printf("  FAILURE U/J=%.1f lambda=%.1f epoch=%d stage=sampling raw|maxleaf|=%.3e scaled|maxleaf|=%.3e\n",
                    H.U/H.J,lambda,ep,st.maxabs,eta*st.maxabs)
            return (model=m,finite=false,failure_epoch=ep,failure_stage="sampling",
                    peak_raw_leaf=peak_raw_leaf,peak_scaled_leaf=peak_scaled_leaf,
                    maxabs_logamp=maxabs_logamp)
        end
        maxabs_logamp=max(maxabs_logamp,maximum(abs.(la)))
        if ep in CHECKPOINTS
            s=exact_model_stats_sym(m,H,gs.basis,mode)
            push!(rows,(L=L[],Nbos=NBOS[],hilbert=size(gs.basis,1),U_over_J=H.U/H.J,M=M,
                lambda=lambda,eta=eta,depth=depth,symmetry=String(mode),seed=seed,epoch=ep,finite=true,
                peak_raw_leaf=peak_raw_leaf,peak_scaled_leaf=peak_scaled_leaf,maxabs_logamp=maxabs_logamp,
                Egs=gs.energy,E_final=s.energy,energy_error=s.energy-gs.energy,
                number_variance_exact=gs.number_variance,number_variance_model=s.number_variance,
                number_variance_abs_error=abs(s.number_variance-gs.number_variance),
                kinetic_exact=gs.kinetic,kinetic_model=s.kinetic,
                kinetic_abs_error=abs(s.kinetic-gs.kinetic),exact_model_variance=s.variance))
        end
    end
    return (model=m,finite=true,failure_epoch=0,failure_stage="none",
            peak_raw_leaf=peak_raw_leaf,peak_scaled_leaf=peak_scaled_leaf,
            maxabs_logamp=maxabs_logamp)
end

function writecsv(path,rows)
    ns=propertynames(rows[1])
    open(path,"w") do io
        println(io,join(string.(ns),','))
        for r in rows
            println(io,join((getproperty(r,n) for n in ns),','))
        end
    end
end

function mc_model_stats(m,H,mode; M_eval=1024, burn=20, sweeps=2, seed=987654)
    rng=MersenneTwister(seed)
    S=initial_samples(rng,M_eval)
    la=[model_logamp(m,@view(S[r,:]),mode) for r in 1:M_eval]
    for _ in 1:burn
        bh_sweep_mode!(rng,m,H,S,la,mode)
    end
    es=Float64[]; nvars=Float64[]; interactions=Float64[]
    for _ in 1:sweeps
        bh_sweep_mode!(rng,m,H,S,la,mode)
        for r in 1:M_eval
            x=copy(@view S[r,:])
            e=real(local_energy_mode(H,m,x,mode))
            push!(es,e)
            push!(nvars,sum((Float64(x[i])-1.0)^2 for i in 1:L[])/L[])
            push!(interactions,0.5*H.U*sum(Float64(x[i])*(Float64(x[i])-1) for i in 1:L[]))
        end
    end
    E=mean(es); varE=mean((es .- E).^2)
    nvar=mean(nvars); kinetic=E-mean(interactions)+H.mu*NBOS[]
    return (energy=E,variance=varE,number_variance=nvar,kinetic=kinetic)
end

function train_scaling(H,lambda,eta,depth,seed,M,rows,mode)
    rng=MersenneTwister(seed)
    S=initial_samples(rng,M)
    m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    la=[model_logamp(m,@view(S[r,:]),mode) for r in 1:M]
    for _ in 1:BURN_IN_SWEEPS
        bh_sweep_mode!(rng,m,H,S,la,mode)
    end
    peak_raw_leaf=0.0; maxabs_logamp=maximum(abs.(la))
    for ep in 1:EPOCHS
        b=bh_batch_mode(H,m,S,mode)
        if !isfinite(real(b.energy)) || any(z->!isfinite(real(z)) || !isfinite(imag(z)),b.local_energy)
            return (finite=false,failure_epoch=ep,failure_stage="batch",peak_raw_leaf=peak_raw_leaf,maxabs_logamp=maxabs_logamp)
        end
        raw=grow_uncertainty_tree_numeric(b.states,b.targets,b.counts;max_depth=depth,min_weight=1.0,min_gain=0.0,lambda=lambda)
        t,_=center_tree(raw,b.states,b.counts)
        st=tree_update_stats(t); peak_raw_leaf=max(peak_raw_leaf,st.maxabs)
        push!(m.logamp.trees,scaled_tree(t,eta))
        @inbounds for r in 1:M
            la[r]=model_logamp(m,@view(S[r,:]),mode)
        end
        all(isfinite,la) || return (finite=false,failure_epoch=ep,failure_stage="tree_update",peak_raw_leaf=peak_raw_leaf,maxabs_logamp=maxabs_logamp)
        for _ in 1:SWEEPS_PER_EPOCH
            bh_sweep_mode!(rng,m,H,S,la,mode)
        end
        all(isfinite,la) || return (finite=false,failure_epoch=ep,failure_stage="sampling",peak_raw_leaf=peak_raw_leaf,maxabs_logamp=maxabs_logamp)
        maxabs_logamp=max(maxabs_logamp,maximum(abs.(la)))
        if ep in CHECKPOINTS
            s=mc_model_stats(m,H,mode)
            hit=CACHE_HITS[]; miss=CACHE_MISSES[]
            push!(rows,(L=L[],Nbos=NBOS[],hilbert=binomial(NBOS[]+L[]-1,NBOS[]),U_over_J=H.U/H.J,M=M,
                lambda=lambda,eta=eta,depth=depth,symmetry=String(mode),seed=seed,epoch=ep,finite=true,
                peak_raw_leaf=peak_raw_leaf,maxabs_logamp=maxabs_logamp,E_per_site=s.energy/L[],
                Eloc_variance=s.variance,Eloc_variance_per_site=s.variance/L[],
                Eloc_variance_per_site2=s.variance/(L[]^2),number_variance=s.number_variance,
                kinetic_per_site=s.kinetic/L[],cache_entries=length(DIHEDRAL_CACHE),cache_hits=hit,
                cache_misses=miss,cache_hit_rate=(hit+miss)==0 ? 0.0 : hit/(hit+miss)))
        end
    end
    return (finite=true,failure_epoch=0,failure_stage="none",peak_raw_leaf=peak_raw_leaf,maxabs_logamp=maxabs_logamp)
end

function main()
    println("="^116)
    println("BOSE-HUBBARD D_L CANONICALIZATION / CACHE BENCHMARK")
    println("L=Nbos=12, U/J=1, M=1024, depth=6, lambda=$LAMBDA, eta=$ETA, epochs=$EPOCHS, seed=1234")
    println("Modes=$SYMMETRY_MODES; bounded cache maximum=$BOUNDED_CACHE_MAX states")
    println("="^116)
    rows=NamedTuple[]
    H=GBTQuantum.BoseHubbardHamiltonian(L[];J=J,U=1.0,mu=0.0,periodic=true)
    for mode in SYMMETRY_MODES
        reset_dihedral_cache!()
        GC.gc()
        t0=time_ns()
        tr=train_scaling(H,LAMBDA,ETA,6,1234,1024,rows,mode)
        elapsed=(time_ns()-t0)/1e9
        if tr.finite
            last=rows[end]
            push!(rows,(L=L[],Nbos=NBOS[],hilbert=binomial(NBOS[]+L[]-1,NBOS[]),U_over_J=1.0,M=1024,
                lambda=LAMBDA,eta=ETA,depth=6,symmetry=String(mode)*"_benchmark",seed=1234,epoch=EPOCHS,finite=true,
                peak_raw_leaf=tr.peak_raw_leaf,maxabs_logamp=tr.maxabs_logamp,E_per_site=last.E_per_site,
                Eloc_variance=last.Eloc_variance,Eloc_variance_per_site=last.Eloc_variance_per_site,
                Eloc_variance_per_site2=last.Eloc_variance_per_site2,number_variance=last.number_variance,
                kinetic_per_site=last.kinetic_per_site,cache_entries=length(DIHEDRAL_CACHE),cache_hits=CACHE_HITS[],
                cache_misses=CACHE_MISSES[],cache_hit_rate=(CACHE_HITS[]+CACHE_MISSES[])==0 ? 0.0 : CACHE_HITS[]/(CACHE_HITS[]+CACHE_MISSES[]),
                elapsed_seconds=elapsed,canonical_calls=CANON_CALLS[],cache_evictions=CACHE_EVICTIONS[]))
            @printf(" mode=%-23s time=%8.2fs E/L=% .8f Var/L=% .3e cache=%d hit=%.4f misses=%d evicted=%d canon_calls=%d\n",
                String(mode),elapsed,last.E_per_site,last.Eloc_variance_per_site,length(DIHEDRAL_CACHE),
                (CACHE_HITS[]+CACHE_MISSES[])==0 ? 0.0 : CACHE_HITS[]/(CACHE_HITS[]+CACHE_MISSES[]),
                CACHE_MISSES[],CACHE_EVICTIONS[],CANON_CALLS[])
        else
            @printf(" mode=%s FAILED epoch=%d stage=%s\n",String(mode),tr.failure_epoch,tr.failure_stage)
        end
    end
    # Only benchmark summary rows have the extended timing fields.
    bench=[r for r in rows if endswith(String(r.symmetry),"_benchmark")]
    dir=joinpath(@__DIR__,"results"); mkpath(dir)
    path=joinpath(dir,"bose_hubbard_dihedral_cache_benchmark_L12.csv")
    writecsv(path,bench)
    println("\nBenchmark results written to $path")
end

end

if abspath(PROGRAM_FILE)==@__FILE__
    BoseHubbardDihedralCacheBenchmark.main()
end
