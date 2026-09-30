module BoseHubbardFastDihedralL8

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Random
using Statistics
using LinearAlgebra
using Printf

# First transfer/smoke test.  These points bracket the thermodynamic 1D
# unit-filling BKT region (~3.3) but L=6 is finite, so none is labelled
# a finite-size "critical point".
const L = 8
const NBOS = 8
const J = 1.0
const U_VALUES = [1.0, 6.0]
const LAMBDA = 4.0
const SEEDS = [1234,2345,3456]

# Frozen TFIM production hyperparameters.
const M = 512
const EPOCHS = 400
const CHECKPOINTS = Set([50,100,150,250,400])
const DEPTHS = [4]
const SYMMETRY_MODES = [:dihedral_canonical]
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

const DIHEDRAL_CACHE = Dict{NTuple{L,Int16},Vector{Int16}}()

function canonical_dihedral_uncached(x)
    # D_L orbit = all cyclic translations of x and of one reflected copy.
    best=canonical_translation(x)
    reflected=reverse(collect(x))
    reflected_best=canonical_translation(reflected)
    return Tuple(reflected_best) < Tuple(best) ? reflected_best : best
end

function canonical_dihedral(x)
    key=NTuple{L,Int16}(Int16.(x))
    return get!(DIHEDRAL_CACHE,key) do
        Int16.(canonical_dihedral_uncached(x))
    end
end

function precompute_dihedral_cache!()
    empty!(DIHEDRAL_CACHE)
    basis=GBTQuantum.bose_hubbard_basis(L,NBOS)
    for r in axes(basis,1)
        x=@view basis[r,:]
        key=Tuple(x)
        DIHEDRAL_CACHE[key]=Int16.(canonical_dihedral_uncached(x))
    end
    @assert length(DIHEDRAL_CACHE)==size(basis,1)
    return length(DIHEDRAL_CACHE)
end

function feature_state(x,mode)
    mode === :raw && return x
    mode === :translation_canonical && return canonical_translation(x)
    mode === :dihedral_canonical && return canonical_dihedral(x)
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
    S=Matrix{Int16}(undef,M,L)
    # Random weak compositions generated by placing NBOS indistinguishable
    # particles independently; this is only a starting distribution and is
    # followed by burn-in.
    for r in 1:M
        fill!(@view(S[r,:]),0)
        for _ in 1:NBOS
            S[r,rand(rng,1:L)] += 1
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
        sitevar=sum((Float64(x[i])-NBOS/L)^2 for i in 1:L)/L
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
    nvar=sum(p[r]*sum((Float64(basis[r,i])-NBOS/L)^2 for i in 1:L)/L for r in 1:d)
    interaction=sum(p[r]*(0.5*H.U*sum(Float64(basis[r,i])*(Float64(basis[r,i])-1) for i in 1:L)) for r in 1:d)
    kinetic=E-interaction+H.mu*NBOS
    return (energy=E,variance=var,number_variance=nvar,kinetic=kinetic)
end

function exact_ground_stats(H)
    gs=GBTQuantum.exact_ground_state(H,NBOS)
    psi=gs.state
    p=abs2.(psi)
    basis=gs.basis

    nvar=0.0
    interaction=0.0
    for r in axes(basis,1)
        x=@view basis[r,:]
        nvar += p[r]*sum((Float64(x[i])-NBOS/L)^2 for i in 1:L)/L
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
            push!(rows,(L=L,Nbos=NBOS,hilbert=size(gs.basis,1),U_over_J=H.U/H.J,M=M,
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

function main()
    println("="^112)
    println("BOSE-HUBBARD L=8 FAST DIHEDRAL REPRESENTATION STUDY")
    println("L=$L Nbos=$NBOS Hilbert=$(binomial(NBOS+L-1,NBOS)) J=$J U/J=$U_VALUES")
    println("lambda=$LAMBDA eta=$ETA seeds=$SEEDS M=$M depth=4 modes=$SYMMETRY_MODES epochs=$EPOCHS checkpoints=$(sort(collect(CHECKPOINTS)))")
    println("Precomputed D_L canonical lookup: $(precompute_dihedral_cache!()) fixed-N states.")
    println("Fixed-N, periodic, mu=0. Exact diagonalization is evaluation-only.")
    println("="^112)

    rows=NamedTuple[]
    for U in U_VALUES
        H=GBTQuantum.BoseHubbardHamiltonian(L;J=J,U=U,mu=0.0,periodic=true)
        gs=exact_ground_stats(H)
        @printf("\nU/J=%4.1f exact E=% .10f  number_var=% .8f  kinetic=% .8f\n",
                U/J,gs.energy,gs.number_variance,gs.kinetic)

        for mode in SYMMETRY_MODES, depth in DEPTHS, seed in SEEDS
            tr=train(H,LAMBDA,ETA,depth,seed,M,gs,rows,mode)
            if tr.finite
                last=rows[end]
                @printf(" mode=%s depth=%d seed=%d E400err=% .3e Var(E_loc)=% .3e peak|leaf|=% .3e max|A|=% .3e\n",
                    String(mode),depth,seed,last.energy_error,last.exact_model_variance,tr.peak_raw_leaf,tr.maxabs_logamp)
            else
                @printf(" mode=%s depth=%d seed=%d FAILED epoch=%d stage=%s peak|leaf|=% .3e max|A|=% .3e\n",
                    String(mode),depth,seed,tr.failure_epoch,tr.failure_stage,tr.peak_raw_leaf,tr.maxabs_logamp)
            end
        end
    end

    dir=joinpath(@__DIR__,"results"); mkpath(dir)
    path=joinpath(dir,"bose_hubbard_fast_dihedral_L8.csv")
    writecsv(path,rows)
    println("\nResults written to $path")
end

end

if abspath(PROGRAM_FILE)==@__FILE__
    BoseHubbardFastDihedralL8.main()
end
