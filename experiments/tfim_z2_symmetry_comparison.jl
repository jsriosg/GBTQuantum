module TFIMZ2SymmetryComparison
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using GBTQuantum, Random, Statistics, Printf
include(joinpath(@__DIR__,"tfim_uncertainty_regularized_training.jl"))
const BaseExp=TFIMUncertaintyRegularizedTraining

# Clean representation experiment: same final regularized TFIM algorithm,
# differing only by raw vs exact global-spin-flip (Z2) canonical representation.
const N=14
const RATIOS=[0.5,1.0,2.0]
const SEEDS=[1234,2345,3456,4567,5678]
const M=512; const EPOCHS=150; const DEPTH=4; const ETA=0.05; const LAMBDA=1.0
const MODES=[:raw,:z2_canonical]

@inline function z2_is_canonical(x)
    # For +/-1 spins, lexicographic min(x,-x) is whichever has first spin -1.
    return x[1] < 0
end
function canonical_z2(x)
    z=Vector{Int8}(undef,length(x))
    s=z2_is_canonical(x) ? Int8(1) : Int8(-1)
    @inbounds for i in eachindex(x); z[i]=s*Int8(x[i]); end
    z
end
@inline feature_state(x,mode) = mode===:raw ? x : mode===:z2_canonical ? canonical_z2(x) : error("unknown mode $mode")

@inline function model_logamp(m,x,mode)
    mode===:raw && return GBTQuantum.logamplitude(m,x)
    z=canonical_z2(x)
    return m.logamp.bias + sum((GBTQuantum.predict(t,z) for t in m.logamp.trees); init=0.0)
end

function local_energy_mode(H,m,x,mode)
    mode===:raw && return GBTQuantum.local_energy!(H,m,x)
    A0=model_logamp(m,x,mode)
    e=ComplexF64(GBTQuantum.diagonal(H,x))
    @inbounds for i=1:H.N
        x[i]=-x[i]
        e -= H.h*exp(model_logamp(m,x,mode)-A0)
        x[i]=-x[i]
    end
    e
end

function compress_mode(samples,mode)
    nr,nf=size(samples); states=Matrix{Int8}(undef,nr,nf); counts=zeros(Int,nr)
    map=Dict{Tuple{Vararg{Int8}},Int}(); K=0
    @inbounds for r=1:nr
        z=mode===:raw ? collect(@view(samples[r,:])) : canonical_z2(@view(samples[r,:]))
        key=Tuple(z); j=get(map,key,0)
        if j==0
            K+=1; map[key]=K; states[K,:].=z; counts[K]=1
        else
            counts[j]+=1
        end
    end
    states[1:K,:],counts[1:K]
end

function batch_mode(H,m,samples,mode)
    states,counts=compress_mode(samples,mode); K=size(states,1)
    eloc=Vector{ComplexF64}(undef,K); Esum=0.0+0im
    @inbounds for j=1:K
        x=copy(@view states[j,:])
        e=local_energy_mode(H,m,x,mode); eloc[j]=e; Esum+=counts[j]*e
    end
    E=Esum/sum(counts)
    y=[-real(eloc[j]-E) for j=1:K]
    (states=states,counts=counts,local_energy=eloc,energy=E,targets=y)
end

function center_tree(t,b)
    w=Float64.(b.counts); p=[GBTQuantum.predict(t,@view b.states[j,:]) for j in axes(b.states,1)]
    mu=GBTQuantum.weighted_mean(p,w); nd=copy(t.nodes)
    for i in eachindex(nd)
        n=nd[i]
        n.isleaf && (nd[i]=GBTQuantum.Node(n.feature,n.value-mu,n.left,n.right,true))
    end
    GBTQuantum.RegressionTree(nd)
end

function sweep_mode!(rng,m,H,S,la,mode)
    mode===:raw && return GBTQuantum.sweep!(rng,m,S,la)
    accepted=0
    @inbounds for r in axes(S,1)
        x=@view S[r,:]; A=la[r]
        for _=1:H.N
            i=rand(rng,1:H.N); x[i]=-x[i]
            Ap=model_logamp(m,x,mode)
            if log(rand(rng)) < min(0.0,2*(Ap-A))
                A=Ap; accepted+=1
            else
                x[i]=-x[i]
            end
        end
        la[r]=A
    end
    accepted/(size(S,1)*H.N)
end

function train_model(H,mode,seed)
    rng=MersenneTwister(seed); S=Matrix{Int8}(undef,M,H.N)
    for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
    m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    la=[model_logamp(m,@view(S[r,:]),mode) for r=1:M]
    for _=1:50; sweep_mode!(rng,m,H,S,la,mode); end
    for ep=1:EPOCHS
        b=batch_mode(H,m,S,mode)
        raw=BaseExp.grow_uncertainty_tree(b.states,b.targets,b.counts;
            max_depth=DEPTH,min_weight=1.0,min_gain=0.0,lambda=LAMBDA)
        t=center_tree(raw,b)
        push!(m.logamp.trees,BaseExp.scaled_tree(t,ETA))
        @inbounds for r=1:M; la[r]=model_logamp(m,@view(S[r,:]),mode); end
        for _=1:2; sweep_mode!(rng,m,H,S,la,mode); end
        any(!isfinite,la) && error("non-finite log amplitudes: ratio=$(H.J/H.h), mode=$mode seed=$seed epoch=$ep")
    end
    m
end

function allstates(N)
    X=Matrix{Int8}(undef,1<<N,N)
    @inbounds for s=0:(1<<N)-1,i=1:N
        X[s+1,i]=((s>>(i-1))&1)==1 ? Int8(1) : Int8(-1)
    end
    X
end

function exact_stats(m,H,mode,X)
    d=size(X,1); A=zeros(d); mz=zeros(d)
    @inbounds for j=1:d
        x=@view X[j,:]; A[j]=model_logamp(m,x,mode); mz[j]=sum(Float64.(x))/H.N
    end
    lw=2 .* A; lw .-= maximum(lw); p=exp.(lw); p./=sum(p)
    el=zeros(d); mx=zeros(d)
    @inbounds for j=1:d
        x=copy(@view X[j,:]); A0=A[j]; e=GBTQuantum.diagonal(H,x); sx=0.0
        for i=1:H.N
            x[i]=-x[i]; r=exp(model_logamp(m,x,mode)-A0); e-=H.h*r; sx+=r/H.N; x[i]=-x[i]
        end
        el[j]=e; mx[j]=sx
    end
    E=sum(p.*el)
    (energy=E,variance=sum(p.*(el.-E).^2),mz=sum(p.*mz),abs_mz=sum(p.*abs.(mz)),
     mz2=sum(p.*mz.^2),mx=sum(p.*mx),z2_pair_max=maximum(abs(p[j]-p[((j-1) ⊻ ((1<<H.N)-1))+1]) for j=1:d))
end

function writecsv(path,rows)
    ns=propertynames(rows[1]); open(path,"w") do io
        println(io,join(string.(ns),','))
        for r in rows; println(io,join((getproperty(r,n) for n in ns),',')); end
    end
end

function main()
    println("="^136)
    println("TFIM Z2 REPRESENTATION COMPARISON")
    println("N=$N J/h=$RATIOS modes=$MODES seeds=$(length(SEEDS))")
    println("Frozen production settings: lambda=$LAMBDA eta=$ETA M=$M depth=$DEPTH epochs=$EPOCHS; no Newton/adaptive step.")
    println("Z2 ansatz: A(x)=F(C_Z2(x)), C_Z2(x)=min_lex{x,-x}. Exact quantities are evaluation-only.")
    println("="^136)
    X=allstates(N); rows=NamedTuple[]
    for ratio in RATIOS
        H=GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true)
        gs=GBTQuantum.exact_ground_observables(H)
        @printf("\nJ/h=%.1f exact E=% .10f mz=% .3e |mz|=%.8f mz2=%.8f mx=%.8f\n",
            ratio,gs.energy,gs.mz,gs.abs_mz,gs.mz2,gs.mx)
        for mode in MODES,seed in SEEDS
            m=train_model(H,mode,seed); o=exact_stats(m,H,mode,X)
            push!(rows,(N=N,hilbert=1<<N,ratio=ratio,lambda=LAMBDA,mode=String(mode),seed=seed,
                Egs=gs.energy,E_final=o.energy,energy_error=o.energy-gs.energy,variance=o.variance,
                mz_exact=gs.mz,mz_model=o.mz,mz_abs_error=abs(o.mz-gs.mz),
                abs_mz_exact=gs.abs_mz,abs_mz_model=o.abs_mz,abs_mz_abs_error=abs(o.abs_mz-gs.abs_mz),
                mz2_exact=gs.mz2,mz2_model=o.mz2,mz2_abs_error=abs(o.mz2-gs.mz2),
                mx_exact=gs.mx,mx_model=o.mx,mx_abs_error=abs(o.mx-gs.mx),z2_pair_max=o.z2_pair_max))
            @printf(" %-12s seed=%d Eerr=% .3e Var=% .3e mz=%+.3e |d|mz||=%.3e |dmz2|=%.3e |dmx|=%.3e Z2pair=%.1e\n",
                String(mode),seed,o.energy-gs.energy,o.variance,o.mz,abs(o.abs_mz-gs.abs_mz),
                abs(o.mz2-gs.mz2),abs(o.mx-gs.mx),o.z2_pair_max)
        end
        for mode in MODES
            q=[r for r in rows if r.ratio==ratio && r.mode==String(mode)]
            @printf(" SUMMARY %-12s Eerr=% .3e +/- %.2e Var=%.3e |mz|odd=%.3e |d|mz||=%.3e |dmz2|=%.3e |dmx|=%.3e\n",
                String(mode),mean(r.energy_error for r in q),std(r.energy_error for r in q),
                mean(r.variance for r in q),mean(abs(r.mz_model) for r in q),
                mean(r.abs_mz_abs_error for r in q),mean(r.mz2_abs_error for r in q),mean(r.mx_abs_error for r in q))
        end
    end
    dir=joinpath(@__DIR__,"results"); mkpath(dir)
    path=joinpath(dir,"tfim_z2_symmetry_comparison.csv"); writecsv(path,rows)
    println("\nResults written to $path")
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMZ2SymmetryComparison.main(); end
