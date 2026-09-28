module TFIMLeafCurvatureErrorMap
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using GBTQuantum, Random, Statistics, Printf

# Leaf-by-leaf microscope for MC curvature error.
# IMPORTANT: reproduces production gauge fixing: after grow_tree, subtract the
# sample-weighted mean prediction from EVERY leaf, so <f>_MC = 0. Exact Hilbert
# enumeration is oracle evaluation only and NEVER re-centers the tree.

const N=8; const NSAMPLES=512; const MAX_DEPTH=4; const ETA_FIXED=0.05; const SEED=1234
const CHECKPOINTS=Dict(1.0=>Set([8,50]),2.0=>Set([4,10]))

function allstates(N)
 X=Matrix{Int8}(undef,1<<N,N)
 for s=0:(1<<N)-1, i=1:N; X[s+1,i]=((s>>(i-1))&1)==1 ? Int8(1) : Int8(-1); end; X
end
@inline flip(j,i)=((j-1) ⊻ (1<<(i-1)))+1
function leaf_index(t,x)
 i=1
 while true
  n=t.nodes[i]; n.isleaf && return i
  i=x[n.feature] <= 0 ? n.left : n.right
 end
end
function center_like_production(t,b)
 w=Float64.(b.counts); pred=[GBTQuantum.predict(t,@view b.states[j,:]) for j in axes(b.states,1)]
 mu=GBTQuantum.weighted_mean(pred,w); nd=copy(t.nodes)
 for i in eachindex(nd); n=nd[i]; n.isleaf && (nd[i]=GBTQuantum.Node(n.feature,n.value-mu,n.left,n.right,true)); end
 GBTQuantum.RegressionTree(nd),mu
end
scaled_tree(t,e)=GBTQuantum.RegressionTree([n.isleaf ? GBTQuantum.Node(n.feature,e*n.value,n.left,n.right,true) : n for n in t.nodes])

function exact_oracle(H,m,t,X)
 d=size(X,1); A=zeros(d); f=zeros(d); leaf=zeros(Int,d)
 for j=1:d
  x=@view X[j,:]; A[j]=GBTQuantum.logamplitude(m,x); f[j]=GBTQuantum.predict(t,x); leaf[j]=leaf_index(t,x)
 end
 p=exp.(2A .- maximum(2A)); p./=sum(p)
 el=zeros(d); hf=zeros(d)
 for j=1:d
  el[j]=GBTQuantum.diagonal(H,@view X[j,:]); hf[j]=GBTQuantum.diagonal(H,@view X[j,:])*f[j]
  for i=1:H.N
   k=flip(j,i); r=exp(A[k]-A[j]); el[j]-=H.h*r; hf[j]-=H.h*r*f[k]
  end
 end
 Ef=sum(p.*f); E=sum(p.*el); Ef2=sum(p.*f.^2); Ef2el=sum(p.*f.^2.*el); Efhf=sum(p.*f.*hf)
 g=2sum(p.*(f.-Ef).*(el.-E))
 c=2Ef2el+2Efhf-4E*Ef2-4Ef*g
 (;A,p,f,leaf,el,hf,Ef,E,Ef2,g,c)
end

function sample_arrays(H,m,t,b)
 n=size(b.states,1); w=Float64.(b.counts); W=sum(w); f=zeros(n); leaf=zeros(Int,n); hf=zeros(n); el=real.(b.local_energy)
 for j=1:n
  x=@view b.states[j,:]; f[j]=GBTQuantum.predict(t,x); leaf[j]=leaf_index(t,x); A=GBTQuantum.logamplitude(m,x)
  z=GBTQuantum.diagonal(H,x)*f[j]
  for i=1:H.N; x[i]=-x[i]; z-=H.h*exp(GBTQuantum.logamplitude(m,x)-A)*GBTQuantum.predict(t,x); x[i]=-x[i]; end
  hf[j]=z
 end
 Ef=sum(w.*f)/W; E=sum(w.*el)/W; g=2sum(w.*(f.-Ef).*(el.-E))/W
 Ef2=sum(w.*f.^2)/W; c=2sum(w.*f.^2.*el)/W + 2sum(w.*f.*hf)/W - 4E*Ef2 - 4Ef*g
 (;w,W,f,leaf,hf,el,Ef,E,g,c)
end

function neighbor_coverage(H,b,X,leaf_exact)
 # State is "represented" if it occurs as a unique root in the compressed batch.
 represented=Set{Int}()
 for j in axes(b.states,1)
  s=0; for i=1:H.N; b.states[j,i]>0 && (s |= 1<<(i-1)); end; push!(represented,s+1)
 end
 out=Dict{Int,Tuple{Int,Int,Float64}}()
 for L in unique(leaf_exact)
  roots=findall(==(L),leaf_exact); total=length(roots)*H.N; hit=0
  for j in roots, i=1:H.N; flip(j,i) in represented && (hit+=1); end
  out[L]=(hit,total,total==0 ? NaN : hit/total)
 end; out
end

function audit(H,m,t,b,X,ratio,ep,shift)
 ex=exact_oracle(H,m,t,X); sm=sample_arrays(H,m,t,b); cov=neighbor_coverage(H,b,X,ex.leaf)
 @printf("\n%s\n", "-"^124)
 @printf("J/h=%.2f epoch=%d | production gauge shift=% .6e | <f>_MC=% .3e | <f>_exact=% .3e\n",ratio,ep,shift,sm.Ef,ex.Ef)
 @printf("total exact g=% .6e c=% .6e | MC g=% .6e c=% .6e | delta_c=% .6e\n",ex.g,ex.c,sm.g,sm.c,sm.c-ex.c)

 # Exact additive decomposition using the SAME sample-gauge-fixed f. The final
 # global -4<Ef>g term is allocated by leaf Born mass so leaf sums equal total c.
 leaves=sort(unique(ex.leaf)); rows=NamedTuple[]
 for L in leaves
  ix=findall(==(L),ex.leaf); mass=sum(ex.p[ix]); fL=ex.f[first(ix)]
  cex=sum(2 .* ex.p[ix].*ex.f[ix].^2.*ex.el[ix] .+ 2 .* ex.p[ix].*ex.f[ix].*ex.hf[ix] .- 4 .* ex.E .* ex.p[ix].*ex.f[ix].^2) - 4*ex.Ef*ex.g*mass
  gex=2sum(ex.p[ix].*(ex.f[ix].-ex.Ef).*(ex.el[ix].-ex.E))

  im=findall(==(L),sm.leaf); mw=sum(sm.w[im]); mmass=mw/sm.W
  if isempty(im)
   gmc=0.0; cmc=-4*sm.Ef*sm.g*mmass
  else
   gmc=2sum(sm.w[im].*(sm.f[im].-sm.Ef).*(sm.el[im].-sm.E))/sm.W
   cmc=sum(2 .* sm.w[im].*sm.f[im].^2.*sm.el[im] .+ 2 .* sm.w[im].*sm.f[im].*sm.hf[im] .- 4 .* sm.E .* sm.w[im].*sm.f[im].^2)/sm.W - 4*sm.Ef*sm.g*mmass
  end
  hit,total,coverage=cov[L]; dc=cmc-cex
  push!(rows,(ratio=ratio,epoch=ep,leaf=L,f=fL,absf=abs(fL),train_weight=mw,train_mass=mmass,exact_born_mass=mass,mass_error=mmass-mass,neighbor_hits=hit,neighbor_edges=total,neighbor_coverage=coverage,g_exact_leaf=gex,g_mc_leaf=gmc,g_error_leaf=gmc-gex,c_exact_leaf=cex,c_mc_leaf=cmc,c_error_leaf=dc,abs_c_error_leaf=abs(dc),exact_mean_f=ex.Ef,mc_mean_f=sm.Ef,g_exact=ex.g,g_mc=sm.g,c_exact=ex.c,c_mc=sm.c))
 end
 @printf("consistency: sum leaf g exact=% .6e MC=% .6e | sum leaf c exact=% .6e MC=% .6e\n",sum(r.g_exact_leaf for r in rows),sum(r.g_mc_leaf for r in rows),sum(r.c_exact_leaf for r in rows),sum(r.c_mc_leaf for r in rows))
 ord=sortperm(rows,by=r->r.abs_c_error_leaf,rev=true)
 println(" Top leaves by |delta c_L|:")
 for k in ord[1:min(8,length(ord))]
  r=rows[k]; @printf("  leaf=%3d dC=% .3e Cex=% .3e Cmc=% .3e support=%6.1f p=% .3e mass_err=% .3e f=% .3e neigh_cov=%6.2f%%\n",r.leaf,r.c_error_leaf,r.c_exact_leaf,r.c_mc_leaf,r.train_weight,r.exact_born_mass,r.mass_error,r.f,100r.neighbor_coverage)
 end
 rows
end

function train(H,ratio,X)
 cfg=GBTQuantum.TrainingConfig(nsamples=NSAMPLES,epochs=150,max_depth=MAX_DEPTH,eta=ETA_FIXED,burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=SEED,exact_diagnostics=false)
 rng=MersenneTwister(SEED); S=Matrix{Int8}(undef,NSAMPLES,H.N); for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
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
 end; out
end
function writecsv(path,rows)
 isempty(rows)&&return; ns=propertynames(rows[1]); open(path,"w") do io
  println(io,join(string.(ns),',')); for r in rows; println(io,join((getproperty(r,n) for n in ns),',')); end
 end
end
function main()
 println("="^124); println("TFIM GAUGE-CONSISTENT LEAF CURVATURE ERROR MAP")
 println("N=$N Hilbert=$(1<<N), trajectory samples=$NSAMPLES, fixed eta=$ETA_FIXED")
 println("Production gauge: sample-weighted <f>_MC=0 by shifting every tree leaf; exact enumeration never re-centers f")
 println("Per leaf: support, exact Born mass, mass error, f, Hamiltonian-neighbor coverage, exact/MC g and c, delta c")
 println("="^124)
 X=allstates(N); rows=NamedTuple[]
 for r in sort(collect(keys(CHECKPOINTS))); append!(rows,train(GBTQuantum.TFIMHamiltonian(N;J=r,h=1.0,periodic=true),r,X)); end
 dir=joinpath(@__DIR__,"results"); mkpath(dir); path=joinpath(dir,"tfim_leaf_curvature_error_map.csv"); writecsv(path,rows); println("\nResults written to $path")
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMLeafCurvatureErrorMap.main(); end
