module TFIMTreeLeafPathologyMap
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using GBTQuantum, Random, Statistics, Printf

# Frozen-checkpoint microscope for the newly fitted log-amplitude tree f.
# Tests whether extreme f values / derivative contributions are concentrated in
# weakly supported leaves. Exact N=8 Hilbert enumeration is diagnostic only.
# Gauge convention: after fitting, f is re-centered exactly so <f>_p = 0.

const N=8; const NSAMPLES=512; const MAX_DEPTH=4; const ETA_FIXED=0.05
const ETA_CAP=0.4; const CURVATURE_FLOOR=1e-12; const SEED=1234
const CHECKPOINTS=Dict(1.0=>Set([8,50]),2.0=>Set([4,10]))

function allstates(N)
 X=Matrix{Int8}(undef,1<<N,N)
 for s=0:(1<<N)-1, i=1:N; X[s+1,i]=((s>>(i-1))&1)==1 ? Int8(1) : Int8(-1); end; X
end
@inline flip(j,i)=((j-1) ⊻ (1<<(i-1)))+1

function center_tree(t,X,w)
 pr=[GBTQuantum.predict(t,@view X[j,:]) for j in axes(X,1)]; mu=GBTQuantum.weighted_mean(pr,w); mu==0 && return t
 nd=copy(t.nodes); for i in eachindex(nd); n=nd[i]; n.isleaf && (nd[i]=GBTQuantum.Node(n.feature,n.value-mu,n.left,n.right,true)); end
 GBTQuantum.RegressionTree(nd)
end
scaled_tree(t,e)=GBTQuantum.RegressionTree([n.isleaf ? GBTQuantum.Node(n.feature,e*n.value,n.left,n.right,true) : n for n in t.nodes])

function leaf_index(t,x)
 i=1
 while true
  n=t.nodes[i]; n.isleaf && return i
  # grow_tree uses spin feature values; reproduce predict traversal.
  i = x[n.feature] <= 0 ? n.left : n.right
 end
end

function flocal(H,m,t,x)
 A=GBTQuantum.logamplitude(m,x); f=GBTQuantum.predict(t,x); z=GBTQuantum.diagonal(H,x)*f
 for i=1:H.N; x[i]=-x[i]; z-=H.h*exp(GBTQuantum.logamplitude(m,x)-A)*GBTQuantum.predict(t,x); x[i]=-x[i]; end; z
end
function decision(H,m,b,t)
 f=[GBTQuantum.predict(t,@view b.states[j,:]) for j in axes(b.states,1)]; w=Float64.(b.counts); el=real.(b.local_energy); W=sum(w)
 Ef=sum(w.*f)/W; E=sum(w.*el)/W; Ef2=sum(w.*f.*f)/W; Ef2el=sum(w.*f.*f.*el)/W
 g=2sum(w.*(f.-Ef).*(el.-E))/W; q=sum(w[j]*f[j]*flocal(H,m,t,@view b.states[j,:]) for j in axes(b.states,1))
 c=2Ef2el+2q/W-4E*Ef2-4Ef*g
 if !isfinite(g)||!isfinite(c)||g>=0||c<=CURVATURE_FLOOR; return ETA_FIXED,g,c,:fallback; end
 raw=-g/c; (!isfinite(raw)||raw<=0) && return (ETA_FIXED,g,c,:fallback)
 clamp(raw,0.0,ETA_CAP),g,c,:newton
end

function exact_arrays(H,m,t,X)
 d=size(X,1); A=zeros(d); f=zeros(d); leaf=zeros(Int,d)
 for j=1:d; x=@view X[j,:]; A[j]=GBTQuantum.logamplitude(m,x); f[j]=GBTQuantum.predict(t,x); leaf[j]=leaf_index(t,x); end
 p=exp.(2A .- maximum(2A)); p./=sum(p); f .-= sum(p.*f) # exact gauge
 el=zeros(d); ef=zeros(d)
 for j=1:d
  e=GBTQuantum.diagonal(H,@view X[j,:]); q=GBTQuantum.diagonal(H,@view X[j,:])*f[j]
  for i=1:H.N; k=flip(j,i); r=exp(A[k]-A[j]); e-=H.h*r; q-=H.h*r*f[k]; end
  el[j]=e; ef[j]=q
 end
 E=sum(p.*el); g=2sum(p.*f.*el); f2=sum(p.*f.^2)
 # Per-root canonical curvature contribution under p, summing exactly to c.
 Croot=2p.*f.^2.*el + 2p.*f.*ef - 4E.*p.*f.^2
 c=sum(Croot)
 A,p,f,leaf,el,ef,E,g,f2,Croot,c
end

function audit(H,m,t,b,X,ratio,ep,gmc,cmc,mode)
 A,p,f,leaf,el,ef,E,g,f2,Croot,c=exact_arrays(H,m,t,X)
 # Training support per leaf: unique sampled configurations and multiplicity weight.
 train_unique=Dict{Int,Int}(); train_weight=Dict{Int,Float64}()
 for j in axes(b.states,1)
  l=leaf_index(t,@view b.states[j,:]); train_unique[l]=get(train_unique,l,0)+1; train_weight[l]=get(train_weight,l,0.0)+Float64(b.counts[j])
 end
 leaves=sort(unique(leaf)); rows=NamedTuple[]
 println("\n","-"^112)
 @printf("J/h=%.2f epoch=%d mode=%s | MC g=% .4e c=% .4e | exact-centered g=% .4e c=% .4e eta=% .5f\n",ratio,ep,String(mode),gmc,cmc,g,c,(g<0&&c>0 ? -g/c : NaN))
 @printf("Gauge <f>=% .3e | leaves=%d | max|f(state)|=% .4e | <f^2>=% .4e\n",sum(p.*f),length(leaves),maximum(abs.(f)),f2)
 for l in leaves
  idx=findall(==(l),leaf); mass=sum(p[idx]); fv=f[first(idx)] # tree constant in leaf before exact gauge, still constant after shift
  gf=sum(2 .* p[idx].*f[idx].*el[idx]); cf=sum(Croot[idx]); f2l=sum(p[idx].*f[idx].^2)
  tw=get(train_weight,l,0.0); tu=get(train_unique,l,0); exactstates=length(idx)
  @printf(" leaf=%3d f=% .4e train_weight=%6.1f train_unique=%3d exact_states=%3d Born_mass=% .4e <f2>part=% .4e g_part=% .4e c_part=% .4e\n",l,fv,tw,tu,exactstates,mass,f2l,gf,cf)
  push!(rows,(ratio=ratio,epoch=ep,leaf=l,f=fv,absf=abs(fv),train_weight=tw,train_unique=tu,exact_states=exactstates,born_mass=mass,f2_contribution=f2l,g_contribution=gf,c_contribution=cf,abs_c_contribution=abs(cf),E_exact=E,g_exact=g,c_exact=c,eta_exact=(g<0&&c>0 ? -g/c : NaN),mc_g=gmc,mc_c=cmc,mode=String(mode)))
 end
 # Compact ranking to make pathology visible immediately.
 ord=sortperm(rows,by=r->r.absf,rev=true)
 println(" Top leaves by |f|:")
 for k in ord[1:min(5,length(ord))]; r=rows[k]; @printf("   leaf=%3d |f|=% .3e support=%5.1f Born=% .3e |c_part|=% .3e\n",r.leaf,r.absf,r.train_weight,r.born_mass,r.abs_c_contribution); end
 rows
end

function train(H,ratio,X)
 cfg=GBTQuantum.TrainingConfig(nsamples=NSAMPLES,epochs=150,max_depth=MAX_DEPTH,eta=ETA_FIXED,burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=SEED,exact_diagnostics=false)
 rng=MersenneTwister(SEED); S=Matrix{Int8}(undef,NSAMPLES,H.N); for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
 m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); la=zeros(NSAMPLES); for _=1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,m,S,la); end
 out=NamedTuple[]; wanted=CHECKPOINTS[ratio]
 for ep=1:maximum(wanted)
  b=GBTQuantum.vmc_batch(H,m,S); y,_=GBTQuantum.make_targets(b); t=GBTQuantum.grow_tree(b.states,y,b.counts;max_depth=cfg.max_depth,min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain); t=center_tree(t,b.states,b.counts)
  eta,g,c,mode=decision(H,m,b,t); ep in wanted && append!(out,audit(H,m,t,b,X,ratio,ep,g,c,mode))
  push!(m.logamp.trees,scaled_tree(t,eta)); GBTQuantum.refresh_logamps!(la,m,S); for _=1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,m,S,la); end
  any(!isfinite,la) && break
 end; out
end
function writecsv(path,rows)
 isempty(rows)&&return; ns=propertynames(rows[1]); open(path,"w") do io; println(io,join(string.(ns),',')); for r in rows; println(io,join((getproperty(r,n) for n in ns),',')); end; end
end
function main()
 println("="^112); println("TFIM EXACT TREE-LEAF PATHOLOGY MAP"); println("N=$N Hilbert=$(1<<N), trajectory samples=$NSAMPLES, max_depth=$MAX_DEPTH"); println("Tests leaf training support vs exact Born mass, |f|, <f^2>, gradient and curvature contributions; exact gauge <f>_p=0"); println("="^112)
 X=allstates(N); rows=NamedTuple[]
 for r in sort(collect(keys(CHECKPOINTS))); append!(rows,train(GBTQuantum.TFIMHamiltonian(N;J=r,h=1.0,periodic=true),r,X)); end
 dir=joinpath(@__DIR__,"results"); mkpath(dir); path=joinpath(dir,"tfim_tree_leaf_pathology_map.csv"); writecsv(path,rows); println("\nLeaf results written to $path")
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMTreeLeafPathologyMap.main(); end
