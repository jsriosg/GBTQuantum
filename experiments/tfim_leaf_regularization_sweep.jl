module TFIMLeafRegularizationSweep
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using GBTQuantum, Random, Statistics, Printf

# Controlled N=8 diagnostic. Training trajectory stays canonical fixed eta=0.05.
# At frozen checkpoints, fit ONE canonical raw tree and evaluate counterfactual
# regularizations on that same tree/state:
#   (1) support-dependent L2 shrinkage f_L <- f_L W_L/(W_L+lambda)
#   (2) hard clipping
#   (3) minimum-support pruning by replacing weak leaves with 0 before recentering
# Every variant is sample-centered AFTER regularization, matching scalable VMC.
# Exact Hilbert enumeration is used only to measure the resulting physical direction.

const N=8; const NSAMPLES=512; const MAX_DEPTH=4; const ETA_FIXED=0.05; const SEED=1234
const CHECKPOINTS=Dict(1.0=>Set([8,50]),2.0=>Set([4,10]))
const LAMBDAS=[0.0,1.0,2.0,4.0,8.0,16.0,32.0]
const CLIPS=[Inf,50.0,25.0,10.0,5.0,2.5]
const MIN_SUPPORTS=[1.0,2.0,4.0,8.0,16.0,32.0]

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
function leaf_support(t,b)
 d=Dict{Int,Float64}()
 for j in axes(b.states,1)
  l=leaf_index(t,@view b.states[j,:]); d[l]=get(d,l,0.0)+Float64(b.counts[j])
 end; d
end
function transform_tree(t,support; lambda=0.0, clip=Inf, min_support=0.0)
 nd=copy(t.nodes)
 for i in eachindex(nd)
  n=nd[i]; n.isleaf || continue
  W=get(support,i,0.0); v=n.value
  if W < min_support
   v=0.0
  elseif lambda>0
   v *= W/(W+lambda)
  end
  isfinite(clip) && (v=clamp(v,-clip,clip))
  nd[i]=GBTQuantum.Node(n.feature,v,n.left,n.right,true)
 end
 GBTQuantum.RegressionTree(nd)
end
function center_sample(t,b)
 w=Float64.(b.counts); pr=[GBTQuantum.predict(t,@view b.states[j,:]) for j in axes(b.states,1)]
 mu=GBTQuantum.weighted_mean(pr,w); nd=copy(t.nodes)
 for i in eachindex(nd); n=nd[i]; n.isleaf && (nd[i]=GBTQuantum.Node(n.feature,n.value-mu,n.left,n.right,true)); end
 GBTQuantum.RegressionTree(nd),mu
end
scaled_tree(t,e)=GBTQuantum.RegressionTree([n.isleaf ? GBTQuantum.Node(n.feature,e*n.value,n.left,n.right,true) : n for n in t.nodes])

function exact_metrics(H,m,t,X)
 d=size(X,1); A=zeros(d); f=zeros(d)
 for j=1:d; A[j]=GBTQuantum.logamplitude(m,@view X[j,:]); f[j]=GBTQuantum.predict(t,@view X[j,:]); end
 p=exp.(2A .- maximum(2A)); p./=sum(p)
 meanf=sum(p.*f); fc=f.-meanf # physical gauge for exact derivative reporting
 el=zeros(d); ef=zeros(d)
 for j=1:d
  e=GBTQuantum.diagonal(H,@view X[j,:]); q=GBTQuantum.diagonal(H,@view X[j,:])*fc[j]
  for i=1:H.N
   k=flip(j,i); r=exp(A[k]-A[j]); e-=H.h*r; q-=H.h*r*fc[k]
  end
  el[j]=e; ef[j]=q
 end
 E=sum(p.*el); f2=sum(p.*fc.^2); g=2sum(p.*fc.*el)
 c=sum(2 .* p .* fc.^2 .* el .+ 2 .* p .* fc .* ef .- 4 .* E .* p .* fc.^2)
 eta=(g<0 && c>0) ? -g/c : NaN
 (meanf=meanf,maxabs=maximum(abs.(fc)),f2=f2,g=g,c=c,eta=eta,E=E)
end
function sample_metrics(H,m,t,b)
 f=[GBTQuantum.predict(t,@view b.states[j,:]) for j in axes(b.states,1)]; w=Float64.(b.counts); el=real.(b.local_energy); W=sum(w)
 Ef=sum(w.*f)/W; E=sum(w.*el)/W; Ef2=sum(w.*f.*f)/W; Ef2el=sum(w.*f.*f.*el)/W
 g=2sum(w.*(f.-Ef).*(el.-E))/W
 q=0.0
 for j in axes(b.states,1)
  x=@view b.states[j,:]; A=GBTQuantum.logamplitude(m,x); fj=f[j]; z=GBTQuantum.diagonal(H,x)*fj
  for i=1:H.N; x[i]=-x[i]; z-=H.h*exp(GBTQuantum.logamplitude(m,x)-A)*GBTQuantum.predict(t,x); x[i]=-x[i]; end
  q += w[j]*fj*z
 end
 c=2Ef2el+2q/W-4E*Ef2-4Ef*g
 eta=(g<0 && c>0) ? -g/c : NaN
 (meanf=Ef,g=g,c=c,eta=eta)
end

function evaluate_variant(H,m,raw,b,X,ratio,ep,kind,param,support)
 t0 = kind=="l2" ? transform_tree(raw,support;lambda=param) : kind=="clip" ? transform_tree(raw,support;clip=param) : transform_tree(raw,support;min_support=param)
 t,shift=center_sample(t0,b)
 ex=exact_metrics(H,m,t,X); sm=sample_metrics(H,m,t,b)
 @printf("  %-7s %7.2f | max|f|=%8.3f <f2>=%9.3e | exact g=% .3e c=% .3e eta=%8.4f | MC c=% .3e eta=%8.4f | c relerr=%8.3f\n",kind,param,ex.maxabs,ex.f2,ex.g,ex.c,ex.eta,sm.c,sm.eta,abs(sm.c-ex.c)/(abs(ex.c)+eps()))
 (ratio=ratio,epoch=ep,kind=kind,param=param,sample_center_shift=shift,exact_residual_mean=ex.meanf,maxabs_f=ex.maxabs,f2=ex.f2,g_exact=ex.g,c_exact=ex.c,eta_exact=ex.eta,g_mc=sm.g,c_mc=sm.c,eta_mc=sm.eta,c_relative_error=abs(sm.c-ex.c)/(abs(ex.c)+eps()))
end

function audit(H,m,raw,b,X,ratio,ep)
 support=leaf_support(raw,b)
 println("\n","-"^118); @printf("J/h=%.2f epoch=%d | raw leaves=%d | support min/median/max = %.1f / %.1f / %.1f\n",ratio,ep,length(support),minimum(values(support)),median(collect(values(support))),maximum(values(support)))
 rows=NamedTuple[]
 println(" L2 shrinkage (then sample recenter):")
 for l in LAMBDAS; push!(rows,evaluate_variant(H,m,raw,b,X,ratio,ep,"l2",l,support)); end
 println(" Hard clipping control (then sample recenter):")
 for z in CLIPS; push!(rows,evaluate_variant(H,m,raw,b,X,ratio,ep,"clip",z,support)); end
 println(" Minimum-support zeroing control (then sample recenter):")
 for s in MIN_SUPPORTS; push!(rows,evaluate_variant(H,m,raw,b,X,ratio,ep,"minsup",s,support)); end
 rows
end

function train(H,ratio,X)
 cfg=GBTQuantum.TrainingConfig(nsamples=NSAMPLES,epochs=150,max_depth=MAX_DEPTH,eta=ETA_FIXED,burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=SEED,exact_diagnostics=false)
 rng=MersenneTwister(SEED); S=Matrix{Int8}(undef,NSAMPLES,H.N); for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
 m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); la=zeros(NSAMPLES); for _=1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,m,S,la); end
 out=NamedTuple[]; wanted=CHECKPOINTS[ratio]
 for ep=1:maximum(wanted)
  b=GBTQuantum.vmc_batch(H,m,S); y,_=GBTQuantum.make_targets(b)
  raw=GBTQuantum.grow_tree(b.states,y,b.counts;max_depth=cfg.max_depth,min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain)
  ep in wanted && append!(out,audit(H,m,raw,b,X,ratio,ep))
  # Preserve canonical control trajectory: sample-center raw tree and apply fixed eta=0.05.
  canonical,_=center_sample(raw,b); push!(m.logamp.trees,scaled_tree(canonical,ETA_FIXED)); GBTQuantum.refresh_logamps!(la,m,S)
  for _=1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,m,S,la); end
  any(!isfinite,la) && break
 end; out
end
function writecsv(path,rows)
 isempty(rows)&&return; ns=propertynames(rows[1]); open(path,"w") do io; println(io,join(string.(ns),',')); for r in rows; println(io,join((getproperty(r,n) for n in ns),',')); end; end
end
function main()
 println("="^118); println("TFIM LEAF-REGULARIZATION COUNTERFACTUAL SWEEP"); println("N=$N Hilbert=$(1<<N), canonical trajectory samples=$NSAMPLES, fixed eta=$ETA_FIXED"); println("Same raw fitted tree at each checkpoint; regularize -> sample-center -> compare exact vs MC derivatives"); println("L2 lambdas=$LAMBDAS | clips=$CLIPS | min supports=$MIN_SUPPORTS"); println("="^118)
 X=allstates(N); rows=NamedTuple[]
 for r in sort(collect(keys(CHECKPOINTS))); append!(rows,train(GBTQuantum.TFIMHamiltonian(N;J=r,h=1.0,periodic=true),r,X)); end
 dir=joinpath(@__DIR__,"results"); mkpath(dir); path=joinpath(dir,"tfim_leaf_regularization_sweep.csv"); writecsv(path,rows); println("\nResults written to $path")
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMLeafRegularizationSweep.main(); end
