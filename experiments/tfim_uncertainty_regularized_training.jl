module TFIMUncertaintyRegularizedTraining
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using GBTQuantum, Random, Statistics, Printf

const N=8; const NSAMPLES=512; const EPOCHS=150; const MAX_DEPTH=4
const ETA_FIXED=0.05; const ETA_CAP=0.40; const SEED=1234
const RATIOS=[1.0,2.0]
const LAMBDAS=[0.0,0.10,0.25,0.50,1.0,2.0,4.0]
const MODES=[:fixed,:adaptive]
const CHECKPOINTS=Set([1,4,8,10,25,50,100,150])
const CFLOOR=1e-12

# Objective convention:
# ordinary grow_tree solves weighted LS with leaf f=S/W and score S^2/W.
# We add to the NORMALIZED loss lambda*sigma(P_L)*f_L^2.
# Multiplying back by M=sum(weights) gives
#   f_L = S_L / (W_L + 2 lambda M sigma_L),
#   sigma_L = sqrt(P_hat_L(1-P_hat_L)/M), P_hat_L=W_L/M.
# The same penalized leaf optimum/score is used while choosing splits.

@inline function denom(W,M,lambda)
 p=clamp(W/M,0.0,1.0); sig=sqrt(max(p*(1-p),0.0)/M)
 W + 2lambda*M*sig
end

function grow_uncertainty_tree(X,y,w; max_depth=4,min_weight=1.0,min_gain=0.0,lambda=0.0)
 nobs,nfeatures=size(X); M=sum(Float64.(w)); nodes=GBTQuantum.Node[]
 idx0=collect(1:nobs); W0=M; S0=sum(Float64(w[i])*Float64(y[i]) for i=1:nobs)
 function build(idx,depth,W,S)
  D=denom(W,M,lambda); pos=Int32(length(nodes)+1)
  push!(nodes,GBTQuantum.Node(0,S/D,0,0,true))
  (depth>=max_depth || length(idx)<=1 || W<2min_weight) && return pos
  parent=S*S/D; best=min_gain; bf=0; bWL=0.0; bSL=0.0
  for f=1:nfeatures
   WL=0.0; SL=0.0
   for i in idx
    if X[i,f]<0; wi=Float64(w[i]); WL+=wi; SL+=wi*Float64(y[i]); end
   end
   WR=W-WL; (WL<min_weight || WR<min_weight) && continue
   SR=S-SL
   gain=SL*SL/denom(WL,M,lambda)+SR*SR/denom(WR,M,lambda)-parent
   if gain>best; best=gain; bf=f; bWL=WL; bSL=SL; end
  end
  bf==0 && return pos
  li=Int[]; ri=Int[]
  for i in idx; (X[i,bf]<0 ? push!(li,i) : push!(ri,i)); end
  left=build(li,depth+1,bWL,bSL); right=build(ri,depth+1,W-bWL,S-bSL)
  nodes[pos]=GBTQuantum.Node(Int32(bf),0.0,left,right,false); pos
 end
 build(idx0,0,W0,S0); GBTQuantum.RegressionTree(nodes)
end

function center_tree(t,b)
 w=Float64.(b.counts); p=[GBTQuantum.predict(t,@view b.states[j,:]) for j in axes(b.states,1)]
 mu=GBTQuantum.weighted_mean(p,w); nd=copy(t.nodes)
 for i in eachindex(nd); n=nd[i]; n.isleaf && (nd[i]=GBTQuantum.Node(n.feature,n.value-mu,n.left,n.right,true)); end
 GBTQuantum.RegressionTree(nd),mu
end
scaled_tree(t,e)=GBTQuantum.RegressionTree([n.isleaf ? GBTQuantum.Node(n.feature,e*n.value,n.left,n.right,true) : n for n in t.nodes])

function f_local(H,m,t,x)
 A=GBTQuantum.logamplitude(m,x); f0=GBTQuantum.predict(t,x); z=GBTQuantum.diagonal(H,x)*f0
 for i=1:H.N
  x[i]=-x[i]; z-=H.h*exp(GBTQuantum.logamplitude(m,x)-A)*GBTQuantum.predict(t,x); x[i]=-x[i]
 end
 z
end

function mc_derivatives(H,m,b,t)
 w=Float64.(b.counts); W=sum(w); f=[GBTQuantum.predict(t,@view b.states[j,:]) for j in axes(b.states,1)]
 el=real.(b.local_energy); Ef=sum(w .* f)/W; E=sum(w.* el)/W
 g=2sum(w.*(f .- Ef).*(el .- E))/W
 f2=sum(w .* f .^ 2)/W; f2e=sum(w .* f .^ 2.* el)/W
 q=sum(w[j]*f[j]*f_local(H,m,t,@view b.states[j,:]) for j in axes(b.states,1))/W
 c=2f2e+2q-4E*f2-4Ef*g
 (g=g,c=c,meanf=Ef)
end

@inline flip(j,i)=((j-1) ⊻ (1<<(i-1)))+1
function allstates(N)
 X=Matrix{Int8}(undef,1<<N,N)
 for s=0:(1<<N)-1,i=1:N; X[s+1,i]=((s>>(i-1))&1)==1 ? Int8(1) : Int8(-1); end
 X
end

function exact_derivatives(H,m,t,X)
 d=size(X,1); A=zeros(d); f=zeros(d)
 for j=1:d; A[j]=GBTQuantum.logamplitude(m,@view X[j,:]); f[j]=GBTQuantum.predict(t,@view X[j,:]); end
 p=exp.(2 .* A .- maximum(2 .* A)); p ./= sum(p); el=zeros(d); hf=zeros(d)
 for j=1:d
  x=@view X[j,:]; el[j]=GBTQuantum.diagonal(H,x); hf[j]=GBTQuantum.diagonal(H,x)*f[j]
  for i=1:H.N
   k=flip(j,i); r=exp(A[k]-A[j]); el[j]-=H.h*r; hf[j]-=H.h*r*f[k]
  end
 end
 Ef=sum(p .* f); E=sum(p.* el); g=2sum(p.*(f .- Ef).*(el .- E))
 c=2sum(p .* f .^ 2.* el)+2sum(p .* f.* hf)-4E*sum(p .* f .^ 2)-4Ef*g
 (g=g,c=c,meanf=Ef,E=E)
end

function leaf_risk_stats(t,b)
 w=Float64.(b.counts); M=sum(w); support=Dict{Int,Float64}()
 function leafidx(x)
  i=1
  while true
   n=t.nodes[i]; n.isleaf && return i
   i=x[n.feature]<0 ? n.left : n.right
  end
 end
 for j in axes(b.states,1); L=leafidx(@view b.states[j,:]); support[L]=get(support,L,0.0)+w[j]; end
 risks=Float64[]; vals=Float64[]
 for (L,W) in support
  f=t.nodes[L].value; p=W/M; sig=sqrt(max(p*(1-p),0.0)/M)
  push!(risks,sig*f^2); push!(vals,abs(f))
 end
 (maxrisk=maximum(risks),sumrisk=sum(risks),maxabs=maximum(vals),nleaves=length(vals))
end

function train(H,ratio,lambda,mode,X)
 cfg=GBTQuantum.TrainingConfig(nsamples=NSAMPLES,epochs=EPOCHS,max_depth=MAX_DEPTH,eta=ETA_FIXED,burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=SEED,exact_diagnostics=false)
 rng=MersenneTwister(SEED); S=Matrix{Int8}(undef,NSAMPLES,H.N)
 for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
 m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); la=zeros(NSAMPLES)
 for _=1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,m,S,la); end
 rows=NamedTuple[]; fallbacks=0
 for ep=1:EPOCHS
  b=GBTQuantum.vmc_batch(H,m,S); y,_=GBTQuantum.make_targets(b)
  raw=grow_uncertainty_tree(b.states,y,b.counts;max_depth=cfg.max_depth,min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain,lambda=lambda)
  t,shift=center_tree(raw,b); mc=mc_derivatives(H,m,b,t); risk=leaf_risk_stats(t,b)
  ex = ep in CHECKPOINTS ? exact_derivatives(H,m,t,X) : nothing
  eta=ETA_FIXED; fallback=false
  if mode==:adaptive
   if isfinite(mc.g)&&isfinite(mc.c)&&mc.g<0&&mc.c>CFLOOR
    eta=clamp(-mc.g/mc.c,0.0,ETA_CAP)
    if !isfinite(eta)||eta<=0; eta=ETA_FIXED; fallback=true; end
   else
    eta=ETA_FIXED; fallback=true
   end
   fallbacks += fallback
  end
  if ep in CHECKPOINTS
   gerr=mc.g-ex.g; cerr=mc.c-ex.c
   push!(rows,(ratio=ratio,lambda=lambda,mode=String(mode),epoch=ep,eta=eta,fallback=fallback,g_mc=mc.g,g_exact=ex.g,g_abs_error=abs(gerr),g_rel_error=abs(gerr)/(abs(ex.g)+eps()),c_mc=mc.c,c_exact=ex.c,c_abs_error=abs(cerr),c_rel_error=abs(cerr)/(abs(ex.c)+eps()),sample_mean_f=mc.meanf,exact_mean_f=ex.meanf,max_leaf_risk=risk.maxrisk,sum_leaf_risk=risk.sumrisk,maxabs_f=risk.maxabs,nleaves=risk.nleaves,center_shift=shift,energy_before=ex.E))
  end
  push!(m.logamp.trees,scaled_tree(t,eta)); GBTQuantum.refresh_logamps!(la,m,S)
  for _=1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,m,S,la); end
  any(!isfinite,la) && break
 end
 E=real(GBTQuantum.exact_model_energy(m,H).energy)
 (rows=rows,E=E,fallbacks=fallbacks,finite=isfinite(E))
end

function writecsv(path,rows)
 isempty(rows)&&return; ns=propertynames(rows[1])
 open(path,"w") do io
  println(io,join(string.(ns),','))
  for r in rows; println(io,join((getproperty(r,n) for n in ns),',')); end
 end
end

function main()
 println("="^124)
 println("TFIM UNCERTAINTY-REGULARIZED END-TO-END TRAINING SWEEP")
 println("Penalty: lambda * sigma(P_hat_L) * f_L^2 in normalized tree loss")
 println("Equivalent unnormalized leaf denominator: W_L + 2 lambda M sigma_L")
 println("Ratios=$RATIOS lambdas=$LAMBDAS modes=$MODES N=$N M=$NSAMPLES epochs=$EPOCHS")
 println("Gradient error is LOGGED as a guardrail; this experiment is designed around curvature risk.")
 println("="^124)
 X=allstates(N); rows=NamedTuple[]; summary=NamedTuple[]
 for ratio in RATIOS
  H=GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true); Egs=GBTQuantum.exact_ground_energy(H)
  println("\nJ/h=$ratio  E_GS=$(Egs)")
  for mode in MODES, lambda in LAMBDAS
   r=train(H,ratio,lambda,mode,X); append!(rows,r.rows)
   err=r.E-Egs
   @printf("  %-8s lambda=%5.2f  E=% .10f err=% .3e finite=%s fallbacks=%d\n",String(mode),lambda,r.E,err,string(r.finite),r.fallbacks)
   push!(summary,(ratio=ratio,lambda=lambda,mode=String(mode),Egs=Egs,E_final=r.E,energy_error=err,finite=r.finite,fallbacks=r.fallbacks))
  end
 end
 dir=joinpath(@__DIR__,"results"); mkpath(dir)
 writecsv(joinpath(dir,"tfim_uncertainty_regularized_training_checkpoints.csv"),rows)
 writecsv(joinpath(dir,"tfim_uncertainty_regularized_training_summary.csv"),summary)
 println("\nResults written to:")
 println("  experiments/results/tfim_uncertainty_regularized_training_checkpoints.csv")
 println("  experiments/results/tfim_uncertainty_regularized_training_summary.csv")
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMUncertaintyRegularizedTraining.main(); end
