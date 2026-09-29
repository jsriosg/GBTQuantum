module TFIMRegularizationRiskInteraction
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using GBTQuantum, Random, Statistics, Printf
include(joinpath(@__DIR__,"tfim_uncertainty_regularized_training.jl"))
const BaseExp=TFIMUncertaintyRegularizedTraining

const N=8; const M=512; const EPOCHS=150; const DEPTH=4; const ETA=0.05; const SEED=1234
const RATIOS=[1.0,2.0]
const LAMBDAS=[0.0,0.10,0.25,0.50,1.0,2.0,4.0]
const CHECKPOINTS=Set([1,4,8,10,25,50,100,150])

function leafidx(t,x)
 i=1
 while true
  n=t.nodes[i]; n.isleaf && return i
  i=x[n.feature]<0 ? n.left : n.right
 end
end

# Production-only risk quantities from the current VMC batch.
# Rc = sigma(P_L) f_L^2
# RgP = 2 |f_L| sigma(P_L) |Ebar_L-E|
# RgE = 2 |f_L| Phat_L SE(Ebar_L)
function risk_stats(t,b)
 w=Float64.(b.counts); W=sum(w); el=real.(b.local_energy)
 E=sum(w.*el)/W
 groups=Dict{Int,Vector{Int}}()
 for j in axes(b.states,1)
  L=leafidx(t,@view b.states[j,:]); push!(get!(groups,L,Int[]),j)
 end
 rc=Float64[]; rgp=Float64[]; rge=Float64[]; rgq=Float64[]
 for (L,idx) in groups
  WL=sum(w[idx]); p=WL/W; f=t.nodes[L].value
  sig=sqrt(max(p*(1-p),0.0)/W)
  Ebar=sum(w[idx].*el[idx])/WL
  varE=sum(w[idx].*(el[idx].-Ebar).^2)/WL
  seE=sqrt(max(varE,0.0)/WL)
  a=sig*f^2
  bp=2abs(f)*sig*abs(Ebar-E)
  be=2abs(f)*p*seE
  push!(rc,a); push!(rgp,bp); push!(rge,be); push!(rgq,hypot(bp,be))
 end
 (max_rc=maximum(rc),sum_rc=sum(rc),
  max_rgp=maximum(rgp),sum_rgp=sum(rgp),
  max_rge=maximum(rge),sum_rge=sum(rge),
  max_rgq=maximum(rgq),sum_rgq=sum(rgq),
  nleaves=length(rc))
end

function train(H,ratio,lambda,X)
 cfg=GBTQuantum.TrainingConfig(nsamples=M,epochs=EPOCHS,max_depth=DEPTH,eta=ETA,
  burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=SEED,exact_diagnostics=false)
 rng=MersenneTwister(SEED); S=Matrix{Int8}(undef,M,H.N)
 for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
 m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); la=zeros(M)
 for _=1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,m,S,la); end
 rows=NamedTuple[]
 for ep=1:EPOCHS
  b=GBTQuantum.vmc_batch(H,m,S); y,_=GBTQuantum.make_targets(b)
  raw=BaseExp.grow_uncertainty_tree(b.states,y,b.counts;max_depth=cfg.max_depth,
   min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain,lambda=lambda)
  t,shift=BaseExp.center_tree(raw,b)
  if ep in CHECKPOINTS
   mc=BaseExp.mc_derivatives(H,m,b,t); ex=BaseExp.exact_derivatives(H,m,t,X); r=risk_stats(t,b)
   push!(rows,(ratio=ratio,lambda=lambda,epoch=ep,
    g_mc=mc.g,g_exact=ex.g,g_abs_error=abs(mc.g-ex.g),
    c_mc=mc.c,c_exact=ex.c,c_abs_error=abs(mc.c-ex.c),
    max_rc=r.max_rc,sum_rc=r.sum_rc,max_rgp=r.max_rgp,sum_rgp=r.sum_rgp,
    max_rge=r.max_rge,sum_rge=r.sum_rge,max_rgq=r.max_rgq,sum_rgq=r.sum_rgq,
    maxabs_f=BaseExp.leaf_risk_stats(t,b).maxabs,nleaves=r.nleaves,
    exact_mean_f=ex.meanf,sample_mean_f=mc.meanf,energy_before=ex.E,center_shift=shift))
  end
  push!(m.logamp.trees,BaseExp.scaled_tree(t,ETA)); GBTQuantum.refresh_logamps!(la,m,S)
  for _=1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,m,S,la); end
 end
 E=real(GBTQuantum.exact_model_energy(m,H).energy)
 rows,E
end

function writecsv(path,rows)
 ns=propertynames(rows[1])
 open(path,"w") do io
  println(io,join(string.(ns),','))
  for r in rows; println(io,join((getproperty(r,n) for n in ns),',')); end
 end
end

function main()
 println("="^132)
 println("TFIM REGULARIZATION / GRADIENT-CURVATURE RISK INTERACTION")
 println("Fixed eta=$ETA, N=$N, M=$M. Tree penalty remains lambda*sigma(P)*f^2.")
 println("Audit asks whether that same penalty suppresses Rc, RgP and RgE together.")
 println("="^132)
 X=BaseExp.allstates(N); rows=NamedTuple[]; summary=NamedTuple[]
 for ratio in RATIOS
  H=GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true)
  Egs=GBTQuantum.exact_ground_energy(H)
  println("\nJ/h=$ratio E_GS=$Egs")
  for lambda in LAMBDAS
   rr,E=train(H,ratio,lambda,X); append!(rows,rr)
   last=rr[end]
   @printf(" lambda=%4.2f Eerr=% .3e | ep150 |dg|=% .3e |dc|=% .3e Rc=% .3e RgP=% .3e RgE=% .3e\n",
    lambda,E-Egs,last.g_abs_error,last.c_abs_error,last.max_rc,last.max_rgp,last.max_rge)
   push!(summary,(ratio=ratio,lambda=lambda,Egs=Egs,E_final=E,energy_error=E-Egs,
    median_g_abs_error=median(getfield.(rr,:g_abs_error)),
    median_c_abs_error=median(getfield.(rr,:c_abs_error)),
    median_max_rc=median(getfield.(rr,:max_rc)),
    median_max_rgp=median(getfield.(rr,:max_rgp)),
    median_max_rge=median(getfield.(rr,:max_rge)),
    median_max_rgq=median(getfield.(rr,:max_rgq)),
    median_maxabs_f=median(getfield.(rr,:maxabs_f))))
  end
 end
 dir=joinpath(@__DIR__,"results"); mkpath(dir)
 writecsv(joinpath(dir,"tfim_regularization_risk_interaction_checkpoints.csv"),rows)
 writecsv(joinpath(dir,"tfim_regularization_risk_interaction_summary.csv"),summary)
 println("\nResults written to experiments/results/tfim_regularization_risk_interaction_{checkpoints,summary}.csv")
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMRegularizationRiskInteraction.main(); end
