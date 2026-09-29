module TFIMRegularizedShrunkNewtonSweep
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using GBTQuantum, Random, Statistics, Printf
include(joinpath(@__DIR__,"tfim_uncertainty_regularized_training.jl"))
const B=TFIMUncertaintyRegularizedTraining

# Final optimizer ablation: keep boosting shrinkage nu separate from the
# sampled Newton line scale eta_N=-g/c. Compare uncapped and capped modulation.
const NS=[8,10,12]
const RATIOS=[0.5,1.0,2.0]
const SEEDS=[1234,2345,3456,4567,5678]
const NUS=[0.05,0.10,0.20]
const ETA_MAX=0.40
const LAMBDA=1.0
const M=512; const EPOCHS=150; const DEPTH=4
const ETA_FIXED=0.05
const CFLOOR=1e-12

function train_case(N,ratio,seed,mode; nu=NaN)
 H=GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true)
 cfg=GBTQuantum.TrainingConfig(nsamples=M,epochs=EPOCHS,max_depth=DEPTH,eta=ETA_FIXED,
  burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=seed,exact_diagnostics=false)
 rng=MersenneTwister(seed); S=Matrix{Int8}(undef,M,N)
 for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
 m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); la=zeros(M)

 for _=1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,m,S,la); end
 effetas=Float64[]; rawetas=Float64[]; invalid=0; caphits=0; finite=true

 for ep=1:EPOCHS
  b=GBTQuantum.vmc_batch(H,m,S); y,_=GBTQuantum.make_targets(b)
  raw=B.grow_uncertainty_tree(b.states,y,b.counts;max_depth=DEPTH,
    min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain,lambda=LAMBDA)
  t,_=B.center_tree(raw,b)

  etaeff=ETA_FIXED
  if mode!=:fixed
   mc=B.mc_derivatives(H,m,b,t)
   if isfinite(mc.g) && isfinite(mc.c) && mc.g<0 && mc.c>CFLOOR
    etaN=-mc.g/mc.c
    if isfinite(etaN) && etaN>0
     push!(rawetas,etaN)
     if mode==:shrunk_newton
      etaeff=nu*etaN
     elseif mode==:capped_shrunk_newton
      caphits += etaN>ETA_MAX
      etaeff=nu*min(etaN,ETA_MAX)
     end
    else
     invalid+=1; etaeff=ETA_FIXED
    end
   else
    invalid+=1; etaeff=ETA_FIXED
   end
  end

  push!(effetas,etaeff)
  push!(m.logamp.trees,B.scaled_tree(t,etaeff))
  GBTQuantum.refresh_logamps!(la,m,S)
  if any(!isfinite,la); finite=false; break; end
  for _=1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,m,S,la); end
  if any(!isfinite,la); finite=false; break; end
 end

 E=finite ? real(GBTQuantum.exact_model_energy(m,H).energy) : NaN
 Egs=GBTQuantum.exact_ground_energy(H); n=length(effetas)
 (N=N,ratio=ratio,mode=String(mode),nu=nu,eta_max=(mode==:capped_shrunk_newton ? ETA_MAX : NaN),
  seed=seed,Egs=Egs,E_final=E,energy_error=E-Egs,finite=finite,epochs_completed=n,
  mean_effective_eta=mean(effetas),median_effective_eta=median(effetas),
  min_effective_eta=minimum(effetas),max_effective_eta=maximum(effetas),
  q90_effective_eta=quantile(effetas,0.90),q99_effective_eta=quantile(effetas,0.99),
  mean_raw_newton=isempty(rawetas) ? NaN : mean(rawetas),
  median_raw_newton=isempty(rawetas) ? NaN : median(rawetas),
  max_raw_newton=isempty(rawetas) ? NaN : maximum(rawetas),
  invalid_newton=invalid,invalid_fraction=invalid/n,
  cap_hits=caphits,cap_hit_fraction=caphits/n)
end

function writecsv(path,rows)
 ns=propertynames(rows[1])
 open(path,"w") do io
  println(io,join(string.(ns),','))
  for r in rows; println(io,join((getproperty(r,n) for n in ns),',')); end
 end
end

function main()
 println("="^142)
 println("TFIM UNCERTAINTY-REGULARIZED SHRUNK-NEWTON SWEEP")
 println("N=$NS ratios=$RATIOS seeds=$(length(SEEDS)) lambda=$LAMBDA M=$M depth=$DEPTH epochs=$EPOCHS")
 println("fixed: eta_eff=$ETA_FIXED")
 println("shrunk Newton: eta_eff = nu*(-g_MC/c_MC), nu=$NUS")
 println("capped + shrunk Newton: eta_eff = nu*min(-g_MC/c_MC,$ETA_MAX)")
 println("Invalid Newton proposals fall back to eta_eff=$ETA_FIXED. No Armijo or ESS rule.")
 println("="^142)

 rows=NamedTuple[]
 for N in NS,ratio in RATIOS
  for seed in SEEDS
   r=train_case(N,ratio,seed,:fixed); push!(rows,r)
   @printf("N=%2d J/h=%.1f fixed seed=%d Eerr=% .4e finite=%s\n",
    N,ratio,seed,r.energy_error,string(r.finite))
  end
  for mode in [:shrunk_newton,:capped_shrunk_newton],nu in NUS,seed in SEEDS
   r=train_case(N,ratio,seed,mode;nu=nu); push!(rows,r)
   @printf("N=%2d J/h=%.1f %-21s nu=%.2f seed=%d Eerr=% .4e finite=%s eta_eff(mean/med/max)=% .4f/% .4f/% .4f rawmax=% .3g cap=%d/%d invalid=%d\n",
    N,ratio,String(mode),nu,seed,r.energy_error,string(r.finite),
    r.mean_effective_eta,r.median_effective_eta,r.max_effective_eta,r.max_raw_newton,
    r.cap_hits,r.epochs_completed,r.invalid_newton)
  end
 end

 dir=joinpath(@__DIR__,"results"); mkpath(dir)
 path=joinpath(dir,"tfim_regularized_shrunk_newton_sweep.csv"); writecsv(path,rows)

 println("\nPaired summaries:")
 for N in NS,ratio in RATIOS
  fixed=[r for r in rows if r.N==N && r.ratio==ratio && r.mode=="fixed"]
  ef=[r.energy_error for r in fixed]
  @printf("\nN=%d J/h=%.1f fixed mean=% .4e\n",N,ratio,mean(ef))
  for mode in ["shrunk_newton","capped_shrunk_newton"],nu in NUS
   test=[r for r in rows if r.N==N && r.ratio==ratio && r.mode==mode && r.nu==nu]
   en=[r.energy_error for r in test]
   wins=sum(isfinite(en[i]) && en[i]<ef[i] for i in eachindex(ef))
   nf=sum(r.finite for r in test)
   @printf("  %-21s nu=%.2f mean=% .4e wins=%d/%d finite=%d/%d mean_eta=% .4f mean_invalid=%.1f%%",
    mode,nu,mean(en),wins,length(ef),nf,length(test),mean(r.mean_effective_eta for r in test),
    100mean(r.invalid_fraction for r in test))
   if mode=="capped_shrunk_newton"
    @printf(" mean_cap_hit=%.1f%%",100mean(r.cap_hit_fraction for r in test))
   end
   println()
  end
 end
 println("\nResults written to experiments/results/tfim_regularized_shrunk_newton_sweep.csv")
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMRegularizedShrunkNewtonSweep.main(); end
