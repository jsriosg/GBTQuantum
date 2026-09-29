module TFIMRegularizedCappedNewtonSweep
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using GBTQuantum, Random, Statistics, Printf
include(joinpath(@__DIR__,"tfim_uncertainty_regularized_training.jl"))
const B=TFIMUncertaintyRegularizedTraining

# Minimal safeguard test after uncapped regularized Newton proved unstable.
# Three physical regimes are retained from now on as the standard TFIM sanity set.
const NS=[8,10,12]
const RATIOS=[0.5,1.0,2.0]
const SEEDS=[1234,2345,3456,4567,5678]
const ETA_CAPS=[0.20,0.30,0.40]
const LAMBDA=1.0
const M=512; const EPOCHS=150; const DEPTH=4
const ETA_FIXED=0.05
const CFLOOR=1e-12

function train_case(N,ratio,seed,mode; eta_cap=NaN)
 H=GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true)
 cfg=GBTQuantum.TrainingConfig(nsamples=M,epochs=EPOCHS,max_depth=DEPTH,eta=ETA_FIXED,
  burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=seed,exact_diagnostics=false)
 rng=MersenneTwister(seed); S=Matrix{Int8}(undef,M,N)
 for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
 m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); la=zeros(M)
 for _=1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,m,S,la); end

 etas=Float64[]; raw_newton=Float64[]; invalid=0; cap_hits=0; finite=true
 for ep=1:EPOCHS
  b=GBTQuantum.vmc_batch(H,m,S); y,_=GBTQuantum.make_targets(b)
  raw=B.grow_uncertainty_tree(b.states,y,b.counts;max_depth=DEPTH,
    min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain,lambda=LAMBDA)
  t,_=B.center_tree(raw,b)
  eta=ETA_FIXED
  if mode==:capped_newton
   mc=B.mc_derivatives(H,m,b,t)
   if isfinite(mc.g) && isfinite(mc.c) && mc.g<0 && mc.c>CFLOOR
    etaN=-mc.g/mc.c
    push!(raw_newton,etaN)
    if isfinite(etaN) && etaN>0
     if etaN>eta_cap; cap_hits+=1; end
     eta=min(etaN,eta_cap)
    else
     eta=ETA_FIXED; invalid+=1
    end
   else
    eta=ETA_FIXED; invalid+=1
   end
  end
  push!(etas,eta)
  push!(m.logamp.trees,B.scaled_tree(t,eta))
  GBTQuantum.refresh_logamps!(la,m,S)
  if any(!isfinite,la); finite=false; break; end
  for _=1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,m,S,la); end
  if any(!isfinite,la); finite=false; break; end
 end

 E=finite ? real(GBTQuantum.exact_model_energy(m,H).energy) : NaN
 Egs=GBTQuantum.exact_ground_energy(H)
 n=length(etas)
 (N=N,ratio=ratio,mode=String(mode),eta_cap=eta_cap,seed=seed,Egs=Egs,E_final=E,
  energy_error=E-Egs,finite=finite,epochs_completed=n,
  mean_eta=mean(etas),median_eta=median(etas),min_eta=minimum(etas),max_eta=maximum(etas),
  q90_eta=quantile(etas,0.90),q99_eta=quantile(etas,0.99),
  invalid_newton=invalid,cap_hits=cap_hits,cap_hit_fraction=cap_hits/n,
  raw_newton_max=isempty(raw_newton) ? NaN : maximum(raw_newton))
end

function writecsv(path,rows)
 ns=propertynames(rows[1])
 open(path,"w") do io
  println(io,join(string.(ns),','))
  for r in rows; println(io,join((getproperty(r,n) for n in ns),',')); end
 end
end

function main()
 println("="^136)
 println("TFIM UNCERTAINTY-REGULARIZED CAPPED-NEWTON SWEEP")
 println("N=$NS ratios=$RATIOS seeds=$(length(SEEDS)) lambda=$LAMBDA M=$M depth=$DEPTH epochs=$EPOCHS")
 println("control eta=$ETA_FIXED | capped Newton eta=min(-g_MC/c_MC, eta_max), eta_max=$ETA_CAPS")
 println("Invalid Newton proposals fall back to eta=$ETA_FIXED. No Armijo, ESS rule, or other adaptive safeguard.")
 println("="^136)
 rows=NamedTuple[]
 for N in NS,ratio in RATIOS
  # One fixed control per seed; do not redundantly rerun it for each cap.
  for seed in SEEDS
   r=train_case(N,ratio,seed,:fixed); push!(rows,r)
   @printf("N=%2d J/h=%.1f fixed seed=%d Eerr=% .4e finite=%s\n",
    N,ratio,seed,r.energy_error,string(r.finite))
  end
  for cap in ETA_CAPS,seed in SEEDS
   r=train_case(N,ratio,seed,:capped_newton;eta_cap=cap); push!(rows,r)
   @printf("N=%2d J/h=%.1f cap=%.2f seed=%d Eerr=% .4e finite=%s eta(mean/med)=% .4f/% .4f caphits=%d/%d (%.1f%%) invalid=%d rawmax=% .3g\n",
    N,ratio,cap,seed,r.energy_error,string(r.finite),r.mean_eta,r.median_eta,
    r.cap_hits,r.epochs_completed,100r.cap_hit_fraction,r.invalid_newton,r.raw_newton_max)
  end
 end

 dir=joinpath(@__DIR__,"results"); mkpath(dir)
 path=joinpath(dir,"tfim_regularized_capped_newton_sweep.csv"); writecsv(path,rows)

 println("\nPaired summaries:")
 for N in NS,ratio in RATIOS
  fixed=[r for r in rows if r.N==N && r.ratio==ratio && r.mode=="fixed"]
  ef=[r.energy_error for r in fixed]
  for cap in ETA_CAPS
   test=[r for r in rows if r.N==N && r.ratio==ratio && r.mode=="capped_newton" && r.eta_cap==cap]
   en=[r.energy_error for r in test]
   wins=sum(isfinite(en[i]) && en[i]<ef[i] for i in eachindex(ef))
   nf=sum(r.finite for r in test)
   @printf("N=%2d J/h=%.1f cap=%.2f fixed mean=% .4e capped mean=% .4e wins=%d/%d finite=%d/%d mean cap-hit=%.1f%%\n",
    N,ratio,cap,mean(ef),mean(en),wins,length(ef),nf,length(test),100mean(r.cap_hit_fraction for r in test))
  end
 end
 println("\nResults written to experiments/results/tfim_regularized_capped_newton_sweep.csv")
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMRegularizedCappedNewtonSweep.main(); end
