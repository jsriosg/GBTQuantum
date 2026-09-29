module TFIMFinalRegularizationObservableComparison
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using GBTQuantum, Random, Statistics, Printf
include(joinpath(@__DIR__,"tfim_uncertainty_regularized_training.jl"))
const BaseExp=TFIMUncertaintyRegularizedTraining

# Definitive TFIM algorithm comparison:
# original GBT-VMC (lambda=0) vs uncertainty-regularized GBT-VMC (lambda=1).
# Both use the same fixed boosting shrinkage eta=0.05; no Newton/adaptive step.
const NS=[8,10,12,14]
const RATIOS=[0.5,1.0,2.0]
const LAMBDAS=[0.0,1.0]
const SEEDS=[1234,2345,3456,4567,5678]
const M=512; const EPOCHS=150; const DEPTH=4; const ETA=0.05

function train_model(H,lambda,seed)
 cfg=GBTQuantum.TrainingConfig(nsamples=M,epochs=EPOCHS,max_depth=DEPTH,eta=ETA,
  burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=seed,exact_diagnostics=false)
 rng=MersenneTwister(seed); S=Matrix{Int8}(undef,M,H.N)
 for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
 m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); la=zeros(M)
 for _=1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,m,S,la); end
 for _=1:EPOCHS
  b=GBTQuantum.vmc_batch(H,m,S); y,_=GBTQuantum.make_targets(b)
  raw=BaseExp.grow_uncertainty_tree(b.states,y,b.counts;max_depth=cfg.max_depth,
   min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain,lambda=lambda)
  t,_=BaseExp.center_tree(raw,b)
  push!(m.logamp.trees,BaseExp.scaled_tree(t,ETA))
  GBTQuantum.refresh_logamps!(la,m,S)
  for _=1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,m,S,la); end
  any(!isfinite,la) && error("non-finite log amplitudes: N=$(H.N), ratio=$(H.J/H.h), lambda=$lambda, seed=$seed")
 end
 m
end

function writecsv(path,rows)
 ns=propertynames(rows[1])
 open(path,"w") do io
  println(io,join(string.(ns),','))
  for r in rows; println(io,join((getproperty(r,n) for n in ns),',')); end
 end
end

function main()
 println("="^140)
 println("FINAL TFIM: ORIGINAL vs UNCERTAINTY-REGULARIZED GBT-VMC")
 println("N=$NS  J/h=$RATIOS  lambda=$LAMBDAS  seeds=$(length(SEEDS))")
 println("Both models: fixed eta=$ETA, M=$M, depth=$DEPTH, epochs=$EPOCHS. No Newton/adaptive step.")
 println("Final exact energy and observables are evaluation-only and never affect training.")
 println("Observables: <mz>, <|mz|>, <mz^2>, <mx>.")
 println("="^140)

 rows=NamedTuple[]
 for N in NS,ratio in RATIOS
  H=GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true)
  gs=GBTQuantum.exact_ground_observables(H)
  @printf("\nN=%d Hilbert=%d J/h=%.1f\n",N,1<<N,ratio)
  @printf(" exact: E=% .10f mz=% .6e |mz|=% .8f mz2=% .8f mx=% .8f\n",
   gs.energy,gs.mz,gs.abs_mz,gs.mz2,gs.mx)

  for lambda in LAMBDAS,seed in SEEDS
   m=train_model(H,lambda,seed)
   E=real(GBTQuantum.exact_model_energy(m,H).energy)
   o=GBTQuantum.exact_model_observables(m,H)
   push!(rows,(N=N,hilbert=1<<N,ratio=ratio,lambda=lambda,seed=seed,
    Egs=gs.energy,E_final=E,energy_error=E-gs.energy,
    mz_exact=gs.mz,mz_model=o.mz,mz_abs_error=abs(o.mz-gs.mz),
    abs_mz_exact=gs.abs_mz,abs_mz_model=o.abs_mz,abs_mz_abs_error=abs(o.abs_mz-gs.abs_mz),
    mz2_exact=gs.mz2,mz2_model=o.mz2,mz2_abs_error=abs(o.mz2-gs.mz2),
    mx_exact=gs.mx,mx_model=real(o.mx),mx_abs_error=abs(real(o.mx)-gs.mx),mx_imag=imag(o.mx)))
   @printf(" lambda=%3.1f seed=%5d Eerr=% .3e |dmz|=% .3e |d|mz||=% .3e |dmz2|=% .3e |dmx|=% .3e\n",
    lambda,seed,E-gs.energy,abs(o.mz-gs.mz),abs(o.abs_mz-gs.abs_mz),
    abs(o.mz2-gs.mz2),abs(real(o.mx)-gs.mx))
  end

  for lambda in LAMBDAS
   q=[r for r in rows if r.N==N && r.ratio==ratio && r.lambda==lambda]
   @printf(" SUMMARY lambda=%3.1f Eerr=% .3e +/- %.2e |dmz|=% .3e |d|mz||=% .3e |dmz2|=% .3e |dmx|=% .3e\n",
    lambda,mean(r.energy_error for r in q),std(r.energy_error for r in q),
    mean(r.mz_abs_error for r in q),mean(r.abs_mz_abs_error for r in q),
    mean(r.mz2_abs_error for r in q),mean(r.mx_abs_error for r in q))
  end

  q0=[r for r in rows if r.N==N && r.ratio==ratio && r.lambda==0.0]
  q1=[r for r in rows if r.N==N && r.ratio==ratio && r.lambda==1.0]
  @printf(" PAIRED regularized wins: energy=%d/5 |mz|=%d/5 mz2=%d/5 mx=%d/5\n",
   sum(q1[i].energy_error<q0[i].energy_error for i=1:length(SEEDS)),
   sum(q1[i].abs_mz_abs_error<q0[i].abs_mz_abs_error for i=1:length(SEEDS)),
   sum(q1[i].mz2_abs_error<q0[i].mz2_abs_error for i=1:length(SEEDS)),
   sum(q1[i].mx_abs_error<q0[i].mx_abs_error for i=1:length(SEEDS)))
 end

 dir=joinpath(@__DIR__,"results"); mkpath(dir)
 path=joinpath(dir,"tfim_final_regularization_observable_comparison.csv")
 writecsv(path,rows)
 println("\nResults written to $path")
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMFinalRegularizationObservableComparison.main(); end
