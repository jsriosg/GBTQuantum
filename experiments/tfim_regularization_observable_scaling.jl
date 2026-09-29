module TFIMRegularizationObservableScaling
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using GBTQuantum, Random, Statistics, Printf
include(joinpath(@__DIR__,"tfim_uncertainty_regularized_training.jl"))
const BaseExp=TFIMUncertaintyRegularizedTraining

const NS=[8,10,12,14]
const RATIOS=[1.0,2.0]
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
  push!(m.logamp.trees,BaseExp.scaled_tree(t,ETA)); GBTQuantum.refresh_logamps!(la,m,S)
  for _=1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,m,S,la); end
  any(!isfinite,la) && error("non-finite log amplitudes: N=$(H.N), lambda=$lambda, seed=$seed")
 end
 return m
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
 println("TFIM FIXED-REGULARIZATION PHYSICAL-OBSERVABLE SCALING")
 println("Same frozen training: lambda=[0,1], eta=$ETA, M=$M, depth=$DEPTH, epochs=$EPOCHS, seeds=$(length(SEEDS)).")
 println("Observables use the existing repository definitions: <mz>, <|mz|>, <mz^2>, <mx>.")
 println("Exact ground-state/model observables are evaluation-only and never affect training.")
 println("="^132)
 rows=NamedTuple[]
 for N in NS, ratio in RATIOS
  H=GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true)
  gs=GBTQuantum.exact_ground_observables(H)
  @printf("\nN=%d Hilbert=%d J/h=%.1f\n",N,1<<N,ratio)
  @printf(" exact: mz=% .6e |mz|=% .8f mz2=% .8f mx=% .8f\n",gs.mz,gs.abs_mz,gs.mz2,gs.mx)
  for lambda in LAMBDAS, seed in SEEDS
   m=train_model(H,lambda,seed)
   E=real(GBTQuantum.exact_model_energy(m,H).energy)
   o=GBTQuantum.exact_model_observables(m,H)
   emz=abs(o.mz-gs.mz); eabs=abs(o.abs_mz-gs.abs_mz)
   emz2=abs(o.mz2-gs.mz2); emx=abs(real(o.mx)-gs.mx)
   push!(rows,(N=N,hilbert=1<<N,ratio=ratio,lambda=lambda,seed=seed,
    Egs=gs.energy,E_final=E,energy_error=E-gs.energy,
    mz_exact=gs.mz,mz_model=o.mz,mz_abs_error=emz,
    abs_mz_exact=gs.abs_mz,abs_mz_model=o.abs_mz,abs_mz_abs_error=eabs,
    mz2_exact=gs.mz2,mz2_model=o.mz2,mz2_abs_error=emz2,
    mx_exact=gs.mx,mx_model=real(o.mx),mx_abs_error=emx,mx_imag=imag(o.mx)))
   @printf(" lambda=%3.1f seed=%5d Eerr=% .3e  |dmz|=% .3e  |d|mz||=% .3e  |dmz2|=% .3e  |dmx|=% .3e\n",
    lambda,seed,E-gs.energy,emz,eabs,emz2,emx)
  end
  for lambda in LAMBDAS
   q=[r for r in rows if r.N==N && r.ratio==ratio && r.lambda==lambda]
   @printf(" SUMMARY lambda=%3.1f mean |dmz|=% .3e |d|mz||=% .3e |dmz2|=% .3e |dmx|=% .3e\n",
    lambda,mean(r.mz_abs_error for r in q),mean(r.abs_mz_abs_error for r in q),
    mean(r.mz2_abs_error for r in q),mean(r.mx_abs_error for r in q))
  end
 end
 dir=joinpath(@__DIR__,"results"); mkpath(dir)
 writecsv(joinpath(dir,"tfim_regularization_observable_scaling.csv"),rows)
 println("\nResults written to experiments/results/tfim_regularization_observable_scaling.csv")
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMRegularizationObservableScaling.main(); end
