module TFIMRegularizationSeedRobustness
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using GBTQuantum, Random, Statistics, Printf
include(joinpath(@__DIR__,"tfim_uncertainty_regularized_training.jl"))
const BaseExp=TFIMUncertaintyRegularizedTraining

const N=8; const M=512; const EPOCHS=150; const DEPTH=4; const ETA=0.05
const RATIOS=[1.0,2.0]
const LAMBDAS=[0.0,0.5,1.0]
const SEEDS=[1234,2345,3456,4567,5678,6789,7890,8901,9012,10123]

function train(H,lambda,seed)
 cfg=GBTQuantum.TrainingConfig(nsamples=M,epochs=EPOCHS,max_depth=DEPTH,eta=ETA,
  burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=seed,exact_diagnostics=false)
 rng=MersenneTwister(seed); S=Matrix{Int8}(undef,M,H.N)
 for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
 m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); la=zeros(M)
 for _=1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,m,S,la); end
 for ep=1:EPOCHS
  b=GBTQuantum.vmc_batch(H,m,S); y,_=GBTQuantum.make_targets(b)
  raw=BaseExp.grow_uncertainty_tree(b.states,y,b.counts;max_depth=cfg.max_depth,
   min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain,lambda=lambda)
  t,_=BaseExp.center_tree(raw,b)
  push!(m.logamp.trees,BaseExp.scaled_tree(t,ETA)); GBTQuantum.refresh_logamps!(la,m,S)
  for _=1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,m,S,la); end
  any(!isfinite,la) && break
 end
 real(GBTQuantum.exact_model_energy(m,H).energy)
end

function writecsv(path,rows)
 ns=propertynames(rows[1])
 open(path,"w") do io
  println(io,join(string.(ns),','))
  for r in rows; println(io,join((getproperty(r,n) for n in ns),',')); end
 end
end

function main()
 println("="^116)
 println("TFIM UNCERTAINTY-REGULARIZATION SEED ROBUSTNESS")
 println("Simple validation only: fixed eta=$ETA, lambda=$LAMBDAS, seeds=$(length(SEEDS)), N=$N, M=$M.")
 println("Exact ground state is used only for final evaluation, never for training.")
 println("="^116)
 rows=NamedTuple[]
 for ratio in RATIOS
  H=GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true)
  Egs=GBTQuantum.exact_ground_energy(H)
  println("\nJ/h=$ratio E_GS=$Egs")
  for lambda in LAMBDAS
   errs=Float64[]
   for seed in SEEDS
    E=train(H,lambda,seed); err=E-Egs; push!(errs,err)
    push!(rows,(ratio=ratio,lambda=lambda,seed=seed,Egs=Egs,E_final=E,energy_error=err))
    @printf(" lambda=%3.1f seed=%5d Eerr=% .6e\n",lambda,seed,err)
   end
   @printf(" SUMMARY lambda=%3.1f mean=% .6e median=% .6e std=% .6e min=% .6e max=% .6e\n",
    lambda,mean(errs),median(errs),std(errs),minimum(errs),maximum(errs))
  end
 end
 dir=joinpath(@__DIR__,"results"); mkpath(dir)
 writecsv(joinpath(dir,"tfim_regularization_seed_robustness.csv"),rows)
 println("\nResults written to experiments/results/tfim_regularization_seed_robustness.csv")
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMRegularizationSeedRobustness.main(); end
