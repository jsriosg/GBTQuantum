module TFIMRegularizedNewtonEndToEnd
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using GBTQuantum, Random, Statistics, Printf
include(joinpath(@__DIR__,"tfim_uncertainty_regularized_training.jl"))
const B=TFIMUncertaintyRegularizedTraining

# First clean end-to-end test of the uncertainty-regularized Newton step.
# Deliberately NO eta_max: the point is to learn whether regularization itself
# repaired the sampled Newton proposal before adding another safeguard.
const NS=[8,10,12]
const RATIOS=[1.0,2.0]
const SEEDS=[1234,2345,3456,4567,5678]
const MODES=[:fixed,:newton]
const LAMBDA=1.0
const M=512; const EPOCHS=150; const DEPTH=4
const ETA_FIXED=0.05
const CFLOOR=1e-12

function train_case(N,ratio,seed,mode)
 H=GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true)
 cfg=GBTQuantum.TrainingConfig(nsamples=M,epochs=EPOCHS,max_depth=DEPTH,eta=ETA_FIXED,
  burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=seed,exact_diagnostics=false)
 rng=MersenneTwister(seed); S=Matrix{Int8}(undef,M,N)
 for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
 m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); la=zeros(M)
 for _=1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,m,S,la); end

 etas=Float64[]; invalid=0; finite=true
 for ep=1:EPOCHS
  b=GBTQuantum.vmc_batch(H,m,S); y,_=GBTQuantum.make_targets(b)
  raw=B.grow_uncertainty_tree(b.states,y,b.counts;max_depth=DEPTH,
    min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain,lambda=LAMBDA)
  t,_=B.center_tree(raw,b)
  eta=ETA_FIXED
  if mode==:newton
   mc=B.mc_derivatives(H,m,b,t)
   if isfinite(mc.g) && isfinite(mc.c) && mc.g<0 && mc.c>CFLOOR
    eta=-mc.g/mc.c
    if !(isfinite(eta) && eta>0)
     eta=ETA_FIXED; invalid+=1
    end
   else
    eta=ETA_FIXED; invalid+=1
   end
  end
  push!(etas,eta)
  push!(m.logamp.trees,B.scaled_tree(t,eta))
  GBTQuantum.refresh_logamps!(la,m,S)
  if any(!isfinite,la)
   finite=false; break
  end
  for _=1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,m,S,la); end
  if any(!isfinite,la)
   finite=false; break
  end
 end

 E=finite ? real(GBTQuantum.exact_model_energy(m,H).energy) : NaN
 Egs=GBTQuantum.exact_ground_energy(H)
 (N=N,ratio=ratio,mode=String(mode),seed=seed,Egs=Egs,E_final=E,
  energy_error=E-Egs,finite=finite,epochs_completed=length(etas),
  mean_eta=mean(etas),median_eta=median(etas),min_eta=minimum(etas),max_eta=maximum(etas),
  q90_eta=quantile(etas,0.90),q99_eta=quantile(etas,0.99),invalid_newton=invalid)
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
 println("TFIM UNCERTAINTY-REGULARIZED NEWTON END-TO-END TEST")
 println("N=$NS ratios=$RATIOS seeds=$(length(SEEDS)) lambda=$LAMBDA M=$M depth=$DEPTH epochs=$EPOCHS")
 println("fixed: eta=$ETA_FIXED | newton: eta=-g_MC/c_MC when g<0,c>0; invalid proposals fall back to $ETA_FIXED")
 println("IMPORTANT: Newton eta is UNcapped. eta_max is intentionally deferred to a later safeguard experiment.")
 println("="^132)
 rows=NamedTuple[]
 for N in NS,ratio in RATIOS,mode in MODES,seed in SEEDS
  r=train_case(N,ratio,seed,mode); push!(rows,r)
  @printf("N=%2d J/h=%.1f %-6s seed=%d Eerr=% .4e finite=%s eta(mean/med/max)=% .4f/% .4f/% .4f invalid=%d\n",
   N,ratio,String(mode),seed,r.energy_error,string(r.finite),r.mean_eta,r.median_eta,r.max_eta,r.invalid_newton)
 end
 dir=joinpath(@__DIR__,"results"); mkpath(dir)
 path=joinpath(dir,"tfim_regularized_newton_end_to_end.csv"); writecsv(path,rows)
 println("\nPaired summaries:")
 for N in NS,ratio in RATIOS
  a=[r for r in rows if r.N==N && r.ratio==ratio && r.mode=="fixed"]
  b=[r for r in rows if r.N==N && r.ratio==ratio && r.mode=="newton"]
  ef=[r.energy_error for r in a]; en=[r.energy_error for r in b]
  wins=sum(en[i]<ef[i] for i in eachindex(ef))
  @printf("N=%2d J/h=%.1f fixed mean=% .4e Newton mean=% .4e wins=%d/%d\n",
   N,ratio,mean(ef),mean(en),wins,length(ef))
 end
 println("\nResults written to experiments/results/tfim_regularized_newton_end_to_end.csv")
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMRegularizedNewtonEndToEnd.main(); end
