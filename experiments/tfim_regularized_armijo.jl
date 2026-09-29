module TFIMRegularizedArmijo
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using GBTQuantum, Random, Statistics, Printf
include(joinpath(@__DIR__,"tfim_uncertainty_regularized_training.jl"))
const BaseExp=TFIMUncertaintyRegularizedTraining

const NS=[8,10,12,14]
const RATIOS=[1.0,2.0]
const SEEDS=[1234,2345,3456,4567,5678]
const MODES=[:fixed,:armijo]
const LAMBDA=1.0; const M=512; const EPOCHS=150; const DEPTH=4
const ETA_FIXED=0.05
# Minimal Armijo: start from a moderate trial step and only backtrack.
const ETA0=0.20; const BETA=0.5; const ARMIJO_ALPHA=0.1; const MAX_BACKTRACKS=8

function trial_energy(H,m,t,b,eta)
 # Self-normalized reweighting from the CURRENT Born sample:
 # p_eta/p_0 is proportional to exp(2 eta f).
 f=[GBTQuantum.predict(t,@view b.states[j,:]) for j in axes(b.states,1)]
 z=2eta .* f; zmax=maximum(z); r=exp.(z .- zmax)
 w=Float64.(b.counts); wr=w .* r
 sw=sum(wr); sw2=sum(w .* r.^2)
 ess=sw^2/sw2
 push!(m.logamp.trees,BaseExp.scaled_tree(t,eta))
 el=Vector{Float64}(undef,length(w))
 try
  for j in eachindex(w)
   el[j]=real(GBTQuantum.local_energy!(H,m,@view b.states[j,:]))
  end
 finally
  pop!(m.logamp.trees)
 end
 return sum(wr .* el)/sw,ess
end

function choose_eta(H,m,t,b,g)
 E0=real(b.energy)
 eta=ETA0
 # g=dE/deta at eta=0 for the already centered tree direction.
 for k=0:MAX_BACKTRACKS
  Etrial,ess=trial_energy(H,m,t,b,eta)
  if isfinite(Etrial) && Etrial <= E0 + ARMIJO_ALPHA*eta*g
   return (eta=eta,backtracks=k,fallback=false,Etrial=Etrial,ess=ess)
  end
  eta *= BETA
 end
 # Keep the ablation safe and interpretable: if Armijo cannot certify a step,
 # use the established fixed step rather than inventing another controller.
 Ef,ess=trial_energy(H,m,t,b,ETA_FIXED)
 return (eta=ETA_FIXED,backtracks=MAX_BACKTRACKS+1,fallback=true,Etrial=Ef,ess=ess)
end

function train(H,mode,seed)
 cfg=GBTQuantum.TrainingConfig(nsamples=M,epochs=EPOCHS,max_depth=DEPTH,eta=ETA_FIXED,
  burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=seed,exact_diagnostics=false)
 rng=MersenneTwister(seed); S=Matrix{Int8}(undef,M,H.N)
 for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
 m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); la=zeros(M)
 for _=1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,m,S,la); end
 eta_hist=Float64[]; bt_hist=Int[]; ess_hist=Float64[]; fallbacks=0
 for _=1:EPOCHS
  b=GBTQuantum.vmc_batch(H,m,S); y,_=GBTQuantum.make_targets(b)
  raw=BaseExp.grow_uncertainty_tree(b.states,y,b.counts;max_depth=DEPTH,
   min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain,lambda=LAMBDA)
  t,_=BaseExp.center_tree(raw,b)
  d=BaseExp.mc_derivatives(H,m,b,t)
  if mode==:armijo
   a=choose_eta(H,m,t,b,d.g); eta=a.eta
   push!(bt_hist,a.backtracks); push!(ess_hist,a.ess); fallbacks += a.fallback
  else
   eta=ETA_FIXED; push!(bt_hist,0); push!(ess_hist,M)
  end
  push!(eta_hist,eta)
  push!(m.logamp.trees,BaseExp.scaled_tree(t,eta))
  GBTQuantum.refresh_logamps!(la,m,S)
  for _=1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,m,S,la); end
  any(!isfinite,la) && error("non-finite log amplitudes: mode=$mode seed=$seed")
 end
 (model=m,mean_eta=mean(eta_hist),median_eta=median(eta_hist),min_eta=minimum(eta_hist),
  max_eta=maximum(eta_hist),mean_backtracks=mean(bt_hist),fallbacks=fallbacks,
  mean_reweight_ess=mean(ess_hist),min_reweight_ess=minimum(ess_hist))
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
 println("TFIM UNCERTAINTY-REGULARIZED FIXED-ETA vs MINIMAL ARMIJO")
 println("Frozen: lambda=$LAMBDA M=$M depth=$DEPTH epochs=$EPOCHS. Fixed eta=$ETA_FIXED.")
 println("Armijo: eta0=$ETA0 beta=$BETA alpha=$ARMIJO_ALPHA max_backtracks=$MAX_BACKTRACKS; trial energies use same-sample reweighting.")
 println("Exact energies/observables are final evaluation only and never enter the line search or training.")
 println("="^136)
 rows=NamedTuple[]
 for N in NS,ratio in RATIOS
  H=GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true)
  gs=GBTQuantum.exact_ground_observables(H)
  @printf("\nN=%d J/h=%.1f Egs=% .10f\n",N,ratio,gs.energy)
  for mode in MODES,seed in SEEDS
   r=train(H,mode,seed); m=r.model
   E=real(GBTQuantum.exact_model_energy(m,H).energy); o=GBTQuantum.exact_model_observables(m,H)
   row=(N=N,ratio=ratio,mode=String(mode),seed=seed,Egs=gs.energy,E_final=E,energy_error=E-gs.energy,
    mz_abs_error=abs(o.mz-gs.mz),abs_mz_abs_error=abs(o.abs_mz-gs.abs_mz),
    mz2_abs_error=abs(o.mz2-gs.mz2),mx_abs_error=abs(real(o.mx)-gs.mx),
    mean_eta=r.mean_eta,median_eta=r.median_eta,min_eta=r.min_eta,max_eta=r.max_eta,
    mean_backtracks=r.mean_backtracks,fallbacks=r.fallbacks,
    mean_reweight_ess=r.mean_reweight_ess,min_reweight_ess=r.min_reweight_ess)
   push!(rows,row)
   @printf(" %-6s seed=%5d Eerr=% .3e |dmz|=% .3e |d|mz||=% .3e |dmz2|=% .3e |dmx|=% .3e eta(mean)=% .3f bt=%.2f fb=%d ESSmin=%.1f\n",
    String(mode),seed,row.energy_error,row.mz_abs_error,row.abs_mz_abs_error,row.mz2_abs_error,row.mx_abs_error,
    row.mean_eta,row.mean_backtracks,row.fallbacks,row.min_reweight_ess)
  end
  for mode in MODES
   q=[r for r in rows if r.N==N && r.ratio==ratio && r.mode==String(mode)]
   @printf(" SUMMARY %-6s Eerr=% .3e |d|mz||=% .3e |dmz2|=% .3e |dmx|=% .3e mean_eta=% .3f\n",
    String(mode),mean(r.energy_error for r in q),mean(r.abs_mz_abs_error for r in q),
    mean(r.mz2_abs_error for r in q),mean(r.mx_abs_error for r in q),mean(r.mean_eta for r in q))
  end
 end
 dir=joinpath(@__DIR__,"results"); mkpath(dir)
 writecsv(joinpath(dir,"tfim_regularized_armijo.csv"),rows)
 println("\nResults written to experiments/results/tfim_regularized_armijo.csv")
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMRegularizedArmijo.main(); end
