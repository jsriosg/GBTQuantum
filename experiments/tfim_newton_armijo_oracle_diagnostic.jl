module TFIMNewtonArmijoOracleDiagnostic
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using GBTQuantum, Random, Statistics, Printf
include(joinpath(@__DIR__,"tfim_uncertainty_regularized_training.jl"))
const B=TFIMUncertaintyRegularizedTraining

const NS=[8,10,12]
const RATIOS=[1.0,2.0]
const SEED=1234
const LAMBDA=1.0; const M=512; const DEPTH=4; const ETA_TRAIN=0.05
const MAXEPOCH=50
const CHECKPOINTS=Set([1,8,25,50])
const ARMIJO_ALPHA=0.1
const MAX_HALVINGS=8
const FRESH_M=512
const FRESH_BURN=50
const FRESH_REPS=3

function trial_exact(H,m,t,eta)
 push!(m.logamp.trees,B.scaled_tree(t,eta))
 try
  real(GBTQuantum.exact_model_energy(m,H).energy)
 finally
  pop!(m.logamp.trees)
 end
end

function trial_reweight(H,m,t,b,eta)
 f=[GBTQuantum.predict(t,@view b.states[j,:]) for j in axes(b.states,1)]
 z=2eta .* f; zmax=maximum(z); r=exp.(z .- zmax)
 w=Float64.(b.counts); wr=w .* r; sw=sum(wr)
 ess=sw^2/sum(w .* r.^2)
 push!(m.logamp.trees,B.scaled_tree(t,eta))
 try
  el=[real(GBTQuantum.local_energy!(H,m,@view b.states[j,:])) for j in axes(b.states,1)]
  return sum(wr .* el)/sw,ess
 finally
  pop!(m.logamp.trees)
 end
end

function trial_fresh(H,m,t,eta,N,ratio,epoch,k)
 vals=Float64[]
 push!(m.logamp.trees,B.scaled_tree(t,eta))
 try
  for rep=1:FRESH_REPS
   # Independent validation chain for each replicate/candidate.
   seed=10_000_000 + 100_000*N + 10_000*round(Int,ratio) + 100*epoch + 10*k + rep
   v=GBTQuantum.validation_vmc(m,H;nsamples=FRESH_M,burn_in_sweeps=FRESH_BURN,
      thinning_sweeps=1,seed=seed)
   push!(vals,v.energy)
  end
 finally
  pop!(m.logamp.trees)
 end
 mean(vals), (length(vals)>1 ? std(vals)/sqrt(length(vals)) : 0.0)
end

armijo_pass(Etrial,E0,g,eta)=isfinite(Etrial) && Etrial <= E0 + ARMIJO_ALPHA*eta*g

function first_pass(rows,field)
 for r in rows
  getproperty(r,field) && return r.k
 end
 missing
end

function writecsv(path,rows)
 isempty(rows)&&return; ns=propertynames(rows[1])
 open(path,"w") do io
  println(io,join(string.(ns),','))
  for r in rows; println(io,join((getproperty(r,n) for n in ns),',')); end
 end
end

function run_case(N,ratio)
 H=GBTQuantum.TFIMHamiltonian(N;J=ratio,h=1.0,periodic=true)
 X=B.allstates(N)
 cfg=GBTQuantum.TrainingConfig(nsamples=M,epochs=MAXEPOCH,max_depth=DEPTH,eta=ETA_TRAIN,
  burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=SEED,exact_diagnostics=false)
 rng=MersenneTwister(SEED); S=Matrix{Int8}(undef,M,N)
 for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
 m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); la=zeros(M)
 for _=1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,m,S,la); end
 allrows=NamedTuple[]; summaries=NamedTuple[]
 for ep=1:MAXEPOCH
  b=GBTQuantum.vmc_batch(H,m,S); y,_=GBTQuantum.make_targets(b)
  raw=B.grow_uncertainty_tree(b.states,y,b.counts;max_depth=DEPTH,
    min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain,lambda=LAMBDA)
  t,_=B.center_tree(raw,b)
  if ep in CHECKPOINTS
   mc=B.mc_derivatives(H,m,b,t); ex=B.exact_derivatives(H,m,t,X)
   eta_mc=(isfinite(mc.g)&&isfinite(mc.c)&&mc.g<0&&mc.c>0) ? -mc.g/mc.c : NaN
   eta_ex=(isfinite(ex.g)&&isfinite(ex.c)&&ex.g<0&&ex.c>0) ? -ex.g/ex.c : NaN
   @printf("\nN=%d J/h=%.1f epoch=%d  gMC=% .4e cMC=% .4e etaN_MC=% .5f | gEX=% .4e cEX=% .4e etaN_EX=% .5f\n",
    N,ratio,ep,mc.g,mc.c,eta_mc,ex.g,ex.c,eta_ex)
   localrows=NamedTuple[]
   if isfinite(eta_mc)
    for k=0:MAX_HALVINGS
     eta=eta_mc/(2.0^k)
     Eex=trial_exact(H,m,t,eta)
     Erw,ess=trial_reweight(H,m,t,b,eta)
     Ef,se=trial_fresh(H,m,t,eta,N,ratio,ep,k)
     pex=armijo_pass(Eex,ex.E,mc.g,eta)
     prw=armijo_pass(Erw,real(b.energy),mc.g,eta)
     pf=armijo_pass(Ef,real(b.energy),mc.g,eta)
     row=(N=N,ratio=ratio,epoch=ep,k=k,eta=eta,g_mc=mc.g,c_mc=mc.c,g_exact=ex.g,c_exact=ex.c,
       eta_newton_mc=eta_mc,eta_newton_exact=eta_ex,E0_mc=real(b.energy),E0_exact=ex.E,
       E_exact=Eex,E_reweight=Erw,reweight_ess=ess,E_fresh=Ef,E_fresh_se=se,
       armijo_exact=pex,armijo_reweight=prw,armijo_fresh=pf)
     push!(localrows,row); push!(allrows,row)
     @printf(" k=%d eta=% .5f Eex=% .8f %s | Erw=% .8f ESS=%6.1f %s | Efresh=% .8f ± %.2e %s\n",
      k,eta,Eex,pex ? "PASS":"FAIL",Erw,ess,prw ? "PASS":"FAIL",Ef,se,pf ? "PASS":"FAIL")
    end
    push!(summaries,(N=N,ratio=ratio,epoch=ep,eta_newton_mc=eta_mc,eta_newton_exact=eta_ex,
      newton_relative_error=abs(eta_mc-eta_ex)/(abs(eta_ex)+eps()),
      exact_first_k=first_pass(localrows,:armijo_exact),
      reweight_first_k=first_pass(localrows,:armijo_reweight),
      fresh_first_k=first_pass(localrows,:armijo_fresh)))
   else
    push!(summaries,(N=N,ratio=ratio,epoch=ep,eta_newton_mc=eta_mc,eta_newton_exact=eta_ex,
      newton_relative_error=NaN,exact_first_k=missing,reweight_first_k=missing,fresh_first_k=missing))
   end
  end
  # Frozen production trajectory: lambda=1 and eta=0.05.
  push!(m.logamp.trees,B.scaled_tree(t,ETA_TRAIN)); GBTQuantum.refresh_logamps!(la,m,S)
  for _=1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,m,S,la); end
 end
 allrows,summaries
end

function main()
 println("="^144)
 println("TFIM REGULARIZED NEWTON + ARMIJO ENERGY-ORACLE DIAGNOSTIC")
 println("N=$NS ratios=$RATIOS lambda=$LAMBDA M=$M frozen training eta=$ETA_TRAIN checkpoints=$(sort(collect(CHECKPOINTS)))")
 println("Newton proposal etaN=-gMC/cMC; candidates etaN/2^k. Exact quantities are diagnostic only.")
 println("Trial-energy oracles: exact enumeration, same-sample reweighting, and $FRESH_REPS independent fresh VMC chains x $FRESH_M samples.")
 println("="^144)
 rows=NamedTuple[]; sums=NamedTuple[]
 for N in NS,ratio in RATIOS
  r,s=run_case(N,ratio); append!(rows,r); append!(sums,s)
 end
 dir=joinpath(@__DIR__,"results"); mkpath(dir)
 writecsv(joinpath(dir,"tfim_newton_armijo_oracle_profiles.csv"),rows)
 writecsv(joinpath(dir,"tfim_newton_armijo_oracle_summary.csv"),sums)
 println("\nResults written to:")
 println(" experiments/results/tfim_newton_armijo_oracle_profiles.csv")
 println(" experiments/results/tfim_newton_armijo_oracle_summary.csv")
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMNewtonArmijoOracleDiagnostic.main(); end
