module TFIMLeafRiskVsCurvatureError
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using GBTQuantum, Random, Statistics, Printf

const N=8; const NSAMPLES=512; const MAX_DEPTH=4; const ETA_FIXED=0.05; const SEED=1234
const CHECKPOINTS=Dict(1.0=>Set([8,50]),2.0=>Set([4,10]))

function allstates(N)
 X=Matrix{Int8}(undef,1<<N,N)
 for s=0:(1<<N)-1, i=1:N; X[s+1,i]=((s>>(i-1))&1)==1 ? Int8(1) : Int8(-1); end
 X
end
@inline flip(j,i)=((j-1) ⊻ (1<<(i-1)))+1

function leaf_index(t,x)
 i=1
 while true
  n=t.nodes[i]; n.isleaf && return i
  i=x[n.feature] <= 0 ? n.left : n.right
 end
end

function center_like_production(t,b)
 w=Float64.(b.counts); pred=[GBTQuantum.predict(t,@view b.states[j,:]) for j in axes(b.states,1)]
 mu=GBTQuantum.weighted_mean(pred,w); nd=copy(t.nodes)
 for i in eachindex(nd)
  n=nd[i]; n.isleaf && (nd[i]=GBTQuantum.Node(n.feature,n.value-mu,n.left,n.right,true))
 end
 GBTQuantum.RegressionTree(nd),mu
end
scaled_tree(t,e)=GBTQuantum.RegressionTree([n.isleaf ? GBTQuantum.Node(n.feature,e*n.value,n.left,n.right,true) : n for n in t.nodes])

function ranks(v)
 n=length(v); o=sortperm(v); r=zeros(Float64,n); k=1
 while k<=n
  q=k
  while q<n && v[o[q+1]]==v[o[k]]; q+=1; end
  rr=(k+q)/2
  for j=k:q; r[o[j]]=rr; end
  k=q+1
 end
 r
end
rankcorr(x,y)=(std(ranks(x))==0 || std(ranks(y))==0) ? NaN : cor(ranks(x),ranks(y))

function exact_arrays(H,m,t,X)
 d=size(X,1); A=zeros(d); f=zeros(d); leaf=zeros(Int,d)
 for j=1:d
  x=@view X[j,:]; A[j]=GBTQuantum.logamplitude(m,x); f[j]=GBTQuantum.predict(t,x); leaf[j]=leaf_index(t,x)
 end
 p=exp.(2 .* A .- maximum(2 .* A)); p./=sum(p)
 el=zeros(d); hf=zeros(d)
 for j=1:d
  x=@view X[j,:]; el[j]=GBTQuantum.diagonal(H,x); hf[j]=GBTQuantum.diagonal(H,x)*f[j]
  for i=1:H.N
   k=flip(j,i); rr=exp(A[k]-A[j]); el[j]-=H.h*rr; hf[j]-=H.h*rr*f[k]
  end
 end
 Ef=sum(p .* f); E=sum(p .* el); g=2*sum(p .* (f .- Ef) .* (el .- E))
 (;p,f,leaf,el,hf,Ef,E,g)
end

function sample_arrays(H,m,t,b)
 n=size(b.states,1); w=Float64.(b.counts); W=sum(w); f=zeros(n); leaf=zeros(Int,n); hf=zeros(n); el=real.(b.local_energy)
 for j=1:n
  x=@view b.states[j,:]; f[j]=GBTQuantum.predict(t,x); leaf[j]=leaf_index(t,x); A=GBTQuantum.logamplitude(m,x)
  z=GBTQuantum.diagonal(H,x)*f[j]
  for i=1:H.N
   x[i]=-x[i]; z-=H.h*exp(GBTQuantum.logamplitude(m,x)-A)*GBTQuantum.predict(t,x); x[i]=-x[i]
  end
  hf[j]=z
 end
 Ef=sum(w .* f)/W; E=sum(w .* el)/W; g=2*sum(w .* (f .- Ef) .* (el .- E))/W
 (;w,W,f,leaf,hf,el,Ef,E,g)
end

function audit(H,m,t,b,X,ratio,ep,shift)
 ex=exact_arrays(H,m,t,X); sm=sample_arrays(H,m,t,b); rows=NamedTuple[]
 for L in sort(unique(ex.leaf))
  ix=findall(==(L),ex.leaf); im=findall(==(L),sm.leaf)
  P=sum(ex.p[ix]); Phat=sum(sm.w[im])/sm.W; dP=Phat-P; fL=ex.f[first(ix)]
  e1=2*sum(ex.p[ix] .* ex.f[ix].^2 .* ex.el[ix])
  e2=2*sum(ex.p[ix] .* ex.f[ix] .* ex.hf[ix])
  e3=-4*ex.E*sum(ex.p[ix] .* ex.f[ix].^2)
  e4=-4*ex.Ef*ex.g*P
  if isempty(im)
   m1=0.0; m2=0.0; m3=0.0; m4=0.0
  else
   m1=2*sum(sm.w[im] .* sm.f[im].^2 .* sm.el[im])/sm.W
   m2=2*sum(sm.w[im] .* sm.f[im] .* sm.hf[im])/sm.W
   m3=-4*sm.E*sum(sm.w[im] .* sm.f[im].^2)/sm.W
   m4=-4*sm.Ef*sm.g*Phat
  end
  dc=(m1-e1)+(m2-e2)+(m3-e3)+(m4-e4)
  sigma=sqrt(max(Phat*(1-Phat),0.0)/sm.W)
  risk=sigma*fL^2; oracle=abs(dP)*fL^2
  push!(rows,(ratio=ratio,epoch=ep,leaf=L,support=sum(sm.w[im]),f=fL,born_mass=P,sample_mass=Phat,abs_mass_error=abs(dP),sigma_iid=sigma,risk_sigma_f2=risk,oracle_absdP_f2=oracle,dc=dc,abs_dc=abs(dc),dC1=m1-e1,dC2=m2-e2,dC3=m3-e3,dC4=m4-e4,exact_mean_f=ex.Ef,mc_mean_f=sm.Ef))
 end

 y=[r.abs_dc for r in rows]
 rho_risk=rankcorr([r.risk_sigma_f2 for r in rows],y)
 rho_oracle=rankcorr([r.oracle_absdP_f2 for r in rows],y)
 rho_f=rankcorr([abs(r.f) for r in rows],y)
 rho_sigma=rankcorr([r.sigma_iid for r in rows],y)
 rho_support=rankcorr([r.support for r in rows],y)
 @printf("\n%s\n","-"^126)
 @printf("J/h=%.2f epoch=%d | gauge shift=% .4e | <f>MC=% .3e <f>exact=% .3e\n",ratio,ep,shift,sm.Ef,ex.Ef)
 @printf("DIRECT Spearman with |dc_L|: sigma*f^2=% .3f | oracle |dP|f^2=% .3f | |f|=% .3f | sigma=% .3f | support=% .3f\n",rho_risk,rho_oracle,rho_f,rho_sigma,rho_support)
 @printf("Risk concentration: top-4 estimated-risk leaves capture %.1f%% of sum |dc_L|\n",begin
   ord=sortperm(rows,by=r->r.risk_sigma_f2,rev=true); 100*sum(rows[k].abs_dc for k in ord[1:min(4,length(ord))])/sum(y)
 end)
 ord=sortperm(rows,by=r->r.risk_sigma_f2,rev=true)
 println(" Top leaves by production-feasible sigma(P_L) f_L^2:")
 for k in ord[1:min(8,length(ord))]
  r=rows[k]; @printf("  leaf=%3d risk=% .3e |dc|=% .3e dc=% .3e sigma=% .3e f=% .3e W=%5.1f oracle=% .3e [dC1=% .2e dC2=% .2e dC3=% .2e dC4=% .2e]\n",r.leaf,r.risk_sigma_f2,r.abs_dc,r.dc,r.sigma_iid,r.f,r.support,r.oracle_absdP_f2,r.dC1,r.dC2,r.dC3,r.dC4)
 end
 rows
end

function train(H,ratio,X)
 cfg=GBTQuantum.TrainingConfig(nsamples=NSAMPLES,epochs=150,max_depth=MAX_DEPTH,eta=ETA_FIXED,burn_in_sweeps=50,sweeps_per_epoch=2,use_phase=false,seed=SEED,exact_diagnostics=false)
 rng=MersenneTwister(SEED); S=Matrix{Int8}(undef,NSAMPLES,H.N)
 for i in eachindex(S); S[i]=rand(rng,Bool) ? Int8(1) : Int8(-1); end
 m=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false); la=zeros(NSAMPLES)
 for _=1:cfg.burn_in_sweeps; GBTQuantum.sweep!(rng,m,S,la); end
 out=NamedTuple[]; wanted=CHECKPOINTS[ratio]
 for ep=1:maximum(wanted)
  b=GBTQuantum.vmc_batch(H,m,S); y,_=GBTQuantum.make_targets(b)
  raw=GBTQuantum.grow_tree(b.states,y,b.counts;max_depth=cfg.max_depth,min_weight=cfg.min_leaf_weight,min_gain=cfg.min_gain)
  t,shift=center_like_production(raw,b)
  ep in wanted && append!(out,audit(H,m,t,b,X,ratio,ep,shift))
  push!(m.logamp.trees,scaled_tree(t,ETA_FIXED)); GBTQuantum.refresh_logamps!(la,m,S)
  for _=1:cfg.sweeps_per_epoch; GBTQuantum.sweep!(rng,m,S,la); end
  any(!isfinite,la) && break
 end
 out
end

function writecsv(path,rows)
 isempty(rows)&&return; ns=propertynames(rows[1])
 open(path,"w") do io
  println(io,join(string.(ns),','))
  for r in rows; println(io,join((getproperty(r,n) for n in ns),',')); end
 end
end

function main()
 println("="^126)
 println("TFIM DIRECT LEAF UNCERTAINTY-RISK VS CURVATURE-ERROR DIAGNOSTIC")
 println("N=$N Hilbert=$(1<<N), trajectory samples=$NSAMPLES, fixed eta=$ETA_FIXED")
 println("Tests production-feasible R_L = sigma_iid(P_hat_L) f_L^2 directly against oracle |delta c_L|")
 println("Production gauge <f>_MC=0; exact enumeration never re-centers the tree")
 println("="^126)
 X=allstates(N); rows=NamedTuple[]
 for r in sort(collect(keys(CHECKPOINTS)))
  append!(rows,train(GBTQuantum.TFIMHamiltonian(N;J=r,h=1.0,periodic=true),r,X))
 end
 dir=joinpath(@__DIR__,"results"); mkpath(dir); path=joinpath(dir,"tfim_leaf_risk_vs_curvature_error.csv")
 writecsv(path,rows); println("\nResults written to $path")
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMLeafRiskVsCurvatureError.main(); end
