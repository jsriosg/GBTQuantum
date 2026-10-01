module BoseHubbardResourcesThesisFigures
using Statistics, CairoMakie
const R=joinpath(@__DIR__,"results"); const OUT=joinpath(R,"thesis_figures")
parsev(s)=lowercase(s)=="true" ? true : lowercase(s)=="false" ? false : tryparse(Int,s)!==nothing ? parse(Int,s) : tryparse(Float64,s)!==nothing ? parse(Float64,s) : s
function readcsv(name)
 l=readlines(joinpath(R,name)); n=Symbol.(split(l[1],','))
 [NamedTuple{Tuple(n)}(Tuple(parsev.(split(x,',')))) for x in l[2:end] if !isempty(strip(x))]
end
function avg(q,s)
 v=[getproperty(x,s) for x in q if x.finite]
 isempty(v) ? NaN : mean(v)
end
function series(rows,U,key,vals; filt=x->true)
 [(v,avg([x for x in rows if x.U_over_J==U && getproperty(x,key)==v && filt(x)],:energy_error)) for v in vals]
end
function panel!(ax,sets)
 colors=Makie.wong_colors()
 for (j,(label,xy,marker)) in enumerate(sets)
  x=first.(xy); y=last.(xy)
  lines!(ax,x,y,color=colors[j]); scatter!(ax,x,y,color=colors[j],marker=marker,markersize=11,label=label)
 end
end
function main()
 mkpath(OUT)
 sample=readcsv("bose_hubbard_sample_scaling_L8.csv")
 hyper=readcsv("bose_hubbard_hyperparameter_convergence_L8.csv")
 depth=readcsv("bose_hubbard_depth_capacity_L8.csv")
 long=readcsv("bose_hubbard_long_horizon_L8.csv")
 f=Figure(size=(1050,850))
 # (a) sample scaling: final runs, original vs regularized
 ax=Axis(f[1,1],title="(a) Sample size",xlabel="Monte Carlo samples M",ylabel="Mean absolute energy error",yscale=log10,
         xscale=log2,xticks=([256,512,1024,2048],["256","512","1024","2048"]))
 for (j,U) in enumerate((1.0,6.0))
  for (k,lam) in enumerate((0.0,1.0))
   xy=series(sample,U,:M,[256,512,1024,2048],filt=x->x.lambda==lam)
   col=Makie.wong_colors()[j]; marker=lam==0 ? :circle : :rect
   good=[p for p in xy if isfinite(last(p))]
   lines!(ax,first.(good),last.(good),color=col,linestyle=lam==0 ? :dash : :solid)
   scatter!(ax,first.(good),last.(good),color=col,marker=marker,markersize=10,
    label="U/J=$(Int(U)), λ=$(Int(lam))")
  end
 end
 axislegend(ax,position=:lb)
 # Explicitly mark all-failed strong-coupling unregularized settings.
 for (i,M) in enumerate((256,512,1024,2048))
  q=[x for x in sample if x.U_over_J==6.0 && x.lambda==0.0 && x.M==M]
  nf=count(x->!x.finite,q)
  nf==length(q) && text!(ax,M,2.5,text="all failed",align=(:center,:center),fontsize=11,color=Makie.wong_colors()[2])
 end
 # (b) lambda at eta=.05, epoch400
 ax=Axis(f[1,2],title="(b) Regularization strength",yscale=log10,xlabel="Regularization strength λ",
         xticks=([0.25,0.5,1,2,4],["0.25","0.5","1","2","4"]))
 for (j,U) in enumerate((1.0,6.0))
  xy=series(hyper,U,:lambda,[0.25,0.5,1.0,2.0,4.0],filt=x->x.eta==0.05 && x.epoch==400)
  lines!(ax,first.(xy),last.(xy),color=Makie.wong_colors()[j]); scatter!(ax,first.(xy),last.(xy),color=Makie.wong_colors()[j],markersize=10,label="U/J=$(Int(U))")
 end
 axislegend(ax,position=:rt)
 # (c) depth at epoch400
 ax=Axis(f[2,1],title="(c) Tree depth",xlabel="Maximum tree depth",ylabel="Mean absolute energy error",yscale=log10,xticks=[2,4,6])
 for (j,U) in enumerate((1.0,6.0))
  xy=series(depth,U,:depth,[2,4,6],filt=x->x.epoch==400)
  lines!(ax,first.(xy),last.(xy),color=Makie.wong_colors()[j]); scatter!(ax,first.(xy),last.(xy),color=Makie.wong_colors()[j],markersize=10,label="U/J=$(Int(U))")
 end
 # (d) long horizon
 ax=Axis(f[2,2],title="(d) Training horizon",xlabel="Epoch",yscale=log10,xticks=[50,150,250,400,600,800])
 for (j,U) in enumerate((1.0,6.0))
  vals=[50,100,150,250,400,500,600,800]; xy=series(long,U,:epoch,vals)
  lines!(ax,first.(xy),last.(xy),color=Makie.wong_colors()[j]); scatter!(ax,first.(xy),last.(xy),color=Makie.wong_colors()[j],markersize=8,label="U/J=$(Int(U))")
 end
 Label(f[0,1:2],"Bose–Hubbard L = N = 8: resource and convergence study",fontsize=20,font=:bold)
 save(joinpath(OUT,"bose_hubbard_resources_convergence_L8.pdf"),f)
 save(joinpath(OUT,"bose_hubbard_resources_convergence_L8.png"),f,px_per_unit=2)
 println("Wrote resource/convergence figure to ",OUT)
end
end
if abspath(PROGRAM_FILE)==@__FILE__; BoseHubbardResourcesThesisFigures.main(); end
