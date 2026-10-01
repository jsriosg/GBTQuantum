module BoseHubbardScalingThesisFigures
using Statistics, CairoMakie
const R=joinpath(@__DIR__,"results"); const OUT=joinpath(R,"thesis_figures")
parsev(s)=lowercase(s)=="true" ? true : lowercase(s)=="false" ? false : tryparse(Int,s)!==nothing ? parse(Int,s) : tryparse(Float64,s)!==nothing ? parse(Float64,s) : s
function readcsv(name)
 l=readlines(joinpath(R,name)); n=Symbol.(split(l[1],','))
 [NamedTuple{Tuple(n)}(Tuple(parsev.(split(x,',')))) for x in l[2:end] if !isempty(strip(x))]
end
function qmean(rows,L,U,sym,field; epoch=400)
 q=[x for x in rows if x.L==L && x.U_over_J==U && String(x.symmetry)==sym && x.epoch==epoch && x.finite]
 mean(getproperty(x,field) for x in q)
end
function main()
 mkpath(OUT)
 rows=vcat(readcsv("bose_hubbard_lazy_dihedral_scaling.csv"),readcsv("bose_hubbard_linear_dihedral_L14.csv"))
 colors=Makie.wong_colors()[[1,2]]; syms=(("raw","Raw",:circle),("dihedral_canonical","Dihedral",:rect))
 f=Figure(size=(1050,850))
 for (row,U) in enumerate((1.0,6.0))
  axE=Axis(f[row,1],title=row==1 ? "(a) Energy per site" : "",ylabel="U/J = $(Int(U))\nE/L",
           xlabel=row==2 ? "System size L" : "",xticks=[8,10,12,14])
  axV=Axis(f[row,2],title=row==1 ? "(b) Local-energy variance per site" : "",
           ylabel="Var(Eₗₒc)/L",xlabel=row==2 ? "System size L" : "",yscale=log10,xticks=[8,10,12,14])
  for (j,(sym,label,marker)) in enumerate(syms)
   Ls=[8,10,12,14]
   e=[qmean(rows,L,U,sym,:E_per_site) for L in Ls]
   v=[qmean(rows,L,U,sym,:Eloc_variance_per_site) for L in Ls]
   lines!(axE,Ls,e,color=colors[j]); scatter!(axE,Ls,e,color=colors[j],marker=marker,markersize=11,label=label)
   lines!(axV,Ls,v,color=colors[j]); scatter!(axV,Ls,v,color=colors[j],marker=marker,markersize=11,label=label)
  end
  row==1 && axislegend(axE,position=:rb)
 end
 Label(f[0,1:2],"Bose–Hubbard symmetry scaling at fixed training resources",fontsize=20,font=:bold)
 save(joinpath(OUT,"bose_hubbard_dihedral_scaling_L8_L14.pdf"),f)
 save(joinpath(OUT,"bose_hubbard_dihedral_scaling_L8_L14.png"),f,px_per_unit=2)

 # Separate checkpoint figure: exposes the L14 fixed-budget limitation rather than hiding it in endpoint data.
 g=Figure(size=(1000,650))
 for (col,U) in enumerate((1.0,6.0))
  ax=Axis(g[1,col],title="U/J = $(Int(U))",xlabel="Epoch",ylabel=col==1 ? "E/L" : "",xticks=[50,100,150,250,400])
  for (j,(sym,label,marker)) in enumerate(syms)
   q=sort([x for x in rows if x.L==14 && x.U_over_J==U && String(x.symmetry)==sym && x.finite],by=x->x.epoch)
   epochs=sort(unique(x.epoch for x in q))
   means=[mean(x.E_per_site for x in q if x.epoch==t) for t in epochs]
   lines!(ax,epochs,means,color=colors[j]); scatter!(ax,epochs,means,color=colors[j],marker=marker,markersize=10,label=label)
  end
  col==1 && axislegend(ax,position=:rb)
 end
 Label(g[0,1:2],"L = N = 14 convergence under the fixed computational budget",fontsize=20,font=:bold)
 save(joinpath(OUT,"bose_hubbard_L14_checkpoint_convergence.pdf"),g)
 save(joinpath(OUT,"bose_hubbard_L14_checkpoint_convergence.png"),g,px_per_unit=2)
 println("Wrote BH scaling and L14 checkpoint figures to ",OUT)
end
end
if abspath(PROGRAM_FILE)==@__FILE__; BoseHubbardScalingThesisFigures.main(); end
