module BoseHubbardSymmetryL8ThesisFigures
using Statistics, CairoMakie
const R=joinpath(@__DIR__,"results"); const OUT=joinpath(R,"thesis_figures")
parsev(s)=lowercase(s)=="true" ? true : lowercase(s)=="false" ? false : tryparse(Int,s)!==nothing ? parse(Int,s) : tryparse(Float64,s)!==nothing ? parse(Float64,s) : s
function readcsv(name)
 l=readlines(joinpath(R,name)); n=Symbol.(split(l[1],','))
 [NamedTuple{Tuple(n)}(Tuple(parsev.(split(x,',')))) for x in l[2:end] if !isempty(strip(x))]
end
function finite_mean(q,s)
 v=[getproperty(x,s) for x in q if x.finite]
 isempty(v) ? NaN : mean(v)
end
function groupmean(rows; U, symmetry, M=1024, depth=6, epoch=400)
 q=[x for x in rows if x.U_over_J==U && x.symmetry==symmetry && x.M==M && x.depth==depth && x.epoch==epoch]
 isempty(q) ? NaN : finite_mean(q,:energy_error)
end
function main()
 mkpath(OUT)
 trans=readcsv("bose_hubbard_translation_symmetry_L8.csv")
 dih=readcsv("bose_hubbard_fast_dihedral_L8.csv")
 cost=readcsv("bose_hubbard_dihedral_cost_grid_L8.csv")
 f=Figure(size=(1050,760)); colors=Makie.wong_colors()
 # Panel a: same computational budget, raw -> translation -> dihedral
 ax=Axis(f[1,1],title="(a) Symmetry at fixed resources",ylabel="Mean absolute energy error",yscale=log10,
  xticks=([1,2,3],["Raw","Translation","Dihedral"]))
 for (j,U) in enumerate((1.0,6.0))
  vals=Float64[]
  for sym in ("raw","translation_canonical","dihedral_canonical")
   if sym=="dihedral_canonical"
    # Fixed-resource comparison uses the cost-grid run at M=1024, d=6.
    push!(vals,groupmean(cost,U=U,symmetry=sym,M=1024,depth=6))
   else
    push!(vals,groupmean(trans,U=U,symmetry=sym,M=1024,depth=6))
   end
  end
  lines!(ax,1:3,vals,color=colors[j]); scatter!(ax,1:3,vals,color=colors[j],markersize=12,label="U/J=$(Int(U))")
 end
 axislegend(ax,position=:rt)
 # Panel b: resource tradeoff inside exact dihedral representation
 ax=Axis(f[1,2],title="(b) Dihedral resource tradeoff",ylabel="Mean absolute energy error",yscale=log10,
  xticks=([1,2,3,4],["M=512\nd=4","M=512\nd=6","M=1024\nd=4","M=1024\nd=6"]))
 for (j,U) in enumerate((1.0,6.0))
  vals=Float64[]
  for (M,d) in ((512,4),(512,6),(1024,4),(1024,6))
   q=[x for x in cost if x.U_over_J==U && x.symmetry=="dihedral_canonical" && x.M==M && x.depth==d && x.epoch==400]
   push!(vals,finite_mean(q,:energy_error))
  end
  lines!(ax,1:4,vals,color=colors[j]); scatter!(ax,1:4,vals,color=colors[j],markersize=12,label="U/J=$(Int(U))")
 end
 axislegend(ax,position=:rt)
 Label(f[0,1:2],"Bose–Hubbard L = N = 8: exploiting spatial symmetry",fontsize=20,font=:bold)
 Label(f[2,1],"Representation",fontsize=16)
 Label(f[2,2],"Monte Carlo samples M and tree depth d",fontsize=16)
 save(joinpath(OUT,"bose_hubbard_symmetry_resources_L8.pdf"),f)
 save(joinpath(OUT,"bose_hubbard_symmetry_resources_L8.png"),f,px_per_unit=2)
 println("Wrote L8 symmetry thesis figure to ",OUT)
end
end
if abspath(PROGRAM_FILE)==@__FILE__; BoseHubbardSymmetryL8ThesisFigures.main(); end
