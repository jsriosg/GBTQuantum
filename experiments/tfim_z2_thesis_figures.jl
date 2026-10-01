module TFIMZ2ThesisFigures
using Statistics, CairoMakie
const R=joinpath(@__DIR__,"results"); const OUT=joinpath(R,"thesis_figures")
const INPUT=joinpath(R,"tfim_z2_symmetry_comparison.csv")
parsev(s)=lowercase(s)=="true" ? true : lowercase(s)=="false" ? false : tryparse(Int,s)!==nothing ? parse(Int,s) : tryparse(Float64,s)!==nothing ? parse(Float64,s) : s
function readcsv(path)
 l=readlines(path); n=Symbol.(split(l[1],','))
 [NamedTuple{Tuple(n)}(Tuple(parsev.(split(x,',')))) for x in l[2:end] if !isempty(strip(x))]
end
function main()
 mkpath(OUT); rows=readcsv(INPUT); @assert length(rows)==30
 colors=Makie.wong_colors()[[1,2]]
 modes=(("raw","Raw",:circle),("z2_canonical","Z₂ canonical",:rect))
 f=Figure(size=(1250,760))
 # (a) odd order parameter: exact finite-size symmetry requires zero
 ax1=Axis(f[1,1],title="(a) Finite-size symmetry",xlabel="J/h",ylabel="|⟨m_z⟩|",yscale=log10,
          xticks=([0.5,1,2],["0.5","1","2"]))
 for (j,(mode,label,marker)) in enumerate(modes)
  for (i,r) in enumerate((0.5,1.0,2.0))
   q=[x for x in rows if x.ratio==r && x.mode==mode]
   y=[max(abs(x.mz_model),1e-17) for x in q]
   scatter!(ax1,fill(r,length(y)),y,color=(colors[j],0.4),marker=marker,markersize=8)
   scatter!(ax1,[r],[mean(y)],color=colors[j],marker=marker,markersize=14,label=i==1 ? label : nothing)
  end
 end
 axislegend(ax1,position=:lt)
 # (b) energy error, individual seeds + mean
 ax2=Axis(f[1,2],title="(b) Variational energy",xlabel="J/h",ylabel="Absolute energy error",yscale=log10,
          xticks=([0.5,1,2],["0.5","1","2"]))
 for (j,(mode,label,marker)) in enumerate(modes)
  for r in (0.5,1.0,2.0)
   q=[x for x in rows if x.ratio==r && x.mode==mode]; y=abs.([x.energy_error for x in q])
   dx=j==1 ? -0.035 : 0.035
   scatter!(ax2,fill(r+dx,length(y)),y,color=(colors[j],0.38),marker=marker,markersize=8)
   scatter!(ax2,[r+dx],[mean(y)],color=colors[j],marker=marker,markersize=14)
  end
 end
 # (c) even-observable errors: mean over seeds
 ax3=Axis(f[1,3],title="(c) Even observables",xlabel="Observable and J/h",ylabel="Mean absolute error",yscale=log10,
          xticks=(1:9,["|m_z|\n0.5","m_z²\n0.5","m_x\n0.5","|m_z|\n1","m_z²\n1","m_x\n1","|m_z|\n2","m_z²\n2","m_x\n2"]))
 fields=(:abs_mz_abs_error,:mz2_abs_error,:mx_abs_error); k=0
 for r in (0.5,1.0,2.0), field in fields
  k+=1
  for (j,(mode,label,marker)) in enumerate(modes)
   q=[x for x in rows if x.ratio==r && x.mode==mode]
   y=[getproperty(x,field) for x in q]; dx=j==1 ? -0.12 : 0.12
   scatter!(ax3,fill(k+dx,length(y)),y,color=(colors[j],0.25),marker=marker,markersize=6)
   scatter!(ax3,[k+dx],[mean(y)],color=colors[j],marker=marker,markersize=11)
  end
 end
 vlines!(ax3,[3.5,6.5],color=(:gray,0.5),linestyle=:dash)
 Label(f[0,1:3],"TFIM N = 14: exact Z₂ symmetry versus variational optimization",fontsize=20,font=:bold)
 save(joinpath(OUT,"tfim_z2_symmetry_thesis_comparison.pdf"),f)
 save(joinpath(OUT,"tfim_z2_symmetry_thesis_comparison.png"),f,px_per_unit=2)
 println("Wrote TFIM Z2 thesis figure to ",OUT)
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMZ2ThesisFigures.main(); end
