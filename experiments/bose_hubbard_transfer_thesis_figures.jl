module BoseHubbardTransferThesisFigures
using Statistics, CairoMakie, LaTeXStrings
const INPUT=joinpath(@__DIR__,"results","bose_hubbard_transfer_L6_multiseed.csv")
const OUTDIR=joinpath(@__DIR__,"results","thesis_figures")
parseval(s)=lowercase(s)=="true" ? true : lowercase(s)=="false" ? false : tryparse(Int,s)!==nothing ? parse(Int,s) : tryparse(Float64,s)!==nothing ? parse(Float64,s) : s
function readcsv(path)
 l=readlines(path); n=Symbol.(split(l[1],','))
 [NamedTuple{Tuple(n)}(Tuple(parseval.(split(x,',')))) for x in l[2:end] if !isempty(strip(x))]
end
function main()
 mkpath(OUTDIR); rows=readcsv(INPUT); @assert length(rows)==30
 colors=Makie.wong_colors()[[1,2]]
 f=Figure(size=(1000,760))
 ax1=Axis(f[1,1],ylabel="Absolute energy error",yscale=log10,
          xticks=([1,2,3],["1.0","3.3","6.0"]),title="(a) Final energy accuracy")
 ax2=Axis(f[1,2],ylabel="Peak raw leaf magnitude",yscale=log10,
          xticks=([1,2,3],["1.0","3.3","6.0"]),title="(b) Update instability")
 for (j,(lam,label,marker,dx)) in enumerate(((0.0,"Original (λ = 0)",:circle,-0.10),(1.0,"Regularized (λ = 1)",:rect,0.10)))
  for (i,U) in enumerate((1.0,3.3,6.0))
   q=[r for r in rows if r.U_over_J==U && r.lambda==lam]
   finite=[r for r in q if r.finite && r.energy_error isa Number && isfinite(r.energy_error)]
   scatter!(ax1,fill(i+dx,length(finite)),[r.energy_error for r in finite],color=(colors[j],0.45),marker=marker,markersize=9)
   if !isempty(finite)
    scatter!(ax1,[i+dx],[mean(r.energy_error for r in finite)],color=colors[j],marker=marker,markersize=16,label=i==1 ? label : nothing)
   end
   failed=count(r->!r.finite,q)
   failed>0 && text!(ax1,i+dx,0.14,text="$(failed) failed",align=(:center,:center),fontsize=14,color=colors[j])
   scatter!(ax2,fill(i+dx,length(q)),[r.peak_raw_leaf for r in q],color=(colors[j],0.45),marker=marker,markersize=9)
   scatter!(ax2,[i+dx],[median(r.peak_raw_leaf for r in q)],color=colors[j],marker=marker,markersize=16,label=i==1 ? label : nothing)
  end
 end
 axislegend(ax1,position=:lt)
 axislegend(ax2,position=:lt)
 Label(f[2,1:2],"Interaction ratio U/J",fontsize=18)
 save(joinpath(OUTDIR,"bose_hubbard_transfer_regularization_L6.pdf"),f)
 save(joinpath(OUTDIR,"bose_hubbard_transfer_regularization_L6.png"),f,px_per_unit=2)
 println("Wrote BH transfer thesis figure to ",OUTDIR)
end
end
if abspath(PROGRAM_FILE)==@__FILE__; BoseHubbardTransferThesisFigures.main(); end
