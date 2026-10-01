module ThesisTables
using Statistics, Printf
const R=joinpath(@__DIR__,"results"); const OUT=joinpath(R,"thesis_tables")
parsev(s)=lowercase(s)=="true" ? true : lowercase(s)=="false" ? false : tryparse(Int,s)!==nothing ? parse(Int,s) : tryparse(Float64,s)!==nothing ? parse(Float64,s) : s
function readcsv(name)
 l=readlines(joinpath(R,name)); n=Symbol.(split(l[1],','))
 [NamedTuple{Tuple(n)}(Tuple(parsev.(split(x,',')))) for x in l[2:end] if !isempty(strip(x))]
end
pm(v)=@sprintf("%.3e \\pm %.1e",mean(v),std(v))
function write_tf_z2()
 a=readcsv("tfim_z2_symmetry_comparison.csv")
 open(joinpath(OUT,"tfim_z2_summary.tex"),"w") do io
  println(io,raw"\begin{tabular}{ccccc}"); println(io,raw"\hline")
  println(io,raw"$J/h$ & Representation & $|\Delta E|$ & $|\langle m_z\rangle|$ & $|\Delta m_x|$ \\"); println(io,raw"\hline")
  for r in (0.5,1.0,2.0), mode in ("raw","z2_canonical")
   q=[x for x in a if x.ratio==r && x.mode==mode]
   lab=mode=="raw" ? "Raw" : raw"$Z_2$ canonical"
   println(io,@sprintf("%.1f & %s & %s & %s & %s \\",r,lab,pm(abs.([x.energy_error for x in q])),pm(abs.([x.mz_model for x in q])),pm([x.mx_abs_error for x in q])))
  end
  println(io,raw"\hline"); println(io,raw"\end{tabular}")
 end
end
function write_bh_transfer()
 a=readcsv("bose_hubbard_transfer_L6_multiseed.csv")
 open(joinpath(OUT,"bose_hubbard_transfer_summary.tex"),"w") do io
  println(io,raw"\begin{tabular}{ccccc}"); println(io,raw"\hline")
  println(io,raw"$U/J$ & $\lambda$ & Finite runs & $|\Delta E|$ (finite) & Median peak $|f_L|$ \\"); println(io,raw"\hline")
  for U in (1.0,3.3,6.0), lam in (0.0,1.0)
   q=[x for x in a if x.U_over_J==U && x.lambda==lam]; fq=[x for x in q if x.finite]
   err=isempty(fq) ? "--" : @sprintf("%.3e",mean(x.energy_error for x in fq))
   leaf=@sprintf("%.3e",median(x.peak_raw_leaf for x in q))
   println(io,@sprintf("%.1f & %.0f & %d/%d & %s & %s \\",U,lam,length(fq),length(q),err,leaf))
  end
  println(io,raw"\hline"); println(io,raw"\end{tabular}")
 end
end
function write_bh_symmetry()
 t=readcsv("bose_hubbard_translation_symmetry_L8.csv"); d=readcsv("bose_hubbard_dihedral_cost_grid_L8.csv")
 open(joinpath(OUT,"bose_hubbard_symmetry_L8_summary.tex"),"w") do io
  println(io,raw"\begin{tabular}{cccc}"); println(io,raw"\hline")
  println(io,raw"$U/J$ & Representation & $M$ & Mean $|\Delta E|$ \\"); println(io,raw"\hline")
  for U in (1.0,6.0), sym in ("raw","translation_canonical","dihedral_canonical")
   src=sym=="dihedral_canonical" ? d : t
   q=[x for x in src if x.U_over_J==U && x.symmetry==sym && x.M==1024 && x.depth==6 && x.epoch==400 && x.finite]
   lab=sym=="raw" ? "Raw" : sym=="translation_canonical" ? "Translation" : "Dihedral"
   println(io,@sprintf("%.0f & %s & 1024 & %.3e \\",U,lab,mean(x.energy_error for x in q)))
  end
  println(io,raw"\hline"); println(io,raw"\end{tabular}")
 end
end
function main()
 mkpath(OUT); write_tf_z2(); write_bh_transfer(); write_bh_symmetry()
 println("Wrote thesis tables to ",OUT)
end
end
if abspath(PROGRAM_FILE)==@__FILE__; ThesisTables.main(); end
