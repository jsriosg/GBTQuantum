module TFIMThesisFigures
using CSV, DataFrames, Statistics, Printf, CairoMakie, LaTeXStrings

const INPUT = joinpath(@__DIR__,"results","tfim_final_regularization_observable_comparison.csv")
const OUTDIR = joinpath(@__DIR__,"results","thesis_figures")

meanstd(v)=(mean(v),std(v))
function summary(df)
 combine(groupby(df,[:N,:ratio,:lambda]),
  :energy_error => mean => :energy_mean, :energy_error => std => :energy_sd,
  :abs_mz_abs_error => mean => :absmz_mean, :abs_mz_abs_error => std => :absmz_sd,
  :mz2_abs_error => mean => :mz2_mean, :mz2_abs_error => std => :mz2_sd,
  :mx_abs_error => mean => :mx_mean, :mx_abs_error => std => :mx_sd)
end

function fig_energy(s)
 f=Figure(size=(900,900))
 for (k,r) in enumerate((0.5,1.0,2.0))
  ax=Axis(f[k,1],xlabel=k==3 ? "System size N" : "",ylabel="Absolute energy error",
   title="J/h = $(r)",yscale=log10,xticks=[8,10,12,14])
  for (lam,label,marker) in ((0.0,"Original GBT-VMC",:circle),(1.0,"Regularized (λ = 1)",:rect))
   q=sort(s[(s.ratio.==r).&(s.lambda.==lam),:],:N)
   errorbars!(ax,q.N,q.energy_mean,q.energy_sd)
   lines!(ax,q.N,q.energy_mean)
   scatter!(ax,q.N,q.energy_mean,marker=marker,label=label)
  end
  k==1 && axislegend(ax,position=:lt)
 end
 save(joinpath(OUTDIR,"tfim_regularization_energy_scaling.pdf"),f)
 save(joinpath(OUTDIR,"tfim_regularization_energy_scaling.png"),f,px_per_unit=2)
end

function fig_observables(s)
 f=Figure(size=(1050,850))
 metrics=[(:absmz_mean,:absmz_sd,L"|\Delta\langle |m_z|\rangle|"),
          (:mz2_mean,:mz2_sd,L"|\Delta\langle m_z^2\rangle|"),
          (:mx_mean,:mx_sd,L"|\Delta\langle m_x\rangle|")]
 for (col,(m,sd,ylab)) in enumerate(metrics), (row,r) in enumerate((1.0,2.0))
  ax=Axis(f[row,col],xlabel=row==2 ? "N" : "",ylabel=ylab,title=row==1 ? "J/h=1.0" : "J/h=2.0",
          yscale=log10,xticks=[8,10,12,14])
  for (lam,label,marker) in ((0.0,"λ=0",:circle),(1.0,"λ=1",:rect))
   q=sort(s[(s.ratio.==r).&(s.lambda.==lam),:],:N)
   errorbars!(ax,q.N,q[!,m],q[!,sd]); lines!(ax,q.N,q[!,m])
   scatter!(ax,q.N,q[!,m],marker=marker,label=label)
  end
  row==1 && col==1 && axislegend(ax,position=:lt)
 end
 save(joinpath(OUTDIR,"tfim_regularization_observable_errors.pdf"),f)
 save(joinpath(OUTDIR,"tfim_regularization_observable_errors.png"),f,px_per_unit=2)
end

function latex_table(s)
 path=joinpath(OUTDIR,"tfim_regularization_summary.tex")
 open(path,"w") do io
  println(io,raw"\begin{table}[htbp]")
  println(io,raw"\centering")
  println(io,raw"\caption{Final TFIM comparison between the original ($\lambda=0$) and uncertainty-regularized ($\lambda=1$) GBT-VMC models. Values are mean $\pm$ standard deviation over five seeds.}")
  println(io,raw"\label{tab:tfim_regularization_summary}")
  println(io,raw"\begin{tabular}{cccrrr}")
  println(io,raw"\toprule")
  println(io,raw"$N$ & $J/h$ & $\lambda$ & $\Delta E$ & $|\Delta\langle m_z^2\rangle|$ & $|\Delta\langle m_x\rangle|$ \\")
  println(io,raw"\midrule")
  for r in eachrow(sort(s,[:ratio,:N,:lambda]))
   row=@sprintf("%d & %.1f & %.0f & %.3e \\pm %.1e & %.3e \\pm %.1e & %.3e \\pm %.1e",
    r.N,r.ratio,r.lambda,r.energy_mean,r.energy_sd,r.mz2_mean,r.mz2_sd,r.mx_mean,r.mx_sd)
   println(io,row * raw" \\")
  end
  println(io,raw"\bottomrule")
  println(io,raw"\end{tabular}")
  println(io,raw"\end{table}")
 end
end

function paired_table(df)
 rows=NamedTuple[]
 for N in sort(unique(df.N)),r in sort(unique(df.ratio))
  q0=sort(df[(df.N.==N).&(df.ratio.==r).&(df.lambda.==0),:],:seed)
  q1=sort(df[(df.N.==N).&(df.ratio.==r).&(df.lambda.==1),:],:seed)
  push!(rows,(N=N,ratio=r,energy=sum(q1.energy_error.<q0.energy_error),
   absmz=sum(q1.abs_mz_abs_error.<q0.abs_mz_abs_error),
   mz2=sum(q1.mz2_abs_error.<q0.mz2_abs_error),mx=sum(q1.mx_abs_error.<q0.mx_abs_error)))
 end
 CSV.write(joinpath(OUTDIR,"tfim_regularization_paired_wins.csv"),DataFrame(rows))
end

function main()
 mkpath(OUTDIR); df=CSV.read(INPUT,DataFrame); @assert nrow(df)==120
 s=summary(df); CSV.write(joinpath(OUTDIR,"tfim_regularization_summary.csv"),s)
 fig_energy(s); fig_observables(s); latex_table(s); paired_table(df)
 println("Thesis outputs written to: ",OUTDIR)
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMThesisFigures.main(); end
