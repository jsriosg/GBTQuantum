module TFIMThesisFigures
using Statistics, Printf, CairoMakie, LaTeXStrings

const INPUT=joinpath(@__DIR__,"results","tfim_final_regularization_observable_comparison.csv")
const OUTDIR=joinpath(@__DIR__,"results","thesis_figures")

parseval(s)=tryparse(Int,s)!==nothing ? parse(Int,s) : tryparse(Float64,s)!==nothing ? parse(Float64,s) : s
function readcsv(path)
 lines=readlines(path); names=Symbol.(split(lines[1],','))
 [NamedTuple{Tuple(names)}(Tuple(parseval.(split(line,',')))) for line in lines[2:end] if !isempty(strip(line))]
end
function writecsv(path,rows)
 isempty(rows)&&return
 ns=propertynames(rows[1]); open(path,"w") do io
  println(io,join(string.(ns),','))
  for r in rows; println(io,join((getproperty(r,n) for n in ns),',')); end
 end
end
function summary(rows)
 out=NamedTuple[]
 for N in sort(unique(r.N for r in rows)),ratio in sort(unique(r.ratio for r in rows)),lambda in sort(unique(r.lambda for r in rows))
  q=[r for r in rows if r.N==N && r.ratio==ratio && r.lambda==lambda]
  isempty(q)&&continue
  ms(sym)=(v=Float64[getproperty(r,sym) for r in q]; (mean(v),std(v)))
  e=ms(:energy_error); a=ms(:abs_mz_abs_error); z=ms(:mz2_abs_error); x=ms(:mx_abs_error)
  push!(out,(N=N,ratio=ratio,lambda=lambda,energy_mean=e[1],energy_sd=e[2],
   absmz_mean=a[1],absmz_sd=a[2],mz2_mean=z[1],mz2_sd=z[2],mx_mean=x[1],mx_sd=x[2]))
 end
 out
end
subset(s,ratio,lambda)=sort([r for r in s if r.ratio==ratio && r.lambda==lambda],by=r->r.N)

function fig_energy(s, rows)
 f=Figure(size=(900,900))
 colors=Makie.wong_colors()[[1,2]]
 for (k,r) in enumerate((0.5,1.0,2.0))
  ax=Axis(f[k,1],xlabel=k==3 ? "System size N" : "",ylabel="Absolute energy error",
   title="J/h = $(r)",yscale=log10,xticks=[8,10,12,14])
  for (j,(lam,label,marker)) in enumerate(((0.0,"Original GBT-VMC",:circle),(1.0,"Regularized (λ = 1)",:rect)))
   q=subset(s,r,lam); N=[x.N for x in q]; y=[x.energy_mean for x in q]; sd=[x.energy_sd for x in q]
   raw=[z for z in rows if z.ratio==r && z.lambda==lam]
   scatter!(ax,[z.N for z in raw],[z.energy_error for z in raw],marker=marker,markersize=6,alpha=0.25,color=(colors[j],0.35))
   lines!(ax,N,y,color=colors[j]); scatter!(ax,N,y,marker=marker,markersize=11,label=label,color=colors[j])
  end
  k==1 && axislegend(ax,position=:lt)
 end
 save(joinpath(OUTDIR,"tfim_regularization_energy_scaling.pdf"),f)
 save(joinpath(OUTDIR,"tfim_regularization_energy_scaling.png"),f,px_per_unit=2)
end

function fig_observables(s, rows)
 f=Figure(size=(1050,850))
 colors=Makie.wong_colors()[[1,2]]
 metrics=[(:absmz_mean,:absmz_sd,L"|\Delta\langle |m_z|\rangle|"),
          (:mz2_mean,:mz2_sd,L"|\Delta\langle m_z^2\rangle|"),
          (:mx_mean,:mx_sd,L"|\Delta\langle m_x\rangle|")]
 for (col,(m,sd,ylab)) in enumerate(metrics), (row,r) in enumerate((1.0,2.0))
  ax=Axis(f[row,col],xlabel=row==2 ? "System size N" : "",ylabel=ylab,
          title=col==2 ? "J/h = $(r)" : "",
          yscale=log10,xticks=[8,10,12,14])
  for (j,(lam,label,marker)) in enumerate(((0.0,"λ = 0",:circle),(1.0,"λ = 1",:rect)))
   q=subset(s,r,lam); N=[x.N for x in q]; y=[getproperty(x,m) for x in q]; e=[getproperty(x,sd) for x in q]
   raw=[z for z in rows if z.ratio==r && z.lambda==lam]
   rawsym = m==:absmz_mean ? :abs_mz_abs_error : m==:mz2_mean ? :mz2_abs_error : :mx_abs_error
   scatter!(ax,[z.N for z in raw],[getproperty(z,rawsym) for z in raw],marker=marker,markersize=5,alpha=0.25,color=(colors[j],0.35))
   lines!(ax,N,y,color=colors[j]); scatter!(ax,N,y,marker=marker,markersize=10,label=label,color=colors[j])
  end
  row==1 && col==1 && axislegend(ax,position=:lt)
 end
 save(joinpath(OUTDIR,"tfim_regularization_observable_errors.pdf"),f)
 save(joinpath(OUTDIR,"tfim_regularization_observable_errors.png"),f,px_per_unit=2)
end

function latex_table(s)
 open(joinpath(OUTDIR,"tfim_regularization_summary.tex"),"w") do io
  println(io,raw"\begin{table}[htbp]"); println(io,raw"\centering")
  println(io,raw"\caption{Final TFIM comparison between the original ($\lambda=0$) and uncertainty-regularized ($\lambda=1$) GBT-VMC models. Values are mean $\pm$ standard deviation over five seeds.}")
  println(io,raw"\label{tab:tfim_regularization_summary}"); println(io,raw"\begin{tabular}{cccrrr}"); println(io,raw"\toprule")
  println(io,raw"$N$ & $J/h$ & $\lambda$ & $\Delta E$ & $|\Delta\langle m_z^2\rangle|$ & $|\Delta\langle m_x\rangle|$ \\"); println(io,raw"\midrule")
  for r in sort(s,by=x->(x.ratio,x.N,x.lambda))
   row=@sprintf("%d & %.1f & %.0f & %.3e \\pm %.1e & %.3e \\pm %.1e & %.3e \\pm %.1e",
    r.N,r.ratio,r.lambda,r.energy_mean,r.energy_sd,r.mz2_mean,r.mz2_sd,r.mx_mean,r.mx_sd)
   println(io,row*raw" \\")
  end
  println(io,raw"\bottomrule"); println(io,raw"\end{tabular}"); println(io,raw"\end{table}")
 end
end

function paired_table(rows)
 out=NamedTuple[]
 for N in sort(unique(r.N for r in rows)),ratio in sort(unique(r.ratio for r in rows))
  q0=sort([r for r in rows if r.N==N&&r.ratio==ratio&&r.lambda==0.0],by=r->r.seed)
  q1=sort([r for r in rows if r.N==N&&r.ratio==ratio&&r.lambda==1.0],by=r->r.seed)
  push!(out,(N=N,ratio=ratio,energy=sum(q1[i].energy_error<q0[i].energy_error for i in eachindex(q0)),
   absmz=sum(q1[i].abs_mz_abs_error<q0[i].abs_mz_abs_error for i in eachindex(q0)),
   mz2=sum(q1[i].mz2_abs_error<q0[i].mz2_abs_error for i in eachindex(q0)),
   mx=sum(q1[i].mx_abs_error<q0[i].mx_abs_error for i in eachindex(q0))))
 end
 writecsv(joinpath(OUTDIR,"tfim_regularization_paired_wins.csv"),out)
end

function main()
 mkpath(OUTDIR); rows=readcsv(INPUT); @assert length(rows)==120
 s=summary(rows); writecsv(joinpath(OUTDIR,"tfim_regularization_summary.csv"),s)
 fig_energy(s,rows); fig_observables(s,rows); latex_table(s); paired_table(rows)
 println("Thesis outputs written to: ",OUTDIR)
end
end
if abspath(PROGRAM_FILE)==@__FILE__; TFIMThesisFigures.main(); end
