using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module TreeTargetScalingAudit

using GBTQuantum
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

const N = 8
const RATIOS = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
const MAX_DEPTH = 4
const MIN_WEIGHT = 1e-14
const MIN_GAIN = 0.0

wmean(y,w)=sum(w .* y)/sum(w)
wmse(y,w)=sum(w .* (y .- wmean(y,w)).^2)/sum(w)

function root_stump_gain(X,y,w,j)
    mask = @view(X[:,j]) .<= 0
    wl=sum(@view w[mask]); wr=sum(@view w[.!mask]); wt=wl+wr
    if wl <= 0 || wr <= 0
        return -Inf
    end
    μ=wmean(y,w)
    μl=sum((@view w[mask]).*(@view y[mask]))/wl
    μr=sum((@view w[.!mask]).*(@view y[.!mask]))/wr
    parent=sum(w .* (y.-μ).^2)
    child=sum((@view w[mask]).*((@view y[mask]).-μl).^2)+sum((@view w[.!mask]).*((@view y[.!mask]).-μr).^2)
    parent-child
end

function tree_stats(tree,X,y,w)
    f=predict_all(tree,X)
    mse0=wmse(y,w)
    mse1=sum(w .* (y.-f).^2)/sum(w)
    μy=wmean(y,w); μf=wmean(f,w)
    vy=sum(w.*(y.-μy).^2)/sum(w)
    vf=sum(w.*(f.-μf).^2)/sum(w)
    cov=sum(w.*(y.-μy).*(f.-μf))/sum(w)
    corr=(vy>0 && vf>0) ? cov/sqrt(vy*vf) : NaN
    r2=mse0>0 ? 1-mse1/mse0 : NaN
    return (f=f,var_f=vf,mse0=mse0,mse1=mse1,corr=corr,r2=r2,
            n_unique_pred=length(unique(round.(f,digits=12))))
end

function exact_target(H,model,X)
    p=exact_probabilities(model,X)
    eloc=Float64[real(GBTQuantum.local_energy!(H,model,@view(X[i,:]))) for i in axes(X,1)]
    E=sum(p.*eloc)
    y=-(eloc.-E)
    p,E,eloc,y
end

function main()
    println("\n============================================================")
    println("DETERMINISTIC TREE TARGET-SCALING AUDIT")
    println("N=$N ratios=$RATIOS max_depth=$MAX_DEPTH")
    println("Full Hilbert space; no MC, no Armijo, no training trajectory")
    println("============================================================")

    X=enumerate_states(N)
    model=GBTQuantum.LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    rows=NamedTuple[]
    normalized_targets=Dict{Float64,Vector{Float64}}()

    for J in RATIOS
        H=GBTQuantum.TFIMHamiltonian(N;J=J,h=1.0,periodic=true)
        p,E,eloc,y=exact_target(H,model,X)
        yn=y./J
        normalized_targets[J]=yn
        analytic=Float64[sum(Float64(X[i,k])*Float64(X[i,k==N ? 1 : k+1]) for k=1:N) for i in axes(X,1)]
        target_err=maximum(abs.(yn.-analytic))

        gains_raw=[root_stump_gain(X,y,p,j) for j=1:N]
        gains_norm=[root_stump_gain(X,yn,p,j) for j=1:N]

        tree_raw=GBTQuantum.grow_tree(X,y,p;max_depth=MAX_DEPTH,min_weight=MIN_WEIGHT,min_gain=MIN_GAIN)
        tree_norm=GBTQuantum.grow_tree(X,yn,p;max_depth=MAX_DEPTH,min_weight=MIN_WEIGHT,min_gain=MIN_GAIN)
        sr=tree_stats(tree_raw,X,y,p)
        sn=tree_stats(tree_norm,X,yn,p)

        println("\nJ/h = $J")
        @printf("  E0=% .12f target RMS=% .6e normalized RMS=% .6e\n",E,sqrt(sum(p.*y.^2)),sqrt(sum(p.*yn.^2)))
        @printf("  max |y/J - sum s_i s_{i+1}| = %.3e\n",target_err)
        println("  raw root gains        = ", [@sprintf("%.3e",g) for g in gains_raw])
        println("  normalized root gains = ", [@sprintf("%.3e",g) for g in gains_norm])
        @printf("  RAW : unique predictions=%d var(f)=%.3e MSE %.3e -> %.3e R2=% .6f corr=% .6f\n",
            sr.n_unique_pred,sr.var_f,sr.mse0,sr.mse1,sr.r2,sr.corr)
        @printf("  NORM: unique predictions=%d var(f)=%.3e MSE %.3e -> %.3e R2=% .6f corr=% .6f\n",
            sn.n_unique_pred,sn.var_f,sn.mse0,sn.mse1,sn.r2,sn.corr)

        push!(rows,(ratio=J,E0=E,target_rms=sqrt(sum(p.*y.^2)),normalized_rms=sqrt(sum(p.*yn.^2)),
            analytic_target_maxerr=target_err,raw_max_root_gain=maximum(gains_raw),norm_max_root_gain=maximum(gains_norm),
            raw_unique_predictions=sr.n_unique_pred,norm_unique_predictions=sn.n_unique_pred,
            raw_var_f=sr.var_f,norm_var_f=sn.var_f,raw_mse_before=sr.mse0,raw_mse_after=sr.mse1,
            norm_mse_before=sn.mse0,norm_mse_after=sn.mse1,raw_r2=sr.r2,norm_r2=sn.r2,
            raw_corr=sr.corr,norm_corr=sn.corr))
    end

    ref=normalized_targets[RATIOS[1]]
    println("\n============================================================")
    println("NORMALIZED TARGET CROSS-COUPLING CONSISTENCY")
    println("============================================================")
    for J in RATIOS
        @printf("J/h=%4.2f max |y_J/J - y_ref/J_ref| = %.3e\n",J,maximum(abs.(normalized_targets[J].-ref)))
    end

    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"tree_target_scaling_audit.csv")
    names=propertynames(rows[1])
    open(path,"w") do io
        println(io,join(string.(names),','))
        for r in rows
            println(io,join((getproperty(r,n) for n in names),','))
        end
    end
    println("\nResults written to experiments/results/tree_target_scaling_audit.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    TreeTargetScalingAudit.main()
end
