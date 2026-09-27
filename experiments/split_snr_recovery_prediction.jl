using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module SplitSNRRecoveryPredictionExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

# Tests the closure proposed by the hierarchical acquisition derivation:
#
#   rho_v = P(sampled best split at v == oracle best split)
#
# against the Gaussian top-two prediction
#
#   rhohat_v = Phi(D_v / sigma_D,v),
#
# where D_v = G1_v-G2_v and sigma_D,v is predicted from the influence
# function of the top-two gain difference under the actual importance
# proposal q. We also record an empirical z-score using the observed
# variance of repeated sampled margins; this separates failures of the
# variance formula from failures of the Gaussian/top-two approximation.
#
# Finally, predicted path survival is formed as the product of predicted
# ancestor recovery probabilities and compared with empirical path survival.

const N = 12
const h = 1.0
const J = 2.0
const training_nsamples = 256
const nepochs = 64
const checkpoint_epochs = Set([8, 32, 64])
const ntraining_runs = 3
const optimizer_max_depth = 4
const eta = 0.05
const burn_in_sweeps = 100
const sweeps_per_epoch = 1
const optimizer_min_leaf_weight = 1.0
const optimizer_min_gain = 0.0
const oracle_depth = 4
const exact_min_leaf_weight = 1e-14
const sampled_min_leaf_weight = 1e-14
const diagnostic_sample_sizes = [256, 1024]
const nsampling_runs = 200
const mass_exponents = [1.0, 2.0, 3.0]
const base_seed = 1_520_000
const diagnostic_seed_base = 31_520_000
const support_floor = 1e-14

function normal_cdf(z::Real)
    z == Inf && return 1.0
    z == -Inf && return 0.0
    isnan(z) && return NaN

    x = Float64(z)
    ax = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * ax)

    poly = t * (
        0.319381530 +
        t * (-0.356563782 +
        t * (1.781477937 +
        t * (-1.821255978 +
        t * 1.330274429)))
    )

    tail = exp(-0.5 * ax^2) / sqrt(2π) * poly
    cdf_positive = 1.0 - tail

    return x >= 0 ? cdf_positive : 1.0 - cdf_positive
end

function csv_escape(x)
    s = string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s) || occursin('\r', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_namedtuple_csv(path, rows)
    open(path, "w") do io
        isempty(rows) && return
        cols = propertynames(first(rows))
        println(io, join(string.(cols), ","))
        for row in rows
            println(io, join((csv_escape(getproperty(row, c)) for c in cols), ","))
        end
    end
end

function exact_frozen_problem(H, model, X)
    p = exact_probabilities(model, X)
    eloc = ComplexF64[local_energy!(H, model, @view(X[i, :])) for i in axes(X, 1)]
    E = sum(p .* eloc)
    y = -real.(eloc .- E)
    return (probabilities=p, target=y, energy=E,
            target_rms=sqrt(sum(p .* y.^2)))
end

function gain_landscape(X, y, w, idx; min_weight=0.0)
    g = fill(-Inf, size(X, 2))
    W = sum(w[i] for i in idx)
    W > 0 || return g
    S = sum(w[i] * y[i] for i in idx)
    parent = S*S/W
    for f in axes(X, 2)
        WL=0.0; SL=0.0
        @inbounds for i in idx
            if X[i,f] < 0
                WL += w[i]; SL += w[i]*y[i]
            end
        end
        WR = W-WL
        (WL < min_weight || WR < min_weight || WL <= 0 || WR <= 0) && continue
        SR = S-SL
        g[f] = SL*SL/WL + SR*SR/WR - parent
    end
    return g
end

function top_two(g)
    v = [i for i in eachindex(g) if isfinite(g[i]) && g[i] >= 0]
    isempty(v) && return (0,0,0.0,0.0)
    sort!(v, by=i->g[i], rev=true)
    f1=v[1]; f2=length(v)>=2 ? v[2] : 0
    return (f1,f2,g[f1],f2==0 ? 0.0 : g[f2])
end

function oracle_regions(tree, X)
    out=NamedTuple[]
    function walk(k,d,idx,path,ancestors)
        n=tree.nodes[k]
        n.isleaf && return
        push!(out,(node_index=k,node_depth=d,node_path=path,
                   state_indices=copy(idx),ancestor_indices=copy(ancestors)))
        L=Int[]; R=Int[]; f=Int(n.feature)
        for i in idx
            X[i,f] < 0 ? push!(L,i) : push!(R,i)
        end
        nextanc=[ancestors;k]
        walk(Int(n.left),d+1,L,path*"L",nextanc)
        walk(Int(n.right),d+1,R,path*"R",nextanc)
    end
    walk(1,0,collect(axes(X,1)),"",Int[])
    return out
end

function conditional_oracle_gains(X,y,p,idx)
    mass=sum(p[idx])
    mass>0 || return fill(-Inf,size(X,2)),mass
    pc=p./mass
    return gain_landscape(X,y,pc,idx;min_weight=exact_min_leaf_weight),mass
end

function gain_influence(X,y,p,idx,f)
    mass=sum(p[idx]); phi=zeros(Float64,size(X,1))
    (mass>0 && f!=0) || return phi
    r=0.0; a=0.0; m=0.0
    @inbounds for i in idx
        pv=p[i]/mass; Li=X[i,f]<0 ? 1.0 : 0.0
        r+=pv*Li; a+=pv*y[i]*Li; m+=pv*y[i]
    end
    (r>eps() && 1-r>eps()) || return phi
    gr=-a^2/r^2+(m-a)^2/(1-r)^2
    ga=2a/r-2(m-a)/(1-r)
    gm=2(m-a)/(1-r)-2m
    @inbounds for i in idx
        Li=X[i,f]<0 ? 1.0 : 0.0
        phi[i]=gr*(Li-r)+ga*(y[i]*Li-a)+gm*(y[i]-m)
    end
    return phi
end

function build_oracle(H,model,X)
    fr=exact_frozen_problem(H,model,X); p=fr.probabilities
    tree=GBTQuantum.grow_tree(X,fr.target,p;max_depth=oracle_depth,
        min_weight=exact_min_leaf_weight,min_gain=0.0)
    nodes=NamedTuple[]
    for n in oracle_regions(tree,X)
        og,mass=conditional_oracle_gains(X,fr.target,p,n.state_indices)
        f1,f2,G1,G2=top_two(og)
        phi1=gain_influence(X,fr.target,p,n.state_indices,f1)
        phi2=gain_influence(X,fr.target,p,n.state_indices,f2)
        phiD=phi1.-phi2
        push!(nodes,merge(n,(probability_mass=mass,f1=f1,f2=f2,G1=G1,G2=G2,
            margin=G1-G2,phiD=phiD,Phi=phiD./mass)))
    end
    return fr,nodes
end

function acquisition_for_alpha(fr,nodes,alpha)
    p=fr.probabilities; A2=zeros(Float64,length(p))
    for n in nodes
        importance=n.probability_mass^alpha*max(n.G1,0.0)
        @. A2 += importance*n.Phi^2
    end
    q=p.*sqrt.(A2); Z=sum(q)
    if Z>eps()
        q./=Z; q.=max.(q,support_floor.*p); q./=sum(q)
    else
        q=copy(p)
    end
    return q
end

# First-order asymptotic variance of the conditional top-two margin estimator.
# The global influence function is Phi_v = phiD_v / P_v inside R_v, hence
# Var(Dhat_v) ~= (1/M) sum_x p(x)^2/q(x) * Phi_v(x)^2.
function predicted_margin_variance(fr,n,q,M)
    p=fr.probabilities
    acc=0.0
    @inbounds for i in n.state_indices
        q[i] > 0 || continue
        acc += p[i]^2/q[i] * n.Phi[i]^2
    end
    return acc/M
end

function sampled_node_statistics(X,y,p,q,draws,idx,f1,f2)
    mask=falses(size(X,1)); mask[idx].=true
    localdraw=[i for i in draws if mask[i]]
    isempty(localdraw) && return (best=0,margin=NaN)
    u,c=compress_indices(localdraw)
    wraw=c.*p[u]./q[u]; Z=sum(wraw)
    Z>0 || return (best=0,margin=NaN)
    w=wraw./Z
    g=gain_landscape(X[u,:],y[u],w,collect(eachindex(u));min_weight=sampled_min_leaf_weight)
    best,_,_,_=top_two(g)
    d=(f1!=0 && f2!=0 && isfinite(g[f1]) && isfinite(g[f2])) ? g[f1]-g[f2] : NaN
    return (best=best,margin=d)
end

diagnostic_seed(run,epoch,M,ai,repetition)=diagnostic_seed_base+
    1_000_000*run+10_000*epoch+100_000*ai+M+repetition

function diagnose(H,model,X,run,epoch)
    fr,nodes=build_oracle(H,model,X)
    nodepos=Dict(n.node_index=>j for (j,n) in enumerate(nodes))
    rows=NamedTuple[]
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e nodes=%d\n",epoch,real(fr.energy),fr.target_rms,length(nodes))

    for (ai,alpha) in enumerate(mass_exponents)
        q=acquisition_for_alpha(fr,nodes,alpha)
        for M in diagnostic_sample_sizes
            correct=falses(nsampling_runs,length(nodes))
            path=falses(nsampling_runs,length(nodes))
            margins=fill(NaN,nsampling_runs,length(nodes))
            for srun in 1:nsampling_runs
                rng=MersenneTwister(diagnostic_seed(run,epoch,M,ai,srun))
                draws=draw_categorical_indices(rng,q,M)
                for (j,n) in enumerate(nodes)
                    st=sampled_node_statistics(X,fr.target,fr.probabilities,q,draws,
                                               n.state_indices,n.f1,n.f2)
                    correct[srun,j]=st.best!=0 && st.best==n.f1
                    margins[srun,j]=st.margin
                end
                for (j,n) in enumerate(nodes)
                    ok=true
                    for a in n.ancestor_indices
                        ok &= correct[srun,nodepos[a]]
                    end
                    path[srun,j]=ok
                end
            end

            rho_emp=vec(mean(correct,dims=1))
            s_emp=vec(mean(path,dims=1))
            rho_pred=zeros(length(nodes))
            rho_empz=fill(NaN,length(nodes))

            for (j,n) in enumerate(nodes)
                varpred=predicted_margin_variance(fr,n,q,M)
                sigpred=sqrt(max(varpred,0.0))
                zpred=sigpred>0 ? n.margin/sigpred : (n.margin>0 ? Inf : 0.0)
                rho_pred[j]=normal_cdf(zpred)

                vals=[margins[r,j] for r in 1:nsampling_runs if isfinite(margins[r,j])]
                sigma_emp=length(vals)>1 ? std(vals;corrected=true) : NaN
                z_emp=isfinite(sigma_emp) && sigma_emp>0 ? n.margin/sigma_emp : NaN
                rho_empz[j]=isfinite(z_emp) ? normal_cdf(z_emp) : NaN
            end

            for (j,n) in enumerate(nodes)
                s_pred=prod(rho_pred[nodepos[a]] for a in n.ancestor_indices)
                s_ind_emp=prod(rho_emp[nodepos[a]] for a in n.ancestor_indices)
                varpred=predicted_margin_variance(fr,n,q,M)
                sigpred=sqrt(max(varpred,0.0))
                zpred=sigpred>0 ? n.margin/sigpred : (n.margin>0 ? Inf : 0.0)
                vals=[margins[r,j] for r in 1:nsampling_runs if isfinite(margins[r,j])]
                sigma_emp=length(vals)>1 ? std(vals;corrected=true) : NaN
                mean_margin_emp=isempty(vals) ? NaN : mean(vals)
                push!(rows,(training_run=run,epoch=epoch,alpha=alpha,M=M,
                    node_index=n.node_index,node_depth=n.node_depth,node_path=n.node_path,
                    probability_mass=n.probability_mass,oracle_best_feature=n.f1,
                    oracle_second_feature=n.f2,oracle_margin=n.margin,
                    predicted_sigma_margin=sigpred,predicted_z=zpred,
                    empirical_mean_margin=mean_margin_emp,empirical_sigma_margin=sigma_emp,
                    empirical_split_recovery=rho_emp[j],predicted_split_recovery=rho_pred[j],
                    empirical_z_split_recovery=rho_empz[j],
                    abs_error_split_recovery=abs(rho_emp[j]-rho_pred[j]),
                    empirical_path_survival=s_emp[j],predicted_path_survival=s_pred,
                    empirical_independence_path_survival=s_ind_emp,
                    abs_error_path_survival=abs(s_emp[j]-s_pred)))
            end

            valid=collect(eachindex(nodes))
            mae_rho=mean(abs(rho_emp[j]-rho_pred[j]) for j in valid)
            nonroot=[j for j in valid if nodes[j].node_depth>0]
            mae_path=mean(abs(s_emp[j]-prod(rho_pred[nodepos[a]] for a in nodes[j].ancestor_indices)) for j in nonroot)
            mae_empind=mean(abs(s_emp[j]-prod(rho_emp[nodepos[a]] for a in nodes[j].ancestor_indices)) for j in nonroot)
            @printf("  alpha=%.1f M=%4d  rho MAE=%.3f  path MAE(pred)=%.3f  path MAE(emp-ind)=%.3f\n",
                    alpha,M,mae_rho,mae_path,mae_empind)
        end
    end
    return rows
end

function run_training(run,X)
    H=TFIMHamiltonian(N;J=J,h=h,periodic=true)
    rng=MersenneTwister(base_seed+10_000*run)
    samples=Matrix{Int8}(undef,training_nsamples,N)
    @inbounds for i in eachindex(samples)
        samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1)
    end
    model=LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    logamps=zeros(training_nsamples)
    for _ in 1:burn_in_sweeps
        GBTQuantum.sweep!(rng,model,samples,logamps)
    end
    rows=NamedTuple[]
    for epoch in 1:nepochs
        batch=vmc_batch(H,model,samples); yA,_=make_targets(batch); w=batch.counts
        if epoch in checkpoint_epochs
            append!(rows,diagnose(H,model,X,run,epoch))
        end
        tree=GBTQuantum.grow_tree(batch.states,yA,w;max_depth=optimizer_max_depth,
            min_weight=optimizer_min_leaf_weight,min_gain=optimizer_min_gain)
        pred=predict_all(tree,batch.states); mu=weighted_mean(pred,w)
        isfinite(mu) && mu!=0 && (tree=shift_tree_leaves(tree,mu))
        push!(model.logamp.trees,scale_tree(tree,eta))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:sweeps_per_epoch
            GBTQuantum.sweep!(rng,model,samples,logamps)
        end
    end
    return rows
end

function main()
    println("\n============================================================")
    println("SPLIT-SNR RECOVERY PREDICTION")
    println("N=$N J/h=$(J/h) runs=$ntraining_runs checkpoints=$(sort(collect(checkpoint_epochs)))")
    println("alpha=$mass_exponents M=$diagnostic_sample_sizes repetitions=$nsampling_runs")
    println("Tests rho_v ~= Phi(D_v/sigma_D,v) and propagated path survival")
    println("============================================================")
    X=enumerate_states(N); rows=NamedTuple[]
    for run in 1:ntraining_runs
        @printf("\nTraining trajectory %d/%d\n",run,ntraining_runs)
        append!(rows,run_training(run,X))
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    write_namedtuple_csv(joinpath(outdir,"split_snr_recovery_prediction.csv"),rows)
    println("\nResults written to experiments/results/split_snr_recovery_prediction.csv")
end

export main,run_training,diagnose

end # module

if abspath(PROGRAM_FILE)==@__FILE__
    SplitSNRRecoveryPredictionExperiment.main()
end
