using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module InfluenceAcquisitionValidationExperiment

using GBTQuantum
using Random
using Statistics
using Printf

include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

# Validate the influence-function derivation for tree split-margin recovery.

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
const hierarchy_alpha = 2.0
const if_born_epsilon = 0.05
const support_floor = 1e-14
const base_seed = 1_620_000
const diagnostic_seed_base = 31_620_000
const gauge_shifts = [-7.25, -1.0, 0.37, 5.5]
const validation_tol = 1e-10

function csv_escape(x)
    s=string(x)
    if occursin(',',s)||occursin('"',s)||occursin('\n',s)||occursin('\r',s)
        return "\""*replace(s,"\""=>"\"\"")*"\""
    end
    return s
end
function write_namedtuple_csv(path,rows)
    open(path,"w") do io
        isempty(rows)&&return
        cols=propertynames(first(rows)); println(io,join(string.(cols),","))
        for r in rows println(io,join((csv_escape(getproperty(r,c)) for c in cols),",")) end
    end
end

function exact_frozen_problem(H,model,X)
    p=exact_probabilities(model,X)
    eloc=ComplexF64[local_energy!(H,model,@view(X[i,:])) for i in axes(X,1)]
    E=sum(p.*eloc); y=-real.(eloc.-E)
    return (probabilities=p,target=y,energy=E,target_rms=sqrt(sum(p.*y.^2)))
end

function gain_landscape(X,y,w,idx;min_weight=0.0)
    g=fill(-Inf,size(X,2)); W=sum(w[i] for i in idx); W>0||return g
    S=sum(w[i]*y[i] for i in idx); parent=S*S/W
    for f in axes(X,2)
        WL=0.0; SL=0.0
        @inbounds for i in idx
            if X[i,f]<0 WL+=w[i]; SL+=w[i]*y[i] end
        end
        WR=W-WL
        (WL<min_weight||WR<min_weight||WL<=0||WR<=0)&&continue
        SR=S-SL; g[f]=SL*SL/WL+SR*SR/WR-parent
    end
    return g
end
function top_two(g)
    v=[i for i in eachindex(g) if isfinite(g[i])&&g[i]>=0]
    isempty(v)&&return (0,0,0.0,0.0); sort!(v,by=i->g[i],rev=true)
    f1=v[1]; f2=length(v)>=2 ? v[2] : 0
    return (f1,f2,g[f1],f2==0 ? 0.0 : g[f2])
end
function oracle_regions(tree,X)
    out=NamedTuple[]
    function walk(k,d,idx,path,ancestors)
        n=tree.nodes[k]; n.isleaf&&return
        push!(out,(node_index=k,node_depth=d,node_path=path,state_indices=copy(idx),ancestor_indices=copy(ancestors)))
        L=Int[];R=Int[];f=Int(n.feature)
        for i in idx X[i,f]<0 ? push!(L,i) : push!(R,i) end
        nextanc=[ancestors;k]
        walk(Int(n.left),d+1,L,path*"L",nextanc); walk(Int(n.right),d+1,R,path*"R",nextanc)
    end
    walk(1,0,collect(axes(X,1)),"",Int[]); return out
end

function gain_and_if_closed(X,y,p,idx,f)
    P=sum(p[idx]); (P>0&&f!=0)||return (gain=-Inf,psi=zeros(length(p)))
    r=0.0;a=0.0;m=0.0
    @inbounds for i in idx
        pi=p[i]/P; L=X[i,f]<0 ? 1.0 : 0.0; r+=pi*L; a+=pi*y[i]*L; m+=pi*y[i]
    end
    (r>eps()&&1-r>eps())||return (gain=-Inf,psi=zeros(length(p)))
    gain=a*a/r+(m-a)^2/(1-r)-m*m
    cr=-a*a/(r*r)+(m-a)^2/((1-r)^2); ca=2a/r-2(m-a)/(1-r); cm=2(m-a)/(1-r)-2m
    psi=zeros(Float64,length(p))
    @inbounds for i in idx
        L=X[i,f]<0 ? 1.0 : 0.0
        psi[i]=cr*(L-r)+ca*(y[i]*L-a)+cm*(y[i]-m)
    end
    return (gain=gain,psi=psi)
end
function gain_if_old(X,y,p,idx,f)
    P=sum(p[idx]);phi=zeros(Float64,length(p));(P>0&&f!=0)||return phi
    r=0.0;a=0.0;m=0.0
    @inbounds for i in idx
        pv=p[i]/P;L=X[i,f]<0 ? 1.0 : 0.0;r+=pv*L;a+=pv*y[i]*L;m+=pv*y[i]
    end
    (r>eps()&&1-r>eps())||return phi
    gr=-a^2/r^2+(m-a)^2/(1-r)^2;ga=2a/r-2(m-a)/(1-r);gm=2(m-a)/(1-r)-2m
    @inbounds for i in idx
        L=X[i,f]<0 ? 1.0 : 0.0;phi[i]=gr*(L-r)+ga*(y[i]*L-a)+gm*(y[i]-m)
    end
    return phi
end
function build_oracle(H,model,X)
    fr=exact_frozen_problem(H,model,X);p=fr.probabilities
    tree=GBTQuantum.grow_tree(X,fr.target,p;max_depth=oracle_depth,min_weight=exact_min_leaf_weight,min_gain=0.0)
    nodes=NamedTuple[]
    for n in oracle_regions(tree,X)
        P=sum(p[n.state_indices]);pc=p./P
        g=gain_landscape(X,fr.target,pc,n.state_indices;min_weight=exact_min_leaf_weight)
        f1,f2,G1,G2=top_two(g);f2==0&&continue
        a=gain_and_if_closed(X,fr.target,p,n.state_indices,f1);b=gain_and_if_closed(X,fr.target,p,n.state_indices,f2)
        psi=a.psi.-b.psi;oldpsi=gain_if_old(X,fr.target,p,n.state_indices,f1).-gain_if_old(X,fr.target,p,n.state_indices,f2)
        push!(nodes,merge(n,(probability_mass=P,oracle_gains=g,f1=f1,f2=f2,G1=G1,G2=G2,margin=G1-G2,psi=psi,oldpsi=oldpsi)))
    end
    return fr,nodes
end

function normalized_q(raw,p)
    q=max.(Float64.(raw),0.0);Z=sum(q)
    if !(Z>eps())||!isfinite(Z) return copy(p) end
    q./=Z;q=max.(q,support_floor.*p);q./=sum(q);return q
end
q_born(fr,node)=copy(fr.probabilities)
q_naive(fr,node)=normalized_q(fr.probabilities.*fr.target.^2,fr.probabilities)
q_if_pure(fr,node)=normalized_q(fr.probabilities.*abs.(node.psi),fr.probabilities)
function q_if_mix(fr,node)
    q0=q_if_pure(fr,node);return (1-if_born_epsilon).*q0.+if_born_epsilon.*fr.probabilities
end
function q_hierarchical(fr,nodes)
    p=fr.probabilities;A2=zeros(Float64,length(p))
    for n in nodes
        importance=n.probability_mass^hierarchy_alpha*max(n.G1,0.0);Phi=n.psi./n.probability_mass
        @. A2+=importance*Phi^2
    end
    return normalized_q(p.*sqrt.(A2),p)
end
function theoretical_V(p,psi,q)
    s=0.0
    @inbounds for i in eachindex(p)
        if p[i]>0&&psi[i]!=0 q[i]>0||return Inf;s+=p[i]^2*psi[i]^2/q[i] end
    end
    return s
end
function sampled_margin(X,y,p,q,draws,node)
    in_node=falses(length(p));in_node[node.state_indices].=true;localdraw=[i for i in draws if in_node[i]]
    isempty(localdraw)&&return (margin=NaN,best=0,selected_gain=NaN)
    u,c=compress_indices(localdraw);wraw=c.*p[u]./q[u];Z=sum(wraw);Z>0||return (margin=NaN,best=0,selected_gain=NaN)
    w=wraw./Z;g=gain_landscape(X[u,:],y[u],w,collect(eachindex(u));min_weight=sampled_min_leaf_weight)
    best,_,_,_=top_two(g);ga=node.f1<=length(g) ? g[node.f1] : -Inf;gb=node.f2<=length(g) ? g[node.f2] : -Inf
    D=(isfinite(ga)&&isfinite(gb)) ? ga-gb : NaN;sg=(best!=0&&isfinite(g[best])) ? g[best] : NaN
    return (margin=D,best=best,selected_gain=sg)
end
diagnostic_seed(run,epoch,M,nodej,samplerj,rep)=diagnostic_seed_base+10_000_000*run+100_000*epoch+10_000*nodej+1_000*samplerj+M+rep

function validate_gauge!(X,fr,nodes)
    max_if_cross=0.0;max_gain_shift=0.0;max_margin_shift=0.0
    max_if_shift_abs=0.0;max_if_shift_rel=0.0
    for n in nodes
        max_if_cross=max(max_if_cross,maximum(abs.(n.psi.-n.oldpsi)))
        for c in gauge_shifts
            ys=fr.target.+c;P=n.probability_mass;pc=fr.probabilities./P
            gs=gain_landscape(X,ys,pc,n.state_indices;min_weight=exact_min_leaf_weight)
            max_gain_shift=max(max_gain_shift,abs(gs[n.f1]-n.G1),abs(gs[n.f2]-n.G2))
            Ds=gs[n.f1]-gs[n.f2];max_margin_shift=max(max_margin_shift,abs(Ds-n.margin))
            as=gain_and_if_closed(X,ys,fr.probabilities,n.state_indices,n.f1);bs=gain_and_if_closed(X,ys,fr.probabilities,n.state_indices,n.f2)
            psis=as.psi.-bs.psi
            abs_err=maximum(abs.(psis.-n.psi))
            scale=max(1.0,maximum(abs.(n.psi)),maximum(abs.(psis)))
            rel_err=abs_err/scale
            max_if_shift_abs=max(max_if_shift_abs,abs_err);max_if_shift_rel=max(max_if_shift_rel,rel_err)
        end
    end
    @printf("  validation: old/closed IF=%.3e gain-shift=%.3e margin-shift=%.3e IF-shift(abs)=%.3e IF-shift(rel)=%.3e\n",max_if_cross,max_gain_shift,max_margin_shift,max_if_shift_abs,max_if_shift_rel)
    max_if_cross<validation_tol||error("closed-form IF disagrees with differential implementation")
    max_gain_shift<validation_tol||error("split gain is not translation invariant")
    max_margin_shift<validation_tol||error("split margin is not translation invariant")
    max_if_shift_rel<validation_tol||error("split-margin IF is not translation invariant beyond scale-aware tolerance")
    return (if_cross=max_if_cross,gain_shift=max_gain_shift,margin_shift=max_margin_shift,if_shift=max_if_shift_abs,if_shift_rel=max_if_shift_rel)
end

function diagnose(H,model,X,run,epoch)
    fr,nodes=build_oracle(H,model,X)
    @printf("\nEpoch %d: E=% .8f target RMS=%.4e nodes=%d\n",epoch,real(fr.energy),fr.target_rms,length(nodes))
    validation=validate_gauge!(X,fr,nodes);qhier=q_hierarchical(fr,nodes);rows=NamedTuple[]
    for (nj,n) in enumerate(nodes)
        samplers=[("born",q_born(fr,n)),("naive_py2",q_naive(fr,n)),("hierarchical_a2",qhier),("if_pure",q_if_pure(fr,n)),("if_mix_005",q_if_mix(fr,n))]
        Vif=theoretical_V(fr.probabilities,n.psi,q_if_pure(fr,n))
        for M in diagnostic_sample_sizes, (sj,(sname,q)) in enumerate(samplers)
            Dh=Float64[];recovered=0;valid=0;rg=Float64[]
            for rep in 1:nsampling_runs
                rng=MersenneTwister(diagnostic_seed(run,epoch,M,nj,sj,rep));draws=draw_categorical_indices(rng,q,M)
                est=sampled_margin(X,fr.target,fr.probabilities,q,draws,n)
                if isfinite(est.margin)
                    push!(Dh,est.margin);valid+=1;recovered+=est.best==n.f1
                    if isfinite(est.selected_gain)&&n.G1>0 push!(rg,est.selected_gain/n.G1) end
                end
            end
            V=theoretical_V(fr.probabilities,n.psi,q);meanD=isempty(Dh) ? NaN : mean(Dh);varD=length(Dh)>1 ? var(Dh;corrected=true) : NaN
            signrec=isempty(Dh) ? NaN : mean(Dh.>0);recovery=valid==0 ? NaN : recovered/valid;empirical_Mvar=isfinite(varD) ? M*varD : NaN
            push!(rows,(training_run=run,epoch=epoch,node_index=n.node_index,node_depth=n.node_depth,node_path=n.node_path,probability_mass=n.probability_mass,f1=n.f1,f2=n.f2,oracle_G1=n.G1,oracle_G2=n.G2,oracle_margin=n.margin,sampler=sname,M=M,repetitions=nsampling_runs,valid_repetitions=valid,split_recovery=recovery,sign_recovery=signrec,mean_margin_hat=meanD,margin_bias=meanD-n.margin,margin_variance=varD,empirical_M_variance=empirical_Mvar,theoretical_V=V,variance_ratio_empirical_theory=empirical_Mvar/V,theoretical_efficiency_vs_IF=Vif/V,mean_relative_selected_gain=isempty(rg) ? NaN : mean(rg),max_if_crosscheck_error=validation.if_cross,max_gain_gauge_error=validation.gain_shift,max_margin_gauge_error=validation.margin_shift,max_if_gauge_error=validation.if_shift,max_if_gauge_relative_error=validation.if_shift_rel))
        end
    end
    for M in diagnostic_sample_sizes
        @printf("  M=%d\n",M)
        for sname in ("born","naive_py2","hierarchical_a2","if_pure","if_mix_005")
            r=[x for x in rows if x.M==M&&x.sampler==sname]
            rec=mean(x.split_recovery for x in r if isfinite(x.split_recovery));sr=mean(x.sign_recovery for x in r if isfinite(x.sign_recovery));vr=mean(x.variance_ratio_empirical_theory for x in r if isfinite(x.variance_ratio_empirical_theory))
            @printf("    %-16s recovery=%.3f sign=%.3f <MVar/V>=%.3f\n",sname,rec,sr,vr)
        end
    end
    return rows
end

function run_training(run,X)
    H=TFIMHamiltonian(N;J=J,h=h,periodic=true);rng=MersenneTwister(base_seed+10_000*run);samples=Matrix{Int8}(undef,training_nsamples,N)
    @inbounds for i in eachindex(samples) samples[i]=rand(rng,Bool) ? Int8(1) : Int8(-1) end
    model=LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false);logamps=zeros(training_nsamples)
    for _ in 1:burn_in_sweeps GBTQuantum.sweep!(rng,model,samples,logamps) end
    rows=NamedTuple[]
    for epoch in 1:nepochs
        batch=vmc_batch(H,model,samples);yA,_=make_targets(batch);w=batch.counts
        if epoch in checkpoint_epochs append!(rows,diagnose(H,model,X,run,epoch)) end
        tree=GBTQuantum.grow_tree(batch.states,yA,w;max_depth=optimizer_max_depth,min_weight=optimizer_min_leaf_weight,min_gain=optimizer_min_gain)
        pred=predict_all(tree,batch.states);mu=weighted_mean(pred,w);isfinite(mu)&&mu!=0&&(tree=shift_tree_leaves(tree,mu))
        push!(model.logamp.trees,scale_tree(tree,eta));GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:sweeps_per_epoch GBTQuantum.sweep!(rng,model,samples,logamps) end
    end
    return rows
end
function main()
    println("\n============================================================");println("INFLUENCE-FUNCTION ACQUISITION VALIDATION")
    println("N=$N J/h=$(J/h) runs=$ntraining_runs checkpoints=$(sort(collect(checkpoint_epochs)))")
    println("M=$diagnostic_sample_sizes repetitions=$nsampling_runs oracle_depth=$oracle_depth")
    println("samplers: Born, p*y^2, hierarchical(alpha=$hierarchy_alpha), IF, IF+$(100if_born_epsilon)% Born");println("============================================================")
    X=enumerate_states(N);rows=NamedTuple[]
    for run in 1:ntraining_runs @printf("\nTraining trajectory %d/%d\n",run,ntraining_runs);append!(rows,run_training(run,X)) end
    outdir=joinpath(@__DIR__,"results");mkpath(outdir);path=joinpath(outdir,"influence_acquisition_validation.csv");write_namedtuple_csv(path,rows)
    println("\nResults written to experiments/results/influence_acquisition_validation.csv")
end
export main,run_training,diagnose
end
if abspath(PROGRAM_FILE)==@__FILE__ InfluenceAcquisitionValidationExperiment.main() end
