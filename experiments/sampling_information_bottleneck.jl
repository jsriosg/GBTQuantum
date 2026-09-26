using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Random
using Statistics
using Printf

# ============================================================
# SAMPLING INFORMATION BOTTLENECK
# ============================================================
# At frozen PRE-UPDATE checkpoints compare:
#   exact   : all Hilbert states with exact p(x) weights
#   iid     : M independent samples from exact p(x)
#   mcmc    : M states from the package Metropolis sampler
#   uniform : M uniform Hilbert-space samples
#
# All sampled trees use the EXACT frozen target
#   u(x) = -Re[E_L(x)-E]
# so differences from the oracle isolate sampling in tree fitting.
#
# Coverage:
#   C_H = observed Hilbert fraction
#   C_p = exact probability mass observed
#   C_q = learning-signal mass observed,
#         q(x) ∝ p(x)[u(x)-<u>_p]^2
# ============================================================

const N = 12
const h = 1.0
const J_values = [0.05, 2.00]

const training_nsamples = 256
const nepochs = 64
const checkpoint_epochs = Set([1, 8, 32, 64])
const ntraining_runs = 3

const optimizer_max_depth = 4
const eta = 0.05
const burn_in_sweeps = 100
const sweeps_per_epoch = 1
const optimizer_min_leaf_weight = 1.0
const optimizer_min_gain = 0.0

const diagnostic_sample_sizes = [64, 256, 1024]
const diagnostic_depths = [4, 8]
const nsampling_runs = 20
const diagnostic_mcmc_burn_in_sweeps = 100

const exact_min_leaf_weight = 1e-14
const exact_min_gain = 0.0
const sampled_min_leaf_weight = 1.0
const sampled_min_gain = 0.0

const base_seed = 910_000
const diagnostic_seed_base = 7_310_000

function enumerate_states(N::Int)
    d = 1 << N
    states = Matrix{Int8}(undef, d, N)
    @inbounds for s in 0:(d-1)
        for i in 1:N
            states[s+1,i] = ((s >> (i-1)) & 1) == 1 ? Int8(1) : Int8(-1)
        end
    end
    return states
end

@inline function state_index(x::AbstractVector{<:Real})
    s = 0
    @inbounds for i in eachindex(x)
        x[i] > 0 && (s |= 1 << (i-1))
    end
    return s + 1
end

function state_indices(X::AbstractMatrix{<:Real})
    idx = Vector{Int}(undef, size(X,1))
    @inbounds for r in axes(X,1)
        idx[r] = state_index(@view X[r,:])
    end
    return idx
end

function weighted_mean_local(y,w)
    total = 0.0
    W = 0.0
    @inbounds for i in eachindex(y,w)
        wi = Float64(w[i])
        total += wi * Float64(y[i])
        W += wi
    end
    return total/W
end

function weighted_r2(ytrue,ypred,w)
    mu = weighted_mean_local(ytrue,w)
    ss_res = 0.0
    ss_tot = 0.0
    @inbounds for i in eachindex(ytrue,ypred,w)
        wi = Float64(w[i])
        dr = Float64(ytrue[i]) - Float64(ypred[i])
        dt = Float64(ytrue[i]) - mu
        ss_res += wi*dr^2
        ss_tot += wi*dt^2
    end
    ss_tot <= eps(Float64) && return NaN
    return 1.0 - ss_res/ss_tot
end

function ordinary_r2(ytrue,ypred)
    mu = mean(ytrue)
    ss_res = sum((ytrue .- ypred).^2)
    ss_tot = sum((ytrue .- mu).^2)
    ss_tot <= eps(Float64) && return NaN
    return 1.0 - ss_res/ss_tot
end

function predict_all(tree::RegressionTree,X::AbstractMatrix{<:Real})
    y = Vector{Float64}(undef,size(X,1))
    @inbounds for i in axes(X,1)
        y[i] = predict(tree,@view(X[i,:]))
    end
    return y
end

function shift_tree_leaves(tree::RegressionTree,shift::Float64)
    nodes = copy(tree.nodes)
    @inbounds for i in eachindex(nodes)
        n = nodes[i]
        if n.isleaf
            nodes[i] = Node(n.feature,n.value-shift,n.left,n.right,true)
        end
    end
    return RegressionTree(nodes)
end

function scale_tree(tree::RegressionTree,scale::Float64)
    nodes = [n.isleaf ? Node(n.feature,scale*n.value,n.left,n.right,true) : n for n in tree.nodes]
    return RegressionTree(nodes)
end

tree_leaf_count(tree::RegressionTree) = count(n -> n.isleaf,tree.nodes)

function exact_probabilities(model::LogGBState,states::Matrix{Int8})
    d = size(states,1)
    logweights = Vector{Float64}(undef,d)
    @inbounds for s in 1:d
        logweights[s] = 2.0*logamplitude(model,@view(states[s,:]))
    end
    m = maximum(logweights)
    weights = exp.(logweights .- m)
    return weights/sum(weights)
end

function exact_frozen_problem(H::TFIMHamiltonian,model::LogGBState,states::Matrix{Int8})
    p = exact_probabilities(model,states)
    d = size(states,1)
    eloc = Vector{ComplexF64}(undef,d)
    @inbounds for s in 1:d
        eloc[s] = local_energy!(H,model,@view(states[s,:]))
    end
    E = sum(p .* eloc)
    target = Vector{Float64}(undef,d)
    @inbounds for s in 1:d
        target[s] = -real(eloc[s]-E)
    end
    target_mean = sum(p .* target)
    centered = target .- target_mean
    signal_density = p .* centered.^2
    signal_total = sum(signal_density)
    q = signal_total > eps(Float64) ? signal_density/signal_total : zeros(Float64,d)
    target_rms = sqrt(sum(p .* target.^2))
    PR = 1.0/sum(abs2,p)
    return (probabilities=p,local_energy=eloc,energy=E,target=target,
            target_mean=target_mean,target_rms=target_rms,q=q,
            signal_total=signal_total,participation_ratio=PR,
            participation_fraction=PR/d)
end

function fit_exact_tree(full_states::Matrix{Int8},frozen,depth::Int)
    tree = GBTQuantum.grow_tree(full_states,frozen.target,frozen.probabilities;
                                max_depth=depth,min_weight=exact_min_leaf_weight,
                                min_gain=exact_min_gain)
    pred = predict_all(tree,full_states)
    mu = weighted_mean_local(pred,frozen.probabilities)
    if mu != 0.0
        tree = shift_tree_leaves(tree,mu)
        pred = predict_all(tree,full_states)
    end
    return tree,pred
end

function draw_categorical_indices(rng::AbstractRNG,p::AbstractVector{<:Real},M::Int)
    cdf = cumsum(Float64.(p))
    cdf[end] = 1.0
    idx = Vector{Int}(undef,M)
    @inbounds for i in 1:M
        idx[i] = searchsortedfirst(cdf,rand(rng))
    end
    return idx
end

draw_uniform_indices(rng::AbstractRNG,d::Int,M::Int) = rand(rng,1:d,M)

function draw_mcmc_indices(rng::AbstractRNG,model::LogGBState,M::Int)
    samples = Matrix{Int8}(undef,M,N)
    @inbounds for i in eachindex(samples)
        samples[i] = rand(rng,Bool) ? Int8(1) : Int8(-1)
    end
    logamps = Vector{Float64}(undef,M)
    GBTQuantum.refresh_logamps!(logamps,model,samples)
    acc = 0.0
    for _ in 1:diagnostic_mcmc_burn_in_sweeps
        acc += GBTQuantum.sweep!(rng,model,samples,logamps)
    end
    return state_indices(samples),acc/diagnostic_mcmc_burn_in_sweeps
end

function compress_indices(indices::Vector{Int})
    d = Dict{Int,Int}()
    @inbounds for idx in indices
        d[idx] = get(d,idx,0)+1
    end
    unique_idx = sort!(collect(keys(d)))
    counts = Float64[d[idx] for idx in unique_idx]
    return unique_idx,counts
end

function coverage_metrics(unique_idx::Vector{Int},frozen,hilbert_dimension::Int)
    C_H = length(unique_idx)/hilbert_dimension
    C_p = sum(frozen.probabilities[unique_idx])
    C_q = sum(frozen.q[unique_idx])
    return C_H,C_p,C_q
end

function fit_sampled_tree(full_states::Matrix{Int8},frozen,indices::Vector{Int},depth::Int)
    unique_idx,counts = compress_indices(indices)
    X = full_states[unique_idx,:]
    y = frozen.target[unique_idx]
    tree = GBTQuantum.grow_tree(X,y,counts;max_depth=depth,
                                min_weight=sampled_min_leaf_weight,
                                min_gain=sampled_min_gain)
    sample_pred = predict_all(tree,X)
    mu = weighted_mean_local(sample_pred,counts)
    mu != 0.0 && (tree = shift_tree_leaves(tree,mu))
    full_pred = predict_all(tree,full_states)
    return tree,full_pred,unique_idx
end

function evaluate_tree(prediction,frozen)
    Rp = weighted_r2(frozen.target,prediction,frozen.probabilities)
    RH = ordinary_r2(frozen.target,prediction)
    return Rp,RH
end

function diagnose_sampling_information(H::TFIMHamiltonian,model::LogGBState,
                                       full_states::Matrix{Int8},J::Float64,
                                       training_run::Int,epoch::Int)
    frozen = exact_frozen_problem(H,model,full_states)
    dH = size(full_states,1)
    @printf("\n  Frozen epoch %2d: E=% .8f  target RMS=%.4e  PR/H=%.6f\n",
            epoch,real(frozen.energy),frozen.target_rms,frozen.participation_fraction)
    abs(frozen.target_mean) > 1e-10 && @warn "Exact frozen target mean is nonzero" J training_run epoch frozen.target_mean

    rows = NamedTuple[]
    oracle = Dict{Int,NamedTuple}()
    for depth in diagnostic_depths
        tree,pred = fit_exact_tree(full_states,frozen,depth)
        Rp,RH = evaluate_tree(pred,frozen)
        oracle[depth] = (Rp=Rp,RH=RH,leaves=tree_leaf_count(tree))
        @printf("    oracle depth=%d: R²p=% .6f  R²H=% .6f\n",depth,Rp,RH)
    end

    for M in diagnostic_sample_sizes
        for sampling_run in 1:nsampling_runs
            seed_core = diagnostic_seed_base + round(Int,100_000*J) +
                        100_000*training_run + 1_000*epoch + 10*sampling_run + M
            iid_idx = draw_categorical_indices(MersenneTwister(seed_core+1),frozen.probabilities,M)
            mcmc_idx,mcmc_acc = draw_mcmc_indices(MersenneTwister(seed_core+2),model,M)
            uniform_idx = draw_uniform_indices(MersenneTwister(seed_core+3),dH,M)

            sample_sets = ((method="iid",indices=iid_idx,acceptance=NaN),
                           (method="mcmc",indices=mcmc_idx,acceptance=mcmc_acc),
                           (method="uniform",indices=uniform_idx,acceptance=NaN))

            for S in sample_sets
                unique_idx,_ = compress_indices(S.indices)
                C_H,C_p,C_q = coverage_metrics(unique_idx,frozen,dH)
                for depth in diagnostic_depths
                    tree,pred,fitted_unique = fit_sampled_tree(full_states,frozen,S.indices,depth)
                    Rp,RH = evaluate_tree(pred,frozen)
                    push!(rows,(
                        J=J,h=h,J_over_h=J/h,training_run=training_run,epoch=epoch,
                        M=M,sampling_run=sampling_run,sampling_method=S.method,tree_depth=depth,
                        frozen_energy=real(frozen.energy),frozen_target_rms=frozen.target_rms,
                        participation_ratio=frozen.participation_ratio,
                        participation_fraction=frozen.participation_fraction,
                        unique_states=length(fitted_unique),hilbert_coverage=C_H,
                        probability_coverage=C_p,signal_coverage=C_q,
                        mcmc_acceptance=S.acceptance,
                        exact_Rp=oracle[depth].Rp,sampled_Rp=Rp,
                        sampling_penalty_Rp=oracle[depth].Rp-Rp,
                        exact_RH=oracle[depth].RH,sampled_RH=RH,
                        sampling_penalty_RH=oracle[depth].RH-RH,
                        exact_leaf_count=oracle[depth].leaves,
                        sampled_leaf_count=tree_leaf_count(tree)))
                end
            end
        end

        selected = [r for r in rows if r.M==M && r.tree_depth==4]
        for method in ("iid","mcmc","uniform")
            subset = [r for r in selected if r.sampling_method==method]
            if !isempty(subset)
                @printf("    M=%4d %-7s d=4: <Cp>=%.3f <Cq>=%.3f <R²p>=%.3f <penalty>=%.3f\n",
                        M,method,mean(r.probability_coverage for r in subset),
                        mean(r.signal_coverage for r in subset),mean(r.sampled_Rp for r in subset),
                        mean(r.sampling_penalty_Rp for r in subset))
            end
        end
    end
    return rows
end

function run_training_trajectory(J::Float64,training_run::Int,full_states::Matrix{Int8})
    H = TFIMHamiltonian(N;J=J,h=h,periodic=true)
    rng = MersenneTwister(base_seed + round(Int,100_000*J) + 10_000*training_run)
    samples = Matrix{Int8}(undef,training_nsamples,N)
    @inbounds for i in eachindex(samples)
        samples[i] = rand(rng,Bool) ? Int8(1) : Int8(-1)
    end
    model = LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    logamps = zeros(Float64,training_nsamples)
    for _ in 1:burn_in_sweeps
        GBTQuantum.sweep!(rng,model,samples,logamps)
    end
    rows = NamedTuple[]
    for epoch in 1:nepochs
        batch = vmc_batch(H,model,samples)
        yA,_ = make_targets(batch)
        weights = batch.counts
        if epoch in checkpoint_epochs
            append!(rows,diagnose_sampling_information(H,model,full_states,J,training_run,epoch))
        end
        tree = GBTQuantum.grow_tree(batch.states,yA,weights;max_depth=optimizer_max_depth,
                                    min_weight=optimizer_min_leaf_weight,min_gain=optimizer_min_gain)
        train_pred = predict_all(tree,batch.states)
        mu = weighted_mean_local(train_pred,weights)
        mu != 0.0 && (tree = shift_tree_leaves(tree,mu))
        push!(model.logamp.trees,scale_tree(tree,eta))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:sweeps_per_epoch
            GBTQuantum.sweep!(rng,model,samples,logamps)
        end
    end
    return rows
end

function write_csv(path,rows)
    isempty(rows) && error("No diagnostic rows generated.")
    names = propertynames(first(rows))
    open(path,"w") do io
        println(io,join(string.(names),","))
        for row in rows
            println(io,join([getproperty(row,n) for n in names],","))
        end
    end
end

function finite_mean(values)
    x = Float64[v for v in values if isfinite(v)]
    return isempty(x) ? NaN : mean(x)
end

function print_summary(rows)
    println("\n\n============================================================")
    println("SAMPLING INFORMATION BOTTLENECK SUMMARY")
    println("============================================================")
    println(" J/h epoch    M method  d      <CH>      <Cp>      <Cq>      <R²p>    penalty")
    println("--------------------------------------------------------------------------------")
    for J in J_values, epoch in sort(collect(checkpoint_epochs)), M in diagnostic_sample_sizes,
        method in ("iid","mcmc","uniform"), depth in diagnostic_depths
        S = [r for r in rows if r.J==J && r.epoch==epoch && r.M==M &&
             r.sampling_method==method && r.tree_depth==depth]
        isempty(S) && continue
        @printf("%4.2f %4d %4d %-7s %d   %7.3f   %7.3f   %7.3f   %8.3f   %8.3f\n",
                J/h,epoch,M,method,depth,
                finite_mean(r.hilbert_coverage for r in S),
                finite_mean(r.probability_coverage for r in S),
                finite_mean(r.signal_coverage for r in S),
                finite_mean(r.sampled_Rp for r in S),
                finite_mean(r.sampling_penalty_Rp for r in S))
    end
end

function main()
    dH = 1 << N
    println("\n============================================================")
    println("SAMPLING INFORMATION BOTTLENECK")
    println("N                     = ",N)
    println("Hilbert dimension     = ",dH)
    println("J/h values            = ",J_values)
    println("Training population   = ",training_nsamples)
    println("Training trajectories = ",ntraining_runs)
    println("Checkpoints           = ",sort(collect(checkpoint_epochs)))
    println("Diagnostic M          = ",diagnostic_sample_sizes)
    println("Diagnostic depths     = ",diagnostic_depths)
    println("Sampling runs         = ",nsampling_runs)
    println("MCMC burn-in sweeps   = ",diagnostic_mcmc_burn_in_sweeps)
    println("============================================================")

    full_states = enumerate_states(N)
    all_rows = NamedTuple[]
    for J in J_values
        println("\n============================================================")
        @printf("J/h = %.4f\n",J/h)
        println("============================================================")
        for training_run in 1:ntraining_runs
            @printf("\nTraining trajectory %d/%d\n",training_run,ntraining_runs)
            append!(all_rows,run_training_trajectory(J,training_run,full_states))
        end
    end

    output_dir = joinpath(@__DIR__,"results")
    mkpath(output_dir)
    csv_path = joinpath(output_dir,"sampling_information_bottleneck.csv")
    write_csv(csv_path,all_rows)
    print_summary(all_rows)
    println("\n============================================================")
    println("RESULTS WRITTEN TO")
    println(csv_path)
    println("============================================================")
end

main()
