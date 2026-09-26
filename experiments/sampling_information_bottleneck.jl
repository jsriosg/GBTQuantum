module SamplingInformationBottleneckExperiment

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using GBTQuantum
using Random
using Statistics
using Printf

include("ExperimentUtils.jl")
using .ExperimentUtils

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

function exact_frozen_problem(H::TFIMHamiltonian, model::LogGBState, states::Matrix{Int8})
    p = exact_probabilities(model, states)
    d = size(states,1)
    eloc = Vector{ComplexF64}(undef,d)
    @inbounds for s in 1:d
        eloc[s] = local_energy!(H,model,@view(states[s,:]))
    end
    E = sum(p .* eloc)
    target = [-real(eloc[s]-E) for s in 1:d]
    target_mean = sum(p .* target)
    centered = target .- target_mean
    signal_density = p .* centered.^2
    signal_total = sum(signal_density)
    q = signal_total > eps(Float64) ? signal_density/signal_total : zeros(Float64,d)
    target_rms = sqrt(sum(p .* target.^2))
    PR = 1.0/sum(abs2,p)
    return (probabilities=p, local_energy=eloc, energy=E, target=target,
            target_mean=target_mean, target_rms=target_rms, q=q,
            signal_total=signal_total, participation_ratio=PR,
            participation_fraction=PR/d)
end

function fit_exact_tree(full_states::Matrix{Int8}, frozen, depth::Int)
    tree = GBTQuantum.grow_tree(full_states,frozen.target,frozen.probabilities;
                                max_depth=depth,min_weight=exact_min_leaf_weight,min_gain=exact_min_gain)
    pred = predict_all(tree,full_states)
    mu = weighted_mean(pred,frozen.probabilities)
    if mu != 0.0
        tree = shift_tree_leaves(tree,mu)
        pred = predict_all(tree,full_states)
    end
    return tree,pred
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

function coverage_metrics(unique_idx,frozen,dH)
    return length(unique_idx)/dH, sum(frozen.probabilities[unique_idx]), sum(frozen.q[unique_idx])
end

function fit_sampled_tree(full_states,frozen,indices,depth)
    unique_idx,counts = compress_indices(indices)
    X = full_states[unique_idx,:]
    y = frozen.target[unique_idx]
    tree = GBTQuantum.grow_tree(X,y,counts;max_depth=depth,
                                min_weight=sampled_min_leaf_weight,min_gain=sampled_min_gain)
    sample_pred = predict_all(tree,X)
    mu = weighted_mean(sample_pred,counts)
    mu != 0.0 && (tree = shift_tree_leaves(tree,mu))
    return tree,predict_all(tree,full_states),unique_idx
end

evaluate_tree(pred,frozen) = (weighted_r2(frozen.target,pred,frozen.probabilities), ordinary_r2(frozen.target,pred))

function diagnose_sampling_information(H,model,full_states,J,training_run,epoch)
    frozen = exact_frozen_problem(H,model,full_states)
    dH = size(full_states,1)
    @printf("\n  Frozen epoch %2d: E=% .8f  target RMS=%.4e  PR/H=%.6f\n",
            epoch,real(frozen.energy),frozen.target_rms,frozen.participation_fraction)
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
            seed_core = diagnostic_seed_base + round(Int,100_000*J) + 100_000*training_run + 1_000*epoch + 10*sampling_run + M
            iid_idx = draw_categorical_indices(MersenneTwister(seed_core+1),frozen.probabilities,M)
            mcmc_idx,mcmc_acc = draw_mcmc_indices(MersenneTwister(seed_core+2),model,M)
            uniform_idx = draw_uniform_indices(MersenneTwister(seed_core+3),dH,M)
            sample_sets = ((method="iid",indices=iid_idx,acceptance=NaN),
                           (method="mcmc",indices=mcmc_idx,acceptance=mcmc_acc),
                           (method="uniform",indices=uniform_idx,acceptance=NaN))
            for S in sample_sets
                unique_idx,_ = compress_indices(S.indices)
                CH,Cp,Cq = coverage_metrics(unique_idx,frozen,dH)
                for depth in diagnostic_depths
                    tree,pred,fitted_unique = fit_sampled_tree(full_states,frozen,S.indices,depth)
                    Rp,RH = evaluate_tree(pred,frozen)
                    push!(rows,(J=J,h=h,J_over_h=J/h,training_run=training_run,epoch=epoch,
                        M=M,sampling_run=sampling_run,sampling_method=S.method,tree_depth=depth,
                        frozen_energy=real(frozen.energy),frozen_target_rms=frozen.target_rms,
                        participation_ratio=frozen.participation_ratio,participation_fraction=frozen.participation_fraction,
                        unique_states=length(fitted_unique),hilbert_coverage=CH,probability_coverage=Cp,signal_coverage=Cq,
                        mcmc_acceptance=S.acceptance,exact_Rp=oracle[depth].Rp,sampled_Rp=Rp,
                        sampling_penalty_Rp=oracle[depth].Rp-Rp,exact_RH=oracle[depth].RH,sampled_RH=RH,
                        sampling_penalty_RH=oracle[depth].RH-RH,exact_leaf_count=oracle[depth].leaves,
                        sampled_leaf_count=tree_leaf_count(tree)))
                end
            end
        end
        selected = [r for r in rows if r.M==M && r.tree_depth==4]
        for method in ("iid","mcmc","uniform")
            subset = [r for r in selected if r.sampling_method==method]
            isempty(subset) && continue
            @printf("    M=%4d %-7s d=4: <Cp>=%.3f <Cq>=%.3f <R²p>=%.3f <penalty>=%.3f\n",M,method,
                    mean(r.probability_coverage for r in subset),mean(r.signal_coverage for r in subset),
                    mean(r.sampled_Rp for r in subset),mean(r.sampling_penalty_Rp for r in subset))
        end
    end
    return rows
end

function run_training_trajectory(J::Float64,training_run::Int,full_states::Matrix{Int8})
    H = TFIMHamiltonian(N;J=J,h=h,periodic=true)
    rng = MersenneTwister(base_seed + round(Int,100_000*J) + 10_000*training_run)
    samples = Matrix{Int8}(undef,training_nsamples,N)
    @inbounds for i in eachindex(samples); samples[i] = rand(rng,Bool) ? Int8(1) : Int8(-1); end
    model = LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    logamps = zeros(Float64,training_nsamples)
    for _ in 1:burn_in_sweeps; GBTQuantum.sweep!(rng,model,samples,logamps); end
    rows = NamedTuple[]
    for epoch in 1:nepochs
        batch = vmc_batch(H,model,samples)
        yA,_ = make_targets(batch); weights = batch.counts
        epoch in checkpoint_epochs && append!(rows,diagnose_sampling_information(H,model,full_states,J,training_run,epoch))
        tree = GBTQuantum.grow_tree(batch.states,yA,weights;max_depth=optimizer_max_depth,
                                    min_weight=optimizer_min_leaf_weight,min_gain=optimizer_min_gain)
        mu = weighted_mean(predict_all(tree,batch.states),weights)
        mu != 0.0 && (tree = shift_tree_leaves(tree,mu))
        push!(model.logamp.trees,scale_tree(tree,eta))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:sweeps_per_epoch; GBTQuantum.sweep!(rng,model,samples,logamps); end
    end
    return rows
end

function print_summary(rows)
    println("\n\n============================================================")
    println("SAMPLING INFORMATION BOTTLENECK SUMMARY")
    println("============================================================")
    println(" J/h epoch    M method  d      <CH>      <Cp>      <Cq>      <R²p>    penalty")
    println("--------------------------------------------------------------------------------")
    for J in J_values, epoch in sort(collect(checkpoint_epochs)), M in diagnostic_sample_sizes,
        method in ("iid","mcmc","uniform"), depth in diagnostic_depths
        S = [r for r in rows if r.J==J && r.epoch==epoch && r.M==M && r.sampling_method==method && r.tree_depth==depth]
        isempty(S) && continue
        @printf("%4.2f %4d %4d %-7s %d   %7.3f   %7.3f   %7.3f   %8.3f   %8.3f\n",J/h,epoch,M,method,depth,
                finite_mean(r.hilbert_coverage for r in S),finite_mean(r.probability_coverage for r in S),
                finite_mean(r.signal_coverage for r in S),finite_mean(r.sampled_Rp for r in S),
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
    println("============================================================")
    full_states = enumerate_states(N); all_rows = NamedTuple[]
    for J in J_values, training_run in 1:ntraining_runs
        @printf("\nJ/h=%.4f  training trajectory %d/%d\n",J/h,training_run,ntraining_runs)
        append!(all_rows,run_training_trajectory(J,training_run,full_states))
    end
    output_dir = joinpath(@__DIR__,"results"); mkpath(output_dir)
    csv_path = joinpath(output_dir,"sampling_information_bottleneck.csv")
    write_namedtuple_csv(csv_path,all_rows)
    print_summary(all_rows)
    println("\nRESULTS WRITTEN TO\n",csv_path)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    SamplingInformationBottleneckExperiment.main()
end
