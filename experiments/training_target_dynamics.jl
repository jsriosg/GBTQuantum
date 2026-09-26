module TrainingTargetDynamicsExperiment

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
const J_values = [0.05, 0.50, 1.00, 2.00]
const sample_sizes = [64, 256]
const nepochs = 64
const diagnostic_epochs = Set([1, 2, 4, 8, 16, 32, 64])
const nruns = 10
const max_depth = 4
const eta = 0.05
const burn_in_sweeps = 100
const sweeps_per_epoch = 1
const min_leaf_weight = 1.0
const min_gain = 0.0
const base_seed = 730_000

function weighted_rms(y,w)
    W = sum(w)
    return sqrt(sum(Float64(w[i])*Float64(y[i])^2 for i in eachindex(y,w))/W)
end

function safe_correlation(x,y)
    (std(x) <= eps(Float64) || std(y) <= eps(Float64)) && return NaN
    return cor(x,y)
end

function probability_weighted_correlation(x,y,p)
    μx = sum(p .* x); μy = sum(p .* y)
    dx = x .- μx; dy = y .- μy
    varx = sum(p .* dx.^2); vary = sum(p .* dy.^2)
    (varx <= eps(Float64) || vary <= eps(Float64)) && return NaN
    return sum(p .* dx .* dy)/sqrt(varx*vary)
end

function exact_target(H::TFIMHamiltonian,model::LogGBState,states::Matrix{Int8},p::Vector{Float64})
    d = size(states,1)
    eloc = Vector{ComplexF64}(undef,d)
    @inbounds for s in 1:d
        eloc[s] = local_energy!(H,model,@view(states[s,:]))
    end
    E = sum(p .* eloc)
    target = [-real(eloc[s]-E) for s in 1:d]
    variance = sum(p .* abs2.(eloc .- real(E)))
    return (target=target,local_energy=eloc,energy=E,variance=variance)
end

function visited_probability_mass(probabilities::Vector{Float64},visited::Set{UInt64})
    mass = 0.0
    @inbounds for key in visited
        mass += probabilities[Int(key)+1]
    end
    return mass
end

participation_ratio(probabilities) = 1.0/sum(abs2,probabilities)

function run_training_diagnostics(J::Float64,M::Int,run::Int,full_states::Matrix{Int8})
    d = size(full_states,1)
    H = TFIMHamiltonian(N;J=J,h=h,periodic=true)
    rng = MersenneTwister(base_seed + round(Int,100_000*J) + 1_000*M + run)
    samples = Matrix{Int8}(undef,M,N)
    @inbounds for i in eachindex(samples); samples[i] = rand(rng,Bool) ? Int8(1) : Int8(-1); end
    model = LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    logamps = zeros(Float64,M)
    for _ in 1:burn_in_sweeps; GBTQuantum.sweep!(rng,model,samples,logamps); end
    visited = Set{UInt64}()
    rows = NamedTuple[]

    for epoch in 1:nepochs
        batch = vmc_batch(H,model,samples)
        yA,_ = make_targets(batch)
        weights = batch.counts
        @inbounds for j in axes(batch.states,1)
            push!(visited,GBTQuantum.spin_key(@view(batch.states[j,:])))
        end

        tree = GBTQuantum.grow_tree(batch.states,yA,weights;max_depth=max_depth,
                                    min_weight=min_leaf_weight,min_gain=min_gain)
        train_prediction = predict_all(tree,batch.states)
        μtree = weighted_mean(train_prediction,weights)
        if μtree != 0.0
            tree = shift_tree_leaves(tree,μtree)
            train_prediction = predict_all(tree,batch.states)
        end

        if epoch in diagnostic_epochs
            probabilities = exact_probabilities(model,full_states)
            exact = exact_target(H,model,full_states,probabilities)
            exact_y = exact.target
            hilbert_prediction = predict_all(tree,full_states)
            sampled_target_rms = weighted_rms(yA,weights)
            exact_target_rms = sqrt(sum(probabilities .* exact_y.^2))
            train_r2 = weighted_r2(yA,train_prediction,weights)
            hilbert_r2 = ordinary_r2(exact_y,hilbert_prediction)
            hilbert_corr = safe_correlation(exact_y,hilbert_prediction)
            probability_r2 = weighted_r2(exact_y,hilbert_prediction,probabilities)
            probability_corr = probability_weighted_correlation(exact_y,hilbert_prediction,probabilities)
            nunique = size(batch.states,1)
            instantaneous_coverage = nunique/d
            cumulative_coverage = length(visited)/d
            probability_coverage = visited_probability_mass(probabilities,visited)
            pr = participation_ratio(probabilities)
            pr_fraction = pr/d
            sampled_energy = real(batch.energy)
            exact_model_energy = real(exact.energy)

            push!(rows,(J=J,h=h,J_over_h=J/h,nsamples=M,run=run,epoch=epoch,
                sampled_energy=sampled_energy,exact_model_energy=exact_model_energy,
                exact_model_variance=exact.variance,sampled_target_rms=sampled_target_rms,
                exact_target_rms=exact_target_rms,
                target_rms_ratio=sampled_target_rms/max(exact_target_rms,eps(Float64)),
                train_r2=train_r2,probability_r2=probability_r2,hilbert_r2=hilbert_r2,
                probability_corr=probability_corr,hilbert_corr=hilbert_corr,
                generalization_gap=train_r2-hilbert_r2,sampling_gap=train_r2-probability_r2,
                relevance_gap=probability_r2-hilbert_r2,
                instantaneous_state_coverage=instantaneous_coverage,
                cumulative_state_coverage=cumulative_coverage,
                cumulative_probability_coverage=probability_coverage,
                participation_ratio=pr,participation_fraction=pr_fraction,unique_fraction=nunique/M))

            @printf("J/h=%4.2f  M=%4d  run=%2d  epoch=%2d  R²tr=% .3f  R²p=% .3f  R²H=% .3f  ρp=% .3f  ρH=% .3f  Cprob=%.3f  PR/H=%.3f\n",
                    J/h,M,run,epoch,train_r2,probability_r2,hilbert_r2,probability_corr,
                    hilbert_corr,probability_coverage,pr_fraction)
        end

        push!(model.logamp.trees,scale_tree(tree,eta))
        GBTQuantum.refresh_logamps!(logamps,model,samples)
        for _ in 1:sweeps_per_epoch; GBTQuantum.sweep!(rng,model,samples,logamps); end
    end
    return rows
end

function print_final_summary(all_rows)
    println("\n============== FINAL-EPOCH SUMMARY ==============\n")
    println(" J/h     M      R²tr      R²p       R²H       ρp        ρH      Cprob     PR/H")
    println("--------------------------------------------------------------------------------")
    for J in J_values, M in sample_sizes
        selected = [r for r in all_rows if r.J==J && r.nsamples==M && r.epoch==nepochs]
        @printf("%5.2f  %4d   %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f\n",
                J/h,M,finite_mean(r.train_r2 for r in selected),finite_mean(r.probability_r2 for r in selected),
                finite_mean(r.hilbert_r2 for r in selected),finite_mean(r.probability_corr for r in selected),
                finite_mean(r.hilbert_corr for r in selected),finite_mean(r.cumulative_probability_coverage for r in selected),
                finite_mean(r.participation_fraction for r in selected))
    end
    println("================================================")
end

function main()
    d = 1 << N
    println("\n================================================")
    println("TRAINING-TARGET DYNAMICS")
    println("N                  = ",N)
    println("Hilbert dimension  = ",d)
    println("h                  = ",h)
    println("Epochs             = ",nepochs)
    println("Independent runs   = ",nruns)
    println("Diagnostic epochs  = ",sort(collect(diagnostic_epochs)))
    println("================================================\n")
    full_states = enumerate_states(N)
    all_rows = NamedTuple[]
    for J in J_values, M in sample_sizes, run in 1:nruns
        append!(all_rows,run_training_diagnostics(J,M,run,full_states))
    end
    output_dir = joinpath(@__DIR__,"results"); mkpath(output_dir)
    csv_path = joinpath(output_dir,"training_target_dynamics.csv")
    write_namedtuple_csv(csv_path,all_rows)
    println("\nRESULTS WRITTEN TO\n",csv_path)
    print_final_summary(all_rows)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    TrainingTargetDynamicsExperiment.main()
end
