module TFIMLogVMCExperiment

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Statistics
using Plots
using Printf

function main()
    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------
    H = TFIMHamiltonian(8; J=1.0, h=1.0, periodic=true)
    cfg = TrainingConfig(
        nsamples=512,
        epochs=150,
        max_depth=4,
        eta=0.05,
        burn_in_sweeps=50,
        sweeps_per_epoch=2,
        use_phase=false,
        seed=1234,
        exact_diagnostics=true,
        exact_every=5,
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    result = train(H, cfg)
    Eexact = exact_ground_energy(H)
    epochs = 1:cfg.epochs
    energy_error = abs.(result.energy_history .- Eexact)

    validation = validation_vmc(
        result.model, H;
        nsamples=10_000,
        burn_in_sweeps=500,
        thinning_sweeps=1,
        seed=9876,
    )
    validation_error = abs(validation.energy - Eexact)
    error_in_SE = validation_error / validation.standard_error

    println()
    println("========== INDEPENDENT VALIDATION ==========")
    println("Validation energy      = ", validation.energy)
    println("Exact ground energy    = ", Eexact)
    println("Absolute error         = ", validation_error)
    println("Local-energy variance  = ", validation.variance)
    println("tau_int                 = ", validation.tau_int)
    println("Effective sample size  = ", validation.ess)
    println("ESS fraction            = ", validation.ess / 10_000)
    println("Energy standard error  = ", validation.standard_error)
    println("Error / SE              = ", error_in_SE)
    println("============================================")

    println()
    println("========== v0.2 TFIM RESULTS ==========")
    println("Final VMC energy      = ", result.energy_history[end])
    println("Exact ground energy  = ", Eexact)
    println("Absolute error       = ", energy_error[end])
    println("Final local-E var.   = ", result.variance_history[end])
    println("Mean acceptance      = ", mean(result.acceptance_history))
    println("Final unique fraction= ", result.unique_fraction_history[end])
    println("Runtime [s]          = ", result.runtime_seconds)
    println("========================================")

    exact_model = exact_model_energy(result.model, H)
    model_error = real(exact_model.energy) - Eexact
    mc_error = validation.energy - real(exact_model.energy)

    println()
    println("========== EXACT MODEL VALIDATION ==========")
    println("Exact ground energy     = ", Eexact)
    println("Exact model energy      = ", real(exact_model.energy))
    println("VMC validation energy   = ", validation.energy)
    println()
    println("Model variational error = ", model_error)
    println("VMC estimation error    = ", mc_error)
    println("Exact model variance    = ", exact_model.variance)
    println("Enumerated states       = ", exact_model.nstates)
    println("============================================")

    hilbert_dim = 2^H.N
    println()
    println("========== HILBERT-SPACE COVERAGE ==========")
    println("Number of spins        = ", H.N)
    println("Hilbert dimension      = ", hilbert_dim)
    println("Maximum coverage       = ", maximum(result.hilbert_coverage))
    println("Initial coverage       = ", result.hilbert_coverage[1])
    println("Final coverage         = ", result.hilbert_coverage[end])
    println("Approx. final unique   = ", round(Int, result.hilbert_coverage[end] * hilbert_dim))
    println("============================================")

    println()
    println("Computing physical observables...")
    gs_obs = exact_ground_observables(H)
    model_obs = exact_model_observables(result.model, H)
    vmc_obs = validation_observables(
        result.model, H;
        nsamples=20_000,
        burn_in_sweeps=200,
        thinning_sweeps=1,
        seed=98765,
    )

    println()
    println("========== PHYSICAL OBSERVABLES ==========")
    println()
    println("                     Exact GS       GBT exact       GBT VMC")
    println("------------------------------------------------------------")
    @printf("<mz>              %12.8f   %12.8f   %12.8f\n", gs_obs.mz, model_obs.mz, vmc_obs.mz)
    @printf("<|mz|>            %12.8f   %12.8f   %12.8f\n", gs_obs.abs_mz, model_obs.abs_mz, vmc_obs.abs_mz)
    @printf("<mz^2>            %12.8f   %12.8f   %12.8f\n", gs_obs.mz2, model_obs.mz2, vmc_obs.mz2)
    @printf("<mx>              %12.8f   %12.8f   %12.8f\n", gs_obs.mx, real(model_obs.mx), real(vmc_obs.mx))

    println()
    println("========== MONTE CARLO DIAGNOSTICS ==========")
    @printf("tau_int(mz)       = %.6f\n", vmc_obs.tau_mz)
    @printf("tau_int(|mz|)     = %.6f\n", vmc_obs.tau_abs_mz)
    @printf("tau_int(mz^2)     = %.6f\n", vmc_obs.tau_mz2)
    println()
    @printf("ESS(mz)           = %.2f / %d\n", vmc_obs.ess_mz, length(vmc_obs.mz_samples))
    @printf("ESS(|mz|)         = %.2f / %d\n", vmc_obs.ess_abs_mz, length(vmc_obs.abs_mz_samples))
    @printf("ESS(mz^2)         = %.2f / %d\n", vmc_obs.ess_mz2, length(vmc_obs.mz2_samples))
    println()
    @printf("SE(mz)            = %.8e\n", vmc_obs.se_mz)
    @printf("SE(|mz|)          = %.8e\n", vmc_obs.se_abs_mz)
    @printf("SE(mz^2)          = %.8e\n", vmc_obs.se_mz2)
    @printf("Im(<mx>) VMC      = %.8e\n", imag(vmc_obs.mx))
    println("=============================================")

    dist = validation_distribution(
        result.model, H;
        nsamples=100_000,
        burn_in_sweeps=500,
        thinning_sweeps=1,
        seed=13579,
    )

    println()
    println("========== DISTRIBUTION VALIDATION ==========")
    println("Samples                  = ", dist.nsamples)
    println("Hilbert dimension        = ", dist.nstates)
    println("Total variation distance = ", dist.total_variation)
    println("L1 distance              = ", dist.l1_distance)
    println("Maximum state error      = ", dist.max_abs_error)
    println("Magnetization TV         = ", dist.magnetization_tv)
    println()
    println("Magnetization distribution:")
    println()
    println("   mz          Exact GBT        VMC         Difference")
    println("-------------------------------------------------------")
    for i in eachindex(dist.magnetizations)
        @printf(
            "%6.2f      %10.6f     %10.6f     %+10.6f\n",
            dist.magnetizations[i],
            dist.exact_magnetization_prob[i],
            dist.sampled_magnetization_prob[i],
            dist.sampled_magnetization_prob[i] - dist.exact_magnetization_prob[i],
        )
    end
    println("=============================================")

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------
    p_energy = plot(
        epochs, result.energy_history;
        label="VMC energy", xlabel="Boosting iteration", ylabel="Energy",
        title="Energy convergence", linewidth=2,
    )
    hline!(p_energy, [Eexact]; label="Exact ground energy", linestyle=:dash, linewidth=2)

    mask = .!isnan.(result.exact_energy)
    @assert all(result.exact_energy[mask] .>= Eexact .- 1e-10) "Variational bound violated in exact diagnostic"
    iters = collect(eachindex(result.energy_history))

    p_exact_energy = plot(
        iters, result.energy_history;
        label="Training VMC", xlabel="Boosting iteration", ylabel="Energy",
        title="Sampled vs exact model energy",
    )
    plot!(p_exact_energy, iters[mask], result.exact_energy[mask]; marker=:circle, label="Exact model energy")
    hline!(p_exact_energy, [Eexact]; linestyle=:dash, label="Exact ground energy")

    p_exact_error = plot(
        iters, abs.(result.energy_history .- Eexact);
        yscale=:log10, label="|E_VMC - E₀|", xlabel="Boosting iteration",
        ylabel="Absolute energy error", title="True vs sampled convergence",
    )
    plot!(p_exact_error, iters[mask], abs.(result.exact_energy[mask] .- Eexact); marker=:circle, label="|E_model - E₀|")

    p_error = plot(
        epochs, energy_error;
        label="|E - E₀|", xlabel="Boosting iteration", ylabel="Absolute energy error",
        title="Energy error", yscale=:log10, linewidth=2,
    )
    p_variance = plot(
        epochs, result.variance_history;
        label="Var(E_loc)", xlabel="Boosting iteration", ylabel="Variance",
        title="Local-energy variance", yscale=:log10, linewidth=2,
    )
    p_acceptance = plot(
        epochs, result.acceptance_history;
        label="Acceptance", xlabel="Boosting iteration", ylabel="Acceptance fraction",
        title="Metropolis acceptance", ylim=(0,1), linewidth=2,
    )
    p_unique = plot(
        epochs, result.unique_fraction_history;
        label="Unique / total", xlabel="Boosting iteration", ylabel="Unique fraction",
        title="Sample diversity", ylim=(0,1), linewidth=2,
    )
    p_tree = plot(
        epochs, result.magnitude_fit_mse;
        label="Magnitude tree MSE", xlabel="Boosting iteration", ylabel="Weighted MSE",
        title="Tree fit quality", yscale=:log10, linewidth=2,
    )

    dashboard = plot(
        p_energy, p_error, p_variance, p_acceptance, p_unique, p_tree,
        p_exact_energy, p_exact_error;
        layout=(4,2), size=(1200,1000),
    )
    display(dashboard)

    figure_dir = joinpath(@__DIR__, "figures")
    mkpath(figure_dir)
    savefig(dashboard, joinpath(figure_dir, "tfim_training_dynamics.png"))

    return (
        result=result,
        validation=validation,
        exact_model=exact_model,
        ground_observables=gs_obs,
        model_observables=model_obs,
        vmc_observables=vmc_obs,
        distribution=dist,
        dashboard=dashboard,
    )
end

end # module TFIMLogVMCExperiment

if abspath(PROGRAM_FILE) == @__FILE__
    TFIMLogVMCExperiment.main()
end
