using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Statistics
using Printf
using Plots

function main()

    # ============================================================
    # Experiment configuration
    # ============================================================

    N = 12
    J = 1.0

    h_values = [0.5, 0.9, 1.5]

    sample_sizes = [
        32,
        64,
        128,
        256,
        512,
        1024,
    ]

    nruns = 5

    nh = length(h_values)
    nM = length(sample_sizes)

    D = 2^N

    println("================================================")
    println("TFIM SAMPLE-SIZE SCALING")
    println("N                  = ", N)
    println("Hilbert dimension  = ", D)
    println("Independent runs   = ", nruns)
    println("================================================")

    # ============================================================
    # Exact references
    # ============================================================

    exact_energy = zeros(nh)
    exact_abs_mz = zeros(nh)
    exact_mx = zeros(nh)

    for (ih, h) in enumerate(h_values)

        H = TFIMHamiltonian(
            N;
            J=J,
            h=h,
            periodic=true,
        )

        gs = exact_ground_observables(H)

        exact_energy[ih] = gs.energy
        exact_abs_mz[ih] = gs.abs_mz
        exact_mx[ih] = gs.mx
    end

    # ============================================================
    # Raw results
    #
    # dimensions:
    #     h × sample-size × independent run
    # ============================================================

    energy_error =
        zeros(nh, nM, nruns)

    abs_mz_error =
        zeros(nh, nM, nruns)

    mx_error =
        zeros(nh, nM, nruns)

    instantaneous_coverage =
        zeros(nh, nM, nruns)

    cumulative_coverage =
        zeros(nh, nM, nruns)

    runtime =
        zeros(nh, nM, nruns)

    success =
    fill(false, nh, nM, nruns)

    final_variance =
        fill(NaN, nh, nM, nruns)

    max_variance =
        fill(NaN, nh, nM, nruns)

    mean_acceptance =
        fill(NaN, nh, nM, nruns)

    final_acceptance =
        fill(NaN, nh, nM, nruns)

    mean_unique_fraction =
        fill(NaN, nh, nM, nruns)

    final_unique_fraction =
        fill(NaN, nh, nM, nruns)

    final_fit_mse =
        fill(NaN, nh, nM, nruns)

    max_fit_mse =
        fill(NaN, nh, nM, nruns)

    # ============================================================
    # Main experiment
    # ============================================================

    for (ih, h) in enumerate(h_values)

        H = TFIMHamiltonian(
            N;
            J=J,
            h=h,
            periodic=true,
        )

        println()
        println("================================================")
        @printf("FIELD h/J = %.2f\n", h / J)
        println("================================================")

        for (iM, M) in enumerate(sample_sizes)

            println()
            @printf(
                "Samples = %d  (nominal Hilbert fraction %.4f)\n",
                M,
                M / D,
            )

            for run in 1:nruns

                seed =
                    1_000_000 * ih +
                    10_000 * iM +
                    run

                cfg = TrainingConfig(
                    nsamples=M,
                    epochs=150,
                    max_depth=4,
                    eta=0.05,
                    burn_in_sweeps=50,
                    sweeps_per_epoch=2,
                    use_phase=false,
                    seed=seed,
                    exact_diagnostics=false,
                )

                result = train(H, cfg)

                # ------------------------------------------------
                # Training diagnostics
                # ------------------------------------------------

                finite_training =
                    all(isfinite, result.energy_history) &&
                    all(isfinite, result.variance_history) &&
                    all(isfinite, result.magnitude_fit_mse)

                success[ih,iM,run] = finite_training

                final_variance[ih,iM,run] =
                    result.variance_history[end]

                max_variance[ih,iM,run] =
                    maximum(result.variance_history)

                mean_acceptance[ih,iM,run] =
                    mean(result.acceptance_history)

                final_acceptance[ih,iM,run] =
                    result.acceptance_history[end]

                mean_unique_fraction[ih,iM,run] =
                    mean(result.unique_fraction_history)

                final_unique_fraction[ih,iM,run] =
                    result.unique_fraction_history[end]

                final_fit_mse[ih,iM,run] =
                    result.magnitude_fit_mse[end]

                max_fit_mse[ih,iM,run] =
                    maximum(result.magnitude_fit_mse)

                # ------------------------------------------------
                # Exact evaluation of learned GBT state
                # ------------------------------------------------

                if finite_training
                    model_energy = exact_model_energy(result.model, H)
                    model_obs = exact_model_observables(result.model, H)

                    energy_error[ih,iM,run] =
                        abs(real(model_energy.energy) - exact_energy[ih])

                    abs_mz_error[ih,iM,run] =
                        abs(model_obs.abs_mz - exact_abs_mz[ih])

                    mx_error[ih,iM,run] =
                        abs(real(model_obs.mx) - exact_mx[ih])
                else
                    energy_error[ih,iM,run] = NaN
                    abs_mz_error[ih,iM,run] = NaN
                    mx_error[ih,iM,run] = NaN
                end

                instantaneous_coverage[ih, iM, run] =
                    result.hilbert_coverage[end]

                cumulative_coverage[ih, iM, run] =
                    result.cumulative_hilbert_coverage[end]

                runtime[ih, iM, run] =
                    result.runtime_seconds

                @printf(
                    "  run %d/%d   ΔE=%9.3e   Δ|mz|=%9.3e   Δmx=%9.3e   cov=%7.4f   cumulative=%7.4f\n",
                    run,
                    nruns,
                    energy_error[ih, iM, run],
                    abs_mz_error[ih, iM, run],
                    mx_error[ih, iM, run],
                    instantaneous_coverage[ih, iM, run],
                    cumulative_coverage[ih, iM, run],
                )
            end
        end
    end

    # ============================================================
    # Statistics over independent trainings
    # ============================================================

    mean_E =
        dropdims(mean(energy_error, dims=3), dims=3)

    std_E =
        dropdims(std(energy_error, dims=3), dims=3)

    mean_abs_mz =
        dropdims(mean(abs_mz_error, dims=3), dims=3)

    std_abs_mz =
        dropdims(std(abs_mz_error, dims=3), dims=3)

    mean_mx =
        dropdims(mean(mx_error, dims=3), dims=3)

    std_mx =
        dropdims(std(mx_error, dims=3), dims=3)

    mean_inst_cov =
        dropdims(
            mean(instantaneous_coverage, dims=3),
            dims=3,
        )

    mean_cum_cov =
        dropdims(
            mean(cumulative_coverage, dims=3),
            dims=3,
        )

    # ============================================================
    # Save CSV
    # ============================================================

    outfile =
        joinpath(
            @__DIR__,
            "sample_size_scaling.csv",
        )

    open(outfile, "w") do io

        println(
            io,
            "h,N,nsamples,run," *
            "energy_error,abs_mz_error,mx_error," *
            "instantaneous_coverage," *
            "cumulative_coverage,runtime"
        )

        for (ih, h) in enumerate(h_values)
            for (iM, M) in enumerate(sample_sizes)
                for run in 1:nruns

                    println(
                        io,
                        join(
                            (
                                h,
                                N,
                                M,
                                run,

                                energy_error[ih,iM,run],
                                abs_mz_error[ih,iM,run],
                                mx_error[ih,iM,run],

                                instantaneous_coverage[ih,iM,run],
                                cumulative_coverage[ih,iM,run],

                                runtime[ih,iM,run],
                            ),
                            ",",
                        )
                    )
                end
            end
        end
    end

    # ============================================================
    # Plot 1 — energy accuracy
    # ============================================================

    pE = plot(
        xlabel="Training population M",
        ylabel="|E_GBT - E0|",
        title="Energy accuracy",
        xscale=:log10,
        yscale=:log10,
    )

    for (ih, h) in enumerate(h_values)

        plot!(
            pE,
            sample_sizes,
            mean_E[ih,:];
            yerror=std_E[ih,:],
            marker=:circle,
            linewidth=2,
            label="h/J=$(h)",
        )
    end

    # ============================================================
    # Plot 2 — longitudinal magnetization
    # ============================================================

    pMz = plot(
        xlabel="Training population M",
        ylabel="Δ<|m_z|>",
        title="Longitudinal observable error",
        xscale=:log10,
        yscale=:log10,
    )

    for (ih, h) in enumerate(h_values)

        plot!(
            pMz,
            sample_sizes,
            mean_abs_mz[ih,:];
            yerror=std_abs_mz[ih,:],
            marker=:circle,
            linewidth=2,
            label="h/J=$(h)",
        )
    end

    # ============================================================
    # Plot 3 — transverse magnetization
    # ============================================================

    pMx = plot(
        xlabel="Training population M",
        ylabel="Δ<m_x>",
        title="Transverse observable error",
        xscale=:log10,
        yscale=:log10,
    )

    for (ih, h) in enumerate(h_values)

        plot!(
            pMx,
            sample_sizes,
            mean_mx[ih,:];
            yerror=std_mx[ih,:],
            marker=:circle,
            linewidth=2,
            label="h/J=$(h)",
        )
    end

    # ============================================================
    # Plot 4 — coverage
    # ============================================================

    pCov = plot(
        xlabel="Training population M",
        ylabel="Fraction of Hilbert space",
        title="Hilbert-space exposure",
        xscale=:log10,
        yscale=:log10,
    )

    for (ih, h) in enumerate(h_values)

        plot!(
            pCov,
            sample_sizes,
            mean_inst_cov[ih,:];
            marker=:circle,
            linestyle=:dash,
            label="Current h/J=$(h)",
        )

        plot!(
            pCov,
            sample_sizes,
            mean_cum_cov[ih,:];
            marker=:square,
            linewidth=2,
            label="Cumulative h/J=$(h)",
        )
    end

    dashboard = plot(
        pE,
        pMz,
        pMx,
        pCov;
        layout=(2,2),
        size=(1250,900),
    )

    display(dashboard)

    savefig(
        dashboard,
        joinpath(
            @__DIR__,
            "sample_size_scaling.png",
        ),
    )

    # ============================================================
    # Numerical summary
    # ============================================================

    println()
    println()
    println("============== SAMPLE-SIZE SUMMARY ==============")

    for (ih, h) in enumerate(h_values)

        println()
        @printf("h/J = %.2f\n", h)
        println()

        println(
            " M       mean ΔE       mean Δ|mz|    mean Δmx      current cov    cumulative cov"
        )

        println(
            "--------------------------------------------------------------------------"
        )

        for (iM, M) in enumerate(sample_sizes)

            @printf(
                "%4d   %11.4e   %11.4e   %11.4e   %10.4f     %10.4f\n",
                M,
                mean_E[ih,iM],
                mean_abs_mz[ih,iM],
                mean_mx[ih,iM],
                mean_inst_cov[ih,iM],
                mean_cum_cov[ih,iM],
            )
        end
    end

    println()
    println("Results written to:")
    println(outfile)
    println("=================================================")

end

main()