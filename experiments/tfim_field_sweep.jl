module TFIMFieldSweepExperiment

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Statistics
using Printf
using DelimitedFiles
using Plots

function main()

    # ============================================================
    # Experiment configuration
    # ============================================================

    J = 1.0

    system_sizes = [8, 10, 12]

    h_values = [
        0.25,
        0.50,
        0.75,
        0.90,
        1.00,
        1.10,
        1.25,
        1.50,
        2.00,
    ]

    nN = length(system_sizes)
    nh = length(h_values)

    # ============================================================
    # Result arrays
    # ============================================================

    E_gbt = fill(NaN, nN, nh)
    E_exact = fill(NaN, nN, nh)

    abs_mz_gbt = fill(NaN, nN, nh)
    abs_mz_exact = fill(NaN, nN, nh)

    mz2_gbt = fill(NaN, nN, nh)
    mz2_exact = fill(NaN, nN, nh)

    mx_gbt = fill(NaN, nN, nh)
    mx_exact = fill(NaN, nN, nh)

    coverage = fill(NaN, nN, nh)
    unique_states = fill(NaN, nN, nh)

    runtime = fill(NaN, nN, nh)

    # ============================================================
    # Sweep
    # ============================================================

    for (iN, N) in enumerate(system_sizes)

        println()
        println("================================================")
        println("SYSTEM SIZE N = ", N)
        println("Hilbert dimension = ", 2^N)
        println("================================================")

        for (ih, h) in enumerate(h_values)

            println()
            @printf(
                "---- N = %d, h/J = %.2f ----\n",
                N,
                h / J,
            )

            H = TFIMHamiltonian(
                N;
                J = J,
                h = h,
                periodic = true,
            )

            # ----------------------------------------------------
            # Keep the original v0.2 hyperparameters fixed.
            # ----------------------------------------------------

            cfg = TrainingConfig(
                nsamples = 512,
                epochs = 150,
                max_depth = 4,
                eta = 0.05,
                burn_in_sweeps = 50,
                sweeps_per_epoch = 2,
                use_phase = false,

                # Different but deterministic seed for every point.
                seed = 10_000 * N + ih,

                # We don't need exact diagnostics during every epoch.
                # Exact validation is performed after training.
                exact_diagnostics = false,
            )

            result = train(H, cfg)

            # ====================================================
            # Exact ground state
            # ====================================================

            gs = exact_ground_observables(H)

            E_exact[iN, ih] = gs.energy

            abs_mz_exact[iN, ih] = gs.abs_mz
            mz2_exact[iN, ih] = gs.mz2
            mx_exact[iN, ih] = gs.mx

            # ====================================================
            # Exact observables of learned GBT
            # ====================================================

            model_energy =
                exact_model_energy(result.model, H)

            model_obs =
                exact_model_observables(result.model, H)

            E_gbt[iN, ih] =
                real(model_energy.energy)

            abs_mz_gbt[iN, ih] =
                model_obs.abs_mz

            mz2_gbt[iN, ih] =
                model_obs.mz2

            mx_gbt[iN, ih] =
                real(model_obs.mx)

            # ====================================================
            # Sampling / coverage diagnostics
            # ====================================================

            coverage[iN, ih] =
                result.hilbert_coverage[end]

            unique_states[iN, ih] =
                coverage[iN, ih] * (2^N)

            runtime[iN, ih] =
                result.runtime_seconds

            # ====================================================
            # Point summary
            # ====================================================

            @printf(
                "E0/N             = %.8f\n",
                gs.energy / N,
            )

            @printf(
                "E_GBT/N          = %.8f\n",
                E_gbt[iN, ih] / N,
            )

            @printf(
                "|ΔE|             = %.6e\n",
                abs(E_gbt[iN, ih] - gs.energy),
            )

            @printf(
                "<|mz|> exact/GBT = %.6f / %.6f\n",
                gs.abs_mz,
                model_obs.abs_mz,
            )

            @printf(
                "<mz²> exact/GBT  = %.6f / %.6f\n",
                gs.mz2,
                model_obs.mz2,
            )

            @printf(
                "<mx> exact/GBT   = %.6f / %.6f\n",
                gs.mx,
                real(model_obs.mx),
            )

            @printf(
                "Hilbert coverage = %.6f\n",
                coverage[iN, ih],
            )

            @printf(
                "Approx unique    = %.0f / %d\n",
                unique_states[iN, ih],
                2^N,
            )

            @printf(
                "Runtime          = %.3f s\n",
                runtime[iN, ih],
            )
        end
    end

    # ============================================================
    # Error matrices
    # ============================================================

    energy_error =
        abs.(E_gbt .- E_exact)

    abs_mz_error =
        abs.(abs_mz_gbt .- abs_mz_exact)

    mz2_error =
        abs.(mz2_gbt .- mz2_exact)

    mx_error =
        abs.(mx_gbt .- mx_exact)

    # ============================================================
    # Save numerical results
    #
    # One row per (N,h).
    # ============================================================

    outfile =
        joinpath(@__DIR__, "tfim_field_sweep.csv")

    open(outfile, "w") do io

        println(
            io,
            "N,h,J,hilbert_dim,coverage,unique_states," *
            "E_exact,E_gbt,E_error," *
            "abs_mz_exact,abs_mz_gbt,abs_mz_error," *
            "mz2_exact,mz2_gbt,mz2_error," *
            "mx_exact,mx_gbt,mx_error,runtime"
        )

        for (iN, N) in enumerate(system_sizes)
            for (ih, h) in enumerate(h_values)

                println(
                    io,
                    join(
                        (
                            N,
                            h,
                            J,
                            2^N,
                            coverage[iN, ih],
                            unique_states[iN, ih],

                            E_exact[iN, ih],
                            E_gbt[iN, ih],
                            energy_error[iN, ih],

                            abs_mz_exact[iN, ih],
                            abs_mz_gbt[iN, ih],
                            abs_mz_error[iN, ih],

                            mz2_exact[iN, ih],
                            mz2_gbt[iN, ih],
                            mz2_error[iN, ih],

                            mx_exact[iN, ih],
                            mx_gbt[iN, ih],
                            mx_error[iN, ih],

                            runtime[iN, ih],
                        ),
                        ",",
                    )
                )
            end
        end
    end

    # ============================================================
    # Plot 1 — longitudinal order
    # ============================================================

    p_abs_mz = plot(
        xlabel = "h/J",
        ylabel = "<|m_z|>",
        title = "Longitudinal magnetic order",
        ylim = (0, 1),
    )

    for (iN, N) in enumerate(system_sizes)

        plot!(
            p_abs_mz,
            h_values,
            abs_mz_exact[iN, :];
            label = "Exact N=$N",
            linewidth = 2,
        )

        scatter!(
            p_abs_mz,
            h_values,
            abs_mz_gbt[iN, :];
            label = "GBT N=$N",
            markersize = 4,
        )
    end

    # ============================================================
    # Plot 2 — transverse magnetization
    # ============================================================

    p_mx = plot(
        xlabel = "h/J",
        ylabel = "<m_x>",
        title = "Transverse magnetization",
        ylim = (0, 1),
    )

    for (iN, N) in enumerate(system_sizes)

        plot!(
            p_mx,
            h_values,
            mx_exact[iN, :];
            label = "Exact N=$N",
            linewidth = 2,
        )

        scatter!(
            p_mx,
            h_values,
            mx_gbt[iN, :];
            label = "GBT N=$N",
            markersize = 4,
        )
    end

    # ============================================================
    # Plot 3 — energy error
    # ============================================================

    p_energy_error = plot(
        xlabel = "h/J",
        ylabel = "|E_GBT - E0|",
        title = "Variational energy error",
        yscale = :log10,
    )

    for (iN, N) in enumerate(system_sizes)

        plot!(
            p_energy_error,
            h_values,
            energy_error[iN, :];
            marker = :circle,
            label = "N=$N",
        )
    end

    # ============================================================
    # Plot 4 — Hilbert-space coverage
    # ============================================================

    p_coverage = plot(
        xlabel = "h/J",
        ylabel = "Unique sampled / dim(H)",
        title = "Final Hilbert-space coverage",
        yscale = :log10,
    )

    for (iN, N) in enumerate(system_sizes)

        plot!(
            p_coverage,
            h_values,
            coverage[iN, :];
            marker = :circle,
            label = "N=$N",
        )
    end

    # ============================================================
    # Dashboard
    # ============================================================

    dashboard = plot(
        p_abs_mz,
        p_mx,
        p_energy_error,
        p_coverage;
        layout = (2, 2),
        size = (1200, 900),
    )

    display(dashboard)

    savefig(
        dashboard,
        joinpath(
            @__DIR__,
            "tfim_field_sweep.png",
        ),
    )

    # ============================================================
    # Final summary
    # ============================================================

    println()
    println("============== SWEEP COMPLETE ==============")
    println("Results: ", outfile)

    for (iN, N) in enumerate(system_sizes)

        println()
        println("N = ", N)
        println("Hilbert dimension = ", 2^N)

        @printf(
            "Mean energy error    = %.6e\n",
            mean(energy_error[iN, :]),
        )

        @printf(
            "Max energy error     = %.6e\n",
            maximum(energy_error[iN, :]),
        )

        @printf(
            "Mean |mz| error      = %.6e\n",
            mean(abs_mz_error[iN, :]),
        )

        @printf(
            "Mean mz² error       = %.6e\n",
            mean(mz2_error[iN, :]),
        )

        @printf(
            "Mean mx error        = %.6e\n",
            mean(mx_error[iN, :]),
        )

        @printf(
            "Mean final coverage  = %.6f\n",
            mean(coverage[iN, :]),
        )
    end

    println()
    println("============================================")
end


end # module TFIMFieldSweepExperiment

if abspath(PROGRAM_FILE) == @__FILE__
    TFIMFieldSweepExperiment.main()
end
