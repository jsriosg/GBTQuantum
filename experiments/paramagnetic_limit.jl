using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Statistics
using Printf
using Plots

function finite_mean_std(v)
    w = filter(isfinite, v)
    isempty(w) && return (NaN, NaN)

    μ = mean(w)
    σ = length(w) > 1 ? std(w) : 0.0

    return μ, σ
end


# ============================================================
# Exact enumeration of the learned log-amplitude
# ============================================================

function exact_logamp_statistics(model::LogGBState, N::Int)

    D = 1 << N
    x = Vector{Int8}(undef, N)
    A = Vector{Float64}(undef, D)

    @inbounds for s in 0:(D - 1)

        for i in 1:N
            x[i] =
                ((s >> (i - 1)) & 1) == 1 ?
                Int8(1) :
                Int8(-1)
        end

        A[s + 1] = logamplitude(model, x)
    end

    return (
        mean = mean(A),
        variance = var(A; corrected=false),
        std = std(A; corrected=false),
        range = maximum(A) - minimum(A),
        minimum = minimum(A),
        maximum = maximum(A),
    )
end


function main()

    # ========================================================
    # Configuration
    # ========================================================

    N = 12
    J = 0.0
    h = 1.0

    sample_sizes = [
        32,
        64,
        128,
        256,
        512,
        1024,
    ]

    nruns = 5

    nM = length(sample_sizes)
    D = 1 << N

    H = TFIMHamiltonian(
        N;
        J=J,
        h=h,
        periodic=true,
    )

    # ========================================================
    # Exact references
    # ========================================================

    Eexact = -N * h

    # We can also use exact diagonalization/observables as an
    # independent numerical check.
    gs = exact_ground_observables(H)

    println("================================================")
    println("PARAMAGNETIC-LIMIT CONTROL")
    println("N                  = ", N)
    println("J                  = ", J)
    println("h                  = ", h)
    println("Hilbert dimension  = ", D)
    println("Independent runs   = ", nruns)
    println("Analytic E0        = ", Eexact)
    println("Numerical E0       = ", gs.energy)
    println("Exact <|mz|>       = ", gs.abs_mz)
    println("Exact <mx>         = ", gs.mx)
    println("================================================")

    # ========================================================
    # Storage
    # ========================================================

    energy_error =
        fill(NaN, nM, nruns)

    abs_mz_error =
        fill(NaN, nM, nruns)

    mx_error =
        fill(NaN, nM, nruns)

    exact_model_variance = 
        fill(NaN, nM, nruns)

    logamp_variance =
        fill(NaN, nM, nruns)

    logamp_std =
        fill(NaN, nM, nruns)

    logamp_range =
        fill(NaN, nM, nruns)

    instantaneous_coverage =
        fill(NaN, nM, nruns)

    cumulative_coverage =
        fill(NaN, nM, nruns)

    final_variance =
        fill(NaN, nM, nruns)

    final_fit_mse =
        fill(NaN, nM, nruns)

    mean_acceptance =
        fill(NaN, nM, nruns)

    final_unique_fraction =
        fill(NaN, nM, nruns)

    runtime =
        fill(NaN, nM, nruns)

    success =
        fill(false, nM, nruns)

    # ========================================================
    # Training
    # ========================================================

    for (iM, M) in enumerate(sample_sizes)

        println()
        @printf(
            "Samples = %d  (nominal Hilbert fraction %.4f)\n",
            M,
            M / D,
        )

        for run in 1:nruns

            seed =
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

            model = result.model

            # ------------------------------------------------
            # Exact model observables
            # ------------------------------------------------

            obs =
                exact_model_observables(
                    model,
                    H,
                )

            model_energy =
                exact_model_energy(
                    model,
                    H,
                )

            # ------------------------------------------------
            # Full-Hilbert log-amplitude statistics
            # ------------------------------------------------

            Astat =
                exact_logamp_statistics(
                    model,
                    N,
                )

            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            energy_error[iM, run] =
                abs(model_energy.energy - Eexact)

            abs_mz_error[iM, run] =
                abs(obs.abs_mz - gs.abs_mz)

            mx_error[iM, run] =
                abs(real(obs.mx) - real(gs.mx))

            exact_model_variance[iM, run] = 
                model_energy.variance

            logamp_variance[iM, run] =
                Astat.variance

            logamp_std[iM, run] =
                Astat.std

            logamp_range[iM, run] =
                Astat.range

            instantaneous_coverage[iM, run] =
                result.hilbert_coverage[end]

            cumulative_coverage[iM, run] =
                result.cumulative_hilbert_coverage[end]

            final_variance[iM, run] =
                result.variance_history[end]

            final_fit_mse[iM, run] =
                result.magnitude_fit_mse[end]

            mean_acceptance[iM, run] =
                mean(result.acceptance_history)

            final_unique_fraction[iM, run] =
                result.unique_fraction_history[end]

            runtime[iM, run] =
                result.runtime_seconds

            success[iM, run] =
                all(isfinite, (
                    energy_error[iM, run],
                    abs_mz_error[iM, run],
                    mx_error[iM, run],
                    logamp_variance[iM, run],
                ))

            @printf(
                "  run %d/%d   ΔE=%9.3e   Var[A]=%9.3e   range[A]=%9.3e   cov=%7.4f   cumulative=%7.4f\n",
                run,
                nruns,
                energy_error[iM, run],
                logamp_variance[iM, run],
                logamp_range[iM, run],
                instantaneous_coverage[iM, run],
                cumulative_coverage[iM, run],
            )
        end
    end

    # ========================================================
    # Summary
    # ========================================================

    println()
    println("============== PARAMAGNETIC CONTROL SUMMARY ==============")
    println()
    println(
        " M       mean ΔE       mean Var[A]    mean range[A]  " *
        "mean Δ|mz|    mean Δmx      success"
    )
    println(
        "--------------------------------------------------------------------------"
    )

    mean_energy = zeros(nM)
    mean_Avar = zeros(nM)
    mean_Arange = zeros(nM)

    for iM in 1:nM

        μE, _ =
            finite_mean_std(
                @view energy_error[iM, :]
            )

        μAvar, _ =
            finite_mean_std(
                @view logamp_variance[iM, :]
            )

        μArange, _ =
            finite_mean_std(
                @view logamp_range[iM, :]
            )

        μmz, _ =
            finite_mean_std(
                @view abs_mz_error[iM, :]
            )

        μmx, _ =
            finite_mean_std(
                @view mx_error[iM, :]
            )

        success_rate =
            count(@view success[iM, :]) /
            nruns

        mean_energy[iM] = μE
        mean_Avar[iM] = μAvar
        mean_Arange[iM] = μArange

        @printf(
            "%4d   %12.4e   %12.4e   %12.4e   %12.4e   %12.4e   %6.0f%%\n",
            sample_sizes[iM],
            μE,
            μAvar,
            μArange,
            μmz,
            μmx,
            100 * success_rate,
        )
    end

    println()
    println("Exact target:")
    println("    Var[log|ψ|]   = 0")
    println("    range[log|ψ|] = 0")
    println("    <mx>           = 1")
    println("    E0             = ", Eexact)
    println("===========================================================")

    # ========================================================
    # CSV
    # ========================================================

    outfile =
        joinpath(
            @__DIR__,
            "paramagnetic_limit.csv",
        )

    open(outfile, "w") do io

        println(
            io,
            "N,J,h,nsamples,run,success," *
            "energy_error,abs_mz_error,mx_error," *
            "logamp_variance,logamp_std,logamp_range," *
            "instantaneous_coverage,cumulative_coverage," *
            "final_variance,final_fit_mse," *
            "mean_acceptance,final_unique_fraction,runtime," *
            "exact_model_variance"
        )

        for iM in 1:nM
            for run in 1:nruns

                values = (
                    N,
                    J,
                    h,
                    sample_sizes[iM],
                    run,
                    success[iM, run],

                    energy_error[iM, run],
                    abs_mz_error[iM, run],
                    mx_error[iM, run],

                    logamp_variance[iM, run],
                    logamp_std[iM, run],
                    logamp_range[iM, run],

                    instantaneous_coverage[iM, run],
                    cumulative_coverage[iM, run],

                    final_variance[iM, run],
                    final_fit_mse[iM, run],

                    mean_acceptance[iM, run],
                    final_unique_fraction[iM, run],

                    runtime[iM, run],

                    exact_model_variance[iM, run]
                )

                println(
                    io,
                    join(values, ","),
                )
            end
        end
    end

    println()
    println("Saved:")
    println(outfile)

    # ========================================================
    # Plot
    # ========================================================

    p = plot(
        sample_sizes,
        mean_Avar;
        xscale=:log10,
        yscale=:log10,
        marker=:circle,
        xlabel="Training sample size M",
        ylabel="Var[log|ψ|]",
        title="J = 0 paramagnetic control",
        legend=false,
    )

    savefig(
        p,
        joinpath(
            @__DIR__,
            "paramagnetic_limit_logamp_variance.png",
        ),
    )

    println(
        joinpath(
            @__DIR__,
            "paramagnetic_limit_logamp_variance.png",
        )
    )

    return nothing
end


main()