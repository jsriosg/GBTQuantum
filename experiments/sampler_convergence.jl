using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using Statistics
using Printf
using Random

# ============================================================
# Configuration
# ============================================================

N = 8
J = 1.0
h = 1.0

H = TFIMHamiltonian(
    N;
    J = J,
    h = h,
    periodic = true,
)

# ------------------------------------------------------------
# Use the SAME training configuration as the main N=8
# experiment so we test the same type of learned state.
# Replace these values if your tfim_log_vmc.jl currently uses
# different parameters.
# ------------------------------------------------------------

cfg = TrainingConfig(
    nsamples = 512,
    epochs = 100,
    max_depth = 3,
    eta = 0.05,
    burn_in_sweeps = 20,
    sweeps_per_epoch = 5,
    use_phase = false,
)

println("Training reference model...")
result = train(H, cfg)

model = result.model

# ============================================================
# Convergence experiment
# ============================================================

sample_sizes = [
    10_000,
    100_000,
    1_000_000,
]

nrepeats = 10

tv_results =
    zeros(Float64, length(sample_sizes), nrepeats)

mag_tv_results =
    zeros(Float64, length(sample_sizes), nrepeats)

max_error_results =
    zeros(Float64, length(sample_sizes), nrepeats)

# Independent deterministic seeds.
base_seed = 100_000

println()
println("Running sampler convergence experiment...")
println()

for (i, M) in enumerate(sample_sizes)

    println("Samples = ", M)

    for r in 1:nrepeats

        seed =
            base_seed +
            10_000 * i +
            r

        dist = validation_distribution(
            model,
            H;
            nsamples = M,
            burn_in_sweeps = 500,
            thinning_sweeps = 1,
            seed = seed,
        )

        tv_results[i, r] =
            dist.total_variation

        mag_tv_results[i, r] =
            dist.magnetization_tv

        max_error_results[i, r] =
            dist.max_abs_error

        @printf(
            "  run %2d/%2d   TV = %.6f   mag-TV = %.6f   max-error = %.6f\n",
            r,
            nrepeats,
            dist.total_variation,
            dist.magnetization_tv,
            dist.max_abs_error,
        )
    end

    println()
end

# ============================================================
# Summary
# ============================================================

println()
println("============== CONVERGENCE SUMMARY ==============")
println()
println(
    " Samples       mean TV      std TV     " *
    "mean mag-TV   std mag-TV   TV*sqrt(M)"
)
println(
    "---------------------------------------------------------------"
)

for (i, M) in enumerate(sample_sizes)

    mean_tv =
        mean(@view tv_results[i, :])

    std_tv =
        std(@view tv_results[i, :])

    mean_mag =
        mean(@view mag_tv_results[i, :])

    std_mag =
        std(@view mag_tv_results[i, :])

    scaled_tv =
        mean_tv * sqrt(M)

    @printf(
        "%8d     %.6f     %.6f     %.6f     %.6f     %.4f\n",
        M,
        mean_tv,
        std_tv,
        mean_mag,
        std_mag,
        scaled_tv,
    )
end

println()
println("==================================================")

# ============================================================
# Scaling exponent
#
# If
#
#       TV ~ M^alpha
#
# independent Monte Carlo scaling predicts approximately
#
#       alpha = -1/2.
#
# Fit log(TV) = alpha log(M) + constant.
# ============================================================

mean_tv =
    vec(mean(tv_results; dims = 2))

mean_mag_tv =
    vec(mean(mag_tv_results; dims = 2))

logM =
    log.(Float64.(sample_sizes))

logTV =
    log.(mean_tv)

logMagTV =
    log.(mean_mag_tv)

α_tv =
    sum(
        (logM .- mean(logM)) .*
        (logTV .- mean(logTV))
    ) /
    sum(
        (logM .- mean(logM)).^2
    )

α_mag =
    sum(
        (logM .- mean(logM)) .*
        (logMagTV .- mean(logMagTV))
    ) /
    sum(
        (logM .- mean(logM)).^2
    )

println()
println("============= SCALING EXPONENTS =============")
println()

@printf(
    "State-distribution TV exponent   = %.4f\n",
    α_tv,
)

@printf(
    "Magnetization TV exponent        = %.4f\n",
    α_mag,
)

println()
println("Reference Monte Carlo exponent    = -0.5000")
println()
println("=============================================")