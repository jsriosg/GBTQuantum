using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module TFIMOldPipelineCouplingDiagnostic

using GBTQuantum
using Statistics
using Printf

# IMPORTANT: this experiment intentionally uses only the original public
# train/exact_ground_energy/exact_model_energy pipeline used by tfim_log_vmc.jl.
# It does not use any of the newer frozen-problem, Armijo, acquisition, or
# independently constructed dense-Hamiltonian helpers.

const N = 8
const RATIOS = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
const NSAMPLES = 512
const EPOCHS = 150
const TOL = 1e-10

function run_ratio(ratio::Float64)
    # Same convention as the previous sweep: h is the energy scale and J/h=ratio.
    H = TFIMHamiltonian(N; J=ratio, h=1.0, periodic=true)

    cfg = TrainingConfig(
        nsamples=NSAMPLES,
        epochs=EPOCHS,
        max_depth=4,
        eta=0.05,
        burn_in_sweeps=50,
        sweeps_per_epoch=2,
        use_phase=false,
        seed=1234,
        exact_diagnostics=true,
        exact_every=5,
    )

    result = train(H, cfg)
    Egs = exact_ground_energy(H)
    exact_model = exact_model_energy(result.model, H)
    Emodel = real(exact_model.energy)

    validation = validation_vmc(
        result.model, H;
        nsamples=10_000,
        burn_in_sweeps=500,
        thinning_sweeps=1,
        seed=9876,
    )
    Evmc = validation.energy

    mask = .!isnan.(result.exact_energy)
    min_diag = any(mask) ? minimum(result.exact_energy[mask]) : NaN
    diag_violation = any(mask) && any(result.exact_energy[mask] .< Egs .- TOL)
    final_violation = Emodel < Egs - TOL

    return (
        ratio=ratio,
        Egs=Egs,
        Emodel=Emodel,
        Evmc=Evmc,
        model_error=Emodel-Egs,
        vmc_minus_model=Evmc-Emodel,
        vmc_se=validation.standard_error,
        min_training_exact=min_diag,
        training_variational_violation=Int(diag_violation),
        final_variational_violation=Int(final_violation),
    )
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "ratio,Egs,Emodel,Evmc,model_error,vmc_minus_model,vmc_se,min_training_exact,training_variational_violation,final_variational_violation")
        for r in rows
            println(io, join((r.ratio,r.Egs,r.Emodel,r.Evmc,r.model_error,r.vmc_minus_model,r.vmc_se,r.min_training_exact,r.training_variational_violation,r.final_variational_violation), ','))
        end
    end
end

function make_plot(rows, outpath)
    try
        @eval using Plots
    catch
        println("Plots.jl unavailable; CSV diagnostic was still written.")
        return
    end
    x = [r.ratio for r in rows]
    p = Plots.plot(x, [r.Egs for r in rows]; marker=:circle, label="Exact ground state", xlabel="J/h", ylabel="Energy", title="Old tfim_log_vmc pipeline: coupling diagnostic")
    Plots.plot!(p, x, [r.Emodel for r in rows]; marker=:circle, label="Exact trained-model energy")
    Plots.plot!(p, x, [r.Evmc for r in rows]; marker=:circle, label="Independent VMC validation")
    Plots.savefig(p, outpath)
end

function main()
    println("\n============================================================")
    println("OLD tfim_log_vmc PIPELINE — COUPLING CONSISTENCY TEST")
    println("N=$N ratios=$RATIOS nsamples=$NSAMPLES epochs=$EPOCHS")
    println("Uses original train() + exact_ground_energy() + exact_model_energy()")
    println("============================================================")

    rows = NamedTuple[]
    for ratio in RATIOS
        println("\nJ/h = $ratio")
        r = run_ratio(ratio)
        push!(rows, r)
        @printf("  E_GS              = % .10f\n", r.Egs)
        @printf("  exact model E     = % .10f\n", r.Emodel)
        @printf("  VMC validation E  = % .10f ± %.3e\n", r.Evmc, r.vmc_se)
        @printf("  E_model - E_GS    = % .6e\n", r.model_error)
        @printf("  E_VMC - E_model   = % .6e\n", r.vmc_minus_model)
        println("  training bound OK = ", r.training_variational_violation == 0)
        println("  final bound OK    = ", r.final_variational_violation == 0)
    end

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    csvpath = joinpath(outdir, "tfim_old_pipeline_coupling_diagnostic.csv")
    pngpath = joinpath(outdir, "tfim_old_pipeline_coupling_diagnostic.png")
    write_csv(csvpath, rows)
    make_plot(rows, pngpath)

    bad = [r for r in rows if r.training_variational_violation != 0 || r.final_variational_violation != 0]
    println("\n============================================================")
    if isempty(bad)
        println("RESULT: no exact variational-bound violation in the old pipeline.")
        println("If the newer sweep violates the bound, the inconsistency is downstream of this pipeline.")
    else
        println("RESULT: variational-bound violation detected in old pipeline at ratios:")
        println([r.ratio for r in bad])
        println("The Hamiltonian/exact-energy implementation itself must then be investigated.")
    end
    println("CSV:  experiments/results/tfim_old_pipeline_coupling_diagnostic.csv")
    println("Plot: experiments/results/tfim_old_pipeline_coupling_diagnostic.png")
    println("============================================================")
end

export main
end

if abspath(PROGRAM_FILE) == @__FILE__
    TFIMOldPipelineCouplingDiagnostic.main()
end
