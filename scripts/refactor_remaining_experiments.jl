# One-time migration helper for the experiment namespace refactor.
#
# Run from the Julia project root with:
#   julia scripts/refactor_remaining_experiments.jl
#
# It is deliberately conservative: it only wraps each legacy experiment in a
# module and replaces a final unconditional `main()` call with a PROGRAM_FILE
# guard. It does NOT alter the experiment body, constants, seeds, or numerical
# logic. Shared-helper extraction can be done separately after regression tests.

const ROOT = normpath(joinpath(@__DIR__, ".."))
const EXPERIMENTS = joinpath(ROOT, "experiments")

const MIGRATIONS = [
    ("exact_tree_capacity.jl", "ExactTreeCapacityExperiment"),
    ("frozen_target_reconstruction.jl", "FrozenTargetReconstructionExperiment"),
    ("initial_target_snr.jl", "InitialTargetSNRExperiment"),
    ("paramagnetic_limit.jl", "ParamagneticLimitExperiment"),
    ("sample_size_scaling.jl", "SampleSizeScalingExperiment"),
    ("sampler_convergence.jl", "SamplerConvergenceExperiment"),
    ("tfim_field_sweep.jl", "TFIMFieldSweepExperiment"),
    ("tfim_log_vmc.jl", "TFIMLogVMCExperiment"),
]

function already_namespaced(text::String, module_name::String)
    occursin("module $module_name", text)
end

function replace_final_main(text::String, module_name::String)
    # All current legacy experiments use an unconditional final main().
    # Match only the last non-whitespace main() so internal calls are untouched.
    m = match(r"(?s)^(.*)\nmain\(\)\s*$", text)
    m === nothing && error("Could not find a final unconditional main() call")
    body = m.captures[1]
    return body * "\n\nend # module $module_name\n\n" *
           "if abspath(PROGRAM_FILE) == @__FILE__\n" *
           "    $module_name.main()\n" *
           "end\n"
end

function migrate_file(filename::String, module_name::String)
    path = joinpath(EXPERIMENTS, filename)
    isfile(path) || error("Missing experiment: $path")
    text = read(path, String)

    if already_namespaced(text, module_name)
        println("SKIP  $filename (already namespaced)")
        return false
    end

    # Keep Pkg activation inside the experiment module. This mirrors the three
    # already-refactored experiments and prevents all subsequent definitions,
    # imports, constants, and helper methods from leaking into Main.
    wrapped = "module $module_name\n\n" * replace_final_main(text, module_name)
    write(path, wrapped)
    println("DONE  $filename -> $module_name")
    return true
end

function main()
    println("Experiment namespace migration")
    println("Project root: ", ROOT)
    println()

    changed = String[]
    for (filename, module_name) in MIGRATIONS
        migrate_file(filename, module_name) && push!(changed, filename)
    end

    println()
    if isempty(changed)
        println("No files changed.")
    else
        println("Migrated $(length(changed)) files:")
        foreach(f -> println("  - ", f), changed)
        println()
        println("Next checks:")
        println("  1. Start a fresh Julia session.")
        println("  2. include every experiments/*.jl file; none should auto-run.")
        println("  3. Run: using Pkg; Pkg.test()")
        println("  4. Inspect: git diff -- experiments")
    end
end

main()
