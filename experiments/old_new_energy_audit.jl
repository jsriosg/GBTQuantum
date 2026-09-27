using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module OldNewEnergyAudit

using GBTQuantum
using Random
using Statistics
using Printf

# Reuse the exact helper functions that the NEW experimental path actually uses.
include(joinpath(@__DIR__, "newton_step_validation.jl"))
const NV = NewtonStepValidationExperiment

const N = 8
const RATIOS = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
const ETA_TEST = 0.05
const TOL = 1e-10

maxabs(v) = maximum(abs.(v))
rms(v) = sqrt(mean(abs2, v))

function old_exact_vectors(H, model, X)
    p = GBTQuantum.exact_probabilities(model, X)
    loga = Float64[GBTQuantum.logamplitude(model, @view(X[i,:])) for i in axes(X,1)]
    eloc = Float64[real(GBTQuantum.local_energy!(H, model, @view(X[i,:]))) for i in axes(X,1)]
    E = sum(p .* eloc)
    y = -(eloc .- E)
    return (p=p, loga=loga, eloc=eloc, E=E, y=y)
end

function audit_state(label, H, model, X, flips; f=nothing)
    old = old_exact_vectors(H, model, X)
    new = NV.exact_frozen_problem(H, model, X)
    exact_model = real(GBTQuantum.exact_model_energy(model, H).energy)
    Egs = GBTQuantum.exact_ground_energy(H)

    @printf("\n  [%s]\n", label)
    @printf("    E_GS                         = % .12f\n", Egs)
    @printf("    old sum(p E_loc)             = % .12f\n", old.E)
    @printf("    GBTQuantum exact_model_energy= % .12f\n", exact_model)
    @printf("    NEW frozen energy            = % .12f\n", new.energy)
    @printf("    |old-exact_model|            = %.3e\n", abs(old.E-exact_model))
    @printf("    |new-old|                    = %.3e\n", abs(new.energy-old.E))
    @printf("    max |p_new-p_old|            = %.3e\n", maxabs(new.probabilities-old.p))
    @printf("    max |logamp_new-logamp_old|  = %.3e\n", maxabs(new.logamp-old.loga))
    @printf("    max |Eloc_new-Eloc_old|      = %.3e\n", maxabs(new.eloc-old.eloc))
    @printf("    max |target_new-target_old|  = %.3e\n", maxabs(new.target-old.y))
    @printf("    variational margins old/new  = %.3e / %.3e\n", old.E-Egs, new.energy-Egs)

    if f !== nothing
        # This is the critical audit: compare the NEW hand-written eta energy
        # against the project's canonical exact energy after applying exactly
        # the same additive log-amplitude direction to a copy of the model.
        Enew_formula = NV.exact_energy_eta(new, X, flips, f, ETA_TEST)
        # f is supplied from an actual tree below, so the updated model can be
        # represented exactly by scaling that tree.
        @printf("    NEW exact_energy_eta(eta=.05) = % .12f\n", Enew_formula)
    end

    return (old=old,new=new,Egs=Egs,exact_model=exact_model)
end

function first_tree_audit(ratio, H, X, flips)
    rng = MersenneTwister(7_700_000 + round(Int,1000ratio))
    model = GBTQuantum.LogGBState(logamp_bias=0.0, phase_bias=0.0, use_phase=false)

    # Epoch 0: absolutely no MC or tree involved.
    a0 = audit_state("epoch 0 / uniform model", H, model, X, flips)

    # Build ONE deterministic full-Hilbert tree from the exact target. This
    # eliminates sampling as a possible explanation for any disagreement.
    tree = GBTQuantum.grow_tree(X, a0.new.target, a0.new.probabilities;
        max_depth=4, min_weight=1e-14, min_gain=0.0)
    f = NV.centered_predictions(tree, X, a0.new.probabilities)

    # Centering is only a constant gauge. To compare to the actual stored tree,
    # use its uncentered prediction and verify the two eta energies coincide.
    raw = NV.predict_all(tree, X)
    mu = sum(a0.new.probabilities .* raw)
    centered = raw .- mu
    @printf("\n  [first exact full-Hilbert tree]\n")
    @printf("    prediction mean <f>_p        = % .6e\n", sum(a0.new.probabilities .* centered))
    @printf("    max |centered - f|           = %.3e\n", maxabs(centered-f))

    E_formula_centered = NV.exact_energy_eta(a0.new, X, flips, centered, ETA_TEST)
    E_formula_raw = NV.exact_energy_eta(a0.new, X, flips, raw, ETA_TEST)

    model1 = deepcopy(model)
    push!(model1.logamp.trees, NV.scale_tree(tree, ETA_TEST))
    E_canonical_after = real(GBTQuantum.exact_model_energy(model1, H).energy)
    a1 = old_exact_vectors(H, model1, X)

    @printf("    NEW eta energy (centered f)   = % .12f\n", E_formula_centered)
    @printf("    NEW eta energy (raw tree)     = % .12f\n", E_formula_raw)
    @printf("    canonical energy after tree   = % .12f\n", E_canonical_after)
    @printf("    sum(p E_loc) after tree       = % .12f\n", a1.E)
    @printf("    |centered-raw energy|         = %.3e\n", abs(E_formula_centered-E_formula_raw))
    @printf("    |NEW eta-canonical|           = %.3e\n", abs(E_formula_raw-E_canonical_after))
    @printf("    variational margin NEW eta    = %.3e\n", E_formula_raw-a0.Egs)
    @printf("    variational margin canonical  = %.3e\n", E_canonical_after-a0.Egs)

    return (
        ratio=ratio,
        epoch0_old=a0.old.E,
        epoch0_new=a0.new.energy,
        epoch0_canonical=a0.exact_model,
        Egs=a0.Egs,
        max_p_diff=maxabs(a0.new.probabilities-a0.old.p),
        max_eloc_diff=maxabs(a0.new.eloc-a0.old.eloc),
        max_target_diff=maxabs(a0.new.target-a0.old.y),
        eta_energy_centered=E_formula_centered,
        eta_energy_raw=E_formula_raw,
        canonical_after_tree=E_canonical_after,
        local_after_tree=a1.E,
        eta_vs_canonical=E_formula_raw-E_canonical_after,
        eta_variational_violation=Int(E_formula_raw < a0.Egs-TOL),
        canonical_variational_violation=Int(E_canonical_after < a0.Egs-TOL),
    )
end

function write_csv(path, rows)
    names = propertynames(rows[1])
    open(path,"w") do io
        println(io, join(string.(names), ','))
        for r in rows
            println(io, join((getproperty(r,n) for n in names), ','))
        end
    end
end

function main()
    println("\n============================================================")
    println("DETERMINISTIC OLD-vs-NEW TFIM ENERGY AUDIT")
    println("N=$N ratios=$RATIOS eta_test=$ETA_TEST")
    println("No Monte Carlo. No Armijo. Full Hilbert enumeration only.")
    println("============================================================")

    X = NV.enumerate_states(N)
    flips = NV.build_flip_index(X)
    rows = NamedTuple[]

    for ratio in RATIOS
        println("\n============================================================")
        @printf("J/h = %.3f\n", ratio)
        H = GBTQuantum.TFIMHamiltonian(N; J=ratio, h=1.0, periodic=true)
        push!(rows, first_tree_audit(ratio,H,X,flips))
    end

    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    path=joinpath(outdir,"old_new_energy_audit.csv")
    write_csv(path,rows)

    println("\n============================================================")
    println("AUDIT SUMMARY")
    println("============================================================")
    for r in rows
        @printf("J/h=%4.2f | epoch0 new-old=% .2e | eta-canonical=% .3e | violation(new/canonical)=%d/%d\n",
            r.ratio,r.epoch0_new-r.epoch0_old,r.eta_vs_canonical,
            r.eta_variational_violation,r.canonical_variational_violation)
    end
    println("\nInterpretation:")
    println("  1) If epoch0_new-old != 0, exact_frozen_problem is wrong.")
    println("  2) If epoch0 agrees but eta-canonical != 0, exact_energy_eta is wrong.")
    println("  3) If both agree, the bug is later (tree application / stochastic Armijo path).")
    println("\nResults written to experiments/results/old_new_energy_audit.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    OldNewEnergyAudit.main()
end
