using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module OldNewEnergyAudit

using GBTQuantum
using Statistics
using Printf

# Use the shared experiment helpers explicitly. exact_probabilities, predict_all,
# scale_tree, and enumerate_states live in ExperimentUtils; they are not exports
# of the GBTQuantum package itself.
include(joinpath(@__DIR__, "ExperimentUtils.jl"))
using .ExperimentUtils

const N = 8
const RATIOS = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
const ETA_TEST = 0.05
const TOL = 1e-10

maxabs(v) = maximum(abs.(v))

function build_flip_index_local(X)
    nstates, nspins = size(X)
    index = Dict{Tuple{Vararg{Int8}},Int}()
    for i in 1:nstates
        index[Tuple(@view X[i,:])] = i
    end
    flips = Matrix{Int}(undef, nstates, nspins)
    for i in 1:nstates, j in 1:nspins
        x = collect(@view X[i,:])
        x[j] = -x[j]
        flips[i,j] = index[Tuple(x)]
    end
    flips
end

function exact_energy_eta_local(fr, X, flips, f, eta, J, h)
    nspins = size(X,2)
    z = fr.logamp .+ eta .* f
    z .-= maximum(z)
    a = exp.(z)
    den = sum(abs2, a)
    num = 0.0
    for i in axes(X,1)
        diag = 0.0
        for j in 1:nspins
            jp = (j == nspins ? 1 : j+1)
            diag += -J * Float64(X[i,j]) * Float64(X[i,jp])
        end
        num += diag * a[i]^2
        for j in 1:nspins
            num += -h * a[i] * a[flips[i,j]]
        end
    end
    num / den
end

function old_exact_vectors(H, model, X)
    p = exact_probabilities(model, X)
    loga = Float64[GBTQuantum.logamplitude(model, @view(X[i,:])) for i in axes(X,1)]
    eloc = Float64[real(GBTQuantum.local_energy!(H, model, @view(X[i,:]))) for i in axes(X,1)]
    E = sum(p .* eloc)
    y = -(eloc .- E)
    return (p=p, loga=loga, eloc=eloc, E=E, y=y)
end

function exact_frozen_problem_local(H, model, X)
    p = exact_probabilities(model, X)
    loga = Float64[GBTQuantum.logamplitude(model, @view(X[i,:])) for i in axes(X,1)]
    eloc = ComplexF64[GBTQuantum.local_energy!(H, model, @view(X[i,:])) for i in axes(X,1)]
    E = real(sum(p .* eloc))
    y = -real.(eloc .- E)
    return (probabilities=p, logamp=loga, eloc=real.(eloc), target=y,
            energy=E, target_rms=sqrt(sum(p .* y.^2)))
end

function audit_state(label, H, model, X)
    old = old_exact_vectors(H, model, X)
    new = exact_frozen_problem_local(H, model, X)
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
    return (old=old,new=new,Egs=Egs,exact_model=exact_model)
end

function first_tree_audit(ratio, H, X, flips)
    model = GBTQuantum.LogGBState(logamp_bias=0.0, phase_bias=0.0, use_phase=false)
    a0 = audit_state("epoch 0 / uniform model", H, model, X)

    tree = GBTQuantum.grow_tree(X, a0.new.target, a0.new.probabilities;
        max_depth=4, min_weight=1e-14, min_gain=0.0)
    raw = predict_all(tree, X)
    mu = sum(a0.new.probabilities .* raw)
    centered = raw .- mu

    @printf("\n  [first exact full-Hilbert tree]\n")
    @printf("    prediction mean <f>_p        = % .6e\n", sum(a0.new.probabilities .* centered))

    E_formula_centered = exact_energy_eta_local(a0.new, X, flips, centered, ETA_TEST, ratio, 1.0)
    E_formula_raw = exact_energy_eta_local(a0.new, X, flips, raw, ETA_TEST, ratio, 1.0)

    model1 = deepcopy(model)
    push!(model1.logamp.trees, scale_tree(tree, ETA_TEST))
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
        ratio=ratio, epoch0_old=a0.old.E, epoch0_new=a0.new.energy,
        epoch0_canonical=a0.exact_model, Egs=a0.Egs,
        max_p_diff=maxabs(a0.new.probabilities-a0.old.p),
        max_eloc_diff=maxabs(a0.new.eloc-a0.old.eloc),
        max_target_diff=maxabs(a0.new.target-a0.old.y),
        eta_energy_centered=E_formula_centered, eta_energy_raw=E_formula_raw,
        canonical_after_tree=E_canonical_after, local_after_tree=a1.E,
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

    X = enumerate_states(N)
    flips = build_flip_index_local(X)
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
    println("  1) If epoch0_new-old != 0, frozen-state construction is wrong.")
    println("  2) If epoch0 agrees but eta-canonical != 0, the hand-written eta-energy formula is wrong.")
    println("  3) If both agree, the bug is later in the stochastic/update path.")
    println("\nResults written to experiments/results/old_new_energy_audit.csv")
end

export main
end

if abspath(PROGRAM_FILE)==@__FILE__
    OldNewEnergyAudit.main()
end
