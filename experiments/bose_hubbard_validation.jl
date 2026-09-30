module BoseHubbardValidation

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GBTQuantum
using LinearAlgebra
using Random
using Printf

const L = 4
const NBOS = 4
const J = 1.0
const U = 3.0
const MU = 0.37
const SEED = 20260929

function check(name, value, tol)
    ok = value <= tol
    @printf("  %-44s %s  value=% .3e  tol=% .1e\n",
            name, ok ? "PASS" : "FAIL", value, tol)
    ok || error("validation failed: $name")
end

function check_bool(name, ok)
    @printf("  %-44s %s\n", name, ok ? "PASS" : "FAIL")
    ok || error("validation failed: $name")
end

function arbitrary_model()
    m = LogGBState(logamp_bias=0.13, phase_bias=0.0, use_phase=false)

    # A deliberately nonuniform positive test wavefunction.  The tree is
    # constructed directly so the local-energy identity is tested independently
    # of tree fitting.
    nodes = Node[
        Node(1, 1.5, 0.0, 2, 3, false),
        Node(0, 0.0, -0.21, 0, 0, true),
        Node(2, 0.5, 0.0, 4, 5, false),
        Node(0, 0.0, 0.17, 0, 0, true),
        Node(0, 0.0, 0.39, 0, 0, true),
    ]
    push!(m.logamp.trees, RegressionTree(nodes))
    return m
end

function dense_local_energy(M, basis, model)
    d = size(basis,1)
    psi = [exp(logamplitude(model, @view basis[r,:])) for r in 1:d]
    Hpsi = M * psi
    return Hpsi ./ psi
end

function validate_hamiltonian()
    H = BoseHubbardHamiltonian(L; J=J, U=U, mu=MU, periodic=true)
    M, basis = GBTQuantum.exact_hamiltonian(H, NBOS)

    println("\n[1] Fixed-number basis and dense Hamiltonian")
    check_bool("basis dimension = binomial(N+L-1,N)",
               size(basis,1) == binomial(NBOS+L-1, NBOS))
    check_bool("every basis state has total Nbos",
               all(sum(@view basis[r,:]) == NBOS for r in axes(basis,1)))
    check_bool("all occupations are non-negative", all(basis .>= 0))
    check("Hermiticity", opnorm(Matrix(M) - Matrix(M)', Inf), 1e-13)

    # Chemical potential must be a pure constant shift in a fixed-N sector.
    H0 = BoseHubbardHamiltonian(L; J=J, U=U, mu=0.0, periodic=true)
    M0, basis0 = GBTQuantum.exact_hamiltonian(H0, NBOS)
    check_bool("mu=0 basis ordering unchanged", basis == basis0)
    expected = -MU * NBOS * Matrix{Float64}(I, size(M,1), size(M,1))
    check("chemical potential is -mu*N identity",
          opnorm(Matrix(M)-Matrix(M0)-expected, Inf), 1e-13)

    # Explicit matrix element: |2,1,1,0> -> |1,2,1,0>
    a = Int16[2,1,1,0]
    b = Int16[1,2,1,0]
    ia = findfirst(r -> all(@view(basis[r,:]) .== a), axes(basis,1))
    ib = findfirst(r -> all(@view(basis[r,:]) .== b), axes(basis,1))
    expected_hop = -J * sqrt(2.0 * (1.0+1.0))
    check("bosonic hopping sqrt(n_i(n_j+1))",
          abs(Matrix(M)[ib,ia] - expected_hop), 1e-13)

    return H, Matrix(M), basis
end

function validate_local_energy(H, M, basis)
    println("\n[2] State-resolved local energy")
    model = arbitrary_model()
    direct = dense_local_energy(M, basis, model)
    via_local = ComplexF64[
        local_energy!(H, model, @view basis[r,:]) for r in axes(basis,1)
    ]
    check("E_loc = (H psi)_n / psi_n for every state",
          maximum(abs.(direct .- via_local)), 5e-13)

    # The local-energy routine mutates in-place internally; verify restoration.
    x = copy(@view basis[17,:])
    before = copy(x)
    local_energy!(H, model, x)
    check_bool("local_energy! restores occupation vector", x == before)
    return model
end

function validate_sampler(H, model, basis)
    println("\n[3] Number-conserving Metropolis sampler")
    rng = MersenneTwister(SEED)
    x = Int16[1,1,1,1]
    A = logamplitude(model, x)
    conserved = true
    nonnegative = true
    accepted = 0
    for _ in 1:20_000
        ok, A = bh_metropolis_step!(rng, model, x, A, H)
        accepted += ok
        conserved &= sum(x) == NBOS
        nonnegative &= all(x .>= 0)
    end
    check_bool("particle number conserved for 20,000 proposals", conserved)
    check_bool("occupations remain non-negative", nonnegative)
    check("cached log amplitude remains exact",
          abs(A - logamplitude(model,x)), 1e-13)
    check_bool("sampler accepts at least one proposal", accepted > 0)

    # For the uniform wavefunction, the symmetric directed-hop proposal and
    # Metropolis ratio imply a uniform stationary distribution over the
    # fixed-N occupation basis.  This is a direct sampler sanity check.
    uniform_model = LogGBState(logamp_bias=0.0, use_phase=false)
    x .= Int16[1,1,1,1]
    A = logamplitude(uniform_model,x)
    counts = Dict{Tuple{Vararg{Int16}},Int}()
    for _ in 1:10_000
        _, A = bh_metropolis_step!(rng, uniform_model, x, A, H)
    end
    for _ in 1:350_000
        _, A = bh_metropolis_step!(rng, uniform_model, x, A, H)
        key = Tuple(x)
        counts[key] = get(counts,key,0) + 1
    end
    empirical = [get(counts,Tuple(@view basis[r,:]),0)/350_000 for r in axes(basis,1)]
    uniform = 1.0 / size(basis,1)
    tv = 0.5 * sum(abs.(empirical .- uniform))
    check("uniform-wavefunction sampler total variation", tv, 0.025)
end

function validate_numeric_tree()
    println("\n[4] Numerical CART thresholds")
    X = Int16[
        0 0;
        0 1;
        1 0;
        1 1;
        2 0;
        2 1;
        3 0;
        3 1
    ]
    y = Float64[-2,-2,-1,-1,1,1,2,2]
    w = ones(Float64, size(X,1))
    t = grow_tree_numeric(X,y,w; max_depth=1)
    root = t.nodes[1]
    check_bool("root is a split", !root.isleaf)
    check_bool("numeric split uses feature 1", root.feature == 1)
    check("best threshold is 1.5", abs(root.threshold - 1.5), 1e-13)
    pred = [predict(t,@view X[r,:]) for r in axes(X,1)]
    check_bool("ordered occupations route consistently",
               pred[1] == pred[2] == pred[3] == pred[4] &&
               pred[5] == pred[6] == pred[7] == pred[8] &&
               pred[1] < pred[5])
end

function main()
    println("="^78)
    println("BOSE-HUBBARD L=4, Nbos=4 IMPLEMENTATION VALIDATION")
    println("No GBT-VMC training is performed by this script.")
    println("="^78)

    H, M, basis = validate_hamiltonian()
    model = validate_local_energy(H, M, basis)
    validate_sampler(H, model, basis)
    validate_numeric_tree()

    exact = exact_ground_state(H, NBOS)
    @printf("\nExact validation ground energy: %.12f\n", exact.energy)
    println("\nALL BOSE-HUBBARD IMPLEMENTATION CHECKS PASSED")
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    BoseHubbardValidation.main()
end
