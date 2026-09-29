struct BoseHubbardHamiltonian
    L::Int
    J::Float64
    U::Float64
    mu::Float64
    periodic::Bool
end

BoseHubbardHamiltonian(L::Integer; J::Real=1.0, U::Real=1.0,
                       mu::Real=0.0, periodic::Bool=true) =
    BoseHubbardHamiltonian(Int(L), Float64(J), Float64(U), Float64(mu), periodic)

@inline function diagonal(H::BoseHubbardHamiltonian, n::AbstractVector{<:Integer})
    length(n) == H.L || throw(DimensionMismatch("occupation vector has wrong length"))
    e = 0.0
    Nbos = 0
    @inbounds for i in 1:H.L
        ni = Int(n[i])
        ni >= 0 || throw(ArgumentError("bosonic occupations must be non-negative"))
        e += 0.5 * H.U * ni * (ni - 1)
        Nbos += ni
    end
    return e - H.mu * Nbos
end

function bh_bonds(H::BoseHubbardHamiltonian)
    bonds = [(i, i+1) for i in 1:(H.L-1)]
    if H.periodic && H.L > 2
        push!(bonds, (H.L, 1))
    end
    return bonds
end

# Local energy in a fixed-particle-number occupation basis.
# For every undirected bond <i,j>, both hopping directions are included.
function local_energy!(H::BoseHubbardHamiltonian,
                       m::LogGBState,
                       n::AbstractVector{<:Integer})
    A0 = logamplitude(m, n)
    Phi0 = phase(m, n)
    z = ComplexF64(diagonal(H, n))

    @inbounds for (i,j) in bh_bonds(H)
        ni = Int(n[i])
        nj = Int(n[j])

        if ni > 0
            n[i] -= 1
            n[j] += 1
            dA = logamplitude(m, n) - A0
            dPhi = phase(m, n) - Phi0
            z -= H.J * sqrt(ni * (nj + 1.0)) * exp(dA) * cis(dPhi)
            n[j] -= 1
            n[i] += 1
        end

        if nj > 0
            n[j] -= 1
            n[i] += 1
            dA = logamplitude(m, n) - A0
            dPhi = phase(m, n) - Phi0
            z -= H.J * sqrt(nj * (ni + 1.0)) * exp(dA) * cis(dPhi)
            n[i] -= 1
            n[j] += 1
        end
    end
    return z
end

# Enumerate weak compositions of Nbos into L sites.
function bose_hubbard_basis(L::Integer, Nbos::Integer)
    L > 0 || throw(ArgumentError("L must be positive"))
    Nbos >= 0 || throw(ArgumentError("Nbos must be non-negative"))
    dim = binomial(Int(Nbos + L - 1), Int(Nbos))
    states = Matrix{Int16}(undef, dim, L)
    x = zeros(Int16, L)
    row = Ref(0)

    function fill_site(site::Int, remaining::Int)
        if site == L
            x[site] = Int16(remaining)
            row[] += 1
            states[row[], :] .= x
            return
        end
        for ni in 0:remaining
            x[site] = Int16(ni)
            fill_site(site + 1, remaining - ni)
        end
    end

    fill_site(1, Int(Nbos))
    return states
end

# Dense exact Hamiltonian restricted to a fixed-N sector.
# This is validation-only and is intentionally limited to small sectors.
function exact_hamiltonian(H::BoseHubbardHamiltonian, Nbos::Integer;
                           max_dimension::Int=10_000)
    states = bose_hubbard_basis(H.L, Nbos)
    d = size(states, 1)
    d <= max_dimension || error("exact Bose-Hubbard validation restricted to dimension <= $max_dimension; got $d")

    index = Dict{Tuple{Vararg{Int16}},Int}()
    @inbounds for r in 1:d
        index[Tuple(@view states[r,:])] = r
    end

    M = zeros(Float64, d, d)
    @inbounds for col in 1:d
        n = @view states[col,:]
        M[col,col] = diagonal(H, n)

        for (i,j) in bh_bonds(H)
            ni = Int(n[i]); nj = Int(n[j])
            if ni > 0
                x = collect(n)
                x[i] -= 1; x[j] += 1
                row = index[Tuple(x)]
                M[row,col] = -H.J * sqrt(ni * (nj + 1.0))
            end
            if nj > 0
                x = collect(n)
                x[j] -= 1; x[i] += 1
                row = index[Tuple(x)]
                M[row,col] = -H.J * sqrt(nj * (ni + 1.0))
            end
        end
    end
    return Symmetric(M), states
end

function exact_ground_state(H::BoseHubbardHamiltonian, Nbos::Integer;
                            max_dimension::Int=10_000)
    M, states = exact_hamiltonian(H, Nbos; max_dimension=max_dimension)
    F = eigen(M)
    k = argmin(F.values)
    return (energy=F.values[k], state=F.vectors[:,k], basis=states)
end

exact_ground_energy(H::BoseHubbardHamiltonian, Nbos::Integer; kwargs...) =
    exact_ground_state(H, Nbos; kwargs...).energy

# Symmetric number-conserving Metropolis proposal: choose one directed
# nearest-neighbour hop uniformly. Empty-source proposals are rejected.
function bh_metropolis_step!(rng::AbstractRNG,
                             m::LogGBState,
                             n::AbstractVector{<:Integer},
                             currentA::Float64,
                             H::BoseHubbardHamiltonian)
    bonds = bh_bonds(H)
    b = bonds[rand(rng, eachindex(bonds))]
    if rand(rng, Bool)
        src, dst = b
    else
        dst, src = b
    end

    @inbounds n[src] == 0 && return false, currentA

    @inbounds begin
        n[src] -= 1
        n[dst] += 1
    end
    proposedA = logamplitude(m, n)
    logratio = 2.0 * (proposedA - currentA)

    if log(rand(rng)) < min(0.0, logratio)
        return true, proposedA
    else
        @inbounds begin
            n[dst] -= 1
            n[src] += 1
        end
        return false, currentA
    end
end

function bh_sweep!(rng::AbstractRNG,
                   m::LogGBState,
                   H::BoseHubbardHamiltonian,
                   samples::Matrix{Int16},
                   logamps::Vector{Float64})
    M, L = size(samples)
    L == H.L || throw(DimensionMismatch("sample width must equal H.L"))
    length(logamps) == M || throw(DimensionMismatch("logamps must have one value per chain"))
    accepted = 0
    @inbounds for r in 1:M
        A = logamps[r]
        n = @view samples[r,:]
        for _ in 1:L
            ok, A = bh_metropolis_step!(rng, m, n, A, H)
            accepted += ok
        end
        logamps[r] = A
    end
    return accepted / (M*L)
end
