struct TFIMHamiltonian
    N::Int
    J::Float64
    h::Float64
    periodic::Bool
end

TFIMHamiltonian(N::Integer; J::Real=1.0, h::Real=1.0, periodic::Bool=true) =
    TFIMHamiltonian(Int(N), Float64(J), Float64(h), periodic)

@inline function diagonal(H::TFIMHamiltonian, x::AbstractVector{<:Real})
    e = 0.0
    @inbounds @simd for i in 1:(H.N-1)
        e -= H.J * x[i] * x[i+1]
    end
    H.periodic && (e -= H.J * x[H.N] * x[1])
    return e
end

# Allocation-free local energy. The state is flipped in-place and restored.
function local_energy!(H::TFIMHamiltonian,
                       m::LogGBState,
                       x::AbstractVector{<:Real})
    A0 = logamplitude(m,x)
    Φ0 = phase(m,x)
    z = ComplexF64(diagonal(H,x))

    @inbounds for i in 1:H.N
        x[i] = -x[i]
        ΔA = logamplitude(m,x) - A0
        ΔΦ = phase(m,x) - Φ0
        # This exponential is physically required by ψ(x')/ψ(x).
        z -= H.h * exp(ΔA) * cis(ΔΦ)
        x[i] = -x[i]
    end
    return z
end

# Dense exact diagonalization is only a validation tool for small systems.
function exact_hamiltonian(H::TFIMHamiltonian)
    H.N <= 14 ||
        error("Dense exact diagonalization is restricted to N <= 14")

    d = 1 << H.N
    M = zeros(Float64, d, d)
    x = Vector{Int8}(undef, H.N)

    @inbounds for s in 0:(d - 1)

        for i in 1:H.N
            x[i] =
                ((s >> (i - 1)) & 1) == 1 ?
                Int8(1) : Int8(-1)
        end

        M[s + 1, s + 1] = diagonal(H, x)

        for i in 1:H.N
            sp = s ⊻ (1 << (i - 1))
            M[sp + 1, s + 1] = -H.h
        end
    end

    return Symmetric(M)
end

function exact_ground_energy(H::TFIMHamiltonian)
    return eigmin(exact_hamiltonian(H))
end
