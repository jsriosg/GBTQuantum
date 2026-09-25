# ============================================================
# Observables.jl
#
# Physical observables for spin systems represented in the
# sigma-z computational basis.
# ============================================================


# ------------------------------------------------------------
# Diagonal observables
# ------------------------------------------------------------

@inline function magnetization_z(x::AbstractVector{<:Real})
    return sum(x) / length(x)
end


@inline function abs_magnetization_z(x::AbstractVector{<:Real})
    return abs(magnetization_z(x))
end


@inline function magnetization_z2(x::AbstractVector{<:Real})
    mz = magnetization_z(x)
    return mz * mz
end


# ------------------------------------------------------------
# Transverse magnetization
# ------------------------------------------------------------

"""
    local_magnetization_x(m, x)

Local estimator of the transverse magnetization per site

    m_x = (1/N) Σᵢ σᵢˣ.

For configuration x,

    (m_x)_loc(x) = (1/N) Σᵢ ψ(x⁽ⁱ⁾)/ψ(x),

where x⁽ⁱ⁾ is x with spin i flipped.

The state x is temporarily modified in-place and restored.
"""
function local_magnetization_x(
    m::LogGBState,
    x::AbstractVector{<:Real},
)
    A0 = logamplitude(m, x)
    Φ0 = phase(m, x)

    mx = 0.0 + 0.0im

    @inbounds for i in eachindex(x)
        x[i] = -x[i]

        ΔA = logamplitude(m, x) - A0
        ΔΦ = phase(m, x) - Φ0

        mx += exp(ΔA) * cis(ΔΦ)

        x[i] = -x[i]
    end

    return mx / length(x)
end


# ------------------------------------------------------------
# Monte Carlo estimators from compressed samples
# ------------------------------------------------------------

function sampled_diagonal_observables(
    states::AbstractMatrix,
    counts::AbstractVector{<:Integer},
)
    M, N = size(states)

    length(counts) == M ||
        throw(DimensionMismatch(
            "counts must contain one value per unique state"
        ))

    total_weight = sum(counts)

    total_weight > 0 ||
        throw(ArgumentError("total sample weight must be positive"))

    mz = 0.0
    abs_mz = 0.0
    mz2 = 0.0

    @inbounds for r in 1:M
        x = @view states[r, :]
        w = counts[r]

        m = sum(x) / N

        mz     += w * m
        abs_mz += w * abs(m)
        mz2    += w * m * m
    end

    invweight = 1.0 / total_weight

    return (
        mz     = mz * invweight,
        abs_mz = abs_mz * invweight,
        mz2    = mz2 * invweight,
    )
end


function sampled_magnetization_x(
    model::LogGBState,
    states::AbstractMatrix,
    counts::AbstractVector{<:Integer},
)
    M = size(states, 1)

    length(counts) == M ||
        throw(DimensionMismatch(
            "counts must contain one value per unique state"
        ))

    total_weight = sum(counts)

    total_weight > 0 ||
        throw(ArgumentError("total sample weight must be positive"))

    mx = 0.0 + 0.0im

    @inbounds for r in 1:M
        x = @view states[r, :]
        mx += counts[r] * local_magnetization_x(model, x)
    end

    return mx / total_weight
end

"""
    exact_model_observables(model, H)

Compute exact expectation values of selected spin observables for the
wavefunction represented by `model`, by enumerating the complete Hilbert
space.

This is a validation routine for small systems only.

Returns:
    mz      = <m_z>
    abs_mz  = <|m_z|>
    mz2     = <m_z^2>
    mx      = <m_x>
    nstates = number of enumerated basis states
"""
function exact_model_observables(
    model::LogGBState,
    H::TFIMHamiltonian,
)
    N = H.N

    N < 64 || throw(
        ArgumentError("exact_model_observables requires N < 64")
    )

    nstates = 1 << N

    states = Matrix{Int8}(undef, nstates, N)
    logweights = Vector{Float64}(undef, nstates)

    # ---------------------------------------------------------
    # Enumerate the Hilbert space and evaluate |ψ(x)|².
    # ---------------------------------------------------------

    @inbounds for k in 0:(nstates - 1)
        row = k + 1

        for j in 1:N
            bit = (k >> (j - 1)) & 1
            states[row, j] =
                bit == 1 ? Int8(1) : Int8(-1)
        end

        x = @view states[row, :]

        logweights[row] =
            2.0 * logamplitude(model, x)
    end

    # Stable normalization of |ψ|².
    maxlogweight = maximum(logweights)

    weights = Vector{Float64}(undef, nstates)

    Z = 0.0

    @inbounds @simd for k in 1:nstates
        w = exp(logweights[k] - maxlogweight)
        weights[k] = w
        Z += w
    end

    # ---------------------------------------------------------
    # Observable expectation values.
    # ---------------------------------------------------------

    mz = 0.0
    abs_mz = 0.0
    mz2 = 0.0
    mx = 0.0 + 0.0im

    @inbounds for k in 1:nstates
        x = @view states[k, :]
        p = weights[k] / Z

        m = magnetization_z(x)

        mz     += p * m
        abs_mz += p * abs(m)
        mz2    += p * m * m

        mx += p * local_magnetization_x(model, x)
    end

    return (
        mz = mz,
        abs_mz = abs_mz,
        mz2 = mz2,
        mx = mx,
        nstates = nstates,
    )
end

"""
    exact_ground_observables(H)

Compute selected observables directly from the exact TFIM ground-state
eigenvector.

This is used only for small-system validation.
"""
function exact_ground_observables(H::TFIMHamiltonian)

    gs = exact_ground_state(H)

    N = H.N
    nstates = 1 << N

    ψ = gs.state

    mz = 0.0
    abs_mz = 0.0
    mz2 = 0.0
    mx = 0.0

    x = Vector{Int8}(undef, N)

    @inbounds for s in 0:(nstates - 1)

        row = s + 1
        p = abs2(ψ[row])

        msum = 0

        for i in 1:N
            spin =
                ((s >> (i - 1)) & 1) == 1 ? 1 : -1

            x[i] = Int8(spin)
            msum += spin
        end

        m = msum / N

        mz     += p * m
        abs_mz += p * abs(m)
        mz2    += p * m * m

        # <σ_i^x> connects s with the configuration
        # obtained by flipping spin i.
        for i in 1:N
            sp = s ⊻ (1 << (i - 1))

            mx += real(conj(ψ[row]) * ψ[sp + 1]) / N
        end
    end

    return (
        energy = gs.energy,
        mz = mz,
        abs_mz = abs_mz,
        mz2 = mz2,
        mx = mx,
        nstates = nstates,
    )
end