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