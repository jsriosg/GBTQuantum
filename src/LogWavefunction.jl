mutable struct LogGBState
    logamp::GBMachine
    phase::GBMachine
    use_phase::Bool
end

LogGBState(; logamp_bias::Real=0.0, phase_bias::Real=0.0, use_phase::Bool=true) =
    LogGBState(GBMachine(logamp_bias), GBMachine(phase_bias), use_phase)

@inline logamplitude(m::LogGBState, x::AbstractVector{<:Real}) = predict(m.logamp, x)
@inline phase(m::LogGBState, x::AbstractVector{<:Real}) =
    m.use_phase ? predict(m.phase, x) : 0.0

@inline function logpsi(m::LogGBState, x::AbstractVector{<:Real})
    return complex(logamplitude(m,x), phase(m,x))
end

@inline function logpsi_ratio(m::LogGBState,
                              xnew::AbstractVector{<:Real},
                              xold::AbstractVector{<:Real})
    return logpsi(m,xnew) - logpsi(m,xold)
end

# Ratios are the primitive quantity needed by VMC. Absolute normalization is never required.
@inline function psi_ratio(m::LogGBState,
                           xnew::AbstractVector{<:Real},
                           xold::AbstractVector{<:Real})
    Δ = logpsi_ratio(m,xnew,xold)
    return exp(real(Δ)) * cis(imag(Δ))
end

@inline wavefunction(m::LogGBState, x::AbstractVector{<:Real}) = exp(logpsi(m,x))
