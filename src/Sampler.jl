@inline function metropolis_step!(rng::AbstractRNG,
                                  m::LogGBState,
                                  x::AbstractVector{<:Real},
                                  currentA::Float64)
    i = rand(rng, eachindex(x))
    @inbounds x[i] = -x[i]
    proposedA = logamplitude(m,x)

    # log acceptance avoids overflow/underflow:
    # log(|ψ'|²/|ψ|²) = 2(A' - A)
    logratio = 2.0 * (proposedA - currentA)
    if log(rand(rng)) < min(0.0, logratio)
        return true, proposedA
    else
        @inbounds x[i] = -x[i]
        return false, currentA
    end
end

function sweep!(rng::AbstractRNG,
                m::LogGBState,
                samples::Matrix{Int8},
                logamps::Vector{Float64})
    M, N = size(samples)
    length(logamps) == M || throw(DimensionMismatch("logamps must have one value per chain"))
    accepted = 0

    @inbounds for r in 1:M
        A = logamps[r]
        x = @view samples[r,:]
        for _ in 1:N
            ok, A = metropolis_step!(rng,m,x,A)
            accepted += ok
        end
        logamps[r] = A
    end
    return accepted / (M*N)
end

function refresh_logamps!(logamps::Vector{Float64}, m::LogGBState, samples::Matrix{Int8})
    @inbounds for r in axes(samples,1)
        logamps[r] = logamplitude(m,@view samples[r,:])
    end
    return logamps
end
