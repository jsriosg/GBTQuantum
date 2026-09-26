Base.@kwdef struct TrainingConfig
    nsamples::Int = 256
    epochs::Int = 200
    max_depth::Int = 4
    eta::Float64 = 0.05
    burn_in_sweeps::Int = 100
    sweeps_per_epoch::Int = 1
    min_leaf_weight::Float64 = 1.0
    min_gain::Float64 = 0.0
    use_phase::Bool = false
    seed::Int = 1234
    exact_diagnostics::Bool = false
    exact_every::Int = 5
end

struct TrainingResult
    model::LogGBState
    energy_history::Vector{Float64}
    energy_imag_history::Vector{Float64}
    variance_history::Vector{Float64}
    acceptance_history::Vector{Float64}
    unique_fraction_history::Vector{Float64}
    magnitude_fit_mse::Vector{Float64}
    phase_fit_mse::Vector{Float64}
    runtime_seconds::Float64

    # Instantaneous fraction of Hilbert space represented
    # by the current training population.
    hilbert_coverage::Vector{Float64}

    # Fraction of Hilbert space encountered at least once
    # during training up to this epoch.
    cumulative_hilbert_coverage::Vector{Float64}

    exact_energy::Vector{Float64}
    exact_variance::Vector{Float64}
end

function train(H::TFIMHamiltonian, cfg::TrainingConfig=TrainingConfig())
    cfg.nsamples > 0 || throw(ArgumentError("nsamples must be positive"))
    cfg.epochs > 0 || throw(ArgumentError("epochs must be positive"))
    cfg.eta > 0 || throw(ArgumentError("eta must be positive"))
    
    if cfg.exact_diagnostics
        cfg.exact_every > 0 ||
            throw(ArgumentError("exact_every must be positive"))

        H.N <= 16 ||
            throw(ArgumentError(
                "Exact diagnostics require enumeration of 2^N states;" *
                "use them only for small validation systems (currently N <= 16)"
            ))
    end
    
    hilbert_coverage = zeros(Float64, cfg.epochs)

    cumulative_hilbert_coverage = zeros(Float64, cfg.epochs)

    # For the system sizes considered here N <= 64, spin_key
    # is a compact UInt64 representation.
    visited_states = Set{UInt64}()

    exact_energy = fill(NaN, cfg.epochs)
    exact_variance = fill(NaN, cfg.epochs)

    rng = MersenneTwister(cfg.seed)
    samples = Matrix{Int8}(undef, cfg.nsamples, H.N)
    @inbounds for i in eachindex(samples)
        samples[i] = rand(rng, Bool) ? Int8(1) : Int8(-1)
    end

    # A=0, Φ=0 => uniform positive initial wavefunction.
    model = LogGBState(logamp_bias=0.0, phase_bias=0.0, use_phase=cfg.use_phase)
    logamps = zeros(Float64, cfg.nsamples)

    for _ in 1:cfg.burn_in_sweeps
        sweep!(rng,model,samples,logamps)
    end

    eh = Vector{Float64}(undef, cfg.epochs)
    eih = Vector{Float64}(undef, cfg.epochs)
    vh = Vector{Float64}(undef, cfg.epochs)
    ah = Vector{Float64}(undef, cfg.epochs)
    uh = Vector{Float64}(undef, cfg.epochs)
    mh = Vector{Float64}(undef, cfg.epochs)
    ph = fill(NaN, cfg.epochs)

    t0 = time_ns()

    for epoch in 1:cfg.epochs
        batch = vmc_batch(H,model,samples)
        yA, yΦ = make_targets(batch)
        weights = batch.counts

        # Gauge fixing: the empirical target is already centered, but removing the
        # fitted tree's weighted constant component prevents normalization drift.
        treeA = grow_tree(batch.states,yA,weights;
                          max_depth=cfg.max_depth,
                          min_weight=cfg.min_leaf_weight,
                          min_gain=cfg.min_gain)
        μA = weighted_mean([predict(treeA,@view batch.states[j,:]) for j in axes(batch.states,1)], weights)
        if μA != 0.0
            # Subtract the constant gauge component by shifting every leaf.
            nodes = copy(treeA.nodes)
            @inbounds for i in eachindex(nodes)
                n = nodes[i]
                n.isleaf && (nodes[i] = Node(n.feature,n.value-μA,n.left,n.right,true))
            end
            treeA = RegressionTree(nodes)
        end
        push!(model.logamp.trees, RegressionTree([
            n.isleaf ? Node(n.feature,cfg.eta*n.value,n.left,n.right,true) : n for n in treeA.nodes
        ]))

        if cfg.use_phase
            treeΦ = grow_tree(batch.states,yΦ,weights;
                              max_depth=cfg.max_depth,
                              min_weight=cfg.min_leaf_weight,
                              min_gain=cfg.min_gain)
            μΦ = weighted_mean([predict(treeΦ,@view batch.states[j,:]) for j in axes(batch.states,1)], weights)
            if μΦ != 0.0
                nodes = copy(treeΦ.nodes)
                @inbounds for i in eachindex(nodes)
                    n = nodes[i]
                    n.isleaf && (nodes[i] = Node(n.feature,n.value-μΦ,n.left,n.right,true))
                end
                treeΦ = RegressionTree(nodes)
            end
            push!(model.phase.trees, RegressionTree([
                n.isleaf ? Node(n.feature,cfg.eta*n.value,n.left,n.right,true) : n for n in treeΦ.nodes
            ]))
            ph[epoch] = weighted_mse(treeΦ,batch.states,yΦ,weights)
        end

       if cfg.exact_diagnostics &&
            (epoch == 1 ||
            epoch % cfg.exact_every == 0 ||
            epoch == cfg.epochs)

            stats = exact_model_energy(model, H)

            exact_energy[epoch] = real(stats.energy)
            exact_variance[epoch] = stats.variance
        end

        eh[epoch] = real(batch.energy)
        eih[epoch] = imag(batch.energy)
        vh[epoch] = batch.variance

        nunique = size(batch.states, 1)
        uh[epoch] = nunique / cfg.nsamples 
        @inbounds for j in axes(batch.states, 1)
            push!(
                visited_states,
                spin_key(@view batch.states[j, :])
            )
        end

        hilbert_coverage[epoch] =
            ldexp(Float64(nunique), -H.N)

        cumulative_hilbert_coverage[epoch] =
            ldexp(Float64(length(visited_states)), -H.N)
        
        mh[epoch] = weighted_mse(treeA,batch.states,yA,weights)

        # The model changed, so cached log amplitudes for the chains are stale.
        refresh_logamps!(logamps,model,samples)

        acc = 0.0
        for _ in 1:cfg.sweeps_per_epoch
            acc += sweep!(rng,model,samples,logamps)
        end
        ah[epoch] = acc / cfg.sweeps_per_epoch
    end

    runtime = (time_ns()-t0)*1e-9
    return TrainingResult(
        model,
        eh,
        eih,
        vh,
        ah,
        uh,
        mh,
        ph,
        runtime,
        hilbert_coverage,
        cumulative_hilbert_coverage,
        exact_energy,
        exact_variance,
    )
end
