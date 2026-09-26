using Test
using GBTQuantum
using Random

@testset "Weighted raw-spin tree" begin
    X = Int8[-1 -1; -1 1; 1 -1; 1 1]
    y = [-1.0,-1.0,1.0,1.0]
    w = [1,3,2,4]
    t = GBTQuantum.grow_tree(X,y,w; max_depth=1)
    @test predict(t,@view X[1,:]) ≈ -1.0
    @test predict(t,@view X[4,:]) ≈ 1.0
end

@testset "Log wavefunction ratios" begin
    m = LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    x = Int8[-1,1,-1]
    y = Int8[1,1,-1]
    @test psi_ratio(m,x,y) ≈ 1.0 + 0im
end

@testset "TFIM uniform state local energy" begin
    H = TFIMHamiltonian(4; J=1.0,h=0.5,periodic=true)
    m = LogGBState(logamp_bias=0.0,phase_bias=0.0,use_phase=false)
    x = Int8[1,1,1,1]
    # diagonal=-4, four spin flips each contribute -0.5
    @test local_energy!(H,m,x) ≈ -6.0 + 0im
    @test x == Int8[1,1,1,1]  # local_energy! restores the state
end

@testset "Sample compression preserves counts" begin
    S = Int8[-1 -1; -1 -1; 1 -1; -1 -1]
    U,c = compress_samples(S)
    @test size(U,1) == 2
    @test sum(c) == 4
    @test sort(c) == [1,3]
end

@testset "Small smoke training" begin
    H = TFIMHamiltonian(4; J=1.0,h=1.0,periodic=true)
    cfg = TrainingConfig(nsamples=32,epochs=3,max_depth=2,eta=0.02,
                         burn_in_sweeps=2,sweeps_per_epoch=1,use_phase=false)
    r = train(H,cfg)
    @test length(r.energy_history) == 3
    @test all(isfinite,r.energy_history)
    @test all(isfinite,r.variance_history)
end

@testset "Monte Carlo statistics" begin

    # Independent data should have ESS reasonably close to N.
    rng = MersenneTwister(1234)
    x = randn(rng, 10_000)

    τ = integrated_autocorrelation_time(x)
    neff = effective_sample_size(x)

    @test τ >= 0.5
    @test 0.5length(x) < neff <= length(x)

    # Artificially correlated AR(1) sequence
    y = zeros(10_000)

    for i in 2:length(y)
        y[i] = 0.9y[i - 1] + randn(rng)
    end

    neff_corr = effective_sample_size(y)

    @test neff_corr < neff
end

@testset "Independent validation VMC" begin

    H = TFIMHamiltonian(
        4;
        J = 1.0,
        h = 1.0,
        periodic = true,
    )

    model = LogGBState(use_phase = false)

    stats = validation_vmc(
        model,
        H;
        nsamples = 256,
        burn_in_sweeps = 20,
        seed = 1234,
    )

    @test isfinite(stats.energy)
    @test stats.variance >= 0.0
    @test stats.tau_int >= 0.5
    @test 0.0 < stats.ess <= 256
    @test stats.standard_error >= 0.0
    @test length(stats.samples) == 256
end

H = TFIMHamiltonian(
    4;
    J = 1.0,
    h = 1.0,
    periodic = true,
)

model = LogGBState(use_phase = false)

stats = validation_vmc(
    model,
    H;
    nsamples = 100_000,
    burn_in_sweeps = 100,
    thinning_sweeps = 1,
    seed = 1234,
)

println("Energy   = ", stats.energy)
println("Variance = ", stats.variance)
println("Min Eloc = ", minimum(stats.samples))
println("Max Eloc = ", maximum(stats.samples))

println(
    "Unique local energies = ",
    sort(unique(stats.samples))
)

@testset "Exact model energy" begin

    H = TFIMHamiltonian(
        4;
        J = 1.0,
        h = 1.0,
        periodic = true,
    )

    model = LogGBState(use_phase = false)

    stats = exact_model_energy(model, H)

    @test stats.nstates == 16
    @test isapprox(real(stats.energy), -4.0; atol=1e-12)
    @test isapprox(imag(stats.energy), 0.0; atol=1e-12)
    @test isapprox(stats.variance, 4.0; atol=1e-12)
end

@testset "Uniform-state validation distribution" begin

    H = TFIMHamiltonian(
        4;
        J = 1.0,
        h = 1.0,
        periodic = true,
    )

    model = LogGBState(use_phase = false)

    stats = validation_vmc(
        model,
        H;
        nsamples = 20_000,
        burn_in_sweeps = 100,
        thinning_sweeps = 1,
        seed = 1234,
    )

    @test abs(stats.energy + 4.0) < 0.1
    @test abs(stats.variance - 4.0) < 0.3
end

@testset "Exact training diagnostics" begin

    H = TFIMHamiltonian(4; J=1.0, h=1.0, periodic=true)

    cfg = TrainingConfig(
        nsamples=32,
        epochs=5,
        max_depth=2,
        eta=0.02,
        burn_in_sweeps=2,
        sweeps_per_epoch=1,
        use_phase=false,
        exact_diagnostics=true,
        exact_every=2,
    )

    result = train(H, cfg)

    @test length(result.hilbert_coverage) == cfg.epochs
    @test length(result.cumulative_hilbert_coverage) == cfg.epochs
    @test length(result.exact_energy) == cfg.epochs
    @test length(result.exact_variance) == cfg.epochs

    @test .!isnan.(result.exact_energy) ==
          [true, true, false, true, true]

    @test .!isnan.(result.exact_variance) ==
          [true, true, false, true, true]

    @test all(0.0 .<= result.hilbert_coverage .<= 1.0)
    @test all(0.0 .<= result.cumulative_hilbert_coverage .<= 1.0)

    # Cumulative coverage can never decrease.
    @test all(diff(result.cumulative_hilbert_coverage) .>= 0.0)

    # Everything represented in the current population
    # must already belong to the cumulative visited set.
    @test all(
        result.cumulative_hilbert_coverage .>=
        result.hilbert_coverage
    )
end

@testset "Spin observables" begin

    # --------------------------------------------------------
    # Basic longitudinal magnetization
    # --------------------------------------------------------

    x_up = Int8[1, 1, 1, 1]
    x_down = Int8[-1, -1, -1, -1]
    x_zero = Int8[1, -1, 1, -1]

    @test magnetization_z(x_up) ≈ 1.0
    @test magnetization_z(x_down) ≈ -1.0
    @test magnetization_z(x_zero) ≈ 0.0

    @test abs_magnetization_z(x_up) ≈ 1.0
    @test abs_magnetization_z(x_down) ≈ 1.0
    @test abs_magnetization_z(x_zero) ≈ 0.0

    @test magnetization_z2(x_up) ≈ 1.0
    @test magnetization_z2(x_down) ≈ 1.0
    @test magnetization_z2(x_zero) ≈ 0.0


    # --------------------------------------------------------
    # Weighted compressed-sample observables
    # --------------------------------------------------------

    states = Int8[
         1  1  1  1;
        -1 -1 -1 -1;
         1 -1  1 -1
    ]

    counts = [2, 1, 1]

    obs = sampled_diagonal_observables(states, counts)

    # Weighted distribution:
    #
    # state        m_z       weight
    # ++++          1          2
    # ----         -1          1
    # +-+-          0          1
    #
    # <m_z>   = (2 - 1)/4 = 1/4
    # <|m_z|> = (2 + 1)/4 = 3/4
    # <m_z²>  = (2 + 1)/4 = 3/4

    @test obs.mz ≈ 0.25
    @test obs.abs_mz ≈ 0.75
    @test obs.mz2 ≈ 0.75


    # --------------------------------------------------------
    # Uniform wavefunction transverse magnetization
    # --------------------------------------------------------

    model = LogGBState(use_phase=false)

    x = Int8[1, -1, 1, -1]
    x_original = copy(x)

    mx = local_magnetization_x(model, x)

    # For a uniform wavefunction:
    #
    # ψ(x^i)/ψ(x) = 1
    #
    # for every spin flip, therefore <σ_x> = 1.

    @test mx ≈ 1.0 + 0im

    # The observable must restore the configuration.
    @test x == x_original


    # --------------------------------------------------------
    # Sampled transverse magnetization
    # --------------------------------------------------------

    mx_sampled = sampled_magnetization_x(
        model,
        states,
        counts,
    )

    @test mx_sampled ≈ 1.0 + 0im

end

@testset "Exact model observables" begin

    H = TFIMHamiltonian(
        4;
        J=1.0,
        h=1.0,
        periodic=true,
    )

    model = LogGBState(use_phase=false)

    obs = exact_model_observables(model, H)

    @test obs.nstates == 16

    @test isapprox(obs.mz, 0.0; atol=1e-12)
    @test isapprox(obs.abs_mz, 0.375; atol=1e-12)
    @test isapprox(obs.mz2, 0.25; atol=1e-12)

    @test isapprox(real(obs.mx), 1.0; atol=1e-12)
    @test isapprox(imag(obs.mx), 0.0; atol=1e-12)
end

@testset "Exact ground-state observables" begin

    N = 4
    J = 1.0
    h = 1.0

    H = TFIMHamiltonian(
        N;
        J=J,
        h=h,
        periodic=true,
    )

    obs = exact_ground_observables(H)

    @test obs.nstates == 16
    @test isapprox(obs.mz, 0.0; atol=1e-12)

    @test 0.0 <= obs.abs_mz <= 1.0
    @test 0.0 <= obs.mz2 <= 1.0
    @test 0.0 <= obs.mx <= 1.0

    # Hellmann-Feynman check.
    ε = 1e-5

    Hp = TFIMHamiltonian(
        N;
        J=J,
        h=h + ε,
        periodic=true,
    )

    Hm = TFIMHamiltonian(
        N;
        J=J,
        h=h - ε,
        periodic=true,
    )

    dEdh =
        (exact_ground_energy(Hp) -
         exact_ground_energy(Hm)) / (2ε)

    mx_HF = -dEdh / N

    @test isapprox(
        obs.mx,
        mx_HF;
        rtol=1e-7,
        atol=1e-9,
    )
end

@testset "VMC observable validation" begin

    H = TFIMHamiltonian(
        4;
        J=1.0,
        h=1.0,
        periodic=true,
    )

    model = LogGBState(use_phase=false)

    obs = validation_observables(
        model,
        H;
        nsamples=20_000,
        burn_in_sweeps=100,
        thinning_sweeps=1,
        seed=24680,
    )

    # Exact values for the uniform wavefunction.
    @test isapprox(obs.mz, 0.0; atol=0.03)
    @test isapprox(obs.abs_mz, 0.375; atol=0.03)
    @test isapprox(obs.mz2, 0.25; atol=0.03)

    # m_x local estimator is exactly 1 for every state.
    @test isapprox(real(obs.mx), 1.0; atol=1e-12)
    @test isapprox(imag(obs.mx), 0.0; atol=1e-12)

    # Statistical diagnostics must be sensible.
    @test obs.tau_mz >= 0.5
    @test obs.tau_abs_mz >= 0.5
    @test obs.tau_mz2 >= 0.5

    @test 0.0 < obs.ess_mz <= 20_000
    @test 0.0 < obs.ess_abs_mz <= 20_000
    @test 0.0 < obs.ess_mz2 <= 20_000

    # The estimator must return one value per measurement.
    @test length(obs.mz_samples) == 20_000
    @test length(obs.abs_mz_samples) == 20_000
    @test length(obs.mz2_samples) == 20_000
    @test length(obs.mx_samples) == 20_000
end

@testset "VMC distribution validation" begin

    H = TFIMHamiltonian(
        4;
        J=1.0,
        h=1.0,
        periodic=true,
    )

    model = LogGBState(use_phase=false)

    dist = validation_distribution(
        model,
        H;
        nsamples=50_000,
        burn_in_sweeps=100,
        thinning_sweeps=1,
        seed=112233,
    )

    @test dist.nstates == 16
    @test sum(dist.counts) == 50_000

    @test isapprox(
        sum(dist.exact_prob),
        1.0;
        atol=1e-12,
    )

    @test isapprox(
        sum(dist.sampled_prob),
        1.0;
        atol=1e-12,
    )

    # Exact uniform distribution.
    @test all(
        isapprox.(dist.exact_prob, 1 / 16; atol=1e-12)
    )

    # MC distribution should approach uniformity.
    @test dist.total_variation < 0.03

    # Magnetization-sector distribution should also agree.
    @test dist.magnetization_tv < 0.02
end