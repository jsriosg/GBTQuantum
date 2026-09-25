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

    # Construct using the same small H/model/config conventions
    # already used by your smoke-training test.

    H = TFIMHamiltonian(4; J=1.0,h=1.0,periodic=true)
    cfg = TrainingConfig(
        nsamples=32,epochs=5,max_depth=2,eta=0.02,
        burn_in_sweeps=2,sweeps_per_epoch=1,use_phase=false,
        exact_diagnostics=true,exact_every=2
    )

    result = train(H,cfg)

    @test length(result.hilbert_coverage) == 5
    @test length(result.exact_energy) == 5
    @test length(result.exact_variance) == 5

    # Diagnostics requested at 1, 2, 4 and final=5.
    expected_mask = [true, true, false, true, true]

    @test .!isnan.(result.exact_energy) == expected_mask

    @test .!isnan.(result.exact_variance) == expected_mask

    @test all(
        0.0 .<= result.hilbert_coverage .<= 1.0
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