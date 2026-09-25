module GBTQuantum

using LinearAlgebra
using Random
using Statistics

include("Trees.jl")
include("LogWavefunction.jl")
include("TFIM.jl")
include("Observables.jl")
include("Sampler.jl")
include("VMC.jl")
include("Optimizer.jl")

export Node, RegressionTree, GBMachine, LogGBState,
       TFIMHamiltonian, TrainingConfig, TrainingResult,
       predict, logamplitude, phase, logpsi, logpsi_ratio, psi_ratio,
       local_energy!, compress_samples, vmc_batch, make_targets,
       train, exact_ground_energy, integrated_autocorrelation_time, effective_sample_size,
       energy_standard_error, validation_vmc, exact_model_energy,
       magnetization_z, abs_magnetization_z, magnetization_z2, local_magnetization_x,
       sampled_diagonal_observables, sampled_magnetization_x, exact_ground_state,
       exact_model_observables, exact_ground_observables      
end
