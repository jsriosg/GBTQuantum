# TFIM Physical-Observable Validation Milestone

This note freezes the conclusions of the fixed-regularization observable-scaling experiment before moving to Armijo.

## Frozen setup

- N = 8, 10, 12, 14
- J/h = 1.0, 2.0
- lambda = 0, 1
- five seeds
- M = 512
- depth = 4
- epochs = 150
- fixed eta = 0.05
- observables: <mz>, <|mz|>, <mz^2>, <mx>
- exact ground-state/model quantities are evaluation-only and never affect training.

## Main result: difficult regime J/h = 2

The uncertainty regularizer improves not only the variational energy but also the symmetry-even physical observables.

At N=14 the mean errors are:

| quantity | lambda=0 | lambda=1 | reduction |
|---|---:|---:|---:|
| energy | 9.769e-2 | 1.131e-2 | 8.64x |
| |Delta |mz|| | 3.477e-3 | 3.918e-4 | 8.87x |
| |Delta mz^2| | 6.472e-3 | 5.294e-4 | 12.22x |
| |Delta mx| | 2.010e-2 | 1.143e-3 | 17.59x |

Thus the energy improvement is accompanied by improved independent physical observables; it is not merely an improvement of the Rayleigh quotient.

## Control regime J/h = 1

The behavior is different. Energy remains comparable, but lambda=1 systematically worsens the symmetry-even observables.

At N=14:

| quantity | lambda=0 | lambda=1 |
|---|---:|---:|
| |Delta |mz|| | 1.248e-2 | 4.307e-2 |
| |Delta mz^2| | 1.248e-2 | 4.607e-2 |
| |Delta mx| | 2.771e-3 | 1.887e-2 |

The regularized errors are also highly reproducible across seeds. This is consistent with a systematic regularization/finite-training bias rather than merely seed noise. The upcoming Armijo experiment should test whether part of this behavior is fixed-step under-convergence.

## Z2 symmetry diagnostic

For the finite periodic TFIM, the exact ground state has <mz> approximately zero. The unconstrained GBT does not reliably preserve this symmetry.

At N=14, J/h=2:
- mean |Delta mz| = 0.3960 for lambda=0
- mean |Delta mz| = 0.7358 for lambda=1.

A particularly clear lambda=1 example has |Delta mz| = 0.9617 while the errors in |mz|, mz^2, and mx are only 4.20e-4, 5.83e-4, and 3.23e-4 respectively.

Therefore symmetry-even observables can be highly accurate while the learned state is strongly biased toward one magnetization sector. This is preserved as a baseline for the later thesis objective on incorporating physical symmetries into tree growth; it is not fixed at this stage.

## Interpretation frozen at this milestone

1. Uncertainty-aware regularization improves statistical robustness and physical-state quality in the difficult J/h=2 regime.
2. Fixed lambda=1 can introduce systematic bias in the easier J/h=1 regime that energy alone conceals.
3. Local/statistical update reliability and global symmetry preservation are distinct problems.
4. The next optimizer experiment is minimal Armijo versus fixed eta=0.05, keeping lambda=1 frozen and evaluating both energy and physical observables.
5. Symmetry enforcement is intentionally deferred to the dedicated symmetry investigation.
