TFIM uncertainty-regularization milestone

Frozen working configuration: lambda=1, eta=0.05, M=512, maximum tree depth=4, 150 epochs. No adaptive lambda or eta.

Scaling validation used N=8,10,12,14; J/h=1,2; five seeds per condition; lambda=0 versus lambda=1; no retuning with system size or coupling.

At J/h=2, lambda=1 beat lambda=0 in all 20 paired seed comparisons. Mean error per site for lambda=1 was 5.236348e-4, 6.682478e-4, 7.838275e-4, and 8.077655e-4 for N=8,10,12,14 respectively. Corresponding baseline values were 3.309356e-3, 2.168854e-2, 7.110231e-3, and 6.977862e-3.

At J/h=1, lambda=1 improved the mean at N=8,10,14 and was slightly worse at N=12. This supports a narrower interpretation: uncertainty regularization primarily improves reliability where finite-sample tree updates become unstable.

Lambda=1 is a practical fixed working choice, not a claimed universal optimum. Exact diagonalization and exact derivative diagnostics are validation tools only and do not affect production training. The current iid uncertainty estimate does not explicitly account for Metropolis autocorrelation or tree-selection uncertainty. Generality beyond TFIM remains to be tested.

Next stage: test generality rather than continue tuning TFIM.
