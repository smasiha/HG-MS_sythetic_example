# Controlled ULA/Gibbs and Ridge-CG Results

All end-to-end entries use 60 single-thread CPU seconds and five held-out seeds (`1000`--`1004`). Values are mean ± symmetric bootstrap 95% CI half-width (5,000 resamples). Every cell contains objective gap / stationarity / optimistic-solution distance; lower is better.

| Method | k=1 | k=2 | k=4 | k=10 |
|---|---:|---:|---:|---:|
| **HG-MS** | **(2.95 ± 2.88)\(\times10^{-3}\) / 0.078 ± 0.064 / 0.254 ± 0.173** | **(0.94 ± 0.93)\(\times10^{-3}\) / 0.046 ± 0.033 / 0.728 ± 0.203** | **(2.92 ± 2.55)\(\times10^{-3}\) / 0.089 ± 0.052 / 0.869 ± 0.167** | **0.011 ± 0.004 / 0.199 ± 0.037 / 1.083 ± 0.144** |
| HG-baseline | 0.056 ± 0.050 / 0.369 ± 0.234 / 1.282 ± 0.620 | 0.047 ± 0.026 / 0.408 ± 0.094 / 1.557 ± 0.354 | 0.052 ± 0.024 / 0.429 ± 0.103 / 1.654 ± 0.243 | 0.025 ± 0.008 / 0.305 ± 0.050 / 1.205 ± 0.167 |
| V-PBGD | 0.152 ± 0.072 / 0.703 ± 0.164 / 1.828 ± 0.316 | 0.116 ± 0.056 / 0.622 ± 0.126 / 1.670 ± 0.238 | 0.096 ± 0.031 / 0.584 ± 0.088 / 1.591 ± 0.160 | 0.096 ± 0.031 / 0.584 ± 0.088 / 1.591 ± 0.160 |
| G-PBGD | 1.76 ± 1.71 / 1.019 ± 0.615 / 2.286 ± 0.660 | 1.75 ± 1.72 / 0.959 ± 0.644 / 2.159 ± 0.721 | 1.75 ± 1.72 / 0.959 ± 0.644 / 2.159 ± 0.721 | 1.75 ± 1.72 / 0.959 ± 0.644 / 2.176 ± 0.704 |
| IAPTT-GM | 1.73 ± 1.73 / 0.837 ± 0.706 / 1.463 ± 1.136 | 1.73 ± 1.73 / 0.837 ± 0.706 / 1.340 ± 1.011 | 1.73 ± 1.73 / 0.837 ± 0.706 / 1.461 ± 1.135 | 2.59 ± 1.73 / 1.190 ± 0.706 / 1.962 ± 1.132 |

HG-MS has the lowest mean in 12 of 12 end-to-end cells. Paired seed-level differences are available in [`tables/paired_comparisons.csv`](results/tables/paired_comparisons.csv).

## Best-of-N Dimension Check

| k | Observed slope | Quadrature slope | Sharp asymptotic | Conservative theorem envelope |
|---:|---:|---:|---:|---:|
| 1 | -2.016 | -1.987 | -2.000 | -1.000 |
| 2 | -1.016 | -1.000 | -1.000 | -0.500 |
| 4 | -0.599 | -0.598 | -0.500 | -0.250 |
| 10 | -0.158 | -0.156 | -0.200 | -0.100 |

The observed slopes match the finite-range quadrature predictions and are consistent with the sharper problem-specific \(N^{-2/k}\) behavior, improving on the conservative \(N^{-1/k}\) upper bound proved in Lemma F.20.

## Hyper-gradient Mechanism Checks

- **Pseudoinverse error** is \(\left|\nabla_\theta f-\nabla_{\theta x}^2g(\nabla_{xx}^2g)^\dagger\nabla_xf-F'(\theta)\right|\) at the exact optimistic manifold points. Its maximum is `6.66e-16`.
- **Ridge bias** is the absolute difference between the dense ridge hyper-gradient and the exact-pseudoinverse hyper-gradient. Its observed log-log slope is `0.993` (first-order prediction: `1`).
- **CG solution error** is \(\|v_{\mathrm{CG}}-v_{\mathrm{dense}}\|_2\) for the same ridge system at rotated non-optimistic manifold points. Its mean at absolute tolerance `1e-8` is `4.45e-12`.
- At `k=2`, `theta=1`, `lambda=1e-3`, `h=0.01`, `K=100`, and `N=32`, the ULA-selected and independently exact-Gibbs-selected hyper-gradient MSEs are `0.633` and `0.669`. Their relative difference is `5.4%`.
- Active-radial/angular/normal marginal KS distances are `0.009` / `0.009` / `0.008`. These marginal diagnostics do **not** establish the full R\'enyi-divergence guarantee.

The complete records, trajectories, oracle counts, timing curves, and diagnostic CSV files are included in this directory. Mechanism diagnostics use controlled sample/iteration budgets, not wall-clock matching.
