# Controlled ULA/Gibbs and Ridge-CG Results

All end-to-end entries use ten held-out seeds and report mean ± symmetric bootstrap 95% CI half-width (5,000 resamples). Every cell contains objective gap / stationarity / optimistic-solution distance; lower is better.

| Method | k=1 | k=2 | k=4 | k=10 |
|---|---:|---:|---:|---:|
| **HG-MS** | **(1.24 ± 0.86)\(\times10^{-3}\) / 0.056 ± 0.026 / 0.238 ± 0.052** | **(2.04 ± 1.65)\(\times10^{-3}\) / 0.077 ± 0.027 / 0.524 ± 0.167** | **(4.73 ± 2.67)\(\times10^{-3}\) / 0.121 ± 0.040 / 0.875 ± 0.103** | **(7.96 ± 2.81)\(\times10^{-3}\) / 0.167 ± 0.036 / 1.103 ± 0.095** |
| HG-baseline | 0.049 ± 0.027 / 0.383 ± 0.116 / 1.168 ± 0.389 | 0.072 ± 0.032 / 0.485 ± 0.109 / 1.391 ± 0.361 | 0.045 ± 0.014 / 0.399 ± 0.068 / 1.454 ± 0.194 | 0.037 ± 0.006 / 0.375 ± 0.031 / 1.428 ± 0.164 |
| V-PBGD | 0.261 ± 0.151 / 0.746 ± 0.280 / 1.523 ± 0.459 | 0.220 ± 0.135 / 0.704 ± 0.255 / 1.440 ± 0.408 | 0.129 ± 0.066 / 0.589 ± 0.188 / 1.530 ± 0.201 | 0.102 ± 0.041 / 0.559 ± 0.140 / 1.509 ± 0.185 |
| G-PBGD | 1.76 ± 1.28 / 0.975 ± 0.481 / 2.123 ± 0.632 | 1.75 ± 1.29 / 0.959 ± 0.483 / 2.155 ± 0.548 | 1.75 ± 1.29 / 0.959 ± 0.483 / 2.155 ± 0.536 | 1.74 ± 1.29 / 0.945 ± 0.483 / 2.132 ± 0.536 |
| IAPTT-GM | 1.30 ± 1.29 / 0.637 ± 0.530 / 1.148 ± 0.884 | 1.31 ± 1.28 / 0.691 ± 0.499 / 1.123 ± 0.851 | 1.73 ± 1.29 / 0.825 ± 0.542 / 1.520 ± 0.949 | 1.73 ± 1.29 / 0.837 ± 0.530 / 1.462 ± 0.881 |

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
