# Predicted-versus-observed mechanism checks

The theorem rates are upper-bound shapes with problem-dependent constants. The table compares empirical slopes with high-accuracy finite-range quadrature references and does not fit constants on the test points.

## Best-of-N selection

| k | Observed squared-error slope | Quadrature-reference slope | Sharp asymptotic slope | Theorem-envelope slope |
|---:|---:|---:|---:|---:|
| 1 | -2.016 | -1.987 | -2.000 | -1.000 |
| 2 | -1.016 | -1.000 | -1.000 | -0.500 |
| 4 | -0.599 | -0.598 | -0.500 | -0.250 |
| 10 | -0.158 | -0.156 | -0.200 | -0.100 |

The observed slopes match the finite-range quadrature predictions and are consistent with the sharper problem-specific $N^{-2/k}$ behavior, improving on the conservative $N^{-1/k}$ upper bound proved in Lemma F.20.

## Pseudoinverse, ridge, and CG

- Maximum exact-pseudoinverse formula error: `6.661e-16`.
- Observed small-ridge bias slope at theta=1: `0.993` (first-order prediction: `1`).
- Mean CG solution error on rotated non-optimistic manifold points: `5.746e+01` at `eta=1`, versus `4.446e-12` at `eta=1e-8`.
- Ridge-CG breakdowns: `0`; ULA-selected-point CG breakdowns: `0`.

## ULA/Gibbs diagnostics

| Init | lambda | h | K | Radial KS | Angular KS | Normal KS | Selection excess | ULA HG MSE | Gibbs HG MSE | Bound-shape proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random-centers | 1e-02 | 0.01 | 10 | 0.996 | 0.014 | 0.008 | 1.652e-01 | 8.643e-01 | 7.251e-01 | 7.034e+00 |
| single-center | 1e-02 | 0.01 | 10 | 0.996 | 0.989 | 0.010 | 8.540e+00 | 2.154e-02 | 7.251e-01 | 7.033e+00 |
| random-centers | 1e-02 | 0.01 | 100 | 0.014 | 0.012 | 0.009 | 1.412e-02 | 6.639e-01 | 7.251e-01 | 1.479e+00 |
| single-center | 1e-02 | 0.01 | 100 | 0.010 | 0.966 | 0.008 | 7.221e+00 | 2.299e-01 | 7.251e-01 | 6.861e+00 |
| random-centers | 1e-02 | 0.01 | 500 | 0.011 | 0.009 | 0.016 | 6.244e-04 | 9.118e-01 | 7.251e-01 | 1.488e+00 |
| single-center | 1e-02 | 0.01 | 500 | 0.017 | 0.883 | 0.007 | 7.182e+00 | 2.938e-01 | 7.251e-01 | 6.394e+00 |
| random-centers | 1e-03 | 0.01 | 10 | 1.000 | 0.006 | 0.014 | 1.836e-01 | 8.292e-01 | 6.693e-01 | 5.906e+01 |
| single-center | 1e-03 | 0.01 | 10 | 1.000 | 0.999 | 0.016 | 9.120e+00 | 1.758e-02 | 6.693e-01 | 5.906e+01 |
| random-centers | 1e-03 | 0.01 | 100 | 0.009 | 0.009 | 0.008 | 3.217e-02 | 6.331e-01 | 6.693e-01 | 2.999e+00 |
| single-center | 1e-03 | 0.01 | 100 | 0.013 | 0.996 | 0.010 | 7.580e+00 | 3.472e-01 | 6.693e-01 | 5.882e+01 |
| random-centers | 1e-03 | 0.01 | 500 | 0.011 | 0.007 | 0.012 | 3.653e-02 | 7.151e-01 | 6.693e-01 | 3.188e+00 |
| single-center | 1e-03 | 0.01 | 500 | 0.012 | 0.983 | 0.015 | 7.562e+00 | 3.428e-01 | 6.693e-01 | 5.810e+01 |

The KS values are one-dimensional marginal diagnostics; they do not establish the full R\'enyi-divergence guarantee. The bound-shape proxy omits unknown constants and is used only for qualitative ordering, not as a calibrated numerical upper bound.
