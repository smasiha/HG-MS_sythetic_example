# Algorithm Audit

This document maps each runnable implementation to its mathematical update and
lists the independent checks used to guard the controlled comparison.

## Controlled Landscape

For \(\theta\in[-3,3]\), \(d=20\), and
\(k\in\{1,2,4,10\}\), write
\(x=(y,z)\in\mathbb R^{k+1}\times\mathbb R^{d-k-1}\). Define

\[
\begin{aligned}
R(\theta)&=\sqrt{1+\theta^2},\\
a(\theta)&=(\cos\theta,\sin\theta,0,\ldots,0),\\
c(\theta)&=R(\theta)/\sqrt{1+R(\theta)^2},\\
u(\theta,x)&=\langle x,a(\theta)\rangle/
\bigl(c(\theta)\sqrt{1+\|x\|^2}\bigr),\\
q_\theta(u)&=(1+0.3\theta)(u+1)
+0.5\sin^2(2\pi(u+1)),\\
g(\theta,x)&=\sqrt{1+(\|y\|^2-R(\theta)^2)^2}-1
+\tfrac12\|z\|^2,\\
f(\theta,x)&=2\sqrt{0.1^2+\|x\|^2}+q_\theta(u(\theta,x)).
\end{aligned}
\]

The exact lower solution manifold, unique non-degenerate optimistic solution,
hyper-objective, and hyper-gradient are

\[
\begin{aligned}
\mathcal S(\theta)&=\{(R(\theta)v,0):v\in\mathbb S^k\},\\
x^\star(\theta)&=(-R(\theta)a(\theta),0),\\
F(\theta)&=2\sqrt{0.1^2+1+\theta^2},\\
F'(\theta)&=2\theta/\sqrt{0.1^2+1+\theta^2}.
\end{aligned}
\]

Thus \(\theta^\star=0\), \(F^\star=2\sqrt{1.01}\), and the tangent
Riemannian Hessian at \(x^\star(\theta)\) has eigenvalue
\(1+0.3\theta\ge0.1\). In the lower Hessian there are exactly \(k\) tangent
zero modes, one positive radial mode, and \(d-k-1\) positive
embedding-normal modes.

## Shared Initialization and Timing

For every final seed, all methods share the same deterministic initial
\(\theta\). Their first lower point is paired through a method-independent
initialization seed. HG-MS candidate zero and the HG-baseline candidate are
also paired.

For HG methods, branch \(i\) starts each outer iteration from

\[
X_i^0\sim\mathcal N((2v_i,0),\lambda I_{20}),
\qquad v_i\sim\operatorname{Unif}(\mathbb S^k),
\qquad\lambda=10^{-3}.
\]

The timer starts before this lower initialization. Each end-to-end method then
receives exactly two single-thread wall-clock seconds. The process-level
method order is deterministically permuted for each \((k,\text{seed})\).

## HG-MS and HG-Baseline

`methods.ula_step` applies true ULA:

\[
X_i^{t+1}=X_i^t-h\nabla_xg(\theta,X_i^t)
+\sqrt{2\lambda h}\,\xi_i^t,
\qquad \xi_i^t\sim\mathcal N(0,I_{20}).
\]

The frozen values are \(h=0.01\), \(\lambda=10^{-3}\), and \(K=100\).
HG-MS uses \(N=16\) independent branches and selects

\[
\widehat x=\operatorname*{arg\,min}_{1\le i\le N}f(\theta,X_i^K).
\]

HG-baseline executes the identical path with \(N=1\).

At \(\widehat x\), `methods.ridge_cg_hypergradient` uses only
Hessian-vector products to solve

\[
(\nabla_{xx}^2g(\theta,\widehat x)+\gamma I)v
=\nabla_xf(\theta,\widehat x),
\qquad \gamma=0.3,
\]

with a zero CG initial vector, maximum 64 iterations, and absolute residual
tolerance \(10^{-8}\). It returns

\[
\widehat h
=\nabla_\theta f(\theta,\widehat x)
-\nabla_{\theta x}^2g(\theta,\widehat x)v,
\qquad
\theta^+=\Pi_{[-3,3]}(\theta-\alpha\widehat h).
\]

The implementation records CG iterations, matrix-vector products, absolute
and relative residuals, breakdowns, and tolerance failures.

## V-PBGD

Let \(\beta>0\) be the penalty,
\(s_f=\min(1,\beta^{-1})\), and
\(s_g=\min(1,\beta)\). At each outer iteration an auxiliary follower starts
from the current lower point and takes the configured lower-gradient steps.
No gradient is propagated through those auxiliary steps. The joint surrogate
is

\[
\Phi_V(\theta,x;\widehat\omega)
=s_f f(\theta,x)
+s_g\bigl[g(\theta,x)-g(\theta,\widehat\omega)\bigr].
\]

`methods.v_pbgd_objective_and_gradients` evaluates the analytical
stop-gradient derivatives of this surrogate. Plain gradient descent updates
\(\theta\) and \(x\), and only \(\theta\) is projected.

## G-PBGD

With the same stable scaling, the implemented surrogate is

\[
\Phi_G(\theta,x)
=s_f f(\theta,x)+\frac{s_g}{2}\|\nabla_xg(\theta,x)\|^2.
\]

Its gradients are

\[
\begin{aligned}
\nabla_\theta\Phi_G
&=s_f\nabla_\theta f
+s_g\langle\nabla_{\theta x}^2g,\nabla_xg\rangle,\\
\nabla_x\Phi_G
&=s_f\nabla_x f
+s_g\nabla_{xx}^2g\,\nabla_xg.
\end{aligned}
\]

`methods.g_pbgd_objective_and_gradients` evaluates these expressions
directly. Plain gradient descent jointly updates \(\theta\) and \(x\).

## IAPTT-GM

Starting from a learned initialization \(x_0\), the implementation unrolls

\[
x_{t+1}=x_t-\eta\nabla_xg(\theta,x_t),
\qquad t=0,\ldots,K-1.
\]

It chooses the pessimistic truncation

\[
\bar t=\operatorname*{arg\,max}_{1\le t\le K}f(\theta,x_t)
\]

and differentiates \(f(\theta,x_{\bar t})\) exactly through the selected
trajectory. Plain gradient descent updates both \(\theta\) and \(x_0\), with
projection only on \(\theta\).

## Pilot and Final Separation

`config/pilot_grid.json` contains exactly eight configurations per method.
Every candidate receives two seconds on seeds `{11, 28, 47}`. Selection uses
only the geometric mean validation objective gap (equivalently, mean
log-objective-gap score). Final evaluation uses the disjoint held-out seeds
`1000`--`1009`. The isolated `config/k10_config.json` changes only the value
of \(k\) and reuses the previously frozen selected configuration without
retuning.

## Mechanism Definitions

- **Pseudoinverse error** is
  \[
  \left|
  \nabla_\theta f-\nabla_{\theta x}^2g
  (\nabla_{xx}^2g)^\dagger\nabla_xf-F'(\theta)
  \right|
  \]
  at an exact optimistic manifold point.
- **Ridge bias** is the absolute difference between the dense ridge
  hyper-gradient and the exact-pseudoinverse hyper-gradient at the same
  point.
- **CG solution error** is
  \(\|v_{\rm CG}-v_{\rm dense}\|_2\) for the same ridge system at a rotated,
  non-optimistic manifold point.
- **ULA-versus-exact-Gibbs MSE difference** is
  \[
  \frac{|\operatorname{MSE}_{\rm ULA}
  -\operatorname{MSE}_{\rm Gibbs}|}
  {\operatorname{MSE}_{\rm Gibbs}},
  \]
  where both MSEs compare the corresponding selected-point hyper-gradient
  estimator to the exact \(F'(\theta)\).
- **Marginal KS distances** compare the active radial, angular, and
  embedding-normal one-dimensional marginals. They are diagnostics only and
  do not verify the full joint Rényi-divergence guarantee.

The best-of-\(N\) diagnostic compares Monte Carlo selection error against
Gauss--Jacobi quadrature. The observed slopes match the finite-range
quadrature predictions and are consistent with the sharper problem-specific
\(N^{-2/k}\) behavior, improving on the conservative \(N^{-1/k}\) upper bound
proved in Lemma F.20.

## Verification Matrix

`tests/test_synthetic.py` independently checks:

| Component | Check |
|---|---|
| Problem geometry | Fixed ambient dimension, manifold nullity, and unique restricted optimistic point |
| Analytical derivatives | Finite differences for all first derivatives, Hessian-vector products, Hessian, and mixed derivative |
| Exact quantities | Closed-form \(F\), \(F'\), \(\theta^\star\), and dense pseudoinverse identity |
| ULA | Exact equality with the stated update under fixed Gaussian noise |
| Warm start | Independent reconstruction of the scaled-Gaussian law and branch-prefix pairing |
| Exact Gibbs | Active radial factorization and normal Gaussian variance |
| Selection | Independent spherical moment and quadrature checks |
| Ridge-CG | Agreement with dense solve and absolute-residual stopping behavior |
| V-PBGD | Finite differences of the stop-gradient value-gap surrogate |
| G-PBGD | Finite differences of the stationarity-penalty surrogate |
| IAPTT-GM | Finite differences through an independently written selected unroll |
| Run cache | Rejection of stale schemas and mismatched scientific metadata |

`tests/test_public_artifact.py` checks lossless sanitization and the frozen
scientific protocol. `scripts/smoke_test.sh` adds integration coverage for all
five algorithms, quick mechanism diagnostics, and the complete committed
result validator.
