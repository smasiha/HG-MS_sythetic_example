# Controlled ULA/Gibbs and Ridge-CG Synthetic Experiment

This CPU-only experiment isolates two mechanisms used by HG-MS when the
lower-level solution is non-singleton:

1. true unadjusted Langevin algorithm (ULA) sampling followed by hard
   optimistic best-of-$N$ selection; and
2. a ridge-CG approximation to the pseudoinverse hyper-gradient.

Every analytical quantity needed to check the implementation is available in
closed form.

Reviewer-facing resources:

- [Complete numerical tables and mechanism diagnostics](RESULTS.md)
- [Intrinsic-Hessian proof](OPTIMISTIC_HESSIAN_PROOF.md)
- [Algorithm-to-code audit](ALGORITHM_AUDIT.md)
- [Machine-readable manifest and checksums](results/MANIFEST.json)

## Exact Controlled Problem

Let $\theta\in[-3,3]$, fix the ambient lower-variable dimension $d=20$,
and vary the intrinsic minimizer-manifold dimension over
$k\in\{1,2,4,10\}$. Write

$$
x=(y,z)\in\mathbb R^{k+1}\times\mathbb R^{d-k-1}
$$

and set

$$
(\kappa,\mu,\epsilon,A,m,\rho)=(1,1,0.1,0.5,4,0.3).
$$

Define

$$
\begin{aligned}
R(\theta)&=\sqrt{1+\theta^2},&
a(\theta)&=(\cos\theta,\sin\theta,0,\ldots,0)\in\mathbb R^{k+1},\\
c(\theta)&=\frac{R(\theta)}{\sqrt{1+R(\theta)^2}},&
u(\theta,x)&=
\frac{\langle y,a(\theta)\rangle}
{c(\theta)\sqrt{1+\lVert x\rVert^2}},\\
q_\theta(u)&=(1+\rho\theta)(u+1)
+A\sin^2\!\left(\frac{m\pi}{2}(u+1)\right).
\end{aligned}
$$

The lower and upper objectives are

$$
\begin{aligned}
g(\theta,x)
&=\kappa\!\left[
\sqrt{1+\bigl(\lVert y\rVert^2-R(\theta)^2\bigr)^2}-1
\right]+\frac{\mu}{2}\lVert z\rVert^2,\\
f(\theta,x)
&=2\sqrt{\epsilon^2+\lVert x\rVert^2}+q_\theta(u(\theta,x)).
\end{aligned}
$$

The lower minimizer set is the known $k$-sphere

$$
\mathcal S(\theta)
=\{(R(\theta)v,0):v\in\mathbb S^k\}.
$$

On this manifold, $u=\langle v,a(\theta)\rangle$. Because
$1+\rho\theta\geq0.1$ over the full domain, the optimistic lower solution
is unique:

$$
x^\star(\theta)=(-R(\theta)a(\theta),0).
$$

With the metric induced by the ambient Euclidean space and any orthonormal
tangent basis, its intrinsic Hessian is

$$
\mathrm{Hess}_{\mathcal S(\theta)}f(x^\star(\theta))
=\frac{1+0.3\theta}{1+\theta^2}I_k
\succeq0.01I_k.
$$

The [complete unit-speed-geodesic proof](OPTIMISTIC_HESSIAN_PROOF.md)
explains both the factor $1/(1+\theta^2)$ and the uniform lower bound.

The exact optimistic hyper-objective, hyper-gradient, optimizer, and optimum
are

$$
\begin{aligned}
F(\theta)&=2\sqrt{\epsilon^2+1+\theta^2},&
F'(\theta)&=\frac{2\theta}{\sqrt{\epsilon^2+1+\theta^2}},\\
\theta^\star&=0,&
F^\star&=2\sqrt{1.01}.
\end{aligned}
$$

Consequently, all reported quantities are exact:

$$
F(\widehat\theta)-F^\star,\qquad
\lvert F'(\widehat\theta)\rvert,\qquad
\lVert\widehat x-x^\star(\widehat\theta)\rVert_2.
$$

## HG-MS Implementation

At every outer update, HG-MS initializes $N=16$ independent branches with
the analyzed $\lambda$-scaled Gaussian warm start

$$
X_i^0\sim\mathcal N((2v_i,0),\lambda I_{20}),\qquad
v_i\sim\mathrm{Unif}(\mathbb S^k),\qquad \lambda=10^{-3}.
$$

The associated Gibbs law is

$$
\pi_\lambda(dx\mid\theta)\propto
\exp[-g(\theta,x)/\lambda]\,dx.
$$

HG-MS applies true ULA for $K=100$ steps with $h=0.01$:

$$
X_i^{t+1}
=X_i^t-h\nabla_xg(\theta,X_i^t)
+\sqrt{2\lambda h}\,\xi_i^t,\qquad
\xi_i^t\sim\mathcal N(0,I_{20}).
$$

Hard optimistic selection returns

$$
\widehat x=\mathop{\mathrm{arg\,min}}_{1\leq i\leq N}
f(\theta,X_i^K).
$$

At the selected point, matrix-free conjugate gradient starts at zero and
solves

$$
\bigl(\nabla_{xx}^2g(\theta,\widehat x)+\gamma I\bigr)v
=\nabla_x f(\theta,\widehat x),\qquad \gamma=0.3,
$$

to absolute residual tolerance $10^{-8}$, with at most 64 iterations. The
resulting hyper-gradient estimator and projected outer update are

$$
\widehat h
=\nabla_\theta f(\theta,\widehat x)
-\nabla_{\theta x}^2g(\theta,\widehat x)v,\qquad
\theta^+=\mathrm{proj}_{[-3,3]}(\theta-\alpha\widehat h).
$$

HG-baseline uses the identical ULA, ridge-CG, warm start, and outer update
with $N=1$.

## Compared Updates

V-PBGD uses a stop-gradient auxiliary follower $\widehat\omega$ and jointly
updates $(\theta,x)$ on

$$
\Phi_V(\theta,x;\widehat\omega)
=s_f f(\theta,x)+s_g
\bigl[g(\theta,x)-g(\theta,\widehat\omega)\bigr],
\quad s_f=\min(1,\beta^{-1}),\quad s_g=\min(1,\beta).
$$

G-PBGD jointly updates $(\theta,x)$ on

$$
\Phi_G(\theta,x)
=s_f f(\theta,x)+\frac{s_g}{2}\lVert\nabla_xg(\theta,x)\rVert^2.
$$

IAPTT-GM differentiates exactly through

$$
x_{t+1}=x_t-\eta\nabla_xg(\theta,x_t)
$$

and uses the pessimistic truncation

$$
\bar t=\mathop{\mathrm{arg\,max}}_{1\leq t\leq K}f(\theta,x_t).
$$

The [algorithm audit](ALGORITHM_AUDIT.md) gives the complete update-by-update
mapping and the frozen constants.

## Fairness and Statistics

- Each method receives 60 single-thread CPU seconds for every
  $(k,\mathrm{seed})$ pair, including initialization.
- Final evaluation uses seeds $\{1000,\ldots,1004\}$ for every method and
  every $k$.
- Configurations are frozen before final evaluation using eight pilot
  configurations per method, disjoint pilot seeds $\{11,28,47\}$, and
  validation objective gap only.
- The $k=10$ runs reuse the same frozen configuration and are not retuned.
- Independent tasks may run concurrently, but every method process is
  single-threaded.
- Uncertainty is mean $\pm$ symmetric bootstrap 95% CI half-width from
  5,000 resamples with bootstrap seed `82731`.

## Results

The complete website-renderable tables are in
[`RESULTS.md`](RESULTS.md). They include:

- objective gap, stationarity, and optimistic-solution distance for all five
  methods and all four manifold dimensions;
- observed and quadrature best-of-$N$ scaling slopes;
- pseudoinverse, ridge-bias, and CG numerical errors;
- ULA-versus-exact-Gibbs hyper-gradient MSE; and
- marginal KS diagnostics with their stated limitation.

HG-MS has the lowest sample mean in all 12 reported
metric-by-dimension cells. Complete trajectories and oracle counts are
retained under [`results/raw/`](results/raw/), and paired seed-level
comparisons are in
[`results/tables/paired_comparisons.csv`](results/tables/paired_comparisons.csv).

## Reproduction

From the repository root:

```bash
python -m pip install -r synthetic_ula_ridge_cg/requirements.txt
bash synthetic_ula_ridge_cg/scripts/smoke_test.sh
RERUN_PILOT=0 JOBS=1 \
  bash synthetic_ula_ridge_cg/scripts/reproduce_all.sh
```

The complete final protocol requires 6,000 CPU-seconds before
mechanism-check overhead. Parallelism changes elapsed reproduction time but
not the single-thread per-method budget.
