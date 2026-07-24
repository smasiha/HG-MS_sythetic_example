# Controlled ULA/Gibbs and Ridge-CG Experiment

This package is a CPU-only, anonymous reproduction artifact for a controlled
bilevel problem with a non-singleton lower solution set. It tests three
mechanisms directly:

1. true unadjusted Langevin algorithm (ULA) sampling and hard best-of-\(N\)
   optimistic selection;
2. the ridge-CG approximation to the pseudoinverse hyper-gradient; and
3. end-to-end performance against four bilevel baselines under an equal
   single-thread wall-clock budget.

The committed artifact contains all 200 end-to-end run records, including
trajectories and oracle counts, plus all mechanism diagnostics. See
[`RESULTS.md`](RESULTS.md) for generated tables,
[`ALGORITHM_AUDIT.md`](ALGORITHM_AUDIT.md) for the update-by-update audit, and
[`results/MANIFEST.json`](results/MANIFEST.json) for checksums and result
counts.

## Exact Problem

Let \(\theta\in[-3,3]\), fix the ambient dimension \(d=20\), and vary the
intrinsic minimizer-manifold dimension over \(k\in\{1,2,4,10\}\). Write

\[
x=(y,z)\in\mathbb R^{k+1}\times\mathbb R^{d-k-1}.
\]

For constants
\[
(\kappa,\mu,\epsilon,A,m,\rho)=(1,1,0.1,0.5,4,0.3),
\]
define

\[
\begin{aligned}
R(\theta)&=\sqrt{1+\theta^2},\\
a(\theta)&=(\cos\theta,\sin\theta,0,\ldots,0)\in\mathbb R^d,\\
c(\theta)&=\frac{R(\theta)}{\sqrt{1+R(\theta)^2}},\\
u(\theta,x)&=
\frac{\langle x,a(\theta)\rangle}
{c(\theta)\sqrt{1+\|x\|^2}},\\
q_\theta(u)&=(1+\rho\theta)(u+1)
+A\sin^2\!\left(\frac{m\pi}{2}(u+1)\right).
\end{aligned}
\]

The lower and upper objectives are

\[
\begin{aligned}
g(\theta,x)
&=\kappa\left[
\sqrt{1+\left(\|y\|^2-R(\theta)^2\right)^2}-1
\right]+\frac{\mu}{2}\|z\|^2,\\
f(\theta,x)
&=2\sqrt{\epsilon^2+\|x\|^2}+q_\theta(u(\theta,x)).
\end{aligned}
\]

The lower minimizer set is the known \(k\)-sphere

\[
\mathcal S(\theta)=
\{(R(\theta)v,0):v\in\mathbb S^k\}.
\]

On this manifold \(u=\langle v,a(\theta)\rangle\). Since
\(1+\rho\theta\ge 0.1\) on the full domain, the unique optimistic lower
solution is

\[
x^\star(\theta)=(-R(\theta)a(\theta),0).
\]

It is non-degenerate on the manifold: its Riemannian Hessian has tangent
eigenvalue \(1+\rho\theta\ge0.1\). The exact optimistic hyper-objective,
hyper-gradient, and global solution are

\[
\begin{aligned}
F(\theta)&=2\sqrt{\epsilon^2+1+\theta^2},\\
F'(\theta)&=\frac{2\theta}{\sqrt{\epsilon^2+1+\theta^2}},\\
\theta^\star&=0,\qquad
F^\star=2\sqrt{1.01}.
\end{aligned}
\]

These formulas make the objective gap \(F(\widehat\theta)-F^\star\),
stationarity \(|F'(\widehat\theta)|\), optimistic-solution distance
\(\|\widehat x-x^\star(\widehat\theta)\|_2\), and exact hyper-gradient error
directly checkable.

## Methods

### HG-MS

Every outer update starts \(N=16\) independent candidates from the analyzed
\(\lambda\)-scaled Gaussian warm start

\[
X_i^0\sim\mathcal N((2v_i,0),\lambda I_{20}),
\qquad v_i\sim\operatorname{Unif}(\mathbb S^k),
\qquad \lambda=10^{-3}.
\]

It applies true ULA for \(K=100\) steps with \(h=0.01\):

\[
X_i^{t+1}=X_i^t-h\nabla_xg(\theta,X_i^t)
+\sqrt{2\lambda h}\,\xi_i^t,
\qquad \xi_i^t\sim\mathcal N(0,I_{20}).
\]

The candidate with the smallest \(f(\theta,\cdot)\) is selected. At that
point, matrix-free CG solves

\[
(\nabla_{xx}^2g+\gamma I)v=\nabla_xf,
\qquad \gamma=0.3,
\]

to absolute residual tolerance \(10^{-8}\), and the estimator is

\[
\widehat h
=\nabla_\theta f-\nabla_{\theta x}^2g\,v.
\]

The outer variable takes a projected gradient step on \([-3,3]\). Candidate
selection is hard upper-objective selection; no soft-selection surrogate is
used.

### Compared methods

- **HG-baseline:** the same ULA, ridge-CG, warm start, and outer update as
  HG-MS, but with one candidate.
- **V-PBGD:** jointly minimizes a scaled upper objective plus a lower
  value-function-gap penalty, using a stop-gradient auxiliary follower.
- **G-PBGD:** jointly minimizes a scaled upper objective plus
  \(\frac12\|\nabla_xg\|^2\).
- **IAPTT-GM:** differentiates through a lower-gradient trajectory and uses
  the pessimistic, maximum-upper-objective truncation point.

All implemented updates are written explicitly in
[`ALGORITHM_AUDIT.md`](ALGORITHM_AUDIT.md).

## Fairness Protocol

- Every end-to-end method receives exactly **two seconds** per seed and \(k\).
- The timer starts before lower-variable initialization; initialization is
  therefore included in the budget.
- Every process fixes all NumPy/BLAS thread counts to one. Each run is
  single-threaded, even when independent tasks execute concurrently.
- The held-out final seeds are exactly `1000`--`1009` for every method and
  every \(k\).
- All methods use the same initial \(\theta\) and paired first lower point for
  a given final seed.
- Method order is deterministically randomized within each \((k,\text{seed})\)
  task to reduce systematic order effects.
- Each method receives the same pilot search size: eight configurations,
  seeds `{11, 28, 47}`, and two seconds per configuration.
- Pilot selection uses validation objective gap only. Pilot and final seeds
  are disjoint.
- The \(k=10\) experiment uses the previously frozen pilot configuration
  selected at \(k=1\); it is not retuned.
- Final entries report mean \(\pm\) symmetric bootstrap 95% confidence
  interval half-width over ten seeds, using 5,000 resamples and bootstrap
  seed `82731`.

The fixed-time comparison is hardware-sensitive by construction. Reproduction
on a different CPU should preserve the protocol and qualitative ordering, but
may execute a different number of outer updates. Full timing curves and oracle
counts are included so that this tradeoff remains visible.

## Mechanism Diagnostics

Mechanism checks intentionally use controlled sample or iteration counts, not
wall-clock matching:

- `selection_error.csv` compares best-of-\(N\) empirical squared selection
  error with a finite-range Gauss--Jacobi quadrature reference for all four
  \(k\) values.
- `ridge_cg_error.csv` separates the exact pseudoinverse identity, ridge bias,
  and CG numerical solution error.
- `ula_gibbs_error.csv` compares ULA-selected candidates against independently
  sampled exact-Gibbs candidates using selection and hyper-gradient errors.

The observed slopes match the finite-range quadrature predictions and are
consistent with the sharper problem-specific \(N^{-2/k}\) behavior, improving
on the conservative \(N^{-1/k}\) upper bound proved in Lemma F.20.

The reported active-radial, angular, and embedding-normal Kolmogorov--Smirnov
distances are only one-dimensional marginal diagnostics. They do **not**
establish the full Rényi-divergence guarantee.

## Installation

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r synthetic_ula_ridge_cg/requirements.txt
```

No GPU, dataset download, network access, or Slurm installation is required.

## Reproduction Commands

All commands below run from the repository root.

### One-command smoke test

```bash
bash synthetic_ula_ridge_cg/scripts/smoke_test.sh
```

This runs the unit tests, all five methods with a short budget, quick
end-to-end and mechanism checks, and validation of the committed artifact.

### One-command full reproduction

```bash
JOBS=1 bash synthetic_ula_ridge_cg/scripts/reproduce_all.sh
```

This reruns the pilot, freezes its selections, executes all 200 final runs,
runs the mechanism diagnostics, and generates tables. `JOBS=1` is the safest
portable default. To reproduce the original concurrent layout, set `JOBS=10`
only when ten dedicated physical CPU cores are available:

```bash
JOBS=10 bash synthetic_ula_ridge_cg/scripts/reproduce_all.sh
```

To skip the pilot and use the committed frozen selection:

```bash
RERUN_PILOT=0 JOBS=10 \
  bash synthetic_ula_ridge_cg/scripts/reproduce_all.sh
```

### Pilot only

```bash
JOBS=5 bash synthetic_ula_ridge_cg/scripts/reproduce_pilot.sh
```

### Aggregate and generate tables

```bash
bash synthetic_ula_ridge_cg/scripts/aggregate.sh
```

The default reproduction root is
`synthetic_ula_ridge_cg/outputs/reproduction`; set `OUTPUT_BASE` to use a
different repository-relative or absolute output directory.

### Regenerate the committed public artifact

The deterministic exporter reads the two saved output trees, strips only
environment-identifying fields, verifies all remaining JSON data exactly,
regenerates every table from raw records, and validates checksums:

```bash
python -m synthetic_ula_ridge_cg.export_public_results \
  --source-fixed-d20 synthetic_ula_ridge_cg/outputs/fixed_d20 \
  --source-k10 synthetic_ula_ridge_cg/outputs/fixed_d20_k10 \
  --output-directory synthetic_ula_ridge_cg/results
```

Validate the committed artifact independently with:

```bash
python -m synthetic_ula_ridge_cg.validate_public_results
```

## Runtime and Hardware

The experiment is lightweight in memory (well below 1 GB) and CPU-only. The
fixed-time pilot and final benchmark account for about 640 CPU-seconds before
Python overhead and mechanism checks. A full run typically takes roughly
15--30 minutes on one recent CPU core and several minutes with ten dedicated
cores. Shared-machine load can increase runtime and alter fixed-time iteration
counts.

## Output Layout

```text
synthetic_ula_ridge_cg/
├── config/                 # final, isolated k=10, and pilot configurations
├── results/
│   ├── raw/                # 200 sanitized lossless run records
│   ├── pilot/              # frozen selection and five pilot summaries
│   ├── tables/             # per-seed, summary, paired, and timing CSV files
│   ├── mechanisms/         # selection, ridge-CG, and ULA/Gibbs diagnostics
│   ├── validation/         # source and public validation reports
│   └── MANIFEST.json       # versions, counts, seeds, and SHA-256 checksums
├── scripts/                # portable local workflows
├── slurm/                  # optional scheduler wrapper
└── tests/                  # unit and integration tests
```

Generated working files under `outputs/`, Python caches, scheduler logs, and
plots are intentionally excluded from version control.

## Limitations

- A fixed wall-clock benchmark cannot force identical iteration counts across
  CPUs; the raw trajectories enable time-aligned reanalysis.
- The mechanism diagnostics verify the predicted behavior on this controlled
  problem, not all assumptions of a general convergence theorem.
- Marginal KS distances diagnose selected one-dimensional distributions but
  do not prove a joint Rényi-divergence bound.
- The final table uses ten independent seeds, while each deterministic
  quadrature or fixed-seed mechanism study uses its stated controlled
  sample/iteration budget.
