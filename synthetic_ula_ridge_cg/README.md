# Controlled ULA/Gibbs and Ridge-CG Experiment

This anonymous, CPU-only package studies bilevel optimization with a known
non-singleton lower solution set. It directly tests:

1. true unadjusted Langevin algorithm (ULA) sampling with hard optimistic
   best-of-\(N\) selection;
2. ridge-CG approximation of the pseudoinverse hyper-gradient; and
3. HG-MS against HG-baseline, V-PBGD, G-PBGD, and IAPTT-GM under the same
   60-second single-thread wall-clock budget.

The committed artifact contains all 100 end-to-end run records
(4 manifold dimensions × 5 seeds × 5 methods), complete trajectories, oracle
counts, mechanism diagnostics, generated CSV and Markdown tables, and
checksums.
The most convenient reviewer-facing entry points are:

- [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md): standalone formulations,
  protocol, generated tables, and interpretation;
- [`OPTIMISTIC_HESSIAN_PROOF.md`](OPTIMISTIC_HESSIAN_PROOF.md): explicit
  geodesic proof of the optimistic point's intrinsic Hessian bound;
- [`RESULTS.md`](RESULTS.md): compact generated results;
- [`ALGORITHM_AUDIT.md`](ALGORITHM_AUDIT.md): each mathematical update mapped
  to its implementation;
- [`results/MANIFEST.json`](results/MANIFEST.json): configurations, counts,
  seeds, and SHA-256 checksums.

## Exact Problem

Let \(\theta\in[-3,3]\), fix the ambient dimension \(d=20\), and vary the
intrinsic minimizer-manifold dimension over \(k\in\{1,2,4,10\}\). Write

\[
x=(y,z)\in\mathbb R^{k+1}\times\mathbb R^{d-k-1}.
\]

For
\[
(\kappa,\mu,\epsilon,A,m,\rho)=(1,1,0.1,0.5,4,0.3),
\]
define

\[
\begin{aligned}
R(\theta)&=\sqrt{1+\theta^2},\\
a(\theta)&=(\cos\theta,\sin\theta,0,\ldots,0)\in\mathbb R^{k+1},\\
c(\theta)&=\frac{R(\theta)}{\sqrt{1+R(\theta)^2}},\\
u(\theta,x)&=
\frac{\langle y,a(\theta)\rangle}
{c(\theta)\sqrt{1+\|x\|^2}},\\
q_\theta(u)&=(1+\rho\theta)(u+1)
+A\sin^2\!\left(\frac{m\pi}{2}(u+1)\right),
\end{aligned}
\]

and

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

The known lower minimizer manifold and unique optimistic solution are

\[
\mathcal S(\theta)=
\{(R(\theta)v,0):v\in\mathbb S^k\},
\qquad
x^\star(\theta)=(-R(\theta)a(\theta),0).
\]

With the ambient-induced metric, the optimistic point is intrinsically
non-degenerate because, in any orthonormal tangent basis,
\[
\operatorname{Hess}_{\mathcal S(\theta)}f(x^\star(\theta))
=\frac{1+0.3\theta}{1+\theta^2}I_k
\succeq0.01I_k.
\]
The complete geodesic calculation is in
[`OPTIMISTIC_HESSIAN_PROOF.md`](OPTIMISTIC_HESSIAN_PROOF.md). The exact
hyper-objective, hyper-gradient, optimizer, and optimum are

\[
\begin{aligned}
F(\theta)&=2\sqrt{\epsilon^2+1+\theta^2},\\
F'(\theta)&=\frac{2\theta}{\sqrt{\epsilon^2+1+\theta^2}},\\
\theta^\star&=0,\qquad F^\star=2\sqrt{1.01}.
\end{aligned}
\]

Thus objective gap \(F(\widehat\theta)-F^\star\), stationarity
\(|F'(\widehat\theta)|\), and optimistic-solution distance
\(\|\widehat x-x^\star(\widehat\theta)\|_2\) are exactly checkable.

## HG-MS

At every outer update, the \(N=16\) branches use the analyzed
\(\lambda\)-scaled Gaussian warm start

\[
X_i^0\sim\mathcal N((2v_i,0),\lambda I_{20}),
\qquad
v_i\sim\operatorname{Unif}(\mathbb S^k),
\qquad \lambda=10^{-3}.
\]

The target Gibbs law is
\(\pi_\lambda(dx\mid\theta)\propto
\exp[-g(\theta,x)/\lambda]\,dx\). True ULA runs for \(K=100\) steps with
\(h=0.01\):

\[
X_i^{t+1}=X_i^t-h\nabla_xg(\theta,X_i^t)
+\sqrt{2\lambda h}\,\xi_i^t,
\qquad \xi_i^t\sim\mathcal N(0,I_{20}).
\]

HG-MS selects the branch minimizing \(f(\theta,\cdot)\). At that point,
matrix-free CG solves

\[
(\nabla_{xx}^2g+\gamma I)v=\nabla_xf,
\qquad \gamma=0.3,
\]

to absolute residual tolerance \(10^{-8}\), and returns

\[
\widehat h
=\nabla_\theta f-\nabla_{\theta x}^2g\,v.
\]

The outer variable takes a projected gradient step on \([-3,3]\). HG-baseline
uses the identical path with \(N=1\). The other four update definitions and
their exact code locations are given in
[`ALGORITHM_AUDIT.md`](ALGORITHM_AUDIT.md).

## Fairness Protocol

- Every final method receives exactly **60 single-thread CPU seconds** per
  seed and \(k\), including initialization.
- Final seeds are `1000`--`1004` for every method and all four \(k\) values.
- Method configurations are frozen before final evaluation. The original
  lightweight pilot evaluates eight configurations per method for two seconds
  on disjoint seeds `{11, 28, 47}` and selects by validation objective gap
  only. No final seed is used for selection.
- The \(k=10\) experiment uses the same frozen configuration; it is not
  retuned.
- Independent tasks may run concurrently, but every method process fixes
  NumPy/BLAS thread counts to one.
- Results use the final iterate and report mean ± symmetric bootstrap 95% CI
  half-width from 5,000 resamples (bootstrap seed `82731`).

Fixed-wall-clock iteration counts are hardware-sensitive. The protocol and
qualitative comparison should reproduce, while exact iteration counts may
differ across CPUs. Complete trajectories and oracle counts make this
tradeoff auditable.

## Mechanism Diagnostics

Mechanism checks use controlled sample or iteration counts rather than
wall-clock matching:

- `selection_error.csv` compares empirical best-of-\(N\) squared selection
  error with finite-range Gauss--Jacobi quadrature for every \(k\).
- `ridge_cg_error.csv` separates pseudoinverse formula error, ridge bias, and
  CG numerical solution error.
- `ula_gibbs_error.csv` compares ULA-selected and independently exact-Gibbs-
  selected candidates and hyper-gradients.

The observed slopes match the finite-range quadrature predictions and are
consistent with the sharper problem-specific \(N^{-2/k}\) behavior, improving
on the conservative \(N^{-1/k}\) upper bound proved in Lemma F.20. The
reported marginal KS distances are diagnostics only; they do not establish
the full Rényi-divergence guarantee.

## Installation

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r synthetic_ula_ridge_cg/requirements.txt
```

No GPU, dataset download, network access, or Slurm installation is required.

## Reproduction

### One-command smoke test

```bash
bash synthetic_ula_ridge_cg/scripts/smoke_test.sh
```

This runs unit tests, all five methods with a short integration budget, quick
mechanism checks, and full validation of the committed artifact.

### One-command full reproduction

Use the committed, frozen pilot selection:

```bash
RERUN_PILOT=0 JOBS=1 \
  bash synthetic_ula_ridge_cg/scripts/reproduce_all.sh
```

`JOBS=1` is the safest portable default. Use `JOBS=20` only with 20 dedicated
physical CPU cores. To rerun the disjoint pilot first, set `RERUN_PILOT=1`.

### One-command aggregation and table generation

```bash
bash synthetic_ula_ridge_cg/scripts/aggregate.sh
```

This writes Markdown and CSV tables under
`synthetic_ula_ridge_cg/outputs/reproduction/report/`.

### Deterministic public export

The exporter removes only environment-identifying metadata, verifies exact
numerical equality before and after sanitization, regenerates every table from
raw records, and builds the checksum manifest:

```bash
python -m synthetic_ula_ridge_cg.export_public_results \
  --source-final synthetic_ula_ridge_cg/outputs/reproduction \
  --source-pilot synthetic_ula_ridge_cg/outputs/reproduction/pilot \
  --source-mechanisms synthetic_ula_ridge_cg/outputs/reproduction/mechanisms \
  --source-k10-selection synthetic_ula_ridge_cg/outputs/reproduction/mechanisms \
  --output-directory synthetic_ula_ridge_cg/results
```

Validate the committed public artifact independently:

```bash
python -m synthetic_ula_ridge_cg.validate_public_results
```

### Reviewer-facing Markdown

The website-renderable report, proof, and generated results are:

- [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md);
- [`OPTIMISTIC_HESSIAN_PROOF.md`](OPTIMISTIC_HESSIAN_PROOF.md);
- [`RESULTS.md`](RESULTS.md).

The numerical tables in `RESULTS.md` are generated directly from the source
CSVs; they are not hand-copied.

## Runtime and Layout

The final benchmark uses 6,000 CPU-seconds (100 method runs × 60 seconds):
about 100 minutes serially or roughly five minutes with 20 dedicated cores,
plus mechanism checks and Python overhead. Memory use is below 1 GB.

```text
synthetic_ula_ridge_cg/
├── EXPERIMENT_REPORT.md    # standalone reviewer-facing report
├── OPTIMISTIC_HESSIAN_PROOF.md
│                             # reusable intrinsic-Hessian proof
├── config/                 # final 60-second and pilot configurations
├── results/
│   ├── raw/                # 100 sanitized lossless records + trajectories
│   ├── pilot/              # frozen selection and pilot summaries
│   ├── tables/             # per-seed, summary, paired, and timing CSVs
│   ├── mechanisms/         # selection, ridge-CG, and ULA/Gibbs diagnostics
│   ├── validation/         # source and public validation reports
│   └── MANIFEST.json       # versions, counts, and SHA-256 checksums
├── scripts/                # portable CPU workflows
├── slurm/                  # optional scheduler wrapper
└── tests/                  # unit and integration tests
```

Generated working outputs, caches, Slurm logs, and plots are excluded from
version control.

## Limitations

- A wall-clock comparison cannot force identical iteration counts on different
  CPUs.
- Five seeds provide less precise uncertainty estimates than a larger study;
  all per-seed records are included.
- Constant-step stochastic methods need not improve monotonically at every
  logged time; the table uses the pre-specified final-iterate rule.
- The controlled mechanism checks support the predicted behavior on this
  problem, not every assumption of a general convergence theorem.
