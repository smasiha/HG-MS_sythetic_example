# Controlled ULA/Gibbs and Ridge-CG Synthetic Experiment

This anonymous, CPU-only artifact studies hyper-gradient optimization when the
lower-level problem has a known non-singleton minimizer manifold. It directly
implements true ULA/Gibbs candidate generation, hard optimistic
best-of-\(N\) selection, and a ridge-CG pseudoinverse approximation, with
fixed-wall-clock comparisons against four bilevel baselines.

- Complete setup and reproduction guide:
  [`synthetic_ula_ridge_cg/README.md`](synthetic_ula_ridge_cg/README.md)
- Generated tables and interpretation:
  [`synthetic_ula_ridge_cg/RESULTS.md`](synthetic_ula_ridge_cg/RESULTS.md)
- Algorithm-to-code audit:
  [`synthetic_ula_ridge_cg/ALGORITHM_AUDIT.md`](synthetic_ula_ridge_cg/ALGORITHM_AUDIT.md)
- Machine-readable inventory and checksums:
  [`synthetic_ula_ridge_cg/results/MANIFEST.json`](synthetic_ula_ridge_cg/results/MANIFEST.json)

Run the complete CPU smoke test from the repository root:

```bash
python -m pip install -r synthetic_ula_ridge_cg/requirements.txt
bash synthetic_ula_ridge_cg/scripts/smoke_test.sh
```

No GPU, external dataset, Slurm installation, or network access is required.
