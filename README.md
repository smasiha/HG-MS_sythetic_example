# Controlled ULA/Gibbs and Ridge-CG Synthetic Experiment

This anonymous, CPU-only artifact studies hyper-gradient optimization when the
lower-level problem has a known non-singleton minimizer manifold. It directly
implements true ULA/Gibbs candidate generation, hard optimistic
best-of-$N$ selection, and a ridge-CG pseudoinverse approximation. The final
comparison gives all five methods 60 single-thread CPU seconds for each of
five held-out seeds and $k\in\{1,2,4,10\}$.

- Standalone formulations and result tables:
  [`synthetic_ula_ridge_cg/EXPERIMENT_REPORT.md`](synthetic_ula_ridge_cg/EXPERIMENT_REPORT.md)
- Intrinsic-Hessian proof:
  [`synthetic_ula_ridge_cg/OPTIMISTIC_HESSIAN_PROOF.md`](synthetic_ula_ridge_cg/OPTIMISTIC_HESSIAN_PROOF.md)
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
