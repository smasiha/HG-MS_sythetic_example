# Optional Slurm Wrapper

The portable workflow in `../scripts/reproduce_all.sh` is canonical and does
not require Slurm. `run_full.sbatch` is only a convenience wrapper.

Submit from the repository root:

```bash
sbatch synthetic_ula_ridge_cg/slurm/run_full.sbatch
```

The template deliberately omits cluster-specific partitions, accounts, QoS
settings, and accelerators. Adjust memory, wall time, and CPU count through
normal scheduler options if required. The experiment is CPU-only; requesting a
GPU does not improve it.
