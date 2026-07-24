#!/usr/bin/env bash
#SBATCH --job-name=ula-ridge-cg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=synthetic_ula_ridge_cg/slurm-%j.out
#SBATCH --error=synthetic_ula_ridge_cg/slurm-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

JOBS="${SLURM_CPUS_PER_TASK:-1}" \
  bash synthetic_ula_ridge_cg/scripts/reproduce_all.sh
