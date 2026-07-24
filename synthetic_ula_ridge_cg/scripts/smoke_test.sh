#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPOSITORY_ROOT"

PYTHON="${PYTHON:-python}"
TEMPORARY_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ula-ridge-cg-smoke.XXXXXX")"
trap 'rm -rf "$TEMPORARY_ROOT"' EXIT

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0

"$PYTHON" -m unittest discover -s synthetic_ula_ridge_cg/tests -v

"$PYTHON" -m synthetic_ula_ridge_cg.run_benchmark \
  --k 2 \
  --seed 1000 \
  --budget-seconds 0.05 \
  --methods hg_ms,hg_baseline,v_pbgd,g_pbgd,iaptt_gm \
  --config synthetic_ula_ridge_cg/config/default_config.json \
  --selected-config synthetic_ula_ridge_cg/results/pilot/selected_config.json \
  --output-root "$TEMPORARY_ROOT/end_to_end" \
  --fixed-order \
  --overwrite

"$PYTHON" -m synthetic_ula_ridge_cg.run_mechanism_checks \
  --quick \
  --selection-k-values 1 2 4 10 \
  --config synthetic_ula_ridge_cg/config/default_config.json \
  --output-directory "$TEMPORARY_ROOT/mechanisms"

"$PYTHON" -m synthetic_ula_ridge_cg.analyze_mechanisms \
  --mechanism-directory "$TEMPORARY_ROOT/mechanisms" \
  --output "$TEMPORARY_ROOT/mechanisms/scaling_summary.md"

"$PYTHON" -m synthetic_ula_ridge_cg.validate_public_results

echo "[smoke] all unit, integration, mechanism, and artifact checks passed"
