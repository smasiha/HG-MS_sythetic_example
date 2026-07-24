#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPOSITORY_ROOT"

PYTHON="${PYTHON:-python}"
OUTPUT_BASE="${OUTPUT_BASE:-synthetic_ula_ridge_cg/outputs/reproduction}"
FIXED_ROOT="$OUTPUT_BASE/fixed_d20"
K10_ROOT="$OUTPUT_BASE/fixed_d20_k10"
SELECTED_CONFIG="$FIXED_ROOT/pilot/selected_config.json"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

"$PYTHON" -m synthetic_ula_ridge_cg.aggregate_results \
  --config synthetic_ula_ridge_cg/config/default_config.json \
  --selected-config "$SELECTED_CONFIG" \
  --main-root "$FIXED_ROOT/main" \
  --output-directory "$FIXED_ROOT/summary"

"$PYTHON" -m synthetic_ula_ridge_cg.aggregate_results \
  --config synthetic_ula_ridge_cg/config/k10_config.json \
  --selected-config "$SELECTED_CONFIG" \
  --main-root "$K10_ROOT/main" \
  --output-directory "$K10_ROOT/summary"

"$PYTHON" -m synthetic_ula_ridge_cg.analyze_mechanisms \
  --mechanism-directory "$FIXED_ROOT/mechanisms" \
  --output "$FIXED_ROOT/mechanisms/scaling_summary.md"

"$PYTHON" -m synthetic_ula_ridge_cg.analyze_mechanisms \
  --selection-only \
  --mechanism-directory "$K10_ROOT/mechanisms" \
  --output "$K10_ROOT/mechanisms/scaling_summary.md"

"$PYTHON" -m synthetic_ula_ridge_cg.build_reproduction_report \
  --fixed-root "$FIXED_ROOT" \
  --k10-root "$K10_ROOT" \
  --output-directory "$OUTPUT_BASE/report"

echo "[aggregate] combined report: $OUTPUT_BASE/report/RESULTS.md"
