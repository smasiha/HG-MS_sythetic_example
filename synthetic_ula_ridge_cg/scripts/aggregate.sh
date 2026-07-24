#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPOSITORY_ROOT"

PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-synthetic_ula_ridge_cg/outputs/reproduction}"
CONFIG="synthetic_ula_ridge_cg/config/default_config.json"
SELECTED_CONFIG="$OUTPUT_ROOT/pilot/selected_config.json"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

"$PYTHON" -m synthetic_ula_ridge_cg.aggregate_results \
  --config "$CONFIG" \
  --selected-config "$SELECTED_CONFIG" \
  --main-root "$OUTPUT_ROOT/main" \
  --output-directory "$OUTPUT_ROOT/summary"

"$PYTHON" -m synthetic_ula_ridge_cg.analyze_mechanisms \
  --mechanism-directory "$OUTPUT_ROOT/mechanisms" \
  --output "$OUTPUT_ROOT/mechanisms/scaling_summary.md"

"$PYTHON" -m synthetic_ula_ridge_cg.build_reproduction_report \
  --reproduction-root "$OUTPUT_ROOT" \
  --output-directory "$OUTPUT_ROOT/report"

echo "[aggregate] report: $OUTPUT_ROOT/report/RESULTS.md"
