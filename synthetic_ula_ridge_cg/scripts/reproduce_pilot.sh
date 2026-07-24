#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPOSITORY_ROOT"

PYTHON="${PYTHON:-python}"
JOBS="${JOBS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-synthetic_ula_ridge_cg/outputs/reproduction/fixed_d20/pilot}"
CONFIG="${CONFIG:-synthetic_ula_ridge_cg/config/default_config.json}"
GRID="${GRID:-synthetic_ula_ridge_cg/config/pilot_grid.json}"

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "JOBS must be a positive integer" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0

methods=(hg_ms hg_baseline v_pbgd g_pbgd iaptt_gm)
pids=()

wait_batch() {
  local pid
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  pids=()
}

for method in "${methods[@]}"; do
  "$PYTHON" -m synthetic_ula_ridge_cg.run_pilot \
    --method "$method" \
    --k 1 \
    --config "$CONFIG" \
    --grid "$GRID" \
    --output-root "$OUTPUT_ROOT" \
    --overwrite &
  pids+=("$!")
  if (( ${#pids[@]} >= JOBS )); then
    wait_batch
  fi
done
wait_batch

"$PYTHON" -m synthetic_ula_ridge_cg.select_pilot \
  --config "$CONFIG" \
  --pilot-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/selected_config.json"

echo "[pilot] complete: $OUTPUT_ROOT/selected_config.json"
