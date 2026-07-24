#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPOSITORY_ROOT"

PYTHON="${PYTHON:-python}"
JOBS="${JOBS:-1}"
RERUN_PILOT="${RERUN_PILOT:-1}"
OUTPUT_BASE="${OUTPUT_BASE:-synthetic_ula_ridge_cg/outputs/reproduction}"
FIXED_ROOT="$OUTPUT_BASE/fixed_d20"
K10_ROOT="$OUTPUT_BASE/fixed_d20_k10"
SELECTED_CONFIG="$FIXED_ROOT/pilot/selected_config.json"

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "JOBS must be a positive integer" >&2
  exit 2
fi
if [[ "$RERUN_PILOT" != "0" && "$RERUN_PILOT" != "1" ]]; then
  echo "RERUN_PILOT must be 0 or 1" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0

"$PYTHON" -m unittest discover -s synthetic_ula_ridge_cg/tests -v

if [[ "$RERUN_PILOT" == "1" ]]; then
  OUTPUT_ROOT="$FIXED_ROOT/pilot" JOBS="$JOBS" PYTHON="$PYTHON" \
    bash synthetic_ula_ridge_cg/scripts/reproduce_pilot.sh
else
  mkdir -p "$FIXED_ROOT/pilot"
  cp synthetic_ula_ridge_cg/results/pilot/selected_config.json \
    "$SELECTED_CONFIG"
fi

run_tasks() {
  local task_count="$1"
  local config="$2"
  local output_root="$3"
  local task_id
  local pid
  local -a pids=()

  for ((task_id = 0; task_id < task_count; task_id++)); do
    "$PYTHON" -m synthetic_ula_ridge_cg.run_task \
      --task-id "$task_id" \
      --config "$config" \
      --selected-config "$SELECTED_CONFIG" \
      --output-root "$output_root" \
      --overwrite &
    pids+=("$!")
    if (( ${#pids[@]} >= JOBS )); then
      for pid in "${pids[@]}"; do
        wait "$pid"
      done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

run_tasks \
  30 \
  synthetic_ula_ridge_cg/config/default_config.json \
  "$FIXED_ROOT/main"
run_tasks \
  10 \
  synthetic_ula_ridge_cg/config/k10_config.json \
  "$K10_ROOT/main"

"$PYTHON" -m synthetic_ula_ridge_cg.run_mechanism_checks \
  --config synthetic_ula_ridge_cg/config/default_config.json \
  --output-directory "$FIXED_ROOT/mechanisms"

"$PYTHON" -m synthetic_ula_ridge_cg.run_mechanism_checks \
  --config synthetic_ula_ridge_cg/config/k10_config.json \
  --selection-k-values 10 \
  --selection-only \
  --output-directory "$K10_ROOT/mechanisms"

OUTPUT_BASE="$OUTPUT_BASE" PYTHON="$PYTHON" \
  bash synthetic_ula_ridge_cg/scripts/aggregate.sh

echo "[reproduce] complete: $OUTPUT_BASE/report/RESULTS.md"
