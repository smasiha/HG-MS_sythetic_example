#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPOSITORY_ROOT"

PYTHON="${PYTHON:-python}"
JOBS="${JOBS:-1}"
RERUN_PILOT="${RERUN_PILOT:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-synthetic_ula_ridge_cg/outputs/reproduction}"
CONFIG="synthetic_ula_ridge_cg/config/default_config.json"
SELECTED_CONFIG="$OUTPUT_ROOT/pilot/selected_config.json"

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
export CUDA_VISIBLE_DEVICES=""

"$PYTHON" -m unittest discover -s synthetic_ula_ridge_cg/tests -v

if [[ "$RERUN_PILOT" == "1" ]]; then
  OUTPUT_ROOT="$OUTPUT_ROOT/pilot" JOBS="$JOBS" PYTHON="$PYTHON" \
    bash synthetic_ula_ridge_cg/scripts/reproduce_pilot.sh
else
  mkdir -p "$OUTPUT_ROOT/pilot"
  cp synthetic_ula_ridge_cg/results/pilot/selected_config.json \
    "$SELECTED_CONFIG"
fi

task_count="$(
  "$PYTHON" - <<'PY'
import json
from pathlib import Path

config = json.loads(
    Path("synthetic_ula_ridge_cg/config/default_config.json").read_text(
        encoding="utf-8"
    )
)
print(
    len(config["experiment"]["k_values"])
    * len(config["experiment"]["seeds"])
)
PY
)"

run_task() {
  "$PYTHON" -m synthetic_ula_ridge_cg.run_task \
    --task-id "$1" \
    --config "$CONFIG" \
    --selected-config "$SELECTED_CONFIG" \
    --output-root "$OUTPUT_ROOT/main" \
    --overwrite
}

pids=()
for ((task_id = 0; task_id < task_count; task_id++)); do
  run_task "$task_id" &
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

"$PYTHON" -m synthetic_ula_ridge_cg.run_mechanism_checks \
  --config "$CONFIG" \
  --selection-k-values 1 2 4 10 \
  --output-directory "$OUTPUT_ROOT/mechanisms"

OUTPUT_ROOT="$OUTPUT_ROOT" PYTHON="$PYTHON" \
  bash synthetic_ula_ridge_cg/scripts/aggregate.sh

echo "[reproduce] complete: $OUTPUT_ROOT/report/RESULTS.md"
