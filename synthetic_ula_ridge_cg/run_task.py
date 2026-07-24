from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .configuration import PACKAGE_ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Map a Slurm array index to k and seed.")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default_config.json",
    )
    parser.add_argument(
        "--selected-config",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "pilot" / "selected_config.json",
    )
    parser.add_argument(
        "--output-root", type=Path, default=PACKAGE_ROOT / "outputs" / "main"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    k_values = [int(value) for value in config["experiment"]["k_values"]]
    seeds = [int(value) for value in config["experiment"]["seeds"]]
    tasks = [(k, seed) for k in k_values for seed in seeds]
    if args.task_id < 0 or args.task_id >= len(tasks):
        raise ValueError(f"task id {args.task_id} outside [0, {len(tasks) - 1}]")
    k, seed = tasks[args.task_id]
    command = [
        sys.executable,
        "-m",
        "synthetic_ula_ridge_cg.run_benchmark",
        "--k",
        str(k),
        "--seed",
        str(seed),
        "--config",
        str(args.config),
        "--selected-config",
        str(args.selected_config),
        "--output-root",
        str(args.output_root),
    ]
    if args.overwrite:
        command.append("--overwrite")
    print(f"[task] task_id={args.task_id} k={k} seed={seed}", flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
