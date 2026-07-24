from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .configuration import (
    PACKAGE_ROOT,
    SCHEMA_VERSION,
    atomic_write_json,
    copy_with_override,
    load_config,
    make_problem,
    read_json,
)
from .methods import METHOD_LABELS, run_method
from .run_benchmark import compatible_cached_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Equal-size validation pilot sweep for one synthetic method."
    )
    parser.add_argument("--method", choices=sorted(METHOD_LABELS), required=True)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default_config.json",
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "pilot_grid.json",
    )
    parser.add_argument(
        "--output-root", type=Path, default=PACKAGE_ROOT / "outputs" / "pilot"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    grid = read_json(args.grid)
    if int(grid.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"pilot-grid schema mismatch: expected {SCHEMA_VERSION}, "
            f"found {grid.get('schema_version')}"
        )
    seeds = [int(seed) for seed in grid["pilot_seeds"]]
    budget = float(grid["pilot_budget_seconds"])
    candidates: List[Dict[str, Any]] = list(grid[args.method])
    final_seeds = {int(seed) for seed in config["experiment"]["seeds"]}
    if len(candidates) != 8:
        raise ValueError(
            f"{args.method} must have exactly eight pilot configurations"
        )
    if len(seeds) != 3 or len(seeds) != len(set(seeds)):
        raise ValueError("the pilot must use exactly three distinct seeds")
    if final_seeds.intersection(seeds):
        raise ValueError("pilot and final seeds must be disjoint")
    if budget <= 0:
        raise ValueError("pilot budget must be positive")
    if grid["selection_metric"] != "hyperobjective_gap":
        raise ValueError("pilot selection must use the exact hyperobjective gap")
    problem = make_problem(config, args.k)
    method_directory = args.output_root / args.method
    method_directory.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    print(
        f"[pilot] method={args.method} "
        f"candidates={len(candidates)} seeds={seeds} budget={budget}s",
        flush=True,
    )
    for candidate_index, override in enumerate(candidates):
        candidate_config = copy_with_override(config, args.method, override)
        candidate_values = []
        for seed in seeds:
            path = method_directory / f"candidate_{candidate_index:02d}_seed_{seed}.json"
            if (
                path.exists()
                and not args.overwrite
                and compatible_cached_run(
                    path,
                    args.method,
                    problem.metadata(),
                    dict(candidate_config["common"]),
                    dict(candidate_config[args.method]),
                    budget,
                    seed,
                )
            ):
                result = read_json(path)
            else:
                result = run_method(
                    args.method,
                    problem,
                    seed,
                    budget,
                    candidate_config,
                )
                result["candidate_index"] = candidate_index
                result["candidate_override"] = override
                atomic_write_json(path, result)
            value = float(result["final"][grid["selection_metric"]])
            if (
                int(result["final"].get("cg_breakdowns", 0)) > 0
                or int(result["final"].get("cg_tolerance_failures", 0)) > 0
            ):
                value = float("inf")
            if args.method in {"hg_ms", "hg_baseline"}:
                residual = float(result["final"]["cg_absolute_residual"])
                tolerance = float(
                    result["method_config"]["cg_absolute_tolerance"]
                )
                if residual > tolerance * (1.0 + 1e-8):
                    value = float("inf")
            candidate_values.append(value)
        score = float(
            np.mean(np.log10(np.maximum(np.asarray(candidate_values), 1e-16)))
        )
        record = {
            "candidate_index": candidate_index,
            "override": override,
            "values": candidate_values,
            "mean_log10_score": score,
            "geometric_mean_score": float(10.0**score),
        }
        records.append(record)
        print(
            f"[pilot] candidate={candidate_index:02d} score={score:.6f} "
            f"override={override}",
            flush=True,
        )

    finite_records = [record for record in records if math.isfinite(record["mean_log10_score"])]
    if not finite_records:
        raise RuntimeError(f"all pilot candidates failed for {args.method}")
    best = min(finite_records, key=lambda record: record["mean_log10_score"])
    serializable_records = []
    for record in records:
        serializable = dict(record)
        serializable["values"] = [
            value if math.isfinite(value) else None
            for value in record["values"]
        ]
        for key in ("mean_log10_score", "geometric_mean_score"):
            if not math.isfinite(record[key]):
                serializable[key] = None
        serializable["failed"] = not math.isfinite(
            record["mean_log10_score"]
        )
        serializable_records.append(serializable)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "method": args.method,
        "method_label": METHOD_LABELS[args.method],
        "k": args.k,
        "pilot_seeds": seeds,
        "pilot_budget_seconds": budget,
        "selection_metric": grid["selection_metric"],
        "candidates": serializable_records,
        "selected_index": best["candidate_index"],
        "selected_override": best["override"],
    }
    atomic_write_json(method_directory / "summary.json", summary)
    print(
        f"[pilot] selected candidate={best['candidate_index']} "
        f"override={best['override']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
