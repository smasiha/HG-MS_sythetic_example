from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .configuration import (
    PACKAGE_ROOT,
    SCHEMA_VERSION,
    atomic_write_json,
    load_config,
    read_json,
)
from .methods import METHOD_LABELS
from .problem import ManifoldBilevelProblem


SUMMARY_METRICS = [
    "hyperobjective_gap",
    "theta_error",
    "exact_hypergradient_abs",
    "manifold_distance",
    "optimistic_distance",
    "iteration",
    "elapsed_seconds",
    "grad_g",
    "f",
    "hessian_g",
    "hvp_g",
    "cg_iterations",
    "lower_steps",
]


def write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    if not records:
        raise ValueError(f"no records for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(records[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def bootstrap_mean_statistics(
    values: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
) -> Tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0]), 0.0
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    means = np.mean(values[indices], axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    center = float(np.mean(values))
    half_width = float(np.quantile(np.abs(means - center), 0.95))
    return float(lower), float(upper), half_width


def bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
) -> Tuple[float, float]:
    lower, upper, _ = bootstrap_mean_statistics(values, rng, repetitions)
    return lower, upper


def discover_runs(main_root: Path) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(main_root.glob("k_*/seed_*/*.json")):
        run = read_json(path)
        run["_path"] = str(path)
        runs.append(run)
    return runs


def validate_runs(
    runs: Sequence[Dict[str, Any]],
    expected_k: Sequence[int],
    expected_seeds: Sequence[int],
    expected_methods: Sequence[str],
    problem_config: Dict[str, Any],
    common_config: Dict[str, Any],
    method_configs: Dict[str, Dict[str, Any]],
) -> List[str]:
    warnings = []
    keys = {(int(run["problem"]["k"]), int(run["seed"]), run["method"]) for run in runs}
    expected_keys = {
        (k, seed, method)
        for k in expected_k
        for seed in expected_seeds
        for method in expected_methods
    }
    if len(runs) != len(expected_keys):
        warnings.append(
            f"expected {len(expected_keys)} run files, found {len(runs)}"
        )
    for k in expected_k:
        for seed in expected_seeds:
            for method in expected_methods:
                if (k, seed, method) not in keys:
                    warnings.append(f"missing k={k} seed={seed} method={method}")
    for run in runs:
        if int(run.get("schema_version", -1)) != SCHEMA_VERSION:
            warnings.append(
                f"stale schema {run.get('schema_version')} for {run['_path']}; "
                f"expected {SCHEMA_VERSION}"
            )
        key = (int(run["problem"]["k"]), int(run["seed"]), run["method"])
        if key not in expected_keys:
            warnings.append(f"unexpected run {key} in {run['_path']}")
        expected_problem = ManifoldBilevelProblem(
            k=int(run["problem"]["k"]), **problem_config
        ).metadata()
        if run["problem"] != expected_problem:
            warnings.append(
                f"problem metadata mismatch for {run['_path']}: "
                f"{run['problem']} != {expected_problem}"
            )
        if run.get("common_config") != common_config:
            warnings.append(
                f"common configuration mismatch for {run['_path']}: "
                f"{run.get('common_config')} != {common_config}"
            )
        expected_method_config = method_configs.get(run["method"])
        if expected_method_config is None:
            warnings.append(
                f"no expected method configuration for {run['method']} "
                f"in {run['_path']}"
            )
        elif run.get("method_config") != expected_method_config:
            warnings.append(
                f"method configuration mismatch for {run['_path']}: "
                f"{run.get('method_config')} != {expected_method_config}"
            )
        budget = float(run["budget_seconds"])
        overrun = float(run["budget_overrun_seconds"])
        allowed = max(0.1, 0.05 * budget)
        if overrun > allowed:
            warnings.append(
                f"large budget overrun {overrun:.3f}s for {run['_path']} "
                f"(allowed {allowed:.3f}s)"
            )
        if not np.isfinite(float(run["final"]["hyperobjective_gap"])):
            warnings.append(f"non-finite final objective for {run['_path']}")
        if int(run["final"].get("cg_breakdowns", 0)) > 0:
            warnings.append(
                f"{run['final']['cg_breakdowns']} CG breakdown(s) for {run['_path']}"
            )
        if int(run["final"].get("cg_tolerance_failures", 0)) > 0:
            warnings.append(
                f"{run['final']['cg_tolerance_failures']} CG tolerance failure(s) "
                f"for {run['_path']}"
            )
        if run["method"] in {"hg_ms", "hg_baseline"}:
            tolerance = float(run["method_config"]["cg_absolute_tolerance"])
            residual_value = run["final"].get("cg_absolute_residual")
            try:
                residual = float(residual_value)
            except (TypeError, ValueError):
                residual = float("nan")
            if not np.isfinite(residual):
                warnings.append(
                    f"missing/non-finite CG absolute residual for {run['_path']}"
                )
            elif residual > tolerance * (1.0 + 1e-8):
                warnings.append(
                    f"CG residual {residual:.3e} exceeds eta={tolerance:.3e} "
                    f"for {run['_path']}"
                )
    return warnings


def aggregate_final(
    runs: Sequence[Dict[str, Any]],
    bootstrap_repetitions: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(int(run["problem"]["k"]), run["method"])].append(run)
    records = []
    for (k, method), group in sorted(grouped.items()):
        record: Dict[str, Any] = {
            "k": k,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "runs": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = np.asarray(
                [float(run["final"].get(metric, float("nan"))) for run in group],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            record[f"{metric}_mean"] = (
                float(np.mean(finite)) if finite.size else float("nan")
            )
            record[f"{metric}_std"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            )
            lower, upper, half_width = bootstrap_mean_statistics(
                values,
                rng,
                bootstrap_repetitions,
            )
            record[f"{metric}_ci95_low"] = lower
            record[f"{metric}_ci95_high"] = upper
            record[f"{metric}_ci95_half_width"] = half_width
        records.append(record)
    return records


def aggregate_paired_differences(
    runs: Sequence[Dict[str, Any]],
    bootstrap_repetitions: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    indexed = {
        (int(run["problem"]["k"]), run["method"], int(run["seed"])): run
        for run in runs
    }
    k_values = sorted({int(run["problem"]["k"]) for run in runs})
    seeds = sorted({int(run["seed"]) for run in runs})
    records = []
    for k in k_values:
        for baseline in METHOD_LABELS:
            if baseline == "hg_ms":
                continue
            differences = []
            for seed in seeds:
                hg_run = indexed.get((k, "hg_ms", seed))
                baseline_run = indexed.get((k, baseline, seed))
                if hg_run is None or baseline_run is None:
                    continue
                differences.append(
                    float(baseline_run["final"]["hyperobjective_gap"])
                    - float(hg_run["final"]["hyperobjective_gap"])
                )
            values = np.asarray(differences, dtype=np.float64)
            lower, upper = bootstrap_interval(
                values, rng, bootstrap_repetitions
            )
            records.append(
                {
                    "k": k,
                    "baseline": baseline,
                    "baseline_label": METHOD_LABELS[baseline],
                    "paired_runs": int(values.size),
                    "mean_baseline_minus_hg_ms_gap": float(
                        np.mean(values)
                    ),
                    "ci95_low": lower,
                    "ci95_high": upper,
                    "hg_ms_win_fraction": float(np.mean(values > 0.0)),
                }
            )
    return records


def hold_last_value(trajectory: Sequence[Dict[str, Any]], time_value: float, metric: str) -> float:
    chosen = trajectory[0]
    for record in trajectory:
        if float(record["elapsed_seconds"]) > time_value:
            break
        chosen = record
    return float(chosen.get(metric, float("nan")))


def aggregate_curves(
    runs: Sequence[Dict[str, Any]],
    bootstrap_repetitions: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(int(run["problem"]["k"]), run["method"])].append(run)
    records = []
    curve_metrics = ["hyperobjective_gap", "theta_error", "exact_hypergradient_abs"]
    for (k, method), group in sorted(grouped.items()):
        common_budget = min(float(run["budget_seconds"]) for run in group)
        time_grid = np.linspace(0.0, common_budget, 121)
        for metric in curve_metrics:
            for time_value in time_grid:
                values = np.asarray(
                    [
                        hold_last_value(run["trajectory"], float(time_value), metric)
                        for run in group
                    ]
                )
                finite = values[np.isfinite(values)]
                lower, upper = bootstrap_interval(values, rng, bootstrap_repetitions)
                records.append(
                    {
                        "k": k,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "metric": metric,
                        "elapsed_seconds": float(time_value),
                        "runs": len(group),
                        "mean": float(np.mean(finite)) if finite.size else float("nan"),
                        "ci95_low": lower,
                        "ci95_high": upper,
                    }
                )
    return records


def write_markdown(path: Path, final_records: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# Fixed-Wall-Clock Synthetic Results",
        "",
        "Mean and nonparametric bootstrap 95% confidence intervals across independent seeds.",
        "",
    ]
    for k in sorted({int(record["k"]) for record in final_records}):
        lines.extend(
            [
                f"## Manifold dimension k={k}",
                "",
                "| Method | Runs | F gap (95% CI) | |theta| (95% CI) | Outer updates | Time (s) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for record in [item for item in final_records if int(item["k"]) == k]:
            lines.append(
                "| {method} | {runs} | {gap:.3e} [{gap_l:.3e}, {gap_u:.3e}] | "
                "{theta:.3e} [{theta_l:.3e}, {theta_u:.3e}] | {iterations:.1f} | "
                "{elapsed:.3f} |".format(
                    method=record["method_label"],
                    runs=record["runs"],
                    gap=record["hyperobjective_gap_mean"],
                    gap_l=record["hyperobjective_gap_ci95_low"],
                    gap_u=record["hyperobjective_gap_ci95_high"],
                    theta=record["theta_error_mean"],
                    theta_l=record["theta_error_ci95_low"],
                    theta_u=record["theta_error_ci95_high"],
                    iterations=record["iteration_mean"],
                    elapsed=record["elapsed_seconds_mean"],
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paired_markdown(
    path: Path, records: Sequence[Dict[str, Any]]
) -> None:
    lines = [
        "# Paired HG-MS Comparisons",
        "",
        "Positive differences mean that HG-MS has the smaller exact "
        "hyperobjective gap. Intervals bootstrap paired final seeds.",
        "",
        "| k | Baseline | Seeds | Baseline gap - HG-MS gap (95% CI) | HG-MS win fraction |",
        "|---:|---|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| {k} | {baseline_label} | {paired_runs} | "
            "{difference:.3e} [{lower:.3e}, {upper:.3e}] | "
            "{win_fraction:.2f} |".format(
                k=record["k"],
                baseline_label=record["baseline_label"],
                paired_runs=record["paired_runs"],
                difference=record["mean_baseline_minus_hg_ms_gap"],
                lower=record["ci95_low"],
                upper=record["ci95_high"],
                win_fraction=record["hg_ms_win_fraction"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate five-seed fixed-wall-clock synthetic results."
    )
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
        "--main-root", type=Path, default=PACKAGE_ROOT / "outputs" / "main"
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "summary",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5_000)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.selected_config)
    runs = discover_runs(args.main_root)
    if not runs:
        raise FileNotFoundError(f"no run JSON files found under {args.main_root}")
    warnings = validate_runs(
        runs,
        [int(value) for value in config["experiment"]["k_values"]],
        [int(value) for value in config["experiment"]["seeds"]],
        list(config["experiment"]["methods"]),
        dict(config["problem"]),
        dict(config["common"]),
        {
            method: dict(config[method])
            for method in config["experiment"]["methods"]
        },
    )
    if warnings and not args.allow_incomplete:
        raise RuntimeError(
            "result validation failed:\n" + "\n".join(f"- {item}" for item in warnings)
        )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(82_731)
    final_records = aggregate_final(runs, args.bootstrap_repetitions, rng)
    paired_records = aggregate_paired_differences(
        runs, args.bootstrap_repetitions, rng
    )
    curve_records = aggregate_curves(runs, args.bootstrap_repetitions, rng)
    write_csv(args.output_directory / "final_summary.csv", final_records)
    write_csv(
        args.output_directory / "paired_comparisons.csv", paired_records
    )
    write_csv(args.output_directory / "wallclock_curves.csv", curve_records)
    write_markdown(args.output_directory / "final_summary.md", final_records)
    write_paired_markdown(
        args.output_directory / "paired_comparisons.md", paired_records
    )
    atomic_write_json(
        args.output_directory / "validation.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_files": len(runs),
            "warnings": warnings,
            "complete": not warnings,
            "bootstrap_repetitions": args.bootstrap_repetitions,
        },
    )
    print(f"[aggregate] runs={len(runs)} warnings={len(warnings)}")
    print(f"[aggregate] wrote {args.output_directory.resolve()}")


if __name__ == "__main__":
    main()
