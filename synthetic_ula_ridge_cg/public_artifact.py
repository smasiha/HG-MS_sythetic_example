from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .aggregate_results import (
    aggregate_curves,
    aggregate_final,
    aggregate_paired_differences,
)
from .analyze_mechanisms import build_summary, log_slope
from .configuration import load_config, read_json, validate_config
from .methods import METHOD_LABELS


PACKAGE_VERSION = "1.0.0"
BOOTSTRAP_SEED = 82_731
BOOTSTRAP_REPETITIONS = 5_000
K_VALUES = (1, 2, 4, 10)
BASE_K_VALUES = (1, 2, 4)
FINAL_SEEDS = tuple(range(1000, 1010))
PILOT_SEEDS = (11, 28, 47)
METHODS = ("hg_ms", "hg_baseline", "v_pbgd", "g_pbgd", "iaptt_gm")
DISPLAY_METRICS = (
    "hyperobjective_gap",
    "exact_hypergradient_abs",
    "optimistic_distance",
)
ENVIRONMENT_KEYS = {
    "host",
    "hostname",
    "cpu_count_visible",
    "job_id",
    "slurm_job_id",
    "user",
    "username",
    "selected_config_path",
}
FORBIDDEN_TEXT_PATTERNS = {
    "home path": re.compile(r"/home/", re.IGNORECASE),
    "macOS user path": re.compile(r"/Users/", re.IGNORECASE),
    "Windows user path": re.compile(r"[A-Za-z]:\\\\Users\\\\", re.IGNORECASE),
    "private SSH remote": re.compile(r"\bgit@", re.IGNORECASE),
    "cluster hostname": re.compile(r"\bkh[0-9]{3}\b", re.IGNORECASE),
    "private cluster name": re.compile(r"\bkuma(?:\.|\b)", re.IGNORECASE),
    "source username": re.compile(r"masiha", re.IGNORECASE),
    "email address": re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    ),
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise ValueError(f"refusing to write empty CSV: {path}")
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


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_json(
    value: Any,
    removed: Counter[str] | None = None,
) -> Any:
    removed = removed if removed is not None else Counter()
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if key.lower() in ENVIRONMENT_KEYS:
                removed[key.lower()] += 1
                continue
            sanitized[key] = sanitize_json(item, removed)
        return sanitized
    if isinstance(value, list):
        return [sanitize_json(item, removed) for item in value]
    if isinstance(value, str):
        for label, pattern in FORBIDDEN_TEXT_PATTERNS.items():
            if pattern.search(value):
                raise ValueError(
                    f"identifying string ({label}) is not attached to a "
                    f"sanitizable environment key: {value!r}"
                )
    return value


def assert_sanitization_is_lossless(source: Any, sanitized: Any) -> None:
    expected = sanitize_json(source)
    if expected != sanitized:
        raise AssertionError("sanitization changed non-environment result data")


def discover_public_runs(raw_directory: Path) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(raw_directory.glob("k_*/seed_*/*.json")):
        run = read_json(path)
        run["_path"] = str(path)
        runs.append(run)
    return runs


def _batch_summaries(
    runs: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    final = aggregate_final(runs, BOOTSTRAP_REPETITIONS, rng)
    paired = aggregate_paired_differences(
        runs,
        BOOTSTRAP_REPETITIONS,
        rng,
    )
    curves = aggregate_curves(runs, BOOTSTRAP_REPETITIONS, rng)
    return final, paired, curves


def regenerate_public_summaries(
    runs: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    outputs: List[List[Dict[str, Any]]] = [[], [], []]
    for k_values in (BASE_K_VALUES, (10,)):
        batch = [run for run in runs if int(run["problem"]["k"]) in k_values]
        batch.sort(key=lambda run: run["_path"])
        generated = _batch_summaries(batch)
        for destination, records in zip(outputs, generated):
            destination.extend(records)
    outputs[0].sort(key=lambda record: (int(record["k"]), record["method"]))
    outputs[1].sort(
        key=lambda record: (int(record["k"]), record["baseline"])
    )
    outputs[2].sort(
        key=lambda record: (
            int(record["k"]),
            record["method"],
            record["metric"],
            float(record["elapsed_seconds"]),
        )
    )
    return outputs[0], outputs[1], outputs[2]


def final_per_seed_records(
    runs: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    records = []
    for run in sorted(
        runs,
        key=lambda item: (
            int(item["problem"]["k"]),
            int(item["seed"]),
            item["method"],
        ),
    ):
        final = run["final"]
        record = {
            "k": int(run["problem"]["k"]),
            "seed": int(run["seed"]),
            "method": run["method"],
            "method_label": run["method_label"],
            "budget_seconds": float(run["budget_seconds"]),
            "elapsed_seconds": float(run["elapsed_seconds"]),
            "budget_overrun_seconds": float(run["budget_overrun_seconds"]),
            "theta_initial": float(run["theta_initial"]),
        }
        for metric in (
            "hyperobjective_gap",
            "theta_error",
            "exact_hypergradient_abs",
            "manifold_distance",
            "optimistic_distance",
            "iteration",
            "grad_g",
            "f",
            "hessian_g",
            "hvp_g",
            "cg_iterations",
            "lower_steps",
            "cg_breakdowns",
            "cg_tolerance_failures",
        ):
            record[metric] = final.get(metric)
        records.append(record)
    return records


def _numeric_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def assert_records_match(
    expected: Sequence[Mapping[str, Any]],
    observed: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    if len(expected) != len(observed):
        raise AssertionError(
            f"{label}: expected {len(expected)} rows, found {len(observed)}"
        )
    for row_index, (left, right) in enumerate(zip(expected, observed)):
        if set(left) != set(right):
            raise AssertionError(
                f"{label} row {row_index}: column mismatch "
                f"{sorted(set(left) ^ set(right))}"
            )
        for key in left:
            left_number = _numeric_value(left[key])
            right_number = _numeric_value(right[key])
            if left_number is not None and right_number is not None:
                if not math.isclose(
                    left_number,
                    right_number,
                    rel_tol=2e-12,
                    abs_tol=2e-14,
                ):
                    raise AssertionError(
                        f"{label} row {row_index} field {key}: "
                        f"{left_number} != {right_number}"
                    )
            elif str(left[key]) != str(right[key]):
                raise AssertionError(
                    f"{label} row {row_index} field {key}: "
                    f"{left[key]!r} != {right[key]!r}"
                )


def selection_slopes(
    selection: Sequence[Mapping[str, Any]],
) -> Dict[int, Dict[str, float]]:
    slopes = {}
    for k in K_VALUES:
        group = sorted(
            (row for row in selection if int(row["k"]) == k),
            key=lambda row: int(row["number_candidates"]),
        )[-6:]
        candidates = np.asarray(
            [float(row["number_candidates"]) for row in group]
        )
        observed = np.asarray(
            [float(row["observed_squared_error_mean"]) for row in group]
        )
        reference = np.asarray(
            [float(row["reference_squared_error_mean"]) for row in group]
        )
        slopes[k] = {
            "observed": log_slope(candidates, observed),
            "quadrature": log_slope(candidates, reference),
            "sharp_asymptotic": -2.0 / k,
            "theorem_envelope": -1.0 / k,
        }
    return slopes


def theory_diagnostics(
    selection: Sequence[Mapping[str, Any]],
    ridge: Sequence[Mapping[str, Any]],
    ula: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    exact_rows = [
        row
        for row in ridge
        if row["point_type"] == "exact_optimistic"
    ]
    pseudoinverse_error = max(
        float(row["pseudoinverse_formula_error"]) for row in exact_rows
    )
    ridge_rows = [
        row
        for row in exact_rows
        if float(row["theta"]) == 1.0
        and float(row["cg_absolute_tolerance_eta"]) == 1e-8
    ]
    ridge_bias_slope = log_slope(
        np.asarray([float(row["ridge_gamma"]) for row in ridge_rows]),
        np.asarray([float(row["ridge_bias_abs"]) for row in ridge_rows]),
    )
    strict_cg_rows = [
        row
        for row in ridge
        if row["point_type"] == "manifold_nonoptimistic"
        and float(row["cg_absolute_tolerance_eta"]) == 1e-8
    ]
    cg_solution_error = float(
        np.mean(
            [float(row["cg_solution_error_l2"]) for row in strict_cg_rows]
        )
    )
    matching_ula = [
        row
        for row in ula
        if row["initialization"] == "random-centers"
        and float(row["temperature_lambda"]) == 1e-3
        and float(row["ula_step_size"]) == 0.01
        and int(row["ula_steps"]) == 100
        and int(row["number_candidates"]) == 32
    ]
    if len(matching_ula) != 1:
        raise AssertionError(
            f"expected one primary ULA/Gibbs diagnostic row, found "
            f"{len(matching_ula)}"
        )
    primary = matching_ula[0]
    ula_mse = float(primary["hypergradient_squared_error_mean"])
    gibbs_mse = float(
        primary["exact_gibbs_hypergradient_squared_error_mean"]
    )
    relative_mse_difference = abs(ula_mse - gibbs_mse) / gibbs_mse
    return {
        "selection_slopes": selection_slopes(selection),
        "pseudoinverse_error_max": pseudoinverse_error,
        "ridge_bias_slope": ridge_bias_slope,
        "cg_solution_error_mean_at_1e-8": cg_solution_error,
        "ula_hypergradient_mse": ula_mse,
        "exact_gibbs_hypergradient_mse": gibbs_mse,
        "relative_mse_difference": relative_mse_difference,
        "marginal_ks": {
            "active_radial": float(primary["radial_ks_to_exact_gibbs"]),
            "angular": float(primary["angular_ks_to_exact_gibbs"]),
            "normal": float(primary["normal_ks_to_exact_gibbs"]),
        },
    }


def _format_pair(mean: float, half_width: float, metric: str) -> str:
    if metric == "hyperobjective_gap" and mean < 0.01:
        return (
            f"({mean * 1000:.2f} ± {half_width * 1000:.2f})"
            r"\(\times10^{-3}\)"
        )
    if metric == "hyperobjective_gap" and mean < 1.0:
        return f"{mean:.3f} ± {half_width:.3f}"
    if metric == "hyperobjective_gap":
        return f"{mean:.2f} ± {half_width:.2f}"
    if metric == "exact_hypergradient_abs":
        return f"{mean:.3f} ± {half_width:.3f}"
    return f"{mean:.3f} ± {half_width:.3f}"


def render_results_markdown(
    final_summary: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    selection: Sequence[Mapping[str, Any]],
    ridge: Sequence[Mapping[str, Any]],
    ula: Sequence[Mapping[str, Any]],
    artifact_prefix: str = "",
) -> str:
    indexed = {
        (int(row["k"]), row["method"]): row for row in final_summary
    }
    lines = [
        "# Controlled ULA/Gibbs and Ridge-CG Results",
        "",
        "All end-to-end entries use ten held-out seeds and report mean ± "
        "symmetric bootstrap 95% CI half-width (5,000 resamples). Every cell "
        "contains objective gap / stationarity / optimistic-solution distance; "
        "lower is better.",
        "",
        "| Method | k=1 | k=2 | k=4 | k=10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        cells = []
        for k in K_VALUES:
            row = indexed[(k, method)]
            values = [
                _format_pair(
                    float(row[f"{metric}_mean"]),
                    float(row[f"{metric}_ci95_half_width"]),
                    metric,
                )
                for metric in DISPLAY_METRICS
            ]
            cells.append(" / ".join(values))
        label = METHOD_LABELS[method]
        if method == "hg_ms":
            label = f"**{label}**"
            cells = [f"**{cell}**" for cell in cells]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lowest_mean_cells = sum(
        min(
            METHODS,
            key=lambda method: float(
                indexed[(k, method)][f"{metric}_mean"]
            ),
        )
        == "hg_ms"
        for k in K_VALUES
        for metric in DISPLAY_METRICS
    )
    diagnostics = theory_diagnostics(selection, ridge, ula)
    lines.extend(
        [
            "",
            f"HG-MS has the lowest mean in {lowest_mean_cells} of 12 "
            "end-to-end cells. Paired seed-level differences are available "
            "in [`tables/paired_comparisons.csv`]("
            f"{artifact_prefix}tables/paired_comparisons.csv).",
            "",
            "## Best-of-N Dimension Check",
            "",
            "| k | Observed slope | Quadrature slope | Sharp asymptotic | Conservative theorem envelope |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for k in K_VALUES:
        values = diagnostics["selection_slopes"][k]
        lines.append(
            f"| {k} | {values['observed']:.3f} | "
            f"{values['quadrature']:.3f} | "
            f"{values['sharp_asymptotic']:.3f} | "
            f"{values['theorem_envelope']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The observed slopes match the finite-range quadrature predictions "
            "and are consistent with the sharper problem-specific "
            r"\(N^{-2/k}\) behavior, improving on the conservative "
            r"\(N^{-1/k}\) upper bound proved in Lemma F.20.",
            "",
            "## Hyper-gradient Mechanism Checks",
            "",
            "- **Pseudoinverse error** is "
            r"\(\left|\nabla_\theta f-\nabla_{\theta x}^2g"
            r"(\nabla_{xx}^2g)^\dagger\nabla_xf-F'(\theta)\right|\) "
            "at the exact optimistic manifold points. Its maximum is "
            f"`{diagnostics['pseudoinverse_error_max']:.2e}`.",
            "- **Ridge bias** is the absolute difference between the dense "
            "ridge hyper-gradient and the exact-pseudoinverse hyper-gradient. "
            "Its observed log-log slope is "
            f"`{diagnostics['ridge_bias_slope']:.3f}` (first-order "
            "prediction: `1`).",
            "- **CG solution error** is "
            r"\(\|v_{\mathrm{CG}}-v_{\mathrm{dense}}\|_2\) for the same "
            "ridge system at rotated non-optimistic manifold points. Its mean "
            "at absolute tolerance `1e-8` is "
            f"`{diagnostics['cg_solution_error_mean_at_1e-8']:.2e}`.",
            "- At `k=2`, `theta=1`, `lambda=1e-3`, `h=0.01`, `K=100`, and "
            "`N=32`, the ULA-selected and independently exact-Gibbs-selected "
            "hyper-gradient MSEs are "
            f"`{diagnostics['ula_hypergradient_mse']:.3f}` and "
            f"`{diagnostics['exact_gibbs_hypergradient_mse']:.3f}`. Their "
            "relative difference is "
            f"`{100 * diagnostics['relative_mse_difference']:.1f}%`.",
            "- Active-radial/angular/normal marginal KS distances are "
            f"`{diagnostics['marginal_ks']['active_radial']:.3f}` / "
            f"`{diagnostics['marginal_ks']['angular']:.3f}` / "
            f"`{diagnostics['marginal_ks']['normal']:.3f}`. These marginal "
            "diagnostics do **not** establish the full "
            "R\\'enyi-divergence guarantee.",
            "",
            "The complete records, trajectories, oracle counts, timing curves, "
            "and diagnostic CSV files are included in this directory. "
            "Mechanism diagnostics use controlled sample/iteration budgets, "
            "not wall-clock matching.",
        ]
    )
    return "\n".join(lines) + "\n"


def validate_scientific_configurations(package_root: Path) -> None:
    default_path = package_root / "config" / "default_config.json"
    k10_path = package_root / "config" / "k10_config.json"
    pilot_path = package_root / "config" / "pilot_grid.json"
    default = read_json(default_path)
    k10 = read_json(k10_path)
    pilot = read_json(pilot_path)
    validate_config(default)
    validate_config(k10)

    if tuple(default["experiment"]["k_values"]) != BASE_K_VALUES:
        raise AssertionError("default configuration must contain k={1,2,4}")
    if tuple(k10["experiment"]["k_values"]) != (10,):
        raise AssertionError("isolated k=10 configuration is malformed")
    for config in (default, k10):
        problem = config["problem"]
        if int(problem["ambient_dimension"]) != 20:
            raise AssertionError("ambient dimension must be 20")
        if float(problem["theta_bound"]) != 3.0:
            raise AssertionError("theta domain must be [-3,3]")
        if tuple(config["experiment"]["seeds"]) != FINAL_SEEDS:
            raise AssertionError("final seeds must be 1000--1009")
        if float(config["experiment"]["budget_seconds"]) != 2.0:
            raise AssertionError("end-to-end budget must be two seconds")
        if tuple(config["experiment"]["methods"]) != METHODS:
            raise AssertionError("method set changed")
        common = config["common"]
        if (
            common["initialization"] != "scaled_gaussian_sphere"
            or float(common["initialization_center_radius"]) != 2.0
            or float(common["initialization_temperature"]) != 1e-3
            or float(common["initialization_variance_scale"]) != 1.0
        ):
            raise AssertionError("analyzed lambda-scaled warm start changed")
        hg_ms = config["hg_ms"]
        if (
            int(hg_ms["branches"]) != 16
            or int(hg_ms["ula_steps"]) != 100
            or float(hg_ms["ula_step_size"]) != 0.01
            or float(hg_ms["temperature"]) != 1e-3
            or float(hg_ms["ridge"]) != 0.3
            or float(hg_ms["cg_absolute_tolerance"]) != 1e-8
        ):
            raise AssertionError("HG-MS scientific configuration changed")
    for key in ("problem", "common", *METHODS):
        if default[key] != k10[key]:
            raise AssertionError(
                f"k=10 must use the same untuned {key} configuration"
            )
    if (
        tuple(pilot["pilot_seeds"]) != PILOT_SEEDS
        or float(pilot["pilot_budget_seconds"]) != 2.0
        or pilot["selection_metric"] != "hyperobjective_gap"
    ):
        raise AssertionError("pilot seed/budget/objective protocol changed")
    for method in METHODS:
        if len(pilot[method]) != 8:
            raise AssertionError(f"{method} must have eight pilot candidates")


def scan_text_for_identifiers(paths: Iterable[Path]) -> None:
    violations = []
    for path in paths:
        if path.suffix.lower() not in {
            ".csv",
            ".json",
            ".md",
            ".py",
            ".sh",
            ".sbatch",
            ".txt",
            ".yml",
            ".yaml",
        }:
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_TEXT_PATTERNS.items():
            if pattern.search(text):
                violations.append(f"{path}: {label}")
    if violations:
        raise AssertionError(
            "identifying metadata found:\n" + "\n".join(violations)
        )


def implementation_files(package_root: Path) -> List[Path]:
    files = []
    for path in package_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root)
        if (
            relative.parts[0] in {"outputs", "results", "__pycache__"}
            or "__pycache__" in relative.parts
            or path.suffix in {".pyc", ".out", ".err", ".log"}
        ):
            continue
        files.append(path)
    return sorted(files)


def build_manifest(
    package_root: Path,
    results_directory: Path,
    removed_fields: Mapping[str, int],
    counts: Mapping[str, int],
) -> Dict[str, Any]:
    result_checksums = {}
    for path in sorted(results_directory.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        result_checksums[str(path.relative_to(results_directory))] = (
            sha256_file(path)
        )
    implementation_checksums = {
        str(path.relative_to(package_root)): sha256_file(path)
        for path in implementation_files(package_root)
    }
    return {
        "package_version": PACKAGE_VERSION,
        "manifest_schema_version": 1,
        "configurations": {
            "fixed_d20": "config/default_config.json",
            "fixed_d20_k10": "config/k10_config.json",
            "pilot_grid": "config/pilot_grid.json",
            "frozen_selected_config": "results/pilot/selected_config.json",
        },
        "dimensions": list(K_VALUES),
        "ambient_dimension": 20,
        "final_seeds": list(FINAL_SEEDS),
        "pilot_seeds": list(PILOT_SEEDS),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "repetitions": BOOTSTRAP_REPETITIONS,
            "interval": "symmetric 95% half-width from absolute bootstrap mean deviations",
        },
        "result_counts": dict(counts),
        "removed_environment_fields": dict(sorted(removed_fields.items())),
        "result_checksums_sha256": result_checksums,
        "implementation_checksums_sha256": implementation_checksums,
    }


def validate_manifest(
    package_root: Path,
    results_directory: Path,
) -> None:
    manifest = read_json(results_directory / "MANIFEST.json")
    if manifest["package_version"] != PACKAGE_VERSION:
        raise AssertionError("manifest package version mismatch")
    for relative, expected in manifest["result_checksums_sha256"].items():
        path = results_directory / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise AssertionError(f"result checksum mismatch: {relative}")
    expected_result_paths = {
        str(path.relative_to(results_directory))
        for path in results_directory.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    if expected_result_paths != set(manifest["result_checksums_sha256"]):
        raise AssertionError("manifest result-file inventory mismatch")
    for relative, expected in manifest[
        "implementation_checksums_sha256"
    ].items():
        path = package_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise AssertionError(
                f"implementation checksum mismatch: {relative}"
            )


def validate_public_results(
    package_root: Path,
    results_directory: Path,
    verify_manifest_checksums: bool = True,
    verify_saved_report: bool = True,
) -> Dict[str, Any]:
    validate_scientific_configurations(package_root)
    selected_path = results_directory / "pilot" / "selected_config.json"
    if not selected_path.is_file():
        raise AssertionError("missing frozen pilot-selected configuration")
    selected = read_json(selected_path)
    for method in METHODS:
        provenance = selected["provenance"][method]
        if (
            tuple(provenance["pilot_seeds"]) != PILOT_SEEDS
            or float(provenance["pilot_budget_seconds"]) != 2.0
            or provenance["selection_metric"] != "hyperobjective_gap"
        ):
            raise AssertionError(f"invalid pilot provenance for {method}")

    runs = discover_public_runs(results_directory / "raw")
    expected_keys = {
        (k, seed, method)
        for k in K_VALUES
        for seed in FINAL_SEEDS
        for method in METHODS
    }
    observed_keys = [
        (int(run["problem"]["k"]), int(run["seed"]), run["method"])
        for run in runs
    ]
    if len(runs) != 200 or set(observed_keys) != expected_keys:
        raise AssertionError("public raw runs are incomplete")
    duplicates = [
        key for key, count in Counter(observed_keys).items() if count != 1
    ]
    if duplicates:
        raise AssertionError(f"duplicate public runs: {duplicates}")

    loaded_configs = {
        **{
            k: load_config(
                package_root / "config" / "default_config.json",
                selected_path,
            )
            for k in BASE_K_VALUES
        },
        10: load_config(
            package_root / "config" / "k10_config.json",
            selected_path,
        ),
    }
    for run in runs:
        if ENVIRONMENT_KEYS.intersection(
            key.lower() for key in run
        ):
            raise AssertionError("environment fields remain in public run")
        k = int(run["problem"]["k"])
        config = loaded_configs[k]
        if (
            int(run["problem"]["ambient_dimension"]) != 20
            or run["common_config"] != config["common"]
            or run["method_config"] != config[run["method"]]
            or float(run["budget_seconds"]) != 2.0
        ):
            raise AssertionError(
                f"configuration mismatch for k={k}, seed={run['seed']}, "
                f"method={run['method']}"
            )
        final = run["final"]
        if int(final.get("cg_breakdowns", 0)) != 0:
            raise AssertionError("CG breakdown found")
        if int(final.get("cg_tolerance_failures", 0)) != 0:
            raise AssertionError("CG tolerance failure found")
        allowed_overrun = max(0.1, 0.05 * float(run["budget_seconds"]))
        if float(run["budget_overrun_seconds"]) > allowed_overrun:
            raise AssertionError("unacceptable wall-clock overrun found")

    for name in ("fixed_d20", "fixed_d20_k10"):
        report = read_json(
            results_directory / "validation" / f"{name}.json"
        )
        if not report["complete"] or report["warnings"]:
            raise AssertionError(f"incomplete source validation: {name}")

    final, paired, curves = regenerate_public_summaries(runs)
    assert_records_match(
        final,
        read_csv(results_directory / "tables" / "final_summary.csv"),
        "final summary",
    )
    assert_records_match(
        paired,
        read_csv(results_directory / "tables" / "paired_comparisons.csv"),
        "paired comparisons",
    )
    assert_records_match(
        curves,
        read_csv(results_directory / "tables" / "wallclock_curves.csv"),
        "wall-clock curves",
    )
    per_seed = final_per_seed_records(runs)
    assert_records_match(
        per_seed,
        read_csv(results_directory / "tables" / "final_per_seed.csv"),
        "per-seed summary",
    )

    selection = read_csv(
        results_directory / "mechanisms" / "selection_error.csv"
    )
    ridge = read_csv(
        results_directory / "mechanisms" / "ridge_cg_error.csv"
    )
    ula = read_csv(
        results_directory / "mechanisms" / "ula_gibbs_error.csv"
    )
    if len(selection) != 48 or len(ridge) != 120 or len(ula) != 24:
        raise AssertionError("mechanism diagnostic row counts changed")
    diagnostics = theory_diagnostics(selection, ridge, ula)
    expected_observed = {1: -2.016, 2: -1.016, 4: -0.599, 10: -0.158}
    expected_quadrature = {
        1: -1.987,
        2: -1.000,
        4: -0.598,
        10: -0.156,
    }
    for k in K_VALUES:
        slopes = diagnostics["selection_slopes"][k]
        if round(slopes["observed"], 3) != expected_observed[k]:
            raise AssertionError(f"observed slope mismatch for k={k}")
        if round(slopes["quadrature"], 3) != expected_quadrature[k]:
            raise AssertionError(f"quadrature slope mismatch for k={k}")

    final_index = {
        (int(row["k"]), row["method"]): row for row in final
    }
    lowest_cells = 0
    for k in K_VALUES:
        for metric in DISPLAY_METRICS:
            values = {
                method: float(final_index[(k, method)][f"{metric}_mean"])
                for method in METHODS
            }
            if min(values, key=values.get) != "hg_ms":
                raise AssertionError(
                    f"HG-MS is not lowest for k={k}, metric={metric}"
                )
            lowest_cells += 1
    k10 = final_index[(10, "hg_ms")]
    approximate = {
        "hyperobjective_gap_mean": 0.007957,
        "hyperobjective_gap_ci95_half_width": 0.00281,
        "exact_hypergradient_abs_mean": 0.1673,
        "exact_hypergradient_abs_ci95_half_width": 0.0359,
        "optimistic_distance_mean": 1.103,
        "optimistic_distance_ci95_half_width": 0.0947,
    }
    for key, target in approximate.items():
        if not math.isclose(
            float(k10[key]),
            target,
            rel_tol=5e-3,
            abs_tol=5e-5,
        ):
            raise AssertionError(f"k=10 HG-MS mismatch for {key}")

    expected_results = render_results_markdown(
        final,
        paired,
        selection,
        ridge,
        ula,
    )
    actual_results = (
        results_directory / "RESULTS.md"
    ).read_text(encoding="utf-8")
    if expected_results != actual_results:
        raise AssertionError(
            "RESULTS.md does not match regenerated means and intervals"
        )
    canonical_results = package_root / "results"
    if results_directory.resolve() == canonical_results.resolve():
        expected_root_results = render_results_markdown(
            final,
            paired,
            selection,
            ridge,
            ula,
            artifact_prefix="results/",
        )
        actual_root_results = (
            package_root / "RESULTS.md"
        ).read_text(encoding="utf-8")
        if expected_root_results != actual_root_results:
            raise AssertionError(
                "package-level RESULTS.md does not match regenerated values"
            )
    expected_scaling = build_summary(selection, ridge, ula)
    actual_scaling = (
        results_directory / "mechanisms" / "scaling_summary.md"
    ).read_text(encoding="utf-8")
    if expected_scaling != actual_scaling:
        raise AssertionError("mechanism scaling summary is stale")

    scan_text_for_identifiers(
        path for path in results_directory.rglob("*") if path.is_file()
    )
    scan_text_for_identifiers(
        path
        for path in implementation_files(package_root)
        if path.name != "public_artifact.py"
    )
    report = {
        "package_version": PACKAGE_VERSION,
        "complete": True,
        "warnings": [],
        "checks": {
            "raw_end_to_end_files": len(runs),
            "unique_method_dimension_seed_cells": len(set(observed_keys)),
            "runs_per_method_and_k": 10,
            "cg_breakdowns": 0,
            "cg_tolerance_failures": 0,
            "unacceptable_wall_clock_overruns": 0,
            "final_per_seed_rows": len(per_seed),
            "final_summary_rows": len(final),
            "paired_comparison_rows": len(paired),
            "wallclock_curve_rows": len(curves),
            "selection_rows": len(selection),
            "ridge_cg_rows": len(ridge),
            "ula_gibbs_rows": len(ula),
            "hg_ms_lowest_mean_cells": lowest_cells,
        },
        "selection_slopes": {
            str(k): {
                key: value
                for key, value in diagnostics["selection_slopes"][k].items()
            }
            for k in K_VALUES
        },
    }
    if verify_saved_report:
        saved = read_json(
            results_directory / "validation" / "public_validation.json"
        )
        if saved != report:
            raise AssertionError("saved public validation report is stale")
    if verify_manifest_checksums:
        validate_manifest(package_root, results_directory)
    return report
