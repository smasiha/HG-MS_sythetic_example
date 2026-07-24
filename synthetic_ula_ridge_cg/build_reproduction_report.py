from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .analyze_mechanisms import build_summary
from .configuration import PACKAGE_ROOT
from .public_artifact import (
    FINAL_SEEDS,
    K_VALUES,
    METHODS,
    discover_public_runs,
    final_per_seed_records,
    read_csv,
    regenerate_public_summaries,
    render_results_markdown,
    write_csv,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine freshly reproduced fixed-dimension result batches."
    )
    parser.add_argument(
        "--fixed-root",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "reproduction" / "fixed_d20",
    )
    parser.add_argument(
        "--k10-root",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "reproduction" / "fixed_d20_k10",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "reproduction" / "report",
    )
    args = parser.parse_args()

    runs = discover_public_runs(args.fixed_root / "main")
    runs.extend(discover_public_runs(args.k10_root / "main"))
    keys = [
        (int(run["problem"]["k"]), int(run["seed"]), run["method"])
        for run in runs
    ]
    expected = {
        (k, seed, method)
        for k in K_VALUES
        for seed in FINAL_SEEDS
        for method in METHODS
    }
    duplicate_keys = [
        key for key, count in Counter(keys).items() if count != 1
    ]
    if len(runs) != 200 or set(keys) != expected or duplicate_keys:
        raise RuntimeError(
            "reproduction is incomplete: expected one run for every "
            "(k, seed, method) cell"
        )
    for run in runs:
        final = run["final"]
        if int(final.get("cg_breakdowns", 0)) != 0:
            raise RuntimeError("CG breakdown in reproduced results")
        if int(final.get("cg_tolerance_failures", 0)) != 0:
            raise RuntimeError("CG tolerance failure in reproduced results")

    final, paired, curves = regenerate_public_summaries(runs)
    per_seed = final_per_seed_records(runs)
    selection = read_csv(
        args.fixed_root / "mechanisms" / "selection_error.csv"
    ) + read_csv(
        args.k10_root / "mechanisms" / "selection_error.csv"
    )
    selection.sort(
        key=lambda row: (int(row["k"]), int(row["number_candidates"]))
    )
    ridge = read_csv(
        args.fixed_root / "mechanisms" / "ridge_cg_error.csv"
    )
    ula = read_csv(
        args.fixed_root / "mechanisms" / "ula_gibbs_error.csv"
    )

    tables = args.output_directory / "tables"
    mechanisms = args.output_directory / "mechanisms"
    write_csv(tables / "final_per_seed.csv", per_seed)
    write_csv(tables / "final_summary.csv", final)
    write_csv(tables / "paired_comparisons.csv", paired)
    write_csv(tables / "wallclock_curves.csv", curves)
    write_csv(mechanisms / "selection_error.csv", selection)
    write_csv(mechanisms / "ridge_cg_error.csv", ridge)
    write_csv(mechanisms / "ula_gibbs_error.csv", ula)
    (args.output_directory / "RESULTS.md").write_text(
        render_results_markdown(final, paired, selection, ridge, ula),
        encoding="utf-8",
    )
    (mechanisms / "scaling_summary.md").write_text(
        build_summary(selection, ridge, ula),
        encoding="utf-8",
    )
    write_json(
        args.output_directory / "validation.json",
        {
            "complete": True,
            "warnings": [],
            "raw_end_to_end_files": len(runs),
            "runs_per_method_and_k": 10,
            "final_summary_rows": len(final),
            "paired_comparison_rows": len(paired),
            "wallclock_curve_rows": len(curves),
            "selection_rows": len(selection),
            "ridge_cg_rows": len(ridge),
            "ula_gibbs_rows": len(ula),
        },
    )
    print(
        f"[reproduction-report] runs={len(runs)} "
        f"output={args.output_directory.resolve()}"
    )


if __name__ == "__main__":
    main()
