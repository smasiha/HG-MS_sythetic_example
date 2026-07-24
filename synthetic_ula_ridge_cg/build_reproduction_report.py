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
        description="Generate tables from a complete 60-second reproduction."
    )
    parser.add_argument(
        "--reproduction-root",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "reproduction",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    output_directory = (
        args.output_directory
        if args.output_directory is not None
        else args.reproduction_root / "report"
    )

    runs = discover_public_runs(args.reproduction_root / "main")
    keys = [
        (int(run["problem"]["k"]), int(run["seed"]), run["method"])
        for run in runs
    ]
    expected = {
        (k_value, seed, method)
        for k_value in K_VALUES
        for seed in FINAL_SEEDS
        for method in METHODS
    }
    duplicates = [
        key for key, count in Counter(keys).items() if count != 1
    ]
    if len(runs) != 100 or set(keys) != expected or duplicates:
        raise RuntimeError(
            "reproduction is incomplete: expected one run for every "
            "(k, seed, method) cell"
        )
    for run in runs:
        final_record = run["final"]
        if int(final_record.get("cg_breakdowns", 0)) != 0:
            raise RuntimeError("CG breakdown in reproduced results")
        if int(final_record.get("cg_tolerance_failures", 0)) != 0:
            raise RuntimeError("CG tolerance failure in reproduced results")

    final, paired, curves = regenerate_public_summaries(runs)
    per_seed = final_per_seed_records(runs)
    mechanism_directory = args.reproduction_root / "mechanisms"
    selection = read_csv(mechanism_directory / "selection_error.csv")
    ridge = read_csv(mechanism_directory / "ridge_cg_error.csv")
    ula = read_csv(mechanism_directory / "ula_gibbs_error.csv")

    tables = output_directory / "tables"
    mechanisms = output_directory / "mechanisms"
    write_csv(tables / "final_per_seed.csv", per_seed)
    write_csv(tables / "final_summary.csv", final)
    write_csv(tables / "paired_comparisons.csv", paired)
    write_csv(tables / "wallclock_curves.csv", curves)
    write_csv(mechanisms / "selection_error.csv", selection)
    write_csv(mechanisms / "ridge_cg_error.csv", ridge)
    write_csv(mechanisms / "ula_gibbs_error.csv", ula)
    (output_directory / "RESULTS.md").write_text(
        render_results_markdown(final, paired, selection, ridge, ula),
        encoding="utf-8",
    )
    (mechanisms / "scaling_summary.md").write_text(
        build_summary(selection, ridge, ula),
        encoding="utf-8",
    )
    write_json(
        output_directory / "validation.json",
        {
            "complete": True,
            "warnings": [],
            "raw_end_to_end_files": len(runs),
            "runs_per_method_and_k": len(FINAL_SEEDS),
            "final_summary_rows": len(final),
            "paired_comparison_rows": len(paired),
            "wallclock_curve_rows": len(curves),
            "selection_rows": len(selection),
            "ridge_cg_rows": len(ridge),
            "ula_gibbs_rows": len(ula),
            "markdown_result_reports": 1,
        },
    )
    print(
        f"[reproduction-report] runs={len(runs)} "
        f"output={output_directory.resolve()}"
    )


if __name__ == "__main__":
    main()
