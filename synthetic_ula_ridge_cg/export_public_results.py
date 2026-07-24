from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .analyze_mechanisms import build_summary
from .configuration import PACKAGE_ROOT, read_json
from .public_artifact import (
    K_VALUES,
    assert_records_match,
    assert_sanitization_is_lossless,
    build_manifest,
    discover_public_runs,
    final_per_seed_records,
    read_csv,
    regenerate_public_summaries,
    render_results_markdown,
    sanitize_json,
    validate_public_results,
    write_csv,
    write_json,
)


def _copy_sanitized_json(
    source: Path,
    destination: Path,
    removed_fields: Counter[str],
) -> None:
    original = read_json(source)
    sanitized = sanitize_json(original, removed_fields)
    assert_sanitization_is_lossless(original, sanitized)
    write_json(destination, sanitized)
    reloaded = read_json(destination)
    assert_sanitization_is_lossless(original, reloaded)


def _source_summary_records(
    fixed_d20: Path,
    fixed_d20_k10: Path,
    filename: str,
) -> List[Dict[str, str]]:
    records = []
    for root in (fixed_d20, fixed_d20_k10):
        records.extend(read_csv(root / "summary" / filename))
    if filename == "final_summary.csv":
        records.sort(key=lambda row: (int(row["k"]), row["method"]))
    elif filename == "paired_comparisons.csv":
        records.sort(key=lambda row: (int(row["k"]), row["baseline"]))
    elif filename == "wallclock_curves.csv":
        records.sort(
            key=lambda row: (
                int(row["k"]),
                row["method"],
                row["metric"],
                float(row["elapsed_seconds"]),
            )
        )
    return records


def _validate_source_report(root: Path, label: str) -> Dict[str, Any]:
    report = read_json(root / "summary" / "validation.json")
    if not report.get("complete") or report.get("warnings"):
        raise RuntimeError(f"source validation is incomplete for {label}")
    return report


def export_results(
    fixed_d20: Path,
    fixed_d20_k10: Path,
    output_directory: Path,
) -> None:
    for root, label in (
        (fixed_d20, "fixed_d20"),
        (fixed_d20_k10, "fixed_d20_k10"),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"missing source output tree: {root}")
        _validate_source_report(root, label)
    if output_directory.resolve() in {
        fixed_d20.resolve(),
        fixed_d20_k10.resolve(),
    }:
        raise ValueError("public output directory must differ from source outputs")

    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True)
    removed_fields: Counter[str] = Counter()

    source_run_paths = sorted(
        (fixed_d20 / "main").glob("k_*/seed_*/*.json")
    ) + sorted(
        (fixed_d20_k10 / "main").glob("k_*/seed_*/*.json")
    )
    if len(source_run_paths) != 200:
        raise AssertionError(
            f"expected 200 source end-to-end files, found "
            f"{len(source_run_paths)}"
        )
    for source in source_run_paths:
        k = int(source.parents[1].name.removeprefix("k_"))
        seed = int(source.parent.name.removeprefix("seed_"))
        destination = (
            output_directory
            / "raw"
            / f"k_{k}"
            / f"seed_{seed}"
            / source.name
        )
        _copy_sanitized_json(source, destination, removed_fields)

    pilot_source = fixed_d20 / "pilot"
    _copy_sanitized_json(
        pilot_source / "selected_config.json",
        output_directory / "pilot" / "selected_config.json",
        removed_fields,
    )
    for method in (
        "hg_ms",
        "hg_baseline",
        "v_pbgd",
        "g_pbgd",
        "iaptt_gm",
    ):
        _copy_sanitized_json(
            pilot_source / method / "summary.json",
            output_directory / "pilot" / f"{method}_summary.json",
            removed_fields,
        )

    for root, label in (
        (fixed_d20, "fixed_d20"),
        (fixed_d20_k10, "fixed_d20_k10"),
    ):
        _copy_sanitized_json(
            root / "summary" / "validation.json",
            output_directory / "validation" / f"{label}.json",
            removed_fields,
        )

    fixed_selection = read_csv(
        fixed_d20 / "mechanisms" / "selection_error.csv"
    )
    k10_selection = read_csv(
        fixed_d20_k10 / "mechanisms" / "selection_error.csv"
    )
    selection = fixed_selection + k10_selection
    selection.sort(
        key=lambda row: (int(row["k"]), int(row["number_candidates"]))
    )
    if {int(row["k"]) for row in selection} != set(K_VALUES):
        raise AssertionError("selection diagnostics do not cover all k values")
    ridge = read_csv(fixed_d20 / "mechanisms" / "ridge_cg_error.csv")
    ula = read_csv(fixed_d20 / "mechanisms" / "ula_gibbs_error.csv")
    write_csv(
        output_directory / "mechanisms" / "selection_error.csv",
        selection,
    )
    write_csv(
        output_directory / "mechanisms" / "ridge_cg_error.csv",
        ridge,
    )
    write_csv(
        output_directory / "mechanisms" / "ula_gibbs_error.csv",
        ula,
    )
    _copy_sanitized_json(
        fixed_d20 / "mechanisms" / "metadata.json",
        output_directory / "mechanisms" / "metadata_fixed_d20.json",
        removed_fields,
    )
    _copy_sanitized_json(
        fixed_d20_k10 / "mechanisms" / "selection_metadata.json",
        output_directory / "mechanisms" / "metadata_k10_selection.json",
        removed_fields,
    )

    runs = discover_public_runs(output_directory / "raw")
    final, paired, curves = regenerate_public_summaries(runs)
    assert_records_match(
        final,
        _source_summary_records(
            fixed_d20,
            fixed_d20_k10,
            "final_summary.csv",
        ),
        "source versus sanitized final summary",
    )
    assert_records_match(
        paired,
        _source_summary_records(
            fixed_d20,
            fixed_d20_k10,
            "paired_comparisons.csv",
        ),
        "source versus sanitized paired comparisons",
    )
    assert_records_match(
        curves,
        _source_summary_records(
            fixed_d20,
            fixed_d20_k10,
            "wallclock_curves.csv",
        ),
        "source versus sanitized wall-clock curves",
    )
    per_seed = final_per_seed_records(runs)
    write_csv(
        output_directory / "tables" / "final_per_seed.csv",
        per_seed,
    )
    write_csv(
        output_directory / "tables" / "final_summary.csv",
        final,
    )
    write_csv(
        output_directory / "tables" / "paired_comparisons.csv",
        paired,
    )
    write_csv(
        output_directory / "tables" / "wallclock_curves.csv",
        curves,
    )
    results_markdown = render_results_markdown(
        final,
        paired,
        selection,
        ridge,
        ula,
    )
    (output_directory / "RESULTS.md").write_text(
        results_markdown,
        encoding="utf-8",
    )
    if output_directory.resolve() == (PACKAGE_ROOT / "results").resolve():
        (PACKAGE_ROOT / "RESULTS.md").write_text(
            render_results_markdown(
                final,
                paired,
                selection,
                ridge,
                ula,
                artifact_prefix="results/",
            ),
            encoding="utf-8",
        )
    (output_directory / "mechanisms" / "scaling_summary.md").write_text(
        build_summary(selection, ridge, ula),
        encoding="utf-8",
    )

    report = validate_public_results(
        PACKAGE_ROOT,
        output_directory,
        verify_manifest_checksums=False,
        verify_saved_report=False,
    )
    write_json(
        output_directory / "validation" / "public_validation.json",
        report,
    )
    counts = dict(report["checks"])
    counts["pilot_summary_files"] = 5
    manifest = build_manifest(
        PACKAGE_ROOT,
        output_directory,
        removed_fields,
        counts,
    )
    write_json(output_directory / "MANIFEST.json", manifest)
    validate_public_results(PACKAGE_ROOT, output_directory)
    print(
        f"[export] wrote anonymous public artifact to "
        f"{output_directory.resolve()}"
    )
    print(
        f"[export] raw_runs={len(runs)} "
        f"removed_environment_fields={sum(removed_fields.values())}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export and verify anonymous public synthetic results."
    )
    parser.add_argument(
        "--source-fixed-d20",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "fixed_d20",
    )
    parser.add_argument(
        "--source-k10",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "fixed_d20_k10",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PACKAGE_ROOT / "results",
    )
    args = parser.parse_args()
    export_results(
        args.source_fixed_d20,
        args.source_k10,
        args.output_directory,
    )


if __name__ == "__main__":
    main()
