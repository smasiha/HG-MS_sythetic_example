from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from .analyze_mechanisms import build_summary
from .configuration import PACKAGE_ROOT, read_json
from .public_artifact import (
    K_VALUES,
    METHODS,
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
    assert_sanitization_is_lossless(original, read_json(destination))


def _validate_source_report(source_root: Path) -> Dict[str, Any]:
    report = read_json(source_root / "summary" / "validation.json")
    if not report.get("complete") or report.get("warnings"):
        raise RuntimeError("final 60-second source validation is incomplete")
    return report


def _read_source_summary(source_root: Path, filename: str):
    records = read_csv(source_root / "summary" / filename)
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


def export_results(
    final_source: Path,
    pilot_source: Path,
    mechanism_source: Path,
    k10_selection_source: Path,
    output_directory: Path,
) -> None:
    required_directories = {
        "final source": final_source,
        "pilot source": pilot_source,
        "mechanism source": mechanism_source,
        "k=10 selection source": k10_selection_source,
    }
    for label, path in required_directories.items():
        if not path.is_dir():
            raise FileNotFoundError(f"missing {label}: {path}")
    _validate_source_report(final_source)
    if output_directory.resolve() == final_source.resolve():
        raise ValueError("public output directory must differ from source")

    source_run_paths = sorted(
        (final_source / "main").glob("k_*/seed_*/*.json")
    )
    if len(source_run_paths) != 100:
        raise AssertionError(
            f"expected 100 final source files, found {len(source_run_paths)}"
        )

    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True)
    removed_fields: Counter[str] = Counter()

    for source in source_run_paths:
        k_value = int(source.parents[1].name.removeprefix("k_"))
        seed = int(source.parent.name.removeprefix("seed_"))
        destination = (
            output_directory
            / "raw"
            / f"k_{k_value}"
            / f"seed_{seed}"
            / source.name
        )
        _copy_sanitized_json(source, destination, removed_fields)

    _copy_sanitized_json(
        pilot_source / "selected_config.json",
        output_directory / "pilot" / "selected_config.json",
        removed_fields,
    )
    for method in METHODS:
        _copy_sanitized_json(
            pilot_source / method / "summary.json",
            output_directory / "pilot" / f"{method}_summary.json",
            removed_fields,
        )
    _copy_sanitized_json(
        final_source / "summary" / "validation.json",
        output_directory / "validation" / "source_validation.json",
        removed_fields,
    )

    selection = read_csv(mechanism_source / "selection_error.csv")
    if k10_selection_source.resolve() != mechanism_source.resolve():
        selection.extend(
            read_csv(k10_selection_source / "selection_error.csv")
        )
    selection.sort(
        key=lambda row: (int(row["k"]), int(row["number_candidates"]))
    )
    if {int(row["k"]) for row in selection} != set(K_VALUES):
        raise AssertionError("selection diagnostics do not cover all k values")
    ridge = read_csv(mechanism_source / "ridge_cg_error.csv")
    ula = read_csv(mechanism_source / "ula_gibbs_error.csv")
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
        mechanism_source / "metadata.json",
        output_directory / "mechanisms" / "metadata_fixed_d20.json",
        removed_fields,
    )
    k10_metadata = k10_selection_source / "selection_metadata.json"
    if k10_metadata.is_file():
        _copy_sanitized_json(
            k10_metadata,
            output_directory / "mechanisms" / "metadata_k10_selection.json",
            removed_fields,
        )

    runs = discover_public_runs(output_directory / "raw")
    final, paired, curves = regenerate_public_summaries(runs)
    assert_records_match(
        final,
        _read_source_summary(final_source, "final_summary.csv"),
        "source versus sanitized final summary",
    )
    assert_records_match(
        paired,
        _read_source_summary(final_source, "paired_comparisons.csv"),
        "source versus sanitized paired comparisons",
    )
    assert_records_match(
        curves,
        _read_source_summary(final_source, "wallclock_curves.csv"),
        "source versus sanitized wall-clock curves",
    )

    write_csv(
        output_directory / "tables" / "final_per_seed.csv",
        final_per_seed_records(runs),
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
    (output_directory / "RESULTS.md").write_text(
        render_results_markdown(final, paired, selection, ridge, ula),
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
    counts["pilot_summary_files"] = len(METHODS)
    write_json(
        output_directory / "MANIFEST.json",
        build_manifest(
            PACKAGE_ROOT,
            output_directory,
            removed_fields,
            counts,
        ),
    )
    validate_public_results(PACKAGE_ROOT, output_directory)
    print(
        f"[export] wrote {len(runs)} anonymous 60-second runs to "
        f"{output_directory.resolve()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export and verify the anonymous 60-second artifact."
    )
    parser.add_argument(
        "--source-final",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "long60s_5seed",
    )
    parser.add_argument(
        "--source-pilot",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "fixed_d20" / "pilot",
    )
    parser.add_argument(
        "--source-mechanisms",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "fixed_d20" / "mechanisms",
    )
    parser.add_argument(
        "--source-k10-selection",
        type=Path,
        default=PACKAGE_ROOT
        / "outputs"
        / "fixed_d20_k10"
        / "mechanisms",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PACKAGE_ROOT / "results",
    )
    args = parser.parse_args()
    export_results(
        args.source_final,
        args.source_pilot,
        args.source_mechanisms,
        args.source_k10_selection,
        args.output_directory,
    )


if __name__ == "__main__":
    main()
