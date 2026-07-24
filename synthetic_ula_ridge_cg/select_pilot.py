from __future__ import annotations

import argparse
from pathlib import Path

from .configuration import (
    PACKAGE_ROOT,
    SCHEMA_VERSION,
    atomic_write_json,
    load_config,
    read_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independently run pilot sweeps.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default_config.json",
    )
    parser.add_argument(
        "--pilot-root", type=Path, default=PACKAGE_ROOT / "outputs" / "pilot"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "pilot" / "selected_config.json",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    methods = list(config["experiment"]["methods"])
    selected = {}
    provenance = {}
    reference_protocol = None
    for method in methods:
        summary_path = args.pilot_root / method / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing pilot summary: {summary_path}")
        summary = read_json(summary_path)
        if int(summary.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(
                f"pilot summary schema mismatch in {summary_path}: "
                f"expected {SCHEMA_VERSION}, found {summary.get('schema_version')}"
            )
        if summary.get("method") != method:
            raise ValueError(
                f"pilot method mismatch in {summary_path}: "
                f"{summary.get('method')} != {method}"
            )
        if len(summary.get("candidates", [])) != 8:
            raise ValueError(f"expected eight candidates in {summary_path}")
        selected_index = int(summary["selected_index"])
        if selected_index < 0 or selected_index >= 8:
            raise ValueError(f"invalid selected index in {summary_path}")
        selected_candidate = summary["candidates"][selected_index]
        if selected_candidate["override"] != summary["selected_override"]:
            raise ValueError(f"selected override mismatch in {summary_path}")
        protocol = (
            int(summary["k"]),
            tuple(int(seed) for seed in summary["pilot_seeds"]),
            float(summary["pilot_budget_seconds"]),
            summary["selection_metric"],
        )
        if reference_protocol is None:
            reference_protocol = protocol
        elif protocol != reference_protocol:
            raise ValueError(
                f"pilot protocol mismatch in {summary_path}: "
                f"{protocol} != {reference_protocol}"
            )
        selected[method] = summary["selected_override"]
        provenance[method] = {
            "selected_index": selected_index,
            "pilot_budget_seconds": summary["pilot_budget_seconds"],
            "pilot_seeds": summary["pilot_seeds"],
            "selection_metric": summary["selection_metric"],
        }
    atomic_write_json(
        args.output,
        {
            "schema_version": SCHEMA_VERSION,
            "selected_overrides": selected,
            "provenance": provenance,
        },
    )
    print(f"[pilot] wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
