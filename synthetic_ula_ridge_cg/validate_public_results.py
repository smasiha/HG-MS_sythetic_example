from __future__ import annotations

import argparse
from pathlib import Path

from .configuration import PACKAGE_ROOT
from .public_artifact import validate_public_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the committed anonymous synthetic artifact."
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=PACKAGE_ROOT / "results",
    )
    args = parser.parse_args()
    report = validate_public_results(PACKAGE_ROOT, args.results_directory)
    checks = report["checks"]
    print(
        "[public-validation] complete "
        f"raw_runs={checks['raw_end_to_end_files']} "
        f"summary_rows={checks['final_summary_rows']} "
        f"hg_ms_lowest_cells={checks['hg_ms_lowest_mean_cells']}"
    )


if __name__ == "__main__":
    main()
