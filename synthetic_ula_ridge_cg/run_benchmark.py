from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from .configuration import (
    PACKAGE_ROOT,
    SCHEMA_VERSION,
    atomic_write_json,
    load_config,
    make_problem,
    read_json,
)
from .methods import METHOD_LABELS, run_method


def parse_methods(value: str, defaults: List[str]) -> List[str]:
    if value.strip().lower() == "all":
        return list(defaults)
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(METHOD_LABELS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    return methods


def compatible_cached_run(
    path: Path,
    method: str,
    problem_metadata: dict,
    common_config: dict,
    method_config: dict,
    budget_seconds: float,
    seed: int,
) -> bool:
    try:
        result = read_json(path)
    except (OSError, ValueError):
        return False
    return (
        int(result.get("schema_version", -1)) == SCHEMA_VERSION
        and result.get("method") == method
        and result.get("problem") == problem_metadata
        and result.get("common_config") == common_config
        and result.get("method_config") == method_config
        and float(result.get("budget_seconds", -1.0)) == budget_seconds
        and int(result.get("seed", -1)) == seed
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all synthetic baselines under the same wall-clock budget."
    )
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget-seconds", type=float, default=None)
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default_config.json",
    )
    parser.add_argument("--selected-config", type=Path, default=None)
    parser.add_argument(
        "--output-root", type=Path, default=PACKAGE_ROOT / "outputs" / "main"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fixed-order", action="store_true")
    args = parser.parse_args()

    if args.budget_seconds is not None and args.budget_seconds <= 0:
        raise ValueError("--budget-seconds must be positive")
    config = load_config(args.config, args.selected_config)
    configured_methods = list(config["experiment"]["methods"])
    methods = parse_methods(args.methods, configured_methods)
    budget_seconds = float(
        args.budget_seconds
        if args.budget_seconds is not None
        else config["experiment"]["budget_seconds"]
    )
    if not args.fixed_order:
        order_rng = np.random.default_rng(args.seed + 99_991 * args.k)
        methods = [str(method) for method in order_rng.permutation(methods)]

    problem = make_problem(config, args.k)
    output_directory = args.output_root / f"k_{args.k}" / f"seed_{args.seed}"
    output_directory.mkdir(parents=True, exist_ok=True)
    print(
        f"[benchmark] k={args.k} seed={args.seed} "
        f"budget={budget_seconds:.3f}s/method order={methods}",
        flush=True,
    )

    for method in methods:
        output_path = output_directory / f"{method}.json"
        if output_path.exists() and not args.overwrite:
            if compatible_cached_run(
                output_path,
                method,
                problem.metadata(),
                dict(config["common"]),
                dict(config[method]),
                budget_seconds,
                args.seed,
            ):
                print(f"[benchmark] skip compatible {output_path}", flush=True)
                continue
            print(f"[benchmark] replace stale {output_path}", flush=True)
        print(f"[benchmark] start {METHOD_LABELS[method]}", flush=True)
        result = run_method(
            method=method,
            problem=problem,
            seed=args.seed,
            budget_seconds=budget_seconds,
            config=config,
        )
        atomic_write_json(output_path, result)
        final = result["final"]
        print(
            f"[benchmark] done {METHOD_LABELS[method]} elapsed={result['elapsed_seconds']:.3f}s "
            f"iters={final['iteration']} F_gap={final['hyperobjective_gap']:.6e} "
            f"|theta|={final['theta_error']:.6e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
