from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .problem import ManifoldBilevelProblem


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "default_config.json"
SCHEMA_VERSION = 2
METHODS = ("hg_ms", "hg_baseline", "v_pbgd", "g_pbgd", "iaptt_gm")
OPTIONAL_METHOD_KEYS = {
    "hg_ms": {"theta_learning_rate_end", "theta_learning_rate_schedule"},
    "hg_baseline": {"theta_learning_rate_end", "theta_learning_rate_schedule"},
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _require_positive(config: Mapping[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if float(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")


def validate_config(config: Mapping[str, Any]) -> None:
    experiment = config["experiment"]
    methods = list(experiment["methods"])
    if methods != list(METHODS):
        raise ValueError(f"experiment methods must be exactly {list(METHODS)}")
    k_values = [int(value) for value in experiment["k_values"]]
    seeds = [int(value) for value in experiment["seeds"]]
    if not k_values or any(value < 1 for value in k_values):
        raise ValueError("all manifold dimensions must be positive")
    if len(k_values) != len(set(k_values)) or len(seeds) != len(set(seeds)):
        raise ValueError("manifold dimensions and final seeds must be unique")
    if not seeds:
        raise ValueError("at least one final seed is required")
    _require_positive(experiment, ("budget_seconds",))
    for k in k_values:
        ManifoldBilevelProblem(k=k, **config["problem"])

    common = config["common"]
    if common["initialization"] not in {"scaled_gaussian_sphere", "cold"}:
        raise ValueError("unsupported lower-level initialization")
    if common["initialization"] == "scaled_gaussian_sphere":
        _require_positive(
            common,
            (
                "initialization_center_radius",
                "initialization_temperature",
                "initialization_variance_scale",
            ),
        )
    _require_positive(common, ("log_interval_seconds", "theta_clip"))
    if float(common["theta_clip"]) != float(config["problem"]["theta_bound"]):
        raise ValueError("theta_clip must equal the synthetic problem theta_bound")
    if float(common["lower_clip"]) < 0 or float(common["gradient_clip"]) < 0:
        raise ValueError("lower_clip and gradient_clip must be nonnegative")

    for method in ("hg_ms", "hg_baseline"):
        method_config = config[method]
        _require_positive(
            method_config,
            (
                "branches",
                "ula_steps",
                "ula_step_size",
                "temperature",
                "ridge",
                "cg_max_iterations",
                "theta_learning_rate",
            ),
        )
        if float(method_config["cg_absolute_tolerance"]) < 0:
            raise ValueError("cg_absolute_tolerance must be nonnegative")
        schedule = str(
            method_config.get("theta_learning_rate_schedule", "constant")
        )
        if schedule not in {"constant", "cosine"}:
            raise ValueError(f"unsupported {method} learning-rate schedule: {schedule}")
        learning_rate = float(method_config["theta_learning_rate"])
        final_learning_rate = float(
            method_config.get("theta_learning_rate_end", learning_rate)
        )
        if final_learning_rate <= 0:
            raise ValueError(f"{method} theta_learning_rate_end must be positive")
        if schedule == "constant" and final_learning_rate != learning_rate:
            raise ValueError(
                f"{method} constant schedule requires equal initial and final "
                "learning rates"
            )
        if schedule == "cosine" and final_learning_rate > learning_rate:
            raise ValueError(
                f"{method} cosine schedule must not increase the learning rate"
            )
        if method == "hg_ms" and int(method_config["branches"]) < 2:
            raise ValueError("HG-MS must use at least two candidates")
        if method == "hg_baseline" and int(method_config["branches"]) != 1:
            raise ValueError("HG-baseline must use exactly one candidate")
        if common["initialization"] == "scaled_gaussian_sphere" and float(
            method_config["temperature"]
        ) != float(common["initialization_temperature"]):
            raise ValueError(
                "HG temperature must equal the scaled-Gaussian "
                "initialization temperature"
            )

    _require_positive(
        config["v_pbgd"],
        (
            "theta_learning_rate",
            "lower_learning_rate",
            "virtual_learning_rate",
            "virtual_steps",
            "penalty",
        ),
    )
    _require_positive(
        config["g_pbgd"],
        ("theta_learning_rate", "lower_learning_rate", "penalty"),
    )
    _require_positive(
        config["iaptt_gm"],
        (
            "theta_learning_rate",
            "initialization_learning_rate",
            "inner_learning_rate",
            "inner_steps",
        ),
    )


def load_config(
    config_path: Optional[Path] = None,
    selected_config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    config = read_json(path)
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"config schema mismatch: expected {SCHEMA_VERSION}, "
            f"found {config.get('schema_version')}"
        )
    if selected_config_path is not None and not selected_config_path.exists():
        raise FileNotFoundError(f"selected config does not exist: {selected_config_path}")
    if selected_config_path is not None:
        selected = read_json(selected_config_path)
        if int(selected.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(
                f"selected-config schema mismatch in {selected_config_path}: "
                f"expected {SCHEMA_VERSION}, found {selected.get('schema_version')}"
            )
        overrides = selected.get("selected_overrides", selected)
        for method, method_overrides in overrides.items():
            if method in config and isinstance(method_overrides, Mapping):
                allowed = set(config[method]) | OPTIONAL_METHOD_KEYS.get(
                    method, set()
                )
                unknown = set(method_overrides) - allowed
                if unknown:
                    raise ValueError(
                        f"unknown selected override(s) for {method}: {sorted(unknown)}"
                    )
                config[method].update(method_overrides)
        config["selected_config_path"] = str(selected_config_path.resolve())
    validate_config(config)
    return config


def copy_with_override(
    config: Mapping[str, Any], method: str, override: Mapping[str, Any]
) -> Dict[str, Any]:
    updated = copy.deepcopy(config)
    allowed = set(updated[method]) | OPTIONAL_METHOD_KEYS.get(method, set())
    unknown = set(override) - allowed
    if unknown:
        raise ValueError(f"unknown override(s) for {method}: {sorted(unknown)}")
    updated[method].update(override)
    validate_config(updated)
    return updated


def make_problem(config: Mapping[str, Any], k: int) -> ManifoldBilevelProblem:
    return ManifoldBilevelProblem(k=k, **config["problem"])
