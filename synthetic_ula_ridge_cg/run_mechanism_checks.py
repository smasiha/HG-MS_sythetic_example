from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from scipy.special import roots_jacobi
from scipy.stats import ks_2samp

from .configuration import (
    PACKAGE_ROOT,
    SCHEMA_VERSION,
    atomic_write_json,
    load_config,
    make_problem,
)
from .methods import ridge_cg_hypergradient, ula_step
from .problem import ManifoldBilevelProblem


def write_csv(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    records = list(records)
    if not records:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(records[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def selection_quadrature_reference(
    problem: ManifoldBilevelProblem,
    theta: float,
    candidate_counts: List[int],
    nodes: int,
) -> Dict[int, Dict[str, float]]:
    exponent = 0.5 * problem.k - 1.0
    coordinates, weights = roots_jacobi(nodes, exponent, exponent)
    weights = weights / np.sum(weights)
    landscape_values = problem.restricted_landscape(theta, coordinates)
    descending = np.argsort(landscape_values)[::-1]
    survival = np.empty_like(weights)
    survival[descending] = (
        np.cumsum(weights[descending]) - 0.5 * weights[descending]
    )
    survival = np.clip(survival, np.finfo(np.float64).tiny, 1.0)
    radius = problem.radius(theta)
    squared_errors = 2.0 * radius * radius * (coordinates + 1.0)

    references: Dict[int, Dict[str, float]] = {}
    for number_candidates in candidate_counts:
        log_selected_density = (
            np.log(float(number_candidates))
            + np.log(weights)
            + (number_candidates - 1) * np.log(survival)
        )
        log_selected_density -= np.max(log_selected_density)
        selected_weights = np.exp(log_selected_density)
        selected_weights /= np.sum(selected_weights)
        references[number_candidates] = {
            "squared_error_mean": float(
                np.dot(selected_weights, squared_errors)
            ),
            "value_gap_mean": float(
                np.dot(selected_weights, landscape_values)
            ),
        }
    return references


def selection_check(
    config: Dict[str, Any],
    output_directory: Path,
    rng: np.random.Generator,
    quick: bool,
    k_values: List[int],
) -> List[Dict[str, Any]]:
    theta = 1.0
    candidate_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    if not quick:
        candidate_counts.extend([512, 1024, 2048])
    repetitions = 250 if quick else 2_000
    quadrature_nodes = 8_192 if quick else 32_768
    records: List[Dict[str, Any]] = []

    for k in k_values:
        problem = make_problem(config, k)
        radius = problem.radius(theta)
        target_direction = -problem.direction(theta)[: problem.y_dim]
        quadrature = selection_quadrature_reference(
            problem,
            theta,
            candidate_counts,
            quadrature_nodes,
        )
        for number_candidates in candidate_counts:
            squared_errors = np.empty(repetitions, dtype=np.float64)
            value_gaps = np.empty(repetitions, dtype=np.float64)
            chunk_size = min(256, repetitions)
            for start in range(0, repetitions, chunk_size):
                current = min(chunk_size, repetitions - start)
                directions = rng.normal(
                    size=(current, number_candidates, problem.y_dim)
                )
                directions /= np.linalg.norm(directions, axis=2, keepdims=True)
                orientation = (
                    directions
                    @ problem.direction(theta)[: problem.y_dim]
                )
                upper_values = problem.restricted_landscape(
                    theta, orientation
                )
                selected_indices = np.argmin(upper_values, axis=1)
                selected = directions[np.arange(current), selected_indices]
                delta = radius * (selected - target_direction)
                squared_errors[start : start + current] = np.sum(delta * delta, axis=1)
                value_gaps[start : start + current] = (
                    upper_values[np.arange(current), selected_indices]
                )

            records.append(
                {
                    "k": k,
                    "number_candidates": number_candidates,
                    "repetitions": repetitions,
                    "observed_squared_error_mean": float(np.mean(squared_errors)),
                    "observed_squared_error_se": float(
                        np.std(squared_errors, ddof=1) / np.sqrt(repetitions)
                    ),
                    "quadrature_nodes": quadrature_nodes,
                    "reference_squared_error_mean": quadrature[
                        number_candidates
                    ]["squared_error_mean"],
                    "observed_value_gap_mean": float(np.mean(value_gaps)),
                    "observed_value_gap_se": float(
                        np.std(value_gaps, ddof=1) / np.sqrt(repetitions)
                    ),
                    "reference_value_gap_mean": quadrature[
                        number_candidates
                    ]["value_gap_mean"],
                    "theorem_rate_proxy_N_minus_1_over_k": float(
                        number_candidates ** (-1.0 / k)
                    ),
                    "sharp_rate_proxy_N_minus_2_over_k": float(
                        number_candidates ** (-2.0 / k)
                    ),
                }
            )
    write_csv(output_directory / "selection_error.csv", records)
    return records


def ridge_cg_check(
    config: Dict[str, Any],
    output_directory: Path,
    quick: bool,
) -> List[Dict[str, Any]]:
    problem = make_problem(config, 2)
    theta_values = [0.5, 1.0] if quick else [0.5, 1.0, 2.0]
    ridge_values = [0.3, 0.1, 0.03, 0.01] if quick else [0.3, 0.1, 0.03, 0.01, 0.003]
    tolerances = [1.0, 1e-1, 1e-4, 1e-8]
    records: List[Dict[str, Any]] = []

    for theta in theta_values:
        radius = problem.radius(theta)
        rotation = 0.3
        exact_point = problem.exact_lower_point(theta)
        nonoptimistic_point = -radius * (
            np.cos(rotation) * problem.direction(theta)
            + np.sin(rotation) * problem.direction_prime(theta)
        )
        exact_hypergradient = problem.exact_hypergradient(theta)
        for point_type, lower_point in (
            ("exact_optimistic", exact_point),
            ("manifold_nonoptimistic", nonoptimistic_point),
        ):
            hessian = problem.hessian_g(theta, lower_point)
            right_hand_side = problem.grad_f_x(theta, lower_point)
            mixed = problem.mixed_grad_g_theta(theta, lower_point)
            direct = float(problem.partial_f_theta(theta, lower_point))
            pseudoinverse_solution = np.linalg.pinv(hessian, rcond=1e-12).dot(
                right_hand_side
            )
            pseudoinverse_hypergradient = direct - float(
                np.dot(mixed, pseudoinverse_solution)
            )

            for ridge in ridge_values:
                ridge_matrix = hessian + ridge * np.eye(problem.dim)
                dense_solution = np.linalg.solve(ridge_matrix, right_hand_side)
                dense_hypergradient = direct - float(np.dot(mixed, dense_solution))
                for tolerance in tolerances:
                    cg_hypergradient, cg_solution, info = ridge_cg_hypergradient(
                        problem,
                        theta,
                        lower_point,
                        ridge,
                        tolerance,
                        max_iterations=problem.dim * 4,
                    )
                    records.append(
                        {
                            "theta": theta,
                            "point_type": point_type,
                            "selected_point_squared_error": float(
                                np.sum((lower_point - exact_point) ** 2)
                            ),
                            "ridge_gamma": ridge,
                            "cg_absolute_tolerance_eta": tolerance,
                            "cg_iterations": info["iterations"],
                            "cg_absolute_residual": info["absolute_residual"],
                            "cg_relative_residual": info["relative_residual"],
                            "cg_breakdown": info["breakdown"],
                            "exact_hypergradient": exact_hypergradient,
                            "pseudoinverse_hypergradient": pseudoinverse_hypergradient,
                            "pseudoinverse_formula_error": abs(
                                pseudoinverse_hypergradient - exact_hypergradient
                            ),
                            "dense_ridge_hypergradient": dense_hypergradient,
                            "ridge_bias_abs": abs(
                                dense_hypergradient - pseudoinverse_hypergradient
                            ),
                            "cg_hypergradient": cg_hypergradient,
                            "cg_solution_error_l2": float(
                                np.linalg.norm(cg_solution - dense_solution)
                            ),
                            "cg_solver_error_abs": abs(
                                cg_hypergradient - dense_hypergradient
                            ),
                            "total_hypergradient_error_abs": abs(
                                cg_hypergradient - exact_hypergradient
                            ),
                            "ridge_proxy_gamma": ridge,
                            "cg_proxy_eta_over_gamma": tolerance / ridge,
                        }
                    )
    write_csv(output_directory / "ridge_cg_error.csv", records)
    return records


def _estimate_hypergradient(
    problem: ManifoldBilevelProblem,
    theta: float,
    lower_point: np.ndarray,
    ridge: float,
    tolerance: float,
) -> tuple[float, Dict[str, Any]]:
    estimate, _, info = ridge_cg_hypergradient(
        problem,
        theta,
        lower_point,
        ridge,
        tolerance,
        max_iterations=problem.dim * 4,
    )
    return estimate, info


def ula_check(
    config: Dict[str, Any],
    output_directory: Path,
    rng: np.random.Generator,
    quick: bool,
) -> List[Dict[str, Any]]:
    problem = make_problem(config, 2)
    theta = 1.0
    temperatures = [1e-2, 1e-3]
    step_sizes = [0.005, 0.01]
    ula_step_counts = [10, 100] if quick else [10, 100, 500]
    initialization_modes = ["random-centers", "single-center"]
    repetitions = 50 if quick else 250
    reference_repetitions = 1_000 if quick else 5_000
    candidates = 16 if quick else 32
    cg_absolute_tolerance = 1e-8
    initialization_variance_scale = float(
        config["common"]["initialization_variance_scale"]
    )
    records: List[Dict[str, Any]] = []

    for temperature in temperatures:
        ridge = max(0.3, 10.0 * np.sqrt(temperature))
        target = problem.sample_exact_gibbs(
            theta,
            max(5_000, reference_repetitions * candidates),
            rng,
            temperature,
        )
        target_radii = np.linalg.norm(target[:, : problem.y_dim], axis=1)
        target_normal_radii = np.linalg.norm(
            target[:, problem.y_dim :],
            axis=1,
        )
        target_directions = target[:, : problem.y_dim] / target_radii[:, None]
        target_angles = (
            target_directions
            @ problem.direction(theta)[: problem.y_dim]
        )
        target_groups = target[: reference_repetitions * candidates].reshape(
            reference_repetitions, candidates, problem.dim
        )
        target_selected_indices = np.argmin(
            problem.f(theta, target_groups), axis=1
        )
        target_selected = target_groups[
            np.arange(reference_repetitions), target_selected_indices
        ]
        target_selected_squared_errors = np.sum(
            (target_selected - problem.exact_lower_point(theta)) ** 2, axis=1
        )
        target_hypergradient_errors = []
        for point in target_selected:
            estimate, info = _estimate_hypergradient(
                problem, theta, point, ridge, cg_absolute_tolerance
            )
            if info["breakdown"]:
                raise RuntimeError(
                    "CG broke down on an exact Gibbs reference sample; "
                    "increase the temperature-coupled ridge"
                )
            target_hypergradient_errors.append(
                abs(estimate - problem.exact_hypergradient(theta))
            )
        target_hypergradient_error_mean = float(
            np.mean(target_hypergradient_errors)
        )
        target_hypergradient_squared_error_mean = float(
            np.mean(np.square(target_hypergradient_errors))
        )

        for step_size in step_sizes:
            for ula_steps in ula_step_counts:
                for initialization_mode in initialization_modes:
                    if initialization_mode == "random-centers":
                        chains = problem.sample_scaled_gaussian_sphere(
                            float(config["common"]["initialization_center_radius"]),
                            repetitions * candidates,
                            rng,
                            temperature,
                            initialization_variance_scale,
                        )
                    else:
                        chains = problem.sample_scaled_gaussian_at_point(
                            float(config["common"]["initialization_center_radius"])
                            * problem.direction(theta),
                            repetitions * candidates,
                            rng,
                            temperature,
                            initialization_variance_scale,
                        )
                    chains = chains.reshape(repetitions, candidates, problem.dim)
                    for _ in range(ula_steps):
                        flat = chains.reshape(-1, problem.dim)
                        flat[...] = ula_step(
                            problem,
                            theta,
                            flat,
                            step_size,
                            temperature,
                            rng.normal(size=flat.shape),
                        )

                    flat = chains.reshape(-1, problem.dim)
                    radii = np.linalg.norm(flat[:, : problem.y_dim], axis=1)
                    normal_radii = np.linalg.norm(
                        flat[:, problem.y_dim :],
                        axis=1,
                    )
                    directions = flat[:, : problem.y_dim] / np.maximum(
                        radii[:, None], 1e-15
                    )
                    angles = (
                        directions
                        @ problem.direction(theta)[: problem.y_dim]
                    )
                    selected_indices = np.argmin(problem.f(theta, chains), axis=1)
                    selected = chains[np.arange(repetitions), selected_indices]
                    selected_squared_errors = np.sum(
                        (selected - problem.exact_lower_point(theta)) ** 2, axis=1
                    )
                    selected_value_gaps = (
                        problem.f(theta, selected)
                        - problem.exact_hyperobjective(theta)
                    )
                    selected_tube_distances = np.asarray(
                        [
                            problem.manifold_distance(theta, point)
                            for point in selected
                        ]
                    )
                    hypergradient_errors = []
                    cg_iterations = []
                    cg_breakdowns = 0
                    for point in selected:
                        estimate, info = _estimate_hypergradient(
                            problem, theta, point, ridge, cg_absolute_tolerance
                        )
                        hypergradient_errors.append(
                            abs(estimate - problem.exact_hypergradient(theta))
                        )
                        cg_iterations.append(info["iterations"])
                        cg_breakdowns += int(info["breakdown"])

                    ks_statistic = float(
                        ks_2samp(radii, target_radii, method="asymp").statistic
                    )
                    angular_ks_statistic = float(
                        ks_2samp(angles, target_angles, method="asymp").statistic
                    )
                    normal_ks_statistic = float(
                        ks_2samp(
                            normal_radii,
                            target_normal_radii,
                            method="asymp",
                        ).statistic
                    )
                    sampler_discrepancy_proxy = max(
                        ks_statistic,
                        angular_ks_statistic,
                        normal_ks_statistic,
                    )
                    temperature_proxy = float(
                        temperature * np.log1p(candidates)
                        + np.sqrt(temperature * np.log1p(candidates))
                    )
                    selection_proxy = float(candidates ** (-1.0 / problem.k))
                    theorem_shape_proxy = float(
                        (
                            temperature_proxy
                            + selection_proxy
                            + np.sqrt(candidates) * sampler_discrepancy_proxy
                            + cg_absolute_tolerance**2
                        )
                        / (ridge * ridge)
                        + ridge * ridge
                    )
                    records.append(
                        {
                            "temperature_lambda": temperature,
                            "ula_step_size": step_size,
                            "ula_steps": ula_steps,
                            "initialization": initialization_mode,
                            "initialization_variance_scale": (
                                initialization_variance_scale
                            ),
                            "initialization_center_radius": float(
                                config["common"]["initialization_center_radius"]
                            ),
                            "repetitions": repetitions,
                            "exact_gibbs_reference_repetitions": (
                                reference_repetitions
                            ),
                            "number_candidates": candidates,
                            "ridge_gamma": ridge,
                            "radial_ks_to_exact_gibbs": ks_statistic,
                            "angular_ks_to_exact_gibbs": angular_ks_statistic,
                            "normal_ks_to_exact_gibbs": normal_ks_statistic,
                            "angular_mean_abs": abs(float(np.mean(angles))),
                            "angular_variance_error": abs(
                                float(np.var(angles)) - 1.0 / problem.y_dim
                            ),
                            "selected_squared_error_mean": float(
                                np.mean(selected_squared_errors)
                            ),
                            "exact_gibbs_selected_squared_error_mean": float(
                                np.mean(target_selected_squared_errors)
                            ),
                            "ula_selection_excess_over_exact_gibbs": float(
                                np.mean(selected_squared_errors)
                                - np.mean(target_selected_squared_errors)
                            ),
                            "selected_squared_error_se": float(
                                np.std(selected_squared_errors, ddof=1)
                                / np.sqrt(repetitions)
                            ),
                            "selected_value_gap_mean": float(
                                np.mean(selected_value_gaps)
                            ),
                            "selected_tube_distance_mean": float(
                                np.mean(selected_tube_distances)
                            ),
                            "hypergradient_error_mean": float(
                                np.mean(hypergradient_errors)
                            ),
                            "exact_gibbs_hypergradient_error_mean": (
                                target_hypergradient_error_mean
                            ),
                            "exact_gibbs_hypergradient_squared_error_mean": (
                                target_hypergradient_squared_error_mean
                            ),
                            "ula_hypergradient_excess_over_exact_gibbs": float(
                                np.mean(hypergradient_errors)
                                - target_hypergradient_error_mean
                            ),
                            "hypergradient_error_se": float(
                                np.std(hypergradient_errors, ddof=1)
                                / np.sqrt(repetitions)
                            ),
                            "hypergradient_squared_error_mean": float(
                                np.mean(np.square(hypergradient_errors))
                            ),
                            "cg_iterations_mean": float(np.mean(cg_iterations)),
                            "cg_breakdowns": cg_breakdowns,
                            "temperature_shape_proxy": temperature_proxy,
                            "selection_shape_proxy_N_minus_1_over_k": selection_proxy,
                            "sampler_discrepancy_proxy": sampler_discrepancy_proxy,
                            "theorem_hypergradient_bound_shape_proxy": theorem_shape_proxy,
                        }
                    )
    write_csv(output_directory / "ula_gibbs_error.csv", records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mechanism checks for selection, ULA/Gibbs, and ridge-CG."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default_config.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "mechanisms",
    )
    parser.add_argument("--seed", type=int, default=202603)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--selection-k-values",
        type=int,
        nargs="+",
        default=None,
        help="manifold dimensions for the best-of-N selection check",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="run only the best-of-N selection mechanism check",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    selection_k_values = (
        [int(value) for value in args.selection_k_values]
        if args.selection_k_values is not None
        else [int(value) for value in config["experiment"]["k_values"]]
    )
    if (
        not selection_k_values
        or any(value < 1 for value in selection_k_values)
        or len(selection_k_values) != len(set(selection_k_values))
    ):
        raise ValueError("selection manifold dimensions must be unique and positive")
    for k in selection_k_values:
        make_problem(config, k)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    print("[mechanisms] selection check", flush=True)
    selection_records = selection_check(
        config,
        args.output_directory,
        rng,
        args.quick,
        selection_k_values,
    )
    if args.selection_only:
        atomic_write_json(
            args.output_directory / "selection_metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "seed": args.seed,
                "quick": args.quick,
                "selection_only": True,
                "selection_k_values": selection_k_values,
                "problem_config": dict(config["problem"]),
                "selection_rows": len(selection_records),
                "important_interpretation": (
                    "For the smooth sphere objective, the sharp squared selection "
                    "error is N^(-2/k). The paper's N^(-1/k) term is a conservative "
                    "upper envelope, not an equality prediction."
                ),
            },
        )
        print(
            f"[mechanisms] wrote selection check to "
            f"{args.output_directory.resolve()}",
            flush=True,
        )
        return

    print("[mechanisms] ridge-CG check", flush=True)
    ridge_records = ridge_cg_check(config, args.output_directory, args.quick)
    print("[mechanisms] ULA/Gibbs check", flush=True)
    ula_records = ula_check(config, args.output_directory, rng, args.quick)
    atomic_write_json(
        args.output_directory / "metadata.json",
        {
            "schema_version": SCHEMA_VERSION,
            "seed": args.seed,
            "quick": args.quick,
            "problem_config": dict(config["problem"]),
            "common_config": dict(config["common"]),
            "selection_k_values": selection_k_values,
            "selection_rows": len(selection_records),
            "ridge_cg_rows": len(ridge_records),
            "ula_rows": len(ula_records),
            "important_interpretation": (
                "For the smooth sphere objective, the sharp squared selection error is "
                "N^(-2/k). The paper's N^(-1/k) term is a conservative upper envelope, "
                "not an equality prediction."
            ),
        },
    )
    print(f"[mechanisms] wrote {args.output_directory.resolve()}", flush=True)


if __name__ == "__main__":
    main()
