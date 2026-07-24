from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np

from .numerics import (
    GradientDescent,
    clip_by_norm,
    conjugate_gradient,
    penalty_scales,
)
from .problem import ManifoldBilevelProblem
from .configuration import SCHEMA_VERSION


Array = np.ndarray


METHOD_LABELS = {
    "hg_ms": "HG-MS",
    "hg_baseline": "HG-baseline",
    "v_pbgd": "V-PBGD",
    "g_pbgd": "G-PBGD",
    "iaptt_gm": "IAPTT-GM",
}


@dataclass
class OracleCounter:
    grad_g: int = 0
    f: int = 0
    hessian_g: int = 0
    hvp_g: int = 0
    cg_iterations: int = 0
    cg_breakdowns: int = 0
    cg_tolerance_failures: int = 0
    lower_steps: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "grad_g": int(self.grad_g),
            "f": int(self.f),
            "hessian_g": int(self.hessian_g),
            "hvp_g": int(self.hvp_g),
            "cg_iterations": int(self.cg_iterations),
            "cg_breakdowns": int(self.cg_breakdowns),
            "cg_tolerance_failures": int(self.cg_tolerance_failures),
            "lower_steps": int(self.lower_steps),
        }


@dataclass
class RunState:
    theta: float
    lower_point: Optional[Array]
    iteration: int = 0
    last_hypergradient: float = float("nan")
    last_cg_iterations: int = 0
    last_cg_absolute_residual: float = float("nan")
    last_cg_relative_residual: float = float("nan")
    last_cg_breakdown: bool = False
    extra: MutableMapping[str, Any] = field(default_factory=dict)


def stable_method_seed(seed: int, method: str) -> int:
    offsets = {
        "hg_ms": 101,
        "hg_baseline": 101,
        "v_pbgd": 211,
        "g_pbgd": 307,
        "iaptt_gm": 401,
    }
    return int(seed * 1009 + offsets[method])


def stable_initialization_seed(seed: int) -> int:
    return int(seed * 1009 + 17)


def initial_theta(seed: int) -> float:
    rng = np.random.default_rng(seed + 71_003)
    magnitude = rng.uniform(1.0, 2.0)
    sign = -1.0 if seed % 2 else 1.0
    return float(sign * magnitude)


def project_box(value: Array, radius: float) -> Array:
    value = np.asarray(value)
    if radius <= 0:
        return value
    return np.clip(value, -radius, radius)


def ula_step(
    problem: ManifoldBilevelProblem,
    theta: float,
    points: Array,
    step_size: float,
    temperature: float,
    standard_normal_noise: Array,
) -> Array:
    if step_size <= 0:
        raise ValueError("ULA step_size must be positive")
    if temperature < 0:
        raise ValueError("ULA temperature must be nonnegative")
    points = np.asarray(points, dtype=np.float64)
    standard_normal_noise = np.asarray(standard_normal_noise, dtype=np.float64)
    if standard_normal_noise.shape != points.shape:
        raise ValueError("ULA noise and points must have the same shape")
    return (
        points
        - step_size * problem.grad_g(theta, points)
        + np.sqrt(2.0 * temperature * step_size) * standard_normal_noise
    )


def scheduled_learning_rate(
    method_config: Mapping[str, Any], progress: float
) -> float:
    initial = float(method_config["theta_learning_rate"])
    schedule = str(method_config.get("theta_learning_rate_schedule", "constant"))
    if schedule == "constant":
        return initial
    if schedule != "cosine":
        raise ValueError(f"unsupported learning-rate schedule: {schedule}")
    final = float(method_config["theta_learning_rate_end"])
    clipped_progress = float(np.clip(progress, 0.0, 1.0))
    cosine_weight = 0.5 * (1.0 + np.cos(np.pi * clipped_progress))
    return float(final + (initial - final) * cosine_weight)


def ridge_cg_hypergradient(
    problem: ManifoldBilevelProblem,
    theta: float,
    lower_point: Array,
    ridge: float,
    absolute_tolerance: float,
    max_iterations: int,
) -> Tuple[float, Array, Dict[str, Any]]:
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    right_hand_side = problem.grad_f_x(theta, lower_point)
    solution, cg_info = conjugate_gradient(
        lambda vector: problem.hvp_g(theta, lower_point, vector) + ridge * vector,
        right_hand_side,
        absolute_tolerance=absolute_tolerance,
        max_iterations=max_iterations,
    )
    hypergradient = float(
        problem.partial_f_theta(theta, lower_point)
        - np.dot(problem.mixed_grad_g_theta(theta, lower_point), solution)
    )
    return hypergradient, solution, cg_info


def v_pbgd_objective_and_gradients(
    problem: ManifoldBilevelProblem,
    theta: float,
    lower_point: Array,
    virtual_point: Array,
    penalty: float,
) -> Dict[str, Any]:
    upper_scale, constraint_scale = penalty_scales(penalty)
    objective = upper_scale * float(problem.f(theta, lower_point))
    objective += constraint_scale * float(
        problem.g(theta, lower_point) - problem.g(theta, virtual_point)
    )
    gradient_theta = upper_scale * float(
        problem.partial_f_theta(theta, lower_point)
    ) + constraint_scale * float(
        problem.partial_g_theta(theta, lower_point)
        - problem.partial_g_theta(theta, virtual_point)
    )
    gradient_lower = (
        upper_scale * problem.grad_f_x(theta, lower_point)
        + constraint_scale * problem.grad_g(theta, lower_point)
    )
    return {
        "objective": float(objective),
        "gradient_theta": float(gradient_theta),
        "gradient_lower": gradient_lower,
        "upper_scale": float(upper_scale),
        "constraint_scale": float(constraint_scale),
    }


def g_pbgd_objective_and_gradients(
    problem: ManifoldBilevelProblem,
    theta: float,
    lower_point: Array,
    penalty: float,
) -> Dict[str, Any]:
    upper_scale, constraint_scale = penalty_scales(penalty)
    lower_gradient = problem.grad_g(theta, lower_point)
    mixed = problem.mixed_grad_g_theta(theta, lower_point)
    objective = upper_scale * float(problem.f(theta, lower_point))
    objective += 0.5 * constraint_scale * float(
        np.dot(lower_gradient, lower_gradient)
    )
    gradient_theta = upper_scale * float(
        problem.partial_f_theta(theta, lower_point)
    ) + constraint_scale * float(np.dot(mixed, lower_gradient))
    gradient_lower = (
        upper_scale * problem.grad_f_x(theta, lower_point)
        + constraint_scale
        * problem.hvp_g(theta, lower_point, lower_gradient)
    )
    return {
        "objective": float(objective),
        "gradient_theta": float(gradient_theta),
        "gradient_lower": gradient_lower,
        "lower_gradient": lower_gradient,
        "upper_scale": float(upper_scale),
        "constraint_scale": float(constraint_scale),
    }


def iaptt_unroll_and_gradients(
    problem: ManifoldBilevelProblem,
    theta: float,
    initialization: Array,
    inner_learning_rate: float,
    inner_steps: int,
    selection: str = "max",
) -> Dict[str, Any]:
    if inner_learning_rate <= 0:
        raise ValueError("inner_learning_rate must be positive")
    if inner_steps < 1:
        raise ValueError("inner_steps must be positive")
    if selection not in {"max", "min"}:
        raise ValueError("selection must be max or min")

    lower_point = np.asarray(initialization, dtype=np.float64).copy()
    theta_sensitivity = np.zeros(problem.dim, dtype=np.float64)
    initialization_jacobian = np.eye(problem.dim, dtype=np.float64)
    trajectory_points: List[Array] = []
    trajectory_theta_sensitivities: List[Array] = []
    trajectory_jacobians: List[Array] = []
    trajectory_values: List[float] = []

    for _ in range(inner_steps):
        lower_gradient = problem.grad_g(theta, lower_point)
        lower_hessian = problem.hessian_g(theta, lower_point)
        mixed = problem.mixed_grad_g_theta(theta, lower_point)
        transition = np.eye(problem.dim) - inner_learning_rate * lower_hessian
        theta_sensitivity = (
            transition.dot(theta_sensitivity) - inner_learning_rate * mixed
        )
        initialization_jacobian = transition.dot(initialization_jacobian)
        lower_point = lower_point - inner_learning_rate * lower_gradient
        trajectory_points.append(lower_point.copy())
        trajectory_theta_sensitivities.append(theta_sensitivity.copy())
        trajectory_jacobians.append(initialization_jacobian.copy())
        trajectory_values.append(float(problem.f(theta, lower_point)))

    selected_index = (
        int(np.argmax(trajectory_values))
        if selection == "max"
        else int(np.argmin(trajectory_values))
    )
    selected_point = trajectory_points[selected_index]
    upper_gradient_x = problem.grad_f_x(theta, selected_point)
    gradient_theta = float(
        problem.partial_f_theta(theta, selected_point)
        + np.dot(upper_gradient_x, trajectory_theta_sensitivities[selected_index])
    )
    gradient_initialization = trajectory_jacobians[selected_index].T.dot(
        upper_gradient_x
    )
    return {
        "selected_index": selected_index,
        "selected_point": selected_point,
        "selected_value": trajectory_values[selected_index],
        "gradient_theta": gradient_theta,
        "gradient_initialization": gradient_initialization,
        "trajectory_values": np.asarray(trajectory_values, dtype=np.float64),
        "trajectory_points": np.asarray(trajectory_points, dtype=np.float64),
    }


def _sample_initial_points(
    problem: ManifoldBilevelProblem,
    theta: float,
    size: int,
    rng: np.random.Generator,
    common: Mapping[str, Any],
) -> Array:
    mode = str(common.get("initialization", "scaled_gaussian_sphere"))
    if mode == "scaled_gaussian_sphere":
        return problem.sample_scaled_gaussian_sphere(
            center_radius=float(common["initialization_center_radius"]),
            size=size,
            rng=rng,
            temperature=float(common["initialization_temperature"]),
            variance_scale=float(common["initialization_variance_scale"]),
        )
    if mode == "cold":
        return problem.sample_cold(theta, size, rng)
    raise ValueError(f"unknown initialization mode: {mode}")


def _sample_initial_points_from_independent_streams(
    problem: ManifoldBilevelProblem,
    theta: float,
    rngs: List[np.random.Generator],
    common: Mapping[str, Any],
) -> Array:
    return np.stack(
        [
            _sample_initial_points(problem, theta, 1, rng, common)[0]
            for rng in rngs
        ]
    )


def _metrics(
    problem: ManifoldBilevelProblem,
    state: RunState,
    elapsed_seconds: float,
    counter: OracleCounter,
) -> Dict[str, Any]:
    theta = float(state.theta)
    estimated_hypergradient = (
        float(state.last_hypergradient)
        if np.isfinite(state.last_hypergradient)
        else None
    )
    cg_absolute_residual = (
        float(state.last_cg_absolute_residual)
        if np.isfinite(state.last_cg_absolute_residual)
        else None
    )
    cg_relative_residual = (
        float(state.last_cg_relative_residual)
        if np.isfinite(state.last_cg_relative_residual)
        else None
    )
    record: Dict[str, Any] = {
        "elapsed_seconds": float(elapsed_seconds),
        "iteration": int(state.iteration),
        "theta": theta,
        "hyperobjective": problem.exact_hyperobjective(theta),
        "hyperobjective_gap": max(
            0.0, problem.exact_hyperobjective(theta) - problem.exact_F_star
        ),
        "theta_error": abs(theta - problem.exact_theta_star),
        "exact_hypergradient_abs": abs(problem.exact_hypergradient(theta)),
        "estimated_hypergradient": estimated_hypergradient,
        "hypergradient_error": (
            abs(state.last_hypergradient - problem.exact_hypergradient(theta))
            if np.isfinite(state.last_hypergradient)
            else None
        ),
        "cg_iterations": int(state.last_cg_iterations),
        "cg_absolute_residual": cg_absolute_residual,
        "cg_relative_residual": cg_relative_residual,
        "cg_breakdown": bool(state.last_cg_breakdown),
        **counter.as_dict(),
    }
    if state.lower_point is None:
        record["manifold_distance"] = None
        record["optimistic_distance"] = None
    else:
        record["manifold_distance"] = problem.manifold_distance(theta, state.lower_point)
        record["optimistic_distance"] = problem.optimistic_distance(theta, state.lower_point)
    record.update(state.extra)
    return record


def _run_timed_loop(
    problem: ManifoldBilevelProblem,
    state: RunState,
    counter: OracleCounter,
    budget_seconds: float,
    log_interval_seconds: float,
    update: Callable[[float], None],
    start_time: float,
) -> List[Dict[str, Any]]:
    trajectory = [_metrics(problem, state, time.perf_counter() - start_time, counter)]
    next_log = log_interval_seconds
    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed >= budget_seconds:
            break
        update(elapsed / budget_seconds)
        state.iteration += 1
        elapsed = time.perf_counter() - start_time
        if elapsed >= next_log:
            trajectory.append(_metrics(problem, state, elapsed, counter))
            while next_log <= elapsed:
                next_log += log_interval_seconds
    final_elapsed = time.perf_counter() - start_time
    final_record = _metrics(problem, state, final_elapsed, counter)
    if trajectory[-1]["iteration"] != final_record["iteration"]:
        trajectory.append(final_record)
    else:
        trajectory[-1] = final_record
    return trajectory


def _run_hg(
    method: str,
    problem: ManifoldBilevelProblem,
    seed: int,
    budget_seconds: float,
    config: Mapping[str, Any],
    common: Mapping[str, Any],
) -> Dict[str, Any]:
    method_config = config[method]
    branches = int(method_config["branches"])
    if method == "hg_baseline" and branches != 1:
        raise ValueError("HG-baseline must use exactly one candidate")
    theta = initial_theta(seed)
    start_time = time.perf_counter()
    initialization_rngs = [
        np.random.default_rng(
            stable_initialization_seed(seed) + 104_729 * index
        )
        for index in range(branches)
    ]
    chains = _sample_initial_points_from_independent_streams(
        problem, theta, initialization_rngs, common
    )
    chain_rngs = [
        np.random.default_rng(stable_method_seed(seed, method) + 104_729 * index)
        for index in range(branches)
    ]
    selected = chains[int(np.argmin(problem.f(theta, chains)))].copy()
    counter = OracleCounter(f=branches)
    state = RunState(theta=theta, lower_point=selected)
    theta_optimizer = GradientDescent(())
    ula_steps = int(method_config["ula_steps"])
    ula_step_size = float(method_config["ula_step_size"])
    temperature = float(method_config["temperature"])
    if (
        str(common.get("initialization")) == "scaled_gaussian_sphere"
        and not np.isclose(
            temperature,
            float(common["initialization_temperature"]),
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise ValueError(
            "HG ULA temperature must equal the scaled-Gaussian "
            "initialization temperature"
        )
    ridge = float(method_config["ridge"])
    cg_absolute_tolerance = float(method_config["cg_absolute_tolerance"])
    cg_max_iterations = int(method_config["cg_max_iterations"])
    theta_clip = float(common["theta_clip"])
    gradient_clip = float(common["gradient_clip"])

    def update(progress: float) -> None:
        nonlocal chains
        theta_before_update = state.theta
        if state.iteration > 0:
            chains = _sample_initial_points_from_independent_streams(
                problem,
                state.theta,
                initialization_rngs,
                common,
            )
        for _ in range(ula_steps):
            chains = ula_step(
                problem,
                state.theta,
                chains,
                ula_step_size,
                temperature,
                np.stack(
                    [chain_rng.normal(size=problem.dim) for chain_rng in chain_rngs]
                ),
            )
        counter.grad_g += branches * ula_steps
        counter.lower_steps += branches * ula_steps

        upper_values = problem.f(state.theta, chains)
        counter.f += branches
        selected_index = int(np.argmin(upper_values))
        selected_point = chains[selected_index].copy()
        hypergradient, _, cg_info = ridge_cg_hypergradient(
            problem,
            state.theta,
            selected_point,
            ridge,
            cg_absolute_tolerance,
            max_iterations=cg_max_iterations,
        )
        counter.hvp_g += int(cg_info["matvecs"])
        counter.cg_iterations += int(cg_info["iterations"])
        counter.cg_breakdowns += int(bool(cg_info["breakdown"]))
        counter.cg_tolerance_failures += int(
            float(cg_info["absolute_residual"])
            > cg_absolute_tolerance * (1.0 + 1e-8)
        )
        clipped = float(clip_by_norm(np.asarray(hypergradient), gradient_clip))
        current_theta_learning_rate = scheduled_learning_rate(
            method_config, progress
        )
        state.theta = float(
            project_box(
                theta_optimizer.step(
                    np.asarray(state.theta),
                    np.asarray(clipped),
                    current_theta_learning_rate,
                ),
                theta_clip,
            )
        )
        state.lower_point = selected_point
        state.last_hypergradient = hypergradient
        state.last_cg_iterations = int(cg_info["iterations"])
        state.last_cg_absolute_residual = float(cg_info["absolute_residual"])
        state.last_cg_relative_residual = float(cg_info["relative_residual"])
        state.last_cg_breakdown = bool(cg_info["breakdown"])
        state.extra = {
            "selected_candidate": selected_index,
            "candidate_count": branches,
            "selected_upper_value": float(upper_values[selected_index]),
            "theta_learning_rate": current_theta_learning_rate,
            "hypergradient_error": abs(
                hypergradient - problem.exact_hypergradient(theta_before_update)
            ),
        }

    trajectory = _run_timed_loop(
        problem,
        state,
        counter,
        budget_seconds,
        float(common["log_interval_seconds"]),
        update,
        start_time,
    )
    return _package_run(method, seed, budget_seconds, problem, config, trajectory, start_time)


def _run_v_pbgd(
    problem: ManifoldBilevelProblem,
    seed: int,
    budget_seconds: float,
    config: Mapping[str, Any],
    common: Mapping[str, Any],
) -> Dict[str, Any]:
    method = "v_pbgd"
    method_config = config[method]
    theta = initial_theta(seed)
    start_time = time.perf_counter()
    initialization_rng = np.random.default_rng(stable_initialization_seed(seed))
    lower_point = _sample_initial_points(
        problem, theta, 1, initialization_rng, common
    )[0]
    virtual_point = lower_point.copy()
    state = RunState(theta=theta, lower_point=lower_point)
    counter = OracleCounter()
    theta_optimizer = GradientDescent(())
    lower_optimizer = GradientDescent((problem.dim,))
    theta_lr = float(method_config["theta_learning_rate"])
    lower_lr = float(method_config["lower_learning_rate"])
    virtual_lr = float(method_config["virtual_learning_rate"])
    virtual_steps = int(method_config["virtual_steps"])
    penalty = float(method_config["penalty"])
    theta_clip = float(common["theta_clip"])
    lower_clip = float(common["lower_clip"])
    gradient_clip = float(common["gradient_clip"])

    def update(_: float) -> None:
        nonlocal lower_point, virtual_point
        theta_before_update = state.theta
        virtual_point = lower_point.copy()
        for _ in range(virtual_steps):
            virtual_point -= virtual_lr * problem.grad_g(state.theta, virtual_point)
        counter.grad_g += virtual_steps
        counter.lower_steps += virtual_steps

        surrogate = v_pbgd_objective_and_gradients(
            problem,
            state.theta,
            lower_point,
            virtual_point,
            penalty,
        )
        gradient_theta = float(surrogate["gradient_theta"])
        gradient_lower = surrogate["gradient_lower"]
        virtual_lower_loss = float(problem.g(state.theta, virtual_point))
        counter.grad_g += 1
        counter.f += 1
        gradient_theta = float(clip_by_norm(np.asarray(gradient_theta), gradient_clip))
        gradient_lower = clip_by_norm(gradient_lower, gradient_clip)
        state.theta = float(
            project_box(
                theta_optimizer.step(
                    np.asarray(state.theta), np.asarray(gradient_theta), theta_lr
                ),
                theta_clip,
            )
        )
        lower_point = project_box(
            lower_optimizer.step(lower_point, gradient_lower, lower_lr),
            lower_clip,
        )
        virtual_point = project_box(virtual_point, lower_clip)
        state.lower_point = lower_point
        state.last_hypergradient = gradient_theta
        state.extra = {
            "penalty": penalty,
            "surrogate_objective": float(surrogate["objective"]),
            "virtual_initialization": "lower_point",
            "virtual_lower_loss": virtual_lower_loss,
            "hypergradient_error": abs(
                gradient_theta - problem.exact_hypergradient(theta_before_update)
            ),
        }

    trajectory = _run_timed_loop(
        problem,
        state,
        counter,
        budget_seconds,
        float(common["log_interval_seconds"]),
        update,
        start_time,
    )
    return _package_run(method, seed, budget_seconds, problem, config, trajectory, start_time)


def _run_g_pbgd(
    problem: ManifoldBilevelProblem,
    seed: int,
    budget_seconds: float,
    config: Mapping[str, Any],
    common: Mapping[str, Any],
) -> Dict[str, Any]:
    method = "g_pbgd"
    method_config = config[method]
    theta = initial_theta(seed)
    start_time = time.perf_counter()
    initialization_rng = np.random.default_rng(stable_initialization_seed(seed))
    lower_point = _sample_initial_points(
        problem, theta, 1, initialization_rng, common
    )[0]
    state = RunState(theta=theta, lower_point=lower_point)
    counter = OracleCounter()
    theta_optimizer = GradientDescent(())
    lower_optimizer = GradientDescent((problem.dim,))
    theta_lr = float(method_config["theta_learning_rate"])
    lower_lr = float(method_config["lower_learning_rate"])
    penalty = float(method_config["penalty"])
    theta_clip = float(common["theta_clip"])
    lower_clip = float(common["lower_clip"])
    gradient_clip = float(common["gradient_clip"])

    def update(_: float) -> None:
        nonlocal lower_point
        theta_before_update = state.theta
        surrogate = g_pbgd_objective_and_gradients(
            problem,
            state.theta,
            lower_point,
            penalty,
        )
        lower_gradient = surrogate["lower_gradient"]
        gradient_theta = float(surrogate["gradient_theta"])
        gradient_lower = surrogate["gradient_lower"]
        counter.grad_g += 1
        counter.f += 1
        counter.hvp_g += 1
        gradient_theta = float(clip_by_norm(np.asarray(gradient_theta), gradient_clip))
        gradient_lower = clip_by_norm(gradient_lower, gradient_clip)
        state.theta = float(
            project_box(
                theta_optimizer.step(
                    np.asarray(state.theta), np.asarray(gradient_theta), theta_lr
                ),
                theta_clip,
            )
        )
        lower_point = project_box(
            lower_optimizer.step(lower_point, gradient_lower, lower_lr),
            lower_clip,
        )
        state.lower_point = lower_point
        state.last_hypergradient = gradient_theta
        state.extra = {
            "penalty": penalty,
            "surrogate_objective": float(surrogate["objective"]),
            "stationarity_norm": float(np.linalg.norm(lower_gradient)),
            "hypergradient_error": abs(
                gradient_theta - problem.exact_hypergradient(theta_before_update)
            ),
        }

    trajectory = _run_timed_loop(
        problem,
        state,
        counter,
        budget_seconds,
        float(common["log_interval_seconds"]),
        update,
        start_time,
    )
    return _package_run(method, seed, budget_seconds, problem, config, trajectory, start_time)


def _run_iaptt_gm(
    problem: ManifoldBilevelProblem,
    seed: int,
    budget_seconds: float,
    config: Mapping[str, Any],
    common: Mapping[str, Any],
) -> Dict[str, Any]:
    method = "iaptt_gm"
    method_config = config[method]
    theta = initial_theta(seed)
    start_time = time.perf_counter()
    initialization_rng = np.random.default_rng(stable_initialization_seed(seed))
    initialization = _sample_initial_points(
        problem, theta, 1, initialization_rng, common
    )[0]
    state = RunState(theta=theta, lower_point=initialization.copy())
    counter = OracleCounter()
    theta_optimizer = GradientDescent(())
    initialization_optimizer = GradientDescent((problem.dim,))
    theta_lr = float(method_config["theta_learning_rate"])
    initialization_lr = float(method_config["initialization_learning_rate"])
    inner_lr = float(method_config["inner_learning_rate"])
    inner_steps = int(method_config["inner_steps"])
    selection = "max"
    theta_clip = float(common["theta_clip"])
    lower_clip = float(common["lower_clip"])
    gradient_clip = float(common["gradient_clip"])

    def update(_: float) -> None:
        nonlocal initialization
        theta_before_update = state.theta
        unroll = iaptt_unroll_and_gradients(
            problem,
            state.theta,
            initialization,
            inner_lr,
            inner_steps,
            selection,
        )
        counter.grad_g += inner_steps
        counter.hessian_g += inner_steps
        counter.lower_steps += inner_steps
        counter.f += inner_steps
        selected_index = int(unroll["selected_index"])
        selected_point = unroll["selected_point"]
        gradient_theta = float(unroll["gradient_theta"])
        gradient_initialization = unroll["gradient_initialization"]
        gradient_theta = float(clip_by_norm(np.asarray(gradient_theta), gradient_clip))
        gradient_initialization = clip_by_norm(gradient_initialization, gradient_clip)
        state.theta = float(
            project_box(
                theta_optimizer.step(
                    np.asarray(state.theta), np.asarray(gradient_theta), theta_lr
                ),
                theta_clip,
            )
        )
        initialization = project_box(
            initialization_optimizer.step(
                initialization, gradient_initialization, initialization_lr
            ),
            lower_clip,
        )
        state.lower_point = selected_point
        state.last_hypergradient = gradient_theta
        state.extra = {
            "trajectory_index": selected_index + 1,
            "trajectory_length": inner_steps,
            "trajectory_selection": selection,
            "trajectory_selected_value": float(unroll["selected_value"]),
            "hypergradient_error": abs(
                gradient_theta - problem.exact_hypergradient(theta_before_update)
            ),
        }

    trajectory = _run_timed_loop(
        problem,
        state,
        counter,
        budget_seconds,
        float(common["log_interval_seconds"]),
        update,
        start_time,
    )
    return _package_run(method, seed, budget_seconds, problem, config, trajectory, start_time)


def _package_run(
    method: str,
    seed: int,
    budget_seconds: float,
    problem: ManifoldBilevelProblem,
    config: Mapping[str, Any],
    trajectory: List[Dict[str, Any]],
    start_time: float,
) -> Dict[str, Any]:
    elapsed = time.perf_counter() - start_time
    return {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "seed": int(seed),
        "theta_initial": initial_theta(seed),
        "budget_seconds": float(budget_seconds),
        "elapsed_seconds": float(elapsed),
        "budget_overrun_seconds": max(0.0, float(elapsed - budget_seconds)),
        "problem": problem.metadata(),
        "common_config": dict(config["common"]),
        "method_config": dict(config[method]),
        "trajectory": trajectory,
        "final": trajectory[-1],
    }


def run_method(
    method: str,
    problem: ManifoldBilevelProblem,
    seed: int,
    budget_seconds: float,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    if method not in METHOD_LABELS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHOD_LABELS)}")
    common = config["common"]
    if method in {"hg_ms", "hg_baseline"}:
        return _run_hg(method, problem, seed, budget_seconds, config, common)
    if method == "v_pbgd":
        return _run_v_pbgd(problem, seed, budget_seconds, config, common)
    if method == "g_pbgd":
        return _run_g_pbgd(problem, seed, budget_seconds, config, common)
    return _run_iaptt_gm(problem, seed, budget_seconds, config, common)
