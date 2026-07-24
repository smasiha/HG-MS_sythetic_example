from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class GradientDescent:
    shape: Tuple[int, ...]

    def step(self, value: Array, gradient: Array, learning_rate: float) -> Array:
        return np.asarray(value) - learning_rate * np.asarray(
            gradient, dtype=np.float64
        )


def clip_by_norm(gradient: Array, max_norm: float) -> Array:
    gradient = np.asarray(gradient, dtype=np.float64)
    norm = float(np.linalg.norm(gradient))
    if max_norm <= 0 or norm <= max_norm:
        return gradient
    return gradient * (max_norm / max(norm, 1e-30))


def conjugate_gradient(
    matvec: Callable[[Array], Array],
    right_hand_side: Array,
    absolute_tolerance: float,
    max_iterations: int,
) -> Tuple[Array, Dict[str, Any]]:
    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be nonnegative")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    right_hand_side = np.asarray(right_hand_side, dtype=np.float64)
    solution = np.zeros_like(right_hand_side)
    residual = right_hand_side.copy()
    direction = residual.copy()
    residual_sq = float(np.dot(residual, residual))
    rhs_norm = float(np.sqrt(residual_sq))
    if rhs_norm <= absolute_tolerance:
        return solution, {
            "iterations": 0,
            "matvecs": 0,
            "absolute_residual": rhs_norm,
            "relative_residual": 0.0 if rhs_norm == 0.0 else 1.0,
            "breakdown": False,
        }

    breakdown = False
    iterations = 0
    matvecs = 0
    for _ in range(max_iterations):
        previous_residual_sq = residual_sq
        matrix_direction = matvec(direction)
        matvecs += 1
        curvature = float(np.dot(direction, matrix_direction))
        if not np.isfinite(curvature) or curvature <= 1e-30:
            breakdown = True
            break
        step_size = residual_sq / curvature
        solution += step_size * direction
        residual -= step_size * matrix_direction
        iterations += 1
        new_residual_sq = float(np.dot(residual, residual))
        residual_sq = new_residual_sq
        if np.sqrt(max(new_residual_sq, 0.0)) <= absolute_tolerance:
            break
        direction = residual + (
            new_residual_sq / max(previous_residual_sq, 1e-300)
        ) * direction

    absolute_residual = float(np.sqrt(max(residual_sq, 0.0)))

    return solution, {
        "iterations": int(iterations),
        "matvecs": int(matvecs),
        "absolute_residual": absolute_residual,
        "relative_residual": float(absolute_residual / rhs_norm),
        "breakdown": bool(breakdown),
    }


def penalty_scales(penalty: float) -> Tuple[float, float]:
    """Stable scaling equivalent to minimizing f + penalty * constraint."""
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    return min(1.0, 1.0 / penalty), min(1.0, penalty)
