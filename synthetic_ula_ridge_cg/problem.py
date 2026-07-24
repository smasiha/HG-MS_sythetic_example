from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.integrate import cumulative_trapezoid


Array = np.ndarray


@dataclass(frozen=True)
class ManifoldBilevelProblem:
    """Smooth bilevel problem with a known spherical minimizer manifold."""

    k: int = 2
    ambient_dimension: int = 20
    kappa: float = 1.0
    normal_curvature: float = 1.0
    epsilon: float = 0.1
    theta_bound: float = 3.0
    landscape_amplitude: float = 0.5
    landscape_oscillations: int = 4
    landscape_coupling: float = 0.3

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be at least one")
        if self.ambient_dimension < self.k + 1:
            raise ValueError("ambient_dimension must be at least k + 1")
        if (
            self.kappa <= 0
            or self.normal_curvature <= 0
            or self.epsilon <= 0
            or self.theta_bound <= 0
        ):
            raise ValueError("problem constants must be positive")
        if self.landscape_amplitude <= 0 or self.landscape_oscillations < 1:
            raise ValueError("landscape parameters must be positive")
        if self.landscape_coupling < 0:
            raise ValueError("landscape_coupling must be nonnegative")
        if self.landscape_coupling * self.theta_bound >= 1.0:
            raise ValueError(
                "landscape tilt must remain positive on the theta domain"
            )

    @property
    def y_dim(self) -> int:
        return self.k + 1

    @property
    def dim(self) -> int:
        return self.ambient_dimension

    def radius(self, theta: float) -> float:
        return float(np.sqrt(1.0 + theta * theta))

    def direction(self, theta: float) -> Array:
        value = np.zeros(self.dim, dtype=np.float64)
        value[0] = np.cos(theta)
        value[1] = np.sin(theta)
        return value

    def direction_prime(self, theta: float) -> Array:
        value = np.zeros(self.dim, dtype=np.float64)
        value[0] = -np.sin(theta)
        value[1] = np.cos(theta)
        return value

    def orientation_normalizer(self, theta: float) -> float:
        radius = self.radius(theta)
        return float(radius / np.sqrt(1.0 + radius * radius))

    def orientation_normalizer_prime(self, theta: float) -> float:
        radius = self.radius(theta)
        return float(
            theta
            / (
                radius
                * np.power(1.0 + radius * radius, 1.5)
            )
        )

    @property
    def landscape_frequency(self) -> float:
        return 0.5 * self.landscape_oscillations * np.pi

    def restricted_landscape(self, theta: float, coordinate: Array) -> Array:
        coordinate = np.asarray(coordinate, dtype=np.float64)
        offset = coordinate + 1.0
        return (
            (1.0 + self.landscape_coupling * theta) * offset
            + self.landscape_amplitude
            * np.sin(self.landscape_frequency * offset) ** 2
        )

    def restricted_landscape_prime(self, theta: float, coordinate: Array) -> Array:
        coordinate = np.asarray(coordinate, dtype=np.float64)
        offset = coordinate + 1.0
        return (
            1.0
            + self.landscape_coupling * theta
            + self.landscape_amplitude
            * self.landscape_frequency
            * np.sin(2.0 * self.landscape_frequency * offset)
        )

    def orientation_coordinate(self, theta: float, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        norm_sq = np.sum(x * x, axis=-1)
        raw_orientation = np.sum(x * self.direction(theta), axis=-1) / np.sqrt(
            1.0 + norm_sq
        )
        return raw_orientation / self.orientation_normalizer(theta)

    @staticmethod
    def _psi(value: Array) -> Array:
        return np.sqrt(1.0 + value * value) - 1.0

    @staticmethod
    def _psi_prime(value: Array) -> Array:
        return value / np.sqrt(1.0 + value * value)

    @staticmethod
    def _psi_second(value: Array) -> Array:
        return np.power(1.0 + value * value, -1.5)

    def radial_residual(self, theta: float, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        active = x[..., : self.y_dim]
        return np.sum(active * active, axis=-1) - (1.0 + theta * theta)

    def g(self, theta: float, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        normal = x[..., self.y_dim :]
        return (
            self.kappa * self._psi(self.radial_residual(theta, x))
            + 0.5 * self.normal_curvature * np.sum(normal * normal, axis=-1)
        )

    def f(self, theta: float, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        norm_sq = np.sum(x * x, axis=-1)
        radial_term = 2.0 * np.sqrt(self.epsilon**2 + norm_sq)
        coordinate = self.orientation_coordinate(theta, x)
        return radial_term + self.restricted_landscape(theta, coordinate)

    def grad_g(self, theta: float, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        radial = self.radial_residual(theta, x)
        radial_factor = 2.0 * self.kappa * self._psi_prime(radial)
        gradient = self.normal_curvature * x.copy()
        gradient[..., : self.y_dim] = (
            radial_factor[..., None] * x[..., : self.y_dim]
        )
        return gradient

    def grad_f_x(self, theta: float, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        norm_sq = np.sum(x * x, axis=-1)
        radial_denom = np.sqrt(self.epsilon**2 + norm_sq)
        orientation_denom = np.sqrt(1.0 + norm_sq)
        orientation = np.sum(x * self.direction(theta), axis=-1)
        coordinate = self.orientation_coordinate(theta, x)
        raw_orientation_gradient = (
            self.direction(theta) / orientation_denom[..., None]
            - orientation[..., None]
            * x
            / np.power(1.0 + norm_sq, 1.5)[..., None]
        )
        coordinate_gradient = (
            raw_orientation_gradient / self.orientation_normalizer(theta)
        )
        return (
            2.0 * x / radial_denom[..., None]
            + self.restricted_landscape_prime(theta, coordinate)[..., None]
            * coordinate_gradient
        )

    def partial_f_theta(self, theta: float, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        norm_sq = np.sum(x * x, axis=-1)
        orientation_denom = np.sqrt(1.0 + norm_sq)
        raw_orientation = np.sum(x * self.direction(theta), axis=-1) / (
            orientation_denom
        )
        raw_orientation_theta = np.sum(
            x * self.direction_prime(theta), axis=-1
        ) / orientation_denom
        normalizer = self.orientation_normalizer(theta)
        coordinate = raw_orientation / normalizer
        coordinate_theta = (
            raw_orientation_theta / normalizer
            - raw_orientation
            * self.orientation_normalizer_prime(theta)
            / (normalizer * normalizer)
        )
        return (
            self.restricted_landscape_prime(theta, coordinate)
            * coordinate_theta
            + self.landscape_coupling * (coordinate + 1.0)
        )

    def partial_g_theta(self, theta: float, x: Array) -> Array:
        radial = self.radial_residual(theta, x)
        return -2.0 * self.kappa * theta * self._psi_prime(radial)

    def mixed_grad_g_theta(self, theta: float, x: Array) -> Array:
        """Return d/dtheta grad_x g(theta, x), with x held fixed."""
        x = np.asarray(x, dtype=np.float64)
        radial = self.radial_residual(theta, x)
        coefficient = -4.0 * self.kappa * theta * self._psi_second(radial)
        mixed = np.zeros_like(x)
        mixed[..., : self.y_dim] = (
            coefficient[..., None] * x[..., : self.y_dim]
        )
        return mixed

    def hessian_g(self, theta: float, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("hessian_g expects one lower-level point")
        radial = float(self.radial_residual(theta, x))
        identity_term = 2.0 * self.kappa * float(self._psi_prime(radial))
        rank_one = 4.0 * self.kappa * float(self._psi_second(radial))
        active = x[: self.y_dim]
        hessian = self.normal_curvature * np.eye(self.dim)
        hessian[: self.y_dim, : self.y_dim] = (
            identity_term * np.eye(self.y_dim)
            + rank_one * np.outer(active, active)
        )
        return hessian

    def hvp_g(self, theta: float, x: Array, vector: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        vector = np.asarray(vector, dtype=np.float64)
        radial = float(self.radial_residual(theta, x))
        identity_term = 2.0 * self.kappa * float(self._psi_prime(radial))
        rank_one = 4.0 * self.kappa * float(self._psi_second(radial))
        active = x[: self.y_dim]
        active_vector = vector[: self.y_dim]
        result = self.normal_curvature * vector.copy()
        result[: self.y_dim] = (
            identity_term * active_vector
            + rank_one * active * float(np.dot(active, active_vector))
        )
        return result

    def exact_lower_point(self, theta: float) -> Array:
        return -self.radius(theta) * self.direction(theta)

    def exact_hyperobjective(self, theta: float) -> float:
        radius = self.radius(theta)
        return float(
            2.0 * np.sqrt(self.epsilon**2 + radius * radius)
        )

    def exact_hypergradient(self, theta: float) -> float:
        radius = self.radius(theta)
        return float(
            2.0 * theta / np.sqrt(self.epsilon**2 + radius * radius)
        )

    @property
    def exact_theta_star(self) -> float:
        return 0.0

    @property
    def exact_F_star(self) -> float:
        return self.exact_hyperobjective(self.exact_theta_star)

    def manifold_distance(self, theta: float, x: Array) -> float:
        x = np.asarray(x, dtype=np.float64)
        radial_distance = np.linalg.norm(x[: self.y_dim]) - self.radius(theta)
        normal_squared = float(np.dot(x[self.y_dim :], x[self.y_dim :]))
        return float(np.sqrt(radial_distance * radial_distance + normal_squared))

    def optimistic_distance(self, theta: float, x: Array) -> float:
        return float(np.linalg.norm(np.asarray(x) - self.exact_lower_point(theta)))

    def project_to_manifold(self, theta: float, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        active = x[: self.y_dim]
        norm = float(np.linalg.norm(active))
        projected = np.zeros(self.dim, dtype=np.float64)
        if norm < 1e-15:
            return self.radius(theta) * self.direction(theta)
        projected[: self.y_dim] = self.radius(theta) * active / norm
        return projected

    def sample_scaled_gaussian_sphere(
        self,
        center_radius: float,
        size: int,
        rng: np.random.Generator,
        temperature: float,
        variance_scale: float,
    ) -> Array:
        if size < 1:
            raise ValueError("sample size must be positive")
        if center_radius <= 0 or temperature <= 0 or variance_scale <= 0:
            raise ValueError(
                "center_radius, temperature, and variance_scale must be positive"
            )
        random_values = rng.normal(size=(size, 2, self.dim))
        directions = random_values[:, 0, : self.y_dim]
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        centers = np.zeros((size, self.dim), dtype=np.float64)
        centers[:, : self.y_dim] = center_radius * directions
        return centers + np.sqrt(variance_scale * temperature) * random_values[
            :, 1, :
        ]

    def sample_scaled_gaussian_at_point(
        self,
        center: Array,
        size: int,
        rng: np.random.Generator,
        temperature: float,
        variance_scale: float,
    ) -> Array:
        if size < 1:
            raise ValueError("sample size must be positive")
        if temperature <= 0 or variance_scale <= 0:
            raise ValueError("temperature and variance_scale must be positive")
        center = np.asarray(center, dtype=np.float64)
        if center.shape != (self.dim,):
            raise ValueError(f"center must have shape {(self.dim,)}")
        return (
            center[None, :]
            + np.sqrt(variance_scale * temperature)
            * rng.normal(size=(size, self.dim))
        )

    def sample_cold(
        self,
        theta: float,
        size: int,
        rng: np.random.Generator,
    ) -> Array:
        del theta
        return rng.normal(size=(size, self.dim))

    def sample_exact_gibbs(
        self,
        theta: float,
        size: int,
        rng: np.random.Generator,
        temperature: float,
        radial_grid_size: int = 100_000,
    ) -> Array:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        directions = rng.normal(size=(size, self.y_dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radius = self.radius(theta)
        tail = max(2.0, 12.0 * np.sqrt(temperature) / max(radius, 0.25))
        radial_grid = np.linspace(1e-12, radius + tail, radial_grid_size)
        radial = radial_grid * radial_grid - radius * radius
        log_density = (
            self.k * np.log(radial_grid)
            - self.kappa * self._psi(radial) / temperature
        )
        density = np.exp(log_density - np.max(log_density))
        cdf = cumulative_trapezoid(density, radial_grid, initial=0.0)
        cdf /= cdf[-1]
        radii = np.interp(rng.random(size), cdf, radial_grid)
        samples = np.zeros((size, self.dim), dtype=np.float64)
        samples[:, : self.y_dim] = radii[:, None] * directions
        samples[:, self.y_dim :] = np.sqrt(
            temperature / self.normal_curvature
        ) * rng.normal(size=(size, self.dim - self.y_dim))
        return samples

    def metadata(self) -> Dict[str, float]:
        return {
            "k": int(self.k),
            "dimension": int(self.dim),
            "active_dimension": int(self.y_dim),
            "ambient_dimension": int(self.ambient_dimension),
            "kappa": float(self.kappa),
            "normal_curvature": float(self.normal_curvature),
            "epsilon": float(self.epsilon),
            "theta_bound": float(self.theta_bound),
            "landscape_amplitude": float(self.landscape_amplitude),
            "landscape_oscillations": int(self.landscape_oscillations),
            "landscape_coupling": float(self.landscape_coupling),
        }
