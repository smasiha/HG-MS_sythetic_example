from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from synthetic_ula_ridge_cg.methods import (
    METHOD_LABELS,
    g_pbgd_objective_and_gradients,
    iaptt_unroll_and_gradients,
    project_box,
    ridge_cg_hypergradient,
    run_method,
    scheduled_learning_rate,
    ula_step,
    v_pbgd_objective_and_gradients,
)
from synthetic_ula_ridge_cg.configuration import (
    load_config,
    make_problem,
)
from synthetic_ula_ridge_cg.aggregate_results import (
    bootstrap_mean_statistics,
    validate_runs,
)
from synthetic_ula_ridge_cg.numerics import conjugate_gradient, penalty_scales
from synthetic_ula_ridge_cg.problem import ManifoldBilevelProblem
from synthetic_ula_ridge_cg.run_benchmark import compatible_cached_run
from synthetic_ula_ridge_cg.run_mechanism_checks import (
    selection_quadrature_reference,
)


class SyntheticProblemTest(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = ManifoldBilevelProblem(k=2)
        self.theta = 0.8
        self.rng = np.random.default_rng(123)
        self.x = self.rng.normal(size=self.problem.dim)

    def test_ambient_dimension_is_fixed_while_manifold_dimension_varies(
        self,
    ) -> None:
        for k in (1, 2, 4, 10):
            with self.subTest(k=k):
                problem = ManifoldBilevelProblem(k=k, ambient_dimension=20)
                self.assertEqual(problem.dim, 20)
                self.assertEqual(problem.y_dim, k + 1)
                point = problem.exact_lower_point(self.theta)
                eigenvalues = np.linalg.eigvalsh(
                    problem.hessian_g(self.theta, point)
                )
                self.assertEqual(
                    int(np.sum(np.abs(eigenvalues) < 1e-10)),
                    k,
                )

    def central_difference(self, function, value: float, step: float = 1e-6) -> float:
        return float((function(value + step) - function(value - step)) / (2.0 * step))

    def vector_central_difference(
        self, function, value: np.ndarray, step: float = 1e-6
    ) -> np.ndarray:
        gradient = np.empty_like(value, dtype=np.float64)
        for index in range(value.size):
            direction = np.zeros_like(value, dtype=np.float64)
            direction[index] = step
            gradient[index] = (
                function(value + direction) - function(value - direction)
            ) / (2.0 * step)
        return gradient

    def test_bootstrap_half_width_matches_absolute_mean_deviations(self) -> None:
        values = np.asarray([0.2, 0.4, 0.8, 1.6])
        repetitions = 2_000
        observed = bootstrap_mean_statistics(
            values,
            np.random.default_rng(907),
            repetitions,
        )
        reference_rng = np.random.default_rng(907)
        indices = reference_rng.integers(
            0,
            values.size,
            size=(repetitions, values.size),
        )
        means = np.mean(values[indices], axis=1)
        expected_interval = np.quantile(means, [0.025, 0.975])
        expected_half_width = np.quantile(
            np.abs(means - np.mean(values)),
            0.95,
        )
        np.testing.assert_allclose(observed[:2], expected_interval)
        self.assertAlmostEqual(observed[2], expected_half_width)

    def test_known_manifold_and_optimistic_solution(self) -> None:
        point = self.problem.exact_lower_point(self.theta)
        self.assertAlmostEqual(float(self.problem.g(self.theta, point)), 0.0, places=13)
        np.testing.assert_allclose(
            self.problem.grad_g(self.theta, point),
            np.zeros(self.problem.dim),
            atol=1e-12,
        )
        self.assertAlmostEqual(
            float(self.problem.f(self.theta, point)),
            self.problem.exact_hyperobjective(self.theta),
            places=12,
        )

    def test_off_manifold_center_is_a_strict_saddle(self) -> None:
        center = np.zeros(self.problem.dim)
        np.testing.assert_array_equal(
            self.problem.grad_g(self.theta, center), center
        )
        self.assertGreater(float(self.problem.g(self.theta, center)), 0.0)
        eigenvalues = np.linalg.eigvalsh(
            self.problem.hessian_g(self.theta, center)
        )
        self.assertEqual(
            int(np.sum(eigenvalues < 0.0)),
            self.problem.y_dim,
        )
        self.assertEqual(
            int(np.sum(eigenvalues > 0.0)),
            self.problem.dim - self.problem.y_dim,
        )

    def test_upper_theta_partial_is_globally_bounded(self) -> None:
        points = self.rng.normal(size=(100, self.problem.dim)) * 1e6
        for theta in (-3.0, -1.0, 0.0, 1.0, 3.0):
            partials = np.abs(self.problem.partial_f_theta(theta, points))
            self.assertTrue(np.all(partials <= 7.0))

    def test_restricted_upper_landscape_is_multimodal_with_unique_optimum(
        self,
    ) -> None:
        coordinates = np.linspace(-1.0, 1.0, 20_001)
        for theta in (-3.0, 0.0, 3.0):
            values = self.problem.restricted_landscape(theta, coordinates)
            self.assertAlmostEqual(float(values[0]), 0.0, places=14)
            self.assertTrue(np.all(values[1:] > 0.0))
            local_minima = np.where(
                (values[1:-1] < values[:-2])
                & (values[1:-1] < values[2:])
            )[0]
            self.assertGreaterEqual(local_minima.size, 4)

    def test_optimistic_point_has_positive_intrinsic_curvature(self) -> None:
        radius = self.problem.radius(self.theta)
        step = 1e-4

        def manifold_curve(angle: float) -> np.ndarray:
            return radius * (
                -np.cos(angle) * self.problem.direction(self.theta)
                + np.sin(angle) * self.problem.direction_prime(self.theta)
            )

        intrinsic_second_derivative = (
            self.problem.f(self.theta, manifold_curve(step))
            - 2.0 * self.problem.f(self.theta, manifold_curve(0.0))
            + self.problem.f(self.theta, manifold_curve(-step))
        ) / (step * step)
        self.assertAlmostEqual(
            float(intrinsic_second_derivative),
            1.0 + self.problem.landscape_coupling * self.theta,
            places=6,
        )

    def test_selection_quadrature_has_correct_single_sample_moment(self) -> None:
        references = selection_quadrature_reference(
            self.problem,
            self.theta,
            [1, 16, 256],
            nodes=2_048,
        )
        self.assertAlmostEqual(
            references[1]["squared_error_mean"],
            2.0 * self.problem.radius(self.theta) ** 2,
            places=10,
        )
        self.assertGreater(
            references[1]["squared_error_mean"],
            references[16]["squared_error_mean"],
        )
        self.assertGreater(
            references[16]["squared_error_mean"],
            references[256]["squared_error_mean"],
        )

    def test_exact_hypergradient_matches_finite_difference(self) -> None:
        finite_difference = self.central_difference(
            self.problem.exact_hyperobjective, self.theta
        )
        self.assertAlmostEqual(
            finite_difference,
            self.problem.exact_hypergradient(self.theta),
            places=7,
        )

    def test_analytic_first_derivatives(self) -> None:
        step = 1e-6
        numeric_grad_g = np.empty(self.problem.dim)
        numeric_grad_f = np.empty(self.problem.dim)
        for index in range(self.problem.dim):
            direction = np.zeros(self.problem.dim)
            direction[index] = step
            numeric_grad_g[index] = (
                self.problem.g(self.theta, self.x + direction)
                - self.problem.g(self.theta, self.x - direction)
            ) / (2.0 * step)
            numeric_grad_f[index] = (
                self.problem.f(self.theta, self.x + direction)
                - self.problem.f(self.theta, self.x - direction)
            ) / (2.0 * step)
        np.testing.assert_allclose(
            self.problem.grad_g(self.theta, self.x), numeric_grad_g, rtol=1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            self.problem.grad_f_x(self.theta, self.x),
            numeric_grad_f,
            rtol=1e-6,
            atol=1e-7,
        )
        numeric_partial_g = self.central_difference(
            lambda theta: self.problem.g(theta, self.x), self.theta
        )
        numeric_partial_f = self.central_difference(
            lambda theta: self.problem.f(theta, self.x), self.theta
        )
        self.assertAlmostEqual(
            numeric_partial_g,
            float(self.problem.partial_g_theta(self.theta, self.x)),
            places=6,
        )
        self.assertAlmostEqual(
            numeric_partial_f,
            float(self.problem.partial_f_theta(self.theta, self.x)),
            places=6,
        )

    def test_hessian_hvp_and_mixed_derivative(self) -> None:
        step = 1e-6
        vector = self.rng.normal(size=self.problem.dim)
        finite_hvp = (
            self.problem.grad_g(self.theta, self.x + step * vector)
            - self.problem.grad_g(self.theta, self.x - step * vector)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            self.problem.hvp_g(self.theta, self.x, vector),
            finite_hvp,
            rtol=2e-6,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            self.problem.hessian_g(self.theta, self.x).dot(vector),
            finite_hvp,
            rtol=2e-6,
            atol=2e-7,
        )
        finite_mixed = (
            self.problem.grad_g(self.theta + step, self.x)
            - self.problem.grad_g(self.theta - step, self.x)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            self.problem.mixed_grad_g_theta(self.theta, self.x),
            finite_mixed,
            rtol=2e-6,
            atol=2e-7,
        )

    def test_pseudoinverse_hypergradient_is_exact(self) -> None:
        point = self.problem.exact_lower_point(self.theta)
        hessian = self.problem.hessian_g(self.theta, point)
        solution = np.linalg.pinv(hessian, rcond=1e-12).dot(
            self.problem.grad_f_x(self.theta, point)
        )
        estimate = float(
            self.problem.partial_f_theta(self.theta, point)
            - np.dot(
                self.problem.mixed_grad_g_theta(self.theta, point),
                solution,
            )
        )
        self.assertAlmostEqual(
            estimate, self.problem.exact_hypergradient(self.theta), places=10
        )
        eigenvalues = np.linalg.eigvalsh(hessian)
        self.assertEqual(int(np.sum(np.abs(eigenvalues) < 1e-10)), self.problem.k)

    def test_conjugate_gradient_matches_dense_ridge(self) -> None:
        point = self.problem.exact_lower_point(self.theta)
        hessian = self.problem.hessian_g(self.theta, point)
        ridge = 0.05
        right_hand_side = self.problem.grad_f_x(self.theta, point)
        expected = np.linalg.solve(
            hessian + ridge * np.eye(self.problem.dim), right_hand_side
        )
        observed, info = conjugate_gradient(
            lambda vector: self.problem.hvp_g(self.theta, point, vector)
            + ridge * vector,
            right_hand_side,
            absolute_tolerance=1e-12,
            max_iterations=4 * self.problem.dim,
        )
        self.assertFalse(info["breakdown"])
        self.assertEqual(info["iterations"], 1)
        self.assertLessEqual(info["absolute_residual"], 1e-12)
        np.testing.assert_allclose(observed, expected, rtol=1e-9, atol=1e-10)

    def test_conjugate_gradient_tolerance_is_absolute(self) -> None:
        right_hand_side = np.asarray([1e-3, 0.0])
        observed, info = conjugate_gradient(
            lambda vector: vector,
            right_hand_side,
            absolute_tolerance=1e-2,
            max_iterations=2,
        )
        np.testing.assert_array_equal(observed, np.zeros(2))
        self.assertEqual(info["iterations"], 0)
        self.assertEqual(info["matvecs"], 0)
        self.assertAlmostEqual(info["absolute_residual"], 1e-3)
        self.assertAlmostEqual(info["relative_residual"], 1.0)

    def test_ula_step_matches_algorithm_equation(self) -> None:
        points = self.rng.normal(size=(4, self.problem.dim))
        noise = self.rng.normal(size=points.shape)
        step_size = 0.013
        temperature = 0.007
        expected = (
            points
            - step_size * self.problem.grad_g(self.theta, points)
            + np.sqrt(2.0 * temperature * step_size) * noise
        )
        observed = ula_step(
            self.problem,
            self.theta,
            points,
            step_size,
            temperature,
            noise,
        )
        np.testing.assert_array_equal(observed, expected)

    def test_cosine_learning_rate_schedule_has_expected_endpoints(self) -> None:
        config = {
            "theta_learning_rate": 0.01,
            "theta_learning_rate_end": 0.001,
            "theta_learning_rate_schedule": "cosine",
        }
        self.assertAlmostEqual(scheduled_learning_rate(config, 0.0), 0.01)
        self.assertAlmostEqual(scheduled_learning_rate(config, 0.5), 0.0055)
        self.assertAlmostEqual(scheduled_learning_rate(config, 1.0), 0.001)
        self.assertAlmostEqual(scheduled_learning_rate(config, 2.0), 0.001)
        self.assertAlmostEqual(
            scheduled_learning_rate({"theta_learning_rate": 0.007}, 0.5),
            0.007,
        )

    def test_scaled_gaussian_initialization_matches_analyzed_law(self) -> None:
        size = 5
        temperature = 0.003
        variance_scale = 0.01
        observed_rng = np.random.default_rng(991)
        reference_rng = np.random.default_rng(991)
        random_values = reference_rng.normal(size=(size, 2, self.problem.dim))
        directions = random_values[:, 0, : self.problem.y_dim]
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        center_radius = 2.0
        centers = np.zeros((size, self.problem.dim))
        centers[:, : self.problem.y_dim] = center_radius * directions
        expected = centers + np.sqrt(
            variance_scale * temperature
        ) * random_values[:, 1, :]
        observed = self.problem.sample_scaled_gaussian_sphere(
            center_radius,
            size,
            observed_rng,
            temperature,
            variance_scale,
        )
        np.testing.assert_array_equal(observed, expected)

        one = self.problem.sample_scaled_gaussian_sphere(
            center_radius,
            1,
            np.random.default_rng(1234),
            temperature,
            variance_scale,
        )
        many = self.problem.sample_scaled_gaussian_sphere(
            center_radius,
            16,
            np.random.default_rng(1234),
            temperature,
            variance_scale,
        )
        np.testing.assert_array_equal(one[0], many[0])

    def test_exact_gibbs_sampler_respects_embedded_factorization(self) -> None:
        temperature = 0.003
        samples = self.problem.sample_exact_gibbs(
            self.theta,
            20_000,
            np.random.default_rng(812),
            temperature,
            radial_grid_size=20_000,
        )
        self.assertEqual(samples.shape, (20_000, self.problem.dim))
        normal = samples[:, self.problem.y_dim :]
        observed_variance = float(np.mean(normal * normal))
        expected_variance = temperature / self.problem.normal_curvature
        self.assertAlmostEqual(
            observed_variance / expected_variance,
            1.0,
            delta=0.02,
        )

    def test_box_projection_can_represent_unconstrained_lower_level(self) -> None:
        value = np.asarray([-5.0, 2.0])
        np.testing.assert_array_equal(project_box(value, 0.0), value)
        np.testing.assert_array_equal(
            project_box(value, 3.0), np.asarray([-3.0, 2.0])
        )

    def test_ridge_cg_hypergradient_matches_dense_system(self) -> None:
        point = self.problem.exact_lower_point(self.theta)
        ridge = 0.07
        expected_solution = np.linalg.solve(
            self.problem.hessian_g(self.theta, point)
            + ridge * np.eye(self.problem.dim),
            self.problem.grad_f_x(self.theta, point),
        )
        expected_hypergradient = float(
            self.problem.partial_f_theta(self.theta, point)
            - np.dot(
                self.problem.mixed_grad_g_theta(self.theta, point),
                expected_solution,
            )
        )
        observed_hypergradient, observed_solution, info = ridge_cg_hypergradient(
            self.problem,
            self.theta,
            point,
            ridge,
            absolute_tolerance=1e-12,
            max_iterations=4 * self.problem.dim,
        )
        self.assertFalse(info["breakdown"])
        self.assertLessEqual(info["absolute_residual"], 1e-12)
        np.testing.assert_allclose(
            observed_solution, expected_solution, rtol=1e-9, atol=1e-10
        )
        self.assertAlmostEqual(
            observed_hypergradient, expected_hypergradient, places=10
        )

    def test_ridge_cg_tolerance_controls_nonoptimistic_manifold_solve(self) -> None:
        radius = self.problem.radius(self.theta)
        angle = 0.3
        point = -radius * (
            np.cos(angle) * self.problem.direction(self.theta)
            + np.sin(angle) * self.problem.direction_prime(self.theta)
        )
        ridge = 0.07
        dense_solution = np.linalg.solve(
            self.problem.hessian_g(self.theta, point)
            + ridge * np.eye(self.problem.dim),
            self.problem.grad_f_x(self.theta, point),
        )

        _, loose_solution, loose_info = ridge_cg_hypergradient(
            self.problem,
            self.theta,
            point,
            ridge,
            absolute_tolerance=1.0,
            max_iterations=4 * self.problem.dim,
        )
        _, strict_solution, strict_info = ridge_cg_hypergradient(
            self.problem,
            self.theta,
            point,
            ridge,
            absolute_tolerance=1e-8,
            max_iterations=4 * self.problem.dim,
        )

        self.assertEqual(loose_info["iterations"], 1)
        self.assertEqual(strict_info["iterations"], 2)
        self.assertLessEqual(loose_info["absolute_residual"], 1.0)
        self.assertLessEqual(strict_info["absolute_residual"], 1e-8)
        self.assertGreater(np.linalg.norm(loose_solution - dense_solution), 1.0)
        np.testing.assert_allclose(
            strict_solution, dense_solution, rtol=1e-10, atol=1e-11
        )

    def test_v_pbgd_gradients_match_penalized_objective(self) -> None:
        lower_point = self.x.copy()
        virtual_point = self.rng.normal(size=self.problem.dim)
        for penalty in (0.4, 7.0):
            with self.subTest(penalty=penalty):
                upper_scale, constraint_scale = penalty_scales(penalty)

                def objective(theta: float, point: np.ndarray) -> float:
                    return float(
                        upper_scale * self.problem.f(theta, point)
                        + constraint_scale
                        * (
                            self.problem.g(theta, point)
                            - self.problem.g(theta, virtual_point)
                        )
                    )

                observed = v_pbgd_objective_and_gradients(
                    self.problem,
                    self.theta,
                    lower_point,
                    virtual_point,
                    penalty,
                )
                numeric_theta = self.central_difference(
                    lambda theta: objective(theta, lower_point),
                    self.theta,
                )
                numeric_lower = self.vector_central_difference(
                    lambda point: objective(self.theta, point),
                    lower_point,
                )
                self.assertAlmostEqual(
                    observed["gradient_theta"], numeric_theta, places=6
                )
                np.testing.assert_allclose(
                    observed["gradient_lower"],
                    numeric_lower,
                    rtol=3e-6,
                    atol=3e-7,
                )

    def test_pbgd_scaling_matches_reference_implementation(self) -> None:
        self.assertEqual(penalty_scales(0.4), (1.0, 0.4))
        self.assertEqual(penalty_scales(7.0), (1.0 / 7.0, 1.0))

    def test_g_pbgd_gradients_match_stationarity_penalty(self) -> None:
        lower_point = self.x.copy()
        for penalty in (0.4, 7.0):
            with self.subTest(penalty=penalty):
                upper_scale, constraint_scale = penalty_scales(penalty)

                def objective(theta: float, point: np.ndarray) -> float:
                    lower_gradient = self.problem.grad_g(theta, point)
                    return float(
                        upper_scale * self.problem.f(theta, point)
                        + 0.5
                        * constraint_scale
                        * np.dot(lower_gradient, lower_gradient)
                    )

                observed = g_pbgd_objective_and_gradients(
                    self.problem,
                    self.theta,
                    lower_point,
                    penalty,
                )
                numeric_theta = self.central_difference(
                    lambda theta: objective(theta, lower_point),
                    self.theta,
                )
                numeric_lower = self.vector_central_difference(
                    lambda point: objective(self.theta, point),
                    lower_point,
                )
                self.assertAlmostEqual(
                    observed["gradient_theta"], numeric_theta, places=6
                )
                np.testing.assert_allclose(
                    observed["gradient_lower"],
                    numeric_lower,
                    rtol=4e-6,
                    atol=5e-7,
                )

    def test_iaptt_unrolled_gradients_match_finite_differences(self) -> None:
        initialization = self.x.copy()
        inner_learning_rate = 0.02
        inner_steps = 6
        observed = iaptt_unroll_and_gradients(
            self.problem,
            self.theta,
            initialization,
            inner_learning_rate,
            inner_steps,
            selection="max",
        )

        def selected_value(theta: float, point: np.ndarray) -> float:
            lower_point = point.copy()
            values = []
            for _ in range(inner_steps):
                lower_point = lower_point - inner_learning_rate * self.problem.grad_g(
                    theta, lower_point
                )
                values.append(float(self.problem.f(theta, lower_point)))
            return max(values)

        numeric_theta = self.central_difference(
            lambda theta: selected_value(theta, initialization), self.theta
        )
        numeric_initialization = self.vector_central_difference(
            lambda point: selected_value(self.theta, point), initialization
        )
        self.assertAlmostEqual(observed["gradient_theta"], numeric_theta, places=6)
        np.testing.assert_allclose(
            observed["gradient_initialization"],
            numeric_initialization,
            rtol=5e-6,
            atol=7e-7,
        )


class CacheCompatibilityTest(unittest.TestCase):
    def test_cache_key_includes_common_initialization_configuration(self) -> None:
        problem = {
            "k": 2,
            "dimension": 20,
            "active_dimension": 3,
            "ambient_dimension": 20,
            "kappa": 1.0,
            "normal_curvature": 1.0,
            "epsilon": 0.1,
        }
        common = {
            "initialization": "scaled_gaussian_sphere",
            "initialization_temperature": 0.001,
        }
        method_config = {"branches": 16}
        result = {
            "schema_version": 2,
            "method": "hg_ms",
            "problem": problem,
            "common_config": common,
            "method_config": method_config,
            "budget_seconds": 60.0,
            "seed": 1000,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "run.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            self.assertTrue(
                compatible_cached_run(
                    path,
                    "hg_ms",
                    problem,
                    common,
                    method_config,
                    60.0,
                    1000,
                )
            )
            changed_common = {
                **common,
                "initialization_temperature": 0.003,
            }
            self.assertFalse(
                compatible_cached_run(
                    path,
                    "hg_ms",
                    problem,
                    changed_common,
                    method_config,
                    60.0,
                    1000,
                )
            )

    def test_stale_hg_result_is_rejected_without_validation_crash(self) -> None:
        config = load_config()
        problem = make_problem(config, 2)
        stale_run = {
            "schema_version": 1,
            "method": "hg_ms",
            "seed": 1000,
            "problem": problem.metadata(),
            "common_config": dict(config["common"]),
            "method_config": dict(config["hg_ms"]),
            "budget_seconds": 2.0,
            "budget_overrun_seconds": 0.0,
            "final": {
                "hyperobjective_gap": 0.1,
                "cg_breakdowns": 0,
                "cg_tolerance_failures": 0,
            },
            "_path": "stale_hg_ms.json",
        }
        warnings = validate_runs(
            [stale_run],
            expected_k=[2],
            expected_seeds=[1000],
            expected_methods=["hg_ms"],
            problem_config=dict(config["problem"]),
            common_config=dict(config["common"]),
            method_configs={"hg_ms": dict(config["hg_ms"])},
        )
        self.assertTrue(
            any("stale schema" in warning for warning in warnings)
        )
        self.assertTrue(
            any("CG absolute residual" in warning for warning in warnings)
        )


class AlgorithmIntegrationTest(unittest.TestCase):
    def test_every_complete_method_path_produces_a_valid_run(self) -> None:
        config = load_config()
        problem = make_problem(config, 2)
        for method in METHOD_LABELS:
            with self.subTest(method=method):
                result = run_method(
                    method,
                    problem,
                    seed=1000,
                    budget_seconds=0.05,
                    config=config,
                )
                self.assertEqual(result["method"], method)
                self.assertGreater(result["final"]["iteration"], 0)
                self.assertTrue(
                    np.isfinite(result["final"]["hyperobjective_gap"])
                )
                self.assertTrue(np.isfinite(result["final"]["theta"]))
                self.assertEqual(result["problem"], problem.metadata())
                if method in {"hg_ms", "hg_baseline"}:
                    self.assertEqual(result["final"]["cg_breakdowns"], 0)
                    self.assertEqual(
                        result["final"]["cg_tolerance_failures"], 0
                    )


if __name__ == "__main__":
    unittest.main()
