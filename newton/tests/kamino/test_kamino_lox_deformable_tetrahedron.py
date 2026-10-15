# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the LOX stable Neo-Hookean tetrahedron constitutive model and proximal updates."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.lox.deformable.energy import (
    mat99f,
    tet_stable_neo_hookean_alpha,
    tet_stable_neo_hookean_differential,
    vec9f,
)
from newton._src.solvers.kamino._src.solvers.lox.deformable.tetrahedron_energy import (
    tet_stable_neo_hookean_eigensystem,
    tet_stable_neo_hookean_hessian,
    tet_stable_neo_hookean_spectral_metrics,
)
from newton._src.solvers.kamino._src.solvers.lox.deformable.tetrahedron_proximal import (
    _initialize_tetrahedron_proximal,
    _penalty_covers_curvature,
    _update_tetrahedron_proximal,
)
from newton.tests.kamino import setup_tests, test_context

# Keep the dense spectral-reference wrapper compact on a cold cache.
wp.set_module_options({"enable_backward": False, "max_unroll": 4})


@wp.kernel
def _evaluate_tetrahedron_constitutive(
    deformation: wp.array[wp.mat33],
    rest_volume: float,
    mu: float,
    k_lambda: float,
    activation: float,
    hessian: wp.array[mat99f],
    majorizer: wp.array[mat99f],
    gradient: wp.array[wp.mat33],
    alpha: wp.array[float],
):
    value = deformation[0]
    hessian[0] = tet_stable_neo_hookean_hessian(value, rest_volume, mu, k_lambda, activation)
    metrics = tet_stable_neo_hookean_spectral_metrics(
        value,
        rest_volume,
        mu,
        k_lambda,
        activation,
        0.0,
        32.0,
    )
    majorizer[0] = metrics.majorizer
    stress, _tangent = tet_stable_neo_hookean_differential(
        value,
        rest_volume,
        mu,
        k_lambda,
        activation,
    )
    gradient[0] = wp.mat33(
        stress[0],
        stress[3],
        stress[6],
        stress[1],
        stress[4],
        stress[7],
        stress[2],
        stress[5],
        stress[8],
    )
    alpha[0] = tet_stable_neo_hookean_alpha(mu, k_lambda)


@wp.kernel
def _evaluate_tetrahedron_eigensystem(
    deformation: wp.array[wp.mat33],
    mu: float,
    k_lambda: float,
    eigenvalues: wp.array[vec9f],
    eigenvectors: wp.array[mat99f],
):
    i = wp.tid()
    values, vectors = tet_stable_neo_hookean_eigensystem(deformation[i], 1.0 / 6.0, mu, k_lambda, 0.0)
    eigenvalues[i] = values
    eigenvectors[i] = vectors


@wp.kernel
def _evaluate_penalty_gate(
    deformation: wp.array[wp.mat33],
    frozen_metric: wp.array[mat99f],
    mu: float,
    k_lambda: float,
    covered: wp.array[wp.int32],
):
    i = wp.tid()
    covered[i] = wp.where(
        _penalty_covers_curvature(deformation[i], frozen_metric[0], 1.0 / 6.0, mu, k_lambda, 0.0), 1, 0
    )


class TestLOXDeformableTetrahedron(unittest.TestCase):
    """Verify the local tetrahedron split and its rejection boundaries."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def _constitutive(self, deformation, *, mu=2.0, k_lambda=3.0, activation=0.0):
        device = self.default_device
        deformation_array = wp.array([deformation], dtype=wp.mat33, device=device)
        hessian = wp.empty(1, dtype=mat99f, device=device)
        majorizer = wp.empty(1, dtype=mat99f, device=device)
        gradient = wp.empty(1, dtype=wp.mat33, device=device)
        alpha = wp.empty(1, dtype=wp.float32, device=device)
        wp.launch(
            _evaluate_tetrahedron_constitutive,
            dim=1,
            inputs=[deformation_array, 1.0 / 6.0, mu, k_lambda, activation],
            outputs=[hessian, majorizer, gradient, alpha],
            device=device,
        )
        return hessian.numpy()[0], majorizer.numpy()[0], gradient.numpy()[0], float(alpha.numpy()[0])

    def _make_update_arrays(self, positions, *, mu=2.0, k_lambda=3.0, activation=0.0):
        device = self.default_device
        positions = np.asarray(positions, dtype=np.float32)
        indices = wp.array([[0, 1, 2, 3]], dtype=wp.int32, device=device)
        rest_pose = wp.array([np.eye(3, dtype=np.float32)], dtype=wp.mat33, device=device)
        materials = wp.array([[mu, k_lambda, 0.0]], dtype=wp.float32, device=device)
        activations = wp.array([activation], dtype=wp.float32, device=device)
        metric_np = self._constitutive(
            np.column_stack((positions[1] - positions[0], positions[2] - positions[0], positions[3] - positions[0])),
            mu=mu,
            k_lambda=k_lambda,
            activation=activation,
        )[1]
        arrays = {
            "position_start": wp.array(positions, dtype=wp.vec3, device=device),
            "position_linearized": wp.array(positions, dtype=wp.vec3, device=device),
            "smooth_velocity": wp.zeros(4, dtype=wp.vec3, device=device),
            "indices": indices,
            "rest_pose": rest_pose,
            "materials": materials,
            "activations": activations,
            "packed_to_newton": wp.array([0, 1, 2, 3], dtype=wp.int32, device=device),
            "packed_world": wp.zeros(4, dtype=wp.int32, device=device),
            "particle_mass": wp.ones(4, dtype=wp.float32, device=device),
            "particle_flags": wp.ones(4, dtype=wp.int32, device=device),
            "world_active": wp.ones(1, dtype=wp.bool, device=device),
            "frozen_coordinate": wp.empty(1, dtype=wp.mat33, device=device),
            "frozen_gradient": wp.empty(1, dtype=wp.mat33, device=device),
            "frozen_metric": wp.array([metric_np], dtype=mat99f, device=device),
            "frozen_factor": wp.empty(1, dtype=mat99f, device=device),
            "proximal_coordinate": wp.empty(1, dtype=wp.mat33, device=device),
            "multiplier": wp.empty(1, dtype=wp.mat33, device=device),
            "nonlinear_rhs": wp.zeros(4, dtype=wp.vec3, device=device),
            "position_residual": wp.zeros(1, dtype=wp.float32, device=device),
            "velocity_residual": wp.zeros(1, dtype=wp.float32, device=device),
            "failed": wp.zeros(1, dtype=wp.bool, device=device),
            "time_step": wp.array([0.01], dtype=wp.float32, device=device),
        }
        wp.launch(
            _initialize_tetrahedron_proximal,
            dim=1,
            inputs=[
                arrays["position_linearized"],
                indices,
                rest_pose,
                materials,
                activations,
                arrays["frozen_metric"],
            ],
            outputs=[
                arrays["frozen_coordinate"],
                arrays["frozen_gradient"],
                arrays["frozen_factor"],
                arrays["proximal_coordinate"],
                arrays["multiplier"],
            ],
            device=device,
        )
        return arrays

    def _update(self, arrays, *, iterations=1, relaxation=1.0):
        wp.launch(
            _update_tetrahedron_proximal,
            dim=1,
            inputs=[
                arrays["position_start"],
                arrays["position_linearized"],
                arrays["smooth_velocity"],
                arrays["indices"],
                arrays["rest_pose"],
                arrays["materials"],
                arrays["activations"],
                arrays["packed_to_newton"],
                arrays["packed_world"],
                arrays["particle_mass"],
                arrays["particle_flags"],
                arrays["world_active"],
                arrays["frozen_coordinate"],
                arrays["frozen_gradient"],
                arrays["frozen_metric"],
                arrays["frozen_factor"],
                iterations,
                relaxation,
                arrays["time_step"],
            ],
            outputs=[
                arrays["proximal_coordinate"],
                arrays["multiplier"],
                arrays["nonlinear_rhs"],
                arrays["position_residual"],
                arrays["velocity_residual"],
                arrays["failed"],
            ],
            device=self.default_device,
        )

    def test_spectral_majorizer_matches_dense_reference(self):
        """Match the frozen spectral metric to a dense eigendecomposition."""
        deformation = np.diag(np.array((0.3, 0.8, 1.2), dtype=np.float32))
        hessian, majorizer, _gradient, _alpha = self._constitutive(deformation)
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
        entry_scale = np.max(np.abs(hessian))
        active = np.abs(eigenvalues) > 1.0e-6 * entry_scale
        negative_curvature = max(0.0, -float(np.min(eigenvalues[active])))
        metric_eigenvalues = np.where(active, np.abs(eigenvalues), 0.0) + 32.0 * negative_curvature
        expected = (eigenvectors * metric_eigenvalues) @ eigenvectors.T
        np.testing.assert_allclose(majorizer, expected, rtol=2.0e-4, atol=2.0e-4)
        np.testing.assert_allclose(majorizer, majorizer.T, rtol=0.0, atol=2.0e-5)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(majorizer))), -2.0e-4)

    def test_hessian_matches_stress_differences(self):
        """Match the Hessian to central differences of the stress at stretched, compressed and inverted states."""
        rng = np.random.default_rng(3)
        samples = [
            np.diag([1.3, 0.9, 1.1]),
            np.diag([0.4, 0.6, 0.5]),
            np.diag([1.0, 1.0, -0.4]),
            np.eye(3) + 0.3 * rng.normal(size=(3, 3)),
        ]
        step = 1.0e-3
        for deformation in samples:
            with self.subTest(determinant=float(np.linalg.det(deformation))):
                hessian, _majorizer, _gradient, _alpha = self._constitutive(deformation.astype(np.float32))
                differences = np.zeros((9, 9))
                for column in range(9):
                    offset = np.zeros(9)
                    offset[column] = step
                    shift = offset.reshape((3, 3), order="F")
                    forward = self._constitutive((deformation + shift).astype(np.float32))[2]
                    backward = self._constitutive((deformation - shift).astype(np.float32))[2]
                    differences[:, column] = (forward - backward).flatten(order="F") / (2.0 * step)
                np.testing.assert_allclose(hessian, differences, rtol=0.0, atol=2.0e-3 * np.abs(hessian).max())

    def test_semi_analytic_eigensystem_reconstructs_hessian(self):
        """Recover the Hessian from orthonormal semi-analytic eigenpairs, including inverted states."""
        rng = np.random.default_rng(5)
        samples = [np.eye(3), np.diag([0.5, 0.5, 0.5]), np.diag([1.0, 1.0, -0.3])]
        samples += list(np.eye(3) + 0.3 * rng.normal(size=(4, 3, 3)))
        samples += list(rng.normal(size=(4, 3, 3)))
        device = self.default_device
        deformations = wp.array(np.asarray(samples, dtype=np.float32), dtype=wp.mat33, device=device)
        eigenvalues = wp.empty(len(samples), dtype=vec9f, device=device)
        eigenvectors = wp.empty(len(samples), dtype=mat99f, device=device)
        wp.launch(
            _evaluate_tetrahedron_eigensystem,
            dim=len(samples),
            inputs=[deformations, 2.0, 3.0],
            outputs=[eigenvalues, eigenvectors],
            device=device,
        )
        for i, deformation in enumerate(samples):
            with self.subTest(determinant=float(np.linalg.det(deformation))):
                hessian = self._constitutive(np.asarray(deformation, dtype=np.float32))[0]
                values = eigenvalues.numpy()[i]
                vectors = eigenvectors.numpy()[i]
                scale = np.abs(hessian).max()
                np.testing.assert_allclose(vectors @ vectors.T, np.eye(9), atol=2.0e-5)
                # The float32 3-by-3 eigensolver resolves strongly inverted scaling blocks to about 1e-4 of the scale
                np.testing.assert_allclose((vectors.T * values) @ vectors, hessian, atol=5.0e-4 * scale)

    def test_penalty_gate_bounds_local_curvature(self):
        """Accept local coordinates near the frozen state and reject those whose curvature exceeds the penalty."""
        frozen = np.eye(3, dtype=np.float32)
        _hessian, majorizer, _gradient, _alpha = self._constitutive(frozen)
        device = self.default_device
        samples = np.asarray(
            [frozen, np.diag([1.05, 0.98, 1.0]), np.diag([2.0, 1.0, 1.0]), np.diag([0.3, 1.0, 1.0])],
            dtype=np.float32,
        )
        covered = wp.empty(len(samples), dtype=wp.int32, device=device)
        wp.launch(
            _evaluate_penalty_gate,
            dim=len(samples),
            inputs=[
                wp.array(samples, dtype=wp.mat33, device=device),
                wp.array([majorizer], dtype=mat99f, device=device),
                2.0,
                3.0,
            ],
            outputs=[covered],
            device=device,
        )
        np.testing.assert_array_equal(covered.numpy(), [1, 1, 0, 0])

    def test_uncovered_curvature_keeps_linearized_response(self):
        """Keep the linearized response when the candidate stretches far beyond the frozen penalty."""
        positions = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float32)
        arrays = self._make_update_arrays(positions)
        arrays["smooth_velocity"].assign(np.array(((0, 0, 0), (100, 0, 0), (0, 0, 0), (0, 0, 0)), dtype=np.float32))
        self._update(arrays, iterations=4)
        self.assertFalse(bool(arrays["failed"].numpy()[0]))
        np.testing.assert_array_equal(arrays["nonlinear_rhs"].numpy(), np.zeros((4, 3), dtype=np.float32))
        center = np.diag(np.array((2.0, 1.0, 1.0), dtype=np.float32))
        np.testing.assert_allclose(arrays["proximal_coordinate"].numpy()[0], center, atol=1.0e-5)
        offset = (center - arrays["frozen_coordinate"].numpy()[0]).flatten(order="F")
        metric_offset = (arrays["frozen_metric"].numpy()[0] @ offset).reshape((3, 3), order="F")
        np.testing.assert_allclose(
            arrays["multiplier"].numpy()[0],
            arrays["frozen_gradient"].numpy()[0] + metric_offset,
            rtol=1.0e-5,
            atol=1.0e-5,
        )

    def test_tiny_coefficients_share_constitutive_alpha(self):
        """Use one regularized volume offset in assembly and the local prox."""
        mu = 4.0e-7
        k_lambda = 3.0e-7
        positions = np.array(((0, 0, 0), (1.2, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float32)
        arrays = self._make_update_arrays(positions, mu=mu, k_lambda=k_lambda)
        deformation = np.diag(np.array((1.2, 1.0, 1.0), dtype=np.float32))
        _hessian, _majorizer, expected_gradient, alpha = self._constitutive(
            deformation,
            mu=mu,
            k_lambda=k_lambda,
        )
        self.assertAlmostEqual(alpha, 1.4, places=6)
        np.testing.assert_allclose(arrays["frozen_gradient"].numpy()[0], expected_gradient, rtol=2.0e-6, atol=1.0e-12)

    def test_fixed_point_is_finite_for_singular_and_inverted_tets(self):
        """Keep finite singular and inverted stable-energy coordinates evaluable."""
        for first_column in (0.0, -0.5):
            with self.subTest(first_column=first_column):
                positions = np.array(
                    ((0, 0, 0), (first_column, 0, 0), (0, 1, 0), (0, 0, 1)),
                    dtype=np.float32,
                )
                arrays = self._make_update_arrays(positions)
                coordinate_before = arrays["proximal_coordinate"].numpy().copy()
                self._update(arrays)
                self.assertFalse(bool(arrays["failed"].numpy()[0]))
                self.assertTrue(np.all(np.isfinite(arrays["nonlinear_rhs"].numpy())))
                np.testing.assert_allclose(arrays["proximal_coordinate"].numpy(), coordinate_before, atol=2.0e-6)
                np.testing.assert_allclose(arrays["nonlinear_rhs"].numpy(), 0.0, atol=2.0e-6)

    def test_exhausted_local_budget_reports_stationarity(self):
        """Report KKT error when a finite local coordinate makes no progress."""
        positions = np.array(((0, 0, 0), (1.2, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float32)
        arrays = self._make_update_arrays(positions)
        arrays["multiplier"].zero_()
        self._update(arrays, iterations=0)
        self.assertFalse(bool(arrays["failed"].numpy()[0]))
        self.assertGreater(float(arrays["position_residual"].numpy()[0]), 1.0e-4)
        self.assertEqual(float(arrays["velocity_residual"].numpy()[0]), 0.0)

    def test_invalid_scatter_keeps_linearized_response(self):
        """Keep the linearized response, without a nodal RHS, when the correction scatter overflows."""
        positions = np.array(((0, 0, 0), (1.0e-30, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float32)
        arrays = self._make_update_arrays(positions)
        rest_pose = np.diag(np.array((1.0e30, 1.0, 1.0), dtype=np.float32))
        arrays["rest_pose"].assign([rest_pose])
        arrays["frozen_metric"].assign([np.eye(9, dtype=np.float32)])
        arrays["frozen_factor"].assign([np.eye(9, dtype=np.float32)])
        arrays["frozen_gradient"].fill_(1.0e12)
        self._update(arrays, iterations=0)
        self.assertFalse(bool(arrays["failed"].numpy()[0]))
        np.testing.assert_array_equal(arrays["nonlinear_rhs"].numpy(), np.zeros((4, 3), dtype=np.float32))
        center = np.column_stack((positions[1], positions[2], positions[3])).astype(np.float64) @ rest_pose
        np.testing.assert_allclose(arrays["proximal_coordinate"].numpy()[0], center, rtol=1.0e-6)
        # With the identity metric, the linearized stress is the frozen gradient plus the deformation offset
        linearized_stress = arrays["frozen_gradient"].numpy()[0] + center - arrays["frozen_coordinate"].numpy()[0]
        np.testing.assert_allclose(arrays["multiplier"].numpy()[0], linearized_stress, rtol=1.0e-6)

    def test_failed_local_solve_keeps_linearized_response(self):
        """Reset the element to its linearized stress at the candidate when the local solve fails."""
        positions = np.array(((0, 0, 0), (1.2, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float32)
        arrays = self._make_update_arrays(positions)
        arrays["smooth_velocity"].assign(np.array(((0, 0, 0), (-5, 1, 0), (0, -3, 2), (1, 0, -4)), dtype=np.float32))
        arrays["frozen_factor"].zero_()
        self._update(arrays)
        self.assertFalse(bool(arrays["failed"].numpy()[0]))
        np.testing.assert_array_equal(arrays["nonlinear_rhs"].numpy(), np.zeros((4, 3), dtype=np.float32))
        candidate = positions + 0.01 * arrays["smooth_velocity"].numpy()
        center = np.column_stack(
            (candidate[1] - candidate[0], candidate[2] - candidate[0], candidate[3] - candidate[0])
        )
        np.testing.assert_allclose(arrays["proximal_coordinate"].numpy()[0], center, rtol=1.0e-5, atol=1.0e-6)
        # The deformation vectors stack the columns of the deformation gradient
        offset = (center - arrays["frozen_coordinate"].numpy()[0]).flatten(order="F")
        metric_offset = (arrays["frozen_metric"].numpy()[0] @ offset).reshape((3, 3), order="F")
        linearized_stress = arrays["frozen_gradient"].numpy()[0] + metric_offset
        np.testing.assert_allclose(arrays["multiplier"].numpy()[0], linearized_stress, rtol=1.0e-5, atol=1.0e-5)

    def test_nonfinite_candidate_fails_world(self):
        """Fail the world and keep the element state when the candidate deformation is not finite."""
        positions = np.array(((0, 0, 0), (1.2, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float32)
        arrays = self._make_update_arrays(positions)
        arrays["smooth_velocity"].fill_(wp.vec3(float("nan"), 0.0, 0.0))
        coordinate_before = arrays["proximal_coordinate"].numpy().copy()
        multiplier_before = arrays["multiplier"].numpy().copy()
        self._update(arrays)
        self.assertTrue(bool(arrays["failed"].numpy()[0]))
        np.testing.assert_array_equal(arrays["nonlinear_rhs"].numpy(), np.zeros((4, 3), dtype=np.float32))
        np.testing.assert_array_equal(arrays["proximal_coordinate"].numpy(), coordinate_before)
        np.testing.assert_array_equal(arrays["multiplier"].numpy(), multiplier_before)

    def test_cuda_graph_replays_tetrahedron_proximal(self):
        """Capture and replay a finite tetrahedron proximal update."""
        device = self.default_device
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        positions = np.array(((0, 0, 0), (1.2, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float32)
        arrays = self._make_update_arrays(positions)
        self._update(arrays)
        with wp.ScopedCapture(device=device) as capture:
            self._update(arrays)
        wp.capture_launch(capture.graph)
        self.assertFalse(bool(arrays["failed"].numpy()[0]))
        self.assertTrue(np.all(np.isfinite(arrays["proximal_coordinate"].numpy())))


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
