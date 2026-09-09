# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused constitutive and safety tests for the LOX tetrahedron proximal."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.lox.deformable_energy import (
    mat99,
    tet_stable_neo_hookean_alpha,
    tet_stable_neo_hookean_differential,
)
from newton._src.solvers.kamino._src.solvers.lox.deformable_tetrahedron_energy import (
    tet_stable_neo_hookean_hessian,
    tet_stable_neo_hookean_spectral_metrics,
)
from newton._src.solvers.kamino._src.solvers.lox.deformable_tetrahedron_proximal import (
    initialize_tetrahedron_proximal,
    update_tetrahedron_proximal,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context

# Keep the dense spectral-reference wrapper compact on a cold cache.
wp.set_module_options({"enable_backward": False, "max_unroll": 4})


@wp.kernel
def _evaluate_tetrahedron_constitutive(
    deformation: wp.array[wp.mat33],
    rest_volume: float,
    mu: float,
    k_lambda: float,
    activation: float,
    hessian: wp.array[mat99],
    majorizer: wp.array[mat99],
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


class TestLOXTetrahedronProximal(unittest.TestCase):
    """Verify the local tetrahedron split and its rejection boundaries."""

    def setUp(self):
        """Select the configured Newton test device."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def _constitutive(self, deformation, *, mu=2.0, k_lambda=3.0, activation=0.0):
        deformation_array = wp.array([deformation], dtype=wp.mat33, device=self.device)
        hessian = wp.empty(1, dtype=mat99, device=self.device)
        majorizer = wp.empty(1, dtype=mat99, device=self.device)
        gradient = wp.empty(1, dtype=wp.mat33, device=self.device)
        alpha = wp.empty(1, dtype=wp.float32, device=self.device)
        wp.launch(
            _evaluate_tetrahedron_constitutive,
            dim=1,
            inputs=[deformation_array, 1.0 / 6.0, mu, k_lambda, activation],
            outputs=[hessian, majorizer, gradient, alpha],
            device=self.device,
        )
        return hessian.numpy()[0], majorizer.numpy()[0], gradient.numpy()[0], float(alpha.numpy()[0])

    def _make_update_arrays(self, positions, *, mu=2.0, k_lambda=3.0, activation=0.0):
        positions = np.asarray(positions, dtype=np.float32)
        indices = wp.array([[0, 1, 2, 3]], dtype=wp.int32, device=self.device)
        rest_pose = wp.array([np.eye(3, dtype=np.float32)], dtype=wp.mat33, device=self.device)
        materials = wp.array([[mu, k_lambda, 0.0]], dtype=wp.float32, device=self.device)
        activations = wp.array([activation], dtype=wp.float32, device=self.device)
        metric_np = self._constitutive(
            np.column_stack((positions[1] - positions[0], positions[2] - positions[0], positions[3] - positions[0])),
            mu=mu,
            k_lambda=k_lambda,
            activation=activation,
        )[1]
        arrays = {
            "position_start": wp.array(positions, dtype=wp.vec3, device=self.device),
            "position_linearized": wp.array(positions, dtype=wp.vec3, device=self.device),
            "smooth_velocity": wp.zeros(4, dtype=wp.vec3, device=self.device),
            "indices": indices,
            "rest_pose": rest_pose,
            "materials": materials,
            "activations": activations,
            "packed_to_newton": wp.array([0, 1, 2, 3], dtype=wp.int32, device=self.device),
            "packed_world": wp.zeros(4, dtype=wp.int32, device=self.device),
            "particle_mass": wp.ones(4, dtype=wp.float32, device=self.device),
            "particle_flags": wp.ones(4, dtype=wp.int32, device=self.device),
            "world_active": wp.ones(1, dtype=wp.int32, device=self.device),
            "frozen_coordinate": wp.empty(1, dtype=wp.mat33, device=self.device),
            "frozen_gradient": wp.empty(1, dtype=wp.mat33, device=self.device),
            "frozen_metric": wp.array([metric_np], dtype=mat99, device=self.device),
            "frozen_factor": wp.empty(1, dtype=mat99, device=self.device),
            "proximal_coordinate": wp.empty(1, dtype=wp.mat33, device=self.device),
            "multiplier": wp.empty(1, dtype=wp.mat33, device=self.device),
            "nonlinear_rhs": wp.zeros(4, dtype=wp.vec3, device=self.device),
            "position_residual": wp.zeros(1, dtype=wp.float32, device=self.device),
            "velocity_residual": wp.zeros(1, dtype=wp.float32, device=self.device),
            "failed": wp.zeros(1, dtype=wp.int32, device=self.device),
            "time_step": wp.array([0.01], dtype=wp.float32, device=self.device),
        }
        wp.launch(
            initialize_tetrahedron_proximal,
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
            device=self.device,
        )
        return arrays

    def _update(self, arrays, *, iterations=1, relaxation=1.0):
        wp.launch(
            update_tetrahedron_proximal,
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
            device=self.device,
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
                self.assertEqual(int(arrays["failed"].numpy()[0]), 0)
                self.assertTrue(np.all(np.isfinite(arrays["nonlinear_rhs"].numpy())))
                np.testing.assert_allclose(arrays["proximal_coordinate"].numpy(), coordinate_before, atol=2.0e-6)
                np.testing.assert_allclose(arrays["nonlinear_rhs"].numpy(), 0.0, atol=2.0e-6)

    def test_exhausted_local_budget_reports_stationarity(self):
        """Report KKT error when a finite local coordinate makes no progress."""
        positions = np.array(((0, 0, 0), (1.2, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float32)
        arrays = self._make_update_arrays(positions)
        arrays["multiplier"].zero_()
        self._update(arrays, iterations=0)
        self.assertEqual(int(arrays["failed"].numpy()[0]), 0)
        self.assertGreater(float(arrays["position_residual"].numpy()[0]), 1.0e-4)
        self.assertEqual(float(arrays["velocity_residual"].numpy()[0]), 0.0)

    def test_invalid_scatter_preserves_element_state(self):
        """Reject overflow before writing element state or a nodal RHS."""
        positions = np.array(((0, 0, 0), (1.0e-30, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float32)
        arrays = self._make_update_arrays(positions)
        arrays["rest_pose"].assign([np.diag(np.array((1.0e30, 1.0, 1.0), dtype=np.float32))])
        arrays["frozen_metric"].assign([np.eye(9, dtype=np.float32)])
        arrays["frozen_factor"].assign([np.eye(9, dtype=np.float32)])
        arrays["frozen_gradient"].fill_(1.0e12)
        coordinate_before = arrays["proximal_coordinate"].numpy().copy()
        multiplier_before = arrays["multiplier"].numpy().copy()
        self._update(arrays, iterations=0)
        self.assertEqual(int(arrays["failed"].numpy()[0]), 1)
        np.testing.assert_array_equal(arrays["nonlinear_rhs"].numpy(), np.zeros((4, 3), dtype=np.float32))
        np.testing.assert_array_equal(arrays["proximal_coordinate"].numpy(), coordinate_before)
        np.testing.assert_array_equal(arrays["multiplier"].numpy(), multiplier_before)

    def test_cuda_graph_replays_tetrahedron_proximal(self):
        """Capture and replay a finite tetrahedron proximal update."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        positions = np.array(((0, 0, 0), (1.2, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float32)
        arrays = self._make_update_arrays(positions)
        self._update(arrays)
        with wp.ScopedCapture(device=self.device) as capture:
            self._update(arrays)
        wp.capture_launch(capture.graph)
        self.assertEqual(int(arrays["failed"].numpy()[0]), 0)
        self.assertTrue(np.all(np.isfinite(arrays["proximal_coordinate"].numpy())))


if __name__ == "__main__":
    unittest.main(verbosity=2)
