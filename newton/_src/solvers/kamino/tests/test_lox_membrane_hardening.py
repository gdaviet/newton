# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for robust LOX membrane geometry and proximal updates."""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.lox.deformable_assembly import assemble_triangle_system
from newton._src.solvers.kamino._src.solvers.lox.deformable_membrane import membrane_area_ratio_gradient
from newton._src.solvers.kamino._src.solvers.lox.deformable_proximal import (
    initialize_membrane_proximal,
    update_membrane_proximal,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


@wp.kernel
def _evaluate_area_gradient(
    deformation: wp.array2d[wp.vec3],
    area_ratio: wp.array[float],
    gradient: wp.array2d[wp.vec3],
):
    sample = wp.tid()
    area, gradient_0, gradient_1 = membrane_area_ratio_gradient(
        deformation[sample, 0],
        deformation[sample, 1],
    )
    area_ratio[sample] = area
    gradient[sample, 0] = gradient_0
    gradient[sample, 1] = gradient_1


class TestLOXMembraneHardening(unittest.TestCase):
    """Exercise membrane degeneracy and transactional update boundaries."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def _geometry(self, deformation_0, deformation_1):
        positions = np.asarray(
            ((0.0, 0.0, 0.0), deformation_0, deformation_1),
            dtype=np.float32,
        )
        return {
            "positions": wp.array(positions, dtype=wp.vec3, device=self.device),
            "indices": wp.array(np.asarray(((0, 1, 2),), dtype=np.int32), dtype=wp.int32, device=self.device),
            "poses": wp.array(
                (wp.mat22(1.0, 0.0, 0.0, 1.0),),
                dtype=wp.mat22,
                device=self.device,
            ),
            "areas": wp.array((1.0,), dtype=wp.float32, device=self.device),
            "packed_to_newton": wp.array((0, 1, 2), dtype=wp.int32, device=self.device),
            "packed_world": wp.zeros(3, dtype=wp.int32, device=self.device),
            "mass": wp.ones(3, dtype=wp.float32, device=self.device),
            "flags": wp.ones(3, dtype=wp.int32, device=self.device),
            "time_step": wp.array((0.01,), dtype=wp.float32, device=self.device),
        }

    def _assemble(self, geometry, materials, *, positions_start=None):
        material_array = wp.array(
            np.asarray((materials,), dtype=np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        force = wp.zeros(3, dtype=wp.vec3, device=self.device)
        blocks = wp.zeros(9, dtype=wp.mat33, device=self.device)
        failed = wp.zeros(1, dtype=wp.int32, device=self.device)
        positions = geometry["positions"]
        wp.launch(
            assemble_triangle_system,
            dim=1,
            inputs=[
                positions if positions_start is None else positions_start,
                positions,
                wp.zeros(3, dtype=wp.vec3, device=self.device),
                geometry["indices"],
                geometry["poses"],
                geometry["areas"],
                material_array,
                wp.zeros(1, dtype=wp.float32, device=self.device),
                geometry["packed_to_newton"],
                geometry["packed_world"],
                geometry["mass"],
                geometry["flags"],
                geometry["time_step"],
                0,
            ],
            outputs=[blocks, force, failed],
            device=self.device,
        )
        return force.numpy(), blocks.numpy(), failed.numpy()

    def _initialize_proximal(self, geometry, materials, activation=0.0):
        material_array = wp.array(
            np.asarray((materials,), dtype=np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        activation_array = wp.array((activation,), dtype=wp.float32, device=self.device)
        arrays = {
            "frozen_coordinate": wp.zeros((1, 2), dtype=wp.vec3, device=self.device),
            "frozen_gradient": wp.zeros((1, 2), dtype=wp.vec3, device=self.device),
            "frozen_area_gradient": wp.zeros((1, 2), dtype=wp.vec3, device=self.device),
            "coordinate": wp.zeros((1, 2), dtype=wp.vec3, device=self.device),
            "multiplier": wp.zeros((1, 2), dtype=wp.vec3, device=self.device),
        }
        wp.launch(
            initialize_membrane_proximal,
            dim=1,
            inputs=[
                geometry["positions"],
                geometry["indices"],
                geometry["poses"],
                geometry["areas"],
                material_array,
                activation_array,
            ],
            outputs=[
                arrays["frozen_coordinate"],
                arrays["frozen_gradient"],
                arrays["frozen_area_gradient"],
                arrays["coordinate"],
                arrays["multiplier"],
            ],
            device=self.device,
        )
        return material_array, activation_array, arrays

    def _update_proximal(
        self,
        geometry,
        material_array,
        activation_array,
        arrays,
        *,
        velocity=None,
        iterations=1,
    ):
        nonlinear_rhs = wp.zeros(3, dtype=wp.vec3, device=self.device)
        position_residual = wp.zeros(1, dtype=wp.float32, device=self.device)
        velocity_residual = wp.zeros(1, dtype=wp.float32, device=self.device)
        failed = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            update_membrane_proximal,
            dim=1,
            inputs=[
                geometry["positions"],
                geometry["positions"],
                wp.zeros(3, dtype=wp.vec3, device=self.device) if velocity is None else velocity,
                geometry["indices"],
                geometry["poses"],
                geometry["areas"],
                material_array,
                activation_array,
                geometry["packed_to_newton"],
                geometry["packed_world"],
                geometry["mass"],
                geometry["flags"],
                wp.ones(1, dtype=wp.int32, device=self.device),
                arrays["frozen_coordinate"],
                arrays["frozen_gradient"],
                arrays["frozen_area_gradient"],
                iterations,
                0.5,
                geometry["time_step"],
            ],
            outputs=[
                arrays["coordinate"],
                arrays["multiplier"],
                nonlinear_rhs,
                position_residual,
                velocity_residual,
                failed,
            ],
            device=self.device,
        )
        return nonlinear_rhs.numpy(), position_residual.numpy(), velocity_residual.numpy(), failed.numpy()

    def test_sliver_area_gradient_and_force_avoid_gram_cancellation(self):
        """Keep a resolvable sliver's area gradient and force at physical scale."""
        geometry = self._geometry((1.0, 0.0, 0.0), (1.0, 1.0e-4, 0.0))
        deformation = wp.array(
            np.asarray((((1.0, 0.0, 0.0), (1.0, 1.0e-4, 0.0)),), dtype=np.float32),
            dtype=wp.vec3,
            device=self.device,
        )
        area = wp.empty(1, dtype=wp.float32, device=self.device)
        gradient = wp.empty((1, 2), dtype=wp.vec3, device=self.device)
        wp.launch(
            _evaluate_area_gradient,
            dim=1,
            inputs=[deformation],
            outputs=[area, gradient],
            device=self.device,
        )

        self.assertAlmostEqual(float(area.numpy()[0]), 1.0e-4, places=7)
        np.testing.assert_allclose(
            gradient.numpy()[0],
            ((1.0e-4, -1.0, 0.0), (0.0, 1.0, 0.0)),
            rtol=2.0e-5,
            atol=2.0e-7,
        )
        force, blocks, failed = self._assemble(geometry, (0.0, 1.0, 0.0, 0.0, 0.0))
        np.testing.assert_array_equal(failed, 0)
        self.assertTrue(np.isfinite(blocks).all())
        self.assertGreater(float(np.max(np.abs(force))), 0.9)
        self.assertLess(float(np.max(np.abs(force))), 2.0)

    def test_area_gradient_and_assembly_rotate_covariantly(self):
        """Rotate membrane gradients, forces, and tangents with the element."""
        angle = 0.7
        rotation = np.asarray(
            (
                (math.cos(angle), -math.sin(angle), 0.0),
                (math.sin(angle), math.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        )
        columns = np.asarray(((1.2, 0.1, 0.3), (-0.2, 0.8, 0.4)), dtype=np.float32)
        base = self._geometry(columns[0], columns[1])
        rotated = self._geometry(rotation @ columns[0], rotation @ columns[1])
        deformation = wp.array(
            np.asarray(((columns[0], columns[1]), (rotation @ columns[0], rotation @ columns[1]))),
            dtype=wp.vec3,
            device=self.device,
        )
        area = wp.empty(2, dtype=wp.float32, device=self.device)
        gradient = wp.empty((2, 2), dtype=wp.vec3, device=self.device)
        wp.launch(
            _evaluate_area_gradient,
            dim=2,
            inputs=[deformation],
            outputs=[area, gradient],
            device=self.device,
        )
        gradient_host = gradient.numpy()
        self.assertAlmostEqual(float(area.numpy()[0]), float(area.numpy()[1]), places=6)
        np.testing.assert_allclose(gradient_host[1], gradient_host[0] @ rotation.T, rtol=2.0e-5, atol=2.0e-6)

        force_base, blocks_base, failed_base = self._assemble(base, (2.0, 3.0, 0.4, 0.0, 0.0))
        force_rotated, blocks_rotated, failed_rotated = self._assemble(rotated, (2.0, 3.0, 0.4, 0.0, 0.0))

        np.testing.assert_array_equal(failed_base, 0)
        np.testing.assert_array_equal(failed_rotated, 0)
        np.testing.assert_allclose(force_rotated, force_base @ rotation.T, rtol=2.0e-5, atol=2.0e-6)
        for block_base, block_rotated in zip(blocks_base, blocks_rotated, strict=True):
            np.testing.assert_allclose(
                block_rotated,
                rotation @ block_base @ rotation.T,
                rtol=3.0e-5,
                atol=2.0e-6,
            )

    def test_collapsed_membrane_is_finite_and_coherently_regularized(self):
        """Keep collapsed membrane geometry, assembly, and proximal state finite."""
        geometry = self._geometry((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        deformation = wp.zeros((1, 2), dtype=wp.vec3, device=self.device)
        area = wp.empty(1, dtype=wp.float32, device=self.device)
        gradient = wp.empty((1, 2), dtype=wp.vec3, device=self.device)
        wp.launch(
            _evaluate_area_gradient,
            dim=1,
            inputs=[deformation],
            outputs=[area, gradient],
            device=self.device,
        )
        self.assertAlmostEqual(float(area.numpy()[0]), 1.0e-8, places=12)
        np.testing.assert_array_equal(gradient.numpy(), 0.0)

        force, blocks, assembly_failed = self._assemble(geometry, (2.0, 3.0, 0.0, 0.0, 0.0))
        np.testing.assert_array_equal(assembly_failed, 0)
        self.assertTrue(np.isfinite(force).all())
        self.assertTrue(np.isfinite(blocks).all())
        materials, activation, arrays = self._initialize_proximal(geometry, (2.0, 3.0, 0.0, 0.0, 0.0))
        rhs, position_residual, velocity_residual, proximal_failed = self._update_proximal(
            geometry,
            materials,
            activation,
            arrays,
        )
        np.testing.assert_array_equal(proximal_failed, 0)
        self.assertTrue(np.isfinite(rhs).all())
        self.assertTrue(np.isfinite(arrays["coordinate"].numpy()).all())
        self.assertTrue(np.isfinite(arrays["multiplier"].numpy()).all())
        self.assertTrue(np.isfinite(position_residual).all())
        self.assertTrue(np.isfinite(velocity_residual).all())

    def test_zero_elastic_coefficients_do_not_enter_proximal_consensus(self):
        """Ignore candidate deformation when a membrane has no elastic energy."""
        geometry = self._geometry((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        materials, activation, arrays = self._initialize_proximal(geometry, (0.0, 0.0, 0.0, 0.0, 0.0))
        coordinate_before = arrays["coordinate"].numpy().copy()
        velocity = wp.array(
            np.asarray(((0.0, 0.0, 0.0), (30.0, 0.0, 0.0), (0.0, -20.0, 0.0)), dtype=np.float32),
            dtype=wp.vec3,
            device=self.device,
        )
        rhs, position_residual, velocity_residual, failed = self._update_proximal(
            geometry,
            materials,
            activation,
            arrays,
            velocity=velocity,
        )

        np.testing.assert_array_equal(failed, 0)
        np.testing.assert_array_equal(rhs, 0.0)
        np.testing.assert_array_equal(position_residual, 0.0)
        np.testing.assert_array_equal(velocity_residual, 0.0)
        np.testing.assert_array_equal(arrays["coordinate"].numpy(), coordinate_before)

    def test_ordinary_membrane_one_iteration_remains_finite(self):
        """Preserve finite balanced feedback for an ordinary distorted membrane."""
        geometry = self._geometry((1.1, 0.2, 0.1), (0.1, 0.9, 0.2))
        materials, activation, arrays = self._initialize_proximal(
            geometry,
            (2.0, 3.0, 0.1, 0.0, 0.0),
            activation=0.05,
        )
        velocity = wp.array(
            np.asarray(((0.0, 0.0, 0.0), (0.2, -0.1, 0.05), (-0.1, 0.15, -0.05)), dtype=np.float32),
            dtype=wp.vec3,
            device=self.device,
        )
        rhs, position_residual, velocity_residual, failed = self._update_proximal(
            geometry,
            materials,
            activation,
            arrays,
            velocity=velocity,
        )

        np.testing.assert_array_equal(failed, 0)
        self.assertTrue(np.isfinite(rhs).all())
        self.assertTrue(np.isfinite(position_residual).all())
        self.assertTrue(np.isfinite(velocity_residual).all())
        np.testing.assert_allclose(np.sum(rhs, axis=0), 0.0, rtol=0.0, atol=2.0e-7)

    def test_stationarity_correction_prevents_false_local_convergence(self):
        """Report unresolved local KKT error even without an accepted local trial."""
        geometry = self._geometry((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        materials, activation, arrays = self._initialize_proximal(
            geometry,
            (2.0, 3.0, 0.0, 0.0, 0.0),
            activation=0.2,
        )
        arrays["multiplier"].zero_()
        _rhs, position_residual, velocity_residual, failed = self._update_proximal(
            geometry,
            materials,
            activation,
            arrays,
            iterations=0,
        )

        np.testing.assert_array_equal(failed, 0)
        self.assertGreater(float(position_residual[0]), 1.0e-3)
        np.testing.assert_array_equal(velocity_residual, 0.0)

    def test_invalid_proximal_scatter_is_transactional(self):
        """Reject all nodal writes when any proximal scatter contribution overflows."""
        geometry = self._geometry((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        geometry["poses"].assign((wp.mat22(1.0e38, 0.0, 0.0, 1.0e38),))
        materials, activation, arrays = self._initialize_proximal(geometry, (1.0, 0.0, 0.0, 0.0, 0.0))
        arrays["frozen_gradient"].fill_(wp.vec3(1.0e10))
        coordinate_before = arrays["coordinate"].numpy().copy()
        multiplier_before = arrays["multiplier"].numpy().copy()
        rhs, position_residual, velocity_residual, failed = self._update_proximal(
            geometry,
            materials,
            activation,
            arrays,
        )

        np.testing.assert_array_equal(failed, 1)
        np.testing.assert_array_equal(rhs, 0.0)
        np.testing.assert_array_equal(position_residual, 0.0)
        np.testing.assert_array_equal(velocity_residual, 0.0)
        np.testing.assert_array_equal(arrays["coordinate"].numpy(), coordinate_before)
        np.testing.assert_array_equal(arrays["multiplier"].numpy(), multiplier_before)

    def test_invalid_assembly_contribution_is_transactional(self):
        """Reject all membrane assembly writes when one tangent block overflows."""
        geometry = self._geometry((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        geometry["poses"].assign((wp.mat22(1.0e38, 0.0, 0.0, 1.0e38),))
        force, blocks, failed = self._assemble(geometry, (1.0, 0.0, 0.0, 0.0, 0.0))

        np.testing.assert_array_equal(failed, 1)
        np.testing.assert_array_equal(force, 0.0)
        np.testing.assert_array_equal(blocks, 0.0)


if __name__ == "__main__":
    unittest.main()
