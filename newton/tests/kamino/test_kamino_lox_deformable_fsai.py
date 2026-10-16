# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the adaptive FSAI preconditioner and the coarse initial guess of the LOX deformable system."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.solvers.lox.deformable.fsai import DeformableFSAI
from newton._src.solvers.kamino._src.solvers.lox.deformable.system import DeformableSystem
from newton.tests.kamino import setup_tests, test_context


def _build_cloth(device, cell_count=6):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=cell_count,
        dim_y=cell_count,
        cell_x=1.0 / cell_count,
        cell_y=1.0 / cell_count,
        mass=0.01,
        fix_left=True,
        tri_ke=1.0e5,
        tri_ka=1.0e5,
        tri_kd=1.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=10.0,
        edge_kd=0.0,
    )
    return builder.finalize(device=device)


def _build_beam(device, cell_count=24):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.add_soft_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=cell_count,
        dim_y=2,
        dim_z=2,
        cell_x=0.04,
        cell_y=0.04,
        cell_z=0.04,
        density=1000.0,
        k_mu=4.0e6,
        k_lambda=4.0e6,
        k_damp=0.0,
        fix_left=True,
        add_surface_mesh_edges=False,
        particle_radius=0.0,
    )
    builder.color()
    return builder.finalize(device=device)


def _assembled_system(model, **kwargs):
    system = DeformableSystem(model, direct_max_particles=0, proximal_iterations=0, **kwargs)
    time_step = wp.array([1.0 / 120.0], dtype=wp.float32, device=model.device)
    system.assemble(model.state(), time_step)
    return system


def _scalar_matrix(matrix):
    offsets = matrix.offsets.numpy()[: matrix.nrow + 1]
    columns = matrix.columns.numpy()
    values = matrix.values.numpy().astype(np.float64)
    dense = np.zeros((3 * matrix.nrow, 3 * matrix.ncol))
    for row in range(matrix.nrow):
        for slot in range(offsets[row], offsets[row + 1]):
            column = columns[slot]
            dense[3 * row : 3 * row + 3, 3 * column : 3 * column + 3] = values[slot]
    return dense


class TestLOXDeformableFSAI(unittest.TestCase):
    """Verify the FSAI preconditioner application, its captured refits, and the coarse correction of the warm start."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def test_apply_is_identity_outside_active_rows(self):
        """Write alpha G^T G x + beta y on CR rows of active worlds and alpha x + beta y elsewhere."""
        with wp.ScopedDevice(self.default_device):
            system = _assembled_system(_build_cloth(self.default_device), preconditioner="fsai")
            data = system.data
            fsai = system.preconditioner
            rng = np.random.default_rng(3)
            x_values = rng.standard_normal((fsai.row_count, 3)).astype(np.float32)
            y_values = rng.standard_normal((fsai.row_count, 3)).astype(np.float32)
            x = wp.array(x_values, dtype=wp.vec3)
            y = wp.array(y_values, dtype=wp.vec3)
            z = wp.zeros_like(x)
            product = wp.zeros_like(x)
            fsai.fsai.matvec(x, product, product, 1.0, 0.0)
            fsai.linear_operator.matvec(x, y, z, 0.5, -2.0)
            active = z.numpy()
            row_active = (np.arange(fsai.row_count) % 2).astype(np.int32)
            masked = DeformableFSAI(
                data.preconditioner_matrix,
                system.topology.packed_solve_world,
                data.world_active,
                system.topology.component_dof_offsets,
                row_active=wp.array(row_active, dtype=wp.int32),
            )
            masked.factorize()
            masked.linear_operator.matvec(x, y, z, 0.5, -2.0)
            rows = z.numpy()
            data.world_active.fill_(False)
            fsai.linear_operator.matvec(x, y, z, 0.5, -2.0)
            inactive = z.numpy()

        expected = 0.5 * product.numpy() - 2.0 * y_values
        identity = 0.5 * x_values - 2.0 * y_values
        self.assertGreater(np.abs(expected - identity).max(), 1.0e-2)
        np.testing.assert_allclose(active, expected, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(rows[row_active != 0], expected[row_active != 0], rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(rows[row_active == 0], identity[row_active == 0], rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(inactive, identity, rtol=1.0e-6, atol=1.0e-6)

    def test_capture_selects_support_once(self):
        """Select the supports inside the first captured factorization and refit them at later replays."""
        device = self.default_device
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        with wp.ScopedDevice(device):
            system = _assembled_system(_build_cloth(device), preconditioner="block_jacobi")
            data = system.data
            arguments = (
                data.preconditioner_matrix,
                system.topology.packed_solve_world,
                data.world_active,
                system.topology.component_dof_offsets,
            )
            x = wp.array(np.random.default_rng(4).standard_normal((data.preconditioner_matrix.nrow, 3)), dtype=wp.vec3)
            eager = DeformableFSAI(*arguments, row_active=data.packed_iterative)
            eager.factorize()
            eager_product = wp.zeros_like(x)
            eager.linear_operator.matvec(x, eager_product, eager_product, 1.0, 0.0)
            captured = DeformableFSAI(*arguments, row_active=data.packed_iterative)
            captured_product = wp.zeros_like(x)
            with wp.ScopedCapture() as capture:
                captured.factorize()
                captured.linear_operator.matvec(x, captured_product, captured_product, 1.0, 0.0)
            self.assertEqual(int(captured.pattern_ready.numpy()[0]), 0)
            wp.capture_launch(capture.graph)
            first = captured_product.numpy()
            # A later replay refits the same supports to the current values.
            values = data.preconditioner_matrix.values
            values.assign(2.0 * values.numpy())
            wp.capture_launch(capture.graph)
            scaled = captured_product.numpy()

        self.assertEqual(int(captured.pattern_ready.numpy()[0]), 1)
        np.testing.assert_array_equal(captured.update_status.numpy(), 0)
        np.testing.assert_allclose(first, eager_product.numpy(), rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(scaled, 0.5 * first, rtol=1.0e-5, atol=1.0e-6)

    def test_candidate_converges_faster_than_block_jacobi(self):
        """Reduce the candidate residual of a stiff beam more than block Jacobi in the same iterations."""
        residuals = {}
        with wp.ScopedDevice(self.default_device):
            model = _build_beam(self.default_device)
            for preconditioner in ("block_jacobi", "fsai"):
                system = _assembled_system(model, preconditioner=preconditioner, cr_iterations=4)
                data = system.data
                center = wp.zeros(model.particle_count, dtype=wp.vec3)
                system.solve_candidate(center)
                product = wp.empty_like(data.smooth_velocity)
                system.system_operator.matvec(data.smooth_velocity, product, product, 1.0, 0.0)
                iterative = data.packed_iterative.numpy() != 0
                residual = (data.candidate_rhs.numpy() - product.numpy())[iterative]
                residuals[preconditioner] = float(np.linalg.norm(residual) / np.linalg.norm(data.candidate_rhs.numpy()))

        self.assertTrue(np.isfinite(residuals["fsai"]))
        self.assertLess(residuals["fsai"], 0.7 * residuals["block_jacobi"])

    def test_coarse_initial_guess_projects_energy_error(self):
        """Remove group translations of the warm-start error without increasing its energy norm."""
        with wp.ScopedDevice(self.default_device):
            model = _build_beam(self.default_device, cell_count=96)
            system = _assembled_system(model, preconditioner="fsai", coarse_initial_guess=True)
            data = system.data
            matrix = _scalar_matrix(data.system_matrix)
            iterative = np.repeat(data.packed_iterative.numpy() != 0, 3)
            local = matrix[np.ix_(iterative, iterative)]
            rhs = data.smooth_rhs.numpy().reshape(-1)
            exact = np.linalg.solve(local, rhs[iterative])
            rng = np.random.default_rng(5)
            offsets = {
                "translation": np.tile(np.asarray([0.3, -0.2, 0.5]), int(np.sum(iterative)) // 3),
                "random": rng.standard_normal(int(np.sum(iterative))),
            }
            data.candidate_rhs.assign(data.smooth_rhs)
            residual = wp.zeros_like(data.smooth_velocity)
            corrected = {}
            for name, offset in offsets.items():
                start = rhs.copy()
                start[iterative] = exact + offset
                data.smooth_velocity.assign(start.reshape(-1, 3))
                system.system_operator.matvec(data.smooth_velocity, data.candidate_rhs, residual, -1.0, 1.0)
                system.coarse_corrector.add_global_coarse_correction(residual, data.smooth_velocity)
                corrected[name] = data.smooth_velocity.numpy().reshape(-1)[iterative] - exact

        def energy(error):
            return float(np.sqrt(error @ (local @ error)))

        self.assertLess(energy(corrected["translation"]), 1.0e-2 * energy(offsets["translation"]))
        self.assertLessEqual(energy(corrected["random"]), 1.0001 * energy(offsets["random"]))


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
