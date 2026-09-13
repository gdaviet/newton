# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the optional LOX deformable cuDSS candidate solver."""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
import warp.sparse as wps

import newton
from newton._src.solvers.kamino._src.solvers.lox.deformable_cudss import _build_scalar_lower_structure
from newton._src.solvers.kamino._src.solvers.lox.deformable_system import DeformableFEMSystem
from newton._src.solvers.kamino._src.solvers.lox.solver import LOX_STATUS_ITERATION_LIMIT


def _build_cloth(device, cell_count=2):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=cell_count,
        dim_y=cell_count,
        cell_x=1.0 / cell_count,
        cell_y=1.0 / cell_count,
        mass=1.0,
        fix_left=True,
        tri_ke=1000.0,
        tri_ka=1000.0,
        tri_kd=1.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=10.0,
        edge_kd=0.0,
    )
    return builder.finalize(device=device)


class TestLOXDeformableCuDSS(unittest.TestCase):
    """Verify scalar expansion and the persistent cuDSS phase lifecycle."""

    def test_expand_block_matrix_lower_triangle(self):
        """Expand each lower 3-by-3 BSR block into ordered scalar CSR entries."""
        offsets, columns, block_slots, block_rows, block_columns = _build_scalar_lower_structure(
            np.asarray([0, 2, 4], dtype=np.int32),
            np.asarray([0, 1, 0, 1], dtype=np.int32),
        )

        np.testing.assert_array_equal(offsets, [0, 1, 3, 6, 10, 15, 21])
        for scalar_row in range(6):
            row_columns = columns[offsets[scalar_row] : offsets[scalar_row + 1]]
            np.testing.assert_array_equal(row_columns, np.arange(scalar_row + 1))
        self.assertEqual(columns.shape, block_slots.shape)
        self.assertEqual(columns.shape, block_rows.shape)
        self.assertEqual(columns.shape, block_columns.shape)

    def test_config_rejects_unknown_linear_solver(self):
        """Reject an unknown LOX deformable linear solver name."""
        config = newton.solvers.SolverKamino.Config(dynamics_solver="lox")
        config.lox.deformable_linear_solver = "unknown"

        with self.assertRaisesRegex(ValueError, "deformable_linear_solver"):
            config.validate()

    def test_config_requires_fixed_iterations(self):
        """Require fixed-count LOX iterations for cuDSS candidate solves."""
        config = newton.solvers.SolverKamino.Config(dynamics_solver="lox")
        config.lox.deformable_linear_solver = "cudss"

        with self.assertRaisesRegex(ValueError, "cudss.*fixed_iterations=True"):
            config.validate()

        config.lox.deformable_linear_solver = "cudss_then_cr"
        config.validate()

    @unittest.skipUnless(importlib.util.find_spec("nvmath"), "nvmath-python is not installed")
    def test_capture_factorization_and_solve(self):
        """Capture numerical factorization and candidate solves after eager analysis."""
        wp.init()
        if not wp.is_cuda_available():
            self.skipTest("CUDA is not available")
        device = wp.get_device("cuda:0")
        if not device.is_mempool_supported:
            self.skipTest("CUDA memory pools are not supported")

        with wp.ScopedDevice(device):
            model = _build_cloth(device, cell_count=32)
            system = DeformableFEMSystem(
                model,
                linear_solver="cudss",
                proximal_iterations=0,
            )
            state = model.state()
            time_step = wp.array([1.0 / 60.0], dtype=wp.float32, device=device)
            center_values = np.linspace(-0.2, 0.3, 3 * model.particle_count, dtype=np.float32).reshape((-1, 3))
            center = wp.array(center_values, dtype=wp.vec3, device=device)

            with wp.ScopedCapture() as capture:
                for _ in range(2):
                    system.assemble(state, time_step)
                    for _ in range(25):
                        system.solve_candidate(center)
            wp.capture_launch(capture.graph)

            candidate_rhs = (
                system.smooth_rhs.numpy()
                + system.nonlinear_rhs.numpy()
                + system.weight.numpy()[:, None] * center_values
            )
            wps.bsr_mv(
                system.system_matrix,
                x=system.smooth_velocity,
                y=system.matrix_velocity,
                alpha=1.0,
                beta=0.0,
            )
            residual = candidate_rhs - system.matrix_velocity.numpy()
            self.assertLess(float(np.max(np.abs(residual))), 2.0e-5)
            system.direct_solver.close()

    @unittest.skipUnless(importlib.util.find_spec("nvmath"), "nvmath-python is not installed")
    def test_fixed_iteration_solver_capture(self):
        """Run exactly the configured LOX iterations in a captured cuDSS step."""
        wp.init()
        if not wp.is_cuda_available():
            self.skipTest("CUDA is not available")
        device = wp.get_device("cuda:0")
        if not device.is_mempool_supported:
            self.skipTest("CUDA memory pools are not supported")

        with wp.ScopedDevice(device):
            model = _build_cloth(device)
            config = newton.solvers.SolverKamino.Config(
                dynamics_solver="lox",
                use_collision_detector=False,
            )
            config.lox.deformable_linear_solver = "cudss"
            config.lox.fixed_iterations = True
            config.lox.max_iterations = 3
            config.lox.deformable_proximal_iterations = 0
            solver = newton.solvers.SolverKamino(model, config=config)
            state_in = model.state()
            state_out = model.state()

            with wp.ScopedCapture() as capture:
                solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
            wp.capture_launch(capture.graph)

            lox = solver._solver_kamino._solver_fd
            np.testing.assert_array_equal(lox.iteration_count.numpy(), [3])
            np.testing.assert_array_equal(lox.world_status.numpy(), [LOX_STATUS_ITERATION_LIMIT])
            self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))
            lox.deformable_system.direct_solver.close()

    @unittest.skipUnless(importlib.util.find_spec("nvmath"), "nvmath-python is not installed")
    def test_cudss_then_conditional_cr_capture(self):
        """Capture one cuDSS candidate before a conditional CR loop."""
        wp.init()
        if not wp.is_cuda_available():
            self.skipTest("CUDA is not available")
        device = wp.get_device("cuda:0")
        if not device.is_mempool_supported or not wp.is_conditional_graph_supported():
            self.skipTest("CUDA memory pools and conditional graphs are required")
        from nvmath.bindings import cudss  # noqa: PLC0415

        phases = []
        execute = cudss.execute

        def record_phase(*args):
            phases.append(args[1])
            return execute(*args)

        with wp.ScopedDevice(device), patch.object(cudss, "execute", side_effect=record_phase):
            model = _build_cloth(device)
            config = newton.solvers.SolverKamino.Config(
                dynamics_solver="lox",
                use_collision_detector=False,
            )
            config.lox.deformable_linear_solver = "cudss_then_cr"
            config.lox.max_iterations = 3
            config.lox.deformable_cr_iterations = 2
            config.lox.deformable_direct_max_particles = 0
            config.lox.deformable_proximal_iterations = 0
            config.lox.selective_weights = False
            config.lox.position_tolerance = 1.0e-12
            config.lox.velocity_tolerance = 1.0e-12
            solver = newton.solvers.SolverKamino(model, config=config)
            state_in = model.state()
            state_out = model.state()

            with wp.ScopedCapture() as capture:
                solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
            wp.capture_launch(capture.graph)

            lox = solver._solver_kamino._solver_fd
            self.assertEqual(phases.count(cudss.Phase.ANALYSIS), 1)
            self.assertEqual(phases.count(cudss.Phase.FACTORIZATION), 1)
            self.assertEqual(phases.count(cudss.Phase.SOLVE), 1)
            np.testing.assert_array_equal(lox.iteration_count.numpy(), [3])
            self.assertEqual(int(lox.deformable_system.recycling_started.numpy()[0]), 1)
            self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))
            lox.deformable_system.cudss_solver.close()

    @unittest.skipUnless(importlib.util.find_spec("nvmath"), "nvmath-python is not installed")
    def test_reuse_analysis_and_factorization(self):
        """Analyze once, factor once per time step, and solve every candidate update."""
        wp.init()
        if not wp.is_cuda_available():
            self.skipTest("CUDA is not available")
        device = wp.get_device("cuda:0")
        from nvmath.bindings import cudss  # noqa: PLC0415

        phases = []
        execute = cudss.execute

        def record_phase(*args):
            phases.append(args[1])
            return execute(*args)

        with wp.ScopedDevice(device), patch.object(cudss, "execute", side_effect=record_phase):
            model = _build_cloth(device)
            system = DeformableFEMSystem(
                model,
                linear_solver="cudss",
                proximal_iterations=0,
            )
            time_step = wp.array([1.0 / 60.0], dtype=wp.float32, device=device)
            center_values = np.linspace(-0.2, 0.3, 3 * model.particle_count, dtype=np.float32).reshape((-1, 3))
            center = wp.array(center_values, dtype=wp.vec3, device=device)

            for _ in range(2):
                system.assemble(model.state(), time_step)
                candidate_rhs = (
                    system.smooth_rhs.numpy()
                    + system.nonlinear_rhs.numpy()
                    + system.weight.numpy()[:, None] * center_values
                )
                for _ in range(3):
                    system.solve_candidate(center)

                wps.bsr_mv(
                    system.system_matrix,
                    x=system.smooth_velocity,
                    y=system.matrix_velocity,
                    alpha=1.0,
                    beta=0.0,
                )
                residual = candidate_rhs - system.matrix_velocity.numpy()
                self.assertLess(float(np.max(np.abs(residual))), 2.0e-5)

            direct_solver = system.direct_solver
            self.assertIsNotNone(direct_solver)
            direct_solver.close()

        self.assertEqual(phases.count(cudss.Phase.ANALYSIS), 1)
        self.assertEqual(phases.count(cudss.Phase.FACTORIZATION), 2)
        self.assertEqual(phases.count(cudss.Phase.SOLVE), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
