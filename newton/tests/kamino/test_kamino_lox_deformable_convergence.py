# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Convergence tests for the LOX damped deformable candidate solves."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.solvers.lox.deformable.system import DeformableSystem
from newton.tests.kamino import setup_tests, test_context


def _build_damped_beam(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.add_soft_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=24,
        dim_y=1,
        dim_z=1,
        cell_x=0.04,
        cell_y=0.04,
        cell_z=0.04,
        density=1000.0,
        k_mu=4.0e6,
        k_lambda=4.0e6,
        k_damp=9.117e4,
        fix_left=True,
        add_surface_mesh_edges=False,
        particle_radius=0.0,
    )
    builder.color()
    return builder.finalize(device=device)


def _make_system(model, *, preconditioner, recycle_cr):
    return DeformableSystem(
        model,
        cr_iterations=4,
        preconditioner=preconditioner,
        direct_max_particles=0,
        proximal_iterations=0,
        recycle_cr=recycle_cr,
    )


def _candidate_residual_history(model, *, preconditioner, recycle_cr, iterations):
    system = _make_system(model, preconditioner=preconditioner, recycle_cr=recycle_cr)
    data = system.data
    time_step = wp.array([1.0 / 480.0], dtype=wp.float32, device=model.device)
    center = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
    matrix_velocity = wp.empty_like(data.smooth_velocity)
    system.assemble(model.state(), time_step)
    inverse_scale = data.full_inverse_weight.numpy()
    residuals = []
    for _ in range(iterations):
        system.solve_candidate(center)
        system.system_operator.matvec(data.smooth_velocity, matrix_velocity, matrix_velocity, 1.0, 0.0)
        exact_residual = data.candidate_rhs.numpy() - matrix_velocity.numpy()
        residuals.append(float(np.max(np.abs(inverse_scale[:, None] * exact_residual))))
    return np.asarray(residuals)


class TestLOXDeformableConvergence(unittest.TestCase):
    """Check the convergence of the restarted conjugate-residual candidate solves."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def test_damped_candidate_recycles_cr_subspace(self):
        """Accelerate a damping-dominated restarted solve with retained CR corrections."""
        device = self.default_device
        with wp.ScopedDevice(device):
            model = _build_damped_beam(device)
            baseline = _candidate_residual_history(
                model,
                preconditioner="two_level",
                recycle_cr=False,
                iterations=8,
            )
            recycled = _candidate_residual_history(
                model,
                preconditioner="two_level",
                recycle_cr=True,
                iterations=8,
            )

        self.assertTrue(np.all(np.isfinite(baseline)))
        self.assertTrue(np.all(np.isfinite(recycled)))
        self.assertLess(recycled[-1], 0.2 * baseline[-1])

    def test_recycled_candidate_rejects_non_descent_projection(self):
        """Reject a corrupted or roundoff-damaged recycle basis that would increase the quadratic."""
        device = self.default_device
        with wp.ScopedDevice(device):
            model = _build_damped_beam(device)
            system = _make_system(model, preconditioner="two_level", recycle_cr=True)
            recycling = system.data.recycling
            time_step = wp.array([1.0 / 480.0], dtype=wp.float32, device=model.device)
            center = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
            system.assemble(model.state(), time_step)
            system.solve_candidate(center)
            system.solve_candidate(center)

            # Make the retained matrix image point opposite to its direction. This
            # models the loss of A-orthogonality that finite-precision recycling
            # must handle without exporting a harmful candidate correction.
            iterative = system.data.packed_iterative.numpy() != 0
            basis = recycling.basis.numpy()
            matrix_basis = recycling.matrix_basis.numpy()
            denominator = recycling.basis_denominator.numpy()
            basis[1, iterative] = 1.0
            matrix_basis[1, iterative] = -1.0
            denominator[1, :] = 1.0
            recycling.basis.assign(basis)
            recycling.matrix_basis.assign(matrix_basis)
            recycling.basis_denominator.assign(denominator)

            system.solve_candidate(center)

        self.assertEqual(int(recycling.projection_accepted.numpy()[0]), 0)
        self.assertTrue(np.all(np.isfinite(system.data.smooth_velocity.numpy())))
        self.assertTrue(np.all(recycling.basis_denominator.numpy()[:, 0] == 0.0))

    def test_damped_candidate_multilevel_is_finite(self):
        """Keep the component-global third level stable for strong damping."""
        device = self.default_device
        with wp.ScopedDevice(device):
            model = _build_damped_beam(device)
            residuals = _candidate_residual_history(
                model,
                preconditioner="multilevel",
                recycle_cr=True,
                iterations=4,
            )

        self.assertTrue(np.all(np.isfinite(residuals)))
        self.assertLess(residuals[-1], residuals[0])


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
