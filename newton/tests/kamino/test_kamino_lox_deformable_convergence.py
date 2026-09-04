# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused convergence tests for LOX damped deformable candidate solves."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.solvers.lox.deformable_system import DeformableFEMSystem
from newton.tests.unittest_utils import add_function_test, get_test_devices


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


def _candidate_residual_history(model, *, preconditioner, recycle_cr, iterations):
    system = DeformableFEMSystem(
        model,
        cr_iterations=4,
        preconditioner=preconditioner,
        direct_max_particles=0,
        proximal_iterations=0,
        recycle_cr=recycle_cr,
    )
    time_step = wp.array([1.0 / 480.0], dtype=wp.float32, device=model.device)
    center = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
    matrix_velocity = wp.empty_like(system.smooth_velocity)
    system.assemble(model.state(), time_step)
    inverse_scale = system.full_inverse_weight.numpy()
    residuals = []
    for _ in range(iterations):
        system.solve_candidate(center)
        system._system_matvec(system.smooth_velocity, matrix_velocity, matrix_velocity, 1.0, 0.0)
        exact_residual = system.candidate_rhs.numpy() - matrix_velocity.numpy()
        residuals.append(float(np.max(np.abs(inverse_scale[:, None] * exact_residual))))
    return np.asarray(residuals)


def test_damped_candidate_recycles_cr_subspace(test, device):
    """Retained CR corrections accelerate a damping-dominated restarted solve."""
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

    test.assertTrue(np.all(np.isfinite(baseline)))
    test.assertTrue(np.all(np.isfinite(recycled)))
    test.assertLess(recycled[-1], 0.2 * baseline[-1])


def test_damped_candidate_multilevel_is_finite(test, device):
    """The component-global third level remains stable for strong damping."""
    with wp.ScopedDevice(device):
        model = _build_damped_beam(device)
        residuals = _candidate_residual_history(
            model,
            preconditioner="multilevel",
            recycle_cr=True,
            iterations=4,
        )

    test.assertTrue(np.all(np.isfinite(residuals)))
    test.assertLess(residuals[-1], residuals[0])


class TestLOXDeformableConvergence(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(
    TestLOXDeformableConvergence,
    "test_damped_candidate_recycles_cr_subspace",
    test_damped_candidate_recycles_cr_subspace,
    devices=devices,
)
add_function_test(
    TestLOXDeformableConvergence,
    "test_damped_candidate_multilevel_is_finite",
    test_damped_candidate_multilevel_is_finite,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
