# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify safe output after a nonlinear update fails in a coupled world."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.test_solvers_lox_deformable_integration import _make_particle_contact


class TestLOXCoupledRejection(unittest.TestCase):
    """Check both rigid and particle output/warm starts after rejection."""

    @classmethod
    def setUpClass(cls):
        """Initialize the test device while preserving the Warp kernel cache."""
        if not test_context.setup_done:
            setup_tests()
        cls.device = test_context.device

    def test_nonfinite_rod_rejects_coupled_trial_without_particle_warmstart_leak(self):
        """Reject an actual nonfinite rod update without leaking particle trial state."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverKamino.register_custom_attributes(builder)
        builder.begin_world()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        shape = builder.add_shape_sphere(body, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        rod = builder.add_joint_rod(-1, body, bend_stiffness=0.0, bend_damping=10000.0)
        builder.add_articulation([rod])
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.1),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, -0.1),
            dim_x=1,
            dim_y=1,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            tri_ke=100.0,
            tri_ka=80.0,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
            edge_ke=2.0,
            edge_kd=0.0,
            particle_radius=0.0,
        )
        builder.end_world()
        model = builder.finalize(device=self.device)
        config = SolverKamino.Config(dynamics_solver="lox", sparse_jacobian=True, use_collision_detector=False)
        config.lox.rod_proximal_relaxation = 1.0
        config.lox.use_graph_conditionals = False
        config.lox.max_iterations = 5
        solver = SolverKamino(model, config=config)
        state_in, state_out = model.state(), model.state()
        poses = state_in.body_q.numpy()
        poses[body, 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5))
        state_in.body_q.assign(poses)
        state_in.body_qd.assign([[0.0, 0.0, 0.0, 10.0, 0.0, 0.0]])
        # Only the contacted node has an active consensus warm start.
        impulse = np.zeros((model.particle_count, 3), dtype=np.float32)
        impulse[0] = (0.02, -0.01, 0.03)
        state_in.particle_lox_dual_impulse = wp.array(impulse, dtype=wp.vec3, device=self.device)
        contacts = _make_particle_contact(model, state_in, shape, gap=-0.005)
        initial_particles = state_in.particle_q.numpy()
        initial_velocities = state_in.particle_qd.numpy()
        lox = solver._solver_kamino._solver_fd
        rods = lox.rigid_adapter.rods
        original_update = rods.update_proximal
        calls = 0

        def inject_nonfinite_multiplier(*args):
            nonlocal calls
            calls += 1
            if calls == 3:
                self.assertFalse(bool(lox.world_failed.numpy()[0]))
                multiplier = rods.multiplier.numpy()
                multiplier[3] = np.nan
                rods.multiplier.assign(multiplier)
            original_update(*args)

        rods.update_proximal = inject_nonfinite_multiplier
        solver.step(state_in, state_out, model.control(), contacts, 0.01)

        self.assertFalse(bool(lox.world_accepted.numpy()[0]))
        self.assertTrue(bool(lox.world_failed.numpy()[0]))
        np.testing.assert_array_equal(lox.iteration_count.numpy(), [3])
        np.testing.assert_array_equal(state_out.particle_q.numpy(), initial_particles)
        np.testing.assert_array_equal(state_out.particle_qd.numpy(), initial_velocities)
        np.testing.assert_array_equal(state_out.particle_lox_dual_impulse.numpy(), impulse)
        np.testing.assert_allclose(state_out.body_qd.numpy(), state_in.body_qd.numpy(), atol=1e-5)
        self.assertTrue(np.isfinite(state_out.body_q.numpy()).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
