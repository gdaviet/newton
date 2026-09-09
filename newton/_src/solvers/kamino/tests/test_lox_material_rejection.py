# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check public material integration and coupled rejection."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.lox_test_utils import build_contact_model
from newton._src.solvers.kamino.tests.test_solvers_lox_deformable_integration import _make_particle_contact


class TestLOXMaterialRejection(unittest.TestCase):
    """Preserve finite material integration and prepared state on failure."""

    @classmethod
    def setUpClass(cls):
        """Initialize the device without clearing the Warp cache."""
        if not test_context.setup_done:
            setup_tests()
        cls.device = test_context.device

    def test_material_failure_preserves_coupled_rigid_warm_start(self):
        """Reject material failure without retaining coupled rigid trial state."""
        model, shapes, _ = build_contact_model(device=self.device, collider="dynamic")
        config = newton.solvers.SolverKamino.Config(dynamics_solver="lox", use_collision_detector=False)
        config.lox.max_iterations = 5
        config.lox.use_graph_conditionals = False
        solver = newton.solvers.SolverKamino(model, config=config)
        lox = solver._solver_kamino._solver_fd
        state_in, state_out = model.state(), model.state()
        velocity = state_in.particle_qd.numpy()
        velocity[0] = (0.3, -0.1, -1.0)
        state_in.particle_qd.assign(velocity)
        impulse = np.zeros_like(velocity)
        impulse[0] = (0.02, -0.01, 0.03)
        state_in.particle_lox_dual_impulse = wp.array(impulse, dtype=wp.vec3, device=self.device)
        contacts = _make_particle_contact(model, state_in, shapes[0], gap=-0.005)
        baseline = {}
        original_prepare = lox._prepare_body_space_projection

        def capture_prepared():
            original_prepare()
            for name in ("splitting_dual_impulse", "splitting_dual", "projected_twist"):
                baseline[name] = getattr(lox.splitting, name).numpy().copy()

        lox._prepare_body_space_projection = capture_prepared
        original_update = lox.deformable_system.update_proximal
        calls = 0

        def fail_material(time_step):
            nonlocal calls
            original_update(time_step)
            calls += 1
            if calls == 3:
                lox.deformable_system.proximal_failed.fill_(1)

        lox.deformable_system.update_proximal = fail_material
        solver.step(state_in, state_out, None, contacts, 0.01)

        self.assertGreaterEqual(calls, 3)
        self.assertTrue(bool(lox.world_failed.numpy()[0]))
        self.assertFalse(bool(lox.world_accepted.numpy()[0]))
        np.testing.assert_array_equal(state_out.particle_q.numpy(), state_in.particle_q.numpy())
        np.testing.assert_array_equal(state_out.particle_qd.numpy(), velocity)
        np.testing.assert_array_equal(state_out.particle_lox_dual_impulse.numpy(), impulse)
        for name, expected in baseline.items():
            np.testing.assert_allclose(getattr(lox.splitting, name).numpy(), expected, atol=1e-6, rtol=1e-6)

    def test_assembly_failure_survives_proximal_resets(self):
        """Keep an assembly rejection terminal with both enabled and disabled proxes."""
        for proximal_iterations in (0, 1):
            with self.subTest(proximal_iterations=proximal_iterations):
                model, _, _ = build_contact_model(device=self.device)
                config = newton.solvers.SolverKamino.Config(dynamics_solver="lox", use_collision_detector=False)
                config.lox.max_iterations = 3
                config.lox.deformable_proximal_iterations = proximal_iterations
                solver = newton.solvers.SolverKamino(model, config=config)
                state_in, state_out = model.state(), model.state()
                positions = state_in.particle_q.numpy()
                positions[:, :2] *= 1.0e10
                state_in.particle_q.assign(positions)
                solver.step(state_in, state_out, None, None, 0.01)
                lox = solver._solver_kamino._solver_fd

                self.assertEqual(int(lox.deformable_system.assembly_failed.numpy()[0]), 1)
                self.assertTrue(bool(lox.world_failed.numpy()[0]))
                self.assertFalse(bool(lox.world_accepted.numpy()[0]))
                np.testing.assert_array_equal(state_out.particle_q.numpy(), positions)
                np.testing.assert_array_equal(state_out.particle_qd.numpy(), state_in.particle_qd.numpy())

                # A fresh valid step must not inherit the previous failure flag.
                solver.step(model.state(), state_out, None, None, 0.01)
                self.assertEqual(int(lox.deformable_system.assembly_failed.numpy()[0]), 0)
                self.assertFalse(bool(lox.world_failed.numpy()[0]))

    def test_public_tetrahedron_preserves_inversion_tolerant_behavior(self):
        """Advance regular, singular and inverted tetrahedra without imposing a volume barrier."""
        for height in (1.0, 0.0, -0.2):
            with self.subTest(height=height):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                for position in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    builder.add_particle(pos=wp.vec3(*position), vel=wp.vec3(0.0), mass=1.0)
                builder.add_tetrahedron(0, 1, 2, 3, k_mu=10.0, k_lambda=20.0)
                model = builder.finalize(device=self.device)
                config = newton.solvers.SolverKamino.Config(dynamics_solver="lox", use_collision_detector=False)
                solver = newton.solvers.SolverKamino(model, config=config)
                state_in, state_out = model.state(), model.state()
                positions = state_in.particle_q.numpy()
                positions[3, 2] = height
                state_in.particle_q.assign(positions)
                for _ in range(5):
                    solver.step(state_in, state_out, None, None, 0.001)
                    lox = solver._solver_kamino._solver_fd
                    self.assertFalse(bool(lox.world_failed.numpy()[0]))
                    self.assertTrue(bool(lox.world_accepted.numpy()[0]))
                    self.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())
                    self.assertTrue(np.isfinite(state_out.particle_qd.numpy()).all())
                    state_in, state_out = state_out, state_in
                if height < 0.0:
                    final_positions = state_in.particle_q.numpy()
                    deformation = (final_positions[1:] - final_positions[0]).T
                    self.assertLess(float(np.linalg.det(deformation)), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
