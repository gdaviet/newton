# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check global-frame covariance of corrected structural feedback."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.test_solver_kamino_lox import (
    _build_two_body_joint_model,
    _quaternion_product,
    _rotate_vector,
)


class TestLOXJointCovariance(unittest.TestCase):
    """Verify geometry and reactions independently of world-coordinate choice."""

    @classmethod
    def setUpClass(cls):
        """Initialize Warp without clearing the kernel cache."""
        if not test_context.setup_done:
            setup_tests()
        cls.device = test_context.device

    def test_joint_feedback_rotates_poses_velocities_and_wrenches(self):
        """Rotate the whole scene without changing its relative physical solution."""
        rotation = np.asarray(wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, -2.0, 0.7)), 1.1))
        for kind in ("fixed", "revolute"):
            with self.subTest(kind=kind):
                results = []
                for rotated in (False, True):
                    model, parent, child, _, _ = _build_two_body_joint_model(
                        joint_kind=kind,
                        mass_ratio=100.0,
                        reverse_body_order=True,
                        device=self.device,
                    )
                    velocity = np.zeros((model.body_count, 6), dtype=np.float32)
                    velocity[parent] = (2.0, -1.0, 0.5, 0.0, 50.0, 0.0)
                    velocity[child] = (-3.0, 2.0, -1.0, 20.0, -10.0, 15.0)
                    if rotated:
                        poses = model.body_q.numpy()
                        for pose in poses:
                            pose[:3] = _rotate_vector(rotation, pose[:3])
                            pose[3:] = _quaternion_product(rotation, pose[3:])
                        model.body_q.assign(poses)
                        root_frames = model.joint_X_p.numpy()
                        for joint in np.flatnonzero(model.joint_parent.numpy() < 0):
                            root_frames[joint, :3] = _rotate_vector(rotation, root_frames[joint, :3])
                            root_frames[joint, 3:] = _quaternion_product(rotation, root_frames[joint, 3:])
                        model.joint_X_p.assign(root_frames)
                        for body in range(model.body_count):
                            velocity[body, :3] = _rotate_vector(rotation, velocity[body, :3])
                            velocity[body, 3:] = _rotate_vector(rotation, velocity[body, 3:])
                    config = SolverKamino.Config(
                        dynamics_solver="lox", sparse_jacobian=True, use_collision_detector=False
                    )
                    config.lox.joint_proximal_relaxation = 1.0
                    config.lox.fixed_iterations = True
                    solver = SolverKamino(model, config=config)
                    previous, current = model.state(), model.state()
                    previous.body_qd.assign(velocity)
                    solver.step(previous, current, model.control(), None, 0.01)
                    lox = solver._solver_kamino._solver_fd
                    results.append(
                        (current.body_q.numpy(), current.body_qd.numpy(), lox.rigid_adapter.data.bodies.w_j_i.numpy())
                    )
                original, transformed = results
                for body in range(model.body_count):
                    np.testing.assert_allclose(
                        transformed[0][body, :3], _rotate_vector(rotation, original[0][body, :3]), atol=2e-5
                    )
                    expected_q = _quaternion_product(rotation, original[0][body, 3:])
                    actual_q = transformed[0][body, 3:]
                    if np.dot(expected_q, actual_q) < 0:
                        actual_q = -actual_q
                    np.testing.assert_allclose(actual_q, expected_q, atol=2e-5)
                    for quantity in (1, 2):
                        for begin in (0, 3):
                            np.testing.assert_allclose(
                                transformed[quantity][body, begin : begin + 3],
                                _rotate_vector(rotation, original[quantity][body, begin : begin + 3]),
                                rtol=5e-4,
                                atol=2e-3,
                            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
