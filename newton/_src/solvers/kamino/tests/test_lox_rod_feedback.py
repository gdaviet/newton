# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused constitutive and stability regressions for LOX rod feedback."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.core.types import vec6f
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context

_TIME_STEP = 0.01


def _principal_angle(angle: float) -> float:
    """Return an angle in the scalar reference's principal elastic chart."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _implicit_twist_reference(
    *,
    initial_angle: float,
    initial_speed: float,
    rest_angle: float,
    stiffness: float,
    damping: float,
    time_step: float,
) -> float:
    """Solve the one-DOF implicit constitutive equation by safeguarded Newton iteration."""

    def residual(speed: float) -> float:
        elastic = _principal_angle(initial_angle + time_step * speed - rest_angle)
        return speed - initial_speed + time_step * (stiffness * elastic + damping * speed)

    speed = initial_speed
    for _ in range(40):
        value = residual(speed)
        if abs(value) < 1.0e-13:
            break
        epsilon = 1.0e-6 * max(1.0, abs(speed))
        derivative = (residual(speed + epsilon) - residual(speed - epsilon)) / (2.0 * epsilon)
        if derivative <= 0.0 or not math.isfinite(derivative):
            raise RuntimeError("Scalar rod reference left its smooth principal-angle branch.")
        speed -= value / derivative
    if abs(residual(speed)) >= 1.0e-10:
        raise RuntimeError("Scalar rod reference did not converge.")
    return speed


def _build_rod_cases(cases: list[dict[str, float | str]], device: wp.DeviceLike) -> tuple[newton.Model, list[int]]:
    """Build independent unit-inertia rod worlds with prescribed rest poses."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverKamino.register_custom_attributes(builder)
    bodies = []
    inertia = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    for case in cases:
        builder.begin_world()
        axis = wp.vec3f(1.0, 0.0, 0.0) if case.get("mode", "twist") == "bend" else wp.vec3f(0.0, 0.0, 1.0)
        rest_angle = float(case.get("rest_angle", 0.0))
        body = builder.add_link(
            xform=wp.transformf(wp.vec3f(0.0), wp.quat_from_axis_angle(axis, rest_angle)),
            mass=1.0,
            inertia=inertia,
            lock_inertia=True,
        )
        joint = builder.add_joint_rod(
            -1,
            body,
            bend_stiffness=float(case.get("stiffness", 0.0)) if case.get("mode", "twist") == "bend" else 0.0,
            bend_damping=float(case.get("damping", 0.0)) if case.get("mode", "twist") == "bend" else 0.0,
            twist_stiffness=(float(case.get("stiffness", 0.0)) if case.get("mode", "twist") == "twist" else 0.0),
            twist_damping=float(case.get("damping", 0.0)) if case.get("mode", "twist") == "twist" else 0.0,
        )
        builder.add_articulation([joint])
        builder.end_world()
        bodies.append(body)
    return builder.finalize(device=device), bodies


def _run_rod_cases(
    cases: list[dict[str, float | str]],
    device: wp.DeviceLike,
    *,
    relaxation: float,
    iterations: int = 80,
) -> tuple[np.ndarray, object]:
    """Run one fixed-budget LOX step and return body velocities and its private solver."""
    model, bodies = _build_rod_cases(cases, device)
    state_previous = model.state()
    state_next = model.state()
    poses = state_previous.body_q.numpy()
    velocities = np.zeros((model.body_count, 6), dtype=np.float32)
    for body, case in zip(bodies, cases, strict=True):
        is_bend = case.get("mode", "twist") == "bend"
        axis = wp.vec3f(1.0, 0.0, 0.0) if is_bend else wp.vec3f(0.0, 0.0, 1.0)
        angle = float(case.get("rest_angle", 0.0)) + float(case.get("initial_strain", 0.0))
        poses[body, 3:] = np.asarray(wp.quat_from_axis_angle(axis, angle))
        velocities[body, 3 if is_bend else 5] = float(case.get("initial_speed", 0.0))
    state_previous.body_q.assign(poses)
    state_previous.body_qd.assign(velocities)

    config = SolverKamino.Config(dynamics_solver="lox", sparse_jacobian=True, use_collision_detector=False)
    config.lox.max_iterations = iterations
    config.lox.fixed_iterations = True
    config.lox.use_graph_conditionals = False
    config.lox.rod_proximal_relaxation = relaxation
    solver = SolverKamino(model, config=config)
    solver.step(state_previous, state_next, model.control(), contacts=None, dt=_TIME_STEP)
    return state_next.body_qd.numpy(), solver._solver_kamino._solver_fd


class TestLOXRodFeedback(unittest.TestCase):
    """Verify continuous twist damping without preemptive geometry rejection."""

    @classmethod
    def setUpClass(cls):
        """Initialize Warp without clearing its kernel cache."""
        if not test_context.setup_done:
            setup_tests()
        cls.device = test_context.device

    def test_twist_damping_crosses_both_principal_branches(self):
        """Damp both branch-crossing directions without increasing kinetic energy."""
        cases = [
            {"initial_strain": 3.1, "initial_speed": 10.0, "damping": 100.0},
            {"initial_strain": -3.1, "initial_speed": -10.0, "damping": 100.0},
            {"initial_strain": 0.5, "initial_speed": 10.0, "damping": 100.0},
        ]
        velocity, _ = _run_rod_cases(cases, self.device, relaxation=1.0)

        np.testing.assert_allclose(velocity[:, 5], [5.0, -5.0, 5.0], rtol=0.0, atol=1.0e-4)
        self.assertTrue(np.all(velocity[:, 5] * velocity[:, 5] <= np.asarray([100.0, 100.0, 100.0]) + 1.0e-5))

    def test_twist_feedback_preserves_rest_angle_and_relaxation_fixed_point(self):
        """Preserve temporal damping across a nonzero rest twist for positive relaxations."""
        cases = [{"rest_angle": 0.7, "initial_strain": 3.1, "initial_speed": 10.0, "damping": 100.0}]
        for relaxation in (0.25, 0.5, 1.0):
            with self.subTest(relaxation=relaxation):
                velocity, _ = _run_rod_cases(cases, self.device, relaxation=relaxation)
                self.assertAlmostEqual(float(velocity[0, 5]), 5.0, delta=1.0e-4)

    def test_mixed_twist_matches_independent_implicit_reference(self):
        """Match mixed elastic-damping solves across unambiguous temporal branch crossings."""
        cases = [
            {
                "rest_angle": -0.35,
                "initial_strain": strain,
                "initial_speed": speed,
                "stiffness": 37.0,
                "damping": 13.0,
            }
            for strain, speed in ((0.8, 7.0), (3.1, 7.0), (-3.1, -7.0))
        ]
        velocity, lox = _run_rod_cases(cases, self.device, relaxation=1.0)
        expected = [
            _implicit_twist_reference(
                initial_angle=float(case["rest_angle"]) + float(case["initial_strain"]),
                initial_speed=float(case["initial_speed"]),
                rest_angle=float(case["rest_angle"]),
                stiffness=float(case["stiffness"]),
                damping=float(case["damping"]),
                time_step=_TIME_STEP,
            )
            for case in cases
        ]
        np.testing.assert_allclose(velocity[:, 5], expected, rtol=0.0, atol=2.0e-4)
        np.testing.assert_array_equal(lox.world_failed.numpy(), [False, False, False])

    def test_twist_proximal_keeps_nonzero_linearization_rate(self):
        """Retain the temporal reference rate when the linearization velocity is nonzero."""
        rest_angle = -0.2
        frozen_angle = 0.4
        linearization_speed = 3.0
        candidate_speed = 7.0
        stiffness = 37.0
        damping = 13.0
        model, _ = _build_rod_cases(
            [{"rest_angle": rest_angle, "stiffness": stiffness, "damping": damping}],
            self.device,
        )
        config = SolverKamino.Config(dynamics_solver="lox", sparse_jacobian=True, use_collision_detector=False)
        config.lox.rod_proximal_relaxation = 1.0
        solver = SolverKamino(model, config=config)
        lox = solver._solver_kamino._solver_fd
        adapter = lox.rigid_adapter
        adapter.data.bodies.q_i.assign(
            [
                wp.transformf(
                    wp.vec3f(0.0),
                    wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), frozen_angle),
                )
            ]
        )
        adapter.data.bodies.u_i.assign([wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, linearization_speed)])
        time_step = wp.array([_TIME_STEP], dtype=wp.float32, device=self.device)
        inverse_time_step = wp.array([1.0 / _TIME_STEP], dtype=wp.float32, device=self.device)
        linearization = wp.array(
            [vec6f(0.0, 0.0, 0.0, 0.0, 0.0, linearization_speed)],
            dtype=vec6f,
            device=self.device,
        )
        candidate = wp.array(
            [vec6f(0.0, 0.0, 0.0, 0.0, 0.0, candidate_speed)],
            dtype=vec6f,
            device=self.device,
        )
        world_mask = wp.ones(1, dtype=wp.bool, device=self.device)
        adapter.begin_time_step(time_step, inverse_time_step)
        adapter.update(time_step, linearization_twist=linearization)

        for _ in range(60):
            adapter.rods.update_proximal(
                adapter.system,
                candidate,
                linearization,
                world_mask,
                time_step,
                1.0e-5,
                1.0e-5,
                1.0e-5,
            )

        candidate_angle = frozen_angle + _TIME_STEP * (candidate_speed - linearization_speed)
        expected_stress = stiffness * _principal_angle(candidate_angle - rest_angle) + damping * candidate_speed
        self.assertAlmostEqual(float(adapter.rods.multiplier.numpy()[5]), expected_stress, delta=2.0e-4)
        np.testing.assert_array_equal(adapter.rods.world_proximal_failed.numpy(), [0])

    def test_zero_twist_coefficients_leave_velocity_unchanged(self):
        """Leave twist velocity unchanged when elastic and damping coefficients vanish."""
        case = {"initial_strain": 3.1, "initial_speed": 10.0}
        baseline, _ = _run_rod_cases([case], self.device, relaxation=0.0)
        feedback, _ = _run_rod_cases([case], self.device, relaxation=1.0)
        np.testing.assert_allclose(feedback, baseline, rtol=0.0, atol=1.0e-7)

    def test_bend_feedback_allows_common_rigid_rotation(self):
        """Allow a large common rotation that leaves binary rod geometry unchanged."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverKamino.register_custom_attributes(builder)
        builder.begin_world()
        inertia = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        parent = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
        child = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
        joint = builder.add_joint_rod(parent, child, bend_damping=100.0, twist_damping=100.0)
        builder.add_articulation([joint])
        builder.end_world()
        model = builder.finalize(device=self.device)
        config = SolverKamino.Config(dynamics_solver="lox", sparse_jacobian=True, use_collision_detector=False)
        config.lox.rod_proximal_relaxation = 1.0
        solver = SolverKamino(model, config=config)
        lox = solver._solver_kamino._solver_fd
        adapter = lox.rigid_adapter
        time_step = wp.array([_TIME_STEP], dtype=wp.float32, device=self.device)
        inverse_time_step = wp.array([1.0 / _TIME_STEP], dtype=wp.float32, device=self.device)
        linearization = wp.zeros(2, dtype=vec6f, device=self.device)
        common_velocity = vec6f(0.0, 0.0, 0.0, 400.0, 0.0, 0.0)
        candidate = wp.array([common_velocity, common_velocity], dtype=vec6f, device=self.device)
        world_mask = wp.ones(1, dtype=wp.bool, device=self.device)
        adapter.begin_time_step(time_step, inverse_time_step)
        adapter.update(time_step, linearization_twist=linearization)

        adapter.rods.update_proximal(
            adapter.system,
            candidate,
            linearization,
            world_mask,
            time_step,
            1.0e-5,
            1.0e-5,
            1.0e-5,
        )

        np.testing.assert_array_equal(adapter.rods.world_proximal_failed.numpy(), [0])

    def test_finite_large_rod_updates_are_not_preemptively_rejected(self):
        """Run finite near-fold and large angular updates without predicting failure."""
        cases = [
            {"mode": "bend", "initial_strain": 3.1, "initial_speed": 10.0, "damping": 10000.0},
            {"mode": "bend", "initial_strain": 0.5, "initial_speed": 600.0, "damping": 100.0},
            {"mode": "twist", "initial_strain": 0.5, "initial_speed": 600.0, "damping": 1.0},
        ]
        feedback, lox = _run_rod_cases(cases, self.device, relaxation=1.0, iterations=1)

        self.assertTrue(np.isfinite(feedback).all())
        self.assertTrue(np.isfinite(lox.rigid_adapter.rods.multiplier.numpy()).all())
        np.testing.assert_array_equal(lox.world_failed.numpy(), [False, False, False])
        np.testing.assert_array_equal(lox.world_accepted.numpy(), [True, True, True])
        np.testing.assert_array_equal(lox.iteration_count.numpy(), [1, 1, 1])

    def test_cuda_graph_replays_twist_branch_crossing(self):
        """Replay continuous twist feedback under CUDA graph capture."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        cases = [{"initial_strain": 3.1, "initial_speed": 10.0, "damping": 100.0}]
        model, bodies = _build_rod_cases(cases, self.device)
        state_previous = model.state()
        state_next = model.state()
        poses = state_previous.body_q.numpy()
        poses[bodies[0], 3:] = np.asarray(
            wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), float(cases[0]["initial_strain"]))
        )
        state_previous.body_q.assign(poses)
        state_previous.body_qd.assign([wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 10.0)])
        config = SolverKamino.Config(dynamics_solver="lox", sparse_jacobian=True, use_collision_detector=False)
        config.lox.max_iterations = 80
        config.lox.fixed_iterations = True
        config.lox.use_graph_conditionals = False
        config.lox.rod_proximal_relaxation = 1.0
        solver = SolverKamino(model, config=config)

        with wp.ScopedCapture() as capture:
            solver.step(state_previous, state_next, model.control(), contacts=None, dt=_TIME_STEP)

        wp.capture_launch(capture.graph)
        self.assertAlmostEqual(float(state_next.body_qd.numpy()[0, 5]), 5.0, delta=1.0e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
