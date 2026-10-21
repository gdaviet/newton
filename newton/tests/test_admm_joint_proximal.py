# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the local ADMM joint quadratic against dense reference equations."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherstone, SolverMuJoCo, SolverVBD, SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM


def _cross(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


class TestAdmmJointProximal(unittest.TestCase):
    def _make_solver(
        self,
        *,
        kind="ball",
        anchor=(0, 0, 0),
        gravity=(0, 0, 0),
        device="cpu",
        factory=None,
        free_bodies=True,
        **config,
    ):
        builder = newton.ModelBuilder(gravity=gravity)
        for mass in (2.0, 5.0):
            builder.add_body(mass=mass, inertia=wp.mat33(np.diag([1.0, 2.0, 3.0])))
        kwargs = {"parent": 0, "child": 1, "parent_xform": wp.transform(anchor, wp.quat_identity())}
        if kind == "ball":
            builder.add_joint_ball(**kwargs)
        elif kind == "fixed":
            builder.add_joint_fixed(**kwargs)
        else:
            builder.add_joint_revolute(**kwargs, axis=wp.vec3(1, 0, 0))
        builder.color()
        model = builder.finalize(device=device)
        solver = SolverCoupledADMM(
            model,
            [
                SolverCoupled.Entry(
                    name=str(body),
                    bodies=[body],
                    joints=[body] if free_bodies else [],
                    solver=factory or (lambda view: SolverMuJoCo(view, use_mujoco_cpu=True, integrator="euler")),
                )
                for body in range(2)
            ],
            SolverCoupledADMM.Config(**config),
        )
        return model, solver

    def test_iterations_match_block_jacobi_equations(self):
        """Match rho and optional gamma updates to an independent scalar recurrence."""
        for rho in (0.7, 20.0):
            for gamma in (0.0, 0.2):
                for dt in (0.01, 0.07):
                    for iterations in (1, 4):
                        with self.subTest(rho=rho, gamma=gamma, dt=dt, iterations=iterations):
                            model, solver = self._make_solver(
                                rho=rho,
                                gamma=gamma,
                                iterations=iterations,
                                joint_stiffness=100.0,
                                gravity=(3.0, 0.0, 0.0),
                            )
                            state = model.state()
                            out = model.state()
                            v_n = np.array([1.3, -0.7])
                            qd = np.zeros((2, 6), dtype=np.float32)
                            qd[:, 0] = v_n
                            state.body_qd.assign(qd)
                            joint_qd = state.joint_qd.numpy()
                            joint_qd[:12] = qd.ravel()
                            state.joint_qd.assign(joint_qd)
                            force = np.array([2.0, -3.0])
                            f = np.zeros((2, 6), dtype=np.float32)
                            f[:, 0] = force
                            state.body_f.assign(f)
                            masses = model.body_mass.numpy().astype(float)
                            weight2 = 1.0 / np.sum(1.0 / masses)
                            weight = np.sqrt(weight2)
                            P = (dt + gamma) * rho * weight2
                            v = v_n.copy()
                            u = dual = 0.0
                            for _ in range(iterations):
                                coupling_force = weight * (dual + rho * weight * (u - (v[1] - v[0])))
                                v = (
                                    masses * v_n
                                    + P * v
                                    + dt * (force + 3.0 * masses + np.array([-coupling_force, coupling_force]))
                                ) / (masses + P)
                                relative = v[1] - v[0]
                                u = (rho * weight2 * relative - weight * dual) / (100.0 + rho * weight2)
                                dual += rho * weight * (u - relative)
                            solver.step(state, out, None, None, dt)
                            np.testing.assert_allclose(out.body_qd.numpy()[:, 0], v, rtol=2e-6, atol=2e-6)

    def test_reduced_coordinate_proxy_without_backend_identity(self):
        """Apply numerical inertia through a second reduced-coordinate solver."""
        dt, rho, gamma = 0.02, 20.0, 0.1
        model, solver = self._make_solver(factory=SolverFeatherstone, rho=rho, gamma=gamma, iterations=1)
        state, out = model.state(), model.state()
        force = np.array([2.0, -3.0])
        body_f = np.zeros((2, 6), dtype=np.float32)
        body_f[:, 0] = force
        state.body_f.assign(body_f)
        mass = model.body_mass.numpy()
        weight2 = 1.0 / np.sum(1.0 / mass)
        numerical_mass = (dt + gamma) * rho * weight2
        solver.step(state, out, None, None, dt)
        np.testing.assert_allclose(
            out.body_qd.numpy()[:, 0], dt * force / (mass + numerical_mass), rtol=2e-6, atol=2e-6
        )

    def test_spatial_inertia_matches_joint_jacobian(self):
        """Retain off-center coupling and the free revolute angular direction exactly."""
        anchor = np.array([0.3, -0.4, 0.2])
        dt, rho = 0.03, 4.0
        for kind in ("ball", "fixed", "revolute"):
            with self.subTest(kind=kind):
                model, solver = self._make_solver(kind=kind, anchor=anchor, rho=rho)
                solver._refresh_proxy_inertia(dt)
                group = solver._admm_rr_groups[0]
                W2 = float(group.W.numpy()[0]) ** 2
                J = np.hstack((np.eye(3), -_cross(anchor)))
                P = dt * rho * W2 * J.T @ J
                if kind != "ball":
                    groups = (
                        solver._admm_rr_angular_groups if kind == "fixed" else solver._admm_rr_revolute_angular_groups
                    )
                    angular_W2 = float(groups[0].W.numpy()[0]) ** 2
                    P[3:, 3:] += dt * rho * angular_W2 * np.diag([kind == "fixed", 1, 1])
                view = solver.view("0")
                body = solver._body_local_id("0", 0)
                buf = solver._admm_buffers["0"]
                mass = float(buf.proxy_mass.numpy()[body])
                moment = _cross(buf.proxy_moment.numpy()[body])
                tensor = buf.proxy_inertia.numpy()[body]
                actual = np.block([[mass * np.eye(3), -moment], [moment, tensor]])
                np.testing.assert_allclose(actual, P, rtol=2e-6, atol=2e-6)
                # Repeated refreshes must not accumulate numerical mass.
                solver._refresh_proxy_inertia(dt)
                self.assertAlmostEqual(float(buf.proxy_mass.numpy()[body]), mass, places=6)
                np.testing.assert_array_equal(view.body_com.numpy()[body], model.body_com.numpy()[0])
                np.testing.assert_array_equal(view.body_inertia.numpy()[body], model.body_inertia.numpy()[0])
                self.assertEqual(float(view.body_mass.numpy()[body]), float(model.body_mass.numpy()[0]))

    def test_off_center_wrench_matches_dense_velocity_solve(self):
        """Match the full six-dimensional response, in physical COM coordinates."""
        anchor = np.array([0.3, -0.4, 0.2])
        dt, rho = 0.02, 7.0
        for gamma in (0.0, 0.2):
            model, solver = self._make_solver(anchor=anchor, iterations=1, rho=rho, gamma=gamma)
            state, out = model.state(), model.state()
            forces = np.array([[1.0, 3.0, 2.0, 0.5, -0.2, 0.4], [-2.0, 1.0, 4.0, -0.4, 0.3, 0.2]])
            state.body_f.assign(forces.astype(np.float32))
            solver.step(state, out, None, None, dt)
            W2 = 1.0 / (1.0 / 2.0 + 1.0 / 5.0)
            for body, r in ((0, anchor), (1, np.zeros(3))):
                mass = float(model.body_mass.numpy()[body])
                M = np.diag([mass] * 3 + [1.0, 2.0, 3.0])
                J = np.hstack((np.eye(3), -_cross(r)))
                G = np.diag([1.0] * 3 + [2.0 / 3.0 * np.dot(r, r)] * 3)
                P = rho * W2 * (dt * J.T @ J + gamma * G)
                expected = np.linalg.solve(M + P, dt * forces[body])
                actual = out.body_qd.numpy()[body].astype(float)
                np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-7)

    def test_numerical_inertia_is_present_on_both_endpoints(self):
        """Give every coupled endpoint rho inertia without changing physical inertia."""
        model, solver = self._make_solver(rho=5.0, gamma=0.0)
        solver.step(model.state(), model.state(), None, None, 0.02)
        self.assertEqual(set(solver._admm_proxy_bodies), {"0", "1"})
        for body in range(2):
            view = solver.view(str(body))
            self.assertEqual(view.body_count, 1)
            self.assertEqual(view.joint_count, 1)
            self.assertAlmostEqual(float(view.body_mass.numpy()[0]), float(model.body_mass.numpy()[body]), places=6)
            self.assertIsNotNone(solver._admm_buffers[str(body)].proxy_mass)

    def test_proxy_selection_matches_scalar_iterations(self):
        """Select rho inertia per entry while retaining gamma and one native advance."""
        cases = ((False, None, set()), (True, (), set()), (True, ("0",), {"0"}), (True, ("1",), {"1"}))
        dt, rho, stiffness = 0.02, 5.0, 100.0
        for enabled, destinations, selected in cases:
            for gamma in (0.0, 0.2):
                with self.subTest(enabled=enabled, destinations=destinations, gamma=gamma):
                    model, solver = self._make_solver(
                        factory=SolverFeatherstone,
                        joint_proximal_bodies=enabled,
                        joint_proximal_destination_entries=destinations,
                        iterations=5,
                        rho=rho,
                        gamma=gamma,
                        joint_stiffness=stiffness,
                        gravity=(3, 0, 0),
                    )
                    self.assertEqual(set(solver._admm_proxy_bodies), selected)
                    state, out = model.state(), model.state()
                    vn = np.array([1.3, -0.7])
                    velocity = np.zeros((2, 6), dtype=np.float32)
                    velocity[:, 0] = vn
                    state.body_qd.assign(velocity)
                    qd = state.joint_qd.numpy()
                    qd[:12] = velocity.ravel()
                    state.joint_qd.assign(qd)
                    force = np.array([2.0, -3.0])
                    body_f = np.zeros((2, 6), dtype=np.float32)
                    body_f[:, 0] = force
                    state.body_f.assign(body_f)
                    mass = model.body_mass.numpy()
                    w2 = 1.0 / np.sum(1.0 / mass)
                    w = np.sqrt(w2)
                    P = rho * w2 * (gamma + dt * np.array([str(i) in selected for i in range(2)]))
                    v = vn.copy()
                    u = dual = 0.0
                    for _ in range(5):
                        t = w * (dual + rho * w * (u - (v[1] - v[0])))
                        v = (mass * vn + P * v + dt * (force + 3 * mass + np.array([-t, t]))) / (mass + P)
                        z = v[1] - v[0]
                        u = (rho * w2 * z - w * dual) / (stiffness + rho * w2)
                        dual += rho * w * (u - z)
                    with ExitStack() as stack:
                        steps = []
                        for name, entry in solver._entries.items():
                            local = solver._admm_proxy_bodies[name].entry if name in selected else entry
                            steps.append(
                                stack.enter_context(patch.object(local.solver, "step", wraps=local.solver.step))
                            )
                        solver.step(state, out, None, None, dt)
                        for step in steps:
                            self.assertEqual(step.call_count, 5)
                    np.testing.assert_allclose(out.body_qd.numpy()[:, 0], v, rtol=3e-6, atol=3e-6)
                    group = solver._admm_rr_groups[0]
                    self.assertAlmostEqual(float(group.u.numpy()[0, 0]), u, places=5)
                    self.assertAlmostEqual(float(group.lambda_.numpy()[0, 0]), dual, places=4)

    def test_proxy_selection_validates_names_and_skips_unused_construction(self):
        """Reject unknown destinations and avoid backend proxy checks when disabled."""
        with self.assertRaisesRegex(ValueError, "Unknown ADMM joint proximal destination"):
            self._make_solver(joint_proximal_destination_entries=("missing",))
        for config in ({"joint_proximal_bodies": False}, {"joint_proximal_destination_entries": ()}):
            with self.subTest(config=config):
                _, solver = self._make_solver(
                    factory=lambda view: SolverMuJoCo(view, use_mujoco_cpu=True, integrator="implicitfast"),
                    **config,
                )
                self.assertFalse(solver._admm_proxy_bodies)

    def test_proxy_selection_preserves_scalar_fixed_point(self):
        """Converge selected and disabled proxies to the same physical equations."""
        dt, rho, stiffness = 0.02, 5.0, 100.0
        force = np.array([2.0, -3.0])
        for destinations in ((), ("0",), ("1",)):
            for gamma in (0.0, 0.2):
                with self.subTest(destinations=destinations, gamma=gamma):
                    model, solver = self._make_solver(
                        factory=SolverFeatherstone,
                        joint_proximal_destination_entries=destinations,
                        iterations=128,
                        rho=rho,
                        gamma=gamma,
                        joint_stiffness=stiffness,
                    )
                    state, out = model.state(), model.state()
                    f = np.zeros((2, 6), dtype=np.float32)
                    f[:, 0] = force
                    state.body_f.assign(f)
                    mass = model.body_mass.numpy()
                    expected = np.linalg.solve(
                        np.diag(mass) + dt * stiffness * np.array([[1, -1], [-1, 1]]), dt * force
                    )
                    solver.step(state, out, None, None, dt)
                    np.testing.assert_allclose(out.body_qd.numpy()[:, 0], expected, atol=2e-6)
                    group = solver._admm_rr_groups[0]
                    np.testing.assert_allclose(group.u.numpy(), group.Jv.numpy(), atol=2e-6)

    def test_revolute_projection_uses_common_world_frame(self):
        """Rotate the same constrained angular plane into each endpoint's body frame."""
        _, solver = self._make_solver(kind="revolute", rho=3.0)
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.6)
        solver._entries["1"].state_0.body_q.assign([wp.transform(wp.vec3(0.0), rotation)])
        solver._refresh_proxy_inertia(0.02)
        R = np.array(wp.quat_to_matrix(rotation)).reshape(3, 3)
        W2 = float(solver._admm_rr_revolute_angular_groups[0].W.numpy()[0]) ** 2
        plane = np.diag([0.0, 1.0, 1.0])
        for name, expected in (("0", R @ plane @ R.T), ("1", plane)):
            actual = solver._admm_buffers[name].proxy_inertia.numpy()[0]
            np.testing.assert_allclose(actual, 0.02 * 3.0 * W2 * expected, rtol=3e-6, atol=2e-7)

    def test_backend_inertial_response(self):
        """Apply the same rho and gamma inertia to each supported rigid backend."""
        factories = {
            "featherstone": SolverFeatherstone,
            "xpbd": lambda view: SolverXPBD(view, iterations=10, angular_damping=0.0),
            "vbd": lambda view: SolverVBD(view, iterations=10, rigid_compliant_alm=True),
            "mujoco": lambda view: SolverMuJoCo(view, use_mujoco_cpu=True, integrator="euler"),
        }
        dt, rho = 0.01, 20.0
        for name, factory in factories.items():
            for gamma in (0.0, 0.2):
                with self.subTest(backend=name, gamma=gamma):
                    model, solver = self._make_solver(
                        factory=factory,
                        free_bodies=True,
                        iterations=1,
                        rho=rho,
                        gamma=gamma,
                    )
                    state, out = model.state(), model.state()
                    f = np.zeros((2, 6), dtype=np.float32)
                    f[:, 0] = [3.0, -4.0]
                    state.body_f.assign(f)
                    solver.step(state, out, None, None, dt)
                    P = (dt + gamma) * rho / (1.0 / 2.0 + 1.0 / 5.0)
                    expected = dt * f[:, 0] / (model.body_mass.numpy() + P)
                    if name == "vbd":
                        # Its finite-compliance proxy changes local curvature;
                        # fixed-point replay below is the acceptance criterion.
                        self.assertTrue(
                            np.all(np.abs(out.body_qd.numpy()[:, 0]) < dt * np.abs(f[:, 0]) / model.body_mass.numpy())
                        )
                    else:
                        np.testing.assert_allclose(out.body_qd.numpy()[:, 0], expected, rtol=2e-5, atol=2e-6)

    def test_recentered_proxies_preserve_native_fixed_point(self):
        """Remove numerical bodies and replay the converged physical equations."""
        anchor = np.array([0.3, -0.4, 0.2])
        initial = np.array([[1.3, 0.2, -0.3, 0.7, -0.4, 0.3], [-0.7, -0.1, 0.2, -0.2, 0.1, 0.3]], dtype=np.float32)
        forces = np.array([[1, 3, 2, 0.5, -0.2, 0.4], [-2, 1, 4, -0.4, 0.3, 0.2]], dtype=np.float32)
        jacobians = [np.hstack((np.eye(3), -_cross(anchor))), np.hstack((np.eye(3), np.zeros((3, 3))))]
        dt, stiffness = 0.02, 100.0
        factories = {
            "featherstone": SolverFeatherstone,
            "xpbd": lambda view: SolverXPBD(view, iterations=30, angular_damping=0.0),
            "vbd": lambda view: SolverVBD(view, iterations=30, rigid_compliant_alm=True, rigid_joint_linear_ke=1e5),
            "mujoco": lambda view: SolverMuJoCo(view, use_mujoco_cpu=True, integrator="euler"),
        }
        for backend, factory in factories.items():
            reference = None
            for rho in (20.0, 1000.0):
                for gamma in (0.0, 0.02):
                    with self.subTest(backend=backend, rho=rho, gamma=gamma):
                        model, solver = self._make_solver(
                            factory=factory,
                            free_bodies=True,
                            anchor=anchor,
                            gravity=(0, -9.81, 0),
                            iterations=256,
                            rho=rho,
                            gamma=gamma,
                            joint_stiffness=stiffness,
                        )
                        self.assertEqual(len(solver._admm_proxy_bodies), 2)
                        state, out = model.state(), model.state()
                        state.body_qd.assign(initial)
                        joint_velocity = state.joint_qd.numpy()
                        joint_velocity[:12] = initial.ravel()
                        state.joint_qd.assign(joint_velocity)
                        state.body_f.assign(forces)
                        calls = [0]
                        for proxy in solver._admm_proxy_bodies.values():
                            native_step = proxy.entry.solver.step

                            def counted_step(*args, step=native_step, counter=calls, **kwargs):
                                counter[0] += 1
                                return step(*args, **kwargs)

                            proxy.entry.solver.step = counted_step
                        solver.step(state, out, None, None, dt)
                        self.assertEqual(calls[0], 512)
                        velocity = out.body_qd.numpy()
                        self.assertTrue(np.all(np.isfinite(velocity)))
                        group = solver._admm_rr_groups[0]
                        np.testing.assert_allclose(group.u.numpy(), group.Jv.numpy(), atol=2e-5)
                        jv = jacobians[1] @ velocity[1] - jacobians[0] @ velocity[0]
                        for body in range(2):
                            entry = solver._entries[str(body)]
                            physical, replay = entry.view.state(), entry.view.state()
                            physical.body_qd.assign(initial[body : body + 1])
                            physical.joint_qd.assign(initial[body])
                            physical.body_f.assign(
                                [forces[body] + (1 if body == 0 else -1) * stiffness * jacobians[body].T @ jv]
                            )
                            if backend == "vbd":
                                entry.solver.coupling_notify_input_state_update(
                                    physical, newton.StateFlags.BODY, dt=0.0
                                )
                            entry.solver.step(physical, replay, None, None, dt)
                            np.testing.assert_allclose(velocity[body], replay.body_qd.numpy()[0], atol=4e-5, rtol=3e-5)
                            np.testing.assert_allclose(out.body_q.numpy()[body], replay.body_q.numpy()[0], atol=2e-6)
                        if reference is None:
                            reference = velocity
                        else:
                            np.testing.assert_allclose(velocity, reference, atol=4e-5, rtol=3e-5)

    def test_gamma_preserves_native_fixed_point_without_all_proxies(self):
        """Cancel gamma for spinning anisotropic bodies with optional proxy destinations."""
        anchor = np.array([0.3, -0.4, 0.2])
        initial = np.array([[1.3, 0.2, -0.3, 0.7, -0.4, 0.3], [-0.7, -0.1, 0.2, -0.2, 0.1, 0.3]], dtype=np.float32)
        forces = np.array([[1, 3, 2, 0.5, -0.2, 0.4], [-2, 1, 4, -0.4, 0.3, 0.2]], dtype=np.float32)
        jacobians = [np.hstack((np.eye(3), -_cross(anchor))), np.hstack((np.eye(3), np.zeros((3, 3))))]
        factories = {
            "featherstone": SolverFeatherstone,
            "mujoco": lambda view: SolverMuJoCo(view, use_mujoco_cpu=True, integrator="euler"),
            "xpbd": lambda view: SolverXPBD(view, iterations=30, angular_damping=0.0),
            "vbd": lambda view: SolverVBD(view, iterations=30, rigid_compliant_alm=True),
        }
        for backend, factory in factories.items():
            for enabled, destinations in ((False, None), (True, ()), (True, ("0",)), (True, ("1",))):
                with self.subTest(backend=backend, enabled=enabled, destinations=destinations):
                    model, solver = self._make_solver(
                        factory=factory,
                        anchor=anchor,
                        gravity=(0, -9.81, 0),
                        rho=20.0,
                        gamma=0.02,
                        iterations=256,
                        joint_stiffness=100.0,
                        joint_proximal_bodies=enabled,
                        joint_proximal_destination_entries=destinations,
                    )
                    _, reference = self._make_solver(
                        factory=factory,
                        anchor=anchor,
                        gravity=(0, -9.81, 0),
                        gamma=0.0,
                        joint_proximal_bodies=False,
                    )
                    state, out = model.state(), model.state()
                    state.body_qd.assign(initial)
                    qd = state.joint_qd.numpy()
                    qd[:12] = initial.ravel()
                    state.joint_qd.assign(qd)
                    state.body_f.assign(forces)
                    solver.step(state, out, None, None, 0.02)
                    velocity = out.body_qd.numpy()
                    group = solver._admm_rr_groups[0]
                    np.testing.assert_allclose(group.u.numpy(), group.Jv.numpy(), atol=2e-5)
                    jv = jacobians[1] @ velocity[1] - jacobians[0] @ velocity[0]
                    for body in range(2):
                        entry = reference._entries[str(body)]
                        physical, replay = entry.view.state(), entry.view.state()
                        physical.body_qd.assign(initial[body : body + 1])
                        physical.joint_qd.assign(initial[body])
                        physical.body_f.assign(
                            [forces[body] + (1 if body == 0 else -1) * 100.0 * jacobians[body].T @ jv]
                        )
                        entry.solver.coupling_notify_input_state_update(
                            physical, newton.StateFlags.BODY | newton.StateFlags.JOINT, dt=0.0
                        )
                        entry.solver.step(physical, replay, None, None, 0.02)
                        np.testing.assert_allclose(velocity[body], replay.body_qd.numpy()[0], atol=4e-5, rtol=3e-5)
                        np.testing.assert_allclose(out.body_q.numpy()[body], replay.body_q.numpy()[0], atol=2e-6)

    def test_gamma_preserves_articulated_fixed_point(self):
        """Cancel numerical inertia bias on moving fixed and floating chains."""
        factories = {
            "featherstone": SolverFeatherstone,
            "mujoco": lambda view: SolverMuJoCo(view, use_mujoco_cpu=True, integrator="euler"),
        }
        for backend, factory in factories.items():
            for floating in (False, True):
                reference = None
                for gamma, proxies in ((0.0, False), (0.02, False), (0.02, True)):
                    with self.subTest(backend=backend, floating=floating, gamma=gamma, proxies=proxies):
                        builder = newton.ModelBuilder(gravity=(0, -9.81, 0))
                        for i in range(2):
                            base = builder.add_link(
                                mass=2.0 + 3.0 * i,
                                com=wp.vec3(0.3, 0.4, 0.2),
                                inertia=wp.mat33(np.diag([1.0, 2.0, 3.0])),
                            )
                            root = (
                                builder.add_joint_free(base)
                                if floating
                                else builder.add_joint_revolute(-1, base, axis=wp.vec3(0, 0, 1))
                            )
                            tip = builder.add_link(
                                mass=1.0, com=wp.vec3(0.2, 0.3, 0.1), inertia=wp.mat33(np.diag([0.2, 0.3, 0.4]))
                            )
                            hinge = builder.add_joint_revolute(
                                base,
                                tip,
                                axis=wp.vec3(0, 1, 0),
                                parent_xform=wp.transform((0.6, 0.2, 0), wp.quat_identity()),
                            )
                            builder.add_articulation([root, hinge])
                        builder.add_joint_ball(1, 3, parent_xform=wp.transform((0.2, -0.1, 0.3), wp.quat_identity()))
                        model = builder.finalize(device="cpu")
                        solver = SolverCoupledADMM(
                            model,
                            [
                                SolverCoupled.Entry(
                                    name=str(i), bodies=[2 * i, 2 * i + 1], joints=[2 * i, 2 * i + 1], solver=factory
                                )
                                for i in range(2)
                            ],
                            SolverCoupledADMM.Config(
                                rho=5.0,
                                gamma=gamma,
                                iterations=512,
                                joint_stiffness=100.0,
                                joint_proximal_bodies=proxies,
                            ),
                        )
                        state, out = model.state(), model.state()
                        q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
                        qs, ds = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
                        for i in range(2):
                            if floating:
                                qd[ds[2 * i] : ds[2 * i] + 6] = [1.3, 0.2, -0.3, 0.7, -0.4, 0.3]
                            else:
                                q[qs[2 * i]] = (0.3, -0.4)[i]
                                qd[ds[2 * i]] = (1.3, -0.9)[i]
                            q[qs[2 * i + 1]] = (-0.2, 0.6)[i]
                            qd[ds[2 * i + 1]] = (-0.7, 0.8)[i]
                        state.joint_q.assign(q)
                        state.joint_qd.assign(qd)
                        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                        solver.step(state, out, None, None, 0.02)
                        group = solver._admm_rr_groups[0]
                        np.testing.assert_allclose(group.u.numpy(), group.Jv.numpy(), atol=2e-5)
                        actual = (out.joint_qd.numpy(), out.body_q.numpy(), out.body_qd.numpy())
                        if reference is None:
                            reference = actual
                        else:
                            for value, expected in zip(actual, reference, strict=True):
                                np.testing.assert_allclose(value, expected, atol=3e-5, rtol=3e-5)

    def test_proxy_native_accuracy_and_restarts(self):
        """Preserve physical replay across native budgets and successive timesteps."""
        for native_iterations in (5, 30):
            model, solver = self._make_solver(
                factory=lambda view, count=native_iterations: SolverVBD(
                    view, iterations=count, rigid_compliant_alm=True
                ),
                free_bodies=True,
                anchor=(0.3, -0.4, 0.2),
                iterations=256,
                rho=20.0,
                gamma=0.02,
                joint_stiffness=100.0,
            )
            state, out = model.state(), model.state()
            state.body_qd.assign([[1, 0.2, -0.3, 0.7, -0.4, 0.3], [-0.7, 0, 0.2, -0.2, 0.1, 0.3]])
            for _ in range(2):
                solver.step(state, out, None, None, 0.02)
                self.assertTrue(np.all(np.isfinite(out.body_qd.numpy())))
                for name, entry in solver._entries.items():
                    buf = solver._admm_buffers[name]
                    physical, replay = entry.view.state(), entry.view.state()
                    wp.copy(physical.body_q, buf.body_q_n)
                    wp.copy(physical.body_qd, buf.body_qd_n)
                    wp.copy(physical.body_f, buf.body_f)
                    entry.solver.coupling_notify_input_state_update(physical, newton.StateFlags.BODY, dt=0.0)
                    entry.solver.step(physical, replay, None, None, 0.02)
                    np.testing.assert_allclose(
                        entry.state_1.body_qd.numpy(), replay.body_qd.numpy(), atol=3e-5, rtol=3e-5
                    )
                state, out = out, state

    def test_proxy_prepare_preserves_vbd_history(self):
        """Notify recentered poses without resetting owned joint or contact history."""
        for compliant in (False, True):
            with self.subTest(compliant=compliant):
                builder = newton.ModelBuilder(gravity=(0, 0, 0))
                parent = builder.add_link(mass=2.0, inertia=wp.mat33(np.eye(3)))
                hinge = builder.add_joint_revolute(-1, parent, axis=wp.vec3(0, 0, 1))
                builder.add_articulation([hinge])
                child = builder.add_body(mass=5.0, inertia=wp.mat33(np.eye(3)))
                builder.add_joint_ball(parent, child)
                builder.add_shape_sphere(parent, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
                builder.add_particle(pos=(2, 0, 0), vel=(0, 0, 0), mass=1.0)
                builder.color()
                model = builder.finalize(device="cpu")
                solver = SolverCoupledADMM(
                    model,
                    [
                        SolverCoupled.Entry(
                            name=str(body),
                            bodies=[body],
                            joints=[body],
                            particles=[0] if body == 0 else [],
                            solver=lambda view, mode=compliant: SolverVBD(
                                view, iterations=10, rigid_compliant_alm=mode, rigid_contact_history=True
                            ),
                        )
                        for body in (parent, child)
                    ],
                    SolverCoupledADMM.Config(iterations=4, rho=20.0, gamma=0.02),
                )
                self.assertEqual(set(solver._admm_proxy_bodies), {"0", "1"})
                proxies = solver._admm_proxy_bodies["0"]
                native = proxies.entry.solver
                fields = ("joint_lambda_lin", "joint_lambda_ang", "joint_C0_lin", "joint_C0_ang", "joint_penalty_k")
                saved = {}
                original = {}
                for field in fields:
                    array = getattr(native, field)
                    original[field] = array.numpy().copy()
                    values = np.arange(array.numpy().size, dtype=np.float32).reshape(array.numpy().shape) + 1.0
                    array.assign(values)
                    saved[field] = values
                contact_mask = native._contact_history_reset_mask.numpy().copy()
                target = wp.transform((0.3, -0.4, 0.2), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.7))
                proxies.target.assign([target])
                with patch.object(native, "reset", side_effect=AssertionError("recenter must not reset")):
                    proxies.prepare(solver._entries["0"], solver._admm_buffers["0"], dt=0.02)
                for field, values in saved.items():
                    np.testing.assert_array_equal(getattr(native, field).numpy(), values)
                np.testing.assert_array_equal(native._contact_history_reset_mask.numpy(), contact_mask)
                np.testing.assert_array_equal(proxies.entry.state_0.body_q.numpy()[-1], np.asarray(target))
                np.testing.assert_array_equal(proxies.entry.state_0.body_qd.numpy()[-1], np.zeros(6))
                np.testing.assert_array_equal(native.body_q_prev.numpy(), proxies.entry.state_0.body_q.numpy())
                np.testing.assert_array_equal(native._coupling_body_q_prev_snapshot.numpy(), native.body_q_prev.numpy())

                for field, values in original.items():
                    getattr(native, field).assign(values)
                state, out = model.state(), model.state()
                state.body_f.assign([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
                with patch.object(native, "reset", side_effect=AssertionError("ADMM must not reset")):
                    with patch.object(native, "step", wraps=native.step) as step:
                        solver.step(state, out, None, None, 0.01)
                        self.assertEqual(step.call_count, 4)
                self.assertTrue(np.all(np.isfinite(out.body_qd.numpy())))
                self.assertTrue(np.all(np.isfinite(out.particle_q.numpy())))

    def test_incomplete_proxy_construction_does_not_change_algorithm(self):
        """Reject incomplete constructions instead of switching to an inner solver."""
        with self.assertRaisesRegex(ValueError, "Euler"):
            self._make_solver(factory=lambda view: SolverMuJoCo(view, use_mujoco_cpu=True, integrator="implicitfast"))
        with self.assertRaisesRegex(ValueError, "free-axis"):
            self._make_solver(kind="revolute", factory=lambda view: SolverXPBD(view, iterations=10), gamma=0.0)

    def test_proxy_multiple_anchors_preserve_physical_com(self):
        """Replay an aggregated proxy with distinct anchors and a nonzero physical COM."""
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for mass in (2.0, 3.0, 5.0):
            builder.add_body(mass=mass, com=wp.vec3(0.1, -0.2, 0.05), inertia=wp.mat33(np.diag([1, 2, 3])))
        for body, point in ((1, (0.3, -0.4, 0.2)), (2, (-0.2, 0.3, 0.1))):
            builder.add_joint_ball(0, body, parent_xform=wp.transform(point, wp.quat_identity()))
        builder.color()
        model = builder.finalize(device="cpu")
        solver = SolverCoupledADMM(
            model,
            [
                SolverCoupled.Entry(
                    name=str(i),
                    bodies=[i],
                    joints=[i],
                    solver=lambda view: SolverXPBD(view, iterations=20, angular_damping=0.0),
                )
                for i in range(3)
            ],
            SolverCoupledADMM.Config(iterations=256, rho=100.0, gamma=0.02, joint_stiffness=100.0),
        )
        self.assertEqual(solver._admm_proxy_bodies["0"].bodies.shape[0], 1)
        state, out = model.state(), model.state()
        state.body_qd.assign([[1, 0, 0, 0.2, -0.1, 0.4], [-1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        solver.step(state, out, None, None, 0.02)
        self.assertTrue(np.all(np.isfinite(out.body_qd.numpy())))
        for name, entry in solver._entries.items():
            buf = solver._admm_buffers[name]
            physical, replay = entry.view.state(), entry.view.state()
            wp.copy(physical.body_q, buf.body_q_n)
            wp.copy(physical.body_qd, buf.body_qd_n)
            wp.copy(physical.body_f, buf.body_f)
            entry.solver.step(physical, replay, None, None, 0.02)
            np.testing.assert_allclose(entry.state_1.body_qd.numpy(), replay.body_qd.numpy(), atol=3e-5, rtol=3e-5)
            np.testing.assert_array_equal(entry.view.body_com.numpy()[0], model.body_com.numpy()[int(name)])

    def test_off_center_convergence_is_independent_of_rho(self):
        """Check stationarity as well as primal feasibility at the converged solution."""
        dt, stiffness = 0.02, 100.0
        anchor = np.array([0.3, -0.4, 0.2])
        M = np.diag([2.0] * 3 + [1.0, 2.0, 3.0] + [5.0] * 3 + [1.0, 2.0, 3.0])
        J = np.hstack([-np.eye(3), _cross(anchor), np.eye(3), np.zeros((3, 3))])
        initial = np.array([[1.3, 0.2, -0.3, 0, 0, 0], [-0.7, -0.1, 0.2, 0, 0, 0]])
        forces = np.array([[1, 3, 2, 0.5, -0.2, 0.4], [-2, 1, 4, -0.4, 0.3, 0.2]])
        system = M + dt * stiffness * J.T @ J
        rhs = M @ initial.ravel() + dt * forces.ravel()
        expected = np.linalg.solve(system, rhs)
        for rho in (20.0, 100.0, 1000.0):
            for gamma in (0.0, 0.02):
                with self.subTest(rho=rho, gamma=gamma):
                    model, solver = self._make_solver(
                        anchor=anchor, rho=rho, gamma=gamma, iterations=512, joint_stiffness=stiffness
                    )
                    state, out = model.state(), model.state()
                    state.body_qd.assign(initial.astype(np.float32))
                    joint_qd = state.joint_qd.numpy()
                    joint_qd[:12] = initial.ravel()
                    state.joint_qd.assign(joint_qd)
                    state.body_f.assign(forces.astype(np.float32))
                    solver.step(state, out, None, None, dt)
                    velocity = out.body_qd.numpy().ravel()
                    np.testing.assert_allclose(velocity, expected, rtol=3e-5, atol=3e-6)
                    np.testing.assert_allclose(system @ velocity, rhs, rtol=3e-5, atol=1e-5)
                    group = solver._admm_rr_groups[0]
                    np.testing.assert_allclose(group.Jv.numpy(), group.u.numpy(), atol=2e-6)
                    # The native integrator must advance the physical COM with
                    # the returned velocity, without a post-step pose correction.
                    np.testing.assert_allclose(out.body_q.numpy()[:, :3], dt * velocity.reshape(2, 6)[:, :3], atol=2e-7)

    def test_vbd_uses_saved_pose_for_joint_jacobian(self):
        """VBD can modify its input pose; the dual update must keep the original J."""
        anchor = np.array([0.3, -0.4, 0.2])
        model, solver = self._make_solver(
            anchor=anchor,
            factory=lambda view: SolverVBD(view, iterations=1, rigid_compliant_alm=True),
            free_bodies=True,
            iterations=4,
            rho=20.0,
            gamma=0.02,
        )
        state, out = model.state(), model.state()
        forces = np.zeros((2, 6), dtype=np.float32)
        forces[0, 3:] = [20, -40, 70]
        state.body_f.assign(forces)
        solver.step(state, out, None, None, 0.02)
        velocity = out.body_qd.numpy()
        expected = velocity[1, :3] - velocity[0, :3] - np.cross(velocity[0, 3:], anchor)
        np.testing.assert_allclose(solver._admm_rr_groups[0].Jv.numpy()[0], expected, atol=1e-7)

    def test_local_solve_preserves_native_hinge(self):
        """The final physical pose must still satisfy the subsolver's own joint."""
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        parent = builder.add_link(
            xform=wp.transform((0.5, 0, 0), wp.quat_identity()),
            mass=2.0,
            inertia=wp.mat33(np.eye(3)),
        )
        hinge = builder.add_joint_revolute(
            -1,
            parent,
            child_xform=wp.transform((-0.5, 0, 0), wp.quat_identity()),
            axis=wp.vec3(0, 0, 1),
            armature=0.0,
            damping=0.0,
        )
        builder.add_articulation([hinge])
        child = builder.add_body(
            xform=wp.transform((0.8, 0.2, 0), wp.quat_identity()),
            mass=5.0,
            inertia=wp.mat33(np.eye(3)),
        )
        builder.add_joint_ball(parent, child, parent_xform=wp.transform((0.3, 0.2, 0), wp.quat_identity()))
        builder.color()
        model = builder.finalize(device="cpu")
        solver = SolverCoupledADMM(
            model,
            [
                SolverCoupled.Entry(
                    name=str(body),
                    bodies=[body],
                    joints=[body],
                    solver=lambda view: SolverMuJoCo(view, use_mujoco_cpu=True, integrator="euler"),
                )
                for body in (parent, child)
            ],
            SolverCoupledADMM.Config(iterations=16, rho=20.0, gamma=0.02, joint_stiffness=100.0),
        )
        self.assertEqual(len(solver._admm_proxy_bodies), 2)
        for proxies in solver._admm_proxy_bodies.values():
            view = proxies.entry.view
            self.assertEqual(view.max_joints_per_articulation, view.joint_count)
            self.assertTrue(np.all(view.joint_ancestor.numpy()[proxies.joint_offset :] >= 0))
        state, out = model.state(), model.state()
        forces = np.zeros((2, 6), dtype=np.float32)
        forces[child, 1] = 100.0
        for _ in range(3):
            state.body_f.assign(forces)
            solver.step(state, out, None, None, 0.02)
            q = out.body_q.numpy()[parent]
            transform = wp.transform(wp.vec3(*q[:3]), wp.quat(*q[3:]))
            anchor = wp.transform_point(transform, wp.vec3(-0.5, 0, 0))
            np.testing.assert_allclose(np.array(anchor), 0.0, atol=1e-7)
            np.testing.assert_array_equal(solver.view(str(parent)).body_com.numpy()[0], [0, 0, 0])
            state, out = out, state
        self.assertGreater(abs(float(state.body_qd.numpy()[parent, 5])), 0.01)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required")
    def test_proxy_cuda_graph(self):
        """Replay private proxy preparation and native solves inside a CUDA graph."""
        factories = {
            "featherstone": SolverFeatherstone,
            "xpbd": lambda view: SolverXPBD(view, iterations=10, angular_damping=0.0),
            "vbd": lambda view: SolverVBD(view, iterations=10, rigid_compliant_alm=True),
            "mujoco": lambda view: SolverMuJoCo(view, use_mujoco_cpu=False, iterations=10, integrator="euler"),
        }
        for backend, factory in factories.items():
            with self.subTest(backend=backend):
                model, solver = self._make_solver(
                    device="cuda:0",
                    factory=factory,
                    free_bodies=True,
                    anchor=(0.3, -0.2, 0.1),
                    rho=20.0,
                    gamma=0.02,
                    iterations=8,
                )
                self.assertEqual(len(solver._admm_proxy_bodies), 2)
                state, out = model.state(), model.state()
                state.body_f.assign([[0, 5, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]])

                def step(solver=solver, state=state, out=out):
                    for group in solver._admm_rr_groups:
                        group.u.zero_()
                        group.lambda_.zero_()
                    solver.step(state, out, None, None, 0.01)

                step()
                step()
                expected = out.body_qd.numpy()
                # A captured transfer into a temporary CPU array can corrupt
                # the host heap when the graph is replayed after it is freed.
                with patch.object(SolverVBD, "_to_numpy", side_effect=AssertionError("host read during capture")):
                    with wp.ScopedCapture(device=model.device) as capture:
                        step()
                wp.capture_launch(capture.graph)
                self.assertTrue(np.all(np.isfinite(out.body_qd.numpy())))
                np.testing.assert_allclose(out.body_qd.numpy(), expected, rtol=3e-5, atol=2e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required")
    def test_gamma_without_proxies_cuda_graph(self):
        """Capture reduced-coordinate gamma compensation with numerical bodies disabled."""
        factories = {
            "featherstone": SolverFeatherstone,
            "mujoco": lambda view: SolverMuJoCo(view, use_mujoco_cpu=False, integrator="euler"),
        }
        for backend, factory in factories.items():
            with self.subTest(backend=backend):
                model, solver = self._make_solver(
                    device="cuda:0",
                    factory=factory,
                    anchor=(0.3, -0.2, 0.1),
                    rho=20.0,
                    gamma=0.02,
                    iterations=8,
                    joint_proximal_bodies=False,
                )
                state, out = model.state(), model.state()
                state.body_qd.assign([[1, 0.2, -0.3, 0.7, -0.4, 0.3], [-0.7, 0, 0.2, -0.2, 0.1, 0.3]])
                qd = state.joint_qd.numpy()
                qd[:12] = state.body_qd.numpy().ravel()
                state.joint_qd.assign(qd)
                state.body_f.assign([[0, 5, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]])

                def step(solver=solver, state=state, out=out):
                    for group in solver._admm_rr_groups:
                        group.u.zero_()
                        group.lambda_.zero_()
                    solver.step(state, out, None, None, 0.01)

                step()
                expected = out.body_qd.numpy()
                with wp.ScopedCapture(device=model.device) as capture:
                    step()
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(out.body_qd.numpy(), expected, rtol=3e-5, atol=2e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required")
    def test_cuda_graph(self):
        """Capture and replay off-center joint inertia updates with optional gamma."""
        for gamma in (0.0, 0.2):
            model, solver = self._make_solver(
                device="cuda:0",
                factory=lambda view: SolverMuJoCo(view, use_mujoco_cpu=False, integrator="euler"),
                anchor=(0.3, -0.2, 0.1),
                kind="revolute",
                rho=10.0,
                gamma=gamma,
            )
            state, out = model.state(), model.state()
            forces = np.zeros((2, 6), dtype=np.float32)
            forces[0, 1] = 5.0
            state.body_f.assign(forces)
            solver.step(state, out, None, None, 0.01)
            groups = [*solver._admm_rr_groups, *solver._admm_rr_revolute_angular_groups]
            for group in groups:
                group.u.zero_()
                group.lambda_.zero_()
            solver.step(state, out, None, None, 0.01)
            expected = out.body_qd.numpy()
            for group in groups:
                group.u.zero_()
                group.lambda_.zero_()
            with wp.ScopedCapture(device=model.device) as capture:
                solver.step(state, out, None, None, 0.01)
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(out.body_qd.numpy(), expected, rtol=2e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
