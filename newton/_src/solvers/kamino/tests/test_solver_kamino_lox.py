# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the LOX Kamino dynamics backend."""

import math
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.geometry.detector import CollisionDetector
from newton._src.solvers.kamino._src.solver_kamino_impl import SolverKaminoImpl
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton.tests.utils.basics import (
    build_box_on_plane,
    build_boxes_hinged,
    build_cartpole,
)


@wp.kernel
def _inject_unsafe_on_iteration(
    iteration_count: wp.array[wp.int32],
    target_iteration: wp.int32,
    unsafe: wp.array[wp.int32],
):
    world = wp.tid()
    if iteration_count[world] == target_iteration:
        unsafe[world] = 1


def _build_revolute_dynamics_model(
    *,
    damping: float,
    friction: float,
    velocity: float,
    armature: float = 0.0,
    target_ke: float = 0.0,
    target_kd: float = 0.0,
    effort_limit: float = math.inf,
    actuator_mode: newton.JointTargetMode | None = None,
    device: wp.DeviceLike = None,
) -> newton.Model:
    """Build a gravity-free world-to-body hinge with consistent initial velocity."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=newton.Axis.Y,
        damping=damping,
        friction=friction,
        armature=armature,
        target_ke=target_ke,
        target_kd=target_kd,
        effort_limit=effort_limit,
        actuator_mode=actuator_mode,
    )
    builder.add_articulation([joint])
    builder.body_qd[body] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, velocity, 0.0)
    builder.joint_qd[builder.joint_qd_start[joint]] = velocity
    builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _build_massless_fixed_child_drive_model(
    *,
    include_massless_fixed_child: bool = True,
    device: wp.DeviceLike = None,
) -> newton.Model:
    """Build a driven link with an optional massless fixed child."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    inertia = wp.mat33f(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
    driven = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
    revolute = builder.add_joint_revolute(
        parent=-1,
        child=driven,
        axis=newton.Axis.Z,
        armature=0.1,
        target_ke=650.0,
        target_kd=100.0,
        effort_limit=math.inf,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    joints = [revolute]
    if include_massless_fixed_child:
        child = builder.add_link(
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.1), wp.quat_identity(dtype=wp.float32)),
            mass=0.0,
            inertia=wp.mat33f(),
            lock_inertia=True,
        )
        joints.append(
            builder.add_joint_fixed(
                parent=driven,
                child=child,
                parent_xform=wp.transformf(
                    wp.vec3f(0.0, 0.0, 0.1),
                    wp.quat_identity(dtype=wp.float32),
                ),
            )
        )
    builder.add_articulation(joints)
    builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _build_driven_link_with_grounded_fixed_base(*, device: wp.DeviceLike = None) -> tuple[newton.Model, int]:
    """Build a driven link whose prescribed base overlaps the ground."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    inertia = wp.mat33f(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
    base = builder.add_link(
        xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.05), wp.quat_identity(dtype=wp.float32)),
        mass=1.0,
        inertia=inertia,
        lock_inertia=True,
    )
    builder.add_shape_box(
        base,
        hx=0.1,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )
    driven = builder.add_link(
        xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.25), wp.quat_identity(dtype=wp.float32)),
        mass=1.0,
        inertia=inertia,
        lock_inertia=True,
    )
    fixed = builder.add_joint_fixed(
        parent=-1,
        child=base,
        parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.05), wp.quat_identity(dtype=wp.float32)),
    )
    revolute = builder.add_joint_revolute(
        parent=base,
        child=driven,
        axis=newton.Axis.Z,
        armature=0.1,
        target_ke=650.0,
        target_kd=100.0,
        effort_limit=math.inf,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    builder.add_articulation([fixed, revolute])
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    builder.end_world()
    model = builder.finalize(device=device)
    model.rigid_contact_max = 16
    return model, driven


def _build_zero_mass_prismatic_anchor_model(*, device: wp.DeviceLike = None) -> tuple[newton.Model, int, int]:
    """Build a zero-mass anchor connected to a dynamic prismatic child."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    anchor = builder.add_link()
    child = builder.add_link()
    builder.add_shape_box(anchor, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    builder.add_shape_box(child)
    fixed = builder.add_joint_fixed(parent=-1, child=anchor)
    prismatic = builder.add_joint_prismatic(parent=anchor, child=child, axis=newton.Axis.Z)
    builder.add_articulation([fixed, prismatic])
    builder.end_world()
    return builder.finalize(device=device), anchor, child


def _build_prescribed_only_model(*, device: wp.DeviceLike = None) -> tuple[newton.Model, int]:
    """Build a world containing only one zero-mass body."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    body = builder.add_link()
    builder.add_shape_box(body, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    fixed = builder.add_joint_fixed(parent=-1, child=body)
    builder.add_articulation([fixed])
    builder.end_world()
    return builder.finalize(device=device), body


def _build_flagged_kinematic_model(*, device: wp.DeviceLike = None) -> tuple[newton.Model, int]:
    """Build a massive free body whose motion is prescribed by its body flag."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    builder.end_world()
    return builder.finalize(device=device), body


def _build_prescribed_rotating_joint_model(
    *,
    angle: float,
    joint_kind: str = "ball",
    nonidentity_frames: bool = False,
    device: wp.DeviceLike = None,
) -> tuple[newton.Model, int, int, wp.transformf, wp.transformf]:
    """Build a rotating prescribed parent with an offset dynamic child."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    inertia = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    parent = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True, is_kinematic=True)
    child = builder.add_link(
        xform=wp.transformf(wp.vec3f(1.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
        mass=1.0,
        inertia=inertia,
        lock_inertia=True,
    )
    root = builder.add_joint_free(parent=-1, child=parent)
    frame_orientation = (
        wp.quat_from_axis_angle(wp.normalize(wp.vec3f(1.0, 2.0, -1.0)), 0.7)
        if nonidentity_frames
        else wp.quat_identity(dtype=wp.float32)
    )
    parent_xform = wp.transformf(wp.vec3f(1.0, 0.0, 0.0), frame_orientation)
    child_xform = wp.transformf(wp.vec3f(0.0), frame_orientation)
    if joint_kind == "ball":
        joint = builder.add_joint_ball(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
        )
    elif joint_kind == "revolute":
        joint = builder.add_joint_revolute(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            axis=newton.Axis.X,
        )
    else:
        raise ValueError(f"Unsupported joint kind: {joint_kind}")
    builder.add_articulation([root, joint])
    builder.body_qd[parent] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, angle / 0.01)
    builder.end_world()
    return builder.finalize(device=device), parent, child, parent_xform, child_xform


def _rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a vector by an xyzw quaternion."""
    axis = quaternion[:3]
    scalar = quaternion[3]
    return vector + 2.0 * (scalar * np.cross(axis, vector) + np.cross(axis, np.cross(axis, vector)))


def _joint_anchor_error(
    body_pose: np.ndarray,
    parent: int,
    child: int,
    parent_xform: wp.transformf,
    child_xform: wp.transformf,
) -> np.ndarray:
    """Return child-minus-parent joint anchor separation in world coordinates [m]."""
    parent_anchor = body_pose[parent, :3] + _rotate_vector(body_pose[parent, 3:], np.asarray(parent_xform.p))
    child_anchor = body_pose[child, :3] + _rotate_vector(body_pose[child, 3:], np.asarray(child_xform.p))
    return child_anchor - parent_anchor


def _quaternion_product(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Multiply xyzw quaternions."""
    first_axis = first[:3]
    second_axis = second[:3]
    return np.concatenate(
        (
            first[3] * second_axis + second[3] * first_axis + np.cross(first_axis, second_axis),
            np.asarray([first[3] * second[3] - np.dot(first_axis, second_axis)]),
        )
    )


def _joint_angular_error(
    body_pose: np.ndarray,
    parent: int,
    child: int,
    parent_xform: wp.transformf,
    child_xform: wp.transformf,
    joint_kind: str,
) -> float:
    """Return the constrained joint-frame angular error [rad]."""
    parent_frame = _quaternion_product(body_pose[parent, 3:], np.asarray(parent_xform.q))
    child_frame = _quaternion_product(body_pose[child, 3:], np.asarray(child_xform.q))
    parent_inverse = parent_frame * np.asarray((-1.0, -1.0, -1.0, 1.0))
    relative = _quaternion_product(parent_inverse, child_frame)
    relative /= np.linalg.norm(relative)
    if joint_kind == "fixed":
        return float(2.0 * np.arccos(np.clip(abs(relative[3]), 0.0, 1.0)))
    follower_axis = _rotate_vector(relative, np.asarray((1.0, 0.0, 0.0)))
    return float(np.linalg.norm(np.cross(np.asarray((1.0, 0.0, 0.0)), follower_axis)))


def _build_two_body_joint_model(
    *,
    joint_kind: str,
    mass_ratio: float,
    reverse_body_order: bool,
    device: wp.DeviceLike = None,
) -> tuple[newton.Model, int, int, wp.transformf, wp.transformf]:
    """Build an offset two-body fixed or revolute joint stress fixture."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    frame_orientation = wp.quat_from_axis_angle(wp.normalize(wp.vec3f(1.0, -2.0, 0.5)), 0.6)
    parent_xform = wp.transformf(wp.vec3f(0.3, -0.2, 0.7), frame_orientation)
    child_xform = wp.transformf(wp.vec3f(-0.1, 0.2, -0.3), frame_orientation)
    child_position = wp.vec3f(0.4, -0.4, 1.0)
    parent_inertia = wp.mat33f(0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4)
    child_inertia = parent_inertia / mass_ratio

    def add_parent() -> int:
        return builder.add_link(mass=1.0, inertia=parent_inertia, lock_inertia=True)

    def add_child() -> int:
        return builder.add_link(
            xform=wp.transformf(child_position, wp.quat_identity(dtype=wp.float32)),
            mass=1.0 / mass_ratio,
            inertia=child_inertia,
            lock_inertia=True,
        )

    if reverse_body_order:
        child = add_child()
        parent = add_parent()
    else:
        parent = add_parent()
        child = add_child()
    root = builder.add_joint_revolute(parent=-1, child=parent, axis=newton.Axis.Y)
    if joint_kind == "fixed":
        joint = builder.add_joint_fixed(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
        )
    elif joint_kind == "revolute":
        joint = builder.add_joint_revolute(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            axis=newton.Axis.X,
        )
    else:
        raise ValueError(f"Unsupported joint kind: {joint_kind}")
    builder.add_articulation([root, joint])
    builder.end_world()
    return builder.finalize(device=device), parent, child, parent_xform, child_xform


def _build_rod_model(
    *,
    binary: bool = False,
    explicit_world: bool = True,
    enabled: bool = True,
    stretch_stiffness: float = 0.0,
    stretch_damping: float = 0.0,
    shear_stiffness: float = 0.0,
    shear_damping: float = 0.0,
    bend_stiffness: float = 0.0,
    bend_damping: float = 0.0,
    twist_stiffness: float = 0.0,
    twist_damping: float = 0.0,
    child_xform: wp.transformf | None = None,
    device: wp.DeviceLike = None,
) -> tuple[newton.Model, int, int]:
    """Build a gravity-free world or binary rod with coincident rest anchors."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverKamino.register_custom_attributes(builder)
    if explicit_world:
        builder.begin_world()
    inertia = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    parent = -1
    if binary:
        parent = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
    if child_xform is None:
        child_xform = wp.transform_identity(dtype=wp.float32)
    child = builder.add_link(xform=child_xform, mass=1.0, inertia=inertia, lock_inertia=True)
    joint = builder.add_joint_rod(
        parent,
        child,
        stretch_stiffness=stretch_stiffness,
        stretch_damping=stretch_damping,
        shear_stiffness=shear_stiffness,
        shear_damping=shear_damping,
        bend_stiffness=bend_stiffness,
        bend_damping=bend_damping,
        twist_stiffness=twist_stiffness,
        twist_damping=twist_damping,
        enabled=enabled,
    )
    builder.add_articulation([joint])
    if explicit_world:
        builder.end_world()
    return builder.finalize(device=device), parent, child


class TestSolverKaminoLOX(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def make_config(
        self,
        compute_solution_metrics: bool = False,
        sparse_jacobian: bool = True,
    ) -> SolverKamino.Config:
        return SolverKamino.Config(
            dynamics_solver="lox",
            compute_solution_metrics=compute_solution_metrics,
            sparse_jacobian=sparse_jacobian,
            lox=kamino_config.LOXSolverConfig(
                max_iterations=25,
                projection_iterations=5,
            ),
        )

    def test_free_fall_advances_projected_velocity_and_pose(self):
        model = ModelKamino.from_newton(build_box_on_plane(ground=False).finalize(device=self.device))
        solver = SolverKaminoImpl(model=model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        time_step = 0.01

        solver.step(state_previous, state_next, control, dt=time_step)

        gravity = model.gravity.vector.numpy()[0]
        expected_velocity = time_step * gravity
        np.testing.assert_allclose(state_next.u_i.numpy()[0, :3], expected_velocity, rtol=0.0, atol=5.0e-4)
        expected_height = state_previous.q_i.numpy()[0, 2] + time_step * expected_velocity[2]
        self.assertAlmostEqual(float(state_next.q_i.numpy()[0, 2]), float(expected_height), places=5)
        self.assertTrue(np.isfinite(state_next.q_i.numpy()).all())
        self.assertTrue(np.isfinite(state_next.u_i.numpy()).all())

    def test_moreau_free_fall_uses_midpoint_configuration(self):
        """Compose LOX forward dynamics with Kamino's Moreau integrator."""
        model = ModelKamino.from_newton(build_box_on_plane(ground=False).finalize(device=self.device))
        config = self.make_config()
        config.integrator = "moreau"
        config.use_collision_detector = True
        solver = SolverKaminoImpl(model=model, config=config)
        state_previous = model.state()
        state_next = model.state()
        time_step = 0.01

        solver.step(state_previous, state_next, model.control(), dt=time_step)

        gravity = model.gravity.vector.numpy()[0]
        expected_velocity = time_step * gravity
        np.testing.assert_allclose(state_next.u_i.numpy()[0, :3], expected_velocity, rtol=0.0, atol=5.0e-4)
        expected_height = state_previous.q_i.numpy()[0, 2] + 0.5 * time_step * expected_velocity[2]
        self.assertAlmostEqual(float(state_next.q_i.numpy()[0, 2]), float(expected_height), places=6)

    def test_standard_integrators_apply_angular_velocity_damping(self):
        """Apply Kamino angular damping after a LOX forward-dynamics solve."""
        for integrator in ("euler", "moreau"):
            with self.subTest(integrator=integrator):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                build_box_on_plane(builder=builder, ground=False)
                model = ModelKamino.from_newton(builder.finalize(device=self.device))
                config = self.make_config()
                config.integrator = integrator
                config.use_collision_detector = True
                config.angular_velocity_damping = 0.5
                solver = SolverKaminoImpl(model=model, config=config)
                state_previous = model.state()
                state_next = model.state()
                state_previous.u_i.assign([wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)])
                time_step = 0.1

                solver.step(state_previous, state_next, model.control(), dt=time_step)

                expected_angular_velocity = 1.0 - config.angular_velocity_damping * time_step
                self.assertAlmostEqual(float(state_next.u_i.numpy()[0, 4]), expected_angular_velocity, places=5)

    def test_free_fall_uses_each_world_time_step(self):
        """Advance each rigid world with its configured device-side time step."""
        builder = build_box_on_plane(ground=False)
        build_box_on_plane(builder=builder, ground=False)
        model = ModelKamino.from_newton(builder.finalize(device=self.device))
        solver = SolverKaminoImpl(model=model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        time_step = np.asarray([0.01, 0.025], dtype=np.float32)
        model.time.dt.assign(time_step)
        model.time.inv_dt.assign(1.0 / time_step)

        solver.step(state_previous, state_next, model.control(), dt=None)

        gravity = model.gravity.vector.numpy()[: model.size.num_worlds]
        body_world = model.bodies.wid.numpy()
        expected_velocity = time_step[body_world, None] * gravity[body_world]
        np.testing.assert_allclose(state_next.u_i.numpy()[:, :3], expected_velocity, rtol=0.0, atol=5.0e-4)
        expected_height = state_previous.q_i.numpy()[:, 2] + time_step[body_world] * expected_velocity[:, 2]
        np.testing.assert_allclose(state_next.q_i.numpy()[:, 2], expected_height, rtol=0.0, atol=1.0e-6)

    def test_zero_mass_joint_anchor_is_prescribed_and_eliminated(self):
        """Use a prescribed anchor velocity in the dynamic child's joint bias."""
        model, anchor, child = _build_zero_mass_prismatic_anchor_model(device=self.device)
        config = self.make_config()
        config.use_collision_detector = False
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        prescribed_velocity = np.zeros((2, 6), dtype=np.float32)
        prescribed_velocity[anchor, 0] = 0.25
        state_previous.body_qd.assign(prescribed_velocity)

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

        np.testing.assert_allclose(state_next.body_qd.numpy()[anchor], prescribed_velocity[anchor], rtol=0.0, atol=0.0)
        self.assertAlmostEqual(float(state_next.body_qd.numpy()[child, 0]), 0.25, places=4)
        self.assertLess(float(state_next.body_qd.numpy()[child, 2]), -1.0e-3)
        self.assertTrue(np.isfinite(state_next.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_next.body_qd.numpy()).all())

    def test_prescribed_only_world_preserves_velocity_without_body_unknowns(self):
        """Preserve velocity in a prescribed-only world with no body unknowns."""
        model, body = _build_prescribed_only_model(device=self.device)
        config = self.make_config()
        config.use_collision_detector = False
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        prescribed_velocity = np.asarray([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        state_previous.body_qd.assign(prescribed_velocity)

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

        np.testing.assert_allclose(state_next.body_qd.numpy()[body], prescribed_velocity[body], rtol=0.0, atol=0.0)

    def test_body_flag_marks_massive_body_as_kinematic(self):
        """Prescribe a massive body's motion through BodyFlags.KINEMATIC."""
        model, body = _build_flagged_kinematic_model(device=self.device)
        config = self.make_config()
        config.use_collision_detector = False
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        prescribed_velocity = np.asarray([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        state_previous.body_qd.assign(prescribed_velocity)

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

        np.testing.assert_allclose(state_next.body_qd.numpy()[body], prescribed_velocity[body], rtol=0.0, atol=0.0)

    def test_kinematic_flag_change_requires_solver_rebuild(self):
        """Reject a runtime immovability change after Kamino has culled constraints."""
        model, _body = _build_flagged_kinematic_model(device=self.device)
        solver = SolverKamino(model, config=self.make_config())

        model.body_flags.fill_(int(newton.BodyFlags.DYNAMIC))
        with self.assertRaisesRegex(RuntimeError, "immovability.*recreate SolverKamino"):
            solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)

    def test_kinematic_joint_friction_is_projection_noop(self):
        """Ignore joint friction when both incident bodies are prescribed."""
        model = _build_revolute_dynamics_model(
            damping=0.0,
            friction=1.0,
            velocity=0.25,
            device=self.device,
        )
        model.body_flags.fill_(int(newton.BodyFlags.KINEMATIC))
        config = self.make_config()
        config.use_collision_detector = False
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

        np.testing.assert_allclose(
            state_next.body_qd.numpy(),
            state_previous.body_qd.numpy(),
            rtol=0.0,
            atol=0.0,
        )

    def test_kinematic_joint_effort_is_projection_noop(self):
        """Ignore bounded actuator effort when both incident bodies are prescribed."""
        model = _build_revolute_dynamics_model(
            damping=0.0,
            friction=0.0,
            velocity=0.25,
            target_ke=100.0,
            effort_limit=1.0,
            actuator_mode=newton.JointTargetMode.POSITION,
            device=self.device,
        )
        model.body_flags.fill_(int(newton.BodyFlags.KINEMATIC))
        config = self.make_config()
        config.use_collision_detector = False
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

        np.testing.assert_allclose(
            state_next.body_qd.numpy(),
            state_previous.body_qd.numpy(),
            rtol=0.0,
            atol=0.0,
        )

    def test_rod_world_parent_stretch_and_damping(self):
        """Restore stretch and reduce axial speed through rod damping."""
        model, _parent, child = _build_rod_model(stretch_stiffness=100.0, device=self.device)
        config = self.make_config()
        config.use_collision_detector = False
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        state_previous.body_q.assign([wp.transformf(wp.vec3f(0.0, 0.0, 0.1), wp.quat_identity(dtype=wp.float32))])

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

        self.assertLess(float(state_next.body_qd.numpy()[child, 2]), 0.0)

        speeds = []
        for damping in (0.0, 10.0):
            damped_model, _parent, damped_child = _build_rod_model(
                stretch_damping=damping,
                device=self.device,
            )
            damped_config = self.make_config()
            damped_config.use_collision_detector = False
            damped_solver = SolverKamino(damped_model, config=damped_config)
            damped_state_previous = damped_model.state()
            damped_state_next = damped_model.state()
            damped_state_previous.body_qd.assign([wp.spatial_vectorf(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)])
            damped_solver.step(
                damped_state_previous,
                damped_state_next,
                damped_model.control(),
                contacts=None,
                dt=0.01,
            )
            speeds.append(float(damped_state_next.body_qd.numpy()[damped_child, 2]))
        self.assertLess(speeds[1], speeds[0])

    def test_rod_accepts_implicit_single_world(self):
        """Accept implicit ownership for a rod in a single-world model."""
        model, _parent, child = _build_rod_model(
            explicit_world=False,
            stretch_stiffness=100.0,
            device=self.device,
        )
        config = self.make_config()
        config.use_collision_detector = False
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        state_previous.body_q.assign([wp.transformf(wp.vec3f(0.0, 0.0, 0.1), wp.quat_identity(dtype=wp.float32))])

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

        self.assertLess(float(state_next.body_qd.numpy()[child, 2]), 0.0)

    def test_rod_bend_and_twist_restore_rotation(self):
        """Restore isolated bend and twist rotations in their material modes."""
        cases = (
            ("bend", wp.vec3f(1.0, 0.0, 0.0), {"bend_stiffness": 20.0}, 3),
            ("twist", wp.vec3f(0.0, 0.0, 1.0), {"twist_stiffness": 20.0}, 5),
        )
        for name, axis, coefficients, velocity_index in cases:
            with self.subTest(mode=name):
                model, _parent, child = _build_rod_model(device=self.device, **coefficients)
                config = self.make_config()
                config.use_collision_detector = False
                solver = SolverKamino(model, config=config)
                state_previous = model.state()
                state_next = model.state()
                state_previous.body_q.assign(
                    [
                        wp.transformf(
                            wp.vec3f(0.0),
                            wp.quat_from_axis_angle(axis, 0.1),
                        )
                    ]
                )

                solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

                velocity = state_next.body_qd.numpy()[child]
                self.assertLess(float(velocity[velocity_index]), 0.0)
                self.assertTrue(np.isfinite(velocity).all())

        precurved_model, _parent, child = _build_rod_model(
            bend_stiffness=20.0,
            twist_stiffness=10.0,
            child_xform=wp.transformf(
                wp.vec3f(0.0),
                wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), 0.2),
            ),
            device=self.device,
        )
        precurved_config = self.make_config()
        precurved_config.use_collision_detector = False
        precurved_solver = SolverKamino(precurved_model, config=precurved_config)
        precurved_previous = precurved_model.state()
        precurved_next = precurved_model.state()

        precurved_solver.step(
            precurved_previous,
            precurved_next,
            precurved_model.control(),
            contacts=None,
            dt=0.01,
        )

        np.testing.assert_allclose(precurved_next.body_qd.numpy()[child], 0.0, atol=1.0e-7)

    def test_binary_rod_balances_wrenches_and_respects_enabled(self):
        """Balance binary rod wrenches and suppress disabled material forces."""
        model, parent, child = _build_rod_model(
            binary=True,
            shear_stiffness=50.0,
            device=self.device,
        )
        config = self.make_config(sparse_jacobian=True)
        config.use_collision_detector = False
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        state_previous.body_q.assign(
            [
                wp.transform_identity(dtype=wp.float32),
                wp.transformf(wp.vec3f(0.1, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
            ]
        )

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

        velocity = state_next.body_qd.numpy()
        self.assertGreater(float(velocity[parent, 0]), 0.0)
        self.assertLess(float(velocity[child, 0]), 0.0)
        np.testing.assert_allclose(velocity[parent, :3] + velocity[child, :3], 0.0, atol=1.0e-6)

        disabled_model, _parent, disabled_child = _build_rod_model(
            enabled=False,
            stretch_stiffness=100.0,
            device=self.device,
        )
        disabled_config = self.make_config()
        disabled_config.use_collision_detector = False
        disabled_solver = SolverKamino(disabled_model, config=disabled_config)
        disabled_previous = disabled_model.state()
        disabled_next = disabled_model.state()
        disabled_previous.body_q.assign([wp.transformf(wp.vec3f(0.0, 0.0, 0.1), wp.quat_identity(dtype=wp.float32))])
        disabled_solver.step(
            disabled_previous,
            disabled_next,
            disabled_model.control(),
            contacts=None,
            dt=0.01,
        )
        np.testing.assert_allclose(disabled_next.body_qd.numpy()[disabled_child], 0.0, atol=1.0e-7)

    def test_joint_damping_is_implicit_in_the_smooth_row(self):
        time_step = 0.1
        damping = 4.0
        initial_velocity = 2.0
        model = _build_revolute_dynamics_model(damping=damping, friction=0.0, velocity=initial_velocity)
        solver = SolverKamino(model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=time_step)

        expected = initial_velocity / (1.0 + time_step * damping)
        self.assertAlmostEqual(float(state_next.joint_qd.numpy()[0]), expected, places=4)

    def test_relaxed_joint_proximal_preserves_nonlinear_fixed_point(self):
        """Converge a relaxed joint correction to the exact candidate-pose constraint."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverKamino.register_custom_attributes(builder)
        builder.begin_world()
        inertia = wp.mat33f(0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4)
        parent = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
        child = builder.add_link(
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
            mass=1.0,
            inertia=inertia,
            lock_inertia=True,
        )
        root = builder.add_joint_revolute(parent=-1, child=parent, axis=newton.Axis.Y)
        fixed = builder.add_joint_fixed(
            parent=parent,
            child=child,
            parent_xform=wp.transformf(
                wp.vec3f(0.0, 0.0, 1.0),
                wp.quat_identity(dtype=wp.float32),
            ),
        )
        builder.add_articulation([root, fixed])
        builder.end_world()
        model = builder.finalize(device=self.device)

        config = self.make_config()
        config.use_collision_detector = False
        config.lox.fixed_iterations = True
        config.lox.max_iterations = 20
        config.lox.joint_proximal_relaxation = 0.5
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        velocity = np.zeros((model.body_count, 6), dtype=np.float32)
        velocity[parent] = (0.0, 0.0, 0.0, 0.0, 5.0, 0.0)
        velocity[child] = (8.0, -5.0, 3.0, 18.0, -12.0, 15.0)
        state_previous.body_qd.assign(velocity)

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.05)

        poses = state_next.body_q.numpy()
        parent_position = poses[parent, :3]
        parent_axis = poses[parent, 3:6]
        parent_scalar = poses[parent, 6]
        joint_offset = np.array((0.0, 0.0, 1.0), dtype=np.float32)
        rotated_offset = joint_offset + 2.0 * (
            parent_scalar * np.cross(parent_axis, joint_offset)
            + np.cross(parent_axis, np.cross(parent_axis, joint_offset))
        )
        position_error = poses[child, :3] - parent_position - rotated_offset
        np.testing.assert_allclose(position_error, 0.0, atol=1.0e-5)

    def test_joint_proximal_uses_frozen_frame_for_full_position_blocks(self):
        """Constrain offset anchors while a prescribed parent rotates substantially."""
        cases = (
            ("ball", 0.1, False),
            ("ball", 0.5, True),
            ("ball", 1.2, True),
            ("revolute", 0.5, True),
        )
        for joint_kind, angle, nonidentity_frames in cases:
            with self.subTest(joint_kind=joint_kind, angle=angle, nonidentity_frames=nonidentity_frames):
                model, parent, child, parent_xform, child_xform = _build_prescribed_rotating_joint_model(
                    angle=angle,
                    joint_kind=joint_kind,
                    nonidentity_frames=nonidentity_frames,
                    device=self.device,
                )
                config = self.make_config()
                config.use_collision_detector = False
                config.lox.fixed_iterations = True
                config.lox.joint_proximal_relaxation = 1.0
                solver = SolverKamino(model, config=config)
                state_previous = model.state()
                state_next = model.state()

                solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

                anchor_error = _joint_anchor_error(
                    state_next.body_q.numpy(),
                    parent,
                    child,
                    parent_xform,
                    child_xform,
                )
                self.assertTrue(np.isfinite(anchor_error).all())
                np.testing.assert_allclose(anchor_error, 0.0, rtol=0.0, atol=1.0e-6)

    def test_joint_proximal_accepts_finite_root_translation_transient(self):
        """Do not reject a large finite first trial solely for its absolute error."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverKamino.register_custom_attributes(builder)
        builder.begin_world()
        inertia = wp.mat33f(0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2)
        parent = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
        child = builder.add_link(
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
            mass=1.0,
            inertia=inertia,
            lock_inertia=True,
        )
        root = builder.add_joint_revolute(parent=-1, child=parent, axis=newton.Axis.Y)
        parent_xform = wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32))
        child_xform = wp.transformf(wp.vec3f(0.0), wp.quat_identity(dtype=wp.float32))
        joint = builder.add_joint_fixed(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
        )
        builder.add_articulation([root, joint])
        builder.end_world()
        model = builder.finalize(device=self.device)
        state_in = model.state()
        velocity = state_in.body_qd.numpy()
        velocity[:] = np.random.default_rng(1).normal(size=(2, 6)) / 0.01
        state_in.body_qd.assign(velocity)
        state_out = model.state()
        config = self.make_config(sparse_jacobian=True)
        config.use_collision_detector = False
        config.lox.fixed_iterations = True
        config.lox.max_iterations = 80
        config.lox.joint_proximal_relaxation = 1.0
        solver = SolverKamino(model, config=config)

        solver.step(state_in, state_out, model.control(), contacts=None, dt=0.01)

        solver_fd = solver._solver_kamino._solver_fd
        self.assertEqual(solver_fd.status.numpy()[0]["accepted"], 1)
        anchor_error = _joint_anchor_error(
            state_out.body_q.numpy(),
            parent,
            child,
            parent_xform,
            child_xform,
        )
        self.assertTrue(np.isfinite(anchor_error).all())
        self.assertLess(float(np.linalg.norm(anchor_error)), 1.0e-5)

    def test_joint_proximal_rejection_runs_in_conditional_loop(self):
        """Reject unsafe feedback inside capture-while without restarting."""
        model, _parent, _child, _parent_xform, _child_xform = _build_prescribed_rotating_joint_model(
            angle=0.5,
            nonidentity_frames=True,
            device=self.device,
        )
        config = self.make_config()
        config.use_collision_detector = False
        config.lox.fixed_iterations = False
        config.lox.use_graph_conditionals = True
        config.lox.max_iterations = 5
        config.lox.joint_proximal_relaxation = 1.0
        solver = SolverKamino(model, config=config)
        solver_fd = solver._solver_kamino._solver_fd
        adapter = solver_fd.rigid_adapter
        original_update = adapter.update_structural_multipliers_from_twist

        def inject_from_device_iteration(*args, **kwargs):
            original_update(*args, **kwargs)
            wp.launch(
                _inject_unsafe_on_iteration,
                dim=solver_fd.num_worlds,
                inputs=[solver_fd.iteration_count, 2],
                outputs=[solver_fd.rigid_adapter.world_structural_failed],
                device=self.device,
            )

        adapter.update_structural_multipliers_from_twist = inject_from_device_iteration
        solver.step(model.state(), model.state(), model.control(), contacts=None, dt=0.01)

        status = solver_fd.status.numpy()[0]
        self.assertEqual(status["accepted"], 0)
        self.assertEqual(status["failed"], 1)
        self.assertEqual(status["converged"], 0)
        self.assertEqual(status["iteration_limit"], 0)
        self.assertEqual(status["iterations"], 3)

    def test_joint_proximal_blocks_nonfinite_row_update(self):
        """Detect a candidate NaN before it reaches reactions or the assembled RHS."""
        model, _parent, child, _parent_xform, _child_xform = _build_prescribed_rotating_joint_model(
            angle=0.5,
            nonidentity_frames=True,
            device=self.device,
        )
        config = self.make_config()
        config.use_collision_detector = False
        config.lox.fixed_iterations = True
        config.lox.max_iterations = 3
        config.lox.joint_multiplier_projected_fraction = 1.0
        config.lox.joint_proximal_relaxation = 1.0
        solver = SolverKamino(model, config=config)
        state_in = model.state()
        state_out = model.state()
        solver_fd = solver._solver_kamino._solver_fd
        adapter = solver_fd.rigid_adapter
        original_evaluate = adapter._evaluate_structural_candidate_pose_residual
        original_update = adapter.update_structural_multipliers_from_twist
        supported_row = int(np.flatnonzero(adapter.structural_feedback_supported.numpy())[0])
        update_calls = 0
        row_reaction_after_second = None

        def inject_candidate_nan(*args):
            original_evaluate(*args)
            if int(solver_fd.iteration_count.numpy()[0]) == 2:
                feedback = args[-1].numpy()
                feedback[supported_row] = np.nan
                args[-1].assign(feedback)

        def inspect_row_update(*args, **kwargs):
            nonlocal update_calls, row_reaction_after_second
            original_update(*args, **kwargs)
            update_calls += 1
            reaction = adapter.structural_reaction.numpy()
            if update_calls == 2:
                row_reaction_after_second = reaction[supported_row]
            elif update_calls == 3:
                self.assertTrue(np.isfinite(reaction).all())
                self.assertTrue(np.isfinite(solver_fd.system.right_hand_side.numpy()).all())
                self.assertEqual(reaction[supported_row], row_reaction_after_second)

        adapter._evaluate_structural_candidate_pose_residual = inject_candidate_nan
        adapter.update_structural_multipliers_from_twist = inspect_row_update
        solver.step(state_in, state_out, model.control(), contacts=None, dt=0.01)

        status = solver_fd.status.numpy()[0]
        self.assertEqual(update_calls, 3)
        self.assertEqual(status["iterations"], 3)
        np.testing.assert_array_equal(adapter.structural_reaction.numpy(), 0.0)
        np.testing.assert_array_equal(solver_fd.splitting.splitting_dual_impulse.numpy(), 0.0)
        self.assertEqual(status["accepted"], 0)
        self.assertEqual(status["failed"], 1)
        np.testing.assert_allclose(state_out.body_q.numpy()[child], state_in.body_q.numpy()[child], atol=0.0)
        np.testing.assert_allclose(state_out.body_qd.numpy()[child], state_in.body_qd.numpy()[child], atol=0.0)

    def test_joint_proximal_handles_two_body_frames_and_orderings(self):
        """Keep fixed and revolute joint errors bounded across mass and storage order."""
        cases = (
            ("fixed", 1.0, False),
            ("fixed", 100.0, True),
            ("revolute", 100.0, False),
            ("revolute", 1.0, True),
        )
        for joint_kind, mass_ratio, reverse_body_order in cases:
            with self.subTest(
                joint_kind=joint_kind,
                mass_ratio=mass_ratio,
                reverse_body_order=reverse_body_order,
            ):
                model, parent, child, parent_xform, child_xform = _build_two_body_joint_model(
                    joint_kind=joint_kind,
                    mass_ratio=mass_ratio,
                    reverse_body_order=reverse_body_order,
                    device=self.device,
                )
                config = self.make_config()
                config.use_collision_detector = False
                config.lox.fixed_iterations = True
                config.lox.joint_proximal_relaxation = 1.0
                solver = SolverKamino(model, config=config)
                state_previous = model.state()
                state_next = model.state()
                velocity = np.zeros((model.body_count, 6), dtype=np.float32)
                velocity[parent] = (2.0, -1.0, 0.5, 0.0, 50.0, 0.0)
                velocity[child] = (-3.0, 2.0, -1.0, 20.0, -10.0, 15.0)
                state_previous.body_qd.assign(velocity)

                solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

                poses = state_next.body_q.numpy()
                position_error = np.linalg.norm(_joint_anchor_error(poses, parent, child, parent_xform, child_xform))
                angular_error = _joint_angular_error(
                    poses,
                    parent,
                    child,
                    parent_xform,
                    child_xform,
                    joint_kind,
                )
                self.assertTrue(math.isfinite(position_error))
                self.assertTrue(math.isfinite(angular_error))
                self.assertLess(position_error, 1.0e-4)
                self.assertLess(angular_error, 0.1)

    def test_joint_proximal_relaxations_share_position_fixed_point(self):
        """Reach the same physical anchor constraint across positive relaxations."""
        child_poses = []
        for relaxation in (0.25, 0.5, 1.0):
            with self.subTest(relaxation=relaxation):
                model, parent, child, parent_xform, child_xform = _build_prescribed_rotating_joint_model(
                    angle=0.5,
                    nonidentity_frames=True,
                    device=self.device,
                )
                config = self.make_config()
                config.use_collision_detector = False
                config.lox.fixed_iterations = True
                config.lox.max_iterations = 80
                config.lox.joint_proximal_relaxation = relaxation
                solver = SolverKamino(model, config=config)
                state_previous = model.state()
                state_next = model.state()

                solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

                poses = state_next.body_q.numpy()
                anchor_error = _joint_anchor_error(poses, parent, child, parent_xform, child_xform)
                np.testing.assert_allclose(anchor_error, 0.0, rtol=0.0, atol=1.0e-6)
                child_poses.append(poses[child])
        child_poses = np.asarray(child_poses)
        np.testing.assert_allclose(
            child_poses,
            np.broadcast_to(child_poses[0], child_poses.shape),
            rtol=0.0,
            atol=2.0e-6,
        )

    def test_joint_proximal_convergence_uses_physical_residual(self):
        """Converge only after the projected candidate satisfies the physical constraint."""
        model, parent, child, parent_xform, child_xform = _build_prescribed_rotating_joint_model(
            angle=0.5,
            nonidentity_frames=True,
            device=self.device,
        )
        config = self.make_config()
        config.use_collision_detector = False
        config.lox.joint_proximal_relaxation = 1.0
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

        solver_fd = solver._solver_kamino._solver_fd
        np.testing.assert_array_equal(solver_fd.world_converged.numpy(), True)
        anchor_error = _joint_anchor_error(
            state_next.body_q.numpy(),
            parent,
            child,
            parent_xform,
            child_xform,
        )
        np.testing.assert_allclose(anchor_error, 0.0, rtol=0.0, atol=config.lox.position_tolerance)

    def test_joint_proximal_support_uses_joint_row_semantics(self):
        """Enable nonlinear feedback only for validated full positional blocks."""
        fixed_model, _parent, _child, _parent_xform, _child_xform = _build_two_body_joint_model(
            joint_kind="fixed",
            mass_ratio=1.0,
            reverse_body_order=False,
            device=self.device,
        )
        fixed_solver = SolverKamino(fixed_model, config=self.make_config())
        fixed_supported = fixed_solver._solver_kamino._solver_fd.rigid_adapter.structural_feedback_supported.numpy()
        np.testing.assert_array_equal(
            fixed_supported,
            np.asarray((True, True, True, False, False, True, True, True, False, False, False)),
        )

        prismatic_model, _anchor, _child = _build_zero_mass_prismatic_anchor_model(device=self.device)
        prismatic_solver = SolverKamino(prismatic_model, config=self.make_config())
        prismatic_supported = (
            prismatic_solver._solver_kamino._solver_fd.rigid_adapter.structural_feedback_supported.numpy()
        )
        np.testing.assert_array_equal(prismatic_supported[-5:], False)

    def test_joint_proximal_keeps_physical_candidate_residual(self):
        """Report unsmoothed geometry for the projected rather than blended candidate."""
        model, parent, child, parent_xform, child_xform = _build_prescribed_rotating_joint_model(
            angle=0.5,
            nonidentity_frames=True,
            device=self.device,
        )
        config = self.make_config()
        config.use_collision_detector = False
        config.lox.fixed_iterations = True
        config.lox.max_iterations = 1
        config.lox.joint_proximal_relaxation = 1.0
        config.lox.joint_multiplier_projected_fraction = 0.0
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

        adapter = solver._solver_kamino._solver_fd.rigid_adapter
        physical_residual = adapter.structural_candidate_residual.numpy()
        update_residual = adapter.structural_update_residual.numpy()
        anchor_error = _joint_anchor_error(
            state_next.body_q.numpy(),
            parent,
            child,
            parent_xform,
            child_xform,
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(physical_residual)),
            float(np.linalg.norm(anchor_error)),
            places=6,
        )
        self.assertGreater(float(np.linalg.norm(physical_residual - update_residual)), 1.0e-7)

    def test_joint_proximal_transport_matches_frozen_linearization_locally(self):
        """Match the frozen structural residual and Jacobian at zero candidate increment."""
        model, _parent, child, _parent_xform, _child_xform = _build_prescribed_rotating_joint_model(
            angle=0.5,
            nonidentity_frames=True,
            device=self.device,
        )
        config = self.make_config()
        config.use_collision_detector = False
        config.lox.fixed_iterations = True
        config.lox.max_iterations = 1
        config.lox.joint_proximal_relaxation = 1.0
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        solver_fd = solver._solver_kamino._solver_fd
        adapter = solver_fd.rigid_adapter
        original_finish = solver_fd._finish_body_space_iteration
        checked = []

        def check_local_linearization(time_step, linearization):
            active = wp.ones(1, dtype=wp.bool, device=self.device)
            adapter._evaluate_structural_candidate_pose_residual(
                time_step,
                linearization,
                linearization,
                active,
                adapter.structural_candidate_residual,
                adapter.structural_feedback_residual,
            )
            np.testing.assert_allclose(
                adapter.structural_feedback_residual.numpy(),
                adapter.structural_residual.numpy(),
                rtol=0.0,
                atol=1.0e-7,
            )

            candidate = linearization.numpy().copy()
            delta = np.zeros_like(candidate)
            delta[child, :3] = (0.1, -0.2, 0.05)
            candidate += delta
            candidate = wp.array(candidate, dtype=wp.spatial_vectorf, device=self.device)
            adapter._evaluate_structural_candidate_pose_residual(
                time_step,
                candidate,
                linearization,
                active,
                adapter.structural_candidate_residual,
                adapter.structural_feedback_residual,
            )
            expected = adapter.structural_residual.numpy() + 0.01 * np.einsum(
                "ij,ij->i",
                adapter.structural_jacobian_second.numpy(),
                delta[adapter.structural_body_second_global.numpy()],
            )
            np.testing.assert_allclose(
                adapter.structural_feedback_residual.numpy(),
                expected,
                rtol=1.0e-3,
                atol=1.0e-8,
            )
            checked.append(True)
            original_finish(time_step, linearization)

        solver_fd._finish_body_space_iteration = check_local_linearization
        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)
        self.assertEqual(checked, [True])

    def test_position_drive_with_massless_fixed_child(self):
        """Track a target when the driven body has a massless fixed child."""
        cases = ((False, "euler"), (True, "euler"), (True, "moreau"))
        for include_massless_fixed_child, integrator in cases:
            with self.subTest(
                include_massless_fixed_child=include_massless_fixed_child,
                integrator=integrator,
            ):
                model = _build_massless_fixed_child_drive_model(
                    include_massless_fixed_child=include_massless_fixed_child,
                    device=self.device,
                )
                config = self.make_config()
                config.integrator = integrator
                config.use_collision_detector = integrator == "moreau"
                solver = SolverKamino(model, config=config)
                state_previous = model.state()
                state_next = model.state()
                control = model.control()
                control.joint_target_q.assign([0.2])

                for _ in range(240):
                    solver.step(state_previous, state_next, control, contacts=None, dt=1.0 / 600.0)
                    state_previous, state_next = state_next, state_previous

                self.assertGreater(float(state_previous.joint_q.numpy()[0]), 0.1)
                if include_massless_fixed_child:
                    body_q = state_previous.body_q.numpy()
                    body_qd = state_previous.body_qd.numpy()
                    np.testing.assert_allclose(body_qd[1], body_qd[0], rtol=0.0, atol=1.0e-4)
                    np.testing.assert_allclose(body_q[1, 3:], body_q[0, 3:], rtol=0.0, atol=1.0e-5)
                    np.testing.assert_allclose(body_q[1, :2], body_q[0, :2], rtol=0.0, atol=1.0e-5)
                    self.assertAlmostEqual(float(body_q[1, 2] - body_q[0, 2]), 0.1, places=5)

    def test_prescribed_contact_does_not_reject_dynamic_world(self):
        """Ignore contacts whose two incident bodies are prescribed."""
        model, _driven = _build_driven_link_with_grounded_fixed_base(device=self.device)
        shape_pairs = wp.array([(0, 1)], dtype=wp.vec2i, device=self.device)
        collision_pipeline = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            shape_pairs_filtered=shape_pairs,
        )
        contacts = collision_pipeline.contacts()
        config = self.make_config(sparse_jacobian=True)
        config.use_collision_detector = False
        config.lox.projection_method = "gauss_seidel"
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        control.joint_target_q.assign([0.2])
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_previous)

        collision_pipeline.collide(state_previous, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        solver.step(state_previous, state_next, control, contacts, dt=1.0 / 600.0)
        state_previous, state_next = state_next, state_previous

        for _ in range(119):
            collision_pipeline.collide(state_previous, contacts)
            solver.step(state_previous, state_next, control, contacts, dt=1.0 / 600.0)
            state_previous, state_next = state_next, state_previous

        self.assertGreater(float(state_previous.joint_q.numpy()[0]), 0.1)

    def test_joint_effort_limit_saturates_implicit_drive(self):
        """Clamp the internal implicit drive while retaining its full stiffness."""
        time_step = 0.1
        model = _build_revolute_dynamics_model(
            damping=0.0,
            friction=0.0,
            velocity=0.0,
            armature=1.0,
            target_ke=100.0,
            effort_limit=1.0,
            actuator_mode=newton.JointTargetMode.POSITION,
            device=self.device,
        )
        config = self.make_config()
        config.lox.max_iterations = 40
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        control.joint_target_q.assign([1.0])

        solver.step(state_previous, state_next, control, contacts=None, dt=time_step)

        self.assertAlmostEqual(float(state_next.joint_qd.numpy()[0]), 0.05, places=4)

    def test_joint_effort_limit_without_dynamic_row(self):
        """Apply a bounded implicit drive without armature or passive damping."""
        model = _build_revolute_dynamics_model(
            damping=0.0,
            friction=0.0,
            velocity=0.0,
            armature=0.0,
            target_ke=100.0,
            effort_limit=1.0,
            actuator_mode=newton.JointTargetMode.POSITION,
            device=self.device,
        )
        solver = SolverKamino(model, config=self.make_config())
        state_next = model.state()
        control = model.control()
        control.joint_target_q.assign([1.0])

        solver.step(model.state(), state_next, control, contacts=None, dt=0.1)

        self.assertAlmostEqual(float(state_next.joint_qd.numpy()[0]), 0.1, places=4)

    def test_unsaturated_joint_effort_preserves_implicit_drive(self):
        """Retain implicit PD behavior when a finite effort bound is inactive."""
        time_step = 0.1
        results = []
        for effort_limit in (math.inf, 1000.0):
            model = _build_revolute_dynamics_model(
                damping=0.0,
                friction=0.0,
                velocity=0.0,
                armature=1.0,
                target_ke=100.0,
                effort_limit=effort_limit,
                actuator_mode=newton.JointTargetMode.POSITION,
                device=self.device,
            )
            config = self.make_config()
            config.lox.max_iterations = 40
            solver = SolverKamino(model, config=config)
            state_previous = model.state()
            state_next = model.state()
            control = model.control()
            control.joint_target_q.assign([1.0])

            solver.step(state_previous, state_next, control, contacts=None, dt=time_step)
            results.append(float(state_next.joint_qd.numpy()[0]))

        self.assertGreater(results[0], 1.0)
        self.assertAlmostEqual(results[1], results[0], places=4)

    def test_joint_effort_limit_excludes_external_joint_force(self):
        """Keep external joint forces outside the actuator effort bound."""
        time_step = 0.1
        model = _build_revolute_dynamics_model(
            damping=0.0,
            friction=0.0,
            velocity=0.0,
            armature=1.0,
            target_ke=100.0,
            effort_limit=0.5,
            actuator_mode=newton.JointTargetMode.POSITION,
            device=self.device,
        )
        config = self.make_config()
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        control.joint_f.assign([20.0])

        solver.step(state_previous, state_next, control, contacts=None, dt=time_step)

        self.assertAlmostEqual(float(state_next.joint_qd.numpy()[0]), 0.975, places=4)

    def test_one_color_gauss_seidel_matches_rigid_jacobi(self):
        """Dispatch one-color rigid Gauss--Seidel through the exact Jacobi path."""
        results = {}
        for projection_method, max_colors in (("jacobi", 0), ("gauss_seidel", 1)):
            model = _build_revolute_dynamics_model(
                damping=0.0,
                friction=5.0,
                velocity=2.0,
                device=self.device,
            )
            config = self.make_config()
            config.lox.projection_method = projection_method
            config.lox.gauss_seidel_max_colors = max_colors
            solver = SolverKamino(model, config=config)
            state_next = model.state()

            solver.step(model.state(), state_next, model.control(), contacts=None, dt=0.1)

            results[projection_method] = (
                state_next.body_qd.numpy(),
                state_next.joint_qd.numpy(),
            )

        for colored_value, jacobi_value in zip(results["gauss_seidel"], results["jacobi"], strict=True):
            np.testing.assert_array_equal(colored_value, jacobi_value)

    def test_box_on_plane_projects_detected_contact(self):
        model = ModelKamino.from_newton(build_box_on_plane(ground=True).finalize(device=self.device))
        detector = CollisionDetector(
            model,
            config=kamino_config.CollisionDetectorConfig(pipeline="unified"),
        )
        contacts = detector.contacts
        solver = SolverKaminoImpl(
            model=model,
            contacts=contacts,
            config=self.make_config(compute_solution_metrics=True),
        )
        state_previous = model.state()
        state_next = model.state()

        solver.step(
            state_previous,
            state_next,
            model.control(),
            contacts=contacts,
            detector=detector,
            dt=0.01,
        )

        self.assertGreater(int(contacts.model_active_contacts.numpy()[0]), 0)
        self.assertGreaterEqual(float(contacts.reaction.numpy()[0, 2]), 0.0)
        self.assertGreaterEqual(float(contacts.velocity.numpy()[0, 2]), -2.0e-4)
        self.assertGreaterEqual(float(state_next.u_i.numpy()[0, 2]), -2.0e-4)
        self.assertGreater(float(state_next.w_i.numpy()[0, 2]), 0.0)
        self.assertGreaterEqual(int(contacts.mode.numpy()[0]), 0)
        self.assertTrue(np.isfinite(state_next.q_i.numpy()).all())
        self.assertTrue(np.isfinite(state_next.u_i.numpy()).all())
        np.testing.assert_array_equal(solver.solver_fd.world_failed.numpy(), [False])
        contact_residual_max = solver.solver_fd.contact_residual_max.numpy()
        self.assertTrue(np.isfinite(contact_residual_max).all())
        self.assertLessEqual(float(contact_residual_max[0]), 1.0e-4)
        metric_values = np.asarray(
            [
                solver.metrics.data.r_eom.numpy()[0],
                solver.metrics.data.r_v_plus.numpy()[0],
                solver.metrics.data.r_ncp_primal.numpy()[0],
                solver.metrics.data.r_ncp_dual.numpy()[0],
                solver.metrics.data.r_ncp_compl.numpy()[0],
                solver.metrics.data.r_vi_natmap.numpy()[0],
            ]
        )
        self.assertTrue(np.isfinite(metric_values).all())
        self.assertLess(float(metric_values[0]), 1.0e-3)
        self.assertLess(float(metric_values[2]), 1.0e-5)

    def test_product_space_structural_split_hinged_contact(self):
        model = ModelKamino.from_newton(build_boxes_hinged(z_offset=0.0, ground=True).finalize(device=self.device))
        detector = CollisionDetector(
            model,
            config=kamino_config.CollisionDetectorConfig(pipeline="unified"),
        )
        contacts = detector.contacts
        config = self.make_config(compute_solution_metrics=True)
        config.lox.max_iterations = 10
        config.lox.projection_iterations = 3
        solver = SolverKaminoImpl(model=model, contacts=contacts, config=config)
        state_previous = model.state()
        state_next = model.state()

        for _ in range(20):
            solver.step(
                state_previous,
                state_next,
                model.control(),
                contacts=contacts,
                detector=detector,
                dt=0.01,
            )
            state_previous, state_next = state_next, state_previous

        splitting = solver.solver_fd.splitting
        self.assertGreater(int(contacts.model_active_contacts.numpy()[0]), 0)
        np.testing.assert_array_equal(solver.solver_fd.world_failed.numpy(), [False])
        self.assertTrue(np.isfinite(splitting.residual_structural.numpy()).all())
        self.assertTrue(np.isfinite(splitting.residual_structural_projected.numpy()).all())
        self.assertLess(float(solver.metrics.data.r_cts_joints.numpy()[0]), 1.0e-3)
        self.assertLess(float(solver.metrics.data.r_eom.numpy()[0]), 1.0e-3)

    def test_joint_proximal_rejection_clears_apgd_contact_state(self):
        """Reject unsafe APGD trials without exporting contact warm starts."""
        model = ModelKamino.from_newton(build_boxes_hinged(z_offset=0.0, ground=True).finalize(device=self.device))
        detector = CollisionDetector(
            model,
            config=kamino_config.CollisionDetectorConfig(pipeline="unified"),
        )
        config = self.make_config()
        config.lox.fixed_iterations = True
        config.lox.max_iterations = 5
        config.lox.projection_method = "apgd"
        config.lox.projection_iterations = 3
        config.lox.joint_proximal_relaxation = 1.0
        solver = SolverKaminoImpl(model=model, contacts=detector.contacts, config=config)
        adapter = solver.solver_fd.rigid_adapter
        original_update = adapter.update_structural_multipliers_from_twist
        calls = 0

        def inject_unsafe(*args, **kwargs):
            nonlocal calls
            original_update(*args, **kwargs)
            calls += 1
            if calls == 3:
                adapter.world_structural_failed.fill_(1)

        adapter.update_structural_multipliers_from_twist = inject_unsafe
        state_in = model.state()
        state_out = model.state()
        solver.step(
            state_in,
            state_out,
            model.control(),
            contacts=detector.contacts,
            detector=detector,
            dt=0.01,
        )

        self.assertGreater(int(detector.contacts.model_active_contacts.numpy()[0]), 0)
        np.testing.assert_array_equal(solver.solver_fd.world_failed.numpy(), [True])
        np.testing.assert_array_equal(solver.solver_fd.iteration_count.numpy(), [3])
        np.testing.assert_array_equal(state_out.u_i.numpy(), state_in.u_i.numpy())
        np.testing.assert_array_equal(state_out.q_i.numpy(), state_in.q_i.numpy())
        np.testing.assert_array_equal(adapter.contact_reaction.numpy(), 0.0)
        np.testing.assert_array_equal(adapter.structural_reaction.numpy(), 0.0)
        np.testing.assert_array_equal(solver.solver_fd.splitting.splitting_dual_impulse.numpy(), 0.0)

    def test_joint_proximal_rejection_is_per_world(self):
        """Reject one packed world without perturbing a safe nonlinear neighbor."""

        def make_model():
            builder = build_boxes_hinged(z_offset=0.5, ground=False)
            build_boxes_hinged(builder=builder, z_offset=0.5, ground=False)
            return ModelKamino.from_newton(builder.finalize(device=self.device))

        def make_solver(model):
            config = self.make_config()
            config.lox.fixed_iterations = True
            config.lox.max_iterations = 5
            config.lox.joint_proximal_relaxation = 1.0
            return SolverKaminoImpl(model=model, config=config)

        rejected_model = make_model()
        rejected = make_solver(rejected_model)
        adapter = rejected.solver_fd.rigid_adapter
        original_update = adapter.update_structural_multipliers_from_twist
        calls = 0

        def inject_second_world_unsafe(*args, **kwargs):
            nonlocal calls
            original_update(*args, **kwargs)
            calls += 1
            if calls == 3:
                rejected.solver_fd.rigid_adapter.world_structural_failed.assign([0, 1])

        adapter.update_structural_multipliers_from_twist = inject_second_world_unsafe
        rejected_state = rejected_model.state()
        rejected.step(rejected_model.state(), rejected_state, rejected_model.control(), dt=0.01)

        reference_model = make_model()
        reference = make_solver(reference_model)
        reference_state = reference_model.state()
        reference.step(reference_model.state(), reference_state, reference_model.control(), dt=0.01)

        np.testing.assert_array_equal(
            rejected.solver_fd.world_failed.numpy(),
            [False, True],
        )
        np.testing.assert_array_equal(rejected.solver_fd.iteration_count.numpy(), [5, 3])
        first_world_bodies = rejected_model.bodies.wid.numpy() == 0
        np.testing.assert_array_equal(
            rejected.solver_fd.splitting.splitting_dual_impulse.numpy()[~first_world_bodies], 0.0
        )
        np.testing.assert_allclose(
            rejected_state.u_i.numpy()[first_world_bodies],
            reference_state.u_i.numpy()[first_world_bodies],
            rtol=2.0e-6,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            rejected_state.q_i.numpy()[first_world_bodies],
            reference_state.q_i.numpy()[first_world_bodies],
            rtol=2.0e-6,
            atol=2.0e-6,
        )

    def test_cartpole_projects_detected_joint_limit(self):
        model = ModelKamino.from_newton(build_cartpole(ground=False, limits=True).finalize(device=self.device))
        solver = SolverKaminoImpl(
            model=model,
            config=self.make_config(compute_solution_metrics=True),
        )
        state_previous = model.state()
        state_next = model.state()
        pose = state_previous.q_i.numpy()
        pose[:, 1] += 4.1
        state_previous.q_i.assign(pose)

        solver.step(state_previous, state_next, model.control(), dt=0.01)

        limits = solver._limits
        self.assertGreater(int(limits.model_active_limits.numpy()[0]), 0)
        self.assertGreater(float(limits.reaction.numpy()[0]), 0.0)
        self.assertGreaterEqual(float(limits.velocity.numpy()[0]), -2.0e-4)
        self.assertLess(float(state_next.dq_j.numpy()[0]), -0.09)
        self.assertTrue(np.isfinite(state_next.q_i.numpy()).all())
        self.assertTrue(np.isfinite(state_next.u_i.numpy()).all())
        np.testing.assert_array_equal(solver.solver_fd.world_failed.numpy(), [False])
        limit_residual_max = solver.solver_fd.limit_residual_max.numpy()
        self.assertTrue(np.isfinite(limit_residual_max).all())
        self.assertLessEqual(float(limit_residual_max[0]), 1.0e-4)
        metric_values = np.asarray(
            [
                solver.metrics.data.r_eom.numpy()[0],
                solver.metrics.data.r_v_plus.numpy()[0],
                solver.metrics.data.r_ncp_primal.numpy()[0],
                solver.metrics.data.r_ncp_dual.numpy()[0],
                solver.metrics.data.r_ncp_compl.numpy()[0],
                solver.metrics.data.r_vi_natmap.numpy()[0],
            ]
        )
        self.assertTrue(np.isfinite(metric_values).all())
        self.assertLess(float(metric_values[0]), 2.0e-2)
        self.assertLess(float(metric_values[2]), 1.0e-5)

    def test_cartpole_sustained_joint_force_remains_bounded(self):
        model = ModelKamino.from_newton(build_cartpole(ground=False, limits=True).finalize(device=self.device))
        solver = SolverKaminoImpl(model=model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        control.tau_j.assign(np.asarray([10.0, 0.0], dtype=np.float32))

        for _ in range(300):
            solver.step(state_previous, state_next, control, dt=0.001)
            state_previous, state_next = state_next, state_previous

        body_velocity = state_previous.u_i.numpy()
        joint_residual = solver._data.joints.r_j.numpy()
        self.assertTrue(np.isfinite(body_velocity).all())
        self.assertTrue(np.isfinite(joint_residual).all())
        self.assertLess(float(np.max(np.abs(body_velocity))), 100.0)
        self.assertLess(float(np.max(np.abs(joint_residual))), 1.0e-2)
        np.testing.assert_array_equal(solver.solver_fd.world_failed.numpy(), [False])

    def test_cuda_graph_capture_uses_conditional_loop(self):
        if not self.device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("CUDA conditional graph nodes require CUDA 12.4 or newer.")
        model = ModelKamino.from_newton(build_box_on_plane(ground=False).finalize(device=self.device))
        solver = SolverKaminoImpl(model=model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        control = model.control()

        with mock.patch.object(wp, "capture_while", wraps=wp.capture_while) as capture_while:
            with wp.ScopedCapture() as capture:
                solver.step(state_previous, state_next, control, dt=0.01)

        capture_while.assert_called_once()
        wp.capture_launch(capture.graph)
        self.assertTrue(np.isfinite(state_next.u_i.numpy()).all())

    def test_cuda_graph_capture_can_unroll_conditional_loop(self):
        """Unroll captured LOX splitting iterations when graph conditionals are disabled."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        model = ModelKamino.from_newton(build_box_on_plane(ground=False).finalize(device=self.device))
        config = self.make_config()
        config.lox.use_graph_conditionals = False
        solver = SolverKaminoImpl(model=model, config=config)
        state_previous = model.state()
        state_next = model.state()
        control = model.control()

        with mock.patch.object(wp, "capture_while", wraps=wp.capture_while) as capture_while:
            with wp.ScopedCapture() as capture:
                solver.step(state_previous, state_next, control, dt=0.01)

        capture_while.assert_not_called()
        wp.capture_launch(capture.graph)
        self.assertTrue(np.isfinite(state_next.u_i.numpy()).all())

    def test_cuda_graph_capture_joint_proximal_transport(self):
        """Replay transported structural feedback under CUDA graph capture."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        model, parent, child, parent_xform, child_xform = _build_prescribed_rotating_joint_model(
            angle=0.5,
            nonidentity_frames=True,
            device=self.device,
        )
        config = self.make_config()
        config.use_collision_detector = False
        config.lox.fixed_iterations = True
        config.lox.use_graph_conditionals = False
        config.lox.joint_proximal_relaxation = 1.0
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()

        with wp.ScopedCapture() as capture:
            solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

        wp.capture_launch(capture.graph)
        anchor_error = _joint_anchor_error(
            state_next.body_q.numpy(),
            parent,
            child,
            parent_xform,
            child_xform,
        )
        np.testing.assert_allclose(anchor_error, 0.0, rtol=0.0, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
