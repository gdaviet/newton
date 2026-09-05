# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Kamino Joint Effort Limits
#
# Compares identical position-driven hinges lifting against gravity with finite
# and unlimited effort. The orange finite-effort arm settles below the target
# while the blue unlimited arm reaches the horizontal target marker.
#
# Command: python -m newton.examples kamino_joint_effort_limits
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    """Compare finite and unlimited LOX joint-drive effort."""

    FINITE_DOF = 0
    UNLIMITED_DOF = 1
    TARGET_ANGLE = 0.5 * math.pi
    TARGET_STIFFNESS = 300.0
    TARGET_DAMPING = 20.0
    FINITE_EFFORT_LIMIT = 5.0

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        newton.solvers.SolverKamino.register_custom_attributes(builder)

        arm_half_length = 0.55
        arm_cfg = newton.ModelBuilder.ShapeConfig(
            density=100.0,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        marker_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        setups = (
            ("finite effort", -0.8, self.FINITE_EFFORT_LIMIT, wp.vec3(0.95, 0.38, 0.08)),
            ("unlimited effort", 0.8, math.inf, wp.vec3(0.12, 0.45, 0.95)),
        )

        for label, x, effort_limit, color in setups:
            pivot = wp.vec3(x, 0.0, 0.85)
            body = builder.add_link(label=f"{label} arm")
            builder.add_shape_box(
                body,
                hx=0.07,
                hy=0.07,
                hz=arm_half_length,
                cfg=arm_cfg,
                color=color,
            )
            builder.add_shape_sphere(
                body,
                xform=wp.transform(p=wp.vec3(0.0, 0.0, -arm_half_length), q=wp.quat_identity()),
                radius=0.105,
                cfg=marker_cfg,
                as_site=True,
                color=color,
            )
            joint = builder.add_joint_revolute(
                parent=-1,
                child=body,
                parent_xform=wp.transform(p=pivot, q=wp.quat_identity()),
                child_xform=wp.transform(
                    p=wp.vec3(0.0, 0.0, arm_half_length),
                    q=wp.quat_identity(),
                ),
                axis=newton.Axis.Y,
                target_pos=self.TARGET_ANGLE,
                target_ke=self.TARGET_STIFFNESS,
                target_kd=self.TARGET_DAMPING,
                damping=1.0,
                armature=0.05,
                effort_limit=effort_limit,
                actuator_mode=newton.JointTargetMode.POSITION,
                label=label,
            )
            builder.add_articulation([joint], label=label)

            # Positive joint rotation carries the arm left toward this marker.
            builder.add_shape_box(
                -1,
                xform=wp.transform(
                    p=wp.vec3(x - arm_half_length, 0.0, pivot[2]),
                    q=wp.quat_identity(),
                ),
                hx=arm_half_length,
                hy=0.018,
                hz=0.018,
                cfg=marker_cfg,
                as_site=True,
                color=wp.vec3(0.35, 0.35, 0.35),
            )
            builder.add_shape_sphere(
                -1,
                xform=wp.transform(p=pivot, q=wp.quat_identity()),
                radius=0.11,
                cfg=marker_cfg,
                as_site=True,
                color=color,
            )

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.control.joint_target_q.assign([self.TARGET_ANGLE, self.TARGET_ANGLE])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)

        config = newton.solvers.SolverKamino.Config.from_model(
            self.model,
            dynamics_solver="lox",
        )
        config.use_collision_detector = False
        config.lox.max_iterations = 40
        self.solver = newton.solvers.SolverKamino(self.model, config=config)
        self._joint_effort_limits = np.array([self.FINITE_EFFORT_LIMIT, math.inf])

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(-0.5, -4.0, 0.8), pitch=0.0, yaw=90.0)

    def simulate(self):
        """Advance one frame."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, contacts=None, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance the example clock."""
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render both driven hinges."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        angles = self.state_0.joint_q.numpy()
        efforts = self._actuator_effort(angles, self.state_0.joint_qd.numpy())
        self.viewer.log_scalar("/target_position [rad]", self.TARGET_ANGLE)
        self.viewer.log_scalar("/orange_finite/position [rad]", angles[self.FINITE_DOF])
        self.viewer.log_scalar("/orange_finite/actuator_effort [N m]", efforts[self.FINITE_DOF])
        self.viewer.log_scalar("/blue_unlimited/position [rad]", angles[self.UNLIMITED_DOF])
        self.viewer.log_scalar("/blue_unlimited/actuator_effort [N m]", efforts[self.UNLIMITED_DOF])
        self.viewer.end_frame()

    def _actuator_effort(self, angles: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """Evaluate the instantaneous position-drive effort."""
        raw_effort = self.TARGET_STIFFNESS * (self.TARGET_ANGLE - angles) - self.TARGET_DAMPING * velocities
        return np.clip(raw_effort, -self._joint_effort_limits, self._joint_effort_limits)

    def test_final(self):
        """Verify gravity keeps the finite-effort hinge below its target."""
        angles = self.state_0.joint_q.numpy()
        efforts = self._actuator_effort(angles, self.state_0.joint_qd.numpy())
        finite_angle = float(angles[self.FINITE_DOF])
        unlimited_angle = float(angles[self.UNLIMITED_DOF])

        assert 0.2 * self.TARGET_ANGLE < finite_angle < 0.4 * self.TARGET_ANGLE, (
            "Finite-effort hinge did not remain gravity-limited below its target: "
            f"finite={finite_angle:.4f}, unlimited={unlimited_angle:.4f} rad."
        )
        assert abs(unlimited_angle - self.TARGET_ANGLE) < 0.05, (
            f"Unlimited-effort hinge did not reach its target: q={unlimited_angle:.4f} rad."
        )
        assert unlimited_angle - finite_angle > 0.55 * self.TARGET_ANGLE, (
            f"Effort-limit contrast was too small: finite={finite_angle:.4f}, unlimited={unlimited_angle:.4f} rad."
        )
        assert abs(float(efforts[self.FINITE_DOF]) - self.FINITE_EFFORT_LIMIT) < 1.0e-4, (
            f"Finite actuator did not remain saturated: effort={efforts[self.FINITE_DOF]:.4f} N m."
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
