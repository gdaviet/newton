# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Franka
#
# Simulates replicated fixed-base Franka FR3 arms tracking smooth joint-space
# targets. The scene has no contacts or hydroelastic geometry, making it a
# compact articulation-dynamics example for MuJoCo and Kamino LOX.
#
# Command: python -m newton.examples robot_franka --world-count 16
#          python -m newton.examples robot_franka --dynamics-backend lox
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode


@wp.kernel
def update_joint_targets(
    initial_targets: wp.array2d[wp.float32],
    amplitudes: wp.array2d[wp.float32],
    time: wp.array[wp.float32],
    dt: wp.float32,
    targets: wp.array2d[wp.float32],
):
    world = wp.tid()
    t = time[world] + dt
    time[world] = t
    for dof in range(targets.shape[1]):
        phase = 0.6 * float(dof) + 0.2 * float(world)
        targets[world, dof] = initial_targets[world, dof] + amplitudes[world, dof] * wp.sin(t + phase)


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.world_count = args.world_count
        self.dynamics_backend = args.dynamics_backend
        self.viewer = viewer
        self.device = wp.get_device()

        franka = newton.ModelBuilder()
        if self.dynamics_backend == "lox":
            newton.solvers.SolverKamino.register_custom_attributes(franka)
        else:
            newton.solvers.SolverMuJoCo.register_custom_attributes(franka)

        franka.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
            enable_self_collisions=False,
            hide_visuals=False,
        )

        initial_q = [-0.004, 0.024, 0.004, -2.368, 0.0, 2.392, 0.785, 0.04, 0.04]
        franka.joint_q[:9] = initial_q
        franka.joint_target_q[:9] = initial_q
        franka.joint_target_ke[:9] = [650.0] * 9
        franka.joint_target_kd[:9] = [100.0] * 9
        franka.joint_armature[:7] = [0.1] * 7
        franka.joint_armature[7:9] = [0.5] * 2
        franka.joint_effort_limit[:7] = [80.0] * 7
        franka.joint_effort_limit[7:9] = [20.0] * 2
        franka.joint_target_mode[:9] = [int(JointTargetMode.POSITION)] * 9

        builder = newton.ModelBuilder()
        builder.replicate(franka, self.world_count, spacing=(1.25, 1.25, 0.0))
        builder.add_ground_plane()

        self.model = builder.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        dof_count = self.model.joint_dof_count // self.world_count
        initial_targets = self.model.joint_q.numpy().reshape((self.world_count, dof_count))
        amplitudes = np.full_like(initial_targets, 0.15)
        amplitudes[:, 7:] = 0.0
        self.initial_targets = wp.array(initial_targets, dtype=wp.float32, device=self.device)
        self.amplitudes = wp.array(amplitudes, dtype=wp.float32, device=self.device)
        self.target_time = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.targets = self.control.joint_target_q.reshape((self.world_count, dof_count))

        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)

        if self.dynamics_backend == "lox":
            solver_config = newton.solvers.SolverKamino.Config.from_model(
                self.model,
                dynamics_solver="lox",
            )
            solver_config.use_collision_detector = False
            self.solver = newton.solvers.SolverKamino(self.model, config=solver_config)
        else:
            self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((1.25, 1.25, 0.0))

        self.capture()

    def capture(self):
        self.graph = None
        if self.dynamics_backend == "lox":
            # Compile and allocate lazy solver data before APIC/CUDA capture.
            self.simulate()
            self.reset()
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def reset(self):
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.solver.reset(self.state_0)
        self.state_1.assign(self.state_0)
        self.target_time.zero_()

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            wp.launch(
                update_joint_targets,
                dim=self.world_count,
                inputs=[self.initial_targets, self.amplitudes, self.target_time, self.sim_dt, self.targets],
                device=self.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        joint_q = self.state_0.joint_q.numpy()
        joint_lower = self.model.joint_limit_lower.numpy()
        joint_upper = self.model.joint_limit_upper.numpy()
        assert np.isfinite(joint_q).all(), "Franka joint coordinates must remain finite."
        assert np.all(joint_q >= joint_lower - 1.0e-3), "Franka exceeded a lower joint limit."
        assert np.all(joint_q <= joint_upper + 1.0e-3), "Franka exceeded an upper joint limit."

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=4)
        parser.add_argument(
            "--dynamics-backend",
            choices=("mujoco", "lox"),
            default="mujoco",
            help="Rigid-body dynamics backend.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
