# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot H1
#
# Shows how to set up a simulation of a H1 articulation
# from a USD file using newton.ModelBuilder.add_usd().
#
# Command: python -m newton.examples robot_h1 --world-count 16
#          python -m newton.examples robot_h1 --dynamics-backend lox
#
###########################################################################

import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.fps = 50
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.dynamics_backend = args.dynamics_backend

        self.viewer = viewer

        self.device = wp.get_device()

        h1 = newton.ModelBuilder()
        if self.dynamics_backend == "lox":
            newton.solvers.SolverKamino.register_custom_attributes(h1)
        else:
            newton.solvers.SolverMuJoCo.register_custom_attributes(h1)
        h1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
        h1.default_shape_cfg.ke = 2.0e3
        h1.default_shape_cfg.kd = 1.0e2
        h1.default_shape_cfg.kf = 1.0e3
        h1.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("unitree_h1")
        asset_file = str(asset_path / "usd_structured" / "h1.usda")
        h1.add_usd(
            asset_file,
            ignore_paths=["/GroundPlane"],
            enable_self_collisions=False,
        )
        # approximate meshes for faster collision detection
        h1.approximate_meshes("bounding_box")

        for i in range(len(h1.joint_target_ke)):
            h1.joint_target_ke[i] = 150
            h1.joint_target_kd[i] = 5
            h1.joint_target_mode[i] = int(JointTargetMode.POSITION)

        builder = newton.ModelBuilder()
        builder.replicate(h1, self.world_count)

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.add_ground_plane()

        self.model = builder.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        use_mujoco_contacts = args.use_mujoco_contacts if args else False
        if self.dynamics_backend == "lox":
            if use_mujoco_contacts:
                raise ValueError("--use-mujoco-contacts requires --dynamics-backend mujoco")
            solver_config = newton.solvers.SolverKamino.Config.from_model(
                self.model,
                dynamics_solver="lox",
            )
            solver_config.use_collision_detector = True
            self.solver = newton.solvers.SolverKamino(self.model, config=solver_config)
            penalty_scales = self.solver.lox_joint_penalty_scale_seed(self.sim_dt)
            formatted_scales = ", ".join(f"{scale:.3g}" for scale in penalty_scales)
            print(f"[INFO] Seeded LOX joint penalty scales per world: [{formatted_scales}]")
        else:
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                iterations=100,
                ls_iterations=50,
                njmax=100,
                nconmax=210,
                use_mujoco_contacts=use_mujoco_contacts,
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Evaluate forward kinematics for collision detection
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)

        self.use_mujoco_contacts = use_mujoco_contacts
        if self.dynamics_backend == "lox":
            self.collision_pipeline = None
            self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
        elif use_mujoco_contacts:
            self.collision_pipeline = None
            self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
        else:
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((3.0, 3.0, 0.0))

        self.capture()

    def capture(self):
        self.graph = None
        if self.dynamics_backend == "lox":
            # Compile and allocate lazy solver data before APIC/CUDA capture.
            self.simulate()
            self.reset()
        if not wp.get_device().is_cuda:
            return
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        if self.dynamics_backend != "lox" and not self.use_mujoco_contacts:
            self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            contacts = None if self.dynamics_backend == "lox" else self.contacts
            self.solver.step(self.state_0, self.state_1, self.control, contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

        if self.dynamics_backend == "lox" or self.use_mujoco_contacts:
            self.solver.update_contacts(self.contacts, self.state_0)

    def reset(self):
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.solver.reset(self.state_0)
        self.state_1.assign(self.state_0)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )
        if self.dynamics_backend == "mujoco":
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                "all body velocities are small",
                lambda q, qd: max(abs(qd)) < 5e-3,
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_mujoco_contacts_arg(parser)
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
