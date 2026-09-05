# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Box Pyramid
#
# Builds pyramids of box-shaped cubes with a wrecking ball on a ramp
# to stress-test narrow-phase contact generation.
#
# Command: python -m newton.examples pyramid
#          python -m newton.examples pyramid --dynamics-backend lox --num-pyramids 1 --pyramid-size 5
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

DEFAULT_NUM_PYRAMIDS = 20
DEFAULT_PYRAMID_SIZE = 20
CUBE_HALF = 0.4
CUBE_SPACING = 2.1 * CUBE_HALF
PYRAMID_SPACING = 2.0 * CUBE_SPACING
Y_STACK = 15.0

WRECKING_BALL_RADIUS = 2.0
WRECKING_BALL_DENSITY_MULT = 100.0
RAMP_LENGTH = 20.0
RAMP_WIDTH = 5.0
RAMP_THICKNESS = 0.5

XPBD_ITERATIONS = 75
XPBD_CONTACT_RELAXATION = 0.8


def add_pyramid(
    builder: newton.ModelBuilder,
    pyramid_size: int,
    *,
    xform: wp.transformf | None = None,
    cube_half: float = CUBE_HALF,
    cube_spacing: float | None = None,
    color: wp.vec3f | None = None,
    shape_cfg: newton.ModelBuilder.ShapeConfig | None = None,
) -> tuple[list[int], list[int]]:
    """Add one pyramid of dynamic cubes.

    Args:
        builder: Model builder that receives the cube bodies and shapes.
        pyramid_size: Number of rows in the pyramid base.
        xform: World transform of the pyramid's bottom-center frame.
        cube_half: Cube half-extent [m].
        cube_spacing: Center spacing between adjacent cubes [m]. Defaults to
            ``2.1 * cube_half``.
        color: Optional display color shared by the cubes.
        shape_cfg: Optional physical properties shared by the cubes.

    Returns:
        A pair containing all cube body indices and the top-row body indices.
    """
    if xform is None:
        xform = wp.transform_identity()
    if cube_spacing is None:
        cube_spacing = 2.1 * cube_half

    body_indices = []
    top_body_indices = []
    for level in range(pyramid_size):
        num_cubes_in_row = pyramid_size - level
        row_width = (num_cubes_in_row - 1) * cube_spacing
        for i in range(num_cubes_in_row):
            local_xform = wp.transform(
                p=wp.vec3(-row_width / 2 + i * cube_spacing, 0.0, level * cube_spacing + cube_half),
                q=wp.quat_identity(),
            )
            body = builder.add_body(xform=wp.transform_multiply(xform, local_xform))
            builder.add_shape_box(
                body,
                hx=cube_half,
                hy=cube_half,
                hz=cube_half,
                color=color,
                cfg=shape_cfg,
            )
            body_indices.append(body)
            if level == pyramid_size - 1:
                top_body_indices.append(body)

    return body_indices, top_body_indices


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.test_mode = args.test
        self.world_count = args.world_count
        self.dynamics_backend = args.dynamics_backend

        num_pyramids = args.num_pyramids
        pyramid_size = args.pyramid_size

        builder = newton.ModelBuilder()

        builder.default_shape_cfg.mu = 0.2

        if self.dynamics_backend == "lox":
            newton.solvers.SolverKamino.register_custom_attributes(builder)
        builder.add_shape_plane(xform=wp.transform_identity(), width=0.0, length=0.0)

        box_body_indices = []
        top_body_indices = []
        pyramid_height = pyramid_size * CUBE_SPACING

        for pyramid in range(num_pyramids):
            y_offset = pyramid * PYRAMID_SPACING
            pyramid_bodies, pyramid_top_bodies = add_pyramid(
                builder,
                pyramid_size,
                xform=wp.transform(p=wp.vec3(0.0, Y_STACK - y_offset, 0.0), q=wp.quat_identity()),
            )
            box_body_indices.extend(pyramid_bodies)
            top_body_indices.extend(pyramid_top_bodies)

        self.box_count = len(box_body_indices)
        self.top_body_indices = top_body_indices
        print(f"Built {num_pyramids} pyramids x {pyramid_size} rows = {self.box_count} boxes")

        if not self.test_mode:
            # Wrecking ball
            ramp_height = 8.4
            ramp_angle = float(np.arctan2(ramp_height, RAMP_LENGTH))
            ball_x = 0.0
            ball_y = Y_STACK + RAMP_LENGTH * 0.9
            ball_z = ramp_height + WRECKING_BALL_RADIUS + 0.1

            body_ball = builder.add_body(
                xform=wp.transform(p=wp.vec3(ball_x, ball_y, ball_z), q=wp.quat_identity()),
            )
            ball_cfg = newton.ModelBuilder.ShapeConfig()
            ball_cfg.density = builder.default_shape_cfg.density * WRECKING_BALL_DENSITY_MULT
            builder.add_shape_sphere(body_ball, radius=WRECKING_BALL_RADIUS, cfg=ball_cfg)

            # Ramp (static)
            ramp_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(ramp_angle))
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(
                    p=wp.vec3(ball_x, Y_STACK + RAMP_LENGTH / 2, ramp_height / 2),
                    q=ramp_quat,
                ),
                hx=RAMP_WIDTH / 2,
                hy=RAMP_LENGTH / 2,
                hz=RAMP_THICKNESS / 2,
            )

        if self.world_count > 1:
            main_builder = newton.ModelBuilder()
            main_builder.replicate(builder, world_count=self.world_count)
            self.model = main_builder.finalize()
        else:
            self.model = builder.finalize()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase=args.broad_phase,
        )

        if self.dynamics_backend == "lox":
            solver_config = newton.solvers.SolverKamino.Config.from_model(
                self.model,
                dynamics_solver="lox",
            )
            solver_config.lox.max_iterations = args.admm_iterations
            self.solver = newton.solvers.SolverKamino(self.model, config=solver_config)
        else:
            self.solver = newton.solvers.SolverXPBD(
                self.model,
                iterations=XPBD_ITERATIONS,
                rigid_contact_relaxation=XPBD_CONTACT_RELAXATION,
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.top_initial_positions = self.state_0.body_q.numpy()[:, :3].copy()

        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)

        cam_dist = max(pyramid_height, num_pyramids * PYRAMID_SPACING * 0.3)
        self.viewer.set_camera(
            pos=wp.vec3(cam_dist, -cam_dist, cam_dist * 0.4),
            pitch=-15.0,
            yaw=135.0,
        )

        self.capture()

    def capture(self):
        self.graph = None
        if self.dynamics_backend == "lox":
            # Compile and allocate lazy solver data before APIC/CUDA capture.
            self.simulate()
            self.solver.reset(self.state_0)
            self.state_1.assign(self.state_0)
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.viewer.apply_forces(self.state_0)
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
        if self.contacts is not None:
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify pyramid top cubes remain near their initial positions.

        In test mode the wrecking ball is omitted so the pyramids should
        settle under gravity without toppling.  Each top cube must stay
        within ``max_displacement`` of its initial position.
        """
        body_q = self.state_0.body_q.numpy()
        max_displacement = 0.5  # [m]
        for idx in self.top_body_indices:
            current_pos = body_q[idx, :3]
            initial_pos = self.top_initial_positions[idx]
            displacement = np.linalg.norm(current_pos - initial_pos)
            assert displacement < max_displacement, (
                f"Top cube body {idx}: displaced {displacement:.4f} m (max allowed {max_displacement:.4f} m)"
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=1)
        newton.examples.add_broad_phase_arg(parser)
        parser.set_defaults(broad_phase="sap")
        parser.add_argument(
            "--dynamics-backend",
            choices=("xpbd", "lox"),
            default="xpbd",
            help="Rigid-body dynamics backend.",
        )
        parser.add_argument(
            "--admm-iterations",
            type=int,
            default=25,
            help="Maximum LOX ADMM iterations per simulation substep.",
        )
        parser.add_argument(
            "--num-pyramids",
            type=int,
            default=DEFAULT_NUM_PYRAMIDS,
            help="Number of pyramids to build.",
        )
        parser.add_argument(
            "--pyramid-size",
            type=int,
            default=DEFAULT_PYRAMID_SIZE,
            help="Number of rows in each pyramid base.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
