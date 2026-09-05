# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Kamino DR Legs Pyramid
#
# Drives DR Legs forward with its trained walking policy and into a dynamic
# box pyramid built with the contact-pyramid example's stack generator.
#
# Command:
#   python -m newton.examples kamino_robot_dr_legs_pyramid --world-count 16
#
###########################################################################

from typing import Any

import numpy as np
import warp as wp
import yaml

import newton
import newton.examples
from newton.examples.contacts.example_pyramid import add_pyramid

PYRAMID_SIZE = 10
PYRAMID_CUBE_HALF = 0.02
PYRAMID_DISTANCE = 0.55
PYRAMID_COLOR = wp.vec3(0.82, 0.42, 0.12)

CLOTH_DISTANCE_FRACTION = 0.82
CLOTH_WIDTH = 0.7
CLOTH_HEIGHT = 0.6
CLOTH_DIM_X = 14
CLOTH_DIM_Y = 12

OBSERVATION_DIM = 94
ACTION_DIM = 12
# The available policy predates the later DR Legs design update.
DR_LEGS_POLICY_ASSET_REF = "261cd1f429619d8ef4f546bd788ab9dea906b5e1"
POLICY_JOINT_NAMES = tuple(
    f"j{joint}_{side}_{branch}"
    for side, branch in (("l", "i"), ("l", "o"), ("r", "i"), ("r", "o"))
    for joint in range(1, 10)
)


def _resolve_policy_joint_indices(
    builder: newton.ModelBuilder,
    *,
    target_coord_layout: bool,
) -> tuple[list[int], list[int], list[int]]:
    """Resolve policy coordinates and actuators independently of USD traversal order."""
    joint_indices_by_name: dict[str, int] = {}
    for joint_index, label in enumerate(builder.joint_label):
        name = label.rsplit("/", maxsplit=1)[-1]
        if name in POLICY_JOINT_NAMES:
            if name in joint_indices_by_name:
                raise ValueError(f"DR Legs joint name is not unique: {name}")
            joint_indices_by_name[name] = joint_index

    missing = [name for name in POLICY_JOINT_NAMES if name not in joint_indices_by_name]
    if missing:
        raise ValueError(f"DR Legs policy joints are missing: {missing}")

    policy_joint_indices = [joint_indices_by_name[name] for name in POLICY_JOINT_NAMES]
    policy_joint_coord_indices = [builder.joint_q_start[joint_index] for joint_index in policy_joint_indices]
    actuated_joint_indices = [
        joint_index
        for joint_index in policy_joint_indices
        if builder.joint_target_ke[builder.joint_qd_start[joint_index]] > 0.0
    ]
    actuated_dof_indices = [builder.joint_qd_start[joint_index] for joint_index in actuated_joint_indices]
    if len(actuated_dof_indices) != ACTION_DIM:
        raise ValueError(f"DR Legs policy expects {ACTION_DIM} actuated joints, found {len(actuated_dof_indices)}")
    target_starts = builder.joint_q_start if target_coord_layout else builder.joint_qd_start
    actuated_target_indices = [target_starts[joint_index] for joint_index in actuated_joint_indices]
    return policy_joint_coord_indices, actuated_dof_indices, actuated_target_indices


def _add_hanging_cloth(builder: newton.ModelBuilder, pyramid_distance: float) -> None:
    """Add an anchored cloth curtain between the robot and pyramid."""
    cloth_rotation = wp.quat_from_matrix(
        wp.mat33(
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        )
    )
    builder.add_cloth_grid(
        pos=wp.vec3(
            pyramid_distance * CLOTH_DISTANCE_FRACTION,
            -0.5 * CLOTH_WIDTH,
            0.04,
        ),
        rot=cloth_rotation,
        vel=wp.vec3(0.0),
        dim_x=CLOTH_DIM_X,
        dim_y=CLOTH_DIM_Y,
        cell_x=CLOTH_WIDTH / CLOTH_DIM_X,
        cell_y=CLOTH_HEIGHT / CLOTH_DIM_Y,
        mass=5.0e-4,
        fix_top=True,
        tri_ke=2.5e1,
        tri_ka=2.5e1,
        tri_kd=1.0e0,
        edge_ke=0.1,
        edge_kd=0.0,
        particle_radius=0.01,
        label="DR Legs interactive curtain",
    )


def _add_visual_mesh_box_colliders(builder: newton.ModelBuilder) -> list[int]:
    """Fit box colliders to robot bodies that only have visual meshes."""
    imported_shape_count = len(builder.shape_body)
    colliding_shapes = [
        shape for shape in range(imported_shape_count) if builder.shape_flags[shape] & newton.ShapeFlags.COLLIDE_SHAPES
    ]
    bodies_with_colliders = {builder.shape_body[shape] for shape in colliding_shapes}

    shape_cfg = newton.ModelBuilder.ShapeConfig()
    shape_cfg.density = 0.0
    shape_cfg.is_visible = False
    shape_cfg.collision_filter_parent = False

    supplemental_shapes = []
    for body in range(len(builder.body_label)):
        if body in bodies_with_colliders:
            continue

        visual_shape = next(
            (
                shape
                for shape in range(imported_shape_count)
                if builder.shape_body[shape] == body
                and builder.shape_type[shape] == newton.GeoType.MESH
                and builder.shape_source[shape] is not None
            ),
            None,
        )
        if visual_shape is None:
            continue

        vertices = np.asarray(builder.shape_source[visual_shape].vertices, dtype=np.float32)
        scale = np.asarray(tuple(builder.shape_scale[visual_shape]), dtype=np.float32)
        scaled_vertices = vertices * scale
        lower = np.min(scaled_vertices, axis=0)
        upper = np.max(scaled_vertices, axis=0)
        center = 0.5 * (lower + upper)
        half_extents = np.maximum(0.45 * (upper - lower), 3.0e-3)

        center_xform = wp.transform(p=wp.vec3(*center), q=wp.quat_identity())
        collider_xform = wp.transform_multiply(builder.shape_transform[visual_shape], center_xform)
        collider = builder.add_shape_box(
            body,
            xform=collider_xform,
            hx=float(half_extents[0]),
            hy=float(half_extents[1]),
            hz=float(half_extents[2]),
            cfg=shape_cfg,
        )

        # The policy was trained without robot self-collision. New shapes are
        # added after USD import, so extend that filtering explicitly.
        for robot_shape in (*colliding_shapes, *supplemental_shapes):
            builder.add_shape_collision_filter_pair(collider, robot_shape)
        supplemental_shapes.append(collider)

    return supplemental_shapes


@wp.kernel
def _mark_dynamic_soft_contact(
    contact_count: wp.array[wp.int32],
    contact_indices: wp.array[wp.vec3i],
    contact_shape: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    contact_seen: wp.array[wp.int32],
    full_surface_contact_seen: wp.array[wp.int32],
):
    contact = wp.tid()
    if contact < contact_count[0]:
        shape = contact_shape[contact]
        if shape >= 0 and shape_body[shape] >= 0:
            wp.atomic_max(contact_seen, 0, 1)
            if contact_indices[contact][1] >= 0:
                wp.atomic_max(full_surface_contact_seen, 0, 1)


@wp.func
def _projected_yaw(q: wp.quat) -> float:
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]
    return wp.atan2(2.0 * (qz * qw + qx * qy), qw * qw + qx * qx - qy * qy - qz * qz)


@wp.func
def _write_vec3(values: wp.array2d[float], world_index: int, offset: int, value: wp.vec3):
    values[world_index, offset] = value[0]
    values[world_index, offset + 1] = value[1]
    values[world_index, offset + 2] = value[2]


@wp.kernel
def _advance_policy_state(
    body_q: wp.array[wp.transformf],
    body_count_per_world: int,
    command_velocity: wp.vec2,
    env_dt: float,
    phase_rate: float,
    path_error_limit: float,
    phase: wp.array[float],
    path_position: wp.array[wp.vec2],
):
    world_index = wp.tid()
    root_body_index = world_index * body_count_per_world
    phase[world_index] = phase[world_index] + env_dt * phase_rate
    phase[world_index] = phase[world_index] - wp.floor(phase[world_index])

    root_position_3d = wp.transform_get_translation(body_q[root_body_index])
    root_position = wp.vec2(root_position_3d[0], root_position_3d[1])
    next_path_position = path_position[world_index] + env_dt * command_velocity
    path_error = next_path_position - root_position
    path_error_norm = wp.length(path_error)
    if path_error_norm > path_error_limit:
        next_path_position = root_position + path_error * (path_error_limit / path_error_norm)
    path_position[world_index] = next_path_position


@wp.kernel
def _compute_observation(
    body_q: wp.array[wp.transformf],
    body_qd: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    body_count_per_world: int,
    joint_coord_count_per_world: int,
    policy_joint_coord_indices: wp.array[int],
    actions: wp.array2d[float],
    previous_actions: wp.array2d[float],
    phase: wp.array[float],
    path_position: wp.array[wp.vec2],
    command_velocity: wp.vec2,
    command_yaw_rate: float,
    command_height: float,
    path_deviation_scale: float,
    height_error_scale: float,
    action_scale: float,
    observation: wp.array2d[float],
):
    world_index = wp.tid()
    root_body_index = world_index * body_count_per_world
    joint_coord_start = world_index * joint_coord_count_per_world
    root_transform = body_q[root_body_index]
    root_position = wp.transform_get_translation(root_transform)
    root_quat = wp.transform_get_rotation(root_transform)

    path_heading = 0.0
    path_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), path_heading)
    root_in_path = wp.mul(wp.quat_inverse(path_quat), root_quat)

    orientation = wp.quat_to_matrix(root_in_path)
    for i in range(3):
        for j in range(3):
            observation[world_index, i * 3 + j] = orientation[i, j]

    path_error_world = wp.vec3(
        root_position[0] - path_position[world_index][0],
        root_position[1] - path_position[world_index][1],
        0.0,
    )
    path_error = wp.quat_rotate_inv(path_quat, path_error_world)
    observation[world_index, 9] = path_error[0] / path_deviation_scale
    observation[world_index, 10] = path_error[1] / path_deviation_scale

    root_heading = _projected_yaw(root_in_path)
    heading_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), root_heading)
    heading_error = wp.quat_rotate_inv(heading_quat, -path_error)
    observation[world_index, 11] = heading_error[0] / path_deviation_scale
    observation[world_index, 12] = heading_error[1] / path_deviation_scale

    command_linear = wp.vec3(command_velocity[0], command_velocity[1], 0.0)
    command_angular = wp.vec3(0.0, 0.0, command_yaw_rate)
    _write_vec3(
        observation,
        world_index,
        13,
        wp.vec3(command_velocity[0], command_velocity[1], command_yaw_rate),
    )
    _write_vec3(observation, world_index, 16, wp.quat_rotate_inv(root_in_path, command_linear))
    _write_vec3(observation, world_index, 19, wp.quat_rotate_inv(root_in_path, command_angular))

    phase_angle = 2.0 * wp.pi * phase[world_index]
    observation[world_index, 22] = wp.cos(phase_angle)
    observation[world_index, 23] = wp.sin(phase_angle)
    observation[world_index, 24] = wp.cos(2.0 * phase_angle)
    observation[world_index, 25] = wp.sin(2.0 * phase_angle)

    root_linear_velocity = wp.quat_rotate_inv(root_quat, wp.spatial_top(body_qd[root_body_index]))
    root_angular_velocity = wp.quat_rotate_inv(root_quat, wp.spatial_bottom(body_qd[root_body_index]))
    _write_vec3(observation, world_index, 26, root_linear_velocity)
    _write_vec3(observation, world_index, 29, root_angular_velocity)

    observation[world_index, 32] = command_height
    observation[world_index, 33] = (root_position[2] - command_height) / height_error_scale

    for i in range(36):
        observation[world_index, 34 + i] = joint_q[joint_coord_start + policy_joint_coord_indices[i]]
    for i in range(ACTION_DIM):
        observation[world_index, 70 + i] = action_scale * actions[world_index, i]
        observation[world_index, 82 + i] = action_scale * previous_actions[world_index, i]


@wp.kernel
def _apply_policy_actions(
    actions: wp.array2d[float],
    actuated_target_indices: wp.array[int],
    joint_target_count_per_world: int,
    action_scale: float,
    joint_target_q: wp.array[float],
):
    world_index, action_index = wp.tid()
    target_index = world_index * joint_target_count_per_world + actuated_target_indices[action_index]
    joint_target_q[target_index] = action_scale * actions[world_index, action_index]


def _load_policy_checkpoint(path: str, device: str) -> tuple[Any, Any]:
    """Load the DR Legs training checkpoint and its PyTorch module.

    Args:
        path: Path to the ``rsl_rl`` checkpoint.
        device: PyTorch device string.

    Returns:
        The callable policy and imported PyTorch module.
    """
    try:
        import torch  # noqa: PLC0415
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The DR Legs policy requires PyTorch. Run this example with "
            "`uv run --extra torch-cu12 -m newton.examples kamino_robot_dr_legs_pyramid` "
            "(or use `torch-cu13` with a CUDA 13 environment)."
        ) from error

    try:
        policy = torch.jit.load(path, map_location=device)
        policy.eval()
        return policy, torch
    except RuntimeError:
        checkpoint = torch.load(path, map_location=device, weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    layer_indices = sorted({int(key.split(".")[1]) for key in state_dict if key.startswith("actor.")})
    layers = []
    for layer_number, layer_index in enumerate(layer_indices):
        weight = state_dict[f"actor.{layer_index}.weight"]
        bias = state_dict[f"actor.{layer_index}.bias"]
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(bias)
        layers.append(layer)
        if layer_number < len(layer_indices) - 1:
            layers.append(torch.nn.ELU())

    actor = torch.nn.Sequential(*layers).to(device)
    actor.eval()
    normalizer = checkpoint.get("obs_norm_state_dict")
    if normalizer is None:
        return actor, torch

    mean = normalizer["_mean"].to(device)
    std = normalizer["_std"].to(device)

    def policy(observation):
        return actor((observation - mean) / (std + 1.0e-2))

    return policy, torch


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.device = wp.get_device()
        self.sim_time = 0.0
        self.test_mode = args.test
        self.dynamics_solver = args.dynamics_solver
        self.world_count = args.world_count
        if self.world_count < 1:
            raise ValueError("world_count must be at least 1")

        asset_path = newton.utils.download_asset("disneyresearch", ref=DR_LEGS_POLICY_ASSET_REF)
        config_path = asset_path / "dr_legs" / "rl_policies" / "drlegs_walk.yaml"
        with open(config_path, encoding="utf-8") as config_file:
            self.policy_config = yaml.safe_load(config_file)

        self.sim_dt = float(self.policy_config["sim_dt"])
        self.sim_substeps = int(self.policy_config["control_decimation"])
        self.frame_dt = self.sim_dt * self.sim_substeps
        self.action_scale = float(self.policy_config["action_scale"])
        self.command_height = float(self.policy_config["standing_height"])
        self.command_velocity = wp.vec2(
            min(float(args.forward_speed), float(self.policy_config["vel_cmd_max"])),
            0.0,
        )
        self.pyramid_far_face = float(args.pyramid_distance + args.cube_half)

        world_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(world_builder)
        world_builder.default_shape_cfg.margin = 0.0
        if self.dynamics_solver == "lox":
            world_builder.default_shape_cfg.configure_sdf(force_sdf=True)

        model_path = asset_path / self.policy_config["usd_model"]
        world_builder.add_usd(
            str(model_path),
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
        self.supplemental_robot_colliders = _add_visual_mesh_box_colliders(world_builder)
        print(f"[INFO] Added {len(self.supplemental_robot_colliders)} supplemental DR Legs colliders")

        self.policy_joint_coord_indices, self.actuated_dof_indices, self.actuated_target_indices = (
            _resolve_policy_joint_indices(world_builder, target_coord_layout=True)
        )
        for dof_index in range(len(world_builder.joint_target_ke)):
            if dof_index in self.actuated_dof_indices:
                world_builder.joint_target_ke[dof_index] = float(self.policy_config["pd_kp"])
                world_builder.joint_target_kd[dof_index] = float(self.policy_config["pd_kd"])
                world_builder.joint_armature[dof_index] = float(self.policy_config["pd_armature"])
                # The policy configuration does not define actuator saturation; preserve its
                # unbounded training/deployment dynamics instead of inheriting USD drive limits.
                world_builder.joint_effort_limit[dof_index] = float("inf")
            else:
                world_builder.joint_target_ke[dof_index] = 0.0
                world_builder.joint_target_kd[dof_index] = 0.0
            world_builder.joint_damping[dof_index] = 0.0

        pyramid_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi)
        body_pose_offset_z = float(self.policy_config["body_pose_offset_z"])
        pyramid_shape_cfg = newton.ModelBuilder.ShapeConfig()
        pyramid_shape_cfg.gap = 0.25 * float(args.cube_half)
        pyramid_body_indices_per_world, pyramid_top_body_indices_per_world = add_pyramid(
            world_builder,
            args.pyramid_size,
            xform=wp.transform(
                # The policy's reference setup offsets every initial body pose
                # after model finalization. Compensate here so the cubes remain
                # seated on the ground while preserving that robot setup.
                p=wp.vec3(float(args.pyramid_distance), 0.0, -body_pose_offset_z),
                q=pyramid_rotation,
            ),
            cube_half=float(args.cube_half),
            color=PYRAMID_COLOR,
            shape_cfg=pyramid_shape_cfg,
        )

        if self.dynamics_solver == "lox":
            _add_hanging_cloth(world_builder, float(args.pyramid_distance))
            world_builder.color(include_bending=True)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.request_contact_attributes("force")
        builder.default_shape_cfg.margin = 0.0
        builder.replicate(world_builder, self.world_count)
        builder.add_ground_plane()

        self.model = builder.finalize(skip_validation_joints=True)
        self.body_count_per_world = self.model.body_count // self.world_count
        self.joint_coord_count_per_world = self.model.joint_coord_count // self.world_count
        body_offsets = np.arange(self.world_count, dtype=np.int32)[:, None] * self.body_count_per_world
        self.root_body_indices = body_offsets[:, 0]
        self.pyramid_body_indices = body_offsets + np.asarray(pyramid_body_indices_per_world, dtype=np.int32)
        self.pyramid_top_body_indices = body_offsets + np.asarray(pyramid_top_body_indices_per_world, dtype=np.int32)
        contacts_per_world = max(4096, 64 * len(pyramid_body_indices_per_world))
        self.model.rigid_contact_max = contacts_per_world * self.world_count
        body_q_initial = self.model.body_q.numpy()
        body_q_initial[:, 2] += body_pose_offset_z
        self.model.body_q.assign(body_q_initial)
        self.model.joint_friction.zero_()

        # Match the dynamics configuration used to deploy this policy. Contact
        # generation follows the contact-pyramid example's Newton pipeline.
        solver_config = newton.solvers.SolverKamino.Config(
            dynamics_solver=self.dynamics_solver,
            use_collision_detector=False,
        )
        solver_config.sparse_jacobian = True
        solver_config.use_fk_solver = False
        solver_config.integrator = "moreau"
        solver_config.constraints.alpha = 0.1
        solver_config.padmm.max_iterations = 200
        solver_config.padmm.primal_tolerance = 1.0e-4
        solver_config.padmm.dual_tolerance = 1.0e-4
        solver_config.padmm.compl_tolerance = 1.0e-4
        solver_config.padmm.eta = 1.0e-5
        solver_config.padmm.rho_0 = 0.05
        solver_config.padmm.use_acceleration = True
        solver_config.padmm.warmstart_mode = "containers"
        solver_config.padmm.contact_warmstart_method = "geom_pair_net_force"
        solver_config.padmm.use_graph_conditionals = False
        solver_config.collect_solver_info = False
        solver_config.compute_solution_metrics = False
        solver_config.constraints.gamma = 0.002
        if self.dynamics_solver == "lox":
            solver_config.integrator = "euler"
            solver_config.lox.max_iterations = args.max_iterations
            solver_config.lox.projection_iterations = args.projection_iterations
            solver_config.lox.deformable_preconditioner = args.deformable_preconditioner
            solver_config.lox.deformable_cr_iterations = args.deformable_cr_iterations
        self.solver = newton.solvers.SolverKamino(self.model, config=solver_config)

        if self.dynamics_solver == "lox":
            penalty_scales = self.solver.lox_joint_penalty_scale_seed(self.sim_dt)
            formatted_scales = ", ".join(f"{scale:.3g}" for scale in penalty_scales)
            print(f"[INFO] Seeded LOX joint penalty scales per world: [{formatted_scales}]")

        self.state_initial = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.solver.reset(self.state_initial, config=newton.solvers.SolverKamino.ResetConfig.preserve())
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase=args.broad_phase,
            contact_matching="latest",
            enable_rigid_soft_full_surface_contact=self.dynamics_solver == "lox",
            soft_contact_max=(
                max(8 * self.model.particle_count, 1024 * self.world_count) if self.model.particle_count > 0 else None
            ),
        )
        self.contacts = self.collision_pipeline.contacts()

        self.observation = wp.zeros((self.world_count, OBSERVATION_DIM), dtype=wp.float32, device=self.device)
        self.actions = wp.zeros((self.world_count, ACTION_DIM), dtype=wp.float32, device=self.device)
        self.previous_actions = wp.zeros((self.world_count, ACTION_DIM), dtype=wp.float32, device=self.device)
        self.phase = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.path_position = wp.array(
            [wp.vec2(0.0, 0.0)] * self.world_count,
            dtype=wp.vec2,
            device=self.device,
        )
        self.actuated_target_indices_wp = wp.array(
            self.actuated_target_indices,
            dtype=wp.int32,
            device=self.device,
        )
        self.policy_joint_coord_indices_wp = wp.array(
            self.policy_joint_coord_indices,
            dtype=wp.int32,
            device=self.device,
        )

        policy_path = asset_path / "dr_legs" / "rl_policies" / self.policy_config["policy_file"]
        torch_device = f"cuda:{self.device.ordinal}" if self.device.is_cuda else "cpu"
        self.policy, self.torch = _load_policy_checkpoint(str(policy_path), torch_device)
        self.observation_torch = wp.to_torch(self.observation)
        print(f"[INFO] Loaded DR Legs walking policy from {policy_path}")

        state_body_q_initial = self.state_0.body_q.numpy()
        self.initial_root_x = state_body_q_initial[self.root_body_indices, 0].copy()
        self.initial_pyramid_positions = state_body_q_initial[self.pyramid_body_indices, :3].copy()
        if self.model.particle_count > 0:
            particle_inverse_mass = self.model.particle_inv_mass.numpy()
            self.fixed_cloth_particle_indices = np.flatnonzero(particle_inverse_mass == 0.0)
            self.initial_fixed_cloth_positions = self.state_0.particle_q.numpy()[
                self.fixed_cloth_particle_indices
            ].copy()
        else:
            self.fixed_cloth_particle_indices = np.empty(0, dtype=np.int64)
            self.initial_fixed_cloth_positions = np.empty((0, 3), dtype=np.float32)
        self.dynamic_soft_contact_seen = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.dynamic_full_surface_contact_seen = wp.zeros(1, dtype=wp.int32, device=self.device)

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((1.5, 1.5, 0.0))
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(-0.35, -1.0, 0.55), -15.0, 45.0)

        self.graph = None
        self._prepare_capture()

    def _restore_initial_state(self):
        self.state_0.assign(self.state_initial)
        self.state_1.assign(self.state_initial)
        self.solver.reset(self.state_0)
        self.dynamic_soft_contact_seen.zero_()
        self.dynamic_full_surface_contact_seen.zero_()

    def _prepare_capture(self):
        # Compile and allocate lazy solver data before graph capture.
        self.simulate()
        self._restore_initial_state()

        if self.device.is_cpu or self.device.is_mempool_enabled:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            self._restore_initial_state()

    def simulate(self):
        for substep in range(self.sim_substeps):
            source = self.state_0 if substep % 2 == 0 else self.state_1
            target = self.state_1 if substep % 2 == 0 else self.state_0
            source.clear_forces()
            self.collision_pipeline.collide(source, self.contacts)
            if self.model.particle_count > 0:
                wp.launch(
                    _mark_dynamic_soft_contact,
                    dim=self.contacts.soft_contact_max,
                    inputs=[
                        self.contacts.soft_contact_count,
                        self.contacts.soft_contact_indices,
                        self.contacts.soft_contact_shape,
                        self.model.shape_body,
                    ],
                    outputs=[
                        self.dynamic_soft_contact_seen,
                        self.dynamic_full_surface_contact_seen,
                    ],
                    device=self.device,
                )
            self.viewer.apply_forces(source)
            self.solver.step(source, target, self.control, self.contacts, self.sim_dt)
            self.solver.update_contacts(self.contacts, target)

        if self.sim_substeps % 2 == 1:
            self.state_0.assign(self.state_1)

    def _policy_step(self):
        wp.launch(
            _advance_policy_state,
            dim=self.world_count,
            inputs=[
                self.state_0.body_q,
                self.body_count_per_world,
                self.command_velocity,
                self.frame_dt,
                1.0 / (2.0 * float(self.policy_config["contact_duration"])),
                float(self.policy_config["linear_path_error_limit"]),
                self.phase,
                self.path_position,
            ],
            device=self.device,
        )
        wp.launch(
            _compute_observation,
            dim=self.world_count,
            inputs=[
                self.state_0.body_q,
                self.state_0.body_qd,
                self.state_0.joint_q,
                self.body_count_per_world,
                self.joint_coord_count_per_world,
                self.policy_joint_coord_indices_wp,
                self.actions,
                self.previous_actions,
                self.phase,
                self.path_position,
                self.command_velocity,
                0.0,
                self.command_height,
                float(self.policy_config["path_deviation_scale"]),
                float(self.policy_config["height_error_scale"]),
                self.action_scale,
                self.observation,
            ],
            device=self.device,
        )

        with self.torch.no_grad():
            action_tensor = self.policy(self.observation_torch).contiguous()
        wp.copy(self.previous_actions, self.actions)
        wp.copy(self.actions, wp.from_torch(action_tensor))
        wp.launch(
            _apply_policy_actions,
            dim=(self.world_count, ACTION_DIM),
            inputs=[
                self.actions,
                self.actuated_target_indices_wp,
                len(self.control.joint_target_q) // self.world_count,
                self.action_scale,
                self.control.joint_target_q,
            ],
            device=self.device,
        )

    def step(self):
        self._policy_step()
        if self.graph is not None:
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
        """Verify locomotion, pyramid disturbance, and cloth validity."""
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        state_arrays = [
            ("body_q", body_q),
            ("body_qd", body_qd),
            ("joint_q", joint_q),
            ("joint_qd", joint_qd),
        ]
        if self.model.particle_count > 0:
            particle_q = self.state_0.particle_q.numpy()
            particle_qd = self.state_0.particle_qd.numpy()
            state_arrays.extend((("particle_q", particle_q), ("particle_qd", particle_qd)))
            fixed_cloth_positions = particle_q[self.fixed_cloth_particle_indices]
            max_anchor_error = np.max(
                np.linalg.norm(fixed_cloth_positions - self.initial_fixed_cloth_positions, axis=1),
                initial=0.0,
            )
            assert max_anchor_error < 1.0e-5, f"Cloth anchor error was {max_anchor_error:.3g} m"
            assert int(self.dynamic_soft_contact_seen.numpy()[0]) == 1, (
                "Cloth never contacted a dynamic robot or pyramid body"
            )
            assert int(self.dynamic_full_surface_contact_seen.numpy()[0]) == 1, (
                "Cloth never generated an edge or face contact with a dynamic robot or pyramid body"
            )

        for name, values in state_arrays:
            assert np.isfinite(values).all(), f"{name} contains non-finite values"

        root_q = body_q[self.root_body_indices]
        forward_distances = root_q[:, 0] - self.initial_root_x
        assert np.all(forward_distances > 0.15), (
            f"DR Legs minimum forward distance was {np.min(forward_distances):.3f} m"
        )
        assert np.all(forward_distances < 1.5), (
            f"DR Legs maximum forward distance was {np.max(forward_distances):.3f} m"
        )
        assert np.all(np.abs(root_q[:, 1]) < 0.75), (
            f"DR Legs maximum lateral deviation was {np.max(np.abs(root_q[:, 1])):.3f} m"
        )
        assert np.all(root_q[:, 0] > self.pyramid_far_face), (
            f"DR Legs minimum root x was {np.min(root_q[:, 0]):.3f} m at the pyramid"
        )

        pyramid_positions = body_q[self.pyramid_body_indices, :3]
        cube_displacements = np.linalg.norm(pyramid_positions - self.initial_pyramid_positions, axis=2)
        max_cube_displacements = np.max(cube_displacements, axis=1, initial=0.0)
        assert np.all(max_cube_displacements > 0.01), (
            "DR Legs did not disturb every pyramid; minimum per-world maximum cube displacement was "
            f"{np.min(max_cube_displacements):.4f} m"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_broad_phase_arg(parser)
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(broad_phase="sap")
        parser.set_defaults(num_frames=240)
        parser.set_defaults(world_count=1)
        parser.add_argument(
            "--dynamics-solver",
            choices=("lox", "padmm"),
            default="lox",
            help="Kamino dynamics solver to use.",
        )
        parser.add_argument(
            "--forward-speed",
            type=float,
            default=0.3,
            help="Forward policy command [m/s].",
        )
        parser.add_argument(
            "--pyramid-size",
            type=int,
            default=PYRAMID_SIZE,
            help="Number of rows in the pyramid base.",
        )
        parser.add_argument(
            "--cube-half",
            type=float,
            default=PYRAMID_CUBE_HALF,
            help="Pyramid cube half-extent [m].",
        )
        parser.add_argument(
            "--pyramid-distance",
            type=float,
            default=PYRAMID_DISTANCE,
            help="Distance from the initial robot position to the pyramid center [m].",
        )
        parser.add_argument(
            "--max-iterations",
            type=int,
            default=25,
            help="Maximum LOX splitting iterations per substep.",
        )
        parser.add_argument(
            "--projection-iterations",
            type=int,
            default=3,
            help="LOX unilateral projection sweeps per splitting iteration.",
        )
        parser.add_argument(
            "--deformable-preconditioner",
            choices=("incomplete_ldlt", "block_jacobi"),
            default="incomplete_ldlt",
            help="LOX cloth candidate preconditioner.",
        )
        parser.add_argument(
            "--deformable-cr-iterations",
            type=int,
            default=4,
            help="Fixed CR iterations per LOX cloth candidate solve.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
