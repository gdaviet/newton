# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Parachute
#
# A flat circular cloth canopy carries DR Legs through four Y-shaped cable
# groups attached to its pelvis. VBD or LOX simulates the cloth, cables,
# articulated robot, and all contacts. Once the robot has landed and settled,
# its trained walking policy starts and drives it forward with the parachute
# still attached.
#
# Command:
#   uv run --extra torch-cu12 -m newton.examples cloth_parachute
#   uv run --extra torch-cu12 -m newton.examples cloth_parachute --solver vbd
#
###########################################################################

from __future__ import annotations

import itertools
import math

import numpy as np
import warp as wp
import yaml
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import newton
import newton.examples
from newton.examples.kamino.example_kamino_robot_dr_legs_pyramid import (
    ACTION_DIM,
    DR_LEGS_POLICY_ASSET_REF,
    OBSERVATION_DIM,
    _add_visual_mesh_box_colliders,
    _advance_policy_state,
    _apply_policy_actions,
    _compute_observation,
    _load_policy_checkpoint,
    _mark_dynamic_soft_contact,
    _resolve_policy_joint_indices,
)


@wp.kernel
def apply_aerodynamic_drag(
    triangle_start: int,
    air_velocity: wp.vec3,
    drag_coefficient: float,
    skin_drag_coefficient: float,
    tri_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
):
    """Apply one-sided quadratic air drag along each triangle normal."""
    triangle = triangle_start + wp.tid()
    i = tri_indices[triangle, 0]
    j = tri_indices[triangle, 1]
    k = tri_indices[triangle, 2]

    area_vector = 0.5 * wp.cross(particle_q[j] - particle_q[i], particle_q[k] - particle_q[i])
    area = wp.length(area_vector)
    if area > 1.0e-8:
        normal = area_vector / area
        face_velocity = (particle_qd[i] + particle_qd[j] + particle_qd[k]) / 3.0
        relative_velocity = air_velocity - face_velocity
        normal_velocity = wp.dot(relative_velocity, normal)
        normal_speed = wp.max(normal_velocity, 0.0)
        tangent_velocity = relative_velocity - normal_velocity * normal
        tangent_speed = wp.length(tangent_velocity)
        air_density = 1.225
        pressure_force = drag_coefficient * normal_speed * normal_speed * normal
        skin_force = skin_drag_coefficient * tangent_speed * tangent_velocity
        vertex_force = 0.5 * air_density * area * (pressure_force + skin_force) / 3.0
        wp.atomic_add(particle_f, i, vertex_force)
        wp.atomic_add(particle_f, j, vertex_force)
        wp.atomic_add(particle_f, k, vertex_force)


@wp.kernel
def apply_cable_wind_drag(
    cable_body_ids: wp.array[wp.int32],
    cable_projected_areas: wp.array[float],
    air_velocity: wp.vec3,
    drag_coefficient: float,
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
):
    """Apply quadratic airflow drag at each cable segment center of mass."""
    cable = wp.tid()
    body = cable_body_ids[cable]
    relative_velocity = air_velocity - wp.spatial_top(body_qd[body])
    speed = wp.length(relative_velocity)
    if speed > 1.0e-6:
        air_density = 1.225
        force = 0.5 * air_density * drag_coefficient * cable_projected_areas[cable] * speed * relative_velocity
        wp.atomic_add(body_f, body, wp.spatial_vector(force, wp.vec3(0.0)))


@wp.kernel
def normalize_scaled_robot_observation(
    robot_scale: float,
    observation: wp.array2d[float],
):
    """Map length-dependent observations back to the policy's training scale."""
    world_index = wp.tid()
    observation[world_index, 26] /= robot_scale
    observation[world_index, 27] /= robot_scale
    observation[world_index, 28] /= robot_scale
    observation[world_index, 32] /= robot_scale


def _create_canopy_mesh(
    radial_resolution: int,
    angular_resolution: int,
    radius: float,
    height: float,
) -> tuple[list[tuple[float, float, float]], list[int], list[tuple[float, float, float]]]:
    """Create a flat circular canopy with eight enclosed rim eyelets."""
    eyelet_count = 8
    radial_step = radius / radial_resolution
    angular_step = 2.0 * math.pi / angular_resolution
    eyelet_sectors = [i * angular_resolution // eyelet_count for i in range(eyelet_count)]
    eyelet_inner_ring = radial_resolution - 2

    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, height)]
    for ring in range(1, radial_resolution + 1):
        ring_radius = ring * radial_step
        for sector in range(angular_resolution):
            angle = sector * angular_step
            vertices.append((ring_radius * math.cos(angle), ring_radius * math.sin(angle), height))

    def ring_vertex(ring: int, sector: int) -> int:
        return 1 + (ring - 1) * angular_resolution + sector % angular_resolution

    indices: list[int] = []
    for sector in range(angular_resolution):
        indices.extend((0, ring_vertex(1, sector), ring_vertex(1, sector + 1)))

    eyelet_sector_set = set(eyelet_sectors)
    for outer_ring in range(2, radial_resolution + 1):
        inner_ring = outer_ring - 1
        for sector in range(angular_resolution):
            if inner_ring == eyelet_inner_ring and sector in eyelet_sector_set:
                continue
            inner_0 = ring_vertex(inner_ring, sector)
            inner_1 = ring_vertex(inner_ring, sector + 1)
            outer_0 = ring_vertex(outer_ring, sector)
            outer_1 = ring_vertex(outer_ring, sector + 1)
            indices.extend((inner_0, outer_0, outer_1, inner_0, outer_1, inner_1))

    eyelet_radius = (eyelet_inner_ring + 0.5) * radial_step
    openings: list[tuple[float, float, float]] = []
    for sector in eyelet_sectors:
        angle = (sector + 0.5) * angular_step
        openings.append((eyelet_radius * math.cos(angle), eyelet_radius * math.sin(angle), height))

    return vertices, indices, openings


def _resample_curve(points: list[np.ndarray], target_length: float) -> list[wp.vec3]:
    """Resample a polyline at approximately uniform arc-length intervals."""
    control = np.asarray(points, dtype=np.float64)
    edge_lengths = np.linalg.norm(control[1:] - control[:-1], axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
    sample_count = max(2, int(math.ceil(float(cumulative[-1]) / target_length)) + 1)
    distances = np.linspace(0.0, float(cumulative[-1]), sample_count)

    samples = np.empty((sample_count, 3), dtype=np.float64)
    for axis in range(3):
        samples[:, axis] = np.interp(distances, cumulative, control[:, axis])
    return [wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in samples]


def _create_y_cable_graph(
    root: tuple[float, float, float],
    junction: tuple[float, float, float],
    eyelet_guides: tuple[tuple[float, float, float], tuple[float, float, float]],
    ends: tuple[tuple[float, float, float], tuple[float, float, float]],
    target_length: float,
) -> tuple[list[wp.vec3], list[tuple[int, int]], list[int]]:
    """Create a Y graph routed vertically through two eyelet centers."""
    root_np = np.asarray(root, dtype=np.float64)
    junction_np = np.asarray(junction, dtype=np.float64)
    trunk = _resample_curve([root_np, junction_np], target_length)

    nodes = list(trunk)
    edges = [(i, i + 1) for i in range(len(trunk) - 1)]
    junction_node = len(trunk) - 1
    endpoint_edges: list[int] = []
    for eyelet_guide, end in zip(eyelet_guides, ends, strict=True):
        branch = [trunk[-1]]
        control = (junction_np, np.asarray(eyelet_guide, dtype=np.float64), np.asarray(end, dtype=np.float64))
        for segment_start, segment_end in itertools.pairwise(control):
            branch.extend(_resample_curve([segment_start, segment_end], target_length)[1:])
        previous_node = junction_node
        for point in branch[1:]:
            nodes.append(point)
            current_node = len(nodes) - 1
            edges.append((previous_node, current_node))
            previous_node = current_node
        endpoint_edges.append(len(edges) - 1)

    return nodes, edges, endpoint_edges


def _add_masonry_arch(
    builder: newton.ModelBuilder,
    center: wp.vec3,
    scale: float,
    shape_cfg: newton.ModelBuilder.ShapeConfig,
) -> tuple[list[int], list[int]]:
    """Build a loose masonry arch from independent rigid blocks."""
    block_half_x = 0.05 * scale
    pier_half_y = 0.065 * scale
    block_half_z = 0.045 * scale
    opening_half_width = 0.32 * scale
    pier_levels = 6
    spring_height = 2.0 * block_half_z * pier_levels
    inner_radius = opening_half_width
    outer_radius = opening_half_width + 2.0 * block_half_z
    arch_radius = 0.5 * (inner_radius + outer_radius)
    arch_block_count = 11
    half_sector_angle = 0.495 * math.pi / arch_block_count

    inner_y = inner_radius * math.sin(half_sector_angle)
    inner_z = inner_radius * math.cos(half_sector_angle) - arch_radius
    outer_y = outer_radius * math.sin(half_sector_angle)
    outer_z = outer_radius * math.cos(half_sector_angle) - arch_radius
    voussoir_vertices = np.array(
        [
            (-block_half_x, -inner_y, inner_z),
            (-block_half_x, -outer_y, outer_z),
            (-block_half_x, outer_y, outer_z),
            (-block_half_x, inner_y, inner_z),
            (block_half_x, -inner_y, inner_z),
            (block_half_x, -outer_y, outer_z),
            (block_half_x, outer_y, outer_z),
            (block_half_x, inner_y, inner_z),
        ],
        dtype=np.float32,
    )
    face_quads = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 3, 7, 4), (1, 5, 6, 2), (0, 4, 5, 1), (3, 2, 6, 7))
    voussoir_indices: list[int] = []
    mesh_center = np.mean(voussoir_vertices, axis=0)
    for a, b, c, d in face_quads:
        for triangle in ((a, b, c), (a, c, d)):
            i, j, k = triangle
            normal = np.cross(voussoir_vertices[j] - voussoir_vertices[i], voussoir_vertices[k] - voussoir_vertices[i])
            face_center = (voussoir_vertices[i] + voussoir_vertices[j] + voussoir_vertices[k]) / 3.0
            if np.dot(normal, face_center - mesh_center) < 0.0:
                j, k = k, j
            voussoir_indices.extend((i, j, k))
    voussoir_mesh = newton.Mesh(voussoir_vertices, voussoir_indices)

    bodies: list[int] = []
    crown_bodies: list[int] = []
    colors = (wp.vec3(0.72, 0.38, 0.20), wp.vec3(0.86, 0.53, 0.28))
    for side in (-1.0, 1.0):
        for level in range(pier_levels):
            buttress_extra = (pier_levels - level - 1) * 0.012 * scale
            course_half_x = block_half_x + 0.25 * buttress_extra
            course_half_y = pier_half_y + buttress_extra
            course_center_y = opening_half_width + course_half_y
            body = builder.add_body(
                xform=wp.transform(
                    wp.vec3(
                        center[0],
                        center[1] + side * course_center_y,
                        center[2] + (2 * level + 1) * block_half_z,
                    ),
                    wp.quat_identity(),
                )
            )
            builder.add_shape_box(
                body,
                hx=course_half_x,
                hy=course_half_y,
                hz=block_half_z,
                cfg=shape_cfg,
                color=colors[level % 2],
            )
            bodies.append(body)

    for block in range(arch_block_count):
        angle = (block + 0.5) * math.pi / arch_block_count
        body = builder.add_body(
            xform=wp.transform(
                wp.vec3(
                    center[0],
                    center[1] + arch_radius * math.cos(angle),
                    center[2] + spring_height + arch_radius * math.sin(angle),
                ),
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle - 0.5 * math.pi),
            )
        )
        builder.add_shape_convex_hull(
            body,
            mesh=voussoir_mesh,
            cfg=shape_cfg,
            color=colors[block % 2],
        )
        bodies.append(body)
        if block == arch_block_count // 2:
            crown_bodies.append(body)

    return bodies, crown_bodies


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.device = wp.get_device()
        self.sim_time = 0.0
        self.solver_type = str(args.solver)

        asset_path = newton.utils.download_asset("disneyresearch", ref=DR_LEGS_POLICY_ASSET_REF)
        config_path = asset_path / "dr_legs" / "rl_policies" / "drlegs_walk.yaml"
        with open(config_path, encoding="utf-8") as config_file:
            self.policy_config = yaml.safe_load(config_file)

        policy_sim_dt = float(self.policy_config["sim_dt"])
        policy_substeps = int(self.policy_config["control_decimation"])
        self.robot_scale = float(args.robot_scale)
        if self.robot_scale <= 0.0:
            raise ValueError(f"Robot scale must be positive, got {self.robot_scale}.")
        self.sim_dt = 0.5 * policy_sim_dt
        self.sim_substeps = 2 * policy_substeps
        self.frame_dt = policy_sim_dt * policy_substeps
        self.action_scale = float(self.policy_config["action_scale"])
        self.command_height = self.robot_scale * float(self.policy_config["standing_height"])
        self.policy_command_velocity = wp.vec2(
            min(float(args.forward_speed), float(self.policy_config["vel_cmd_max"])),
            0.0,
        )
        self.command_velocity = self.robot_scale * self.policy_command_velocity
        self.settle_frame_count = max(1, math.ceil(float(args.settle_time) / self.frame_dt))
        self.settle_speed = float(args.settle_speed)
        self.landing_height = self.command_height + float(args.landing_height_tolerance)
        self.settled_frames = 0
        self.policy_active = False
        self.policy_start_root_x: float | None = None

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        if self.solver_type == "lox":
            newton.solvers.SolverKamino.register_custom_attributes(builder)
        builder.request_contact_attributes("force")
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 20.0
        builder.default_shape_cfg.mu = 0.8
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        builder.rigid_gap = 0.005

        model_path = asset_path / self.policy_config["usd_model"]
        source_stage = Usd.Stage.Open(str(model_path))
        if source_stage is None:
            raise FileNotFoundError(f"Could not open DR Legs USD stage: {model_path}")
        session_layer = Sdf.Layer.CreateAnonymous("scaled_dr_legs.usda")
        robot_stage = Usd.Stage.Open(source_stage.GetRootLayer(), session_layer)
        if robot_stage is None:
            raise RuntimeError(f"Could not create a scaled view of DR Legs USD stage: {model_path}")
        robot_stage.SetEditTarget(session_layer)
        robot_root_prim = robot_stage.GetDefaultPrim()
        robot_root = UsdGeom.Xformable(robot_root_prim)
        robot_root.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "parachute").Set(Gf.Vec3d(self.robot_scale))
        pelvis_prim = next(prim for prim in robot_stage.Traverse() if prim.GetName() == "pelvis")
        if pelvis_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            pelvis_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        builder.add_usd(
            robot_stage,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
        drop_offset = wp.vec3(0.0, 0.0, float(args.drop_height))
        for body_index, body_transform in enumerate(builder.body_q):
            builder.body_q[body_index] = wp.transform(body_transform.p + drop_offset, body_transform.q)
        self.robot_root_body = next(
            (index for index, label in enumerate(builder.body_label) if label.endswith("/pelvis")),
            -1,
        )
        if self.robot_root_body != 0:
            raise ValueError(f"Expected the DR Legs pelvis to be body 0, found {self.robot_root_body}.")

        supplemental_colliders = _add_visual_mesh_box_colliders(builder)
        print(
            f"[INFO] Imported DR Legs at {self.robot_scale:.2f}x scale and added "
            f"{len(supplemental_colliders)} supplemental colliders"
        )
        robot_shapes = list(range(len(builder.shape_body)))

        (
            self.policy_joint_coord_indices,
            self.actuated_dof_indices,
            self.actuated_target_indices,
        ) = _resolve_policy_joint_indices(builder, target_coord_layout=True)
        for dof_index in range(len(builder.joint_target_ke)):
            if dof_index in self.actuated_dof_indices:
                builder.joint_target_ke[dof_index] = float(self.policy_config["pd_kp"])
                builder.joint_target_kd[dof_index] = float(self.policy_config["pd_kd"])
                builder.joint_armature[dof_index] = float(self.policy_config["pd_armature"])
            else:
                builder.joint_target_ke[dof_index] = 0.0
                builder.joint_target_kd[dof_index] = 0.0
            builder.joint_damping[dof_index] = 0.0

        canopy_radial_resolution = 10
        canopy_angular_resolution = 48
        canopy_radius = 0.8
        pelvis_position = builder.body_q[self.robot_root_body].p
        canopy_height = float(pelvis_position[2]) + 1.65
        canopy_vertices, canopy_indices, openings = _create_canopy_mesh(
            canopy_radial_resolution,
            canopy_angular_resolution,
            canopy_radius,
            canopy_height,
        )

        self.cloth_particle_start = builder.particle_count
        self.cloth_triangle_start = builder.tri_count
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=canopy_vertices,
            indices=canopy_indices,
            density=0.05,
            tri_ke=5.0e3,
            tri_ka=5.0e3,
            tri_kd=5.0,
            edge_ke=0.01,
            edge_kd=0.001,
            particle_radius=0.01,
            validate_mesh=True,
            label="parachute_canopy",
        )
        self.cloth_particle_count = builder.particle_count - self.cloth_particle_start
        self.cloth_triangle_count = builder.tri_count - self.cloth_triangle_start
        self.cloth_render_uvs = wp.array(
            [(0.5 + x / (2.0 * canopy_radius), 0.5 + y / (2.0 * canopy_radius)) for x, y, _ in canopy_vertices],
            dtype=wp.vec2,
            device=self.device,
        )
        # RTX binds texture-backed materials to dynamic meshes, while a constant
        # log_mesh color currently falls back to the default gray USD material.
        self.cloth_render_texture = np.full((2, 2, 4), (20, 166, 255, 255), dtype=np.uint8)

        cable_cfg = builder.default_shape_cfg.copy()
        cable_cfg.density = 10.0
        cable_cfg.mu = 0.9
        cable_cfg.margin = 0.004
        cable_cfg.gap = 0.002

        stopper_cfg = cable_cfg.copy()
        stopper_cfg.density = 0.1
        stopper_cfg.mu = 1.0

        cable_radius = 0.005
        stopper_radius = 0.065
        stopper_clearance = 0.08
        eyelet_lead = 0.12
        target_segment_length = 0.095
        junction_radius = 0.32
        junction_height = float(pelvis_position[2]) + 1.15
        cable_colors = (
            wp.vec3(0.95, 0.50, 0.08),
            wp.vec3(0.95, 0.70, 0.10),
            wp.vec3(0.90, 0.58, 0.08),
            wp.vec3(1.00, 0.78, 0.16),
        )

        self.cable_bodies: list[int] = []
        self.cable_projected_areas: list[float] = []
        self.robot_cable_joints: list[int] = []
        self.stopper_body_ids: list[int] = []
        pelvis_transform = builder.body_q[self.robot_root_body]
        for group in range(4):
            opening_a = openings[2 * group]
            opening_b = openings[2 * group + 1]
            group_angle = math.atan2(opening_a[1] + opening_b[1], opening_a[0] + opening_b[0])
            group_direction = (math.cos(group_angle), math.sin(group_angle))
            pelvis_anchor = wp.vec3(
                self.robot_scale * 0.035 * group_direction[0],
                self.robot_scale * 0.09 * group_direction[1],
                self.robot_scale * 0.028,
            )
            root_position = wp.transform_point(pelvis_transform, pelvis_anchor)
            root = (float(root_position[0]), float(root_position[1]), float(root_position[2]))
            junction = (
                junction_radius * group_direction[0],
                junction_radius * group_direction[1],
                junction_height,
            )
            ends = (
                (opening_a[0], opening_a[1], opening_a[2] + stopper_clearance),
                (opening_b[0], opening_b[1], opening_b[2] + stopper_clearance),
            )
            eyelet_guides = (
                (opening_a[0], opening_a[1], opening_a[2] - eyelet_lead),
                (opening_b[0], opening_b[1], opening_b[2] - eyelet_lead),
            )
            nodes, edges, endpoint_edges = _create_y_cable_graph(
                root,
                junction,
                eyelet_guides,
                ends,
                target_segment_length,
            )
            bodies, _joints = builder.add_rod_graph(
                node_positions=nodes,
                edges=edges,
                radius=cable_radius,
                cfg=cable_cfg,
                stretch_stiffness=2.0e5,
                stretch_damping=100.0,
                bend_stiffness=0.001,
                bend_damping=0.001,
                label=f"suspension_group_{group}",
                wrap_in_articulation=False,
                color=cable_colors[group],
                body_frame_origin="com",
            )
            self.cable_bodies.extend(bodies)
            self.cable_projected_areas.extend(
                2.0 * cable_radius * float(wp.length(nodes[edge_end] - nodes[edge_start]))
                for edge_start, edge_end in edges
            )

            for branch, edge_index in enumerate(endpoint_edges):
                edge_start, edge_end = edges[edge_index]
                edge_length = float(wp.length(nodes[edge_end] - nodes[edge_start]))
                stopper_body = bodies[edge_index]
                builder.add_shape_sphere(
                    stopper_body,
                    xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * edge_length), wp.quat_identity()),
                    radius=stopper_radius,
                    cfg=stopper_cfg,
                    color=wp.vec3(0.95, 0.35, 0.05),
                    label=f"stopper_group_{group}_{branch}",
                )
                self.stopper_body_ids.append(stopper_body)

            root_start, root_end = edges[0]
            root_length = float(wp.length(nodes[root_end] - nodes[root_start]))
            robot_cable_joint = builder.add_joint_ball(
                parent=self.robot_root_body,
                child=bodies[0],
                parent_xform=wp.transform(pelvis_anchor, wp.quat_identity()),
                child_xform=wp.transform(
                    wp.vec3(0.0, 0.0, -0.5 * root_length),
                    wp.quat_identity(),
                ),
                damping=0.0,
                collision_filter_parent=True,
                label=f"robot_suspension_group_{group}",
            )
            self.robot_cable_joints.append(robot_cable_joint)

            # The ball joint carries the suspension load. Filtering cable-robot
            # and within-group cable contacts avoids initial overlaps around the
            # compact pelvis anchors and the idealized Y junction.
            cable_shapes = [shape for body in bodies for shape in builder.body_shapes[body]]
            for cable_shape in cable_shapes:
                for robot_shape in robot_shapes:
                    builder.add_shape_collision_filter_pair(cable_shape, robot_shape)
            for cable_shape_index, cable_shape in enumerate(cable_shapes):
                for other_cable_shape in cable_shapes[cable_shape_index + 1 :]:
                    builder.add_shape_collision_filter_pair(cable_shape, other_cable_shape)

        all_cable_shapes = [shape for body in self.cable_bodies for shape in builder.body_shapes[body]]
        for cable_shape_index, cable_shape in enumerate(all_cable_shapes):
            for other_cable_shape in all_cable_shapes[cable_shape_index + 1 :]:
                builder.add_shape_collision_filter_pair(cable_shape, other_cable_shape)

        masonry_cfg = builder.default_shape_cfg.copy()
        masonry_cfg.density = float(args.arch_density)
        masonry_cfg.mu = 0.9
        masonry_cfg.gap = 0.001
        arch_scale = self.robot_scale / 1.5
        arch_center_x = float(pelvis_position[0]) + float(args.arch_distance)
        self.arch_far_face = arch_center_x + 0.05 * arch_scale
        self.arch_bodies, self.arch_crown_bodies = _add_masonry_arch(
            builder,
            wp.vec3(
                arch_center_x,
                float(pelvis_position[1]),
                0.0,
            ),
            arch_scale,
            masonry_cfg,
        )

        builder.add_ground_plane(color=wp.vec3(0.35, 0.38, 0.42))
        builder.color(include_bending=True)

        self.model = builder.finalize(skip_validation_joints=True)
        self.model.rigid_contact_max = 4096
        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = 25.0
        self.model.soft_contact_mu = 0.9
        self.model.joint_friction.zero_()

        if self.solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=args.max_iterations,
                particle_enable_self_contact=True,
                particle_self_contact_radius=0.02,
                particle_self_contact_margin=0.03,
                particle_topological_contact_filter_threshold=2,
                rigid_body_contact_buffer_size=256,
                rigid_body_particle_contact_buffer_size=1024,
            )
        else:
            solver_config = newton.solvers.SolverKamino.Config.from_model(self.model, dynamics_solver="lox")
            solver_config.use_collision_detector = False
            solver_config.sparse_jacobian = True
            solver_config.use_fk_solver = False
            solver_config.integrator = "euler"
            solver_config.constraints.gamma = 0.002
            solver_config.lox.max_iterations = args.max_iterations
            # solver_config.lox.projection_iterations = args.projection_iterations
            # solver_config.lox.projection_method = "apgd"
            solver_config.lox.deformable_enable_self_contact = True
            self.solver = newton.solvers.SolverKamino(self.model, config=solver_config)
            penalty_scales = self.solver.lox_joint_penalty_scale_seed(self.sim_dt)
            formatted_scales = ", ".join(f"{scale:.3g}" for scale in penalty_scales)
            print(f"[INFO] Seeded LOX joint penalty scales per world: [{formatted_scales}]")

        self.air_velocity = wp.vec3(-float(args.backward_wind_speed), 0.0, float(args.air_speed))
        self.drag_coefficient = float(args.drag_coefficient)
        self.cloth_skin_drag_coefficient = float(args.cloth_skin_drag_coefficient)
        self.cable_drag_coefficient = float(args.cable_drag_coefficient)
        self.cable_body_ids_wp = wp.array(self.cable_bodies, dtype=wp.int32, device=self.device)
        self.cable_projected_areas_wp = wp.array(
            self.cable_projected_areas,
            dtype=wp.float32,
            device=self.device,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase=args.broad_phase,
            contact_matching="latest",
            enable_rigid_soft_full_surface_contact=True,
            soft_contact_margin=0.012,
            soft_contact_max=max(8 * self.model.particle_count, 4096),
        )
        self.contacts = self.collision_pipeline.contacts()
        self.dynamic_soft_contact_seen = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.dynamic_full_surface_contact_seen = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._initial_particle_q = wp.clone(self.state_0.particle_q)
        self._initial_particle_qd = wp.clone(self.state_0.particle_qd)
        self._initial_body_q = wp.clone(self.state_0.body_q)
        self._initial_body_qd = wp.clone(self.state_0.body_qd)
        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)

        self.body_count_per_world = self.model.body_count
        self.joint_coord_count_per_world = self.model.joint_coord_count
        self.observation = wp.zeros((1, OBSERVATION_DIM), dtype=wp.float32, device=self.device)
        self.actions = wp.zeros((1, ACTION_DIM), dtype=wp.float32, device=self.device)
        self.previous_actions = wp.zeros((1, ACTION_DIM), dtype=wp.float32, device=self.device)
        self.phase = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.path_position = wp.zeros(1, dtype=wp.vec2, device=self.device)
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

        initial_body_q = self.state_0.body_q.numpy()
        self.initial_root_position = initial_body_q[self.robot_root_body, :3].copy()
        self.initial_arch_positions = initial_body_q[self.arch_bodies, :3].copy()
        self.arch_positions_at_policy_start: np.ndarray | None = None
        self.arch_crown_heights_at_policy_start: np.ndarray | None = None
        self.arch_pre_policy_max_displacement = 0.0

        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        self.viewer.set_camera(
            pos=wp.vec3(4.0, -5.0, 2.3),
            pitch=-5.0,
            yaw=140.0,
        )
        camera = getattr(self.viewer, "camera", None)
        if camera is not None:
            camera.look_at(wp.vec3(0.0, 0.0, 2.15))

        self.graph = None
        self._prepare_capture()

    def _restore_initial_state(self):
        """Restore simulation state after warm-up and graph capture."""
        for state in (self.state_0, self.state_1):
            wp.copy(state.particle_q, self._initial_particle_q)
            wp.copy(state.particle_qd, self._initial_particle_qd)
            wp.copy(state.body_q, self._initial_body_q)
            wp.copy(state.body_qd, self._initial_body_qd)
            wp.copy(state.joint_q, self._initial_joint_q)
            wp.copy(state.joint_qd, self._initial_joint_qd)
            state.clear_forces()
        self.solver.reset(self.state_0)
        self.dynamic_soft_contact_seen.zero_()
        self.dynamic_full_surface_contact_seen.zero_()

    def _prepare_capture(self):
        """Compile lazy kernels and capture the fixed solver substeps when supported."""
        self.simulate()
        self._restore_initial_state()

        if self.device.is_cpu or self.device.is_mempool_enabled:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            self._restore_initial_state()

    def simulate(self):
        """Advance one policy-rate frame with the selected solver."""
        for substep in range(self.sim_substeps):
            source = self.state_0 if substep % 2 == 0 else self.state_1
            target = self.state_1 if substep % 2 == 0 else self.state_0
            source.clear_forces()
            self.viewer.apply_forces(source)
            wp.launch(
                apply_aerodynamic_drag,
                dim=self.cloth_triangle_count,
                inputs=[
                    self.cloth_triangle_start,
                    self.air_velocity,
                    self.drag_coefficient,
                    self.cloth_skin_drag_coefficient,
                    self.model.tri_indices,
                    source.particle_q,
                    source.particle_qd,
                ],
                outputs=[source.particle_f],
                device=self.device,
            )
            wp.launch(
                apply_cable_wind_drag,
                dim=len(self.cable_bodies),
                inputs=[
                    self.cable_body_ids_wp,
                    self.cable_projected_areas_wp,
                    self.air_velocity,
                    self.cable_drag_coefficient,
                    source.body_qd,
                ],
                outputs=[source.body_f],
                device=self.device,
            )
            self.collision_pipeline.collide(source, self.contacts)
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
            self.solver.step(source, target, self.control, self.contacts, self.sim_dt)
            if self.solver_type == "lox":
                self.solver.update_contacts(self.contacts, target)

        if self.sim_substeps % 2 == 1:
            self.state_0.assign(self.state_1)

    def _activate_policy_if_settled(self):
        """Start walking after the landed pelvis remains vertically settled."""
        body_q = self.state_0.body_q.numpy()
        root_q = body_q[self.robot_root_body]
        root_qd = self.state_0.body_qd.numpy()[self.robot_root_body]
        landed = root_q[2] <= self.landing_height
        settled = abs(root_qd[2]) <= self.settle_speed
        self.settled_frames = self.settled_frames + 1 if landed and settled else 0
        if self.settled_frames < self.settle_frame_count:
            return

        self.policy_active = True
        self.policy_start_root_x = float(root_q[0])
        self.arch_positions_at_policy_start = body_q[self.arch_bodies, :3].copy()
        self.arch_crown_heights_at_policy_start = body_q[self.arch_crown_bodies, 2].copy()
        self.arch_pre_policy_max_displacement = float(
            np.max(
                np.linalg.norm(self.arch_positions_at_policy_start - self.initial_arch_positions, axis=1),
                initial=0.0,
            )
        )
        self.path_position.assign([wp.vec2(float(root_q[0]), float(root_q[1]))])
        print(f"[INFO] Landing settled at t={self.sim_time:.2f} s; starting the DR Legs walking policy")

    def _policy_step(self):
        """Evaluate the DR Legs walking policy and update joint targets."""
        wp.launch(
            _advance_policy_state,
            dim=1,
            inputs=[
                self.state_0.body_q,
                self.body_count_per_world,
                self.command_velocity,
                self.frame_dt,
                1.0 / (2.0 * float(self.policy_config["contact_duration"])),
                self.robot_scale * float(self.policy_config["linear_path_error_limit"]),
                self.phase,
                self.path_position,
            ],
            device=self.device,
        )
        wp.launch(
            _compute_observation,
            dim=1,
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
                self.policy_command_velocity,
                0.0,
                self.command_height,
                self.robot_scale * float(self.policy_config["path_deviation_scale"]),
                self.robot_scale * float(self.policy_config["height_error_scale"]),
                self.action_scale,
                self.observation,
            ],
            device=self.device,
        )
        wp.launch(
            normalize_scaled_robot_observation,
            dim=1,
            inputs=[self.robot_scale, self.observation],
            device=self.device,
        )

        with self.torch.no_grad():
            action_tensor = self.policy(self.observation_torch).contiguous()
        wp.copy(self.previous_actions, self.actions)
        wp.copy(self.actions, wp.from_torch(action_tensor))
        wp.launch(
            _apply_policy_actions,
            dim=(1, ACTION_DIM),
            inputs=[
                self.actions,
                self.actuated_target_indices_wp,
                len(self.control.joint_target_q),
                self.action_scale,
                self.control.joint_target_q,
            ],
            device=self.device,
        )

    def step(self):
        """Advance the parachute and start the policy after landing."""
        if self.policy_active:
            self._policy_step()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        if not self.policy_active:
            self._activate_policy_if_settled()

    def render(self):
        """Render the current parachute and robot state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/parachute/canopy",
            self.state_0.particle_q,
            self.model.tri_indices[
                self.cloth_triangle_start : self.cloth_triangle_start + self.cloth_triangle_count
            ].flatten(),
            uvs=self.cloth_render_uvs,
            texture=self.cloth_render_texture,
            backface_culling=False,
        )
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify a finite landing followed by policy-driven locomotion."""
        state_arrays = (
            ("body_q", self.state_0.body_q.numpy()),
            ("body_qd", self.state_0.body_qd.numpy()),
            ("joint_q", self.state_0.joint_q.numpy()),
            ("joint_qd", self.state_0.joint_qd.numpy()),
            ("particle_q", self.state_0.particle_q.numpy()),
            ("particle_qd", self.state_0.particle_qd.numpy()),
        )
        for name, values in state_arrays:
            assert np.isfinite(values).all(), f"{name} contains non-finite values"

        if self.solver_type == "lox":
            failed = self.solver._solver_kamino.solver_fd.world_failed.numpy()
            assert not failed.any(), "LOX reported a failed parachute rollout"
        assert int(self.dynamic_full_surface_contact_seen.numpy()[0]) == 1, (
            "No full-surface cloth contact with a dynamic rigid body was generated"
        )

        root_position = state_arrays[0][1][self.robot_root_body, :3]
        root_orientation = state_arrays[0][1][self.robot_root_body, 3:7]
        root_up_z = 1.0 - 2.0 * (root_orientation[0] ** 2 + root_orientation[1] ** 2)
        assert root_position[2] <= self.landing_height, f"DR Legs did not land: pelvis z={root_position[2]:.3f} m"
        assert root_position[2] > 0.5 * self.command_height, f"DR Legs fell: pelvis z={root_position[2]:.3f} m"
        assert root_up_z > 0.6, f"DR Legs fell: root up-axis z={root_up_z:.3f}"
        assert self.policy_active, "The DR Legs walking policy did not start after landing"
        if self.policy_start_root_x is not None:
            forward_distance = float(root_position[0] - self.policy_start_root_x)
            assert forward_distance > 0.01, f"DR Legs did not walk forward after landing: {forward_distance:.3f} m"
        assert root_position[0] > self.arch_far_face, (
            f"DR Legs did not pass through the arch: pelvis x={root_position[0]:.3f} m"
        )

        assert self.arch_positions_at_policy_start is not None
        assert self.arch_crown_heights_at_policy_start is not None
        assert self.arch_pre_policy_max_displacement < 0.05, (
            f"The masonry arch collapsed before DR Legs walked: {self.arch_pre_policy_max_displacement:.3f} m"
        )
        arch_positions = state_arrays[0][1][self.arch_bodies, :3]
        arch_displacements = np.linalg.norm(arch_positions - self.arch_positions_at_policy_start, axis=1)
        crown_height_drop = self.arch_crown_heights_at_policy_start - state_arrays[0][1][self.arch_crown_bodies, 2]
        assert np.max(arch_displacements, initial=0.0) > 0.05, "DR Legs did not disturb the masonry arch"
        assert np.max(crown_height_drop, initial=0.0) > 0.05, "The masonry arch did not topple"

    @staticmethod
    def create_parser():
        """Create the example command-line parser."""
        parser = newton.examples.create_parser()
        newton.examples.add_broad_phase_arg(parser)
        parser.set_defaults(broad_phase="sap")
        parser.set_defaults(num_frames=200)
        parser.add_argument(
            "--solver",
            choices=("lox", "vbd"),
            default="lox",
            help="Dynamics solver used for the shared parachute scene.",
        )
        parser.add_argument(
            "--drop-height",
            type=float,
            default=3.0,
            help="Initial DR Legs pelvis offset above the ground [m].",
        )
        parser.add_argument(
            "--robot-scale",
            type=float,
            default=1.5,
            help="Uniform DR Legs geometry scale.",
        )
        parser.add_argument(
            "--air-speed",
            type=float,
            default=0.0,
            help="Upward ambient airflow speed [m/s].",
        )
        parser.add_argument(
            "--backward-wind-speed",
            type=float,
            default=0.15,
            help="Airflow speed opposite the robot's forward direction [m/s].",
        )
        parser.add_argument(
            "--drag-coefficient",
            type=float,
            default=15.0,
            help="Dimensionless cloth-normal drag coefficient.",
        )
        parser.add_argument(
            "--cloth-skin-drag-coefficient",
            type=float,
            default=0.2,
            help="Dimensionless tangential cloth drag coefficient.",
        )
        parser.add_argument(
            "--cable-drag-coefficient",
            type=float,
            default=1.2,
            help="Dimensionless suspension-cable crossflow drag coefficient.",
        )
        parser.add_argument(
            "--forward-speed",
            type=float,
            default=0.3,
            help="Forward policy command [m/s].",
        )
        parser.add_argument(
            "--arch-distance",
            type=float,
            default=0.25,
            help="Masonry arch distance along the robot's forward path [m].",
        )
        parser.add_argument(
            "--arch-density",
            type=float,
            default=50.0,
            help="Masonry block density [kg/m^3].",
        )
        parser.add_argument(
            "--settle-time",
            type=float,
            default=0.2,
            help="Continuous landed time required before policy activation [s].",
        )
        parser.add_argument(
            "--settle-speed",
            type=float,
            default=0.25,
            help="Maximum absolute pelvis vertical speed considered settled [m/s].",
        )
        parser.add_argument(
            "--landing-height-tolerance",
            type=float,
            default=0.1,
            help="Pelvis height tolerance above the policy standing height [m].",
        )
        parser.add_argument(
            "--max-iterations",
            type=int,
            default=10,
            help="VBD iterations or maximum LOX splitting iterations per substep.",
        )
        parser.add_argument(
            "--projection-iterations",
            type=int,
            default=15,
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
