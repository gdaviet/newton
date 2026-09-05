# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Filtered surface self-contact generation for LOX cloth."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import warp as wp

from ......geometry.kernels import (
    EDGE_COLLISION_BUFFER_OVERFLOW_INDEX,
    VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX,
    triangle_closest_point,
)
from ......utils.mesh import MeshAdjacency, MeshAdjacencyData, get_vertex_num_adjacent_edges
from .....vbd.tri_mesh_collision import TriMeshCollisionDetector
from ...geometry.contacts import make_contact_frame_znorm
from .bias import compute_contact_velocity_target
from .deformable_contact import DEFORMABLE_CONTACT_STATUS_UNUSED, DEFORMABLE_CONTACT_STATUS_VALID
from .soft_contact_filter import (
    SOFT_CONTACT_GEOMETRY_EPSILON,
    compute_soft_edge_normal_cones,
    edge_feature_normal_cone_contains,
    triangle_feature_normal_cone_contains,
    vertex_normal_cone_contains,
)
from .time import validate_world_time_step

if TYPE_CHECKING:
    from ......sim import Model
    from .deformable_contact import DeformableContactSystem

wp.set_module_options({"enable_backward": False})

_COEFFICIENT_TOLERANCE = 1.0e-5
_PARTICLE_FLAG_ACTIVE = 1


class _SurfaceCollisionModel:
    """Present LOX's triangulated surface edges through the VBD detector API."""

    def __init__(self, model, edge_indices: wp.array[wp.int32], adjacency: MeshAdjacency):
        self._model = model
        self.edge_indices = edge_indices
        self.edge_count = int(edge_indices.shape[0])
        self.soft_mesh_adjacency = adjacency

    def __getattr__(self, name: str):
        return getattr(self._model, name)


@wp.kernel
def _warn_candidate_buffer_overflow(
    resize_flags: wp.array[wp.int32],
    vertex_buffer_size: int,
    edge_buffer_size: int,
    warning_emitted: wp.array[wp.int32],
):
    if resize_flags[VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX] != 0 and warning_emitted[0] == 0:
        warning_emitted[0] = 1
        wp.printf(
            "Warning: LOX dropped deformable vertex-triangle contact candidates because the per-vertex buffer "
            "is full (capacity %d). Increase deformable_self_contact_vertex_buffer_size.\n",
            vertex_buffer_size,
        )
    if resize_flags[EDGE_COLLISION_BUFFER_OVERFLOW_INDEX] != 0 and warning_emitted[1] == 0:
        warning_emitted[1] = 1
        wp.printf(
            "Warning: LOX dropped deformable edge-edge contact candidates because the per-edge buffer "
            "is full (capacity %d). Increase deformable_self_contact_edge_buffer_size.\n",
            edge_buffer_size,
        )


@wp.func
def _write_self_contact(
    contact: int,
    source_particles: wp.vec4i,
    source_coefficients: wp.vec4,
    contact_normal: wp.vec3,
    contact_gap: float,
    particle_velocity: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    newton_to_packed: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    inverse_weight: wp.array[float],
    time_step: wp.array[wp.float32],
    stabilization_fraction: float,
    dead_zone: float,
    impact_velocity_threshold: float,
    recoverable_response: bool,
    friction: float,
    restitution: float,
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_shape: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    contact_distances: wp.array[float],
    bias: wp.array[wp.vec3],
    contact_friction: wp.array[float],
    contact_status: wp.array[wp.int32],
    particle_multiplicity: wp.array[wp.int32],
    particle_majorizer_weight_sum: wp.array[float],
    world_contact_count: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
):
    world = int(-1)
    relative_velocity = wp.vec3(0.0)
    prescribed_relative_velocity = wp.vec3(0.0)
    inverse_delassus = float(0.0)
    for slot in range(4):
        source_particle = source_particles[slot]
        packed_particle = newton_to_packed[source_particle]
        particle_world = packed_world[packed_particle]
        if world < 0:
            world = particle_world
        elif particle_world != world:
            return
        coefficient = source_coefficients[slot]
        particle_indices[contact, slot] = packed_particle
        coefficients[contact, slot] = coefficient
        if (particle_flags[source_particle] & _PARTICLE_FLAG_ACTIVE) != 0:
            relative_velocity += coefficient * particle_velocity[source_particle]
            if inverse_weight[packed_particle] <= 0.0:
                prescribed_relative_velocity += coefficient * particle_velocity[source_particle]
        inverse_delassus += coefficient * coefficient * inverse_weight[packed_particle]

    # A fully prescribed pair is already satisfied by construction and must not fail its world.
    if world < 0 or inverse_delassus <= 0.0:
        return

    previous_normal_velocity = wp.dot(contact_normal, relative_velocity)
    velocity_target = compute_contact_velocity_target(
        contact_gap,
        previous_normal_velocity,
        restitution,
        time_step[world],
        stabilization_fraction,
        dead_zone,
        impact_velocity_threshold,
        recoverable_response,
    )
    contact_bias = prescribed_relative_velocity - velocity_target * contact_normal
    contact_frame = make_contact_frame_znorm(contact_normal)

    contact_world[contact] = world
    contact_shape[contact] = -1
    contact_body[contact] = -1
    frame[contact] = contact_frame
    contact_distances[contact] = contact_gap
    bias[contact] = wp.transpose(contact_frame) @ contact_bias
    contact_friction[contact] = friction
    contact_status[contact] = DEFORMABLE_CONTACT_STATUS_VALID
    wp.atomic_max(world_status, world, DEFORMABLE_CONTACT_STATUS_VALID)
    wp.atomic_add(world_contact_count, world, 1)
    for slot in range(4):
        packed_particle = particle_indices[contact, slot]
        if inverse_weight[packed_particle] > 0.0 and wp.abs(coefficients[contact, slot]) > _COEFFICIENT_TOLERANCE:
            wp.atomic_add(particle_multiplicity, packed_particle, 1)
            wp.atomic_add(
                particle_majorizer_weight_sum,
                packed_particle,
                wp.abs(coefficients[contact, slot]),
            )


@wp.kernel
def _adapt_self_contacts(
    output_offset: int,
    vertex_capacity: int,
    vertex_records: wp.array[wp.int32],
    edge_records: wp.array[wp.int32],
    position: wp.array[wp.vec3],
    rest_position: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    cone_axis: wp.array[wp.vec3],
    cone_cosine: wp.array[float],
    enable_normal_cone_filtering: bool,
    normal_cone_filtering_min_distance: float,
    newton_to_packed: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    inverse_weight: wp.array[float],
    rest_contact_exclusion_radius: float,
    margin: float,
    gap: float,
    edge_parallel_epsilon: float,
    time_step: wp.array[wp.float32],
    stabilization_fraction: float,
    dead_zone: float,
    impact_velocity_threshold: float,
    recoverable_response: bool,
    friction: float,
    restitution: float,
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_shape: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    contact_distances: wp.array[float],
    bias: wp.array[wp.vec3],
    contact_friction: wp.array[float],
    contact_status: wp.array[wp.int32],
    particle_multiplicity: wp.array[wp.int32],
    particle_majorizer_weight_sum: wp.array[float],
    world_contact_count: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
):
    candidate = wp.tid()
    contact = output_offset + candidate
    for slot in range(4):
        particle_indices[contact, slot] = -1
        coefficients[contact, slot] = 0.0
    contact_world[contact] = -1
    contact_shape[contact] = -1
    contact_body[contact] = -1
    frame[contact] = wp.mat33f(0.0)
    contact_distances[contact] = 0.0
    bias[contact] = wp.vec3(0.0)
    contact_friction[contact] = 0.0
    contact_status[contact] = DEFORMABLE_CONTACT_STATUS_UNUSED

    if candidate < vertex_capacity:
        vertex = vertex_records[2 * candidate]
        triangle = vertex_records[2 * candidate + 1]
        if vertex < 0 or triangle < 0:
            return
        if get_vertex_num_adjacent_edges(adjacency, vertex) == 0:
            return

        triangle_0 = tri_indices[triangle, 0]
        triangle_1 = tri_indices[triangle, 1]
        triangle_2 = tri_indices[triangle, 2]
        point_0 = position[triangle_0]
        point_1 = position[triangle_1]
        point_2 = position[triangle_2]
        closest, barycentric, _feature = triangle_closest_point(
            point_0,
            point_1,
            point_2,
            position[vertex],
        )
        if rest_contact_exclusion_radius > 0.0:
            rest_closest, _rest_barycentric, _rest_feature = triangle_closest_point(
                rest_position[triangle_0],
                rest_position[triangle_1],
                rest_position[triangle_2],
                rest_position[vertex],
            )
            if wp.length(rest_position[vertex] - rest_closest) < rest_contact_exclusion_radius:
                return

        difference = position[vertex] - closest
        distance = wp.length(difference)
        if distance <= SOFT_CONTACT_GEOMETRY_EPSILON or distance >= margin + gap:
            return
        contact_normal = difference / distance
        if enable_normal_cone_filtering and distance > normal_cone_filtering_min_distance:
            # Each cone is tested from its feature toward the opposing point;
            # the solver normal points from the triangle toward the vertex.
            if not vertex_normal_cone_contains(vertex, -contact_normal, position, edge_indices, adjacency):
                return
            if not triangle_feature_normal_cone_contains(
                triangle_0,
                triangle_1,
                triangle_2,
                barycentric,
                contact_normal,
                position,
                edge_indices,
                adjacency,
                cone_axis,
                cone_cosine,
            ):
                return

        _write_self_contact(
            contact,
            wp.vec4i(triangle_0, triangle_1, triangle_2, vertex),
            wp.vec4(-barycentric[0], -barycentric[1], -barycentric[2], 1.0),
            contact_normal,
            distance - margin,
            velocity,
            particle_flags,
            newton_to_packed,
            packed_world,
            inverse_weight,
            time_step,
            stabilization_fraction,
            dead_zone,
            impact_velocity_threshold,
            recoverable_response,
            friction,
            restitution,
            particle_indices,
            coefficients,
            contact_world,
            contact_shape,
            contact_body,
            frame,
            contact_distances,
            bias,
            contact_friction,
            contact_status,
            particle_multiplicity,
            particle_majorizer_weight_sum,
            world_contact_count,
            world_status,
        )
        return

    edge_candidate = candidate - vertex_capacity
    edge_0 = edge_records[2 * edge_candidate]
    edge_1 = edge_records[2 * edge_candidate + 1]
    # The detector records both directed orders; retain one constraint per pair.
    if edge_0 < 0 or edge_1 < 0 or edge_0 >= edge_1:
        return

    edge_0_vertex_0 = edge_indices[edge_0, 2]
    edge_0_vertex_1 = edge_indices[edge_0, 3]
    edge_1_vertex_0 = edge_indices[edge_1, 2]
    edge_1_vertex_1 = edge_indices[edge_1, 3]
    if rest_contact_exclusion_radius > 0.0:
        rest_closest_parameters = wp.closest_point_edge_edge(
            rest_position[edge_0_vertex_0],
            rest_position[edge_0_vertex_1],
            rest_position[edge_1_vertex_0],
            rest_position[edge_1_vertex_1],
            edge_parallel_epsilon,
        )
        if rest_closest_parameters[2] < rest_contact_exclusion_radius:
            return
    closest_parameters = wp.closest_point_edge_edge(
        position[edge_0_vertex_0],
        position[edge_0_vertex_1],
        position[edge_1_vertex_0],
        position[edge_1_vertex_1],
        edge_parallel_epsilon,
    )
    parameter_0 = closest_parameters[0]
    parameter_1 = closest_parameters[1]
    distance = closest_parameters[2]
    if distance <= SOFT_CONTACT_GEOMETRY_EPSILON or distance >= margin + gap:
        return

    point_0 = (1.0 - parameter_0) * position[edge_0_vertex_0] + parameter_0 * position[edge_0_vertex_1]
    point_1 = (1.0 - parameter_1) * position[edge_1_vertex_0] + parameter_1 * position[edge_1_vertex_1]
    contact_normal = (point_0 - point_1) / distance
    if enable_normal_cone_filtering and distance > normal_cone_filtering_min_distance:
        # The solver normal points from edge 1 toward edge 0, opposite the
        # feature-to-query direction for edge 0.
        if not edge_feature_normal_cone_contains(
            edge_0,
            parameter_0,
            -contact_normal,
            position,
            edge_indices,
            adjacency,
            cone_axis,
            cone_cosine,
        ):
            return
        if not edge_feature_normal_cone_contains(
            edge_1,
            parameter_1,
            contact_normal,
            position,
            edge_indices,
            adjacency,
            cone_axis,
            cone_cosine,
        ):
            return

    _write_self_contact(
        contact,
        wp.vec4i(edge_0_vertex_0, edge_0_vertex_1, edge_1_vertex_0, edge_1_vertex_1),
        wp.vec4(1.0 - parameter_0, parameter_0, -1.0 + parameter_1, -parameter_1),
        contact_normal,
        distance - margin,
        velocity,
        particle_flags,
        newton_to_packed,
        packed_world,
        inverse_weight,
        time_step,
        stabilization_fraction,
        dead_zone,
        impact_velocity_threshold,
        recoverable_response,
        friction,
        restitution,
        particle_indices,
        coefficients,
        contact_world,
        contact_shape,
        contact_body,
        frame,
        contact_distances,
        bias,
        contact_friction,
        contact_status,
        particle_multiplicity,
        particle_majorizer_weight_sum,
        world_contact_count,
        world_status,
    )


class DeformableSelfContactDetector:
    """Own fixed-capacity proximity candidates and dual-cone filters for LOX cloth."""

    def __init__(
        self,
        model: Model,
        *,
        margin: float,
        gap: float,
        vertex_contact_buffer_size: int,
        edge_contact_buffer_size: int,
        topological_contact_filter_threshold: int,
        rest_contact_exclusion_radius: float,
        edge_parallel_epsilon: float,
        enable_normal_cone_filtering: bool = True,
        normal_cone_filtering_min_distance: float = 1.0e-4,
    ):
        """Allocate cloth self-contact detection and filtering data.

        Args:
            model: Newton cloth model.
            margin: Additional self-contact surface thickness [m].
            gap: Additional speculative detection distance [m].
            vertex_contact_buffer_size: Candidate triangles stored per vertex.
            edge_contact_buffer_size: Candidate edges stored per edge.
            topological_contact_filter_threshold: Adjacent mesh rings excluded from queries.
            rest_contact_exclusion_radius: Rest-space proximity exclusion radius [m].
            edge_parallel_epsilon: Parallel-edge tolerance.
            enable_normal_cone_filtering: Whether to prune candidates using surface normal cones.
            normal_cone_filtering_min_distance: Separation below which normal-cone filtering is bypassed [m].
        """
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("LOX cloth self-contact margin must be finite and non-negative.")
        if not math.isfinite(gap) or gap < 0.0:
            raise ValueError("LOX cloth self-contact gap must be finite and non-negative.")
        if margin + gap <= 0.0:
            raise ValueError("LOX cloth self-contact margin plus gap must be positive.")
        if (
            not isinstance(vertex_contact_buffer_size, int)
            or isinstance(vertex_contact_buffer_size, bool)
            or not isinstance(edge_contact_buffer_size, int)
            or isinstance(edge_contact_buffer_size, bool)
            or vertex_contact_buffer_size < 1
            or edge_contact_buffer_size < 1
        ):
            raise ValueError("LOX cloth self-contact candidate buffer sizes must be positive.")
        if (
            not isinstance(topological_contact_filter_threshold, int)
            or isinstance(topological_contact_filter_threshold, bool)
            or topological_contact_filter_threshold < 0
        ):
            raise ValueError("LOX cloth topological contact filter threshold must be non-negative.")
        if not math.isfinite(rest_contact_exclusion_radius) or rest_contact_exclusion_radius < 0.0:
            raise ValueError("LOX cloth rest contact exclusion radius must be finite and non-negative.")
        if not math.isfinite(edge_parallel_epsilon) or edge_parallel_epsilon <= 0.0:
            raise ValueError("LOX cloth parallel-edge tolerance must be finite and positive.")
        if not isinstance(enable_normal_cone_filtering, bool):
            raise ValueError("LOX cloth normal-cone filtering flag must be a bool.")
        if not math.isfinite(normal_cone_filtering_min_distance) or normal_cone_filtering_min_distance < 0.0:
            raise ValueError("LOX cloth normal-cone filtering minimum distance must be finite and non-negative.")
        if model.tri_count < 1:
            raise ValueError("LOX deformable self-contact requires a triangulated surface.")

        self.model = model
        self.device = model.device
        self.margin = float(margin)
        self.gap = float(gap)
        self.query_radius = self.margin + self.gap
        self.rest_contact_exclusion_radius = float(rest_contact_exclusion_radius)
        self.enable_normal_cone_filtering = enable_normal_cone_filtering
        self.normal_cone_filtering_min_distance = float(normal_cone_filtering_min_distance)
        self.surface_adjacency = MeshAdjacency(tri_indices=model.tri_indices.numpy()).init_vertex_adjacency(
            model.particle_count
        )
        self.edge_indices = wp.array(self.surface_adjacency.edge_indices, dtype=wp.int32, device=self.device)
        self.adjacency = self.surface_adjacency.to(self.device)
        self.edge_cone_axis = wp.zeros(self.edge_indices.shape[0], dtype=wp.vec3, device=self.device)
        self.edge_cone_cosine = wp.full(self.edge_indices.shape[0], -1.0, dtype=float, device=self.device)
        self.detector = TriMeshCollisionDetector(
            _SurfaceCollisionModel(model, self.edge_indices, self.surface_adjacency),
            vertex_collision_buffer_pre_alloc=vertex_contact_buffer_size,
            edge_collision_buffer_pre_alloc=edge_contact_buffer_size,
            edge_edge_parallel_epsilon=edge_parallel_epsilon,
            topological_contact_filter_threshold=topological_contact_filter_threshold,
        )
        self.vertex_capacity = self.detector.vertex_colliding_triangles.shape[0] // 2
        self.edge_capacity = self.detector.edge_colliding_edges.shape[0] // 2
        self.capacity = self.vertex_capacity + self.edge_capacity
        self.vertex_contact_buffer_size = vertex_contact_buffer_size
        self.edge_contact_buffer_size = edge_contact_buffer_size
        self.buffer_overflow_warning_emitted = wp.zeros(2, dtype=wp.int32, device=self.device)

    def detect(self, position: wp.array[wp.vec3]) -> None:
        """Refit proximity structures and freeze filtered candidate pairs for one step."""
        self.detector.resize_flags.zero_()
        if self.enable_normal_cone_filtering:
            wp.launch(
                compute_soft_edge_normal_cones,
                dim=self.edge_indices.shape[0],
                inputs=[position, self.edge_indices],
                outputs=[self.edge_cone_axis, self.edge_cone_cosine],
                device=self.device,
            )
        self.detector.refit(position)
        self.detector.vertex_triangle_collision_detection(
            self.query_radius,
            min_query_radius=self.rest_contact_exclusion_radius,
            min_distance_filtering_ref_pos=self.model.particle_q,
        )
        self.detector.edge_edge_collision_detection(
            self.query_radius,
            min_query_radius=self.rest_contact_exclusion_radius,
            min_distance_filtering_ref_pos=self.model.particle_q,
        )
        wp.launch(
            _warn_candidate_buffer_overflow,
            dim=1,
            inputs=[
                self.detector.resize_flags,
                self.vertex_contact_buffer_size,
                self.edge_contact_buffer_size,
            ],
            outputs=[self.buffer_overflow_warning_emitted],
            device=self.device,
        )

    def adapt(
        self,
        contact_system: DeformableContactSystem,
        position: wp.array[wp.vec3],
        velocity: wp.array[wp.vec3],
        time_step: wp.array[wp.float32],
        friction: float,
        restitution: float,
    ) -> None:
        """Write filtered self-contact constraints after rigid-soft contact slots."""
        validate_world_time_step(time_step, self.model.world_count, self.device)
        wp.launch(
            _adapt_self_contacts,
            dim=self.capacity,
            inputs=[
                contact_system.rigid_contact_capacity,
                self.vertex_capacity,
                self.detector.vertex_colliding_triangles,
                self.detector.edge_colliding_edges,
                position,
                self.model.particle_q,
                velocity,
                self.model.particle_flags,
                self.model.tri_indices,
                self.edge_indices,
                self.adjacency,
                self.edge_cone_axis,
                self.edge_cone_cosine,
                self.enable_normal_cone_filtering,
                self.normal_cone_filtering_min_distance,
                contact_system.cloth_system.topology.newton_to_packed,
                contact_system.cloth_system.topology.packed_solve_world,
                contact_system.cloth_system.full_inverse_weight,
                self.rest_contact_exclusion_radius,
                self.margin,
                self.gap,
                self.detector.edge_edge_parallel_epsilon,
                time_step,
                contact_system.stabilization_fraction,
                contact_system.dead_zone,
                contact_system.impact_velocity_threshold,
                contact_system.recoverable_response,
                friction,
                restitution,
            ],
            outputs=[
                contact_system.particle_indices,
                contact_system.coefficients,
                contact_system.contact_world,
                contact_system.contact_shape,
                contact_system.body,
                contact_system.frame,
                contact_system.gap,
                contact_system.bias,
                contact_system.friction,
                contact_system.status,
                contact_system.particle_multiplicity,
                contact_system.particle_majorizer_weight_sum,
                contact_system.world_contact_count,
                contact_system.world_status,
            ],
            device=self.device,
        )
