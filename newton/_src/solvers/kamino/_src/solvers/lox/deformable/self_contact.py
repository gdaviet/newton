# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Self-contacts of the triangulated surfaces of the LOX deformable bodies.

:class:`DeformableSelfContacts` detects the vertex-triangle and edge-edge proximity
candidates of the deformable surfaces with VBD's triangle-mesh collision detector at the
beginning of each time step, keeps those whose direction lies in the normal cones of
their surface features, and writes them as particle contact rows after the rigid-soft
rows. A vertex-triangle row couples the vertex with coefficient one and the triangle
vertices with their negated barycentric coordinates; an edge-edge row couples the two
edges with their opposite interpolation coefficients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .......geometry import ParticleFlags
from .......geometry.kernels import (
    EDGE_COLLISION_BUFFER_OVERFLOW_INDEX,
    VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX,
    triangle_closest_point,
)
from .......geometry.tri_mesh_collision import TriMeshCollisionDetector
from .......utils.mesh import MeshAdjacency, MeshAdjacencyData, get_vertex_num_adjacent_edges
from ....core.types import mat36f
from ....geometry.contacts import make_contact_frame_znorm
from ..contact import compute_contact_velocity_target
from ..types import ParticleContactStatus
from .contact_filter import (
    SOFT_CONTACT_GEOMETRY_EPSILON,
    _compute_edge_normal_cones,
    edge_feature_normal_cone_contains,
    triangle_feature_normal_cone_contains,
    vertex_normal_cone_contains,
)

if TYPE_CHECKING:
    from .......sim import Model
    from ..types import ParticleContactRows

###
# Module interface
###

__all__ = ["DeformableSelfContacts"]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


class _SurfaceCollisionModel:
    """Present the triangulated surface edges of the deformable bodies through the VBD detector API."""

    def __init__(self, model: Model, edge_indices: wp.array[wp.int32], adjacency: MeshAdjacency):
        self._model = model
        self.edge_indices = edge_indices
        self.edge_count = int(edge_indices.shape[0])
        self.soft_mesh_adjacency = adjacency

    def __getattr__(self, name: str):
        return getattr(self._model, name)


###
# Functions
###


@wp.func
def _write_self_contact(
    row: int,
    source_particles: wp.vec4i,
    source_coefficients: wp.vec4f,
    normal: wp.vec3f,
    gap: wp.float32,
    particle_qd: wp.array[wp.vec3f],
    particle_flags: wp.array[wp.int32],
    newton_to_packed: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    particle_inverse_weight: wp.array[wp.float32],
    model_time_dt: wp.array[wp.float32],
    stabilization_fraction: wp.float32,
    dead_zone: wp.float32,
    friction: wp.float32,
    restitution: wp.float32,
    row_world: wp.array[wp.int32],
    row_status: wp.array[wp.int32],
    row_particle: wp.array2d[wp.int32],
    row_coefficient: wp.array2d[wp.float32],
    row_frame: wp.array[wp.mat33f],
    row_gap: wp.array[wp.float32],
    row_bias: wp.array[wp.vec3f],
    row_friction: wp.array[wp.float32],
):
    """Write one self-contact row; the prescribed particles enter its bias."""
    wid = int(-1)
    relative_velocity = wp.vec3f(0.0)
    prescribed_velocity = wp.vec3f(0.0)
    for slot in range(4):
        source_particle = source_particles[slot]
        pid = newton_to_packed[source_particle]
        if wid < 0:
            wid = particle_world[pid]
        elif particle_world[pid] != wid:
            # Proximity candidates of different worlds leave their row unused
            return
        coefficient = source_coefficients[slot]
        row_particle[row, slot] = pid
        row_coefficient[row, slot] = coefficient
        if (particle_flags[source_particle] & ParticleFlags.ACTIVE) != 0:
            relative_velocity += coefficient * particle_qd[source_particle]
            if particle_inverse_weight[pid] <= 0.0:
                prescribed_velocity += coefficient * particle_qd[source_particle]
    velocity_target = compute_contact_velocity_target(
        gap, wp.dot(normal, relative_velocity), restitution, model_time_dt[wid], stabilization_fraction, dead_zone
    )
    frame = make_contact_frame_znorm(normal)
    row_world[row] = wid
    row_frame[row] = frame
    row_gap[row] = gap
    row_bias[row] = wp.transpose(frame) @ (prescribed_velocity - velocity_target * normal)
    row_friction[row] = friction
    row_status[row] = ParticleContactStatus.ACTIVE


###
# Kernels
###


@wp.kernel
def _warn_candidate_buffer_overflow(
    # Inputs:
    resize_flags: wp.array[wp.int32],
    vertex_buffer_size: wp.int32,
    edge_buffer_size: wp.int32,
    # Outputs:
    warning_emitted: wp.array[wp.int32],
):
    """Warn once when the detector drops vertex-triangle or edge-edge candidates."""
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


@wp.kernel
def _load_self_contacts(
    # Inputs:
    row_offset: wp.int32,
    vertex_capacity: wp.int32,
    vertex_records: wp.array[wp.int32],
    edge_records: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3f],
    particle_q_rest: wp.array[wp.vec3f],
    particle_qd: wp.array[wp.vec3f],
    particle_flags: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    edge_cone_axis: wp.array[wp.vec3f],
    edge_cone_cosine: wp.array[wp.float32],
    normal_cone_filtering: wp.bool,
    normal_cone_filtering_min_distance: wp.float32,
    newton_to_packed: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    particle_inverse_weight: wp.array[wp.float32],
    rest_exclusion_radius: wp.float32,
    margin: wp.float32,
    gap: wp.float32,
    edge_parallel_epsilon: wp.float32,
    model_time_dt: wp.array[wp.float32],
    stabilization_fraction: wp.float32,
    dead_zone: wp.float32,
    friction: wp.float32,
    restitution: wp.float32,
    # Outputs:
    row_world: wp.array[wp.int32],
    row_status: wp.array[wp.int32],
    row_particle: wp.array2d[wp.int32],
    row_coefficient: wp.array2d[wp.float32],
    row_body: wp.array[wp.int32],
    row_shape: wp.array[wp.int32],
    row_frame: wp.array[wp.mat33f],
    row_body_jacobian: wp.array[mat36f],
    row_gap: wp.array[wp.float32],
    row_bias: wp.array[wp.vec3f],
    row_friction: wp.array[wp.float32],
    row_reaction: wp.array[wp.vec3f],
):
    """Load the filtered vertex-triangle and edge-edge candidates of the time step into self-contact rows.

    The candidates within the rest exclusion radius, beyond the detection distance, or outside the
    normal cones of their surface features leave their rows unused.
    """
    candidate = wp.tid()
    row = row_offset + candidate
    for slot in range(4):
        row_particle[row, slot] = -1
        row_coefficient[row, slot] = 0.0
    row_world[row] = -1
    row_shape[row] = -1
    row_body[row] = -1
    row_frame[row] = wp.mat33f(0.0)
    row_body_jacobian[row] = mat36f(0.0)
    row_gap[row] = 0.0
    row_bias[row] = wp.vec3f(0.0)
    row_friction[row] = 0.0
    row_reaction[row] = wp.vec3f(0.0)
    row_status[row] = ParticleContactStatus.UNUSED

    if candidate < vertex_capacity:
        vertex = vertex_records[2 * candidate]
        triangle = vertex_records[2 * candidate + 1]
        if vertex < 0 or triangle < 0 or get_vertex_num_adjacent_edges(adjacency, vertex) == 0:
            return
        triangle_0 = tri_indices[triangle, 0]
        triangle_1 = tri_indices[triangle, 1]
        triangle_2 = tri_indices[triangle, 2]
        closest, barycentric, _feature = triangle_closest_point(
            particle_q[triangle_0], particle_q[triangle_1], particle_q[triangle_2], particle_q[vertex]
        )
        if rest_exclusion_radius > 0.0:
            rest_closest, _rest_barycentric, _rest_feature = triangle_closest_point(
                particle_q_rest[triangle_0],
                particle_q_rest[triangle_1],
                particle_q_rest[triangle_2],
                particle_q_rest[vertex],
            )
            if wp.length(particle_q_rest[vertex] - rest_closest) < rest_exclusion_radius:
                return
        difference = particle_q[vertex] - closest
        distance = wp.length(difference)
        if distance <= SOFT_CONTACT_GEOMETRY_EPSILON or distance >= margin + gap:
            return
        normal = difference / distance
        if normal_cone_filtering and distance > normal_cone_filtering_min_distance:
            # Each cone is tested from its feature toward the opposing point; the normal
            # points from the triangle towards the vertex.
            if not vertex_normal_cone_contains(vertex, -normal, particle_q, edge_indices, adjacency):
                return
            if not triangle_feature_normal_cone_contains(
                triangle_0,
                triangle_1,
                triangle_2,
                barycentric,
                normal,
                particle_q,
                edge_indices,
                adjacency,
                edge_cone_axis,
                edge_cone_cosine,
            ):
                return
        _write_self_contact(
            row,
            wp.vec4i(triangle_0, triangle_1, triangle_2, vertex),
            wp.vec4f(-barycentric[0], -barycentric[1], -barycentric[2], 1.0),
            normal,
            distance - margin,
            particle_qd,
            particle_flags,
            newton_to_packed,
            particle_world,
            particle_inverse_weight,
            model_time_dt,
            stabilization_fraction,
            dead_zone,
            friction,
            restitution,
            row_world,
            row_status,
            row_particle,
            row_coefficient,
            row_frame,
            row_gap,
            row_bias,
            row_friction,
        )
        return

    edge_candidate = candidate - vertex_capacity
    edge_0 = edge_records[2 * edge_candidate]
    edge_1 = edge_records[2 * edge_candidate + 1]
    # The detector records both orders of a pair; one row per pair is kept
    if edge_0 < 0 or edge_1 < 0 or edge_0 >= edge_1:
        return
    edge_0_vertex_0 = edge_indices[edge_0, 2]
    edge_0_vertex_1 = edge_indices[edge_0, 3]
    edge_1_vertex_0 = edge_indices[edge_1, 2]
    edge_1_vertex_1 = edge_indices[edge_1, 3]
    if rest_exclusion_radius > 0.0:
        rest_closest = wp.closest_point_edge_edge(
            particle_q_rest[edge_0_vertex_0],
            particle_q_rest[edge_0_vertex_1],
            particle_q_rest[edge_1_vertex_0],
            particle_q_rest[edge_1_vertex_1],
            edge_parallel_epsilon,
        )
        if rest_closest[2] < rest_exclusion_radius:
            return
    closest = wp.closest_point_edge_edge(
        particle_q[edge_0_vertex_0],
        particle_q[edge_0_vertex_1],
        particle_q[edge_1_vertex_0],
        particle_q[edge_1_vertex_1],
        edge_parallel_epsilon,
    )
    parameter_0 = closest[0]
    parameter_1 = closest[1]
    distance = closest[2]
    if distance <= SOFT_CONTACT_GEOMETRY_EPSILON or distance >= margin + gap:
        return
    point_0 = (1.0 - parameter_0) * particle_q[edge_0_vertex_0] + parameter_0 * particle_q[edge_0_vertex_1]
    point_1 = (1.0 - parameter_1) * particle_q[edge_1_vertex_0] + parameter_1 * particle_q[edge_1_vertex_1]
    normal = (point_0 - point_1) / distance
    if normal_cone_filtering and distance > normal_cone_filtering_min_distance:
        # The normal points from edge 1 towards edge 0, opposite the feature-to-query direction of edge 0
        if not edge_feature_normal_cone_contains(
            edge_0, parameter_0, -normal, particle_q, edge_indices, adjacency, edge_cone_axis, edge_cone_cosine
        ):
            return
        if not edge_feature_normal_cone_contains(
            edge_1, parameter_1, normal, particle_q, edge_indices, adjacency, edge_cone_axis, edge_cone_cosine
        ):
            return
    _write_self_contact(
        row,
        wp.vec4i(edge_0_vertex_0, edge_0_vertex_1, edge_1_vertex_0, edge_1_vertex_1),
        wp.vec4f(1.0 - parameter_0, parameter_0, -1.0 + parameter_1, -parameter_1),
        normal,
        distance - margin,
        particle_qd,
        particle_flags,
        newton_to_packed,
        particle_world,
        particle_inverse_weight,
        model_time_dt,
        stabilization_fraction,
        dead_zone,
        friction,
        restitution,
        row_world,
        row_status,
        row_particle,
        row_coefficient,
        row_frame,
        row_gap,
        row_bias,
        row_friction,
    )


###
# Interfaces
###


class DeformableSelfContacts:
    """Detect the self-contacts of the deformable surfaces and load them into particle contact rows.

    Args:
        model: Newton model of the deformable bodies.
        margin: Detection margin of the self-contacts [m].
        gap: Speculative self-contact distance [m].
        vertex_buffer_size: Largest number of candidate triangles of each surface vertex.
        edge_buffer_size: Largest number of candidate edges of each surface edge.
        topological_filter_threshold: Mesh-ring distance within which the candidates are excluded.
        rest_exclusion_radius: Rest-pose distance within which the candidates are excluded [m].
        edge_parallel_epsilon: Tolerance that classifies nearly parallel edges.
        normal_cone_filtering: Whether to keep only the candidates in the normal cones of their surface features.
        normal_cone_filtering_min_distance: Separation below which the candidates bypass the normal-cone
            filter [m].
    """

    def __init__(
        self,
        model: Model,
        *,
        margin: float,
        gap: float,
        vertex_buffer_size: int,
        edge_buffer_size: int,
        topological_filter_threshold: int,
        rest_exclusion_radius: float,
        edge_parallel_epsilon: float,
        normal_cone_filtering: bool = True,
        normal_cone_filtering_min_distance: float = 1.0e-4,
    ):
        if model.tri_count < 1:
            raise ValueError("LOX deformable self-contacts require a triangulated surface.")
        self.model = model
        self.device = wp.get_device(model.device)
        self.margin = float(margin)
        self.gap = float(gap)
        self.rest_exclusion_radius = float(rest_exclusion_radius)
        self.edge_parallel_epsilon = float(edge_parallel_epsilon)
        self.normal_cone_filtering = normal_cone_filtering
        self.normal_cone_filtering_min_distance = float(normal_cone_filtering_min_distance)
        self.vertex_buffer_size = vertex_buffer_size
        self.edge_buffer_size = edge_buffer_size
        surface_adjacency = MeshAdjacency(tri_indices=model.tri_indices.numpy()).init_vertex_adjacency(
            model.particle_count
        )
        self.edge_indices = wp.array(surface_adjacency.edge_indices, dtype=wp.int32, device=self.device)
        """Surface edges as ``(opposite_0, opposite_1, vertex_0, vertex_1)``."""
        self.adjacency = surface_adjacency.to(self.device)
        """Vertex adjacency of the surface edges."""
        edge_count = self.edge_indices.shape[0]
        self.edge_cone_axis = wp.zeros(edge_count, dtype=wp.vec3f, device=self.device)
        """Normal-cone axis of each surface edge."""
        self.edge_cone_cosine = wp.full(edge_count, -1.0, dtype=wp.float32, device=self.device)
        """Cosine of the normal-cone half angle of each surface edge, or ``-1`` for an unavailable cone."""
        self.detector = TriMeshCollisionDetector(
            _SurfaceCollisionModel(model, self.edge_indices, surface_adjacency),
            init_collision_info=True,
            vertex_collision_buffer_pre_alloc=vertex_buffer_size,
            edge_collision_buffer_pre_alloc=edge_buffer_size,
            edge_edge_parallel_epsilon=edge_parallel_epsilon,
            topological_contact_filter_threshold=topological_filter_threshold,
        )
        """Proximity detector of the vertex-triangle and edge-edge candidates."""
        self.vertex_capacity = self.detector.vertex_colliding_triangles.shape[0] // 2
        """Number of vertex-triangle candidates."""
        self.edge_capacity = self.detector.edge_colliding_edges.shape[0] // 2
        """Number of edge-edge candidates."""
        self.capacity = self.vertex_capacity + self.edge_capacity
        """Number of self-contact rows."""
        self._overflow_warning_emitted = wp.zeros(2, dtype=wp.int32, device=self.device)

    ###
    # Public API
    ###

    def detect(self, particle_q: wp.array[wp.vec3f]) -> None:
        """Refit the proximity structures and freeze the candidates of the time step."""
        detector = self.detector
        detector.resize_flags.zero_()
        if self.normal_cone_filtering:
            wp.launch(
                _compute_edge_normal_cones,
                dim=self.edge_indices.shape[0],
                inputs=[particle_q, self.edge_indices],
                outputs=[self.edge_cone_axis, self.edge_cone_cosine],
                device=self.device,
            )
        query_radius = self.margin + self.gap
        detector.refit(particle_q)
        detector.vertex_triangle_collision_detection(
            query_radius,
            min_query_radius=self.rest_exclusion_radius,
            min_distance_filtering_ref_pos=self.model.particle_q,
        )
        detector.edge_edge_collision_detection(
            query_radius,
            min_query_radius=self.rest_exclusion_radius,
            min_distance_filtering_ref_pos=self.model.particle_q,
        )
        wp.launch(
            _warn_candidate_buffer_overflow,
            dim=1,
            inputs=[detector.resize_flags, self.vertex_buffer_size, self.edge_buffer_size],
            outputs=[self._overflow_warning_emitted],
            device=self.device,
        )

    def load(
        self,
        rows: ParticleContactRows,
        particle_q: wp.array[wp.vec3f],
        particle_qd: wp.array[wp.vec3f],
        newton_to_packed: wp.array[wp.int32],
        particle_world: wp.array[wp.int32],
        particle_inverse_weight: wp.array[wp.float32],
        time_step: wp.array[wp.float32],
        stabilization_fraction: float,
        dead_zone: float,
    ) -> None:
        """Load the detected candidates into the self-contact rows, after the rigid-soft rows.

        Args:
            rows: Particle contact rows.
            particle_q: Begin-step particle positions [m].
            particle_qd: Begin-step particle velocities [m/s].
            newton_to_packed: Packed particle of each Newton particle.
            particle_world: World of each packed particle.
            particle_inverse_weight: Inverse weight of each packed particle; zero for prescribed particles.
            time_step: Time step of each world [s].
            stabilization_fraction: Fraction of the contact penetration corrected in one step.
            dead_zone: Symmetric contact distance dead zone of the stabilization [m].
        """
        if self.capacity == 0:
            return
        model = self.model
        wp.launch(
            _load_self_contacts,
            dim=self.capacity,
            inputs=[
                rows.rigid_capacity,
                self.vertex_capacity,
                self.detector.vertex_colliding_triangles,
                self.detector.edge_colliding_edges,
                particle_q,
                model.particle_q,
                particle_qd,
                model.particle_flags,
                model.tri_indices,
                self.edge_indices,
                self.adjacency,
                self.edge_cone_axis,
                self.edge_cone_cosine,
                self.normal_cone_filtering,
                self.normal_cone_filtering_min_distance,
                newton_to_packed,
                particle_world,
                particle_inverse_weight,
                self.rest_exclusion_radius,
                self.margin,
                self.gap,
                self.edge_parallel_epsilon,
                time_step,
                stabilization_fraction,
                dead_zone,
                float(model.soft_contact_mu),
                float(model.soft_contact_restitution),
            ],
            outputs=[
                rows.world,
                rows.status,
                rows.particle,
                rows.coefficient,
                rows.body,
                rows.shape,
                rows.frame,
                rows.body_jacobian,
                rows.gap,
                rows.bias,
                rows.friction,
                rows.reaction,
            ],
            device=self.device,
        )
