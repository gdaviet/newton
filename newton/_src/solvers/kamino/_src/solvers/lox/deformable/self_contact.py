# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Self-contacts of the triangulated surfaces of the LOX deformable bodies.

:class:`DeformableSelfContacts` detects the vertex-triangle and edge-edge proximity
candidates of the deformable surfaces with VBD's triangle-mesh collision detector at the
beginning of each time step, keeps those whose direction lies in the normal cones of
the surface features on both sides (with the face normal of a concave feature), and
writes them as particle contact rows after the rigid-soft
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
    CONTACT_CONE_INSIDE,
    CONTACT_CONE_OUTSIDE,
    CONTACT_CONE_PENETRATING,
    SOFT_CONTACT_GEOMETRY_EPSILON,
    classify_surface_contact,
    surface_point_is_behind,
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
def _pair_normal(
    first_verdict: int,
    first_face_normal: wp.vec3,
    second_verdict: int,
    second_face_normal: wp.vec3,
    normal: wp.vec3,
):
    """Select whether a self-contact keeps its row and the normal from the second towards the first primitive.

    A penetrated feature keeps the contact along its outward normal whatever the other feature;
    otherwise a feature outside its cone drops the contact, and a concave feature presses along
    its face normal.
    """
    if first_verdict == CONTACT_CONE_PENETRATING:
        return True, -first_face_normal
    if second_verdict == CONTACT_CONE_PENETRATING:
        return True, second_face_normal
    if first_verdict == CONTACT_CONE_OUTSIDE or second_verdict == CONTACT_CONE_OUTSIDE:
        return False, normal
    if first_verdict != CONTACT_CONE_INSIDE:
        return True, -first_face_normal
    if second_verdict != CONTACT_CONE_INSIDE:
        return True, second_face_normal
    return True, normal


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
    max_separation_velocity: wp.float32,
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
    row_body: wp.array[wp.int32],
    row_shape: wp.array[wp.int32],
    row_body_jacobian: wp.array[mat36f],
    row_reaction: wp.array[wp.vec3f],
):
    """Write one self-contact row; the prescribed particles enter its bias.

    The separation recovery of a row closer than its target stays within the largest separation velocity.
    """
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
    time_step = model_time_dt[wid]
    recovery_gap = gap
    if stabilization_fraction > 0.0:
        recovery_gap = wp.max(gap, -(max_separation_velocity * time_step / stabilization_fraction + dead_zone))
    velocity_target = compute_contact_velocity_target(
        recovery_gap, wp.dot(normal, relative_velocity), restitution, time_step, stabilization_fraction, dead_zone
    )
    frame = make_contact_frame_znorm(normal)
    row_world[row] = wid
    row_frame[row] = frame
    row_gap[row] = gap
    row_bias[row] = wp.transpose(frame) @ (prescribed_velocity - velocity_target * normal)
    row_friction[row] = friction
    row_body[row] = -1
    row_shape[row] = -1
    row_body_jacobian[row] = mat36f(0.0)
    row_reaction[row] = wp.vec3f(0.0)
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
    face_side: wp.array[wp.float32],
    normal_cone_filtering: wp.bool,
    normal_cone_filtering_min_distance: wp.float32,
    newton_to_packed: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    particle_inverse_weight: wp.array[wp.float32],
    margin: wp.float32,
    gap: wp.float32,
    edge_parallel_epsilon: wp.float32,
    model_time_dt: wp.array[wp.float32],
    stabilization_fraction: wp.float32,
    max_separation_velocity: wp.float32,
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

    Each row targets the margin. The candidates beyond the detection distance or outside the normal
    cones of their surface features leave their rows unused.
    """
    candidate = wp.tid()
    row = row_offset + candidate
    # The consumers read the rows with an active status only; the row writer fills every field of
    # the kept candidates
    row_world[row] = -1
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
        difference = particle_q[vertex] - closest
        distance = wp.length(difference)
        if distance <= SOFT_CONTACT_GEOMETRY_EPSILON or distance >= margin + gap:
            return
        normal = difference / distance
        if normal_cone_filtering and distance > normal_cone_filtering_min_distance:
            # A pair facing across the interior of a soft body in the rest pose is no contact
            rest_closest, rest_barycentric, _rest_feature = triangle_closest_point(
                particle_q_rest[triangle_0],
                particle_q_rest[triangle_1],
                particle_q_rest[triangle_2],
                particle_q_rest[vertex],
            )
            if surface_point_is_behind(
                wp.vec3i(triangle_0, triangle_1, triangle_2),
                rest_barycentric,
                rest_closest,
                particle_q_rest[vertex],
                particle_q_rest,
                tri_indices,
                edge_indices,
                adjacency,
                face_side,
            ) or surface_point_is_behind(
                wp.vec3i(vertex, -1, -1),
                wp.vec3(1.0, 0.0, 0.0),
                particle_q_rest[vertex],
                rest_closest,
                particle_q_rest,
                tri_indices,
                edge_indices,
                adjacency,
                face_side,
            ):
                return
            # Each feature is tested with the direction towards the other primitive; the normal
            # points from the triangle towards the vertex.
            vertex_verdict, vertex_face_normal = classify_surface_contact(
                wp.vec3i(vertex, -1, -1),
                wp.vec3(1.0, 0.0, 0.0),
                -normal,
                particle_q,
                tri_indices,
                edge_indices,
                adjacency,
                face_side,
            )
            triangle_verdict, triangle_face_normal = classify_surface_contact(
                wp.vec3i(triangle_0, triangle_1, triangle_2),
                barycentric,
                normal,
                particle_q,
                tri_indices,
                edge_indices,
                adjacency,
                face_side,
            )
            keep, normal = _pair_normal(
                vertex_verdict, vertex_face_normal, triangle_verdict, triangle_face_normal, normal
            )
            if not keep:
                return
        _write_self_contact(
            row,
            wp.vec4i(triangle_0, triangle_1, triangle_2, vertex),
            wp.vec4f(-barycentric[0], -barycentric[1], -barycentric[2], 1.0),
            normal,
            wp.dot(normal, difference) - margin,
            particle_qd,
            particle_flags,
            newton_to_packed,
            particle_world,
            particle_inverse_weight,
            model_time_dt,
            stabilization_fraction,
            max_separation_velocity,
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
            row_body,
            row_shape,
            row_body_jacobian,
            row_reaction,
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
        # A pair facing across the interior of a soft body in the rest pose is no contact
        rest_closest = wp.closest_point_edge_edge(
            particle_q_rest[edge_0_vertex_0],
            particle_q_rest[edge_0_vertex_1],
            particle_q_rest[edge_1_vertex_0],
            particle_q_rest[edge_1_vertex_1],
            edge_parallel_epsilon,
        )
        rest_parameter_0 = rest_closest[0]
        rest_parameter_1 = rest_closest[1]
        rest_point_0 = (1.0 - rest_parameter_0) * particle_q_rest[edge_0_vertex_0] + rest_parameter_0 * particle_q_rest[
            edge_0_vertex_1
        ]
        rest_point_1 = (1.0 - rest_parameter_1) * particle_q_rest[edge_1_vertex_0] + rest_parameter_1 * particle_q_rest[
            edge_1_vertex_1
        ]
        if surface_point_is_behind(
            wp.vec3i(edge_0_vertex_0, edge_0_vertex_1, -1),
            wp.vec3(1.0 - rest_parameter_0, rest_parameter_0, 0.0),
            rest_point_0,
            rest_point_1,
            particle_q_rest,
            tri_indices,
            edge_indices,
            adjacency,
            face_side,
        ) or surface_point_is_behind(
            wp.vec3i(edge_1_vertex_0, edge_1_vertex_1, -1),
            wp.vec3(1.0 - rest_parameter_1, rest_parameter_1, 0.0),
            rest_point_1,
            rest_point_0,
            particle_q_rest,
            tri_indices,
            edge_indices,
            adjacency,
            face_side,
        ):
            return
        # The normal points from edge 1 towards edge 0, so edge 0 sees the other edge along its opposite
        verdict_0, face_normal_0 = classify_surface_contact(
            wp.vec3i(edge_0_vertex_0, edge_0_vertex_1, -1),
            wp.vec3(1.0 - parameter_0, parameter_0, 0.0),
            -normal,
            particle_q,
            tri_indices,
            edge_indices,
            adjacency,
            face_side,
        )
        verdict_1, face_normal_1 = classify_surface_contact(
            wp.vec3i(edge_1_vertex_0, edge_1_vertex_1, -1),
            wp.vec3(1.0 - parameter_1, parameter_1, 0.0),
            normal,
            particle_q,
            tri_indices,
            edge_indices,
            adjacency,
            face_side,
        )
        keep, normal = _pair_normal(verdict_0, face_normal_0, verdict_1, face_normal_1, normal)
        if not keep:
            return
    _write_self_contact(
        row,
        wp.vec4i(edge_0_vertex_0, edge_0_vertex_1, edge_1_vertex_0, edge_1_vertex_1),
        wp.vec4f(1.0 - parameter_0, parameter_0, -1.0 + parameter_1, -parameter_1),
        normal,
        wp.dot(normal, point_0 - point_1) - margin,
        particle_qd,
        particle_flags,
        newton_to_packed,
        particle_world,
        particle_inverse_weight,
        model_time_dt,
        stabilization_fraction,
        max_separation_velocity,
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
        row_body,
        row_shape,
        row_body_jacobian,
        row_reaction,
    )


###
# Interfaces
###


class DeformableSelfContacts:
    """Detect the self-contacts of the deformable surfaces and load them into particle contact rows.

    Args:
        model: Newton model of the deformable bodies.
        margin: Detection margin and target separation of the self-contacts [m].
        gap: Speculative self-contact distance [m].
        vertex_buffer_size: Largest number of candidate triangles of each surface vertex.
        edge_buffer_size: Largest number of candidate edges of each surface edge.
        topological_filter_threshold: Mesh-ring distance within which the candidates are excluded.
        edge_parallel_epsilon: Tolerance that classifies nearly parallel edges.
        stabilization_fraction: Fraction of the separation deficit recovered in one step.
        max_separation_velocity: Largest normal velocity that recovers the separation [m/s].
        normal_cone_filtering: Whether to keep only the candidates in the normal cones of their surface features.
        normal_cone_filtering_min_distance: Separation below which the candidates bypass the normal-cone
            filter [m].
        record_triangle_distances: Whether the detector reduces the smallest vertex distance of each triangle.
        face_side: Side of each surface triangle (zero for cloth, outward sign for a soft body boundary),
            or ``None`` for two-sided triangles.
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
        edge_parallel_epsilon: float,
        stabilization_fraction: float,
        max_separation_velocity: float,
        normal_cone_filtering: bool = True,
        normal_cone_filtering_min_distance: float = 1.0e-4,
        record_triangle_distances: bool = False,
        face_side: wp.array[wp.float32] | None = None,
    ):
        if model.tri_count < 1:
            raise ValueError("LOX deformable self-contacts require a triangulated surface.")
        self.model = model
        self.device = wp.get_device(model.device)
        if face_side is None:
            face_side = wp.zeros(model.tri_count, dtype=wp.float32, device=self.device)
        self.face_side = face_side
        """Side of each surface triangle; zero for a two-sided triangle."""
        self.margin = float(margin)
        self.gap = float(gap)
        self.edge_parallel_epsilon = float(edge_parallel_epsilon)
        self.stabilization_fraction = float(stabilization_fraction)
        self.max_separation_velocity = float(max_separation_velocity)
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
        """Vertex, edge, and triangle adjacency of the surface."""
        self.detector = TriMeshCollisionDetector(
            _SurfaceCollisionModel(model, self.edge_indices, surface_adjacency),
            record_triangle_contacting_vertices=record_triangle_distances,
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
        query_radius = self.margin + self.gap
        detector.refit(particle_q)
        detector.vertex_triangle_collision_detection(query_radius)
        detector.edge_edge_collision_detection(query_radius)
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
                self.face_side,
                self.normal_cone_filtering,
                self.normal_cone_filtering_min_distance,
                newton_to_packed,
                particle_world,
                particle_inverse_weight,
                self.margin,
                self.gap,
                self.edge_parallel_epsilon,
                time_step,
                self.stabilization_fraction,
                self.max_separation_velocity,
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
