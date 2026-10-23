# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Penetration-free step limiting of the contacts of the LOX deformable bodies.

:class:`DeformablePenetrationLimiter` keeps the straight displacement of each self-contact
candidate from the begin-step surface on its side of a certified division plane, with the
divide-and-truncate planes of VBD. The certified separators of the self-contact candidates are
found once per time step into a dense list, and each iteration places their planes within the
gaps from the current displacements, as the separators depend on the begin-step positions only.
Each rigid-soft record of the
collision pipeline adds a plane through its rigid surface point, placed at every iteration
beyond the predicted advance of that point under the accepted body twist, so that only the
particles are limited. An averaged projection sweep first removes the normal displacement
that crosses each plane, which keeps the tangential motion; a truncation of the remaining
step along its direction then certifies every plane. The displacement of each particle stays
within half the self-contact detection distance, and within the conservative bound of the
smallest detected distances when a candidate buffer overflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .......geometry.kernels import EDGE_COLLISION_BUFFER_OVERFLOW_INDEX, VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX
from .......utils.mesh import (
    MeshAdjacencyData,
    get_vertex_adjacent_edge_id_order,
    get_vertex_adjacent_face_id_order,
    get_vertex_num_adjacent_edges,
    get_vertex_num_adjacent_faces,
)
from ......vbd.rigid_vbd_kernels import (
    dat_separation_epsilon,
    find_edge_edge_separator,
    find_vertex_triangle_separator,
    place_dat_division_plane,
    planar_truncation_t,
)
from ....core.types import vec6f

if TYPE_CHECKING:
    from .......sim import Contacts
    from .self_contact import DeformableSelfContacts
    from .types import DeformableTopology

###
# Module interface
###

__all__ = ["DeformablePenetrationLimiter"]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


@wp.func
def _candidate_particles(
    candidate: int,
    vertex_capacity: int,
    vertex_records: wp.array[wp.int32],
    edge_records: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
):
    """Return the four particles of a candidate and their sides of the division plane.

    The plane normal points towards the vertex of a vertex-triangle candidate and towards the first
    edge of an edge-edge candidate.
    """
    if candidate < vertex_capacity:
        triangle = vertex_records[2 * candidate + 1]
        return (
            wp.vec4i(
                vertex_records[2 * candidate],
                tri_indices[triangle, 0],
                tri_indices[triangle, 1],
                tri_indices[triangle, 2],
            ),
            wp.vec4(1.0, -1.0, -1.0, -1.0),
        )
    edge_candidate = candidate - vertex_capacity
    edge_0 = edge_records[2 * edge_candidate]
    edge_1 = edge_records[2 * edge_candidate + 1]
    return (
        wp.vec4i(edge_indices[edge_0, 2], edge_indices[edge_0, 3], edge_indices[edge_1, 2], edge_indices[edge_1, 3]),
        wp.vec4(1.0, 1.0, -1.0, -1.0),
    )


@wp.func
def _plane_index(
    tid: int,
    self_capacity: int,
    self_count: wp.array[wp.int32],
    rigid_count: wp.array[wp.int32],
    plane_normal: wp.array[wp.vec3],
) -> int:
    """Return the plane of a thread, or ``-1`` past the self-contact and rigid-soft plane counts.

    The self-contact planes fill the front of the plane arrays densely; the rigid-soft planes
    follow at the offset of the self-contact capacity, one slot per collision record.
    """
    if tid < self_capacity:
        if tid >= self_count[0]:
            return -1
        return tid
    if tid - self_capacity >= rigid_count[0]:
        return -1
    if wp.length_sq(plane_normal[tid]) == 0.0:
        return -1
    return tid


@wp.func
def _vertex_triangle_separator(
    vertex: int,
    triangle: int,
    reference_position: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
):
    """Find the certified separator of a vertex and a triangle at the begin-step surface.

    Returns the normal towards the vertex (zero for an unconstrained pair), the triangle support
    point, the gap between the supports, and the separation clearance. A pair without a strict
    separator stays unconstrained; its contact row separates it, and a later detection certifies
    a separator.
    """
    if vertex < 0 or triangle < 0 or get_vertex_num_adjacent_edges(adjacency, vertex) == 0:
        return wp.vec3(0.0), wp.vec3(0.0), 0.0, 0.0
    v = reference_position[vertex]
    a = reference_position[tri_indices[triangle, 0]]
    b = reference_position[tri_indices[triangle, 1]]
    c = reference_position[tri_indices[triangle, 2]]
    coordinate_scale = wp.max(
        1.0, wp.max(wp.max(wp.max(wp.abs(v)), wp.max(wp.abs(a))), wp.max(wp.max(wp.abs(b)), wp.max(wp.abs(c))))
    )
    epsilon = dat_separation_epsilon(coordinate_scale)
    valid, normal, _vertex_support, triangle_support, gap, _candidate = find_vertex_triangle_separator(
        v, a, b, c, wp.vec3(0.0), epsilon
    )
    if not valid:
        return wp.vec3(0.0), wp.vec3(0.0), 0.0, 0.0
    return normal, triangle_support, gap, epsilon


@wp.func
def _edge_edge_separator(
    edge_0: int,
    edge_1: int,
    reference_position: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
):
    """Find the certified separator of two edges at the begin-step surface.

    Returns the normal towards the first edge (zero for an unconstrained pair), the support point
    of the second edge, the gap between the supports, and the separation clearance.
    """
    if edge_0 < 0 or edge_1 < 0 or edge_0 == edge_1:
        return wp.vec3(0.0), wp.vec3(0.0), 0.0, 0.0
    a0 = reference_position[edge_indices[edge_0, 2]]
    b0 = reference_position[edge_indices[edge_0, 3]]
    a1 = reference_position[edge_indices[edge_1, 2]]
    b1 = reference_position[edge_indices[edge_1, 3]]
    coordinate_scale = wp.max(
        1.0, wp.max(wp.max(wp.max(wp.abs(a0)), wp.max(wp.abs(b0))), wp.max(wp.max(wp.abs(a1)), wp.max(wp.abs(b1))))
    )
    epsilon = dat_separation_epsilon(coordinate_scale)
    valid, normal, _edge_0_support, edge_1_support, gap, _candidate = find_edge_edge_separator(
        a0, b0, a1, b1, wp.vec3(0.0), epsilon
    )
    if not valid:
        return wp.vec3(0.0), wp.vec3(0.0), 0.0, 0.0
    return normal, edge_1_support, gap, epsilon


###
# Kernels
###


@wp.kernel
def _compute_particle_displacements(
    # Inputs:
    model_time_dt: wp.array[wp.float32],
    packed_to_newton: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    projected_velocity: wp.array[wp.vec3],
    # Outputs:
    displacement: wp.array[wp.vec3],
):
    """Form the displacement ``dt p`` of each particle, in Newton order."""
    pid = wp.tid()
    displacement[packed_to_newton[pid]] = model_time_dt[particle_world[pid]] * projected_velocity[pid]


@wp.kernel
def _build_division_separators(
    # Inputs:
    vertex_capacity: wp.int32,
    vertex_records: wp.array[wp.int32],
    edge_records: wp.array[wp.int32],
    reference_position: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    # Outputs:
    plane_count: wp.array[wp.int32],
    plane_particles: wp.array[wp.vec4i],
    plane_sides: wp.array[wp.vec4],
    plane_normal: wp.array[wp.vec3],
    plane_support: wp.array[wp.vec3],
    plane_gap: wp.array[wp.float32],
    plane_epsilon: wp.array[wp.float32],
):
    """Find the certified separator of each self-contact candidate at the begin-step surface, once per time step.

    The separators depend on the begin-step positions only and are appended densely; a candidate
    without a separator leaves no plane. Each iteration places its plane within the gap.
    """
    candidate = wp.tid()
    normal = wp.vec3(0.0)
    support = wp.vec3(0.0)
    gap = float(0.0)
    epsilon = float(0.0)
    if candidate < vertex_capacity:
        normal, support, gap, epsilon = _vertex_triangle_separator(
            vertex_records[2 * candidate],
            vertex_records[2 * candidate + 1],
            reference_position,
            tri_indices,
            adjacency,
        )
    else:
        edge_candidate = candidate - vertex_capacity
        normal, support, gap, epsilon = _edge_edge_separator(
            edge_records[2 * edge_candidate],
            edge_records[2 * edge_candidate + 1],
            reference_position,
            edge_indices,
        )
    if wp.length_sq(normal) == 0.0:
        return
    particles, sides = _candidate_particles(
        candidate, vertex_capacity, vertex_records, edge_records, tri_indices, edge_indices
    )
    slot = wp.atomic_add(plane_count, 0, 1)
    plane_particles[slot] = particles
    plane_sides[slot] = sides
    plane_normal[slot] = normal
    plane_support[slot] = support
    plane_gap[slot] = gap
    plane_epsilon[slot] = epsilon


@wp.kernel
def _build_rigid_soft_planes(
    # Inputs:
    plane_offset: wp.int32,
    contact_count: wp.array[wp.int32],
    contact_particles: wp.array[wp.vec3i],
    contact_coefficients: wp.array[wp.vec3f],
    contact_shape: wp.array[wp.int32],
    contact_body_position: wp.array[wp.vec3f],
    contact_normal: wp.array[wp.vec3f],
    shape_body: wp.array[wp.int32],
    body_com: wp.array[wp.vec3f],
    body_pose: wp.array[wp.transformf],
    projected_twist: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    projected_twist_fraction: wp.float32,
    newton_to_packed: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    model_time_dt: wp.array[wp.float32],
    reference_position: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
    # Outputs:
    plane_particles: wp.array[wp.vec4i],
    plane_sides: wp.array[wp.vec4],
    plane_normal: wp.array[wp.vec3],
    plane_point: wp.array[wp.vec3],
    plane_epsilon: wp.array[wp.float32],
):
    """Place the division plane of each rigid-soft record from the begin-step surfaces.

    The plane passes through the rigid surface point of the record with the contact normal,
    which points from the rigid surface towards the particle feature. The rigid side keeps the
    predicted advance of its surface point under the accepted twist, including a bound on the
    deviation of its arc from the chord, so the plane only limits the particles. A record that
    starts inside the collider holds its particles from moving deeper.
    """
    cid = wp.tid()
    if cid >= contact_count[0]:
        return
    slot = plane_offset + cid
    plane_normal[slot] = wp.vec3(0.0)
    particles = contact_particles[cid]
    normal_length = wp.length(contact_normal[cid])
    if particles[0] < 0 or normal_length == 0.0:
        return
    normal = contact_normal[cid] / normal_length

    feature_position = wp.vec3(0.0)
    soft_approach = float(0.0)
    coordinate_scale = float(1.0)
    for corner in range(3):
        particle = particles[corner]
        if particle >= 0:
            position = reference_position[particle]
            feature_position += contact_coefficients[cid][corner] * position
            soft_approach = wp.max(soft_approach, -wp.dot(normal, displacement[particle]))
            coordinate_scale = wp.max(coordinate_scale, wp.max(wp.abs(position)))

    anchor = contact_body_position[cid]
    rigid_approach = float(0.0)
    body = shape_body[contact_shape[cid]]
    if body >= 0 and body < body_pose.shape[0]:
        pose = body_pose[body]
        center_of_mass = wp.transform_get_translation(pose)
        lever = wp.quat_rotate(wp.transform_get_rotation(pose), contact_body_position[cid] - body_com[body])
        anchor = center_of_mass + lever
        twist = projected_twist_fraction * projected_twist[body] + (1.0 - projected_twist_fraction) * global_twist[body]
        time_step = model_time_dt[particle_world[newton_to_packed[particles[0]]]]
        angular_step = time_step * wp.vec3(twist[3], twist[4], twist[5])
        chord = time_step * wp.vec3(twist[0], twist[1], twist[2]) + wp.cross(angular_step, lever)
        rigid_approach = wp.max(wp.dot(normal, chord), 0.0) + 0.5 * wp.length_sq(angular_step) * wp.length(lever)
    coordinate_scale = wp.max(coordinate_scale, wp.max(wp.abs(anchor)))
    epsilon = dat_separation_epsilon(coordinate_scale)

    gap = wp.max(wp.dot(normal, feature_position - anchor), 0.0)
    point, _fraction = place_dat_division_plane(normal, anchor, gap, soft_approach, rigid_approach, epsilon)
    # The rigid side keeps its predicted advance and the separation band below the plane
    offset = wp.max(wp.dot(normal, point - anchor), rigid_approach + epsilon)
    plane_particles[slot] = wp.vec4i(particles[0], particles[1], particles[2], -1)
    plane_sides[slot] = wp.vec4(1.0, 1.0, 1.0, 0.0)
    plane_normal[slot] = normal
    plane_point[slot] = anchor + offset * normal
    plane_epsilon[slot] = epsilon


@wp.kernel
def _accumulate_plane_projections(
    # Inputs:
    self_capacity: wp.int32,
    self_count: wp.array[wp.int32],
    rigid_count: wp.array[wp.int32],
    plane_particles: wp.array[wp.vec4i],
    plane_sides: wp.array[wp.vec4],
    plane_normal: wp.array[wp.vec3],
    plane_support: wp.array[wp.vec3],
    plane_gap: wp.array[wp.float32],
    plane_epsilon: wp.array[wp.float32],
    reference_position: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
    relaxation: wp.float32,
    # Outputs:
    plane_point: wp.array[wp.vec3],
    correction: wp.array[wp.vec3],
    correction_count: wp.array[wp.float32],
):
    """Place each self-contact plane for the current displacements and accumulate the normal correction that
    keeps each particle within the relaxed side of its planes.

    A self-contact plane splits the gap of its separator in proportion to the approach of each
    side along the normal. A particle may approach a plane by the relaxed fraction of its
    begin-step distance, and a particle on the wrong side may only move towards its side.
    """
    plane = _plane_index(wp.tid(), self_capacity, self_count, rigid_count, plane_normal)
    if plane < 0:
        return
    particles = plane_particles[plane]
    sides = plane_sides[plane]
    normal = plane_normal[plane]
    epsilon = plane_epsilon[plane]
    if plane < self_capacity:
        positive_approach = float(0.0)
        negative_approach = float(0.0)
        for slot in range(4):
            particle = particles[slot]
            if particle >= 0:
                approach = wp.dot(normal, displacement[particle])
                if sides[slot] > 0.0:
                    positive_approach = wp.max(positive_approach, -approach)
                else:
                    negative_approach = wp.max(negative_approach, approach)
        placed, _fraction = place_dat_division_plane(
            normal, plane_support[plane], plane_gap[plane], positive_approach, negative_approach, epsilon
        )
        plane_point[plane] = placed
    point = plane_point[plane]
    for slot in range(4):
        particle = particles[slot]
        if particle >= 0:
            side_normal = sides[slot] * normal
            distance = wp.dot(side_normal, reference_position[particle] - point) - epsilon
            allowed = -relaxation * wp.max(distance, 0.0)
            violation = allowed - wp.dot(side_normal, displacement[particle])
            if violation > 0.0:
                wp.atomic_add(correction, particle, violation * side_normal)
                wp.atomic_add(correction_count, particle, 1.0)


@wp.kernel
def _apply_plane_projections(
    # Outputs:
    correction: wp.array[wp.vec3],
    correction_count: wp.array[wp.float32],
    displacement: wp.array[wp.vec3],
    truncation: wp.array[wp.float32],
):
    """Move each particle by the average of its plane corrections, clear the accumulators, and reset its step
    fraction."""
    particle = wp.tid()
    count = correction_count[particle]
    if count > 0.0:
        displacement[particle] = displacement[particle] + correction[particle] / count
        correction[particle] = wp.vec3(0.0)
        correction_count[particle] = 0.0
    truncation[particle] = 1.0


@wp.kernel
def _accumulate_directional_bounds(
    # Inputs:
    self_capacity: wp.int32,
    self_count: wp.array[wp.int32],
    rigid_count: wp.array[wp.int32],
    plane_particles: wp.array[wp.vec4i],
    plane_sides: wp.array[wp.vec4],
    plane_normal: wp.array[wp.vec3],
    plane_point: wp.array[wp.vec3],
    plane_epsilon: wp.array[wp.float32],
    reference_position: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
    relaxation: wp.float32,
    # Outputs:
    truncation: wp.array[wp.float32],
):
    """Reduce the smallest step fraction of each particle over its planes."""
    plane = _plane_index(wp.tid(), self_capacity, self_count, rigid_count, plane_normal)
    if plane < 0:
        return
    particles = plane_particles[plane]
    sides = plane_sides[plane]
    normal = plane_normal[plane]
    point = plane_point[plane]
    epsilon = plane_epsilon[plane]
    for slot in range(4):
        particle = particles[slot]
        if particle >= 0:
            bound = planar_truncation_t(
                reference_position[particle], displacement[particle], sides[slot] * normal, point, relaxation, epsilon
            )
            if bound < 1.0:
                wp.atomic_min(truncation, particle, bound)


@wp.kernel
def _apply_directional_bounds(
    # Inputs:
    model_time_dt: wp.array[wp.float32],
    max_displacement: wp.float32,
    relaxation: wp.float32,
    resize_flags: wp.array[wp.int32],
    vertex_min_distance: wp.array[wp.float32],
    triangle_min_distance: wp.array[wp.float32],
    edge_min_distance: wp.array[wp.float32],
    adjacency: MeshAdjacencyData,
    newton_to_packed: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    inverse_weight: wp.array[wp.float32],
    displacement: wp.array[wp.vec3],
    truncation: wp.array[wp.float32],
    # Outputs:
    projected_velocity: wp.array[wp.vec3],
):
    """Set the projected velocity of each particle from its truncated displacement, within the displacement cap.

    After a candidate buffer overflow, the cap also holds each particle within half the smallest
    detected distance of its vertex, triangles, and edges, which bounds the dropped candidates.
    """
    particle = wp.tid()
    pid = newton_to_packed[particle]
    wid = particle_world[pid]
    if not world_active[wid]:
        return
    if inverse_weight[pid] <= 0.0:
        projected_velocity[pid] = wp.vec3(0.0)
        return
    step = truncation[particle] * displacement[particle]
    displacement_bound = max_displacement
    if (
        resize_flags[VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX] != 0
        or resize_flags[EDGE_COLLISION_BUFFER_OVERFLOW_INDEX] != 0
    ):
        min_distance = vertex_min_distance[particle]
        for face in range(get_vertex_num_adjacent_faces(adjacency, particle)):
            triangle, _order = get_vertex_adjacent_face_id_order(adjacency, particle, face)
            min_distance = wp.min(min_distance, triangle_min_distance[triangle])
        for adjacent_edge in range(get_vertex_num_adjacent_edges(adjacency, particle)):
            edge, order = get_vertex_adjacent_edge_id_order(adjacency, particle, adjacent_edge)
            if order == 2 or order == 3:
                min_distance = wp.min(min_distance, edge_min_distance[edge])
        displacement_bound = wp.min(displacement_bound, 0.5 * relaxation * min_distance)
    step_length = wp.length(step)
    if step_length > displacement_bound:
        step *= displacement_bound / step_length
    projected_velocity[pid] = step / model_time_dt[wid]


###
# Interfaces
###


class DeformablePenetrationLimiter:
    """Limit the projected particle motion against the self-contact candidates of the time step.

    Args:
        self_contacts: Self-contact detector whose candidates bound the steps; its detector records
            the smallest vertex distance of each triangle.
        topology: Packed particle layout of the deformable bodies.
        inverse_weight: Inverse consensus weight of each packed particle.
        relaxation: Safety relaxation of the directional and isotropic step bounds, in ``(0, 1]``.
        body_pose: Begin-step pose of each body about its center of mass, which places the rigid-soft
            records.
    """

    def __init__(
        self,
        self_contacts: DeformableSelfContacts,
        topology: DeformableTopology,
        inverse_weight: wp.array[wp.float32],
        relaxation: float,
        body_pose: wp.array[wp.transformf],
    ):
        self.self_contacts = self_contacts
        self.topology = topology
        self.inverse_weight = inverse_weight
        self.device = self_contacts.device
        self.relaxation = float(relaxation)
        self.body_pose = body_pose
        model = self_contacts.model
        self.body_com = model.body_com if model.body_count > 0 else wp.empty(0, dtype=wp.vec3f, device=self.device)
        self.rigid_contacts: Contacts | None = None
        """Newton contacts whose rigid-soft records bound the steps, or ``None``."""
        particle_count = int(model.particle_count)
        self.reference_position = wp.zeros(particle_count, dtype=wp.vec3, device=self.device)
        """Begin-step particle positions, in Newton order."""
        self.displacement = wp.zeros(particle_count, dtype=wp.vec3, device=self.device)
        """Displacement of each particle over the time step, in Newton order."""
        self.truncation = wp.ones(particle_count, dtype=wp.float32, device=self.device)
        """Step fraction of each particle, in Newton order."""
        self.correction = wp.zeros(particle_count, dtype=wp.vec3, device=self.device)
        """Accumulated plane correction of each particle, in Newton order."""
        self.correction_count = wp.zeros(particle_count, dtype=wp.float32, device=self.device)
        """Number of plane corrections of each particle, in Newton order."""
        self.self_plane_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        """Number of self-contact planes of the time step."""
        self._no_rigid_planes = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.set_rigid_capacity(0)

    ###
    # Public API
    ###

    def set_rigid_capacity(self, capacity: int) -> None:
        """Size the planes for the self-contact candidates and the soft contact capacity of the Newton contacts."""
        self.rigid_capacity = int(capacity)
        size = self.self_contacts.capacity + self.rigid_capacity
        self.plane_particles = wp.full(size, wp.vec4i(-1), dtype=wp.vec4i, device=self.device)
        """Particles of each plane, in Newton order, padded with ``-1``."""
        self.plane_sides = wp.zeros(size, dtype=wp.vec4, device=self.device)
        """Side of the plane of each particle of each plane."""
        self.plane_normal = wp.zeros(size, dtype=wp.vec3, device=self.device)
        """Division-plane normal of each plane, or zero for an unused rigid-soft slot."""
        self.plane_point = wp.zeros(size, dtype=wp.vec3, device=self.device)
        """Point on each division plane [m]."""
        self.plane_support = wp.zeros(size, dtype=wp.vec3, device=self.device)
        """Support point of the negative side of each self-contact separator [m]."""
        self.plane_gap = wp.zeros(size, dtype=wp.float32, device=self.device)
        """Gap between the supports of each self-contact separator [m]."""
        self.plane_epsilon = wp.zeros(size, dtype=wp.float32, device=self.device)
        """Separation clearance of each division plane [m]."""

    def bind_rigid_contacts(self, contacts: Contacts | None) -> None:
        """Bound the steps of the time step with the rigid-soft records of the Newton contacts."""
        self.rigid_contacts = contacts if contacts is not None and self.rigid_capacity > 0 else None

    def begin_time_step(
        self, particle_q: wp.array[wp.vec3], particle_qd: wp.array[wp.vec3], time_step: wp.array[wp.float32]
    ) -> None:
        """Freeze the begin-step surface of the step bounds and find the self-contact separators.

        Each iteration places the plane of a separator within its gap from the current displacements.

        Args:
            particle_q: Begin-step particle positions [m], in Newton order.
            particle_qd: Begin-step particle velocities [m/s], in Newton order.
            time_step: Time step of each world [s].
        """
        self_contacts = self.self_contacts
        detector = self_contacts.detector
        model = self_contacts.model
        wp.copy(self.reference_position, particle_q)
        self.self_plane_count.zero_()
        wp.launch(
            _build_division_separators,
            dim=self_contacts.capacity,
            inputs=[
                self_contacts.vertex_capacity,
                detector.vertex_colliding_triangles,
                detector.edge_colliding_edges,
                self.reference_position,
                model.tri_indices,
                self_contacts.edge_indices,
                self_contacts.adjacency,
            ],
            outputs=[
                self.self_plane_count,
                self.plane_particles,
                self.plane_sides,
                self.plane_normal,
                self.plane_support,
                self.plane_gap,
                self.plane_epsilon,
            ],
            device=self.device,
        )

    def truncate(
        self,
        projected_velocity: wp.array[wp.vec3],
        world_active: wp.array[wp.bool],
        time_step: wp.array[wp.float32],
        projected_twist: wp.array[vec6f],
        global_twist: wp.array[vec6f],
        projected_twist_fraction: float,
    ) -> None:
        """Limit the projected particle velocities of the active worlds from the begin-step surfaces.

        Args:
            projected_velocity: Projected velocity of each packed particle [m/s], limited in place.
            world_active: Whether each world iterates.
            time_step: Time step of each world [s].
            projected_twist: Projected twist of each body.
            global_twist: Smooth twist of each body.
            projected_twist_fraction: Weight of the projected twist in the accepted body twist.
        """
        self_contacts = self.self_contacts
        detector = self_contacts.detector
        model = self_contacts.model
        particle_count = model.particle_count
        self_capacity = self_contacts.capacity
        wp.launch(
            _compute_particle_displacements,
            dim=projected_velocity.shape[0],
            inputs=[time_step, self.topology.packed_to_newton, self.topology.packed_solve_world, projected_velocity],
            outputs=[self.displacement],
            device=self.device,
        )
        contacts = self.rigid_contacts
        rigid_count = self._no_rigid_planes
        if contacts is not None:
            rigid_count = contacts.soft_contact_count
            wp.launch(
                _build_rigid_soft_planes,
                dim=self.rigid_capacity,
                inputs=[
                    self_capacity,
                    contacts.soft_contact_count,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_barycentric,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_normal,
                    model.shape_body,
                    self.body_com,
                    self.body_pose,
                    projected_twist,
                    global_twist,
                    float(projected_twist_fraction),
                    self.topology.newton_to_packed,
                    self.topology.packed_solve_world,
                    time_step,
                    self.reference_position,
                    self.displacement,
                ],
                outputs=[
                    self.plane_particles,
                    self.plane_sides,
                    self.plane_normal,
                    self.plane_point,
                    self.plane_epsilon,
                ],
                device=self.device,
            )
        plane_slots = self_capacity + (self.rigid_capacity if contacts is not None else 0)
        wp.launch(
            _accumulate_plane_projections,
            dim=plane_slots,
            inputs=[
                self_capacity,
                self.self_plane_count,
                rigid_count,
                self.plane_particles,
                self.plane_sides,
                self.plane_normal,
                self.plane_support,
                self.plane_gap,
                self.plane_epsilon,
                self.reference_position,
                self.displacement,
                self.relaxation,
            ],
            outputs=[self.plane_point, self.correction, self.correction_count],
            device=self.device,
        )
        plane_inputs = [
            self_capacity,
            self.self_plane_count,
            rigid_count,
            self.plane_particles,
            self.plane_sides,
            self.plane_normal,
            self.plane_point,
            self.plane_epsilon,
            self.reference_position,
            self.displacement,
            self.relaxation,
        ]
        wp.launch(
            _apply_plane_projections,
            dim=particle_count,
            outputs=[self.correction, self.correction_count, self.displacement, self.truncation],
            device=self.device,
        )
        wp.launch(
            _accumulate_directional_bounds,
            dim=plane_slots,
            inputs=plane_inputs,
            outputs=[self.truncation],
            device=self.device,
        )
        wp.launch(
            _apply_directional_bounds,
            dim=particle_count,
            inputs=[
                time_step,
                0.5 * (self_contacts.margin + self_contacts.gap) * self.relaxation,
                self.relaxation,
                detector.resize_flags,
                detector.vertex_colliding_triangles_min_dist,
                detector.triangle_colliding_vertices_min_dist,
                detector.edge_colliding_edges_min_dist,
                self_contacts.adjacency,
                self.topology.newton_to_packed,
                self.topology.packed_solve_world,
                world_active,
                self.inverse_weight,
                self.displacement,
                self.truncation,
            ],
            outputs=[projected_velocity],
            device=self.device,
        )
