# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Penetration-free step limiting of the self-contacts of the LOX deformable bodies.

:class:`DeformablePenetrationLimiter` truncates the projected particle velocities so that the
straight displacement of each self-contact candidate from the begin-step surface stays on its
side of a certified division plane, with the divide-and-truncate planes of VBD, and caps the
displacement of each particle at half the self-contact detection distance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .......utils.mesh import MeshAdjacencyData, get_vertex_num_adjacent_edges
from ......vbd.particle_vbd_kernels import (
    create_edge_edge_division_plane_closest_pt,
    create_vertex_triangle_division_plane_closest_pt,
    planar_truncation_t,
)

if TYPE_CHECKING:
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
def _truncate_vertex(
    particle: int,
    position: wp.vec3,
    displacement: wp.vec3,
    normal: wp.vec3,
    plane_point: wp.vec3,
    relaxation: float,
    separation_epsilon: float,
    truncation: wp.array[wp.float32],
):
    bound = planar_truncation_t(position, displacement, normal, plane_point, relaxation, separation_epsilon)
    if bound < 1.0:
        wp.atomic_min(truncation, particle, bound)


@wp.func
def _accumulate_vertex_triangle_bound(
    vertex: int,
    triangle: int,
    reference_position: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    relaxation: float,
    truncation: wp.array[wp.float32],
):
    """Bound the steps of a vertex and a triangle on the two sides of their division plane.

    A pair without a strict separator at the begin-step surface stays unconstrained; its
    contact row separates it, and a later detection certifies a separator.
    """
    if vertex < 0 or triangle < 0 or get_vertex_num_adjacent_edges(adjacency, vertex) == 0:
        return
    triangle_0 = tri_indices[triangle, 0]
    triangle_1 = tri_indices[triangle, 1]
    triangle_2 = tri_indices[triangle, 2]
    valid, normal, plane_point, separation_epsilon = create_vertex_triangle_division_plane_closest_pt(
        reference_position[vertex],
        displacement[vertex],
        reference_position[triangle_0],
        displacement[triangle_0],
        reference_position[triangle_1],
        displacement[triangle_1],
        reference_position[triangle_2],
        displacement[triangle_2],
    )
    if not valid:
        return
    # The plane normal points towards the vertex
    _truncate_vertex(
        vertex,
        reference_position[vertex],
        displacement[vertex],
        normal,
        plane_point,
        relaxation,
        separation_epsilon,
        truncation,
    )
    for corner in range(3):
        particle = tri_indices[triangle, corner]
        _truncate_vertex(
            particle,
            reference_position[particle],
            displacement[particle],
            -normal,
            plane_point,
            relaxation,
            separation_epsilon,
            truncation,
        )


@wp.func
def _accumulate_edge_edge_bound(
    edge_0: int,
    edge_1: int,
    reference_position: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    relaxation: float,
    truncation: wp.array[wp.float32],
):
    """Bound the steps of two edges on the two sides of their division plane."""
    if edge_0 < 0 or edge_1 < 0 or edge_0 == edge_1:
        return
    edge_0_vertex_0 = edge_indices[edge_0, 2]
    edge_0_vertex_1 = edge_indices[edge_0, 3]
    edge_1_vertex_0 = edge_indices[edge_1, 2]
    edge_1_vertex_1 = edge_indices[edge_1, 3]
    valid, normal, plane_point, separation_epsilon = create_edge_edge_division_plane_closest_pt(
        reference_position[edge_0_vertex_0],
        displacement[edge_0_vertex_0],
        reference_position[edge_0_vertex_1],
        displacement[edge_0_vertex_1],
        reference_position[edge_1_vertex_0],
        displacement[edge_1_vertex_0],
        reference_position[edge_1_vertex_1],
        displacement[edge_1_vertex_1],
    )
    if not valid:
        return
    # The plane normal points towards edge 0
    for side in range(2):
        sign = 1.0
        vertex_0 = edge_0_vertex_0
        vertex_1 = edge_0_vertex_1
        if side == 1:
            sign = -1.0
            vertex_0 = edge_1_vertex_0
            vertex_1 = edge_1_vertex_1
        _truncate_vertex(
            vertex_0,
            reference_position[vertex_0],
            displacement[vertex_0],
            sign * normal,
            plane_point,
            relaxation,
            separation_epsilon,
            truncation,
        )
        _truncate_vertex(
            vertex_1,
            reference_position[vertex_1],
            displacement[vertex_1],
            sign * normal,
            plane_point,
            relaxation,
            separation_epsilon,
            truncation,
        )


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
def _accumulate_directional_bounds(
    # Inputs:
    vertex_capacity: wp.int32,
    vertex_records: wp.array[wp.int32],
    edge_records: wp.array[wp.int32],
    reference_position: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    relaxation: wp.float32,
    # Outputs:
    truncation: wp.array[wp.float32],
):
    """Reduce the smallest step fraction of each particle over its self-contact candidates."""
    candidate = wp.tid()
    if candidate < vertex_capacity:
        _accumulate_vertex_triangle_bound(
            vertex_records[2 * candidate],
            vertex_records[2 * candidate + 1],
            reference_position,
            displacement,
            tri_indices,
            adjacency,
            relaxation,
            truncation,
        )
    else:
        edge_candidate = candidate - vertex_capacity
        _accumulate_edge_edge_bound(
            edge_records[2 * edge_candidate],
            edge_records[2 * edge_candidate + 1],
            reference_position,
            displacement,
            edge_indices,
            relaxation,
            truncation,
        )


@wp.kernel
def _apply_directional_bounds(
    # Inputs:
    model_time_dt: wp.array[wp.float32],
    max_displacement: wp.float32,
    newton_to_packed: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    inverse_weight: wp.array[wp.float32],
    truncation: wp.array[wp.float32],
    # Outputs:
    projected_velocity: wp.array[wp.vec3],
):
    """Scale the projected velocity of each particle by its step fraction, within the displacement cap."""
    particle = wp.tid()
    pid = newton_to_packed[particle]
    wid = particle_world[pid]
    if not world_active[wid]:
        return
    if inverse_weight[pid] <= 0.0:
        projected_velocity[pid] = wp.vec3(0.0)
        return
    velocity = truncation[particle] * projected_velocity[pid]
    displacement_length = model_time_dt[wid] * wp.length(velocity)
    if displacement_length > max_displacement:
        velocity *= max_displacement / displacement_length
    projected_velocity[pid] = velocity


###
# Interfaces
###


class DeformablePenetrationLimiter:
    """Limit the projected particle motion against the self-contact candidates of the time step.

    Args:
        self_contacts: Self-contact detector whose candidates bound the steps.
        topology: Packed particle layout of the deformable bodies.
        inverse_weight: Inverse consensus weight of each packed particle.
        relaxation: Safety relaxation of the directional and isotropic step bounds, in ``(0, 1]``.
    """

    def __init__(
        self,
        self_contacts: DeformableSelfContacts,
        topology: DeformableTopology,
        inverse_weight: wp.array[wp.float32],
        relaxation: float,
    ):
        self.self_contacts = self_contacts
        self.topology = topology
        self.inverse_weight = inverse_weight
        self.device = self_contacts.device
        self.relaxation = float(relaxation)
        particle_count = int(self_contacts.model.particle_count)
        self.reference_position = wp.zeros(particle_count, dtype=wp.vec3, device=self.device)
        """Begin-step particle positions, in Newton order."""
        self.displacement = wp.zeros(particle_count, dtype=wp.vec3, device=self.device)
        """Displacement of each particle over the time step, in Newton order."""
        self.truncation = wp.ones(particle_count, dtype=wp.float32, device=self.device)
        """Step fraction of each particle, in Newton order."""

    ###
    # Public API
    ###

    def begin_time_step(self, particle_q: wp.array[wp.vec3]) -> None:
        """Freeze the begin-step surface of the step bounds."""
        wp.copy(self.reference_position, particle_q)

    def truncate(
        self,
        projected_velocity: wp.array[wp.vec3],
        world_active: wp.array[wp.bool],
        time_step: wp.array[wp.float32],
    ) -> None:
        """Truncate the projected particle velocities of the active worlds from the begin-step surface."""
        self_contacts = self.self_contacts
        detector = self_contacts.detector
        model = self_contacts.model
        wp.launch(
            _compute_particle_displacements,
            dim=projected_velocity.shape[0],
            inputs=[time_step, self.topology.packed_to_newton, self.topology.packed_solve_world, projected_velocity],
            outputs=[self.displacement],
            device=self.device,
        )
        self.truncation.fill_(1.0)
        wp.launch(
            _accumulate_directional_bounds,
            dim=self_contacts.capacity,
            inputs=[
                self_contacts.vertex_capacity,
                detector.vertex_colliding_triangles,
                detector.edge_colliding_edges,
                self.reference_position,
                self.displacement,
                model.tri_indices,
                self_contacts.edge_indices,
                self_contacts.adjacency,
                self.relaxation,
            ],
            outputs=[self.truncation],
            device=self.device,
        )
        wp.launch(
            _apply_directional_bounds,
            dim=model.particle_count,
            inputs=[
                time_step,
                0.5 * (self_contacts.margin + self_contacts.gap) * self.relaxation,
                self.topology.newton_to_packed,
                self.topology.packed_solve_world,
                world_active,
                self.inverse_weight,
                self.truncation,
            ],
            outputs=[projected_velocity],
            device=self.device,
        )
