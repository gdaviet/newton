# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Directional step limiting for LOX deformable self-contact."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import warp as wp

from ......utils.mesh import MeshAdjacencyData, get_vertex_num_adjacent_edges
from .....vbd.particle_vbd_kernels import (
    create_edge_edge_division_plane_closest_pt,
    create_vertex_triangle_division_plane_closest_pt,
    planar_truncation_t,
)
from .time import validate_world_time_step

if TYPE_CHECKING:
    from .deformable_self_contact import DeformableSelfContactDetector
    from .deformable_system import DeformableTopology

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _form_deformable_displacement(
    time_step: wp.array[wp.float32],
    packed_to_newton: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    projected_velocity: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
):
    packed_particle = wp.tid()
    particle = packed_to_newton[packed_particle]
    displacement[particle] = time_step[packed_world[packed_particle]] * projected_velocity[packed_particle]


@wp.func
def _accumulate_vertex_triangle_bound(
    vertex: int,
    triangle: int,
    reference_position: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    parallel_epsilon: float,
    relaxation: float,
    truncation: wp.array[float],
):
    if vertex < 0 or triangle < 0 or get_vertex_num_adjacent_edges(adjacency, vertex) == 0:
        return

    triangle_0 = tri_indices[triangle, 0]
    triangle_1 = tri_indices[triangle, 1]
    triangle_2 = tri_indices[triangle, 2]

    vertex_position = reference_position[vertex]
    triangle_position_0 = reference_position[triangle_0]
    triangle_position_1 = reference_position[triangle_1]
    triangle_position_2 = reference_position[triangle_2]
    vertex_displacement = displacement[vertex]
    triangle_displacement_0 = displacement[triangle_0]
    triangle_displacement_1 = displacement[triangle_1]
    triangle_displacement_2 = displacement[triangle_2]
    dummy, normal, plane_point = create_vertex_triangle_division_plane_closest_pt(
        vertex_position,
        vertex_displacement,
        triangle_position_0,
        triangle_displacement_0,
        triangle_position_1,
        triangle_displacement_1,
        triangle_position_2,
        triangle_displacement_2,
    )
    if not dummy[0]:
        bound = planar_truncation_t(
            vertex_position,
            vertex_displacement,
            normal,
            plane_point,
            parallel_epsilon,
            relaxation,
        )
        wp.atomic_min(truncation, vertex, bound)
    if not dummy[1]:
        bound = planar_truncation_t(
            triangle_position_0,
            triangle_displacement_0,
            normal,
            plane_point,
            parallel_epsilon,
            relaxation,
        )
        wp.atomic_min(truncation, triangle_0, bound)
    if not dummy[2]:
        bound = planar_truncation_t(
            triangle_position_1,
            triangle_displacement_1,
            normal,
            plane_point,
            parallel_epsilon,
            relaxation,
        )
        wp.atomic_min(truncation, triangle_1, bound)
    if not dummy[3]:
        bound = planar_truncation_t(
            triangle_position_2,
            triangle_displacement_2,
            normal,
            plane_point,
            parallel_epsilon,
            relaxation,
        )
        wp.atomic_min(truncation, triangle_2, bound)


@wp.func
def _accumulate_edge_edge_bound(
    edge_0: int,
    edge_1: int,
    reference_position: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    parallel_epsilon: float,
    relaxation: float,
    truncation: wp.array[float],
):
    if edge_0 < 0 or edge_1 < 0 or edge_0 == edge_1:
        return

    edge_0_vertex_0 = edge_indices[edge_0, 2]
    edge_0_vertex_1 = edge_indices[edge_0, 3]
    edge_1_vertex_0 = edge_indices[edge_1, 2]
    edge_1_vertex_1 = edge_indices[edge_1, 3]

    edge_0_position_0 = reference_position[edge_0_vertex_0]
    edge_0_position_1 = reference_position[edge_0_vertex_1]
    edge_1_position_0 = reference_position[edge_1_vertex_0]
    edge_1_position_1 = reference_position[edge_1_vertex_1]
    edge_0_displacement_0 = displacement[edge_0_vertex_0]
    edge_0_displacement_1 = displacement[edge_0_vertex_1]
    edge_1_displacement_0 = displacement[edge_1_vertex_0]
    edge_1_displacement_1 = displacement[edge_1_vertex_1]
    dummy, normal, plane_point = create_edge_edge_division_plane_closest_pt(
        edge_0_position_0,
        edge_0_displacement_0,
        edge_0_position_1,
        edge_0_displacement_1,
        edge_1_position_0,
        edge_1_displacement_0,
        edge_1_position_1,
        edge_1_displacement_1,
    )
    if not dummy[0]:
        bound = planar_truncation_t(
            edge_0_position_0,
            edge_0_displacement_0,
            normal,
            plane_point,
            parallel_epsilon,
            relaxation,
        )
        wp.atomic_min(truncation, edge_0_vertex_0, bound)
    if not dummy[1]:
        bound = planar_truncation_t(
            edge_0_position_1,
            edge_0_displacement_1,
            normal,
            plane_point,
            parallel_epsilon,
            relaxation,
        )
        wp.atomic_min(truncation, edge_0_vertex_1, bound)
    if not dummy[2]:
        bound = planar_truncation_t(
            edge_1_position_0,
            edge_1_displacement_0,
            normal,
            plane_point,
            parallel_epsilon,
            relaxation,
        )
        wp.atomic_min(truncation, edge_1_vertex_0, bound)
    if not dummy[3]:
        bound = planar_truncation_t(
            edge_1_position_1,
            edge_1_displacement_1,
            normal,
            plane_point,
            parallel_epsilon,
            relaxation,
        )
        wp.atomic_min(truncation, edge_1_vertex_1, bound)


@wp.kernel
def _accumulate_directional_bounds(
    vertex_capacity: int,
    vertex_records: wp.array[wp.int32],
    edge_records: wp.array[wp.int32],
    reference_position: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    parallel_epsilon: float,
    relaxation: float,
    truncation: wp.array[float],
):
    candidate = wp.tid()
    if candidate < vertex_capacity:
        _accumulate_vertex_triangle_bound(
            vertex_records[2 * candidate],
            vertex_records[2 * candidate + 1],
            reference_position,
            displacement,
            tri_indices,
            adjacency,
            parallel_epsilon,
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
            parallel_epsilon,
            relaxation,
            truncation,
        )


@wp.kernel
def _apply_directional_bounds(
    time_step: wp.array[wp.float32],
    max_displacement: float,
    newton_to_packed: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    inverse_weight: wp.array[float],
    truncation: wp.array[float],
    projected_velocity: wp.array[wp.vec3],
):
    particle = wp.tid()
    packed_particle = newton_to_packed[particle]
    world = packed_world[packed_particle]
    if world_active[world] == 0:
        return
    if inverse_weight[packed_particle] <= 0.0:
        projected_velocity[packed_particle] = wp.vec3(0.0)
        return

    velocity = truncation[particle] * projected_velocity[packed_particle]
    displacement_length = time_step[world] * wp.length(velocity)
    if displacement_length > max_displacement:
        velocity *= max_displacement / displacement_length
    projected_velocity[packed_particle] = velocity


class DeformablePenetrationFreeLimiter:
    """Limit projected deformable motion against frozen speculative pairs."""

    def __init__(
        self,
        detector: DeformableSelfContactDetector,
        topology: DeformableTopology,
        inverse_weight: wp.array[float],
        relaxation: float,
    ):
        """Allocate fixed-capacity VBD-style DAT work buffers.

        Args:
            detector: Frozen speculative surface-pair detector.
            topology: LOX packed-to-Newton particle mappings.
            inverse_weight: Packed inverse consensus weights.
            relaxation: Directional and isotropic safety relaxation.
        """
        if not math.isfinite(relaxation) or not 0.0 < relaxation <= 1.0:
            raise ValueError("LOX deformable penetration-free relaxation must be finite and in (0, 1].")
        self.detector = detector
        self.topology = topology
        self.inverse_weight = inverse_weight
        self.device = detector.device
        self.relaxation = float(relaxation)
        particle_count = int(detector.model.particle_count)
        self.reference_position = wp.empty(particle_count, dtype=wp.vec3, device=self.device)
        self.displacement = wp.empty(particle_count, dtype=wp.vec3, device=self.device)
        self.truncation = wp.ones(particle_count, dtype=float, device=self.device)

    def begin_time_step(self, position: wp.array[wp.vec3]) -> None:
        """Freeze the collision-free beginning-of-step surface."""
        wp.copy(self.reference_position, position)

    def truncate(
        self,
        projected_velocity: wp.array[wp.vec3],
        world_active: wp.array[wp.int32],
        time_step: wp.array[wp.float32],
    ) -> None:
        """Truncate the current projected velocity from the frozen step origin."""
        validate_world_time_step(time_step, self.detector.model.world_count, self.device)
        detector = self.detector
        wp.launch(
            _form_deformable_displacement,
            dim=detector.model.particle_count,
            inputs=[
                time_step,
                self.topology.packed_to_newton,
                self.topology.packed_solve_world,
                projected_velocity,
            ],
            outputs=[self.displacement],
            device=self.device,
        )
        self.truncation.fill_(1.0)
        wp.launch(
            _accumulate_directional_bounds,
            dim=detector.capacity,
            inputs=[
                detector.vertex_capacity,
                detector.detector.vertex_colliding_triangles,
                detector.detector.edge_colliding_edges,
                self.reference_position,
                self.displacement,
                detector.model.tri_indices,
                detector.edge_indices,
                detector.adjacency,
                detector.detector.edge_edge_parallel_epsilon,
                self.relaxation,
            ],
            outputs=[self.truncation],
            device=self.device,
        )
        wp.launch(
            _apply_directional_bounds,
            dim=detector.model.particle_count,
            inputs=[
                time_step,
                0.5 * detector.query_radius * self.relaxation,
                self.topology.newton_to_packed,
                self.topology.packed_solve_world,
                world_active,
                self.inverse_weight,
                self.truncation,
            ],
            outputs=[projected_velocity],
            device=self.device,
        )
