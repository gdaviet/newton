# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernels adapting LOX results to Kamino integrators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ...core.types import vec6f

if TYPE_CHECKING:
    pass

__all__ = []

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _reset_newton_body_dual_impulse(
    body_world: wp.array[wp.int32],
    source_world_mask: wp.array[wp.bool],
    body_dual_impulse: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    world = wp.max(body_world[body], 0)
    selected = source_world_mask[world]
    if world == 0:
        selected = selected or source_world_mask[source_world_mask.shape[0] - 1]
    if selected:
        body_dual_impulse[body] = wp.spatial_vectorf(0.0)


@wp.kernel
def _write_integrator_body_inputs(
    body_vector_index: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    world_accepted: wp.array[wp.bool],
    world_time_step: wp.array[wp.float32],
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33f],
    body_inverse_mass: wp.array[wp.float32],
    body_inverse_inertia: wp.array[wp.mat33f],
    world_gravity: wp.array[wp.vec3f],
    velocity_begin: wp.array[vec6f],
    velocity_projected: wp.array[vec6f],
    body_wrench: wp.array[wp.spatial_vectorf],
    body_velocity: wp.array[wp.spatial_vectorf],
):
    """Encode the accepted LOX velocity as inputs to Kamino integration."""
    body = wp.tid()
    world = body_world[body]
    velocity_end = velocity_begin[body]
    if body_vector_index[body] >= 0 and world_accepted[world]:
        velocity_end = velocity_projected[body]
    linear_velocity_begin = wp.vec3f(
        velocity_begin[body][0],
        velocity_begin[body][1],
        velocity_begin[body][2],
    )
    angular_velocity_begin = wp.vec3f(velocity_begin[body][3], velocity_begin[body][4], velocity_begin[body][5])
    linear_velocity_end = wp.vec3f(velocity_end[0], velocity_end[1], velocity_end[2])
    angular_velocity_end = wp.vec3f(velocity_end[3], velocity_end[4], velocity_end[5])
    inverse_time_step = 1.0 / world_time_step[world]
    force = body_mass[body] * (inverse_time_step * (linear_velocity_end - linear_velocity_begin) - world_gravity[world])
    inertia = body_inertia[body]
    torque = inertia @ (inverse_time_step * (angular_velocity_end - angular_velocity_begin)) + wp.skew(
        angular_velocity_begin
    ) @ (inertia @ angular_velocity_begin)
    body_wrench[body] = wp.spatial_vectorf(
        force[0],
        force[1],
        force[2],
        torque[0],
        torque[1],
        torque[2],
    )
    if body_vector_index[body] >= 0 and (
        body_inverse_mass[body] == 0.0 or wp.determinant(body_inverse_inertia[body]) == 0.0
    ):
        body_velocity[body] = wp.spatial_vectorf(
            velocity_end[0],
            velocity_end[1],
            velocity_end[2],
            velocity_end[3],
            velocity_end[4],
            velocity_end[5],
        )
