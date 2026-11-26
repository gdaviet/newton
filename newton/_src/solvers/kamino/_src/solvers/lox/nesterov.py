# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Nesterov extrapolation of the LOX projection sweeps.

After every sweep but the last, the reactions are extrapolated along their
last change, ``lambda <- lambda + beta (lambda - lambda_previous)``, with the
usual ``theta`` recursion. The momentum restarts per world whenever the last
sweep moved away from the previous extrapolation. When recycled across
splitting iterations, the extrapolation of the last sweep is applied at the
start of the next projection, after its warm start.

The body twists are affine in the reactions, with weights fixed over a LOX
solve, so each body twist is extrapolated by the same momentum,
``v <- v + beta (v - v_previous)``, with no impulse scatter. Across splitting
iterations, the change ``v - v_previous`` of the last sweep carries over to
the warm start of the next projection, whose candidate twist differs.

The kernels run over the stacked reaction vector ``[box rows | contacts]``.
With deformable bodies, the particle contact rows and the projected particle
velocities follow the same extrapolation, in separate launches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ...core.types import mat36f, mat66f, vec6f
from ..padmm.math import project_to_coulomb_cone
from .contact import (
    angular_contact_reaction,
    clamp_contact_reaction,
    linear_contact_reaction,
    pack_contact_reaction,
    pack_spatial_friction,
    spatial_body_velocity,
    spatial_friction_scaling,
)
from .projection_kernels import is_active_row, particle_contact_velocity
from .system_kernels import apply_body_weight
from .types import ParticleContactStatus

if TYPE_CHECKING:
    from .types import LOXProblemData, NesterovData

###
# Module interface
###

__all__ = [
    "NesterovAcceleration",
    "NesterovRowState",
    "box_restart_dot",
    "contact_restart_dot",
    "extrapolate_box_nesterov_row",
    "extrapolate_contact_nesterov_row",
    "initialize_box_nesterov_row",
    "initialize_contact_nesterov_row",
    "nesterov_momentum",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@wp.struct
class NesterovRowState:
    """Trial and previous reactions of the Nesterov extrapolation, for the per-world projection."""

    box_trial: wp.array[wp.float32]
    box_previous: wp.array[wp.float32]
    contact_trial: wp.array[wp.vec3f]
    contact_previous: wp.array[wp.vec3f]
    angular_trial: wp.array[wp.vec3f]
    angular_previous: wp.array[wp.vec3f]
    theta: wp.array[wp.float32]
    history: wp.array[wp.int32]
    twist_previous: wp.array[vec6f]
    twist_trial: wp.array[vec6f]
    twist_final: wp.array[vec6f]
    drift_previous: wp.array[wp.float32]
    body_weight: wp.array[mat66f]


###
# Functions
###


@wp.func
def nesterov_momentum(theta: wp.float32, restart_dot: wp.float32):
    """Return the next ``theta`` and the extrapolation ``beta``, restarting when the last sweep moved away.

    ``theta`` starts at one and the recursion keeps it in ``(0, 1]``.
    """
    # The negated comparison also restarts on a NaN dot product
    if not (restart_dot > 0.0):
        return wp.float32(1.0), wp.float32(0.0)
    next_theta = 2.0 * theta / (wp.sqrt(theta * theta + 4.0) + theta)
    return next_theta, theta * (1.0 - theta) / (theta * theta + next_theta)


@wp.func
def de_saxce_shift(velocity: wp.vec3f, friction: wp.float32) -> wp.float32:
    """Return the de Saxce shift ``mu |u_T|`` of a normal-last contact velocity.

    The Coulomb law of a contact is the associated law of the velocity shifted by ``mu |u_T|`` along its normal,
    so the Coulomb solution minimizes the projection energy with that shift frozen.
    """
    return friction * wp.sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1])


@wp.func
def energy_momentum(
    theta: wp.float32, slope: wp.float32, slope_trial: wp.float32, curvature: wp.float32, first: wp.bool
):
    """Return the next ``theta`` and the extrapolation ``beta`` from the energy slopes and curvature of the last step.

    The projection minimizes the energy ``E = 1/2 |p|_W^2 + b . lambda + 1/2 lambda . C lambda`` of its current
    candidate. Along the last change ``d`` of the reactions, ``slope`` is ``grad E . d`` at the current reactions,
    ``slope_trial`` is ``grad E . d`` at the extrapolated reactions the last sweep started from, and ``curvature``
    is ``h = d . (D + C) d``. The momentum restarts unless the sweep descended from the extrapolated point,
    ``slope_trial < 0``, and ``d`` still descends, ``slope < 0``; the extrapolation stops at the minimum of ``E``
    along ``d``, ``beta <= -slope / h``. The ``first`` extrapolation of a projection tests ``slope`` alone: its
    theta, recycled across splitting iterations, is already gated on the drift of the candidates.
    """
    descent = -slope
    if not first and not (slope_trial < 0.0):
        descent = wp.float32(0.0)
    next_theta, beta = nesterov_momentum(theta, descent)
    if beta > 0.0:
        if curvature > 0.0:
            beta = wp.min(beta, -slope / curvature)
        else:
            beta = wp.float32(0.0)
    return next_theta, beta


@wp.func
def initialize_box_nesterov_row(
    row: wp.int32,
    box_lower: wp.array[wp.float32],
    box_upper: wp.array[wp.float32],
    box_reaction: wp.array[wp.float32],
    box_trial: wp.array[wp.float32],
    box_previous: wp.array[wp.float32],
):
    """Clamp the warm-start reaction of one box row and reset its momentum."""
    value = wp.clamp(box_reaction[row], box_lower[row], box_upper[row])
    box_reaction[row] = value
    box_trial[row] = value
    box_previous[row] = value


@wp.func
def initialize_contact_nesterov_row(
    cid: wp.int32,
    contact_body_a: wp.array[wp.int32],
    contact_body_b: wp.array[wp.int32],
    contact_friction: wp.array[wp.float32],
    angular_friction: wp.array[wp.vec2f],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    angular_trial: wp.array[wp.vec3f],
    angular_previous: wp.array[wp.vec3f],
):
    """Project the warm-start reaction of one contact onto its cone and reset its momentum."""
    value = wp.vec3f(0.0)
    if contact_body_a[cid] >= 0 or contact_body_b[cid] >= 0:
        if angular_reaction:
            spatial = clamp_contact_reaction(
                pack_contact_reaction(contact_reaction[cid], angular_reaction[cid]),
                pack_spatial_friction(cid, contact_friction, angular_friction),
            )
            value = linear_contact_reaction(spatial)
            angular = angular_contact_reaction(spatial)
            angular_reaction[cid] = angular
            angular_trial[cid] = angular
            angular_previous[cid] = angular
        else:
            value = project_to_coulomb_cone(contact_reaction[cid], contact_friction[cid])
    contact_reaction[cid] = value
    contact_trial[cid] = value
    contact_previous[cid] = value


@wp.func
def box_restart_dot(
    row: wp.int32,
    box_reaction: wp.array[wp.float32],
    box_trial: wp.array[wp.float32],
    box_previous: wp.array[wp.float32],
) -> wp.float32:
    """Return ``(lambda - trial) (lambda - previous)`` of one box row."""
    current = box_reaction[row]
    return (current - box_trial[row]) * (current - box_previous[row])


@wp.func
def contact_restart_dot(
    cid: wp.int32,
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    angular_trial: wp.array[wp.vec3f],
    angular_previous: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    angular_friction: wp.array[wp.vec2f],
) -> wp.float32:
    """Return ``(lambda - trial) . (lambda - previous)`` of one contact, in units of the normal impulse."""
    current = contact_reaction[cid]
    if not angular_reaction:
        return wp.dot(current - contact_trial[cid], current - contact_previous[cid])
    spatial = pack_contact_reaction(current, angular_reaction[cid])
    trial = pack_contact_reaction(contact_trial[cid], angular_trial[cid])
    previous = pack_contact_reaction(contact_previous[cid], angular_previous[cid])
    scale = spatial_friction_scaling(pack_spatial_friction(cid, contact_friction, angular_friction))
    dot = wp.float32(0.0)
    for axis in range(6):
        if scale[axis] > 0.0:
            dot += ((spatial[axis] - trial[axis]) / scale[axis]) * ((spatial[axis] - previous[axis]) / scale[axis])
    return dot


@wp.func
def extrapolate_box_nesterov_row(
    row: wp.int32,
    beta: wp.float32,
    box_reaction: wp.array[wp.float32],
    box_trial: wp.array[wp.float32],
    box_previous: wp.array[wp.float32],
):
    """Extrapolate the reaction of one box row."""
    current = box_reaction[row]
    extrapolated = current + beta * (current - box_previous[row])
    box_previous[row] = current
    box_trial[row] = extrapolated
    box_reaction[row] = extrapolated


@wp.func
def extrapolate_contact_nesterov_row(
    cid: wp.int32,
    beta: wp.float32,
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    angular_trial: wp.array[wp.vec3f],
    angular_previous: wp.array[wp.vec3f],
):
    """Extrapolate the linear and, with spatial friction, angular reaction of one contact."""
    current = contact_reaction[cid]
    extrapolated = current + beta * (current - contact_previous[cid])
    contact_previous[cid] = current
    contact_trial[cid] = extrapolated
    contact_reaction[cid] = extrapolated
    if angular_reaction:
        angular = angular_reaction[cid]
        angular_extrapolated = angular + beta * (angular - angular_previous[cid])
        angular_previous[cid] = angular
        angular_trial[cid] = angular_extrapolated
        angular_reaction[cid] = angular_extrapolated


@wp.func
def extrapolate_body_twist(
    bid: wp.int32,
    beta: wp.float32,
    twist: wp.array[vec6f],
    twist_previous: wp.array[vec6f],
):
    """Extrapolate the twist of one body, which follows the extrapolation of the reactions of its world."""
    current = twist[bid]
    twist[bid] = current + beta * (current - twist_previous[bid])
    twist_previous[bid] = current


@wp.func
def world_restart_dot(
    restart_dot: wp.array2d[wp.float32], read: wp.int32, wid: wp.int32, speed: wp.bool, first: wp.bool
):
    """Return the restart criterion of a world; the first speed-restart extrapolation of a projection keeps its
    momentum, since it has no previous step to compare with."""
    if speed and first:
        return wp.float32(1.0)
    return restart_dot[read, wid]


@wp.func
def world_momentum(
    theta: wp.array2d[wp.float32],
    restart_dot: wp.array2d[wp.float32],
    curvature: wp.array2d[wp.float32],
    slope_trial: wp.array2d[wp.float32],
    read: wp.int32,
    wid: wp.int32,
    speed: wp.bool,
    first: wp.bool,
    energy: wp.bool,
):
    """Return the next ``theta`` and the extrapolation ``beta`` of a world.

    With ``energy``, ``restart_dot`` holds the slope ``g`` of the projection energy along the last step and
    ``curvature`` its curvature ``h``; see :func:`energy_momentum`.
    """
    if energy:
        return energy_momentum(
            theta[read, wid], restart_dot[read, wid], slope_trial[read, wid], curvature[read, wid], first
        )
    return nesterov_momentum(theta[read, wid], world_restart_dot(restart_dot, read, wid, speed, first))


###
# Kernels
###


@wp.kernel
def _initialize_worlds(
    # Inputs:
    world_active: wp.array[wp.bool],
    recycle: wp.bool,
    first_buffer: wp.int32,
    # Outputs:
    theta: wp.array2d[wp.float32],
    restart_dot: wp.array2d[wp.float32],
    history: wp.array[wp.int32],
):
    """Load the momentum of each world into the buffer of the first extrapolation and clear the restart dots.

    The first buffer holds the momentum between projections, where the last extrapolation writes it.
    """
    wid = wp.tid()
    if world_active[wid]:
        value = wp.float32(1.0)
        if recycle:
            value = theta[0, wid]
        theta[first_buffer, wid] = value
        restart_dot[0, wid] = 0.0
        restart_dot[1, wid] = 0.0
        history[wid] = wp.int32(recycle)


@wp.kernel
def _initialize_reactions(
    # Inputs:
    box_capacity: wp.int32,
    box_world: wp.array[wp.int32],
    box_local: wp.array[wp.int32],
    world_box_count: wp.array[wp.int32],
    box_lower: wp.array[wp.float32],
    box_upper: wp.array[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_a: wp.array[wp.int32],
    contact_body_b: wp.array[wp.int32],
    contact_friction: wp.array[wp.float32],
    angular_friction: wp.array[wp.vec2f],
    world_active: wp.array[wp.bool],
    recycle: wp.bool,
    history: wp.array[wp.int32],
    # Outputs:
    box_reaction: wp.array[wp.float32],
    box_trial: wp.array[wp.float32],
    box_previous: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    angular_trial: wp.array[wp.vec3f],
    angular_previous: wp.array[wp.vec3f],
):
    """Clamp the warm-start reactions into their feasible sets and reset the momentum.

    With ``recycle``, the worlds that hold the row history of the previous splitting iteration keep it.
    """
    tid = wp.tid()
    if tid < box_capacity:
        row = tid
        wid = box_world[row]
        if box_local[row] < world_box_count[wid] and world_active[wid] and not (recycle and history[wid] == 1):
            initialize_box_nesterov_row(row, box_lower, box_upper, box_reaction, box_trial, box_previous)
        return
    cid = tid - box_capacity
    wid = contact_world[cid]
    if contact_local[cid] < world_contact_count[wid] and world_active[wid] and not (recycle and history[wid] == 1):
        initialize_contact_nesterov_row(
            cid,
            contact_body_a,
            contact_body_b,
            contact_friction,
            angular_friction,
            contact_reaction,
            contact_trial,
            contact_previous,
            angular_reaction,
            angular_trial,
            angular_previous,
        )


@wp.kernel
def _accumulate_restart_dot(
    # Inputs:
    box_capacity: wp.int32,
    box_world: wp.array[wp.int32],
    box_local: wp.array[wp.int32],
    world_box_count: wp.array[wp.int32],
    box_reaction: wp.array[wp.float32],
    box_trial: wp.array[wp.float32],
    box_previous: wp.array[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    angular_trial: wp.array[wp.vec3f],
    angular_previous: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    angular_friction: wp.array[wp.vec2f],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    # Outputs:
    read: wp.int32,
    restart_dot: wp.array2d[wp.float32],
):
    """Accumulate ``(lambda - trial) . (lambda - previous)`` per world, into the ``read`` buffer."""
    tid = wp.tid()
    if tid < box_capacity:
        row = tid
        wid = box_world[row]
        if is_active_row(box_local[row], wid, world_box_count, world_active, world_failed):
            wp.atomic_add(restart_dot, read, wid, box_restart_dot(row, box_reaction, box_trial, box_previous))
        return
    cid = tid - box_capacity
    wid = contact_world[cid]
    if is_active_row(contact_local[cid], wid, world_contact_count, world_active, world_failed):
        wp.atomic_add(
            restart_dot,
            read,
            wid,
            contact_restart_dot(
                cid,
                contact_reaction,
                contact_trial,
                contact_previous,
                angular_reaction,
                angular_trial,
                angular_previous,
                contact_friction,
                angular_friction,
            ),
        )


@wp.kernel
def _extrapolate_reactions_and_twists(
    # Inputs:
    box_capacity: wp.int32,
    box_world: wp.array[wp.int32],
    box_local: wp.array[wp.int32],
    world_box_count: wp.array[wp.int32],
    contact_capacity: wp.int32,
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    body_count: wp.int32,
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    read: wp.int32,
    speed: wp.bool,
    first: wp.bool,
    speed_step: wp.bool,
    energy: wp.bool,
    # Outputs:
    theta: wp.array2d[wp.float32],
    restart_dot: wp.array2d[wp.float32],
    curvature: wp.array2d[wp.float32],
    slope_trial: wp.array2d[wp.float32],
    step_previous: wp.array[wp.float32],
    twist_trial: wp.array[vec6f],
    box_reaction: wp.array[wp.float32],
    box_trial: wp.array[wp.float32],
    box_previous: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    angular_trial: wp.array[wp.vec3f],
    angular_previous: wp.array[wp.vec3f],
    projected_twist: wp.array[vec6f],
    twist_previous: wp.array[vec6f],
):
    """Extrapolate each reaction and each body twist by the momentum of its world.

    Every thread derives the momentum of its world from the ``read`` buffers of ``theta`` and ``restart_dot``;
    one trailing thread per world writes the next ``theta`` and resets the restart dot of the next sweep, in the
    other buffers.

    With ``speed``, the ``read`` restart dot holds the previous weighted step minus the current one, and the
    trailing thread loads the current step into the next restart dot, from which the next sweep subtracts its own.
    """
    tid = wp.tid()
    if tid < box_capacity:
        row = tid
        wid = box_world[row]
        if is_active_row(box_local[row], wid, world_box_count, world_active, world_failed):
            _next_theta, beta = world_momentum(
                theta, restart_dot, curvature, slope_trial, read, wid, speed, first, energy
            )
            extrapolate_box_nesterov_row(row, beta, box_reaction, box_trial, box_previous)
        return
    cid = tid - box_capacity
    if cid < contact_capacity:
        wid = contact_world[cid]
        if is_active_row(contact_local[cid], wid, world_contact_count, world_active, world_failed):
            _next_theta, beta = world_momentum(
                theta, restart_dot, curvature, slope_trial, read, wid, speed, first, energy
            )
            extrapolate_contact_nesterov_row(
                cid,
                beta,
                contact_reaction,
                contact_trial,
                contact_previous,
                angular_reaction,
                angular_trial,
                angular_previous,
            )
        return
    bid = cid - contact_capacity
    if bid < body_count:
        wid = body_world[bid]
        if world_active[wid] and not world_failed[wid]:
            _next_theta, beta = world_momentum(
                theta, restart_dot, curvature, slope_trial, read, wid, speed, first, energy
            )
            extrapolate_body_twist(bid, beta, projected_twist, twist_previous)
            if speed or energy:
                twist_trial[bid] = projected_twist[bid]
        return
    wid = bid - body_count
    if world_active[wid] and not world_failed[wid]:
        next_theta, _beta = world_momentum(theta, restart_dot, curvature, slope_trial, read, wid, speed, first, energy)
        theta[1 - read, wid] = next_theta
        next_dot = wp.float32(0.0)
        if speed_step:
            previous = wp.float32(0.0)
            if not first:
                previous = step_previous[wid]
            next_dot = previous - restart_dot[read, wid]
            step_previous[wid] = next_dot
        restart_dot[1 - read, wid] = next_dot
        curvature[1 - read, wid] = 0.0
        slope_trial[1 - read, wid] = 0.0


@wp.func
def _add_energy(
    wid: wp.int32,
    value: wp.float32,
    trial_value: wp.float32,
    curve: wp.float32,
    read: wp.int32,
    slope: wp.array2d[wp.float32],
    slope_trial: wp.array2d[wp.float32],
    curvature: wp.array2d[wp.float32],
):
    """Add the nonzero slope and curvature terms of an item to its world."""
    if wid >= 0:
        if value != 0.0:
            wp.atomic_add(slope, read, wid, value)
        if curve != 0.0:
            wp.atomic_add(curvature, read, wid, curve)
        if trial_value != 0.0:
            wp.atomic_add(slope_trial, read, wid, trial_value)


@wp.func
def _gate_recycled_world(
    wid: wp.int32,
    world_active: wp.array[wp.bool],
    first_buffer: wp.int32,
    drift_current: wp.array[wp.float32],
    drift_previous: wp.array[wp.float32],
    step: wp.array[wp.float32],
    theta: wp.array2d[wp.float32],
):
    """Keep the recycled momentum of one world while the splitting iterations converge.

    The drift is the squared weighted change between the warm start of this projection and the result of the
    previous one, the change of the candidate. A world keeps its momentum while the drift shrinks and stays
    below the squared weighted last step that the momentum carries; otherwise it restarts.
    """
    if not world_active[wid]:
        return
    current = drift_current[wid]
    previous = drift_previous[wid]
    if not (previous >= 0.0 and current <= previous and current <= step[wid]):
        theta[first_buffer, wid] = 1.0
    drift_previous[wid] = current
    drift_current[wid] = 0.0
    step[wid] = 0.0


@wp.func
def box_row_energy(
    row: wp.int32,
    change: wp.float32,
    bias: wp.float32,
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    twist: wp.array[vec6f],
    twist_previous: wp.array[vec6f],
    twist_trial: wp.array[vec6f],
) -> wp.vec3f:
    """Return the energy slope at the current and at the extrapolated twists, and the curvature, of one box row
    along its reaction change ``d``: ``(b + J v) d``, ``(b + J v_trial) d``, and ``(J dv) d``."""
    velocity = bias
    trial = bias
    step = wp.float32(0.0)
    bid_a = body_a[row]
    bid_b = body_b[row]
    if bid_a >= 0:
        jacobian = jacobian_a[row]
        current = twist[bid_a]
        velocity += wp.dot(jacobian, current)
        trial += wp.dot(jacobian, twist_trial[bid_a])
        step += wp.dot(jacobian, current - twist_previous[bid_a])
    if bid_b >= 0:
        jacobian = jacobian_b[row]
        current = twist[bid_b]
        velocity += wp.dot(jacobian, current)
        trial += wp.dot(jacobian, twist_trial[bid_b])
        step += wp.dot(jacobian, current - twist_previous[bid_b])
    return wp.vec3f(velocity * change, trial * change, step * change)


@wp.func
def contact_energy(
    cid: wp.int32,
    spatial: wp.bool,
    current: wp.vec3f,
    change: wp.vec3f,
    angular_change: wp.vec3f,
    trial_normal: wp.float32,
    bias: wp.vec3f,
    compliance: wp.float32,
    de_saxce: wp.bool,
    friction: wp.float32,
    body_a: wp.array[wp.int32],
    body_b: wp.array[wp.int32],
    jacobian_a: wp.array[mat36f],
    jacobian_b: wp.array[mat36f],
    frame: wp.array[wp.mat33f],
    twist: wp.array[vec6f],
    twist_previous: wp.array[vec6f],
    twist_trial: wp.array[vec6f],
) -> wp.vec3f:
    """Return the energy slope at the current and at the extrapolated twists, and the curvature, of one contact
    along its reaction change ``d``: ``(b + J v + C lambda) . d``, the same at the extrapolated point, and
    ``(J dv) . d + C d_n^2``, each slope with the de Saxce shift of its velocity.

    A spatial contact pairs its normal-first velocity with its linear and angular reaction change, and keeps
    the normal bias only.
    """
    bid_a = body_a[cid]
    bid_b = body_b[cid]
    linear = bias
    linear_trial = bias
    value = wp.float32(0.0)
    trial_value = wp.float32(0.0)
    curve = wp.float32(0.0)
    if spatial:
        contact_frame = frame[cid]
        velocity = vec6f(0.0)
        velocity_trial = vec6f(0.0)
        spatial_step = vec6f(0.0)
        if bid_a >= 0:
            jacobian = jacobian_a[cid]
            twist_a = twist[bid_a]
            velocity += spatial_body_velocity(jacobian, contact_frame, -1.0, twist_a)
            velocity_trial += spatial_body_velocity(jacobian, contact_frame, -1.0, twist_trial[bid_a])
            spatial_step += spatial_body_velocity(jacobian, contact_frame, -1.0, twist_a - twist_previous[bid_a])
        if bid_b >= 0:
            jacobian = jacobian_b[cid]
            twist_b = twist[bid_b]
            velocity += spatial_body_velocity(jacobian, contact_frame, 1.0, twist_b)
            velocity_trial += spatial_body_velocity(jacobian, contact_frame, 1.0, twist_trial[bid_b])
            spatial_step += spatial_body_velocity(jacobian, contact_frame, 1.0, twist_b - twist_previous[bid_b])
        packed = pack_contact_reaction(change, angular_change)
        value = wp.dot(velocity, packed) + bias[2] * change[2]
        trial_value = wp.dot(velocity_trial, packed) + bias[2] * change[2]
        curve = wp.dot(spatial_step, packed)
        linear += linear_contact_reaction(velocity)
        linear_trial += linear_contact_reaction(velocity_trial)
    else:
        linear_step = wp.vec3f(0.0)
        if bid_a >= 0:
            jacobian = jacobian_a[cid]
            twist_a = twist[bid_a]
            linear += jacobian @ twist_a
            linear_trial += jacobian @ twist_trial[bid_a]
            linear_step += jacobian @ (twist_a - twist_previous[bid_a])
        if bid_b >= 0:
            jacobian = jacobian_b[cid]
            twist_b = twist[bid_b]
            linear += jacobian @ twist_b
            linear_trial += jacobian @ twist_trial[bid_b]
            linear_step += jacobian @ (twist_b - twist_previous[bid_b])
        value = wp.dot(linear, change)
        trial_value = wp.dot(linear_trial, change)
        curve = wp.dot(linear_step, change)
    value += compliance * current[2] * change[2]
    trial_value += compliance * trial_normal * change[2]
    curve += compliance * change[2] * change[2]
    if de_saxce and change[2] != 0.0:
        value += de_saxce_shift(linear, friction) * change[2]
        trial_value += de_saxce_shift(linear_trial, friction) * change[2]
    return wp.vec3f(value, trial_value, curve)


@wp.func
def particle_row_energy(
    row: wp.int32,
    change: wp.vec3f,
    bias: wp.vec3f,
    de_saxce: wp.bool,
    friction: wp.float32,
    particle: wp.array2d[wp.int32],
    coefficient: wp.array2d[wp.float32],
    body: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    velocity: wp.array[wp.vec3f],
    velocity_previous: wp.array[wp.vec3f],
    velocity_trial: wp.array[wp.vec3f],
    twist: wp.array[vec6f],
    twist_previous: wp.array[vec6f],
    twist_trial: wp.array[vec6f],
) -> wp.vec3f:
    """Return the energy slope at the current and at the extrapolated velocities, and the curvature, of one
    particle contact row along its reaction change ``d``, as :func:`contact_energy`."""
    current = bias + particle_contact_velocity(row, particle, coefficient, body, frame, body_jacobian, velocity, twist)
    trial = bias + particle_contact_velocity(
        row, particle, coefficient, body, frame, body_jacobian, velocity_trial, twist_trial
    )
    previous = bias + particle_contact_velocity(
        row, particle, coefficient, body, frame, body_jacobian, velocity_previous, twist_previous
    )
    value = wp.dot(current, change)
    trial_value = wp.dot(trial, change)
    if de_saxce and change[2] != 0.0:
        value += de_saxce_shift(current, friction) * change[2]
        trial_value += de_saxce_shift(trial, friction) * change[2]
    return wp.vec3f(value, trial_value, wp.dot(current - previous, change))


@wp.kernel
def _accumulate_energy_slope(
    # Inputs:
    box_capacity: wp.int32,
    box_world: wp.array[wp.int32],
    box_local: wp.array[wp.int32],
    world_box_count: wp.array[wp.int32],
    box_bias: wp.array[wp.float32],
    box_reaction: wp.array[wp.float32],
    box_previous: wp.array[wp.float32],
    box_body_a: wp.array[wp.int32],
    box_body_b: wp.array[wp.int32],
    box_jacobian_a: wp.array[vec6f],
    box_jacobian_b: wp.array[vec6f],
    contact_capacity: wp.int32,
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_bias: wp.array[wp.vec3f],
    normal_compliance: wp.array[wp.float32],
    spatial: wp.bool,
    contact_reaction: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    angular_previous: wp.array[wp.vec3f],
    de_saxce: wp.bool,
    contact_body_a: wp.array[wp.int32],
    contact_body_b: wp.array[wp.int32],
    contact_jacobian_a: wp.array[mat36f],
    contact_jacobian_b: wp.array[mat36f],
    contact_frame: wp.array[wp.mat33f],
    contact_friction: wp.array[wp.float32],
    projected_twist: wp.array[vec6f],
    twist_previous: wp.array[vec6f],
    twist_trial: wp.array[vec6f],
    row_capacity: wp.int32,
    row_world: wp.array[wp.int32],
    row_status: wp.array[wp.int32],
    row_bias: wp.array[wp.vec3f],
    row_reaction: wp.array[wp.vec3f],
    row_previous: wp.array[wp.vec3f],
    row_de_saxce: wp.bool,
    row_particle: wp.array2d[wp.int32],
    row_coefficient: wp.array2d[wp.float32],
    row_body: wp.array[wp.int32],
    row_frame: wp.array[wp.mat33f],
    row_body_jacobian: wp.array[mat36f],
    row_friction: wp.array[wp.float32],
    projected_velocity: wp.array[wp.vec3f],
    velocity_previous: wp.array[wp.vec3f],
    velocity_trial: wp.array[wp.vec3f],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    read: wp.int32,
    gate_offset: wp.int32,
    # Outputs:
    slope: wp.array2d[wp.float32],
    slope_trial: wp.array2d[wp.float32],
    curvature: wp.array2d[wp.float32],
    drift_current: wp.array[wp.float32],
    drift_previous: wp.array[wp.float32],
    step: wp.array[wp.float32],
    theta: wp.array2d[wp.float32],
):
    """Accumulate per world the slope ``g = grad E . d`` of the projection energy at the current and at the
    extrapolated point, and its curvature ``h = d . (D + C) d``, along the last reaction change ``d``.

    The twists are affine in the reactions, ``W dv = J^T d``, so every term is a row term over the stacked
    ``[box rows | contacts | particle contact rows]``, and the rows that the last sweep left unchanged add
    nothing. With ``gate_offset >= 0``, at the deferred extrapolation of a recycled projection, one trailing
    thread per world gates its recycled ``theta`` (:func:`_gate_recycled_world`), which this kernel never reads.
    """
    tid = wp.tid()
    if tid < box_capacity:
        row = tid
        wid = box_world[row]
        if is_active_row(box_local[row], wid, world_box_count, world_active, world_failed):
            box_change = box_reaction[row] - box_previous[row]
            if box_change != 0.0:
                terms = box_row_energy(
                    row,
                    box_change,
                    box_bias[row],
                    box_body_a,
                    box_body_b,
                    box_jacobian_a,
                    box_jacobian_b,
                    projected_twist,
                    twist_previous,
                    twist_trial,
                )
                _add_energy(wid, terms[0], terms[1], terms[2], read, slope, slope_trial, curvature)
        return
    cid = tid - box_capacity
    if cid < contact_capacity:
        wid = contact_world[cid]
        if not is_active_row(contact_local[cid], wid, world_contact_count, world_active, world_failed):
            return
        current = contact_reaction[cid]
        change = current - contact_previous[cid]
        angular_change = wp.vec3f(0.0)
        if spatial:
            angular_change = angular_reaction[cid] - angular_previous[cid]
        if change == wp.vec3f(0.0) and angular_change == wp.vec3f(0.0):
            return
        compliance = wp.float32(0.0)
        if normal_compliance:
            compliance = normal_compliance[cid]
        contact_terms = contact_energy(
            cid,
            spatial,
            current,
            change,
            angular_change,
            contact_trial[cid][2],
            contact_bias[cid],
            compliance,
            de_saxce,
            contact_friction[cid],
            contact_body_a,
            contact_body_b,
            contact_jacobian_a,
            contact_jacobian_b,
            contact_frame,
            projected_twist,
            twist_previous,
            twist_trial,
        )
        _add_energy(wid, contact_terms[0], contact_terms[1], contact_terms[2], read, slope, slope_trial, curvature)
        return
    prow = cid - contact_capacity
    if gate_offset >= 0 and prow >= gate_offset:
        _gate_recycled_world(prow - gate_offset, world_active, read, drift_current, drift_previous, step, theta)
        return
    if row_status[prow] != ParticleContactStatus.ACTIVE:
        return
    wid = row_world[prow]
    if not world_active[wid] or world_failed[wid]:
        return
    row_change = row_reaction[prow] - row_previous[prow]
    if row_change == wp.vec3f(0.0):
        return
    row_terms = particle_row_energy(
        prow,
        row_change,
        row_bias[prow],
        row_de_saxce,
        row_friction[prow],
        row_particle,
        row_coefficient,
        row_body,
        row_frame,
        row_body_jacobian,
        projected_velocity,
        velocity_previous,
        velocity_trial,
        projected_twist,
        twist_previous,
        twist_trial,
    )
    _add_energy(wid, row_terms[0], row_terms[1], row_terms[2], read, slope, slope_trial, curvature)


@wp.kernel
def _initialize_particle_reactions(
    # Inputs:
    row_world: wp.array[wp.int32],
    row_status: wp.array[wp.int32],
    friction: wp.array[wp.float32],
    world_active: wp.array[wp.bool],
    recycle: wp.bool,
    history: wp.array[wp.int32],
    # Outputs:
    reaction: wp.array[wp.vec3f],
    trial: wp.array[wp.vec3f],
    previous: wp.array[wp.vec3f],
):
    """Project the warm-start reaction of each particle contact row onto its cone and reset its momentum.

    With ``recycle``, the worlds that hold the row history of the previous splitting iteration keep it.
    """
    row = wp.tid()
    if row_status[row] != ParticleContactStatus.ACTIVE:
        return
    wid = row_world[row]
    if not world_active[wid] or (recycle and history[wid] == 1):
        return
    value = project_to_coulomb_cone(reaction[row], friction[row])
    reaction[row] = value
    trial[row] = value
    previous[row] = value


@wp.kernel
def _accumulate_particle_restart_dot(
    # Inputs:
    row_world: wp.array[wp.int32],
    row_status: wp.array[wp.int32],
    reaction: wp.array[wp.vec3f],
    trial: wp.array[wp.vec3f],
    previous: wp.array[wp.vec3f],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    read: wp.int32,
    # Outputs:
    restart_dot: wp.array2d[wp.float32],
):
    """Accumulate ``(lambda - trial) . (lambda - previous)`` of the particle contact rows per world."""
    row = wp.tid()
    if row_status[row] != ParticleContactStatus.ACTIVE:
        return
    wid = row_world[row]
    if not world_active[wid] or world_failed[wid]:
        return
    current = reaction[row]
    wp.atomic_add(restart_dot, read, wid, wp.dot(current - trial[row], current - previous[row]))


@wp.kernel
def _extrapolate_particle_reactions_and_velocities(
    # Inputs:
    row_capacity: wp.int32,
    row_world: wp.array[wp.int32],
    row_status: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    read: wp.int32,
    speed: wp.bool,
    first: wp.bool,
    energy: wp.bool,
    theta: wp.array2d[wp.float32],
    restart_dot: wp.array2d[wp.float32],
    curvature: wp.array2d[wp.float32],
    slope_trial: wp.array2d[wp.float32],
    # Outputs:
    reaction: wp.array[wp.vec3f],
    trial: wp.array[wp.vec3f],
    previous: wp.array[wp.vec3f],
    projected_velocity: wp.array[wp.vec3f],
    velocity_previous: wp.array[wp.vec3f],
    velocity_trial: wp.array[wp.vec3f],
):
    """Extrapolate each particle contact reaction and each projected particle velocity by the momentum of its
    world.

    The particle velocities are affine in the reactions, with weights fixed over a LOX solve, so they follow
    the same extrapolation, with no impulse scatter.
    """
    tid = wp.tid()
    if tid < row_capacity:
        row = tid
        if row_status[row] != ParticleContactStatus.ACTIVE:
            return
        wid = row_world[row]
        if not world_active[wid] or world_failed[wid]:
            return
        _next_theta, beta = world_momentum(theta, restart_dot, curvature, slope_trial, read, wid, speed, first, energy)
        current = reaction[row]
        extrapolated = current + beta * (current - previous[row])
        previous[row] = current
        trial[row] = extrapolated
        reaction[row] = extrapolated
        return
    pid = tid - row_capacity
    wid = particle_world[pid]
    if world_active[wid] and not world_failed[wid]:
        _next_theta, beta = world_momentum(theta, restart_dot, curvature, slope_trial, read, wid, speed, first, energy)
        current = projected_velocity[pid]
        extrapolated = current + beta * (current - velocity_previous[pid])
        projected_velocity[pid] = extrapolated
        velocity_previous[pid] = current
        if speed or energy:
            velocity_trial[pid] = extrapolated


@wp.kernel
def _difference_motion(
    # Inputs:
    body_count: wp.int32,
    body_world: wp.array[wp.int32],
    body_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    particle_world: wp.array[wp.int32],
    particle_inverse_weight: wp.array[wp.float32],
    projected_velocity: wp.array[wp.vec3f],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    measure: wp.bool,
    keep_final: wp.bool,
    # Outputs:
    twist_previous: wp.array[vec6f],
    velocity_previous: wp.array[wp.vec3f],
    twist_final: wp.array[vec6f],
    velocity_final: wp.array[wp.vec3f],
    drift: wp.array[wp.float32],
    step: wp.array[wp.float32],
):
    """Replace each previous body twist and particle velocity by its difference with the projected one.

    With ``keep_final``, at the end of a projection, also keep the projected twists and velocities. With
    ``measure``, at the warm start of a recycled projection, also accumulate per world the squared weighted
    drift from the kept ones and the squared weighted last step, which the previous values hold.
    """
    tid = wp.tid()
    if tid < body_count:
        bid = tid
        wid = body_world[bid]
        if world_active[wid] and not world_failed[wid]:
            current = projected_twist[bid]
            previous = twist_previous[bid]
            twist_previous[bid] = current - previous
            if keep_final:
                twist_final[bid] = current
            if measure:
                weight = body_weight[bid]
                change = current - twist_final[bid]
                wp.atomic_add(drift, wid, wp.dot(change, apply_body_weight(weight, change)))
                wp.atomic_add(step, wid, wp.dot(previous, apply_body_weight(weight, previous)))
        return
    pid = tid - body_count
    wid = particle_world[pid]
    if world_active[wid] and not world_failed[wid]:
        velocity = projected_velocity[pid]
        last = velocity_previous[pid]
        velocity_previous[pid] = velocity - last
        if keep_final:
            velocity_final[pid] = velocity
        inverse = particle_inverse_weight[pid]
        if measure and inverse > 0.0:
            velocity_drift = velocity - velocity_final[pid]
            wp.atomic_add(drift, wid, wp.dot(velocity_drift, velocity_drift) / inverse)
            wp.atomic_add(step, wid, wp.dot(last, last) / inverse)


@wp.kernel
def _accumulate_weighted_change(
    # Inputs:
    body_count: wp.int32,
    body_world: wp.array[wp.int32],
    body_weight: wp.array[mat66f],
    twist: wp.array[vec6f],
    twist_reference: wp.array[vec6f],
    particle_world: wp.array[wp.int32],
    particle_inverse_weight: wp.array[wp.float32],
    velocity: wp.array[wp.vec3f],
    velocity_reference: wp.array[wp.vec3f],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    row: wp.int32,
    sign: wp.float32,
    gradient: wp.bool,
    twist_previous: wp.array[vec6f],
    velocity_previous: wp.array[wp.vec3f],
    # Outputs:
    world_sum: wp.array2d[wp.float32],
):
    """Accumulate ``sign`` times the squared ``W``-norm of the change of the body twists and particle velocities
    per world, into ``world_sum[row]``.

    With ``gradient``, accumulate instead the weighted restart product ``(v - v_ref) . W (v - v_previous)``.
    """
    tid = wp.tid()
    if tid < body_count:
        wid = body_world[tid]
        if world_active[wid] and not world_failed[wid]:
            twist_change = twist[tid] - twist_reference[tid]
            twist_other = twist_change
            if gradient:
                twist_other = twist[tid] - twist_previous[tid]
            wp.atomic_add(
                world_sum, row, wid, sign * wp.dot(twist_other, apply_body_weight(body_weight[tid], twist_change))
            )
        return
    pid = tid - body_count
    wid = particle_world[pid]
    inverse = particle_inverse_weight[pid]
    if world_active[wid] and not world_failed[wid] and inverse > 0.0:
        velocity_change = velocity[pid] - velocity_reference[pid]
        velocity_other = velocity_change
        if gradient:
            velocity_other = velocity[pid] - velocity_previous[pid]
        wp.atomic_add(world_sum, row, wid, sign * wp.dot(velocity_other, velocity_change) / inverse)


@wp.kernel
def _gate_recycled_momentum(
    # Inputs:
    world_active: wp.array[wp.bool],
    first_buffer: wp.int32,
    # Outputs:
    drift_current: wp.array[wp.float32],
    drift_previous: wp.array[wp.float32],
    theta: wp.array2d[wp.float32],
):
    """Keep the recycled momentum of each world while the warm-started projections drift less and less.

    The drift is the weighted change between the warm start of this projection and the result of the
    previous one; a world whose drift grew, or without a previous drift, restarts its momentum.
    """
    wid = wp.tid()
    if not world_active[wid]:
        return
    current = drift_current[wid]
    previous = drift_previous[wid]
    if not (previous >= 0.0 and current <= previous):
        theta[first_buffer, wid] = 1.0
    drift_previous[wid] = current
    drift_current[wid] = 0.0


###
# Interfaces
###


class NesterovAcceleration:
    """Own the momentum state of Nesterov-accelerated projection."""

    checked_recycle = False
    """Whether a recycled projection keeps only ``theta``, gated on a shrinking warm-start drift, and skips the
    deferred extrapolation of the previous projection."""
    speed_restart = False
    """Whether the momentum restarts on a growing weighted sweep step, in place of the gradient restart."""
    energy_rule = True
    """Whether the momentum follows the projection energy: it restarts unless the last sweep descended from the
    extrapolated point and its change still descends, each extrapolation stops at the energy minimum along it, and
    recycled momentum is kept only while the splitting iterations converge. Overrides the other restart and
    recycle switches."""
    energy_de_saxce = True
    """With :attr:`energy_rule`, whether the energy slope includes the de Saxce shift ``mu |u_T|`` of each
    contact, frozen at the current reactions, whose Coulomb solution then zeroes the slope."""
    weighted_gradient_restart = False
    """Whether the momentum restarts on ``(v - v_trial) . W (v - v_previous) <= 0`` over the body twists and
    particle velocities, in place of the speed restart and the reaction-space gradient restart."""
    gated_recycle = False
    """Without :attr:`checked_recycle`, whether a recycled projection keeps its row history and deferred
    extrapolation only while the warm-start drift shrinks; otherwise ``theta`` restarts before it."""

    def __init__(self, problem_data: LOXProblemData, data: NesterovData):
        self.problem_data = problem_data
        self._data = data
        self._body_weight: wp.array[mat66f] | None = None
        self._particle_inverse_weight: wp.array[wp.float32] | None = None
        self._capacity = problem_data.box_rows.capacity + problem_data.contact_rows.capacity
        self._gate_pending = False
        self.row_state = NesterovRowState()
        """Trial and previous reactions, for the per-world projection."""
        for name in (
            "box_trial",
            "box_previous",
            "contact_trial",
            "contact_previous",
            "history",
            "twist_previous",
            "twist_trial",
            "twist_final",
            "drift_previous",
        ):
            setattr(self.row_state, name, getattr(data, name))
        # The per-world projection keeps theta in registers and stores it in the first buffer
        self.row_state.theta = data.theta[0]
        if data.angular_trial is not None:
            self.row_state.angular_trial = data.angular_trial
            self.row_state.angular_previous = data.angular_previous

    ###
    # Public API
    ###

    def set_weights(
        self, body_weight: wp.array[mat66f] | None, particle_inverse_weight: wp.array[wp.float32] | None
    ) -> None:
        """Set the body weights and particle inverse weights of the weighted step norms."""
        self._body_weight = body_weight
        self._particle_inverse_weight = particle_inverse_weight
        self.row_state.body_weight = body_weight

    def reset(self) -> None:
        """Restart the momentum and the row history of every world, once per LOX solve."""
        self._data.theta.fill_(1.0)
        self._data.drift_previous.fill_(-1.0)
        self._data.history.zero_()
        self._data.twist_previous.zero_()
        if self._data.velocity_previous is not None:
            self._data.velocity_previous.zero_()

    def begin(
        self, world_active: wp.array[wp.bool], extrapolations: int, recycle: bool = False, keep_rows: bool | None = None
    ) -> None:
        """Clamp the warm-start reactions and restart the row history.

        With ``recycle``, the worlds keep ``theta`` and the row history of the previous splitting iteration,
        whose deferred extrapolation :meth:`extrapolate` then applies after the warm start and
        :meth:`begin_twists`.

        Args:
            world_active: Worlds to project.
            extrapolations: Number of :meth:`extrapolate` calls of the projection.
            recycle: Whether the worlds keep the momentum and row history of the previous splitting iteration.
        """
        keep_rows = recycle if keep_rows is None else keep_rows
        # The buffers alternate at each extrapolation, starting so that the last one writes the first buffer
        self._extrapolations = extrapolations
        self._extrapolation = 0
        self._gate_pending = False
        p = self.problem_data
        angular_reaction, angular_friction, _frame = self._spatial_arrays()
        if self._capacity > 0:
            self._initialize_reactions(world_active, keep_rows, angular_reaction, angular_friction)
        rows = p.particle_contact_rows
        if rows is not None and rows.capacity > 0:
            wp.launch(
                _initialize_particle_reactions,
                dim=rows.capacity,
                inputs=[rows.world, rows.status, rows.friction, world_active, keep_rows, self._data.history],
                outputs=[rows.reaction, self._data.particle_trial, self._data.particle_previous],
                device=p.device,
            )
        # After the rows, which read the history flags of the previous splitting iteration
        wp.launch(
            _initialize_worlds,
            dim=p.num_worlds,
            inputs=[world_active, recycle, extrapolations % 2],
            outputs=[self._data.theta, self._data.restart_dot, self._data.history],
            device=p.device,
        )
        if not keep_rows:
            self._data.history.zero_()
        self._data.curvature.zero_()
        self._data.slope_trial.zero_()

    def begin_checked(
        self,
        body_world: wp.array[wp.int32],
        world_active: wp.array[wp.bool],
        world_failed: wp.array[wp.bool],
        recycle: bool,
        projected_twist: wp.array[vec6f],
        projected_velocity: wp.array[wp.vec3f] | None,
    ) -> None:
        """Start the momentum history at the warm start and, with ``recycle``, gate the recycled ``theta``."""
        self._copy_trial(projected_twist, projected_velocity)
        if recycle:
            self.gate_recycled(body_world, world_active, world_failed, projected_twist, projected_velocity)

    def gate_recycled(
        self,
        body_world: wp.array[wp.int32],
        world_active: wp.array[wp.bool],
        world_failed: wp.array[wp.bool],
        projected_twist: wp.array[vec6f],
        projected_velocity: wp.array[wp.vec3f] | None,
    ) -> None:
        """Restart ``theta`` of each world whose warm start drifted more from the previous projection than before."""
        self._weighted_change(
            body_world,
            world_active,
            world_failed,
            projected_twist,
            self._data.twist_final,
            projected_velocity,
            self._data.velocity_final,
            self._data.drift_current.reshape((1, self.problem_data.num_worlds)),
            0,
            1.0,
        )
        wp.launch(
            _gate_recycled_momentum,
            dim=self.problem_data.num_worlds,
            inputs=[world_active, self._extrapolations % 2],
            outputs=[self._data.drift_current, self._data.drift_previous, self._data.theta],
            device=self.problem_data.device,
        )

    def end_checked(self, projected_twist: wp.array[vec6f], projected_velocity: wp.array[wp.vec3f] | None) -> None:
        """Keep the result of the projection, for the drift of the next recycled projection."""
        wp.copy(self._data.twist_final, projected_twist)
        if self._has_particles(projected_velocity):
            wp.copy(self._data.velocity_final, projected_velocity)

    def _copy_trial(self, projected_twist, projected_velocity) -> None:
        wp.copy(self._data.twist_trial, projected_twist)
        if self._has_particles(projected_velocity):
            wp.copy(self._data.velocity_trial, projected_velocity)

    def _weighted_change(
        self,
        body_world,
        world_active,
        world_failed,
        twist,
        twist_reference,
        velocity,
        velocity_reference,
        out,
        row,
        sign,
        gradient=False,
    ) -> None:
        p = self.problem_data
        particles = self._has_particles(velocity)
        body_count = twist.shape[0]
        wp.launch(
            _accumulate_weighted_change,
            dim=body_count + (velocity.shape[0] if particles else 0),
            inputs=[
                body_count,
                body_world,
                self._body_weight,
                twist,
                twist_reference,
                p.particle_world if particles else None,
                self._particle_inverse_weight if particles else None,
                velocity if particles else None,
                velocity_reference if particles else None,
                world_active,
                world_failed,
                row,
                sign,
                gradient,
                self._data.twist_previous,
                self._data.velocity_previous if particles else None,
            ],
            outputs=[out],
            device=p.device,
        )

    def _initialize_reactions(self, world_active, recycle, angular_reaction, angular_friction) -> None:
        p = self.problem_data
        wp.launch(
            _initialize_reactions,
            dim=self._capacity,
            inputs=[
                p.box_rows.capacity,
                p.box_rows.world,
                p.box_rows.local,
                p.box_rows.world_count,
                p.box_rows.lower,
                p.box_rows.upper,
                p.contact_rows.world,
                p.contact_rows.local,
                p.contact_rows.world_count,
                p.contact_rows.body_a,
                p.contact_rows.body_b,
                p.contact_rows.friction,
                angular_friction,
                world_active,
                recycle,
                self._data.history,
            ],
            outputs=[
                p.box_rows.reaction,
                self._data.box_trial,
                self._data.box_previous,
                p.contact_rows.reaction,
                self._data.contact_trial,
                self._data.contact_previous,
                angular_reaction,
                self._data.angular_trial,
                self._data.angular_previous,
            ],
            device=p.device,
        )

    def begin_twists(
        self,
        body_world: wp.array[wp.int32],
        world_active: wp.array[wp.bool],
        world_failed: wp.array[wp.bool],
        recycle: bool,
        projected_twist: wp.array[vec6f],
        projected_velocity: wp.array[wp.vec3f] | None = None,
        measure: bool = False,
    ) -> None:
        """Set the previous twists and particle velocities of the warm-started reactions, after the warm start.

        With ``recycle``, the previous twists and velocities hold the change of the last sweep of the previous
        splitting iteration (:meth:`end_twists`), and become the warm-started ones minus that change. With
        ``measure``, also accumulate the drift of the candidate and the last-step norm, which gate the recycled
        momentum at the deferred extrapolation.
        """
        self._gate_pending = recycle and measure
        if not recycle:
            wp.copy(self._data.twist_previous, projected_twist)
            if self._has_particles(projected_velocity):
                wp.copy(self._data.velocity_previous, projected_velocity)
            return
        self._difference_motion(body_world, world_active, world_failed, projected_twist, projected_velocity, measure)

    def end_twists(
        self,
        body_world: wp.array[wp.int32],
        world_active: wp.array[wp.bool],
        world_failed: wp.array[wp.bool],
        projected_twist: wp.array[vec6f],
        projected_velocity: wp.array[wp.vec3f] | None = None,
        keep_final: bool = False,
    ) -> None:
        """Keep the twist and particle velocity changes of the last sweep, for the deferred extrapolation of a
        recycled projection, and with ``keep_final`` the projected twists and velocities, for its drift."""
        self._difference_motion(
            body_world, world_active, world_failed, projected_twist, projected_velocity, False, keep_final
        )

    def extrapolate(
        self,
        body_world: wp.array[wp.int32],
        world_active: wp.array[wp.bool],
        world_failed: wp.array[wp.bool],
        projected_twist: wp.array[vec6f],
        projected_velocity: wp.array[wp.vec3f] | None = None,
    ) -> None:
        """Update the momentum and extrapolate the reactions, the body twists, and the particle velocities of the
        last sweep."""
        p = self.problem_data
        read = (self._extrapolations + self._extrapolation) % 2
        first = self._extrapolation == 0
        self._extrapolation += 1
        angular_reaction, angular_friction, _frame = self._spatial_arrays()
        energy = self.energy_rule
        weighted = self.weighted_gradient_restart and not energy
        speed_step = self.speed_restart and not weighted and not energy
        speed = speed_step or weighted
        if energy:
            # The deferred extrapolation of a measured recycled projection also gates its momentum
            gate = first and self._gate_pending
            self._gate_pending = False
            self._accumulate_energy_slope(
                body_world, world_active, world_failed, read, projected_twist, projected_velocity, gate
            )
        elif weighted:
            self._weighted_change(
                body_world,
                world_active,
                world_failed,
                projected_twist,
                self._data.twist_trial,
                projected_velocity,
                self._data.velocity_trial,
                self._data.restart_dot,
                read,
                1.0,
                True,
            )
        elif speed:
            # Subtract the current weighted step from the previous one, loaded by the last extrapolation
            self._weighted_change(
                body_world,
                world_active,
                world_failed,
                projected_twist,
                self._data.twist_trial,
                projected_velocity,
                self._data.velocity_trial,
                self._data.restart_dot,
                read,
                -1.0,
            )
        elif self._capacity > 0:
            self._accumulate_restart_dot(world_active, world_failed, read, angular_reaction, angular_friction)
        rows = p.particle_contact_rows
        row_capacity = rows.capacity if rows is not None and not speed and not energy else 0
        if row_capacity > 0:
            wp.launch(
                _accumulate_particle_restart_dot,
                dim=row_capacity,
                inputs=[
                    rows.world,
                    rows.status,
                    rows.reaction,
                    self._data.particle_trial,
                    self._data.particle_previous,
                    world_active,
                    world_failed,
                    read,
                ],
                outputs=[self._data.restart_dot],
                device=p.device,
            )
        row_capacity = rows.capacity if rows is not None else 0
        # Reads the momentum buffers before the launch below writes the next ones
        if self._has_particles(projected_velocity) or row_capacity > 0:
            particle_count = projected_velocity.shape[0] if projected_velocity is not None else 0
            wp.launch(
                _extrapolate_particle_reactions_and_velocities,
                dim=row_capacity + particle_count,
                inputs=[
                    row_capacity,
                    rows.world,
                    rows.status,
                    p.particle_world,
                    world_active,
                    world_failed,
                    read,
                    speed,
                    first,
                    energy,
                    self._data.theta,
                    self._data.restart_dot,
                    self._data.curvature,
                    self._data.slope_trial,
                ],
                outputs=[
                    rows.reaction,
                    self._data.particle_trial,
                    self._data.particle_previous,
                    projected_velocity,
                    self._data.velocity_previous,
                    self._data.velocity_trial,
                ],
                device=p.device,
            )
        body_count = projected_twist.shape[0]
        wp.launch(
            _extrapolate_reactions_and_twists,
            dim=self._capacity + body_count + p.num_worlds,
            inputs=[
                p.box_rows.capacity,
                p.box_rows.world,
                p.box_rows.local,
                p.box_rows.world_count,
                p.contact_rows.capacity,
                p.contact_rows.world,
                p.contact_rows.local,
                p.contact_rows.world_count,
                body_count,
                body_world,
                world_active,
                world_failed,
                read,
                speed,
                first,
                speed_step,
                energy,
            ],
            outputs=[
                self._data.theta,
                self._data.restart_dot,
                self._data.curvature,
                self._data.slope_trial,
                self._data.step_previous,
                self._data.twist_trial,
                p.box_rows.reaction,
                self._data.box_trial,
                self._data.box_previous,
                p.contact_rows.reaction,
                self._data.contact_trial,
                self._data.contact_previous,
                angular_reaction,
                self._data.angular_trial,
                self._data.angular_previous,
                projected_twist,
                self._data.twist_previous,
            ],
            device=p.device,
        )

    def _accumulate_energy_slope(
        self, body_world, world_active, world_failed, read, projected_twist, projected_velocity, gate=False
    ) -> None:
        """Accumulate the slopes and curvature of the projection energy along the last change, per world.

        At the deferred extrapolation of a recycled projection, the twists hold the warm start of the new
        candidate, so the slope is that of the new energy along the momentum of the previous projection.
        """
        p = self.problem_data
        box = p.box_rows
        contacts = p.contact_rows
        rows = p.particle_contact_rows
        row_capacity = rows.capacity if rows is not None else 0
        particles = self._has_particles(projected_velocity)
        has_rows = row_capacity > 0
        spatial = contacts.angular_reaction is not None
        gate_offset = row_capacity if gate else -1
        wp.launch(
            _accumulate_energy_slope,
            dim=self._capacity + row_capacity + (p.num_worlds if gate else 0),
            inputs=[
                box.capacity,
                box.world,
                box.local,
                box.world_count,
                box.bias,
                box.reaction,
                self._data.box_previous,
                box.body_a,
                box.body_b,
                box.jacobian_a,
                box.jacobian_b,
                contacts.capacity,
                contacts.world,
                contacts.local,
                contacts.world_count,
                contacts.bias,
                contacts.normal_compliance,
                spatial,
                contacts.reaction,
                self._data.contact_previous,
                self._data.contact_trial,
                contacts.angular_reaction,
                self._data.angular_previous,
                self.energy_de_saxce,
                contacts.body_a,
                contacts.body_b,
                contacts.jacobian_a,
                contacts.jacobian_b,
                contacts.frame,
                contacts.friction,
                projected_twist,
                self._data.twist_previous,
                self._data.twist_trial,
                row_capacity,
                rows.world if has_rows else None,
                rows.status if has_rows else None,
                rows.bias if has_rows else None,
                rows.reaction if has_rows else None,
                self._data.particle_previous if has_rows else None,
                self.energy_de_saxce and has_rows,
                rows.particle if has_rows else None,
                rows.coefficient if has_rows else None,
                rows.body if has_rows else None,
                rows.frame if has_rows else None,
                rows.body_jacobian if has_rows else None,
                rows.friction if has_rows else None,
                projected_velocity if particles else None,
                self._data.velocity_previous if particles else None,
                self._data.velocity_trial if particles else None,
                world_active,
                world_failed,
                read,
                gate_offset,
            ],
            outputs=[
                self._data.restart_dot,
                self._data.slope_trial,
                self._data.curvature,
                self._data.drift_current,
                self._data.drift_previous,
                self._data.step_previous,
                self._data.theta,
            ],
            device=p.device,
        )

    def _accumulate_restart_dot(self, world_active, world_failed, read, angular_reaction, angular_friction) -> None:
        p = self.problem_data
        wp.launch(
            _accumulate_restart_dot,
            dim=self._capacity,
            inputs=[
                p.box_rows.capacity,
                p.box_rows.world,
                p.box_rows.local,
                p.box_rows.world_count,
                p.box_rows.reaction,
                self._data.box_trial,
                self._data.box_previous,
                p.contact_rows.world,
                p.contact_rows.local,
                p.contact_rows.world_count,
                p.contact_rows.reaction,
                self._data.contact_trial,
                self._data.contact_previous,
                angular_reaction,
                self._data.angular_trial,
                self._data.angular_previous,
                p.contact_rows.friction,
                angular_friction,
                world_active,
                world_failed,
                read,
            ],
            outputs=[self._data.restart_dot],
            device=p.device,
        )

    ###
    # Internals
    ###

    def _difference_motion(
        self, body_world, world_active, world_failed, projected_twist, projected_velocity, measure, keep_final=False
    ) -> None:
        p = self.problem_data
        particles = self._has_particles(projected_velocity)
        body_count = projected_twist.shape[0]
        wp.launch(
            _difference_motion,
            dim=body_count + (projected_velocity.shape[0] if particles else 0),
            inputs=[
                body_count,
                body_world,
                self._body_weight,
                projected_twist,
                p.particle_world if particles else None,
                self._particle_inverse_weight if particles else None,
                projected_velocity if particles else None,
                world_active,
                world_failed,
                measure,
                keep_final,
            ],
            outputs=[
                self._data.twist_previous,
                self._data.velocity_previous if particles else None,
                self._data.twist_final,
                self._data.velocity_final if particles else None,
                self._data.drift_current,
                self._data.step_previous,
            ],
            device=p.device,
        )

    def _has_particles(self, projected_velocity: wp.array[wp.vec3f] | None) -> bool:
        return projected_velocity is not None and projected_velocity.shape[0] > 0

    def _spatial_arrays(self):
        """Return the angular reaction, angular friction, and frame of spatial contacts, or ``None``."""
        rows = self.problem_data.contact_rows
        return rows.angular_reaction, rows.angular_friction, rows.frame
