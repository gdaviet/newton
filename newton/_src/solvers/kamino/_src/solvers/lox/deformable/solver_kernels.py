# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernels of the particle splitting iterations of the LOX deformable bodies.

The particles of the deformable bodies are the 3D counterparts of the body twists of
:mod:`..solver_kernels`: their splitting iterates are seeded from the begin-step
velocity, the smooth solve gives their candidate velocity ``v``, the projection their
projected velocity ``p``, and the splitting dual ``u`` accumulates ``p - v``. Their
convergence residuals reduce into the per-world residuals of the bodies. Particles
without a consensus weight keep ``p = v``, and prescribed particles keep ``p = 0``;
their velocity enters the bias of their contact rows.
"""

import warp as wp

from .......core.reset import reset_world_selected
from ..solver_kernels import atomic_max_nonnegative
from .types import DeformablePreconditionerStatus

###
# Module interface
###

__all__ = [
    "_blend_accepted_particle_velocity",
    "_compute_particle_consensus_center",
    "_initialize_particles",
    "_prepare_particle_projection",
    "_reduce_deformable_status",
    "_reduce_final_particle_residuals",
    "_reduce_particle_residuals",
    "_reset_particle_dual_force",
    "_reset_particle_state",
    "_restore_particle_dual",
    "_select_global_particle_solve_world",
    "_store_particle_dual_force",
    "_update_particle_splitting_dual",
    "_write_particle_outputs",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


_PARTICLE_FLAG_ACTIVE = wp.constant(1)
"""Newton particle flag of the simulated particles."""


###
# Kernels
###


@wp.kernel
def _initialize_particles(
    # Inputs:
    packed_to_newton: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    particle_mass: wp.array[wp.float32],
    full_inverse_weight: wp.array[wp.float32],
    velocity_begin: wp.array[wp.vec3f],
    external_force: wp.array[wp.vec3f],
    dual_force: wp.array[wp.vec3f],
    model_time_dt: wp.array[wp.float32],
    inertial_fraction: wp.float32,
    # Outputs:
    projected_velocity: wp.array[wp.vec3f],
    projected_velocity_previous: wp.array[wp.vec3f],
    global_velocity: wp.array[wp.vec3f],
    global_velocity_previous: wp.array[wp.vec3f],
):
    """Seed the particle iterates with the begin-step velocity advanced by a fraction of the predicted step.

    A unit fraction seeds the predicted velocity ``v + dt m^-1 (f + f_u)`` of the external force ``f``
    and the warm-started force ``f_u`` of the contacts, like :func:`..solver_kernels._initialize_bodies`.
    Prescribed particles, with zero inverse weight, keep a zero iterate.
    """
    pid = wp.tid()
    wid = particle_world[pid]
    if world_mask and not world_mask[wid]:
        return
    value = wp.vec3f(0.0)
    if full_inverse_weight[pid] > 0.0:
        value = velocity_begin[pid]
        if inertial_fraction > 0.0:
            force = external_force[pid]
            if dual_force:
                force += dual_force[pid]
            value += inertial_fraction * model_time_dt[wid] * force / particle_mass[packed_to_newton[pid]]
    projected_velocity[pid] = value
    projected_velocity_previous[pid] = value
    global_velocity[pid] = value
    global_velocity_previous[pid] = value


@wp.kernel
def _restore_particle_dual(
    # Inputs:
    particle_world: wp.array[wp.int32],
    consensus_enabled: wp.array[wp.int32],
    model_time_dt: wp.array[wp.float32],
    inverse_weight: wp.array[wp.float32],
    dual_force: wp.array[wp.vec3f],
    # Outputs:
    splitting_dual: wp.array[wp.vec3f],
):
    """Convert the warm-start force ``w u / dt`` back to the scaled splitting dual ``u``.

    The splitting dual of particles without a consensus weight is zero.
    """
    pid = wp.tid()
    splitting_dual[pid] = wp.vec3f(0.0)
    if consensus_enabled[pid] != 0:
        splitting_dual[pid] = model_time_dt[particle_world[pid]] * inverse_weight[pid] * dual_force[pid]


@wp.kernel
def _compute_particle_consensus_center(
    # Inputs:
    consensus_fraction: wp.float32,
    projected_velocity: wp.array[wp.vec3f],
    splitting_dual: wp.array[wp.vec3f],
    # Outputs:
    consensus_center: wp.array[wp.vec3f],
):
    """Form the splitting target ``u + alpha p`` of the candidate solve of each particle."""
    pid = wp.tid()
    consensus_center[pid] = splitting_dual[pid] + consensus_fraction * projected_velocity[pid]


@wp.kernel
def _reduce_deformable_status(
    # Inputs:
    position_tolerance: wp.float32,
    velocity_tolerance: wp.float32,
    world_active: wp.array[wp.bool],
    proximal_failed: wp.array[wp.bool],
    assembly_failed: wp.array[wp.bool],
    preconditioner_status: wp.array[wp.int32],
    proximal_position_residual: wp.array[wp.float32],
    proximal_velocity_residual: wp.array[wp.float32],
    candidate_linear_residual: wp.array[wp.float32],
    # Outputs:
    world_failed: wp.array[wp.bool],
    deformable_residual: wp.array[wp.float32],
):
    """Fail the worlds whose deformable assembly, preconditioner, or element updates failed, and reduce the
    scaled residuals of the element updates and of the candidate solve of the others."""
    wid = wp.tid()
    if not world_active[wid]:
        return
    preconditioner_failed = False
    if preconditioner_status:
        preconditioner_failed = preconditioner_status[wid] == DeformablePreconditionerStatus.FAILED
    if proximal_failed[wid] or assembly_failed[wid] or preconditioner_failed:
        world_failed[wid] = True
        return
    residual = wp.max(
        proximal_position_residual[wid] / position_tolerance, proximal_velocity_residual[wid] / velocity_tolerance
    )
    deformable_residual[wid] = wp.max(residual, candidate_linear_residual[wid] / velocity_tolerance)


@wp.kernel
def _prepare_particle_projection(
    # Inputs:
    particle_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    consensus_enabled: wp.array[wp.int32],
    full_inverse_weight: wp.array[wp.float32],
    candidate_velocity: wp.array[wp.vec3f],
    # Outputs:
    splitting_dual: wp.array[wp.vec3f],
    global_velocity_previous: wp.array[wp.vec3f],
    global_velocity: wp.array[wp.vec3f],
    projected_velocity_previous: wp.array[wp.vec3f],
    projected_velocity: wp.array[wp.vec3f],
):
    """Store the candidate velocity ``v`` of the smooth solve and seed the projection with ``p = v - u``."""
    pid = wp.tid()
    if not world_active[particle_world[pid]]:
        return
    candidate = candidate_velocity[pid]
    global_velocity_previous[pid] = global_velocity[pid]
    global_velocity[pid] = candidate
    projected_velocity_previous[pid] = projected_velocity[pid]
    if consensus_enabled[pid] != 0:
        projected_velocity[pid] = candidate - splitting_dual[pid]
    elif full_inverse_weight[pid] > 0.0:
        projected_velocity[pid] = candidate
    else:
        projected_velocity[pid] = wp.vec3f(0.0)
        splitting_dual[pid] = wp.vec3f(0.0)


@wp.kernel
def _update_particle_splitting_dual(
    # Inputs:
    particle_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    consensus_enabled: wp.array[wp.int32],
    global_velocity: wp.array[wp.vec3f],
    projected_velocity: wp.array[wp.vec3f],
    # Outputs:
    splitting_dual: wp.array[wp.vec3f],
):
    """Accumulate the consensus gap ``p - v`` into the scaled splitting dual ``u`` of each particle."""
    pid = wp.tid()
    wid = particle_world[pid]
    if world_active[wid] and not world_failed[wid] and consensus_enabled[pid] != 0:
        splitting_dual[pid] += projected_velocity[pid] - global_velocity[pid]


@wp.kernel
def _reduce_particle_residuals(
    # Inputs:
    model_time_dt: wp.array[wp.float32],
    position_tolerance: wp.float32,
    velocity_tolerance: wp.float32,
    particle_world: wp.array[wp.int32],
    consensus_enabled: wp.array[wp.int32],
    global_velocity_previous: wp.array[wp.vec3f],
    global_velocity: wp.array[wp.vec3f],
    projected_velocity_previous: wp.array[wp.vec3f],
    projected_velocity: wp.array[wp.vec3f],
    world_active: wp.array[wp.bool],
    # Outputs:
    splitting_dual: wp.array[wp.vec3f],
    world_failed: wp.array[wp.bool],
    residual_change: wp.array[wp.float32],
    residual_split: wp.array[wp.float32],
    residual_cross_iterate: wp.array[wp.float32],
):
    """Advance the splitting dual, fail the worlds with non-finite iterates, and reduce the convergence
    residuals of the particle velocities, with the scaling of the body twists."""
    pid = wp.tid()
    wid = particle_world[pid]
    if not world_active[wid]:
        return
    current = global_velocity[pid]
    projected = projected_velocity[pid]
    dual = splitting_dual[pid]
    if consensus_enabled[pid] != 0:
        dual += projected - current
        splitting_dual[pid] = dual
    if not (wp.isfinite(current) and wp.isfinite(projected) and wp.isfinite(dual)):
        world_failed[wid] = True
        return
    dt = model_time_dt[wid]
    change = dt * wp.max(wp.abs(current - global_velocity_previous[pid])) / position_tolerance
    split = wp.max(wp.abs(current - projected)) / velocity_tolerance
    cross_iterate = dt * wp.max(wp.abs(current - projected_velocity_previous[pid])) / position_tolerance
    atomic_max_nonnegative(residual_change, wid, change)
    atomic_max_nonnegative(residual_split, wid, split)
    atomic_max_nonnegative(residual_cross_iterate, wid, cross_iterate)


@wp.kernel
def _reduce_final_particle_residuals(
    # Inputs:
    particle_world: wp.array[wp.int32],
    weight: wp.array[wp.float32],
    global_velocity_previous: wp.array[wp.vec3f],
    global_velocity: wp.array[wp.vec3f],
    projected_velocity: wp.array[wp.vec3f],
    splitting_dual: wp.array[wp.vec3f],
    # Outputs:
    world_failed: wp.array[wp.bool],
    primal_residual: wp.array[wp.float32],
    dual_residual: wp.array[wp.float32],
):
    """Fail the worlds with non-finite particle iterates and reduce the reported consensus gap ``|w (v - p)|``
    and the change ``|v - v_prev|`` of the others."""
    pid = wp.tid()
    wid = particle_world[pid]
    if world_failed[wid]:
        return
    current = global_velocity[pid]
    projected = projected_velocity[pid]
    if not (wp.isfinite(current) and wp.isfinite(projected) and wp.isfinite(splitting_dual[pid])):
        world_failed[wid] = True
        return
    atomic_max_nonnegative(primal_residual, wid, weight[pid] * wp.max(wp.abs(current - projected)))
    atomic_max_nonnegative(dual_residual, wid, wp.max(wp.abs(current - global_velocity_previous[pid])))


@wp.kernel
def _store_particle_dual_force(
    # Inputs:
    particle_world: wp.array[wp.int32],
    consensus_enabled: wp.array[wp.int32],
    world_failed: wp.array[wp.bool],
    model_time_inv_dt: wp.array[wp.float32],
    weight: wp.array[wp.float32],
    splitting_dual: wp.array[wp.vec3f],
    # Outputs:
    dual_force: wp.array[wp.vec3f],
):
    """Convert the scaled splitting dual ``u`` of each particle to the warm-start force ``w u / dt``.

    Failed worlds keep the warm start of their last accepted time step.
    """
    pid = wp.tid()
    wid = particle_world[pid]
    if world_failed[wid]:
        return
    dual_force[pid] = wp.vec3f(0.0)
    if consensus_enabled[pid] != 0:
        dual_force[pid] = model_time_inv_dt[wid] * weight[pid] * splitting_dual[pid]


@wp.kernel
def _reset_particle_dual_force(
    # Inputs:
    particle_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    # Outputs:
    dual_force: wp.array[wp.vec3f],
):
    """Clear the warm-start force of the particles of the masked worlds."""
    pid = wp.tid()
    if world_mask[particle_world[pid]]:
        dual_force[pid] = wp.vec3f(0.0)


@wp.kernel
def _reset_particle_state(
    # Inputs:
    particle_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    particle_q_default: wp.array[wp.vec3f],
    particle_qd_default: wp.array[wp.vec3f],
    reset_q: wp.bool,
    reset_qd: wp.bool,
    reset_f: wp.bool,
    # Outputs:
    particle_q: wp.array[wp.vec3f],
    particle_qd: wp.array[wp.vec3f],
    particle_f: wp.array[wp.vec3f],
):
    """Reset the Newton state of the particles of the masked worlds, with the global world in the last entry."""
    particle = wp.tid()
    if reset_world_selected(particle_world[particle], world_mask, world_mask.shape[0] - 1):
        if reset_q:
            particle_q[particle] = particle_q_default[particle]
        if reset_qd:
            particle_qd[particle] = particle_qd_default[particle]
        if reset_f:
            particle_f[particle] = wp.vec3f(0.0)


@wp.kernel
def _select_global_particle_solve_world(
    # Inputs:
    world_mask: wp.array[wp.bool],
    # Outputs:
    solve_world_mask: wp.array[wp.bool],
):
    """Select the first solve world, which holds the global particles, when the global world is masked."""
    wid = wp.tid()
    solve_world_mask[wid] = wid == 0 and world_mask[world_mask.shape[0] - 1]


@wp.kernel
def _blend_accepted_particle_velocity(
    # Inputs:
    projected_fraction: wp.float32,
    projected_velocity: wp.array[wp.vec3f],
    # Outputs:
    global_velocity: wp.array[wp.vec3f],
):
    """Blend the projected velocity into the smooth velocity, ``v <- f p + (1 - f) v``, as the accepted velocity."""
    pid = wp.tid()
    global_velocity[pid] = (
        projected_fraction * projected_velocity[pid] + (1.0 - projected_fraction) * global_velocity[pid]
    )


@wp.kernel
def _write_particle_outputs(
    # Inputs:
    packed_to_newton: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    world_failed: wp.array[wp.bool],
    model_time_dt: wp.array[wp.float32],
    particle_mass: wp.array[wp.float32],
    particle_flags: wp.array[wp.int32],
    particle_q_begin: wp.array[wp.vec3f],
    particle_qd_begin: wp.array[wp.vec3f],
    accepted_velocity: wp.array[wp.vec3f],
    # Outputs:
    particle_q: wp.array[wp.vec3f],
    particle_qd: wp.array[wp.vec3f],
):
    """Integrate the accepted particle velocities with ``x = x_0 + dt v``.

    Failed worlds keep their begin-step particle state, inactive particles keep their position
    with zero velocity, and massless particles move with their begin-step velocity. The output
    may alias the begin-step state.
    """
    pid = wp.tid()
    particle = packed_to_newton[pid]
    wid = particle_world[pid]
    position = particle_q_begin[particle]
    velocity = particle_qd_begin[particle]
    if not world_failed[wid]:
        if (particle_flags[particle] & _PARTICLE_FLAG_ACTIVE) == 0:
            velocity = wp.vec3f(0.0)
        else:
            if particle_mass[particle] > 0.0:
                velocity = accepted_velocity[pid]
            position += model_time_dt[wid] * velocity
    particle_q[particle] = position
    particle_qd[particle] = velocity
