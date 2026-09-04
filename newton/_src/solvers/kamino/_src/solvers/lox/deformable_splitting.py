# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistent nodal consensus state for the LOX cloth solve."""

from __future__ import annotations

import numpy as np
import warp as wp

from .deformable_preconditioner import DEFORMABLE_PRECONDITIONER_STATUS_FAILED
from .time import validate_world_time_step

__all__ = ["DeformableSplittingState"]

PARTICLE_FLAG_ACTIVE = 1

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _gather_state_dual_impulse(
    packed_to_newton: wp.array[wp.int32],
    state_dual_impulse: wp.array[wp.vec3],
    packed_dual_impulse: wp.array[wp.vec3],
):
    particle = wp.tid()
    packed_dual_impulse[particle] = state_dual_impulse[packed_to_newton[particle]]


@wp.kernel
def _scatter_state_dual_impulse(
    packed_to_newton: wp.array[wp.int32],
    packed_dual_impulse: wp.array[wp.vec3],
    state_dual_impulse: wp.array[wp.vec3],
):
    particle = wp.tid()
    state_dual_impulse[packed_to_newton[particle]] = packed_dual_impulse[particle]


@wp.kernel
def _begin_particles(
    consensus_enabled: wp.array[wp.int32],
    full_inverse_weight: wp.array[float],
    initial_velocity: wp.array[wp.vec3],
    external_force: wp.array[wp.vec3],
    packed_to_newton: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    particle_mass: wp.array[float],
    initialized: wp.array[wp.int32],
    retain_projected: bool,
    time_step: wp.array[wp.float32],
    external_force_fraction: float,
    projected_velocity: wp.array[wp.vec3],
    projected_velocity_previous: wp.array[wp.vec3],
    global_velocity: wp.array[wp.vec3],
    global_velocity_previous: wp.array[wp.vec3],
    dual: wp.array[wp.vec3],
    dual_impulse: wp.array[wp.vec3],
    accepted_velocity: wp.array[wp.vec3],
):
    particle = wp.tid()
    world = packed_world[particle]
    dt = time_step[world]
    inverse_particle_weight = full_inverse_weight[particle]
    initial = initial_velocity[particle]
    enabled = consensus_enabled[particle] != 0
    dynamic = inverse_particle_weight > 0.0
    if dynamic and external_force_fraction > 0.0:
        mass = particle_mass[packed_to_newton[particle]]
        initial += external_force_fraction * dt * external_force[particle] / mass
    projected = projected_velocity[particle] if retain_projected and enabled and initialized[world] != 0 else initial
    if not dynamic:
        projected = wp.vec3(0.0)
    if not enabled:
        dual[particle] = wp.vec3(0.0)
        dual_impulse[particle] = wp.vec3(0.0)
    projected_velocity[particle] = projected
    projected_velocity_previous[particle] = projected
    global_velocity[particle] = projected
    global_velocity_previous[particle] = projected
    if enabled:
        dual[particle] = inverse_particle_weight * dual_impulse[particle]
    if accepted_velocity:
        accepted_velocity[particle] = initial_velocity[particle]


@wp.kernel
def _reset_particles_masked(
    packed_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    projected_velocity: wp.array[wp.vec3],
    projected_velocity_previous: wp.array[wp.vec3],
    global_velocity: wp.array[wp.vec3],
    global_velocity_previous: wp.array[wp.vec3],
    dual: wp.array[wp.vec3],
    dual_impulse: wp.array[wp.vec3],
    consensus_center: wp.array[wp.vec3],
    accepted_velocity: wp.array[wp.vec3],
):
    particle = wp.tid()
    if world_mask[packed_world[particle]]:
        projected_velocity[particle] = wp.vec3(0.0)
        projected_velocity_previous[particle] = wp.vec3(0.0)
        global_velocity[particle] = wp.vec3(0.0)
        global_velocity_previous[particle] = wp.vec3(0.0)
        dual[particle] = wp.vec3(0.0)
        dual_impulse[particle] = wp.vec3(0.0)
        consensus_center[particle] = wp.vec3(0.0)
        accepted_velocity[particle] = wp.vec3(0.0)


@wp.kernel
def _reset_worlds_masked(
    world_mask: wp.array[wp.bool],
    initialized: wp.array[wp.int32],
    outer_accepted: wp.array[wp.bool],
    iteration_failed: wp.array[wp.int32],
    consensus_residual: wp.array[wp.float32],
    iterate_residual: wp.array[wp.float32],
    displacement_residual: wp.array[wp.float32],
    cloth_residual: wp.array[wp.float32],
    cloth_converged: wp.array[wp.bool],
    cloth_failed: wp.array[wp.bool],
):
    world = wp.tid()
    if world_mask[world]:
        initialized[world] = 0
        outer_accepted[world] = False
        iteration_failed[world] = 0
        consensus_residual[world] = 0.0
        iterate_residual[world] = 0.0
        displacement_residual[world] = 0.0
        cloth_residual[world] = 0.0
        cloth_converged[world] = False
        cloth_failed[world] = False


@wp.kernel
def _build_consensus_center(
    particle_count: int,
    world_count: int,
    shared_world_active: wp.array[wp.bool],
    projected_velocity: wp.array[wp.vec3],
    dual: wp.array[wp.vec3],
    deformable_world_active: wp.array[wp.int32],
    consensus_center: wp.array[wp.vec3],
):
    index = wp.tid()
    if index < world_count:
        deformable_world_active[index] = wp.where(shared_world_active[index], 1, 0)
    if index < particle_count:
        consensus_center[index] = projected_velocity[index] + dual[index]


@wp.kernel
def _prepare_projection(
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    consensus_enabled: wp.array[wp.int32],
    full_inverse_weight: wp.array[float],
    candidate_velocity: wp.array[wp.vec3],
    global_velocity_previous: wp.array[wp.vec3],
    global_velocity: wp.array[wp.vec3],
    projected_velocity_previous: wp.array[wp.vec3],
    projected_velocity: wp.array[wp.vec3],
    dual: wp.array[wp.vec3],
):
    particle = wp.tid()
    if world_active[packed_world[particle]] == 0:
        return
    global_velocity_previous[particle] = global_velocity[particle]
    global_velocity[particle] = candidate_velocity[particle]
    projected_velocity_previous[particle] = projected_velocity[particle]
    if consensus_enabled[particle] != 0:
        projected_velocity[particle] = candidate_velocity[particle] - dual[particle]
    elif full_inverse_weight[particle] > 0.0:
        projected_velocity[particle] = candidate_velocity[particle]
    else:
        projected_velocity[particle] = wp.vec3(0.0)
        dual[particle] = wp.vec3(0.0)


@wp.kernel
def _initialize_residuals(
    world_active: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
    iteration_failed: wp.array[wp.int32],
    consensus_residual: wp.array[float],
    iterate_residual: wp.array[float],
    displacement_residual: wp.array[float],
):
    world = wp.tid()
    if iteration_count and world_active[world]:
        iteration_count[world] += 1
    iteration_failed[world] = 0
    consensus_residual[world] = 0.0
    iterate_residual[world] = 0.0
    displacement_residual[world] = 0.0


@wp.kernel
def _finish_particles(
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    global_velocity: wp.array[wp.vec3],
    projected_velocity: wp.array[wp.vec3],
    global_velocity_previous: wp.array[wp.vec3],
    projected_velocity_previous: wp.array[wp.vec3],
    consensus_enabled: wp.array[wp.int32],
    time_step: wp.array[wp.float32],
    dual: wp.array[wp.vec3],
    iteration_failed: wp.array[wp.int32],
    consensus_residual: wp.array[float],
    iterate_residual: wp.array[float],
    displacement_residual: wp.array[float],
):
    particle = wp.tid()
    world = packed_world[particle]
    if world_active[world] == 0:
        return

    global_current = global_velocity[particle]
    projected_current = projected_velocity[particle]
    global_previous = global_velocity_previous[particle]
    projected_previous = projected_velocity_previous[particle]
    finite = True
    for axis in range(3):
        finite = (
            finite
            and wp.isfinite(global_current[axis])
            and wp.isfinite(projected_current[axis])
            and wp.isfinite(global_previous[axis])
            and wp.isfinite(projected_previous[axis])
            and wp.isfinite(dual[particle][axis])
        )
    if not finite:
        wp.atomic_max(iteration_failed, world, 1)
        return

    consensus = wp.vec3(global_current - projected_current)
    iterate = wp.vec3(projected_current - projected_previous)
    global_change = wp.vec3(global_current - global_previous)
    consensus_norm = wp.max(wp.abs(consensus[0]), wp.abs(consensus[1]))
    consensus_norm = wp.max(consensus_norm, wp.abs(consensus[2]))
    iterate_norm = wp.max(wp.abs(iterate[0]), wp.abs(iterate[1]))
    iterate_norm = wp.max(iterate_norm, wp.abs(iterate[2]))
    global_change_norm = wp.max(wp.abs(global_change[0]), wp.abs(global_change[1]))
    global_change_norm = wp.max(global_change_norm, wp.abs(global_change[2]))
    wp.atomic_max(consensus_residual, world, consensus_norm)
    wp.atomic_max(iterate_residual, world, iterate_norm)
    wp.atomic_max(displacement_residual, world, time_step[world] * (global_change_norm + consensus_norm))
    if consensus_enabled[particle] != 0:
        dual[particle] += projected_current - global_current
    else:
        dual[particle] = wp.vec3(0.0)


@wp.kernel
def _finalize_worlds(
    world_has_particles: wp.array[wp.int32],
    preconditioner_status: wp.array[wp.int32],
    proximal_failed: wp.array[wp.int32],
    proximal_position_residual: wp.array[float],
    proximal_velocity_residual: wp.array[float],
    contact_world_status: wp.array[wp.int32],
    contact_global_status: wp.array[wp.int32],
    contact_residual: wp.array[float],
    consensus_residual: wp.array[float],
    iterate_residual: wp.array[float],
    displacement_residual: wp.array[float],
    position_tolerance: float,
    velocity_tolerance: float,
    iteration_failed: wp.array[wp.int32],
    cloth_converged: wp.array[wp.bool],
    cloth_failed: wp.array[wp.bool],
    cloth_residual: wp.array[float],
    world_active: wp.array[wp.bool],
    world_converged: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    residual_total: wp.array[float],
):
    world = wp.tid()
    deformable_converged = wp.bool(True)
    deformable_failed = wp.bool(False)
    deformable_residual = wp.float32(0.0)
    if world_has_particles[world] != 0:
        contact_failed = wp.bool(False)
        if contact_global_status:
            contact_failed = contact_global_status[0] > 1 or contact_world_status[world] > 1
        deformable_failed = (
            iteration_failed[world] != 0
            or preconditioner_status[world] == DEFORMABLE_PRECONDITIONER_STATUS_FAILED
            or proximal_failed[world] != 0
            or contact_failed
        )
        if deformable_failed:
            deformable_converged = False
        else:
            deformable_residual = wp.max(
                consensus_residual[world] / velocity_tolerance,
                iterate_residual[world] / velocity_tolerance,
            )
            deformable_residual = wp.max(
                deformable_residual,
                displacement_residual[world] / position_tolerance,
            )
            deformable_residual = wp.max(
                deformable_residual,
                proximal_position_residual[world] / position_tolerance,
            )
            deformable_residual = wp.max(
                deformable_residual,
                proximal_velocity_residual[world] / velocity_tolerance,
            )
            if contact_residual:
                deformable_residual = wp.max(
                    deformable_residual,
                    contact_residual[world] / velocity_tolerance,
                )
            deformable_converged = deformable_residual <= 1.0

    cloth_converged[world] = deformable_converged
    cloth_failed[world] = deformable_failed
    cloth_residual[world] = deformable_residual
    failed = world_failed[world] or deformable_failed
    converged = world_converged[world] and deformable_converged and not failed
    world_failed[world] = failed
    world_converged[world] = converged
    world_active[world] = not failed and not converged
    residual_total[world] = wp.max(residual_total[world], deformable_residual)


@wp.kernel
def _accept_projected_velocity(
    packed_to_newton: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    velocity_start: wp.array[wp.vec3],
    projected_velocity: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    weight: wp.array[float],
    dual: wp.array[wp.vec3],
    world_accepted: wp.array[wp.bool],
    accepted_velocity: wp.array[wp.vec3],
    dual_impulse: wp.array[wp.vec3],
    outer_accepted: wp.array[wp.bool],
):
    packed_particle = wp.tid()
    particle = packed_to_newton[packed_particle]
    world = packed_world[packed_particle]
    dual_impulse[packed_particle] = weight[packed_particle] * dual[packed_particle]
    if world_accepted[world]:
        outer_accepted[world] = True
        if particle_mass[particle] > 0.0 and (particle_flags[particle] & PARTICLE_FLAG_ACTIVE) != 0:
            accepted_velocity[packed_particle] = projected_velocity[packed_particle]
        else:
            accepted_velocity[packed_particle] = velocity_start[packed_particle]


@wp.kernel
def _write_particles(
    packed_to_newton: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    position_start: wp.array[wp.vec3],
    velocity_start: wp.array[wp.vec3],
    projected_velocity: wp.array[wp.vec3],
    world_accepted: wp.array[wp.bool],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    time_step: wp.array[wp.float32],
    position_out: wp.array[wp.vec3],
    velocity_out: wp.array[wp.vec3],
):
    packed_particle = wp.tid()
    particle = packed_to_newton[packed_particle]
    world = packed_world[packed_particle]
    dt = time_step[world]
    if not world_accepted[world]:
        position_out[particle] = position_start[packed_particle]
        velocity_out[particle] = velocity_start[packed_particle]
    elif (particle_flags[particle] & PARTICLE_FLAG_ACTIVE) == 0:
        position_out[particle] = position_start[packed_particle]
        velocity_out[particle] = wp.vec3(0.0)
    elif particle_mass[particle] <= 0.0:
        velocity = velocity_start[packed_particle]
        position_out[particle] = position_start[packed_particle] + dt * velocity
        velocity_out[particle] = velocity
    else:
        velocity = projected_velocity[packed_particle]
        position_out[particle] = position_start[packed_particle] + dt * velocity
        velocity_out[particle] = velocity


class DeformableSplittingState:
    """Own cloth consensus iterates, residuals, and impulse warm starts."""

    def __init__(self, cloth_system):
        self.cloth_system = cloth_system
        self.device = cloth_system.device
        self.particle_count = cloth_system.particle_count
        self.num_worlds = int(cloth_system.model.world_count)

        packed_world = cloth_system.topology.packed_solve_world.numpy()
        world_has_particles = np.bincount(packed_world, minlength=self.num_worlds).astype(bool)
        self.world_has_particles = wp.array(world_has_particles, dtype=wp.int32, device=self.device)
        self.world_active = cloth_system.world_active
        self.initialized = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.projected_velocity = wp.zeros(self.particle_count, dtype=wp.vec3, device=self.device)
        self.projected_velocity_previous = wp.zeros_like(self.projected_velocity)
        self.global_velocity = wp.zeros_like(self.projected_velocity)
        self.global_velocity_previous = wp.zeros_like(self.projected_velocity)
        self.dual = wp.zeros_like(self.projected_velocity)
        self.dual_impulse = wp.zeros_like(self.projected_velocity)
        self.consensus_center = wp.zeros_like(self.projected_velocity)
        self.accepted_velocity = wp.zeros_like(self.projected_velocity)
        self.outer_accepted = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)

        self.iteration_failed = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.consensus_residual = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.iterate_residual = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.displacement_residual = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.cloth_residual = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.cloth_converged = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        self.cloth_failed = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)

    def reset(self, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Clear cloth warm starts and diagnostics."""
        if world_mask is None:
            self.initialized.zero_()
            self.projected_velocity.zero_()
            self.projected_velocity_previous.zero_()
            self.global_velocity.zero_()
            self.global_velocity_previous.zero_()
            self.dual.zero_()
            self.dual_impulse.zero_()
            self.consensus_center.zero_()
            self.accepted_velocity.zero_()
            self.outer_accepted.zero_()
            self.iteration_failed.zero_()
            self.consensus_residual.zero_()
            self.iterate_residual.zero_()
            self.displacement_residual.zero_()
            self.cloth_residual.zero_()
            self.cloth_converged.zero_()
            self.cloth_failed.zero_()
            return

        wp.launch(
            _reset_particles_masked,
            dim=self.particle_count,
            inputs=[self.cloth_system.topology.packed_solve_world, world_mask],
            outputs=[
                self.projected_velocity,
                self.projected_velocity_previous,
                self.global_velocity,
                self.global_velocity_previous,
                self.dual,
                self.dual_impulse,
                self.consensus_center,
                self.accepted_velocity,
            ],
            device=self.device,
        )
        wp.launch(
            _reset_worlds_masked,
            dim=self.num_worlds,
            inputs=[world_mask],
            outputs=[
                self.initialized,
                self.outer_accepted,
                self.iteration_failed,
                self.consensus_residual,
                self.iterate_residual,
                self.displacement_residual,
                self.cloth_residual,
                self.cloth_converged,
                self.cloth_failed,
            ],
            device=self.device,
        )

    def load_state_dual_impulse(self, state_dual_impulse: wp.array[wp.vec3]) -> None:
        """Load the Newton-ordered nodal impulse warm start into packed storage."""
        if state_dual_impulse.shape != (self.particle_count,) or state_dual_impulse.dtype != wp.vec3:
            raise ValueError(f"particle_lox_dual_impulse must have shape ({self.particle_count},) and dtype {wp.vec3}.")
        if state_dual_impulse.device != self.device:
            raise ValueError(
                f"particle_lox_dual_impulse must be allocated on {self.device}, found {state_dual_impulse.device}."
            )
        wp.launch(
            _gather_state_dual_impulse,
            dim=self.particle_count,
            inputs=[self.cloth_system.topology.packed_to_newton, state_dual_impulse],
            outputs=[self.dual_impulse],
            device=self.device,
        )

    def write_state_dual_impulse(self, state_dual_impulse: wp.array[wp.vec3]) -> None:
        """Write the packed nodal impulse warm start in Newton particle order."""
        if state_dual_impulse.shape != (self.particle_count,) or state_dual_impulse.dtype != wp.vec3:
            raise ValueError(f"particle_lox_dual_impulse must have shape ({self.particle_count},) and dtype {wp.vec3}.")
        if state_dual_impulse.device != self.device:
            raise ValueError(
                f"particle_lox_dual_impulse must be allocated on {self.device}, found {state_dual_impulse.device}."
            )
        wp.launch(
            _scatter_state_dual_impulse,
            dim=self.particle_count,
            inputs=[self.cloth_system.topology.packed_to_newton, self.dual_impulse],
            outputs=[state_dual_impulse],
            device=self.device,
        )

    def begin(self, time_step: wp.array[wp.float32]) -> None:
        """Restore the scaled nodal dual and retain the projected warm start."""
        validate_world_time_step(time_step, self.num_worlds, self.device)
        wp.launch(
            _begin_particles,
            dim=self.particle_count,
            inputs=[
                self.cloth_system.consensus_enabled,
                self.cloth_system.full_inverse_weight,
                self.cloth_system.velocity_linearized,
                self.cloth_system.external_force,
                self.cloth_system.topology.packed_to_newton,
                self.cloth_system.topology.packed_solve_world,
                self.cloth_system.model.particle_mass,
                self.initialized,
                True,
                time_step,
                0.0,
            ],
            outputs=[
                self.projected_velocity,
                self.projected_velocity_previous,
                self.global_velocity,
                self.global_velocity_previous,
                self.dual,
                self.dual_impulse,
                None,
            ],
            device=self.device,
        )
        self.initialized.fill_(1)

    def begin_time_step(
        self,
        time_step: wp.array[wp.float32],
        inertial_warmstart_fraction: float,
    ) -> None:
        """Begin splitting from step-start velocity plus fractional external acceleration."""
        validate_world_time_step(time_step, self.num_worlds, self.device)
        wp.launch(
            _begin_particles,
            dim=self.particle_count,
            inputs=[
                self.cloth_system.consensus_enabled,
                self.cloth_system.full_inverse_weight,
                self.cloth_system.velocity_start,
                self.cloth_system.external_force,
                self.cloth_system.topology.packed_to_newton,
                self.cloth_system.topology.packed_solve_world,
                self.cloth_system.model.particle_mass,
                self.initialized,
                False,
                time_step,
                inertial_warmstart_fraction,
            ],
            outputs=[
                self.projected_velocity,
                self.projected_velocity_previous,
                self.global_velocity,
                self.global_velocity_previous,
                self.dual,
                self.dual_impulse,
                self.accepted_velocity,
            ],
            device=self.device,
        )
        self.initialized.fill_(1)
        self.outer_accepted.zero_()

    def build_consensus_center(self, shared_world_active: wp.array[wp.bool]) -> wp.array[wp.vec3]:
        """Copy the active mask and form the current nodal center ``p + lambda``."""
        wp.launch(
            _build_consensus_center,
            dim=max(self.particle_count, self.num_worlds),
            inputs=[
                self.particle_count,
                self.num_worlds,
                shared_world_active,
                self.projected_velocity,
                self.dual,
            ],
            outputs=[self.world_active, self.consensus_center],
            device=self.device,
        )
        return self.consensus_center

    def prepare_projection(self, candidate_velocity: wp.array[wp.vec3]) -> None:
        """Store the smooth candidate and initialize ``p = v - lambda``."""
        wp.launch(
            _prepare_projection,
            dim=self.particle_count,
            inputs=[
                self.cloth_system.topology.packed_solve_world,
                self.world_active,
                self.cloth_system.consensus_enabled,
                self.cloth_system.full_inverse_weight,
                candidate_velocity,
            ],
            outputs=[
                self.global_velocity_previous,
                self.global_velocity,
                self.projected_velocity_previous,
                self.projected_velocity,
                self.dual,
            ],
            device=self.device,
        )

    def finish_iteration(
        self,
        world_active: wp.array[wp.bool],
        world_converged: wp.array[wp.bool],
        world_failed: wp.array[wp.bool],
        iteration_count: wp.array[wp.int32],
        residual_total: wp.array[float],
        time_step: wp.array[wp.float32],
        position_tolerance: float,
        velocity_tolerance: float,
        contact_system=None,
        rigid_projected_twist=None,
        increment_iteration_count: bool = False,
    ) -> None:
        """Update the nodal dual and merge cloth status into shared worlds."""
        wp.launch(
            _initialize_residuals,
            dim=self.num_worlds,
            inputs=[world_active],
            outputs=[
                iteration_count if increment_iteration_count else None,
                self.iteration_failed,
                self.consensus_residual,
                self.iterate_residual,
                self.displacement_residual,
            ],
            device=self.device,
        )
        wp.launch(
            _finish_particles,
            dim=self.particle_count,
            inputs=[
                self.cloth_system.topology.packed_solve_world,
                self.world_active,
                self.global_velocity,
                self.projected_velocity,
                self.global_velocity_previous,
                self.projected_velocity_previous,
                self.cloth_system.consensus_enabled,
                time_step,
            ],
            outputs=[
                self.dual,
                self.iteration_failed,
                self.consensus_residual,
                self.iterate_residual,
                self.displacement_residual,
            ],
            device=self.device,
        )
        if contact_system is None:
            contact_residual = None
            contact_world_status = None
            contact_global_status = None
        else:
            contact_system.compute_contact_residuals(
                self.projected_velocity,
                projected_twist=rigid_projected_twist,
            )
            contact_residual = contact_system.world_contact_residual
            contact_world_status = contact_system.world_status
            contact_global_status = contact_system.global_status

        wp.launch(
            _finalize_worlds,
            dim=self.num_worlds,
            inputs=[
                self.world_has_particles,
                self.cloth_system.preconditioner.world_status,
                self.cloth_system.proximal_failed,
                self.cloth_system.proximal_position_residual,
                self.cloth_system.proximal_velocity_residual,
                contact_world_status,
                contact_global_status,
                contact_residual,
                self.consensus_residual,
                self.iterate_residual,
                self.displacement_residual,
                position_tolerance,
                velocity_tolerance,
                self.iteration_failed,
            ],
            outputs=[
                self.cloth_converged,
                self.cloth_failed,
                self.cloth_residual,
                world_active,
                world_converged,
                world_failed,
                residual_total,
            ],
            device=self.device,
        )

    def accept_projected(self, world_accepted: wp.array[wp.bool]) -> None:
        """Persist the nodal dual and retain velocities accepted by the outer solve."""
        wp.launch(
            _accept_projected_velocity,
            dim=self.particle_count,
            inputs=[
                self.cloth_system.topology.packed_to_newton,
                self.cloth_system.topology.packed_solve_world,
                self.cloth_system.velocity_start,
                self.projected_velocity,
                self.cloth_system.model.particle_mass,
                self.cloth_system.model.particle_flags,
                self.cloth_system.weight,
                self.dual,
                world_accepted,
            ],
            outputs=[self.accepted_velocity, self.dual_impulse, self.outer_accepted],
            device=self.device,
        )

    def write_output(self, state_out, time_step: wp.array[wp.float32]) -> None:
        """Write accepted projected particle positions and velocities."""
        if state_out.particle_q is None or state_out.particle_qd is None:
            raise ValueError("LOX cloth requires particle positions and velocities in the output state.")
        validate_world_time_step(time_step, self.num_worlds, self.device)
        wp.launch(
            _write_particles,
            dim=self.particle_count,
            inputs=[
                self.cloth_system.topology.packed_to_newton,
                self.cloth_system.topology.packed_solve_world,
                self.cloth_system.position_start,
                self.cloth_system.velocity_start,
                self.accepted_velocity,
                self.outer_accepted,
                self.cloth_system.model.particle_mass,
                self.cloth_system.model.particle_flags,
                time_step,
            ],
            outputs=[state_out.particle_q, state_out.particle_qd],
            device=self.device,
        )
