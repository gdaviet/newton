# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernels adapting LOX results to Kamino integrators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ......core.reset import reset_world_selected
from ...core.types import vec6f

if TYPE_CHECKING:
    from ......sim import Contacts, Model, State
    from .solver import LOXSolver

__all__ = []

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _reset_newton_particle_state(
    particle_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    particle_q_default: wp.array[wp.vec3],
    particle_qd_default: wp.array[wp.vec3],
    reset_q: bool,
    reset_qd: bool,
    reset_f: bool,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
):
    particle = wp.tid()
    if reset_world_selected(particle_world[particle], world_mask, world_mask.shape[0] - 1):
        if reset_q:
            particle_q[particle] = particle_q_default[particle]
        if reset_qd:
            particle_qd[particle] = particle_qd_default[particle]
        if reset_f:
            particle_f[particle] = wp.vec3(0.0)


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
def _reset_newton_particle_dual_impulse(
    particle_world: wp.array[wp.int32],
    source_world_mask: wp.array[wp.bool],
    particle_dual_impulse: wp.array[wp.vec3],
):
    particle = wp.tid()
    world = wp.max(particle_world[particle], 0)
    selected = source_world_mask[world]
    if world == 0:
        selected = selected or source_world_mask[source_world_mask.shape[0] - 1]
    if selected:
        particle_dual_impulse[particle] = wp.vec3(0.0)


@wp.kernel
def _select_global_deformable_solve_world(
    source_world_mask: wp.array[wp.bool],
    solve_world_mask: wp.array[wp.bool],
):
    """Select partition zero only when its global deformables were reset."""
    world = wp.tid()
    solve_world_mask[world] = world == 0 and source_world_mask[source_world_mask.shape[0] - 1]


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


class _LOXDeformableIntegration:
    """Manage the Newton state needed by LOX deformables."""

    def __init__(self, model: Model):
        if model.particle_count == 0:
            raise ValueError("LOX deformable integration requires particles.")
        self._newton_model = model
        self._newton_state_in: State | None = None
        self._newton_state_out: State | None = None
        self._newton_contacts: Contacts | None = None
        self._newton_body_pose = (
            wp.empty_like(model.body_q) if model.particle_count > 0 and model.body_count > 0 else None
        )
        has_global_particles = bool((model.particle_world.numpy() == -1).any())
        self._global_reset_world_mask = (
            wp.empty(model.world_count, dtype=wp.bool, device=model.device) if has_global_particles else None
        )

    def _ensure_dual_impulse_state(
        self,
        state: State,
    ) -> tuple[wp.array[wp.spatial_vector] | None, wp.array[wp.vec3] | None]:
        model = self._newton_model

        def ensure(name: str, count: int, dtype: type) -> wp.array | None:
            if count == 0:
                return None
            value = getattr(state, name, None)
            if value is None:
                value = wp.zeros(count, dtype=dtype, device=model.device)
                setattr(state, name, value)
            if not isinstance(value, wp.array) or value.shape != (count,) or value.dtype != dtype:
                raise ValueError(f"State.{name} must have shape ({count},) and dtype {dtype}.")
            if value.device != model.device:
                raise ValueError(f"State.{name} must be allocated on {model.device}, found {value.device}.")
            return value

        return (
            ensure("body_lox_dual_impulse", model.body_count, wp.spatial_vector),
            ensure("particle_lox_dual_impulse", model.particle_count, wp.vec3),
        )

    def bind_step(
        self,
        solver: LOXSolver,
        state_in: State,
        state_out: State,
        contacts: Contacts | None,
    ) -> None:
        """Bind Newton warm starts and deformable step inputs and outputs."""
        body_in, particle_in = self._ensure_dual_impulse_state(state_in)
        self._ensure_dual_impulse_state(state_out)
        self._newton_state_in = state_in
        self._newton_state_out = state_out
        self._newton_contacts = contacts
        if self._newton_body_pose is not None:
            if state_in.body_q is None:
                raise ValueError("The LOX deformable path requires Newton body-origin poses.")
            wp.copy(self._newton_body_pose, state_in.body_q)
        solver.load_state_dual_impulses(body_in, particle_in)

    def begin_step(
        self,
        solver: LOXSolver,
        time_step: wp.array[wp.float32],
        inverse_time_step: wp.array[wp.float32],
        *,
        contact_stabilization_fraction: float,
        contact_dead_zone: float,
        impact_velocity_threshold: float,
        contact_recoverable_response: bool,
    ) -> None:
        """Prepare deformable coupling from retained Newton inputs."""
        if solver.deformable_system is None:
            return
        state_in = self._newton_state_in
        if state_in is None:
            raise ValueError("The LOX deformable path requires the original Newton input state.")
        solver.begin_deformable_time_step(
            state_in,
            self._newton_contacts,
            time_step,
            inverse_time_step,
            contact_stabilization_fraction=contact_stabilization_fraction,
            contact_dead_zone=contact_dead_zone,
            impact_velocity_threshold=impact_velocity_threshold,
            contact_recoverable_response=contact_recoverable_response,
            body_pose=self._newton_body_pose,
        )

    def finish_step(
        self,
        solver: LOXSolver,
        time_step: wp.array[wp.float32],
        inverse_time_step: wp.array[wp.float32],
    ) -> None:
        """Write deformable state and persistent impulses to Newton output."""
        state_out = self._newton_state_out
        if solver.deformable_system is not None:
            if state_out is None:
                raise ValueError("The LOX deformable path requires the original Newton output state.")
            solver.write_deformable_output(state_out, time_step, inverse_time_step)
        if state_out is not None:
            body_out, particle_out = self._ensure_dual_impulse_state(state_out)
            solver.write_state_dual_impulses(body_out, particle_out)

    def reset_deformable_state(
        self,
        state: State,
        state_flags: int,
        source_world_mask: wp.array[wp.bool] | None,
    ) -> wp.array[wp.bool] | None:
        """Reset deformable state and return any extra LOX partition reset."""
        from ......sim import StateFlags  # noqa: PLC0415

        model = self._newton_model
        if source_world_mask is not None:
            self._validate_source_world_mask(source_world_mask, "LOX reset")
        body_dual_impulse, particle_dual_impulse = self._ensure_dual_impulse_state(state)
        reset_body_qd = bool(state_flags & int(StateFlags.BODY_QD))
        reset_particle_qd = bool(state_flags & int(StateFlags.PARTICLE_QD))
        if model.particle_count > 0:
            arrays = {
                "particle_q": state.particle_q,
                "particle_qd": state.particle_qd,
                "particle_f": state.particle_f,
            }
            expected_shape = (model.particle_count,)
            for name, value in arrays.items():
                if value is None or value.shape != expected_shape:
                    raise ValueError(f"LOX deformable reset requires State.{name} with shape {expected_shape}.")
                if value.device != model.device:
                    raise ValueError(
                        f"LOX deformable reset expected State.{name} on {model.device}, found {value.device}."
                    )

            reset_q = bool(state_flags & int(StateFlags.PARTICLE_Q))
            reset_f = bool(state_flags & int(StateFlags.PARTICLE_F))
            if source_world_mask is None:
                if reset_q:
                    wp.copy(state.particle_q, model.particle_q)
                if reset_particle_qd:
                    wp.copy(state.particle_qd, model.particle_qd)
                if reset_f:
                    state.particle_f.zero_()
            else:
                wp.launch(
                    _reset_newton_particle_state,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_world,
                        source_world_mask,
                        model.particle_q,
                        model.particle_qd,
                        reset_q,
                        reset_particle_qd,
                        reset_f,
                    ],
                    outputs=[state.particle_q, state.particle_qd, state.particle_f],
                    device=model.device,
                )

        if source_world_mask is None:
            if reset_body_qd and body_dual_impulse is not None:
                body_dual_impulse.zero_()
            if reset_particle_qd and particle_dual_impulse is not None:
                particle_dual_impulse.zero_()
            return None

        if reset_body_qd and body_dual_impulse is not None:
            wp.launch(
                _reset_newton_body_dual_impulse,
                dim=model.body_count,
                inputs=[model.body_world, source_world_mask],
                outputs=[body_dual_impulse],
                device=model.device,
            )
        if reset_particle_qd and particle_dual_impulse is not None:
            wp.launch(
                _reset_newton_particle_dual_impulse,
                dim=model.particle_count,
                inputs=[model.particle_world, source_world_mask],
                outputs=[particle_dual_impulse],
                device=model.device,
            )

        global_reset_world_mask = self._global_reset_world_mask
        if global_reset_world_mask is None:
            return None
        wp.launch(
            _select_global_deformable_solve_world,
            dim=model.world_count,
            inputs=[source_world_mask],
            outputs=[global_reset_world_mask],
            device=model.device,
        )
        return global_reset_world_mask

    def _validate_source_world_mask(self, world_mask: wp.array[wp.bool], operation: str) -> None:
        model = self._newton_model
        expected_shape = (model.world_count + 1,)
        if world_mask.shape != expected_shape or world_mask.dtype != wp.bool:
            raise ValueError(f"{operation} world_mask must have shape {expected_shape} and dtype bool.")
        if world_mask.device != model.device:
            raise ValueError(f"{operation} expected world_mask on {model.device}, found {world_mask.device}.")
