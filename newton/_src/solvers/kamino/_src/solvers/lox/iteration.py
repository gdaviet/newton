# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body-space state for LOX splitting iterations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp

from ...core.types import mat66f, vec6f
from .projection import PROJECTION_STATUS_VALID
from .time import validate_world_time_step

__all__ = ["SplittingState"]

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _initialize_bodies(
    body_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    initial_twist: wp.array[vec6f],
    reset_dual: wp.bool,
    projected_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    global_twist_previous: wp.array[vec6f],
    splitting_dual: wp.array[vec6f],
    splitting_dual_impulse: wp.array[vec6f],
):
    body = wp.tid()
    if world_mask and not world_mask[body_world[body]]:
        return
    value = vec6f(0.0)
    if initial_twist:
        value = initial_twist[body]
    projected_twist[body] = value
    projected_twist_previous[body] = value
    global_twist[body] = value
    global_twist_previous[body] = value
    if reset_dual:
        splitting_dual[body] = vec6f(0.0)
        splitting_dual_impulse[body] = vec6f(0.0)


@wp.kernel
def _initialize_worlds(
    world_mask: wp.array[wp.bool],
    world_active: wp.array[wp.bool],
    world_converged: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    world_iteration_limit: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
    residual_change: wp.array[wp.float32],
    residual_split: wp.array[wp.float32],
    residual_structural: wp.array[wp.float32],
    residual_structural_projected: wp.array[wp.float32],
    residual_cross_iterate: wp.array[wp.float32],
    residual_lagged_velocity: wp.array[wp.float32],
    residual_total: wp.array[wp.float32],
    iteration_failed: wp.array[wp.int32],
):
    world = wp.tid()
    if world_mask and not world_mask[world]:
        return
    world_active[world] = True
    world_converged[world] = False
    world_failed[world] = False
    world_iteration_limit[world] = False
    iteration_count[world] = 0
    residual_change[world] = 0.0
    residual_split[world] = 0.0
    residual_structural[world] = 0.0
    residual_structural_projected[world] = 0.0
    residual_cross_iterate[world] = 0.0
    residual_lagged_velocity[world] = 0.0
    residual_total[world] = 0.0
    iteration_failed[world] = 0


@wp.kernel
def _prepare_projection(
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    body_split_enabled: wp.array[wp.int32],
    global_solution: wp.array[vec6f],
    splitting_dual: wp.array[vec6f],
    global_twist_previous: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
):
    body = wp.tid()
    world = body_world[body]
    if not world_active[world]:
        return
    global_twist_previous[body] = global_twist[body]
    global_twist[body] = global_solution[body]
    projected_twist_previous[body] = projected_twist[body]
    if not body_split_enabled or body_split_enabled[body] != 0:
        projected_twist[body] = global_solution[body] - splitting_dual[body]
    else:
        projected_twist[body] = global_solution[body]
        splitting_dual[body] = vec6f(0.0)


@wp.kernel
def _store_dual_impulse(
    body_has_unilateral: wp.array[wp.int32],
    weight: wp.array[mat66f],
    splitting_dual: wp.array[vec6f],
    splitting_dual_impulse: wp.array[vec6f],
):
    body = wp.tid()
    if body_has_unilateral[body] != 0:
        splitting_dual_impulse[body] = weight[body] @ splitting_dual[body]
    else:
        splitting_dual[body] = vec6f(0.0)
        splitting_dual_impulse[body] = vec6f(0.0)


@wp.kernel
def _restore_dual_from_impulse(
    body_has_unilateral: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    splitting_dual_impulse: wp.array[vec6f],
    splitting_dual: wp.array[vec6f],
):
    body = wp.tid()
    if body_has_unilateral[body] != 0:
        splitting_dual[body] = inverse_weight[body] @ splitting_dual_impulse[body]
    else:
        splitting_dual_impulse[body] = vec6f(0.0)
        splitting_dual[body] = vec6f(0.0)


@wp.func
def _is_finite_twist(value: vec6f) -> wp.bool:
    finite = wp.bool(True)
    for axis in range(6):
        finite = finite and wp.isfinite(value[axis])
    return finite


@wp.kernel
def _begin_residual_iteration(
    projection_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
    iteration_failed: wp.array[wp.int32],
    residual_change: wp.array[wp.float32],
    residual_split: wp.array[wp.float32],
    residual_cross_iterate: wp.array[wp.float32],
):
    world = wp.tid()
    if not world_active[world]:
        return
    iteration_count[world] += 1
    iteration_failed[world] = 0
    residual_change[world] = 0.0
    residual_split[world] = 0.0
    residual_cross_iterate[world] = 0.0
    if projection_status[world] != PROJECTION_STATUS_VALID:
        world_active[world] = False
        world_failed[world] = True


@wp.kernel
def _update_bodies_and_reduce_residuals(
    time_step: wp.array[wp.float32],
    position_tolerance: wp.float32,
    rotation_tolerance: wp.float32,
    velocity_tolerance: wp.float32,
    body_world: wp.array[wp.int32],
    global_twist_previous: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
    world_active: wp.array[wp.bool],
    splitting_dual: wp.array[vec6f],
    iteration_failed: wp.array[wp.int32],
    residual_change: wp.array[wp.float32],
    residual_split: wp.array[wp.float32],
    residual_cross_iterate: wp.array[wp.float32],
):
    body = wp.tid()
    world = body_world[body]
    dt = time_step[world]
    if not world_active[world]:
        return

    previous = global_twist_previous[body]
    current = global_twist[body]
    projected_previous = projected_twist_previous[body]
    projected = projected_twist[body]
    dual = splitting_dual[body]
    finite = (
        _is_finite_twist(previous)
        and _is_finite_twist(current)
        and _is_finite_twist(projected_previous)
        and _is_finite_twist(projected)
        and _is_finite_twist(dual)
    )
    if not finite:
        wp.atomic_max(iteration_failed, world, 1)
        return

    linear_change = wp.float32(0.0)
    angular_change = wp.float32(0.0)
    linear_split = wp.float32(0.0)
    angular_split = wp.float32(0.0)
    linear_cross_iterate = wp.float32(0.0)
    angular_cross_iterate = wp.float32(0.0)
    for axis in range(3):
        linear_change = wp.max(linear_change, wp.abs(current[axis] - previous[axis]))
        angular_change = wp.max(angular_change, wp.abs(current[axis + 3] - previous[axis + 3]))
        linear_split = wp.max(linear_split, wp.abs(current[axis] - projected[axis]))
        angular_split = wp.max(angular_split, wp.abs(current[axis + 3] - projected[axis + 3]))
        linear_cross_iterate = wp.max(
            linear_cross_iterate,
            wp.abs(current[axis] - projected_previous[axis]),
        )
        angular_cross_iterate = wp.max(
            angular_cross_iterate,
            wp.abs(current[axis + 3] - projected_previous[axis + 3]),
        )
    change = wp.max(
        dt * linear_change / position_tolerance,
        dt * angular_change / rotation_tolerance,
    )
    split = wp.max(linear_split, angular_split) / velocity_tolerance
    cross_iterate = wp.max(
        dt * linear_cross_iterate / position_tolerance,
        dt * angular_cross_iterate / rotation_tolerance,
    )
    wp.atomic_max(residual_change, world, change)
    wp.atomic_max(residual_split, world, split)
    wp.atomic_max(residual_cross_iterate, world, cross_iterate)
    splitting_dual[body] += projected - current


@wp.kernel
def _begin_fixed_iteration(
    projection_status: wp.array[wp.int32],
    proximal_failed: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
):
    world = wp.tid()
    if not world_active[world]:
        return
    iteration_count[world] += 1
    if projection_status[world] != PROJECTION_STATUS_VALID or (proximal_failed and proximal_failed[world] != 0):
        world_active[world] = False
        world_failed[world] = True


@wp.kernel
def _update_fixed_iteration_bodies(
    body_world: wp.array[wp.int32],
    global_twist: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    splitting_dual: wp.array[vec6f],
):
    body = wp.tid()
    world = body_world[body]
    if not world_active[world]:
        return
    current = global_twist[body]
    projected = projected_twist[body]
    dual = splitting_dual[body]
    finite = _is_finite_twist(current) and _is_finite_twist(projected) and _is_finite_twist(dual)
    if finite:
        splitting_dual[body] = dual + projected - current
    else:
        world_active[world] = False
        world_failed[world] = True


@wp.kernel
def _finalize_residual_iteration(
    structural_residual: wp.array[wp.float32],
    projected_structural_residual: wp.array[wp.float32],
    lagged_velocity_residual: wp.array[wp.float32],
    lagged_velocity_required: wp.array[wp.int32],
    effort_residual: wp.array[wp.float32],
    proximal_residual: wp.array[wp.float32],
    proximal_failed: wp.array[wp.int32],
    iteration_count: wp.array[wp.int32],
    iteration_failed: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_converged: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    residual_change: wp.array[wp.float32],
    residual_split: wp.array[wp.float32],
    residual_structural: wp.array[wp.float32],
    residual_structural_projected: wp.array[wp.float32],
    residual_cross_iterate: wp.array[wp.float32],
    residual_lagged_velocity: wp.array[wp.float32],
    residual_total: wp.array[wp.float32],
):
    world = wp.tid()
    if not world_active[world]:
        return
    if iteration_failed[world] != 0:
        world_active[world] = False
        world_failed[world] = True
        return
    if proximal_failed and proximal_failed[world] != 0:
        world_active[world] = False
        world_failed[world] = True
        return

    change = residual_change[world]
    split = residual_split[world]
    cross_iterate = residual_cross_iterate[world]
    structural = wp.float32(0.0)
    if structural_residual:
        structural = structural_residual[world]
    projected_structural = wp.float32(0.0)
    if projected_structural_residual:
        projected_structural = projected_structural_residual[world]
    lagged_velocity = wp.float32(0.0)
    if lagged_velocity_residual:
        lagged_velocity = lagged_velocity_residual[world]
    total = wp.max(wp.max(change, split), wp.max(wp.max(structural, cross_iterate), lagged_velocity))
    if effort_residual:
        total = wp.max(total, effort_residual[world])
    if proximal_residual:
        total = wp.max(total, proximal_residual[world])
    residual_structural[world] = structural
    residual_structural_projected[world] = projected_structural
    residual_lagged_velocity[world] = lagged_velocity
    residual_total[world] = total
    lagged_velocity_is_valid = (
        not lagged_velocity_required or lagged_velocity_required[world] == 0 or iteration_count[world] >= 2
    )
    if total <= 1.0 and lagged_velocity_is_valid:
        world_active[world] = False
        world_converged[world] = True


@wp.kernel
def _mark_iteration_limit(
    world_active: wp.array[wp.bool],
    world_iteration_limit: wp.array[wp.bool],
):
    world = wp.tid()
    if world_active[world]:
        world_active[world] = False
        world_iteration_limit[world] = True


class SplittingState:
    """Persistent body/world state for a batched LOX solve."""

    def __init__(self, body_counts: Sequence[int], device: wp.DeviceLike = None):
        if len(body_counts) == 0:
            raise ValueError("At least one world is required.")
        if any(not isinstance(count, int) or count < 0 for count in body_counts) or sum(body_counts) == 0:
            raise ValueError("Body counts must be non-negative and include at least one active body.")

        self.device = wp.get_device(device)
        self.body_counts = tuple(body_counts)
        self.num_worlds = len(body_counts)
        self.num_bodies = sum(body_counts)
        body_world = np.repeat(
            np.arange(self.num_worlds, dtype=np.int32),
            np.asarray(body_counts, dtype=np.int32),
        )

        self.body_world = wp.array(body_world, dtype=wp.int32, device=self.device)
        self.projected_twist = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.projected_twist_previous = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.global_twist = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.global_twist_previous = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.splitting_dual = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.splitting_dual_impulse = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.world_active = wp.ones(self.num_worlds, dtype=wp.bool, device=self.device)
        self.world_converged = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        self.world_failed = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        self.world_iteration_limit = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        self.iteration_count = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.residual_change = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.residual_split = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.residual_structural = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.residual_structural_projected = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.residual_cross_iterate = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.residual_lagged_velocity = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.residual_total = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self._iteration_failed = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)

    def _initialize(
        self,
        world_mask: wp.array[wp.bool] | None,
        initial_twist: wp.array[vec6f] | None,
        reset_dual: bool,
    ) -> None:
        wp.launch(
            _initialize_bodies,
            dim=self.num_bodies,
            inputs=[self.body_world, world_mask, initial_twist, reset_dual],
            outputs=[
                self.projected_twist,
                self.projected_twist_previous,
                self.global_twist,
                self.global_twist_previous,
                self.splitting_dual,
                self.splitting_dual_impulse,
            ],
            device=self.device,
        )
        wp.launch(
            _initialize_worlds,
            dim=self.num_worlds,
            inputs=[world_mask],
            outputs=[
                self.world_active,
                self.world_converged,
                self.world_failed,
                self.world_iteration_limit,
                self.iteration_count,
                self.residual_change,
                self.residual_split,
                self.residual_structural,
                self.residual_structural_projected,
                self.residual_cross_iterate,
                self.residual_lagged_velocity,
                self.residual_total,
                self._iteration_failed,
            ],
            device=self.device,
        )

    def reset(self, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Reset body-space warm starts and world diagnostics."""
        if world_mask is not None and world_mask.shape[0] != self.num_worlds:
            raise ValueError("world_mask must contain one entry per world.")
        self._initialize(world_mask, initial_twist=None, reset_dual=True)

    def begin(self, initial_twist: wp.array[vec6f], reset_dual: bool = False) -> None:
        """Initialize one nonlinear solve while optionally retaining its impulse warm start."""
        if initial_twist.shape[0] != self.num_bodies:
            raise ValueError("initial_twist must contain one entry per packed active body.")
        self._initialize(world_mask=None, initial_twist=initial_twist, reset_dual=reset_dual)

    def store_dual_impulse(
        self,
        weight: wp.array[mat66f],
        body_has_unilateral: wp.array[wp.int32],
    ) -> None:
        """Store the physical body impulse ``W u`` for later warm starting."""
        if weight.shape[0] != self.num_bodies or body_has_unilateral.shape[0] != self.num_bodies:
            raise ValueError("Weight and unilateral mask arrays must contain one entry per body.")
        wp.launch(
            _store_dual_impulse,
            dim=self.num_bodies,
            inputs=[body_has_unilateral, weight],
            outputs=[self.splitting_dual, self.splitting_dual_impulse],
            device=self.device,
        )

    def restore_dual_from_impulse(
        self,
        inverse_weight: wp.array[mat66f],
        body_has_unilateral: wp.array[wp.int32],
    ) -> None:
        """Recover the scaled dual ``u = W^-1 (W u)`` for the current weight."""
        if inverse_weight.shape[0] != self.num_bodies or body_has_unilateral.shape[0] != self.num_bodies:
            raise ValueError("Inverse weight and unilateral mask arrays must contain one entry per body.")
        wp.launch(
            _restore_dual_from_impulse,
            dim=self.num_bodies,
            inputs=[body_has_unilateral, inverse_weight],
            outputs=[self.splitting_dual_impulse, self.splitting_dual],
            device=self.device,
        )

    def prepare_projection(
        self,
        global_solution: wp.array[vec6f],
        body_split_enabled: wp.array[wp.int32] | None = None,
    ) -> None:
        """Store the global solution and initialize ``p = v - u``."""
        if global_solution.shape[0] != self.num_bodies:
            raise ValueError("global_solution must contain one entry per packed active body.")
        if body_split_enabled is not None and body_split_enabled.shape[0] != self.num_bodies:
            raise ValueError("body_split_enabled must contain one entry per packed active body.")
        wp.launch(
            _prepare_projection,
            dim=self.num_bodies,
            inputs=[self.body_world, self.world_active, body_split_enabled, global_solution],
            outputs=[
                self.splitting_dual,
                self.global_twist_previous,
                self.global_twist,
                self.projected_twist_previous,
                self.projected_twist,
            ],
            device=self.device,
        )

    def finish_iteration(
        self,
        projection_status: wp.array[wp.int32],
        time_step: wp.array[wp.float32],
        position_tolerance: float,
        rotation_tolerance: float,
        velocity_tolerance: float,
        effort_residual: wp.array[wp.float32] | None = None,
        structural_residual: wp.array[wp.float32] | None = None,
        projected_structural_residual: wp.array[wp.float32] | None = None,
        lagged_velocity_residual: wp.array[wp.float32] | None = None,
        lagged_velocity_required: wp.array[wp.int32] | None = None,
        proximal_residual: wp.array[wp.float32] | None = None,
        proximal_failed: wp.array[wp.int32] | None = None,
    ) -> None:
        """Update duals and residuals, then test per-world convergence."""
        if projection_status.shape[0] != self.num_worlds:
            raise ValueError("projection_status must contain one entry per world.")
        optional_world_arrays = (
            ("effort_residual", effort_residual),
            ("structural_residual", structural_residual),
            ("projected_structural_residual", projected_structural_residual),
            ("lagged_velocity_residual", lagged_velocity_residual),
            ("lagged_velocity_required", lagged_velocity_required),
            ("proximal_residual", proximal_residual),
            ("proximal_failed", proximal_failed),
        )
        for name, array in optional_world_arrays:
            if array is not None and array.shape[0] != self.num_worlds:
                raise ValueError(f"{name} must contain one entry per world.")
        validate_world_time_step(time_step, self.num_worlds, self.device)
        if position_tolerance <= 0.0 or rotation_tolerance <= 0.0 or velocity_tolerance <= 0.0:
            raise ValueError("Convergence tolerances must be positive.")
        wp.launch(
            _begin_residual_iteration,
            dim=self.num_worlds,
            inputs=[
                projection_status,
            ],
            outputs=[
                self.world_active,
                self.world_failed,
                self.iteration_count,
                self._iteration_failed,
                self.residual_change,
                self.residual_split,
                self.residual_cross_iterate,
            ],
            device=self.device,
        )
        wp.launch(
            _update_bodies_and_reduce_residuals,
            dim=self.num_bodies,
            inputs=[
                time_step,
                position_tolerance,
                rotation_tolerance,
                velocity_tolerance,
                self.body_world,
                self.global_twist_previous,
                self.global_twist,
                self.projected_twist_previous,
                self.projected_twist,
                self.world_active,
            ],
            outputs=[
                self.splitting_dual,
                self._iteration_failed,
                self.residual_change,
                self.residual_split,
                self.residual_cross_iterate,
            ],
            device=self.device,
        )
        wp.launch(
            _finalize_residual_iteration,
            dim=self.num_worlds,
            inputs=[
                structural_residual,
                projected_structural_residual,
                lagged_velocity_residual,
                lagged_velocity_required,
                effort_residual,
                proximal_residual,
                proximal_failed,
                self.iteration_count,
                self._iteration_failed,
            ],
            outputs=[
                self.world_active,
                self.world_converged,
                self.world_failed,
                self.residual_change,
                self.residual_split,
                self.residual_structural,
                self.residual_structural_projected,
                self.residual_cross_iterate,
                self.residual_lagged_velocity,
                self.residual_total,
            ],
            device=self.device,
        )

    def finish_fixed_iteration(
        self,
        projection_status: wp.array[wp.int32],
        proximal_failed: wp.array[wp.int32] | None = None,
    ) -> None:
        """Update the dual state and failures for a fixed-count iteration."""
        if projection_status.shape[0] != self.num_worlds:
            raise ValueError("projection_status must contain one entry per world.")
        if proximal_failed is not None and proximal_failed.shape[0] != self.num_worlds:
            raise ValueError("proximal_failed must contain one entry per world.")
        wp.launch(
            _begin_fixed_iteration,
            dim=self.num_worlds,
            inputs=[projection_status, proximal_failed],
            outputs=[self.world_active, self.world_failed, self.iteration_count],
            device=self.device,
        )
        wp.launch(
            _update_fixed_iteration_bodies,
            dim=self.num_bodies,
            inputs=[self.body_world, self.global_twist, self.projected_twist],
            outputs=[self.world_active, self.world_failed, self.splitting_dual],
            device=self.device,
        )

    def mark_iteration_limit(self) -> None:
        """Deactivate worlds that remain unconverged at the iteration limit."""
        wp.launch(
            _mark_iteration_limit,
            dim=self.num_worlds,
            inputs=[],
            outputs=[self.world_active, self.world_iteration_limit],
            device=self.device,
        )
