# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Element-local nonlinear membrane proximal updates of the LOX deformables.

Each membrane triangle keeps a local copy of its two deformation columns and a force
multiplier. Every splitting iteration solves the local proximal problem of each triangle,
centered at the deformation of the candidate velocity with the fixed metric of the
linearization, by a few damped Gauss-Newton steps, then updates its multiplier and
scatters the difference to the linearized membrane forces into the right-hand side of
the next candidate solve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .membrane import membrane_area_ratio_gradient
from .system_kernels import is_dynamic_particle, triangle_vertex_coefficients

if TYPE_CHECKING:
    from .......sim import Model
    from .types import DeformableMembraneProximalData, DeformableSystemData, DeformableTopology

###
# Module interface
###

__all__ = ["DeformableMembraneProximal"]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


_PROXIMAL_EPSILON = 1.0e-8
"""Relative determinant floor of the local Gauss-Newton system."""


_LOCAL_SOLVE_REGULARIZATION = 1.0e-6
"""Relative shear regularization of the local Gauss-Newton system."""


###
# Functions
###


@wp.func
def _pair_dot(
    first_0: wp.vec3,
    first_1: wp.vec3,
    second_0: wp.vec3,
    second_1: wp.vec3,
) -> float:
    return wp.dot(first_0, second_0) + wp.dot(first_1, second_1)


@wp.func
def _vector_is_finite(value: wp.vec3) -> bool:
    finite = True
    for axis in range(3):
        finite = finite and wp.isfinite(value[axis])
    return finite


@wp.func
def _pair_is_finite(first: wp.vec3, second: wp.vec3) -> bool:
    return _vector_is_finite(first) and _vector_is_finite(second)


@wp.func
def _deformation(
    position_0: wp.vec3,
    position_1: wp.vec3,
    position_2: wp.vec3,
    rest_pose: wp.mat22,
):
    """Return the two deformation columns of a triangle."""
    edge_10 = position_1 - position_0
    edge_20 = position_2 - position_0
    deformation_0 = edge_10 * rest_pose[0, 0] + edge_20 * rest_pose[1, 0]
    deformation_1 = edge_10 * rest_pose[0, 1] + edge_20 * rest_pose[1, 1]
    return deformation_0, deformation_1


@wp.func
def _apply_metric(
    value_0: wp.vec3,
    value_1: wp.vec3,
    area_gradient_0: wp.vec3,
    area_gradient_1: wp.vec3,
    shear_scale: float,
    area_scale: float,
):
    """Apply the fixed membrane metric to a pair of deformation columns."""
    area_component = _pair_dot(area_gradient_0, area_gradient_1, value_0, value_1)
    result_0 = shear_scale * value_0 + area_scale * area_component * area_gradient_0
    result_1 = shear_scale * value_1 + area_scale * area_component * area_gradient_1
    return result_0, result_1


@wp.func
def _membrane_energy(
    deformation_0: wp.vec3,
    deformation_1: wp.vec3,
    rest_area: float,
    mu: float,
    lmbd: float,
    alpha: float,
    activation: float,
) -> float:
    area_ratio, _, _ = membrane_area_ratio_gradient(deformation_0, deformation_1)
    area_constraint = area_ratio - alpha + activation
    return rest_area * (
        0.5 * mu * (wp.dot(deformation_0, deformation_0) + wp.dot(deformation_1, deformation_1) - 2.0)
        + 0.5 * lmbd * area_constraint * area_constraint
    )


@wp.func
def _proximal_objective(
    deformation_0: wp.vec3,
    deformation_1: wp.vec3,
    center_0: wp.vec3,
    center_1: wp.vec3,
    multiplier_0: wp.vec3,
    multiplier_1: wp.vec3,
    frozen_area_gradient_0: wp.vec3,
    frozen_area_gradient_1: wp.vec3,
    rest_area: float,
    mu: float,
    lmbd: float,
    alpha: float,
    activation: float,
) -> float:
    delta_0 = deformation_0 - center_0
    delta_1 = deformation_1 - center_1
    metric_delta_0, metric_delta_1 = _apply_metric(
        delta_0,
        delta_1,
        frozen_area_gradient_0,
        frozen_area_gradient_1,
        rest_area * mu,
        rest_area * lmbd,
    )
    return (
        _membrane_energy(
            deformation_0,
            deformation_1,
            rest_area,
            mu,
            lmbd,
            alpha,
            activation,
        )
        - _pair_dot(multiplier_0, multiplier_1, deformation_0, deformation_1)
        + 0.5 * _pair_dot(delta_0, delta_1, metric_delta_0, metric_delta_1)
    )


@wp.func
def _proximal_stationarity_norm_squared(
    deformation_0: wp.vec3,
    deformation_1: wp.vec3,
    center_0: wp.vec3,
    center_1: wp.vec3,
    multiplier_0: wp.vec3,
    multiplier_1: wp.vec3,
    frozen_area_gradient_0: wp.vec3,
    frozen_area_gradient_1: wp.vec3,
    rest_area: float,
    mu: float,
    lmbd: float,
    alpha: float,
    activation: float,
) -> float:
    area_ratio, area_gradient_0, area_gradient_1 = membrane_area_ratio_gradient(
        deformation_0,
        deformation_1,
    )
    constraint = area_ratio - alpha + activation
    gradient_0 = rest_area * (mu * deformation_0 + lmbd * constraint * area_gradient_0)
    gradient_1 = rest_area * (mu * deformation_1 + lmbd * constraint * area_gradient_1)
    metric_delta_0, metric_delta_1 = _apply_metric(
        deformation_0 - center_0,
        deformation_1 - center_1,
        frozen_area_gradient_0,
        frozen_area_gradient_1,
        rest_area * mu,
        rest_area * lmbd,
    )
    residual_0 = gradient_0 - multiplier_0 + metric_delta_0
    residual_1 = gradient_1 - multiplier_1 + metric_delta_1
    return _pair_dot(residual_0, residual_1, residual_0, residual_1)


@wp.func
def _solve_gauss_newton_step(
    right_hand_side_0: wp.vec3,
    right_hand_side_1: wp.vec3,
    current_area_gradient_0: wp.vec3,
    current_area_gradient_1: wp.vec3,
    frozen_area_gradient_0: wp.vec3,
    frozen_area_gradient_1: wp.vec3,
    shear_scale: float,
    area_scale: float,
):
    """Solve the local Gauss-Newton system, a shear identity plus two rank-one area terms."""
    current_norm = _pair_dot(
        current_area_gradient_0,
        current_area_gradient_1,
        current_area_gradient_0,
        current_area_gradient_1,
    )
    frozen_norm = _pair_dot(
        frozen_area_gradient_0,
        frozen_area_gradient_1,
        frozen_area_gradient_0,
        frozen_area_gradient_1,
    )
    cross = _pair_dot(
        current_area_gradient_0,
        current_area_gradient_1,
        frozen_area_gradient_0,
        frozen_area_gradient_1,
    )
    metric_scale = shear_scale + area_scale * (current_norm + frozen_norm)
    regularized_shear_scale = shear_scale + _LOCAL_SOLVE_REGULARIZATION * wp.max(metric_scale, 1.0e-6)
    system_00 = regularized_shear_scale + area_scale * current_norm
    system_01 = area_scale * cross
    system_11 = regularized_shear_scale + area_scale * frozen_norm
    projected_0 = _pair_dot(
        current_area_gradient_0,
        current_area_gradient_1,
        right_hand_side_0,
        right_hand_side_1,
    )
    projected_1 = _pair_dot(
        frozen_area_gradient_0,
        frozen_area_gradient_1,
        right_hand_side_0,
        right_hand_side_1,
    )
    determinant = system_00 * system_11 - system_01 * system_01
    coefficient_0 = (projected_0 * system_11 - system_01 * projected_1) / determinant
    coefficient_1 = (system_00 * projected_1 - system_01 * projected_0) / determinant
    delta_0 = (
        right_hand_side_0
        - area_scale * (coefficient_0 * current_area_gradient_0 + coefficient_1 * frozen_area_gradient_0)
    ) / regularized_shear_scale
    delta_1 = (
        right_hand_side_1
        - area_scale * (coefficient_0 * current_area_gradient_1 + coefficient_1 * frozen_area_gradient_1)
    ) / regularized_shear_scale
    return delta_0, delta_1, determinant, system_00 * system_11


###
# Kernels
###


@wp.kernel
def _initialize_membrane_proximal(
    # Inputs:
    position_linearized: wp.array[wp.vec3],
    triangle_indices: wp.array2d[wp.int32],
    model_tri_poses: wp.array[wp.mat22],
    model_tri_areas: wp.array[float],
    model_tri_materials: wp.array2d[float],
    model_tri_activations: wp.array[float],
    # Outputs:
    frozen_coordinate: wp.array2d[wp.vec3],
    frozen_gradient: wp.array2d[wp.vec3],
    frozen_area_gradient: wp.array2d[wp.vec3],
    proximal_coordinate: wp.array2d[wp.vec3],
    multiplier: wp.array2d[wp.vec3],
):
    """Initialize local coordinates and force multipliers at the frozen state."""
    triangle = wp.tid()
    vertex_0 = triangle_indices[triangle, 0]
    vertex_1 = triangle_indices[triangle, 1]
    vertex_2 = triangle_indices[triangle, 2]
    deformation_0, deformation_1 = _deformation(
        position_linearized[vertex_0],
        position_linearized[vertex_1],
        position_linearized[vertex_2],
        model_tri_poses[triangle],
    )
    area_ratio, area_gradient_0, area_gradient_1 = membrane_area_ratio_gradient(
        deformation_0,
        deformation_1,
    )
    mu = model_tri_materials[triangle, 0]
    lmbd = model_tri_materials[triangle, 1] + mu
    alpha = 1.0
    if lmbd > 1.0e-6:
        alpha = 1.0 + mu / lmbd
    constraint = area_ratio - alpha + model_tri_activations[triangle]
    rest_area = model_tri_areas[triangle]
    gradient_0 = rest_area * (mu * deformation_0 + lmbd * constraint * area_gradient_0)
    gradient_1 = rest_area * (mu * deformation_1 + lmbd * constraint * area_gradient_1)

    frozen_coordinate[triangle, 0] = deformation_0
    frozen_coordinate[triangle, 1] = deformation_1
    frozen_gradient[triangle, 0] = gradient_0
    frozen_gradient[triangle, 1] = gradient_1
    frozen_area_gradient[triangle, 0] = area_gradient_0
    frozen_area_gradient[triangle, 1] = area_gradient_1
    proximal_coordinate[triangle, 0] = deformation_0
    proximal_coordinate[triangle, 1] = deformation_1
    multiplier[triangle, 0] = gradient_0
    multiplier[triangle, 1] = gradient_1


@wp.kernel
def _update_membrane_proximal(
    # Inputs:
    position_start: wp.array[wp.vec3],
    position_linearized: wp.array[wp.vec3],
    smooth_velocity: wp.array[wp.vec3],
    triangle_indices: wp.array2d[wp.int32],
    model_tri_poses: wp.array[wp.mat22],
    model_tri_areas: wp.array[float],
    model_tri_materials: wp.array2d[float],
    model_tri_activations: wp.array[float],
    packed_to_newton: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    model_particle_mass: wp.array[float],
    model_particle_flags: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    frozen_coordinate: wp.array2d[wp.vec3],
    frozen_gradient: wp.array2d[wp.vec3],
    frozen_area_gradient: wp.array2d[wp.vec3],
    proximal_iterations: int,
    proximal_relaxation: float,
    model_time_dt: wp.array[wp.float32],
    # Outputs:
    proximal_coordinate: wp.array2d[wp.vec3],
    multiplier: wp.array2d[wp.vec3],
    nonlinear_rhs: wp.array[wp.vec3],
    world_position_residual: wp.array[float],
    world_velocity_residual: wp.array[float],
    world_failed: wp.array[wp.bool],
):
    """Evaluate one membrane prox and scatter its next global RHS correction."""
    triangle = wp.tid()
    vertex_0 = triangle_indices[triangle, 0]
    vertex_1 = triangle_indices[triangle, 1]
    vertex_2 = triangle_indices[triangle, 2]
    wid = packed_world[vertex_0]
    dt = model_time_dt[wid]
    if not world_active[wid]:
        return

    mu = model_tri_materials[triangle, 0]
    lmbd = model_tri_materials[triangle, 1] + mu
    rest_area = model_tri_areas[triangle]
    shear_scale = 2.0 * rest_area * mu
    area_scale = rest_area * lmbd
    if mu == 0.0 and lmbd == 0.0:
        return

    position_0 = position_linearized[vertex_0]
    position_1 = position_linearized[vertex_1]
    position_2 = position_linearized[vertex_2]
    if is_dynamic_particle(vertex_0, packed_to_newton, model_particle_mass, model_particle_flags):
        position_0 = position_start[vertex_0] + dt * smooth_velocity[vertex_0]
    if is_dynamic_particle(vertex_1, packed_to_newton, model_particle_mass, model_particle_flags):
        position_1 = position_start[vertex_1] + dt * smooth_velocity[vertex_1]
    if is_dynamic_particle(vertex_2, packed_to_newton, model_particle_mass, model_particle_flags):
        position_2 = position_start[vertex_2] + dt * smooth_velocity[vertex_2]
    center_0, center_1 = _deformation(
        position_0,
        position_1,
        position_2,
        model_tri_poses[triangle],
    )

    frozen_area_gradient_0 = frozen_area_gradient[triangle, 0]
    frozen_area_gradient_1 = frozen_area_gradient[triangle, 1]
    multiplier_0 = multiplier[triangle, 0]
    multiplier_1 = multiplier[triangle, 1]
    previous_0 = proximal_coordinate[triangle, 0]
    previous_1 = proximal_coordinate[triangle, 1]
    deformation_0 = previous_0
    deformation_1 = previous_1
    alpha = 1.0
    if lmbd > 1.0e-6:
        alpha = 1.0 + mu / lmbd
    activation = model_tri_activations[triangle]

    failed = wp.bool(False)
    for _iteration in range(proximal_iterations):
        area_ratio, current_area_gradient_0, current_area_gradient_1 = membrane_area_ratio_gradient(
            deformation_0, deformation_1
        )
        constraint = area_ratio - alpha + activation
        gradient_0 = rest_area * (mu * deformation_0 + lmbd * constraint * current_area_gradient_0)
        gradient_1 = rest_area * (mu * deformation_1 + lmbd * constraint * current_area_gradient_1)
        metric_delta_0, metric_delta_1 = _apply_metric(
            deformation_0 - center_0,
            deformation_1 - center_1,
            frozen_area_gradient_0,
            frozen_area_gradient_1,
            rest_area * mu,
            area_scale,
        )
        residual_0 = gradient_0 - multiplier_0 + metric_delta_0
        residual_1 = gradient_1 - multiplier_1 + metric_delta_1
        step_0, step_1, determinant, determinant_scale = _solve_gauss_newton_step(
            -residual_0,
            -residual_1,
            current_area_gradient_0,
            current_area_gradient_1,
            frozen_area_gradient_0,
            frozen_area_gradient_1,
            shear_scale,
            area_scale,
        )
        if (
            not wp.isfinite(determinant)
            or determinant <= _PROXIMAL_EPSILON * determinant_scale
            or not _pair_is_finite(step_0, step_1)
        ):
            failed = wp.bool(True)
            break

        objective = _proximal_objective(
            deformation_0,
            deformation_1,
            center_0,
            center_1,
            multiplier_0,
            multiplier_1,
            frozen_area_gradient_0,
            frozen_area_gradient_1,
            rest_area,
            mu,
            lmbd,
            alpha,
            activation,
        )
        stationarity_norm_squared = _pair_dot(
            residual_0,
            residual_1,
            residual_0,
            residual_1,
        )
        accepted = wp.bool(False)
        step_scale = wp.float32(1.0)
        for _trial in range(4):
            candidate_0 = deformation_0 + step_scale * step_0
            candidate_1 = deformation_1 + step_scale * step_1
            candidate_objective = _proximal_objective(
                candidate_0,
                candidate_1,
                center_0,
                center_1,
                multiplier_0,
                multiplier_1,
                frozen_area_gradient_0,
                frozen_area_gradient_1,
                rest_area,
                mu,
                lmbd,
                alpha,
                activation,
            )
            candidate_accepted = wp.isfinite(candidate_objective) and candidate_objective <= objective
            if wp.isfinite(candidate_objective) and not candidate_accepted:
                # Float32 objective differences can round away near the prox;
                # retain steps that still make progress toward its KKT equation.
                candidate_stationarity_norm_squared = _proximal_stationarity_norm_squared(
                    candidate_0,
                    candidate_1,
                    center_0,
                    center_1,
                    multiplier_0,
                    multiplier_1,
                    frozen_area_gradient_0,
                    frozen_area_gradient_1,
                    rest_area,
                    mu,
                    lmbd,
                    alpha,
                    activation,
                )
                candidate_accepted = (
                    wp.isfinite(candidate_stationarity_norm_squared)
                    and candidate_stationarity_norm_squared < stationarity_norm_squared
                )
            if candidate_accepted:
                deformation_0 = candidate_0
                deformation_1 = candidate_1
                accepted = wp.bool(True)
                break
            step_scale *= 0.5
        if not accepted:
            break

    if failed or not _pair_is_finite(deformation_0, deformation_1):
        world_failed[wid] = True
        return

    area_ratio, current_area_gradient_0, current_area_gradient_1 = membrane_area_ratio_gradient(
        deformation_0,
        deformation_1,
    )
    constraint = area_ratio - alpha + activation
    gradient_0 = rest_area * (mu * deformation_0 + lmbd * constraint * current_area_gradient_0)
    gradient_1 = rest_area * (mu * deformation_1 + lmbd * constraint * current_area_gradient_1)
    metric_delta_0, metric_delta_1 = _apply_metric(
        deformation_0 - center_0,
        deformation_1 - center_1,
        frozen_area_gradient_0,
        frozen_area_gradient_1,
        rest_area * mu,
        area_scale,
    )
    residual_0 = gradient_0 - multiplier_0 + metric_delta_0
    residual_1 = gradient_1 - multiplier_1 + metric_delta_1
    stationarity_correction_0, stationarity_correction_1, determinant, determinant_scale = _solve_gauss_newton_step(
        -residual_0,
        -residual_1,
        current_area_gradient_0,
        current_area_gradient_1,
        frozen_area_gradient_0,
        frozen_area_gradient_1,
        shear_scale,
        area_scale,
    )
    if (
        not wp.isfinite(determinant)
        or not wp.isfinite(determinant_scale)
        or determinant <= _PROXIMAL_EPSILON * determinant_scale
        or not _pair_is_finite(stationarity_correction_0, stationarity_correction_1)
    ):
        world_failed[wid] = True
        return
    stationarity_correction_norm_0 = wp.length(stationarity_correction_0)
    stationarity_correction_norm_1 = wp.length(stationarity_correction_1)
    if not wp.isfinite(stationarity_correction_norm_0) or not wp.isfinite(stationarity_correction_norm_1):
        world_failed[wid] = True
        return
    stationarity_correction_norm = wp.max(
        stationarity_correction_norm_0,
        stationarity_correction_norm_1,
    )

    metric_primal_0, metric_primal_1 = _apply_metric(
        center_0 - deformation_0,
        center_1 - deformation_1,
        frozen_area_gradient_0,
        frozen_area_gradient_1,
        rest_area * mu,
        area_scale,
    )
    multiplier_new_0 = multiplier_0 + proximal_relaxation * metric_primal_0
    multiplier_new_1 = multiplier_1 + proximal_relaxation * metric_primal_1
    if not _pair_is_finite(multiplier_new_0, multiplier_new_1):
        world_failed[wid] = True
        return

    frozen_0 = frozen_coordinate[triangle, 0]
    frozen_1 = frozen_coordinate[triangle, 1]
    metric_correction_0, metric_correction_1 = _apply_metric(
        deformation_0 - frozen_0,
        deformation_1 - frozen_1,
        frozen_area_gradient_0,
        frozen_area_gradient_1,
        rest_area * mu,
        area_scale,
    )
    correction_0 = metric_correction_0 - multiplier_new_0 + frozen_gradient[triangle, 0]
    correction_1 = metric_correction_1 - multiplier_new_1 + frozen_gradient[triangle, 1]
    if not _pair_is_finite(correction_0, correction_1):
        world_failed[wid] = True
        return

    characteristic_length = wp.sqrt(rest_area)
    primal_norm_0 = wp.length(center_0 - deformation_0)
    primal_norm_1 = wp.length(center_1 - deformation_1)
    local_change_norm_0 = wp.length(deformation_0 - previous_0)
    local_change_norm_1 = wp.length(deformation_1 - previous_1)
    if (
        not wp.isfinite(characteristic_length)
        or not wp.isfinite(primal_norm_0)
        or not wp.isfinite(primal_norm_1)
        or not wp.isfinite(local_change_norm_0)
        or not wp.isfinite(local_change_norm_1)
    ):
        world_failed[wid] = True
        return
    primal_norm = wp.max(primal_norm_0, primal_norm_1)
    local_change_norm = wp.max(local_change_norm_0, local_change_norm_1)
    position_residual_value = characteristic_length * wp.max(primal_norm, stationarity_correction_norm)
    velocity_residual_value = characteristic_length * local_change_norm / dt
    if not wp.isfinite(position_residual_value) or not wp.isfinite(velocity_residual_value):
        world_failed[wid] = True
        return

    rest_pose = model_tri_poses[triangle]
    scatter_finite = wp.bool(True)
    for order in range(3):
        vertex = triangle_indices[triangle, order]
        if is_dynamic_particle(vertex, packed_to_newton, model_particle_mass, model_particle_flags):
            coefficients = triangle_vertex_coefficients(order, rest_pose)
            contribution = dt * (coefficients[0] * correction_0 + coefficients[1] * correction_1)
            scatter_finite = scatter_finite and _vector_is_finite(contribution)
    if not scatter_finite:
        world_failed[wid] = True
        return

    for order in range(3):
        vertex = triangle_indices[triangle, order]
        if is_dynamic_particle(vertex, packed_to_newton, model_particle_mass, model_particle_flags):
            coefficients = triangle_vertex_coefficients(order, rest_pose)
            wp.atomic_add(
                nonlinear_rhs,
                vertex,
                dt * (coefficients[0] * correction_0 + coefficients[1] * correction_1),
            )

    proximal_coordinate[triangle, 0] = deformation_0
    proximal_coordinate[triangle, 1] = deformation_1
    multiplier[triangle, 0] = multiplier_new_0
    multiplier[triangle, 1] = multiplier_new_1

    wp.atomic_max(
        world_position_residual,
        wid,
        position_residual_value,
    )
    wp.atomic_max(
        world_velocity_residual,
        wid,
        velocity_residual_value,
    )


###
# Interfaces
###


class DeformableMembraneProximal:
    """Fixed-metric membrane proximal updates of a deformable system.

    The proximal state lives in :class:`DeformableMembraneProximalData`, exposed as
    :attr:`data`.

    Args:
        model: Newton model of the deformables.
        topology: Packed topology of the deformables.
        system_data: Arrays of the deformable system, which receive the right-hand-side
            corrections, residuals, and failures of the proximal updates.
        data: Proximal state of the membrane triangles.
        iterations: Fixed number of local Gauss-Newton iterations per update.
        relaxation: Relaxation factor in ``[0, 1]`` of the local multiplier update.
    """

    def __init__(
        self,
        model: Model,
        topology: DeformableTopology,
        system_data: DeformableSystemData,
        data: DeformableMembraneProximalData,
        iterations: int,
        relaxation: float,
    ):
        self.model = model
        self.device = model.device
        self.topology = topology
        """Packed topology of the deformables."""
        self.system_data = system_data
        """Arrays of the deformable system."""
        self.triangle_count = int(model.tri_count)
        self.iterations = iterations
        self.relaxation = relaxation
        self._data = data

    ###
    # Properties
    ###

    @property
    def data(self) -> DeformableMembraneProximalData:
        """Returns the proximal state of the membrane triangles."""
        return self._data

    ###
    # Public API
    ###

    def initialize(self) -> None:
        """Initialize the local ADMM state from the frozen linearization."""
        system = self.system_data
        model = self.model
        data = self._data
        wp.launch(
            _initialize_membrane_proximal,
            dim=self.triangle_count,
            inputs=[
                system.position_linearized,
                self.topology.triangle_indices,
                model.tri_poses,
                model.tri_areas,
                model.tri_materials,
                model.tri_activations,
            ],
            outputs=[
                data.frozen_coordinate,
                data.frozen_gradient,
                data.frozen_area_gradient,
                data.proximal_coordinate,
                data.multiplier,
            ],
            device=self.device,
        )

    def update(self, time_step: wp.array[wp.float32]) -> None:
        """Evaluate all active membrane proxes and add their RHS corrections.

        Args:
            time_step: Time step of each world [s].
        """
        system = self.system_data
        model = self.model
        data = self._data
        wp.launch(
            _update_membrane_proximal,
            dim=self.triangle_count,
            inputs=[
                system.position_start,
                system.position_linearized,
                system.smooth_velocity,
                self.topology.triangle_indices,
                model.tri_poses,
                model.tri_areas,
                model.tri_materials,
                model.tri_activations,
                self.topology.packed_to_newton,
                self.topology.packed_solve_world,
                model.particle_mass,
                model.particle_flags,
                system.world_active,
                data.frozen_coordinate,
                data.frozen_gradient,
                data.frozen_area_gradient,
                self.iterations,
                self.relaxation,
                time_step,
            ],
            outputs=[
                data.proximal_coordinate,
                data.multiplier,
                system.nonlinear_rhs,
                system.proximal_position_residual,
                system.proximal_velocity_residual,
                system.proximal_failed,
            ],
            device=self.device,
        )
