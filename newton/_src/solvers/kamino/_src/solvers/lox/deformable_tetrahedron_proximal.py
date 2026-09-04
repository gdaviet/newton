# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Element-local nonlinear tetrahedron proximal updates for LOX deformables."""

from __future__ import annotations

import warp as wp

from .deformable_energy import (
    mat99,
    tet_cofactor,
    tet_vertex_coefficient,
    vec9,
)

# Keep nested 9-by-9 eigensolver loops from expanding into multi-megabyte kernels.
wp.set_module_options({"enable_backward": False, "max_unroll": 4})

PARTICLE_FLAG_ACTIVE = 1
_LOCAL_SOLVE_REGULARIZATION = 1.0e-6


@wp.func
def _particle_is_dynamic(
    packed_particle: int,
    packed_to_newton: wp.array[wp.int32],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
) -> bool:
    particle = packed_to_newton[packed_particle]
    return particle_mass[particle] > 0.0 and (particle_flags[particle] & PARTICLE_FLAG_ACTIVE) != 0


@wp.func
def _matrix_dot(first: wp.mat33, second: wp.mat33) -> float:
    result = 0.0
    for row in range(3):
        for column in range(3):
            result += first[row, column] * second[row, column]
    return result


@wp.func
def _matrix_is_finite(value: wp.mat33) -> bool:
    finite = True
    for row in range(3):
        for column in range(3):
            finite = finite and wp.isfinite(value[row, column])
    return finite


@wp.func
def _matrix_to_vector(value: wp.mat33) -> vec9:
    return vec9(
        value[0, 0],
        value[1, 0],
        value[2, 0],
        value[0, 1],
        value[1, 1],
        value[2, 1],
        value[0, 2],
        value[1, 2],
        value[2, 2],
    )


@wp.func
def _vector_to_matrix(value: vec9) -> wp.mat33:
    return wp.mat33(
        value[0],
        value[3],
        value[6],
        value[1],
        value[4],
        value[7],
        value[2],
        value[5],
        value[8],
    )


@wp.func
def _deformation(
    position_0: wp.vec3,
    position_1: wp.vec3,
    position_2: wp.vec3,
    position_3: wp.vec3,
    rest_pose: wp.mat33,
) -> wp.mat33:
    return (
        wp.matrix_from_cols(
            position_1 - position_0,
            position_2 - position_0,
            position_3 - position_0,
        )
        * rest_pose
    )


@wp.func
def _apply_metric(
    value: wp.mat33,
    metric: mat99,
) -> wp.mat33:
    return _vector_to_matrix(metric * _matrix_to_vector(value))


@wp.func
def _tetrahedron_energy(
    deformation: wp.mat33,
    rest_volume: float,
    mu: float,
    lmbd: float,
    alpha: float,
    activation: float,
) -> float:
    volume_constraint = wp.determinant(deformation) - alpha + activation
    return rest_volume * (
        0.5 * mu * (_matrix_dot(deformation, deformation) - 3.0) + 0.5 * lmbd * volume_constraint * volume_constraint
    )


@wp.func
def _proximal_objective(
    deformation: wp.mat33,
    center: wp.mat33,
    multiplier: wp.mat33,
    frozen_metric: mat99,
    rest_volume: float,
    mu: float,
    lmbd: float,
    alpha: float,
    activation: float,
) -> float:
    delta = deformation - center
    metric_delta = _apply_metric(delta, frozen_metric)
    return (
        _tetrahedron_energy(deformation, rest_volume, mu, lmbd, alpha, activation)
        - _matrix_dot(multiplier, deformation)
        + 0.5 * _matrix_dot(delta, metric_delta)
    )


@wp.func
def _proximal_stationarity_norm_squared(
    deformation: wp.mat33,
    center: wp.mat33,
    multiplier: wp.mat33,
    frozen_metric: mat99,
    rest_volume: float,
    mu: float,
    lmbd: float,
    alpha: float,
    activation: float,
) -> float:
    cofactor = tet_cofactor(deformation)
    constraint = wp.determinant(deformation) - alpha + activation
    gradient = rest_volume * (mu * deformation + lmbd * constraint * cofactor)
    metric_delta = _apply_metric(deformation - center, frozen_metric)
    residual = gradient - multiplier + metric_delta
    return _matrix_dot(residual, residual)


@wp.func
def _factor_gauss_newton_base(
    frozen_metric: mat99,
    frozen_cofactor: wp.mat33,
    rest_volume: float,
    mu: float,
    k_lambda: float,
):
    base = frozen_metric + rest_volume * mu * wp.identity(n=9, dtype=float)
    frozen_cofactor_vector = _matrix_to_vector(frozen_cofactor)
    volume_scale = rest_volume * (k_lambda + mu)

    system_scale = float(0.0)
    system_is_finite = wp.bool(True)
    for row in range(9):
        for column in range(9):
            frozen_system_value = (
                base[row, column] + volume_scale * frozen_cofactor_vector[row] * frozen_cofactor_vector[column]
            )
            system_scale = wp.max(system_scale, wp.abs(frozen_system_value))
            system_is_finite = system_is_finite and wp.isfinite(frozen_system_value)
    if not system_is_finite or system_scale == 0.0:
        return mat99(0.0), wp.bool(False)

    regularization = _LOCAL_SOLVE_REGULARIZATION * system_scale
    lower = mat99(0.0)
    for row in range(9):
        for column in range(row + 1):
            value = 0.5 * (base[row, column] + base[column, row])
            if row == column:
                value += regularization
            for inner in range(column):
                value -= lower[row, inner] * lower[column, inner]
            if row == column:
                if not wp.isfinite(value) or value <= 0.0:
                    return mat99(0.0), wp.bool(False)
                lower[row, column] = wp.sqrt(value)
            else:
                lower[row, column] = value / lower[column, column]
    return lower, wp.bool(True)


@wp.func
def _solve_factored_base(lower: mat99, right_hand_side: vec9) -> vec9:
    intermediate = vec9(0.0)
    for row in range(9):
        value = right_hand_side[row]
        for column in range(row):
            value -= lower[row, column] * intermediate[column]
        intermediate[row] = value / lower[row, row]

    result = vec9(0.0)
    for offset in range(9):
        row = 8 - offset
        value = intermediate[row]
        for column in range(row + 1, 9):
            value -= lower[column, row] * result[column]
        result[row] = value / lower[row, row]
    return result


@wp.func
def _solve_gauss_newton_step(
    right_hand_side: wp.mat33,
    deformation: wp.mat33,
    frozen_factor: mat99,
    rest_volume: float,
    mu: float,
    k_lambda: float,
):
    right_hand_side_vector = _matrix_to_vector(right_hand_side)
    base_step = _solve_factored_base(frozen_factor, right_hand_side_vector)
    cofactor_vector = _matrix_to_vector(tet_cofactor(deformation))
    inverse_base_cofactor = _solve_factored_base(frozen_factor, cofactor_vector)
    volume_scale = rest_volume * (k_lambda + mu)
    denominator = 1.0 + volume_scale * wp.dot(cofactor_vector, inverse_base_cofactor)
    coefficient = volume_scale * wp.dot(cofactor_vector, base_step) / denominator
    step_vector = base_step - coefficient * inverse_base_cofactor

    step = _vector_to_matrix(step_vector)
    return step, wp.isfinite(denominator) and denominator > 0.0 and _matrix_is_finite(step)


@wp.kernel
def initialize_tetrahedron_proximal(
    position_linearized: wp.array[wp.vec3],
    tetrahedron_indices: wp.array2d[wp.int32],
    tetrahedron_poses: wp.array[wp.mat33],
    tetrahedron_materials: wp.array2d[float],
    tetrahedron_activations: wp.array[float],
    frozen_metric: wp.array[mat99],
    frozen_coordinate: wp.array[wp.mat33],
    frozen_gradient: wp.array[wp.mat33],
    frozen_factor: wp.array[mat99],
    proximal_coordinate: wp.array[wp.mat33],
    multiplier: wp.array[wp.mat33],
):
    """Initialize local coordinates and force multipliers at the frozen state."""
    tetrahedron = wp.tid()
    vertex_0 = tetrahedron_indices[tetrahedron, 0]
    deformation = _deformation(
        position_linearized[vertex_0],
        position_linearized[tetrahedron_indices[tetrahedron, 1]],
        position_linearized[tetrahedron_indices[tetrahedron, 2]],
        position_linearized[tetrahedron_indices[tetrahedron, 3]],
        tetrahedron_poses[tetrahedron],
    )
    cofactor = tet_cofactor(deformation)
    mu = tetrahedron_materials[tetrahedron, 0]
    k_lambda = tetrahedron_materials[tetrahedron, 1]
    lmbd = k_lambda + mu
    if mu == 0.0 and k_lambda == 0.0:
        frozen_coordinate[tetrahedron] = deformation
        frozen_gradient[tetrahedron] = wp.mat33(0.0)
        frozen_factor[tetrahedron] = mat99(0.0)
        proximal_coordinate[tetrahedron] = deformation
        multiplier[tetrahedron] = wp.mat33(0.0)
        return

    alpha = 1.0
    if lmbd > 1.0e-6:
        alpha = 1.0 + mu / lmbd
    constraint = wp.determinant(deformation) - alpha + tetrahedron_activations[tetrahedron]
    rest_volume = 1.0 / (6.0 * wp.determinant(tetrahedron_poses[tetrahedron]))
    gradient = rest_volume * (mu * deformation + lmbd * constraint * cofactor)

    frozen_coordinate[tetrahedron] = deformation
    frozen_gradient[tetrahedron] = gradient
    frozen_metric_value = frozen_metric[tetrahedron]
    frozen_factor_value, _factor_succeeded = _factor_gauss_newton_base(
        frozen_metric_value,
        cofactor,
        rest_volume,
        mu,
        k_lambda,
    )
    frozen_factor[tetrahedron] = frozen_factor_value
    proximal_coordinate[tetrahedron] = deformation
    multiplier[tetrahedron] = gradient


@wp.kernel
def update_tetrahedron_proximal(
    position_start: wp.array[wp.vec3],
    position_linearized: wp.array[wp.vec3],
    smooth_velocity: wp.array[wp.vec3],
    tetrahedron_indices: wp.array2d[wp.int32],
    tetrahedron_poses: wp.array[wp.mat33],
    tetrahedron_materials: wp.array2d[float],
    tetrahedron_activations: wp.array[float],
    packed_to_newton: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    frozen_coordinate: wp.array[wp.mat33],
    frozen_gradient: wp.array[wp.mat33],
    frozen_metric: wp.array[mat99],
    frozen_factor: wp.array[mat99],
    proximal_iterations: int,
    proximal_relaxation: float,
    time_step: wp.array[wp.float32],
    proximal_coordinate: wp.array[wp.mat33],
    multiplier: wp.array[wp.mat33],
    nonlinear_rhs: wp.array[wp.vec3],
    world_position_residual: wp.array[float],
    world_velocity_residual: wp.array[float],
    world_failed: wp.array[wp.int32],
):
    """Evaluate one tetrahedron prox and scatter its next global RHS correction."""
    tetrahedron = wp.tid()
    vertices = wp.vec4i(
        tetrahedron_indices[tetrahedron, 0],
        tetrahedron_indices[tetrahedron, 1],
        tetrahedron_indices[tetrahedron, 2],
        tetrahedron_indices[tetrahedron, 3],
    )
    world = packed_world[vertices[0]]
    dt = time_step[world]
    if world_active[world] == 0:
        return
    if tetrahedron_materials[tetrahedron, 0] == 0.0 and tetrahedron_materials[tetrahedron, 1] == 0.0:
        return

    position_0 = position_linearized[vertices[0]]
    position_1 = position_linearized[vertices[1]]
    position_2 = position_linearized[vertices[2]]
    position_3 = position_linearized[vertices[3]]
    if _particle_is_dynamic(vertices[0], packed_to_newton, particle_mass, particle_flags):
        position_0 = position_start[vertices[0]] + dt * smooth_velocity[vertices[0]]
    if _particle_is_dynamic(vertices[1], packed_to_newton, particle_mass, particle_flags):
        position_1 = position_start[vertices[1]] + dt * smooth_velocity[vertices[1]]
    if _particle_is_dynamic(vertices[2], packed_to_newton, particle_mass, particle_flags):
        position_2 = position_start[vertices[2]] + dt * smooth_velocity[vertices[2]]
    if _particle_is_dynamic(vertices[3], packed_to_newton, particle_mass, particle_flags):
        position_3 = position_start[vertices[3]] + dt * smooth_velocity[vertices[3]]
    rest_pose = tetrahedron_poses[tetrahedron]
    center = _deformation(
        position_0,
        position_1,
        position_2,
        position_3,
        rest_pose,
    )

    frozen_metric_value = frozen_metric[tetrahedron]
    frozen_factor_value = frozen_factor[tetrahedron]
    multiplier_value = multiplier[tetrahedron]
    previous = proximal_coordinate[tetrahedron]
    deformation = previous
    rest_volume = 1.0 / (6.0 * wp.determinant(rest_pose))
    mu = tetrahedron_materials[tetrahedron, 0]
    k_lambda = tetrahedron_materials[tetrahedron, 1]
    lmbd = k_lambda + mu
    alpha = 1.0
    if lmbd > 1.0e-6:
        alpha = 1.0 + mu / lmbd
    activation = tetrahedron_activations[tetrahedron]

    failed = wp.bool(False)
    for _iteration in range(proximal_iterations):
        current_cofactor = tet_cofactor(deformation)
        constraint = wp.determinant(deformation) - alpha + activation
        gradient = rest_volume * (mu * deformation + lmbd * constraint * current_cofactor)
        metric_delta = _apply_metric(deformation - center, frozen_metric_value)
        residual = gradient - multiplier_value + metric_delta
        step, solve_succeeded = _solve_gauss_newton_step(
            -residual,
            deformation,
            frozen_factor_value,
            rest_volume,
            mu,
            k_lambda,
        )
        if not solve_succeeded:
            failed = wp.bool(True)
            break

        objective = _proximal_objective(
            deformation,
            center,
            multiplier_value,
            frozen_metric_value,
            rest_volume,
            mu,
            lmbd,
            alpha,
            activation,
        )
        stationarity_norm_squared = _matrix_dot(residual, residual)
        accepted = wp.bool(False)
        step_scale = wp.float32(1.0)
        for _trial in range(4):
            candidate = deformation + step_scale * step
            candidate_objective = _proximal_objective(
                candidate,
                center,
                multiplier_value,
                frozen_metric_value,
                rest_volume,
                mu,
                lmbd,
                alpha,
                activation,
            )
            candidate_accepted = wp.isfinite(candidate_objective) and candidate_objective <= objective
            if wp.isfinite(candidate_objective) and not candidate_accepted:
                candidate_stationarity_norm_squared = _proximal_stationarity_norm_squared(
                    candidate,
                    center,
                    multiplier_value,
                    frozen_metric_value,
                    rest_volume,
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
                deformation = candidate
                accepted = wp.bool(True)
                break
            step_scale *= 0.5
        if not accepted:
            break

    if failed or not _matrix_is_finite(deformation):
        wp.atomic_max(world_failed, world, 1)
        return

    metric_primal = _apply_metric(center - deformation, frozen_metric_value)
    multiplier_new = multiplier_value + proximal_relaxation * metric_primal
    if not _matrix_is_finite(multiplier_new):
        wp.atomic_max(world_failed, world, 1)
        return

    metric_correction = _apply_metric(deformation - frozen_coordinate[tetrahedron], frozen_metric_value)
    correction = metric_correction - multiplier_new + frozen_gradient[tetrahedron]
    if not _matrix_is_finite(correction):
        wp.atomic_max(world_failed, world, 1)
        return

    for order in range(4):
        vertex = vertices[order]
        if _particle_is_dynamic(vertex, packed_to_newton, particle_mass, particle_flags):
            wp.atomic_add(
                nonlinear_rhs,
                vertex,
                dt * (correction * tet_vertex_coefficient(order, rest_pose)),
            )

    proximal_coordinate[tetrahedron] = deformation
    multiplier[tetrahedron] = multiplier_new

    characteristic_length = wp.pow(rest_volume, 1.0 / 3.0)
    primal_norm = wp.sqrt(_matrix_dot(center - deformation, center - deformation))
    local_change_norm = wp.sqrt(_matrix_dot(deformation - previous, deformation - previous))
    wp.atomic_max(world_position_residual, world, characteristic_length * primal_norm)
    wp.atomic_max(world_velocity_residual, world, characteristic_length * local_change_norm / dt)


class DeformableTetrahedronProximal:
    """Own shared fixed-majorizer tetrahedron proximal state."""

    def __init__(self, deformable_system, iterations: int, relaxation: float):
        self.deformable_system = deformable_system
        self.device = deformable_system.device
        self.iterations = iterations
        self.relaxation = relaxation
        shape = deformable_system.tetrahedron_count
        self.frozen_coordinate = wp.zeros(shape, dtype=wp.mat33, device=self.device)
        self.frozen_gradient = wp.zeros(shape, dtype=wp.mat33, device=self.device)
        self.frozen_metric = deformable_system.tetrahedron_metric
        self.frozen_factor = wp.zeros(shape, dtype=mat99, device=self.device)
        self.proximal_coordinate = wp.zeros(shape, dtype=wp.mat33, device=self.device)
        self.multiplier = wp.zeros(shape, dtype=wp.mat33, device=self.device)

    def initialize(self) -> None:
        """Initialize the local ADMM state from the frozen linearization."""
        system = self.deformable_system
        wp.launch(
            initialize_tetrahedron_proximal,
            dim=system.tetrahedron_count,
            inputs=[
                system.position_linearized,
                system.topology.tetrahedron_indices,
                system.model.tet_poses,
                system.model.tet_materials,
                system.model.tet_activations,
                self.frozen_metric,
            ],
            outputs=[
                self.frozen_coordinate,
                self.frozen_gradient,
                self.frozen_factor,
                self.proximal_coordinate,
                self.multiplier,
            ],
            device=self.device,
        )

    def update(self, time_step: wp.array[wp.float32]) -> None:
        """Evaluate all active tetrahedron proxes and add their RHS corrections."""
        system = self.deformable_system
        wp.launch(
            update_tetrahedron_proximal,
            dim=system.tetrahedron_count,
            inputs=[
                system.position_start,
                system.position_linearized,
                system.smooth_velocity,
                system.topology.tetrahedron_indices,
                system.model.tet_poses,
                system.model.tet_materials,
                system.model.tet_activations,
                system.topology.packed_to_newton,
                system.topology.packed_solve_world,
                system.model.particle_mass,
                system.model.particle_flags,
                system.world_active,
                self.frozen_coordinate,
                self.frozen_gradient,
                self.frozen_metric,
                self.frozen_factor,
                self.iterations,
                self.relaxation,
                time_step,
            ],
            outputs=[
                self.proximal_coordinate,
                self.multiplier,
                system.nonlinear_rhs,
                system.proximal_position_residual,
                system.proximal_velocity_residual,
                system.proximal_failed,
            ],
            device=self.device,
        )
