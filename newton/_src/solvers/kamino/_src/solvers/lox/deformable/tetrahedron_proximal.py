# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Element-local nonlinear tetrahedron proximal updates of the LOX deformables.

Each tetrahedron keeps a local copy of its deformation gradient and a force multiplier.
Every splitting iteration solves the local proximal problem of each tetrahedron, centered
at the deformation of the candidate velocity with the majorizing metric of the
linearization, by a few damped Gauss-Newton steps, then updates its multiplier and
scatters the difference to the linearized forces into the right-hand side of the next
candidate solve. The refinement of a tetrahedron requires its frozen penalty to bound the
local curvature, ``-B <= H(z) <= B``, which keeps the local updates contractive; a tetrahedron
whose local update fails or leaves that region keeps its linearized response for that
iteration, and a non-finite candidate deformation fails its world.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .energy import mat99f, tet_cofactor, tet_stable_neo_hookean_alpha, tet_vertex_coefficient, vec9f
from .system_kernels import is_dynamic_particle
from .tetrahedron_energy import tet_stable_neo_hookean_hessian

if TYPE_CHECKING:
    from .......sim import Model
    from .types import DeformableSystemData, DeformableTetrahedronProximalData, DeformableTopology

###
# Module interface
###

__all__ = ["DeformableTetrahedronProximal"]

###
# Module configs
###

# Keep nested 9-by-9 eigensolver loops from expanding into multi-megabyte kernels.
wp.set_module_options({"enable_backward": False, "max_unroll": 4})


###
# Constants
###


_LOCAL_SOLVE_REGULARIZATION = 1.0e-6
"""Relative diagonal regularization of the frozen local Gauss-Newton base matrix."""


###
# Functions
###


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
def _vector_is_finite(value: wp.vec3) -> bool:
    return wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])


@wp.func
def _matrix_to_vector(value: wp.mat33) -> vec9f:
    return vec9f(
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
def _vector_to_matrix(value: vec9f) -> wp.mat33:
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
    """Return the deformation gradient of a tetrahedron."""
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
    metric: mat99f,
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
    # The stable Neo-Hookean polynomial stays evaluable at singular and inverted
    # deformations, which the local problem admits without an orientation barrier.
    volume_constraint = wp.determinant(deformation) - alpha + activation
    return rest_volume * (
        0.5 * mu * (_matrix_dot(deformation, deformation) - 3.0) + 0.5 * lmbd * volume_constraint * volume_constraint
    )


@wp.func
def _proximal_objective(
    deformation: wp.mat33,
    center: wp.mat33,
    multiplier: wp.mat33,
    frozen_metric: mat99f,
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
    frozen_metric: mat99f,
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
    frozen_metric: mat99f,
    frozen_cofactor: wp.mat33,
    rest_volume: float,
    mu: float,
    k_lambda: float,
):
    """Return the regularized Cholesky factor of the frozen metric plus the shear identity."""
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
        return mat99f(0.0), wp.bool(False)

    regularization = _LOCAL_SOLVE_REGULARIZATION * system_scale
    lower = mat99f(0.0)
    for row in range(9):
        for column in range(row + 1):
            value = 0.5 * (base[row, column] + base[column, row])
            if row == column:
                value += regularization
            for inner in range(column):
                value -= lower[row, inner] * lower[column, inner]
            if row == column:
                if not wp.isfinite(value) or value <= 0.0:
                    return mat99f(0.0), wp.bool(False)
                lower[row, column] = wp.sqrt(value)
            else:
                lower[row, column] = value / lower[column, column]
    return lower, wp.bool(True)


@wp.func
def _solve_factored_base(lower: mat99f, right_hand_side: vec9f) -> vec9f:
    intermediate = vec9f(0.0)
    for row in range(9):
        value = right_hand_side[row]
        for column in range(row):
            value -= lower[row, column] * intermediate[column]
        intermediate[row] = value / lower[row, row]

    result = vec9f(0.0)
    for offset in range(9):
        row = 8 - offset
        value = intermediate[row]
        for column in range(row + 1, 9):
            value -= lower[column, row] * result[column]
        result[row] = value / lower[row, row]
    return result


@wp.func
def _is_positive_definite(matrix: mat99f) -> bool:
    """Return whether the Cholesky factorization of a symmetric 9-by-9 matrix succeeds."""
    lower = mat99f(0.0)
    for row in range(9):
        for column in range(row + 1):
            value = matrix[row, column]
            for inner in range(column):
                value -= lower[row, inner] * lower[column, inner]
            if row == column:
                if not (value > 0.0):
                    return False
                lower[row, row] = wp.sqrt(value)
            else:
                lower[row, column] = value / lower[column, column]
    return True


@wp.func
def _penalty_covers_curvature(
    deformation: wp.mat33,
    frozen_metric: mat99f,
    rest_volume: float,
    mu: float,
    k_lambda: float,
    activation: float,
) -> bool:
    """Return whether the frozen base ``B``, the metric plus the shear identity, bounds the local Hessian.

    The local updates contract while ``-B <= H(z) <= B``, which the two Cholesky factorizations decide.
    """
    hessian = tet_stable_neo_hookean_hessian(deformation, rest_volume, mu, k_lambda, activation)
    base = frozen_metric + rest_volume * mu * wp.identity(n=9, dtype=float)
    return _is_positive_definite(base - hessian) and _is_positive_definite(base + hessian)


@wp.func
def _solve_gauss_newton_step(
    right_hand_side: wp.mat33,
    deformation: wp.mat33,
    frozen_factor: mat99f,
    rest_volume: float,
    mu: float,
    k_lambda: float,
):
    """Solve the local Gauss-Newton system: the factored base plus a rank-one volume term."""
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


###
# Kernels
###


@wp.kernel
def _initialize_tetrahedron_proximal(
    # Inputs:
    position_linearized: wp.array[wp.vec3],
    tetrahedron_indices: wp.array2d[wp.int32],
    model_tet_poses: wp.array[wp.mat33],
    model_tet_materials: wp.array2d[float],
    model_tet_activations: wp.array[float],
    frozen_metric: wp.array[mat99f],
    # Outputs:
    frozen_coordinate: wp.array[wp.mat33],
    frozen_gradient: wp.array[wp.mat33],
    frozen_factor: wp.array[mat99f],
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
        model_tet_poses[tetrahedron],
    )
    cofactor = tet_cofactor(deformation)
    mu = model_tet_materials[tetrahedron, 0]
    k_lambda = model_tet_materials[tetrahedron, 1]
    lmbd = k_lambda + mu
    if mu == 0.0 and k_lambda == 0.0:
        frozen_coordinate[tetrahedron] = deformation
        frozen_gradient[tetrahedron] = wp.mat33(0.0)
        frozen_factor[tetrahedron] = mat99f(0.0)
        proximal_coordinate[tetrahedron] = deformation
        multiplier[tetrahedron] = wp.mat33(0.0)
        return

    alpha = tet_stable_neo_hookean_alpha(mu, k_lambda)
    constraint = wp.determinant(deformation) - alpha + model_tet_activations[tetrahedron]
    rest_volume = 1.0 / (6.0 * wp.determinant(model_tet_poses[tetrahedron]))
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


@wp.func
def _keep_linearized_response(
    tetrahedron: int,
    center: wp.mat33,
    frozen_coordinate: wp.mat33,
    frozen_gradient: wp.mat33,
    frozen_metric: mat99f,
    proximal_coordinate: wp.array[wp.mat33],
    multiplier: wp.array[wp.mat33],
) -> bool:
    """Reset a tetrahedron to the linearized stress at the candidate deformation, which zeroes its correction.

    Returns whether the linearized stress is finite; only then does the element state change.
    """
    linearized_stress = frozen_gradient + _apply_metric(center - frozen_coordinate, frozen_metric)
    if not _matrix_is_finite(linearized_stress):
        return False
    proximal_coordinate[tetrahedron] = center
    multiplier[tetrahedron] = linearized_stress
    return True


@wp.func
def _refine_tetrahedron(
    tetrahedron: int,
    wid: int,
    vertices: wp.vec4i,
    rest_pose: wp.mat33,
    center: wp.mat33,
    previous: wp.mat33,
    multiplier_value: wp.mat33,
    frozen_metric_value: mat99f,
    frozen_factor_value: mat99f,
    rest_volume: float,
    mu: float,
    k_lambda: float,
    lmbd: float,
    alpha: float,
    activation: float,
    dt: float,
    proximal_iterations: int,
    proximal_relaxation: float,
    packed_to_newton: wp.array[wp.int32],
    model_particle_mass: wp.array[float],
    model_particle_flags: wp.array[wp.int32],
    frozen_coordinate: wp.array[wp.mat33],
    frozen_gradient: wp.array[wp.mat33],
    proximal_coordinate: wp.array[wp.mat33],
    multiplier: wp.array[wp.mat33],
    nonlinear_rhs: wp.array[wp.vec3],
    world_position_residual: wp.array[float],
    world_velocity_residual: wp.array[float],
) -> bool:
    """Solve the local prox of one tetrahedron, update its multiplier, and scatter its RHS correction.

    Returns whether every step stayed finite and the frozen penalty bounds the curvature at the local
    coordinate; a failed update leaves the element state and the RHS unchanged.
    """
    deformation = previous
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
        return False

    # Measure the stationarity of the local problem against the multiplier that
    # defined it. A finite stalled line search is a soft outcome, and its residual
    # keeps reporting the remaining local KKT error.
    stationarity_residual = (
        rest_volume
        * (mu * deformation + lmbd * (wp.determinant(deformation) - alpha + activation) * tet_cofactor(deformation))
        - multiplier_value
        + _apply_metric(deformation - center, frozen_metric_value)
    )
    stationarity_correction, stationarity_solve_succeeded = _solve_gauss_newton_step(
        -stationarity_residual,
        deformation,
        frozen_factor_value,
        rest_volume,
        mu,
        k_lambda,
    )
    if not stationarity_solve_succeeded:
        return False

    if not _penalty_covers_curvature(deformation, frozen_metric_value, rest_volume, mu, k_lambda, activation):
        return False

    metric_primal = _apply_metric(center - deformation, frozen_metric_value)
    multiplier_new = multiplier_value + proximal_relaxation * metric_primal
    if not _matrix_is_finite(multiplier_new):
        return False

    metric_correction = _apply_metric(deformation - frozen_coordinate[tetrahedron], frozen_metric_value)
    correction = metric_correction - multiplier_new + frozen_gradient[tetrahedron]
    if not _matrix_is_finite(correction):
        return False

    characteristic_length = wp.pow(rest_volume, 1.0 / 3.0)
    primal_norm = wp.sqrt(_matrix_dot(center - deformation, center - deformation))
    local_change_norm = wp.sqrt(_matrix_dot(deformation - previous, deformation - previous))
    stationarity_correction_norm = wp.sqrt(_matrix_dot(stationarity_correction, stationarity_correction))
    position_residual = characteristic_length * wp.max(primal_norm, stationarity_correction_norm)
    velocity_residual = characteristic_length * local_change_norm / dt
    if (
        not wp.isfinite(characteristic_length)
        or not wp.isfinite(primal_norm)
        or not wp.isfinite(local_change_norm)
        or not wp.isfinite(stationarity_correction_norm)
        or not wp.isfinite(position_residual)
        or not wp.isfinite(velocity_residual)
    ):
        return False

    dynamic_0 = is_dynamic_particle(vertices[0], packed_to_newton, model_particle_mass, model_particle_flags)
    dynamic_1 = is_dynamic_particle(vertices[1], packed_to_newton, model_particle_mass, model_particle_flags)
    dynamic_2 = is_dynamic_particle(vertices[2], packed_to_newton, model_particle_mass, model_particle_flags)
    dynamic_3 = is_dynamic_particle(vertices[3], packed_to_newton, model_particle_mass, model_particle_flags)
    scatter_0 = dt * (correction * tet_vertex_coefficient(0, rest_pose))
    scatter_1 = dt * (correction * tet_vertex_coefficient(1, rest_pose))
    scatter_2 = dt * (correction * tet_vertex_coefficient(2, rest_pose))
    scatter_3 = dt * (correction * tet_vertex_coefficient(3, rest_pose))
    if (
        (dynamic_0 and not _vector_is_finite(scatter_0))
        or (dynamic_1 and not _vector_is_finite(scatter_1))
        or (dynamic_2 and not _vector_is_finite(scatter_2))
        or (dynamic_3 and not _vector_is_finite(scatter_3))
    ):
        return False

    if dynamic_0:
        wp.atomic_add(nonlinear_rhs, vertices[0], scatter_0)
    if dynamic_1:
        wp.atomic_add(nonlinear_rhs, vertices[1], scatter_1)
    if dynamic_2:
        wp.atomic_add(nonlinear_rhs, vertices[2], scatter_2)
    if dynamic_3:
        wp.atomic_add(nonlinear_rhs, vertices[3], scatter_3)

    proximal_coordinate[tetrahedron] = deformation
    multiplier[tetrahedron] = multiplier_new

    wp.atomic_max(world_position_residual, wid, position_residual)
    wp.atomic_max(world_velocity_residual, wid, velocity_residual)
    return True


@wp.kernel
def _update_tetrahedron_proximal(
    # Inputs:
    position_start: wp.array[wp.vec3],
    position_linearized: wp.array[wp.vec3],
    smooth_velocity: wp.array[wp.vec3],
    tetrahedron_indices: wp.array2d[wp.int32],
    model_tet_poses: wp.array[wp.mat33],
    model_tet_materials: wp.array2d[float],
    model_tet_activations: wp.array[float],
    packed_to_newton: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    model_particle_mass: wp.array[float],
    model_particle_flags: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    frozen_coordinate: wp.array[wp.mat33],
    frozen_gradient: wp.array[wp.mat33],
    frozen_metric: wp.array[mat99f],
    frozen_factor: wp.array[mat99f],
    proximal_iterations: int,
    proximal_relaxation: float,
    model_time_dt: wp.array[wp.float32],
    # Outputs:
    proximal_coordinate: wp.array[wp.mat33],
    multiplier: wp.array[wp.mat33],
    nonlinear_rhs: wp.array[wp.vec3],
    world_position_residual: wp.array[float],
    world_velocity_residual: wp.array[float],
    world_failed: wp.array[wp.bool],
):
    """Evaluate one tetrahedron prox and scatter its next global RHS correction."""
    tetrahedron = wp.tid()
    vertices = wp.vec4i(
        tetrahedron_indices[tetrahedron, 0],
        tetrahedron_indices[tetrahedron, 1],
        tetrahedron_indices[tetrahedron, 2],
        tetrahedron_indices[tetrahedron, 3],
    )
    wid = packed_world[vertices[0]]
    dt = model_time_dt[wid]
    if not world_active[wid]:
        return
    if model_tet_materials[tetrahedron, 0] == 0.0 and model_tet_materials[tetrahedron, 1] == 0.0:
        return

    position_0 = position_linearized[vertices[0]]
    position_1 = position_linearized[vertices[1]]
    position_2 = position_linearized[vertices[2]]
    position_3 = position_linearized[vertices[3]]
    if is_dynamic_particle(vertices[0], packed_to_newton, model_particle_mass, model_particle_flags):
        position_0 = position_start[vertices[0]] + dt * smooth_velocity[vertices[0]]
    if is_dynamic_particle(vertices[1], packed_to_newton, model_particle_mass, model_particle_flags):
        position_1 = position_start[vertices[1]] + dt * smooth_velocity[vertices[1]]
    if is_dynamic_particle(vertices[2], packed_to_newton, model_particle_mass, model_particle_flags):
        position_2 = position_start[vertices[2]] + dt * smooth_velocity[vertices[2]]
    if is_dynamic_particle(vertices[3], packed_to_newton, model_particle_mass, model_particle_flags):
        position_3 = position_start[vertices[3]] + dt * smooth_velocity[vertices[3]]
    rest_pose = model_tet_poses[tetrahedron]
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
    rest_volume = 1.0 / (6.0 * wp.determinant(rest_pose))
    mu = model_tet_materials[tetrahedron, 0]
    k_lambda = model_tet_materials[tetrahedron, 1]
    lmbd = k_lambda + mu
    alpha = tet_stable_neo_hookean_alpha(mu, k_lambda)
    activation = model_tet_activations[tetrahedron]

    if not _matrix_is_finite(center):
        world_failed[wid] = True
        return

    # A failed local update keeps the linearized response of the element for this iteration
    if not _refine_tetrahedron(
        tetrahedron,
        wid,
        vertices,
        rest_pose,
        center,
        previous,
        multiplier_value,
        frozen_metric_value,
        frozen_factor_value,
        rest_volume,
        mu,
        k_lambda,
        lmbd,
        alpha,
        activation,
        dt,
        proximal_iterations,
        proximal_relaxation,
        packed_to_newton,
        model_particle_mass,
        model_particle_flags,
        frozen_coordinate,
        frozen_gradient,
        proximal_coordinate,
        multiplier,
        nonlinear_rhs,
        world_position_residual,
        world_velocity_residual,
    ):
        if not _keep_linearized_response(
            tetrahedron,
            center,
            frozen_coordinate[tetrahedron],
            frozen_gradient[tetrahedron],
            frozen_metric_value,
            proximal_coordinate,
            multiplier,
        ):
            world_failed[wid] = True


###
# Interfaces
###


class DeformableTetrahedronProximal:
    """Fixed-majorizer tetrahedron proximal updates of a deformable system.

    The proximal metric of each tetrahedron is its majorizing metric from the assembly,
    :attr:`DeformableSystemData.tetrahedron_metric`. The proximal state lives in
    :class:`DeformableTetrahedronProximalData`, exposed as :attr:`data`.

    Args:
        model: Newton model of the deformables.
        topology: Packed topology of the deformables.
        system_data: Arrays of the deformable system, which receive the right-hand-side
            corrections, residuals, and failures of the proximal updates.
        data: Proximal state of the tetrahedra.
        iterations: Fixed number of local Gauss-Newton iterations per update.
        relaxation: Relaxation factor in ``[0, 1]`` of the local multiplier update.
    """

    def __init__(
        self,
        model: Model,
        topology: DeformableTopology,
        system_data: DeformableSystemData,
        data: DeformableTetrahedronProximalData,
        iterations: int,
        relaxation: float,
    ):
        self.model = model
        self.device = model.device
        self.topology = topology
        """Packed topology of the deformables."""
        self.system_data = system_data
        """Arrays of the deformable system."""
        self.tetrahedron_count = int(model.tet_count)
        self.iterations = iterations
        self.relaxation = relaxation
        self._data = data

    ###
    # Properties
    ###

    @property
    def data(self) -> DeformableTetrahedronProximalData:
        """Returns the proximal state of the tetrahedra."""
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
            _initialize_tetrahedron_proximal,
            dim=self.tetrahedron_count,
            inputs=[
                system.position_linearized,
                self.topology.tetrahedron_indices,
                model.tet_poses,
                model.tet_materials,
                model.tet_activations,
                system.tetrahedron_metric,
            ],
            outputs=[
                data.frozen_coordinate,
                data.frozen_gradient,
                data.frozen_factor,
                data.proximal_coordinate,
                data.multiplier,
            ],
            device=self.device,
        )

    def update(self, time_step: wp.array[wp.float32]) -> None:
        """Evaluate all active tetrahedron proxes and add their RHS corrections.

        Args:
            time_step: Time step of each world [s].
        """
        system = self.system_data
        model = self.model
        data = self._data
        wp.launch(
            _update_tetrahedron_proximal,
            dim=self.tetrahedron_count,
            inputs=[
                system.position_start,
                system.position_linearized,
                system.smooth_velocity,
                self.topology.tetrahedron_indices,
                model.tet_poses,
                model.tet_materials,
                model.tet_activations,
                self.topology.packed_to_newton,
                self.topology.packed_solve_world,
                model.particle_mass,
                model.particle_flags,
                system.world_active,
                data.frozen_coordinate,
                data.frozen_gradient,
                system.tetrahedron_metric,
                data.frozen_factor,
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
