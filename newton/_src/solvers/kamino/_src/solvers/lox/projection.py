# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free one-constraint projection primitives.

Contact-facing functions use Kamino's normal-last order
``(tangent_x, tangent_y, normal_z)``. Body twists and Jacobian columns use
Kamino's linear-first 6D convention.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from ...core.types import mat36f, mat66f, vec6f
from .contact import solve_contact_coulomb_isotropic, solve_contact_coulomb_newton

__all__ = [
    "PROJECTION_STATUS_INVALID",
    "PROJECTION_STATUS_REGULARIZED",
    "PROJECTION_STATUS_VALID",
    "apply_contact_desaxce_correction",
    "compute_contact_delassus",
    "compute_limit_delassus",
    "prepare_contact_coulomb_delassus",
    "project_contact_coulomb_isotropic_local",
    "project_contact_coulomb_local",
    "project_friction_local",
    "project_limit_local",
]

PROJECTION_STATUS_INVALID = 0
"""The local block or an input value was non-finite or non-positive."""

PROJECTION_STATUS_VALID = 1
"""The one-constraint update completed successfully."""

PROJECTION_STATUS_REGULARIZED = 2
"""The contact update used a numerically regularized Delassus block."""

wp.set_module_options({"enable_backward": False})

_FUSED_RIGID_WORLD_MIN_BLOCKS_PER_SM = 4


def _can_fuse_rigid_projection_by_world(
    device: wp.Device,
    world_count: int,
    *,
    required_world_arrays: tuple[wp.array[Any] | None, ...],
    parallel_constraint_capacity: int | None = None,
    world_block_dim: int | None = None,
    minimum_blocks_per_sm: int = _FUSED_RIGID_WORLD_MIN_BLOCKS_PER_SM,
) -> bool:
    """Return whether launch fusion or world-level occupancy favors one block per world."""
    enough_world_blocks = device.is_cuda and world_count >= device.sm_count * minimum_blocks_per_sm
    small_world_work = (
        parallel_constraint_capacity is not None
        and world_block_dim is not None
        and world_count > 0
        and parallel_constraint_capacity <= world_count * world_block_dim
    )
    return (
        device.is_cuda
        and (enough_world_blocks or small_world_work)
        and all(array is not None for array in required_world_arrays)
    )


@wp.struct
class ContactProjectionData:
    """Contact data that is invariant throughout a body-space solve."""

    delassus: wp.mat33f
    status: wp.int32


@wp.struct
class ScalarProjectionResult:
    """Minimal result of a prepared scalar projection."""

    reaction: wp.float32
    reaction_delta: wp.float32
    status: wp.int32


@wp.struct
class ContactProjectionResult:
    """Minimal result of a prepared Coulomb projection."""

    reaction: wp.vec3f
    reaction_delta: wp.vec3f
    status: wp.int32


@wp.func
def _is_finite_vec3(value: wp.vec3f) -> wp.bool:
    return wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])


@wp.func
def apply_contact_desaxce_correction(velocity: wp.vec3f, friction: wp.float32) -> wp.vec3f:
    """Apply the de Saxce correction to a normal-last contact velocity.

    Args:
        velocity: Raw normal-last relative contact velocity.
        friction: Nonnegative isotropic Coulomb friction coefficient.

    Returns:
        The corrected velocity. Invalid inputs are preserved for the caller's
        projection-status check.
    """
    if not _is_finite_vec3(velocity) or not wp.isfinite(friction) or friction <= 0.0:
        return velocity

    tangent_norm = wp.sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1])
    return wp.vec3f(velocity[0], velocity[1], velocity[2] + friction * tangent_norm)


@wp.func
def _is_finite_vec6(value: vec6f) -> wp.bool:
    finite = wp.bool(True)
    for index in range(6):
        finite = finite and wp.isfinite(value[index])
    return finite


@wp.func
def _is_finite_mat33(value: wp.mat33f) -> wp.bool:
    finite = wp.bool(True)
    for row in range(3):
        for col in range(3):
            finite = finite and wp.isfinite(value[row, col])
    return finite


@wp.func
def _is_finite_mat36(value: mat36f) -> wp.bool:
    finite = wp.bool(True)
    for row in range(3):
        for col in range(6):
            finite = finite and wp.isfinite(value[row, col])
    return finite


@wp.func
def _is_finite_mat66(value: mat66f) -> wp.bool:
    finite = wp.bool(True)
    for row in range(6):
        for col in range(6):
            finite = finite and wp.isfinite(value[row, col])
    return finite


@wp.func
def compute_contact_delassus(
    jacobian_first: mat36f,
    inverse_weight_first: mat66f,
    jacobian_second: mat36f,
    inverse_weight_second: mat66f,
) -> wp.mat33f:
    """Compute the full normal-last local contact Delassus block."""
    return jacobian_first @ inverse_weight_first @ wp.transpose(
        jacobian_first
    ) + jacobian_second @ inverse_weight_second @ wp.transpose(jacobian_second)


@wp.func
def prepare_contact_coulomb_delassus(
    delassus: wp.mat33f,
    velocity_bias: wp.vec3f,
    friction: wp.float32,
) -> ContactProjectionData:
    """Validate and, when numerically marginal, regularize a contact block."""
    result = ContactProjectionData()
    result.delassus = delassus
    result.status = PROJECTION_STATUS_INVALID
    if not _is_finite_vec3(velocity_bias) or not wp.isfinite(friction) or friction < 0.0:
        return result
    if not _is_finite_mat33(delassus):
        return result

    scale = wp.float32(0.0)
    asymmetry = wp.float32(0.0)
    symmetric = wp.mat33f(0.0)
    for row in range(3):
        for col in range(3):
            scale = wp.max(scale, wp.abs(delassus[row, col]))
            asymmetry = wp.max(asymmetry, wp.abs(delassus[row, col] - delassus[col, row]))
            symmetric[row, col] = 0.5 * (delassus[row, col] + delassus[col, row])
    if scale <= 0.0 or asymmetry > 1.0e-5 * scale:
        return result

    eigenvectors, eigenvalues = wp.eig3(symmetric)
    if not _is_finite_vec3(eigenvalues):
        return result
    minimum_eigenvalue = wp.min(eigenvalues[0], wp.min(eigenvalues[1], eigenvalues[2]))
    if minimum_eigenvalue < -1.0e-5 * scale:
        return result

    eigenvalue_floor = 1.0e-6 * scale
    clamped_eigenvalues = wp.vec3f(
        wp.max(eigenvalues[0], eigenvalue_floor),
        wp.max(eigenvalues[1], eigenvalue_floor),
        wp.max(eigenvalues[2], eigenvalue_floor),
    )
    regularized = minimum_eigenvalue < eigenvalue_floor
    if regularized:
        symmetric = eigenvectors @ wp.diag(clamped_eigenvalues) @ wp.transpose(eigenvectors)
        result.status = PROJECTION_STATUS_REGULARIZED
    else:
        result.status = PROJECTION_STATUS_VALID
    result.delassus = symmetric
    return result


@wp.func
def _contact_projection_inputs_are_finite(
    jacobian_first: mat36f,
    inverse_weight_first: mat66f,
    jacobian_second: mat36f,
    inverse_weight_second: mat66f,
) -> wp.bool:
    return (
        _is_finite_mat36(jacobian_first)
        and _is_finite_mat66(inverse_weight_first)
        and _is_finite_mat36(jacobian_second)
        and _is_finite_mat66(inverse_weight_second)
    )


@wp.func
def prepare_contact_coulomb(
    jacobian_first: mat36f,
    inverse_weight_first: mat66f,
    jacobian_second: mat36f,
    inverse_weight_second: mat66f,
    velocity_bias: wp.vec3f,
    friction: wp.float32,
) -> ContactProjectionData:
    """Validate fixed inputs and prepare the normal-last contact block."""
    result = prepare_contact_coulomb_delassus(
        compute_contact_delassus(
            jacobian_first,
            inverse_weight_first,
            jacobian_second,
            inverse_weight_second,
        ),
        velocity_bias,
        friction,
    )
    if not _contact_projection_inputs_are_finite(
        jacobian_first,
        inverse_weight_first,
        jacobian_second,
        inverse_weight_second,
    ):
        result.status = PROJECTION_STATUS_INVALID
    return result


@wp.func
def compute_limit_delassus(
    jacobian_first: vec6f,
    inverse_weight_first: mat66f,
    jacobian_second: vec6f,
    inverse_weight_second: mat66f,
) -> wp.float32:
    """Compute the scalar local joint-limit Delassus coefficient."""
    return wp.dot(jacobian_first, inverse_weight_first @ jacobian_first) + wp.dot(
        jacobian_second, inverse_weight_second @ jacobian_second
    )


@wp.func
def project_limit_local(
    current_velocity: wp.float32,
    reaction_old: wp.float32,
    delassus: wp.float32,
) -> ScalarProjectionResult:
    """Project one prepared scalar unilateral constraint."""
    result = ScalarProjectionResult()
    result.reaction = reaction_old
    result.reaction_delta = 0.0
    result.status = PROJECTION_STATUS_INVALID
    if (
        not wp.isfinite(current_velocity)
        or not wp.isfinite(reaction_old)
        or not wp.isfinite(delassus)
        or delassus <= 0.0
    ):
        return result

    free_velocity = current_velocity - delassus * reaction_old
    reaction_new = wp.max(-free_velocity / delassus, 0.0)
    reaction_delta = reaction_new - reaction_old
    if not wp.isfinite(reaction_new) or not wp.isfinite(reaction_delta):
        return result
    result.reaction = reaction_new
    result.reaction_delta = reaction_delta
    result.status = PROJECTION_STATUS_VALID
    return result


@wp.func
def project_friction_local(
    current_velocity: wp.float32,
    reaction_old: wp.float32,
    delassus: wp.float32,
    impulse_bound: wp.float32,
) -> ScalarProjectionResult:
    """Project one prepared bounded scalar friction constraint."""
    result = ScalarProjectionResult()
    result.reaction = reaction_old
    result.reaction_delta = 0.0
    result.status = PROJECTION_STATUS_INVALID
    if (
        not wp.isfinite(current_velocity)
        or not wp.isfinite(reaction_old)
        or not wp.isfinite(delassus)
        or delassus <= 0.0
        or not wp.isfinite(impulse_bound)
        or impulse_bound < 0.0
    ):
        return result

    free_velocity = current_velocity - delassus * reaction_old
    reaction_new = wp.clamp(-free_velocity / delassus, -impulse_bound, impulse_bound)
    reaction_delta = reaction_new - reaction_old
    if not wp.isfinite(reaction_new) or not wp.isfinite(reaction_delta):
        return result
    result.reaction = reaction_new
    result.reaction_delta = reaction_delta
    result.status = PROJECTION_STATUS_VALID
    return result


@wp.func
def project_contact_coulomb_isotropic_local(
    current_velocity: wp.vec3f,
    reaction_old: wp.vec3f,
    delassus: wp.float32,
    friction: wp.float32,
) -> ContactProjectionResult:
    """Project one prepared normal-last contact with scalar Delassus."""
    result = ContactProjectionResult()
    result.reaction = reaction_old
    result.reaction_delta = wp.vec3f(0.0)
    result.status = PROJECTION_STATUS_INVALID
    if not wp.isfinite(delassus) or delassus <= 0.0 or not wp.isfinite(friction) or friction < 0.0:
        return result
    free_velocity = current_velocity - delassus * reaction_old
    if not _is_finite_vec3(free_velocity):
        return result

    reaction_new = solve_contact_coulomb_isotropic(
        delassus,
        free_velocity,
        wp.vec3f(0.0, 0.0, 1.0),
        friction,
    )
    reaction_delta = reaction_new - reaction_old
    if not _is_finite_vec3(reaction_new) or not _is_finite_vec3(reaction_delta):
        return result
    result.reaction = reaction_new
    result.reaction_delta = reaction_delta
    result.status = PROJECTION_STATUS_VALID
    return result


@wp.func
def project_contact_coulomb_local(
    current_velocity: wp.vec3f,
    reaction_old: wp.vec3f,
    delassus: wp.mat33f,
    friction: wp.float32,
) -> ContactProjectionResult:
    """Project one prepared normal-last Coulomb contact."""
    result = ContactProjectionResult()
    result.reaction = reaction_old
    result.reaction_delta = wp.vec3f(0.0)
    result.status = PROJECTION_STATUS_INVALID
    free_velocity = current_velocity - delassus @ reaction_old
    if not _is_finite_vec3(free_velocity):
        return result

    reaction_new = solve_contact_coulomb_newton(delassus, free_velocity, friction)
    reaction_delta = reaction_new - reaction_old
    if not _is_finite_vec3(reaction_new) or not _is_finite_vec3(reaction_delta):
        return result
    result.reaction = reaction_new
    result.reaction_delta = reaction_delta
    result.status = PROJECTION_STATUS_VALID
    return result
