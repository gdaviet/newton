# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body-space numerical primitives for LOX rigid dynamics."""

from __future__ import annotations

import warp as wp

from ...core.types import mat66f, vec6f

__all__ = [
    "PrimalRowContribution",
    "compute_augmented_joint_multiplier",
    "compute_augmented_joint_row",
    "compute_body_explicit_wrench",
    "compute_body_inertial_system",
    "compute_dynamic_joint_row",
    "compute_velocity_distance",
    "make_spatial_mass_matrix",
]

wp.set_module_options({"enable_backward": False})


@wp.struct
class PrimalRowContribution:
    """Matrix and right-hand-side contribution of one body-space term."""

    matrix: mat66f
    """Symmetric contribution to the body-space operator."""
    right_hand_side: vec6f
    """Contribution to the body-space right-hand side."""


@wp.func
def make_spatial_mass_matrix(mass: wp.float32, inertia_world: wp.mat33f) -> mat66f:
    """Construct a linear-first spatial mass matrix about the center of mass.

    Args:
        mass: Body mass [kg].
        inertia_world: World-space inertia about the center of mass [kg m^2].

    Returns:
        ``diag(mass I_3, inertia_world)``.
    """
    matrix = mat66f(0.0)
    for index in range(3):
        matrix[index, index] = mass
    for row in range(3):
        for col in range(3):
            matrix[row + 3, col + 3] = inertia_world[row, col]
    return matrix


@wp.func
def _outer_product(vector: vec6f) -> mat66f:
    matrix = mat66f(0.0)
    for row in range(6):
        for col in range(6):
            matrix[row, col] = vector[row] * vector[col]
    return matrix


@wp.func
def compute_body_explicit_wrench(
    mass: wp.float32,
    inertia_world: wp.mat33f,
    velocity_previous: vec6f,
    external_wrench: vec6f,
    actuation_wrench: vec6f,
    gravity: wp.vec3f,
) -> vec6f:
    """Combine explicit body forces in world coordinates.

    Args:
        mass: Body mass [kg].
        inertia_world: World-space inertia about the center of mass [kg m^2].
        velocity_previous: Begin-of-step linear-first body twist [m/s, rad/s].
        external_wrench: External body wrench excluding gravity [N, N m].
        actuation_wrench: Joint-actuation body wrench [N, N m].
        gravity: World-space gravitational acceleration [m/s^2].

    Returns:
        Explicit force and torque, including gravity and the gyroscopic torque [N, N m].
    """
    result = external_wrench + actuation_wrench
    for index in range(3):
        result[index] += mass * gravity[index]

    angular_velocity = wp.vec3f(velocity_previous[3], velocity_previous[4], velocity_previous[5])
    gyroscopic_torque = -wp.cross(angular_velocity, inertia_world @ angular_velocity)
    for index in range(3):
        result[index + 3] += gyroscopic_torque[index]
    return result


@wp.func
def compute_body_inertial_system(
    mass: wp.float32,
    inertia_world: wp.mat33f,
    velocity_previous: vec6f,
    force_explicit: vec6f,
    time_step: wp.float32,
) -> PrimalRowContribution:
    """Compute the inertial operator and explicit-force right-hand side.

    Args:
        mass: Body mass [kg].
        inertia_world: World-space inertia about the center of mass [kg m^2].
        velocity_previous: Begin-of-step linear-first body twist [m/s, rad/s].
        force_explicit: Explicit body wrench [N, N m].
        time_step: Time step [s].

    Returns:
        The mass matrix and ``M v_previous + h force_explicit``.
    """
    result = PrimalRowContribution()
    result.matrix = make_spatial_mass_matrix(mass, inertia_world)
    result.right_hand_side = result.matrix @ velocity_previous + time_step * force_explicit
    return result


@wp.func
def compute_dynamic_joint_row(
    jacobian: vec6f,
    effective_inertia: wp.float32,
    free_velocity: wp.float32,
) -> PrimalRowContribution:
    """Eliminate one implicit joint-dynamics coordinate into body space.

    The joint equation is ``effective_inertia * dq = h_joint`` with
    ``free_velocity = h_joint / effective_inertia`` and ``dq = J v``.

    Args:
        jacobian: Linear-first joint velocity Jacobian row.
        effective_inertia: Positive implicit joint inertia.
        free_velocity: Implicit joint free velocity.

    Returns:
        ``effective_inertia J^T J`` and
        ``effective_inertia free_velocity J^T``.
    """
    result = PrimalRowContribution()
    result.matrix = effective_inertia * _outer_product(jacobian)
    result.right_hand_side = effective_inertia * free_velocity * jacobian
    return result


@wp.func
def compute_augmented_joint_row(
    jacobian: vec6f,
    residual: wp.float32,
    multiplier: wp.float32,
    penalty: wp.float32,
    time_step: wp.float32,
    linearization_velocity: wp.float32,
) -> PrimalRowContribution:
    """Linearize one structural joint row with augmented Lagrangian forces.

    For ``C(q(v)) ~= residual + h J (v - v_k)``, this returns the terms in

    ``(M + h^2 penalty J^T J) v``
    ``= f - h J^T (multiplier + penalty residual)``
    ``+ h^2 penalty J^T J v_k``.

    Args:
        jacobian: Linear-first structural joint Jacobian row.
        residual: Current joint position residual [m or rad].
        multiplier: Persistent structural multiplier [N or N m].
        penalty: Positive augmented penalty [N/m or N m/rad].
        time_step: Time step [s].
        linearization_velocity: Row velocity ``J v_k`` at the linearization
            twist [m/s or rad/s]. Pass zero for the first linearly implicit
            assembly about the begin-step pose.

    Returns:
        The structural matrix and right-hand-side contributions.
    """
    result = PrimalRowContribution()
    result.matrix = time_step * time_step * penalty * _outer_product(jacobian)
    result.right_hand_side = (
        -time_step * (multiplier + penalty * residual) + time_step * time_step * penalty * linearization_velocity
    ) * jacobian
    return result


@wp.func
def compute_augmented_joint_multiplier(
    multiplier: wp.float32,
    penalty: wp.float32,
    candidate_residual: wp.float32,
) -> wp.float32:
    """Update one accepted structural multiplier.

    Args:
        multiplier: Previous multiplier [N or N m].
        penalty: Positive augmented penalty [N/m or N m/rad].
        candidate_residual: Residual at the accepted candidate [m or rad].

    Returns:
        ``multiplier + penalty * candidate_residual``.
    """
    return multiplier + penalty * candidate_residual


@wp.func
def compute_velocity_distance(
    first: vec6f,
    second: vec6f,
    time_step: wp.float32,
    position_tolerance: wp.float32,
    rotation_tolerance: wp.float32,
) -> wp.float32:
    """Compute the tolerance-normalized rigid-twist distance.

    Args:
        first: First linear-first body twist [m/s, rad/s].
        second: Second linear-first body twist [m/s, rad/s].
        time_step: Time step [s].
        position_tolerance: Translational displacement tolerance [m].
        rotation_tolerance: Rotational displacement tolerance [rad].

    Returns:
        Maximum normalized translational or rotational end-of-step change.
    """
    linear_max = wp.float32(0.0)
    angular_max = wp.float32(0.0)
    for index in range(3):
        linear_max = wp.max(linear_max, wp.abs(first[index] - second[index]))
        angular_max = wp.max(angular_max, wp.abs(first[index + 3] - second[index + 3]))
    return wp.max(
        time_step * linear_max / position_tolerance,
        time_step * angular_max / rotation_tolerance,
    )
