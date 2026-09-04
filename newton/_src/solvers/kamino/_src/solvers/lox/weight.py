# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-body consensus weights for the LOX split.

The primitive uses Kamino's linear-first twist order
``(linear_x, linear_y, linear_z, angular_x, angular_y, angular_z)`` but is
otherwise independent of Kamino containers. The body mass matrix is
``M = diag(m I_3, I_world)`` and is referenced at the body's center of mass.
"""

from __future__ import annotations

import warp as wp

from ...core.types import mat66f

__all__ = [
    "BODY_WEIGHT_BETA_DEFAULT",
    "BODY_WEIGHT_SIGMA_DEFAULT",
    "BODY_WEIGHT_STATUS_INVALID",
    "BODY_WEIGHT_STATUS_REGULARIZED",
    "BODY_WEIGHT_STATUS_VALID",
    "DEFORMABLE_WEIGHT_BETA_DEFAULT",
    "BodyWeightResult",
    "compute_body_weight_mass_proportional",
]

BODY_WEIGHT_SIGMA_DEFAULT = 1.0e-3
"""Default lower spectral fraction from Daviet (2020)."""

BODY_WEIGHT_BETA_DEFAULT = 4.0
"""Default normalized smooth-weight transition threshold."""

DEFORMABLE_WEIGHT_BETA_DEFAULT = 25.0
"""Default normalized smooth-weight transition threshold for deformable nodes."""

BODY_WEIGHT_STATUS_INVALID = 0
"""The input contained a non-finite value or an invalid policy parameter."""

BODY_WEIGHT_STATUS_VALID = 1
"""The input was finite and required no spectral regularization."""

BODY_WEIGHT_STATUS_REGULARIZED = 2
"""At least one finite mass, inertia, symmetry, or smooth eigenvalue was floored."""

wp.set_module_options({"enable_backward": False})


@wp.struct
class BodyWeightResult:
    """Result of one mass-proportional rigid-body weight computation."""

    weight: mat66f
    """Mass-proportional body weight ``W = alpha M``."""
    inverse_weight: mat66f
    """Inverse mass-proportional body weight ``W^-1``."""
    eta: wp.float32
    """Minimum eigenvalue of the regularized mass-normalized smooth block."""
    alpha: wp.float32
    """Scalar weight selected by the paper's clamping policy."""
    status: wp.int32
    """One of the ``BODY_WEIGHT_STATUS_*`` values."""


@wp.func
def _make_invalid_body_weight_result() -> BodyWeightResult:
    result = BodyWeightResult()
    result.weight = mat66f(0.0)
    result.inverse_weight = mat66f(0.0)
    result.eta = 0.0
    result.alpha = 0.0
    result.status = BODY_WEIGHT_STATUS_INVALID
    return result


@wp.func
def _is_finite_mat33(value: wp.mat33f) -> wp.bool:
    finite = wp.bool(True)
    for row in range(3):
        for col in range(3):
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
def _symmetrize_mat33(value: wp.mat33f) -> wp.mat33f:
    return 0.5 * (value + wp.transpose(value))


@wp.func
def _symmetrize_mat66(value: mat66f) -> mat66f:
    return 0.5 * (value + wp.transpose(value))


@wp.func
def _requires_symmetry_regularization_mat33(value: wp.mat33f, tolerance: wp.float32) -> wp.bool:
    scale = wp.float32(1.0)
    asymmetry = wp.float32(0.0)
    for row in range(3):
        for col in range(3):
            scale = wp.max(scale, wp.abs(value[row, col]))
            asymmetry = wp.max(asymmetry, wp.abs(value[row, col] - value[col, row]))
    return asymmetry > tolerance * scale


@wp.func
def _requires_symmetry_regularization_mat66(value: mat66f, tolerance: wp.float32) -> wp.bool:
    scale = wp.float32(1.0)
    asymmetry = wp.float32(0.0)
    for row in range(6):
        for col in range(6):
            scale = wp.max(scale, wp.abs(value[row, col]))
            asymmetry = wp.max(asymmetry, wp.abs(value[row, col] - value[col, row]))
    return asymmetry > tolerance * scale


@wp.func
def _compute_symmetric_eigenvalue_min_fixed(matrix: mat66f) -> wp.float32:
    """Approximate the minimum eigenvalue with twelve cyclic Jacobi sweeps."""
    diagonalized = matrix
    sweep = int(0)
    while sweep < 12:
        first = int(0)
        while first < 5:
            second = first + 1
            while second < 6:
                off_diagonal = diagonalized[first, second]
                if off_diagonal != 0.0:
                    tau = (diagonalized[second, second] - diagonalized[first, first]) / (2.0 * off_diagonal)
                    tangent = wp.float32(1.0)
                    if tau != 0.0:
                        tangent = wp.sign(tau) / (wp.abs(tau) + wp.sqrt(1.0 + tau * tau))
                    cosine = 1.0 / wp.sqrt(1.0 + tangent * tangent)
                    sine = tangent * cosine

                    first_diagonal = diagonalized[first, first]
                    second_diagonal = diagonalized[second, second]
                    diagonalized[first, first] = (
                        cosine * cosine * first_diagonal
                        - 2.0 * cosine * sine * off_diagonal
                        + sine * sine * second_diagonal
                    )
                    diagonalized[second, second] = (
                        sine * sine * first_diagonal
                        + 2.0 * cosine * sine * off_diagonal
                        + cosine * cosine * second_diagonal
                    )
                    diagonalized[first, second] = 0.0
                    diagonalized[second, first] = 0.0

                    other = int(0)
                    while other < 6:
                        if other != first and other != second:
                            other_first = diagonalized[other, first]
                            other_second = diagonalized[other, second]
                            rotated_first = cosine * other_first - sine * other_second
                            rotated_second = sine * other_first + cosine * other_second
                            diagonalized[other, first] = rotated_first
                            diagonalized[first, other] = rotated_first
                            diagonalized[other, second] = rotated_second
                            diagonalized[second, other] = rotated_second
                        other += 1
                second += 1
            first += 1
        sweep += 1

    eigenvalue_min = diagonalized[0, 0]
    for index in range(1, 6):
        eigenvalue_min = wp.min(eigenvalue_min, diagonalized[index, index])
    return eigenvalue_min


@wp.func
def compute_body_weight_mass_proportional(
    smooth_diagonal: mat66f,
    mass: wp.float32,
    inertia_world: wp.mat33f,
    sigma: wp.float32 = BODY_WEIGHT_SIGMA_DEFAULT,
    beta: wp.float32 = BODY_WEIGHT_BETA_DEFAULT,
    mass_floor: wp.float32 = 1.0e-8,
    inertia_floor: wp.float32 = 1.0e-10,
    eta_floor: wp.float32 = 1.0e-6,
    symmetry_tolerance: wp.float32 = 1.0e-5,
) -> BodyWeightResult:
    """Compute a mass-proportional rigid-body weight and its inverse.

    The implementation forms the symmetric normalization
    ``M^-1/2 smooth_diagonal M^-1/2``. The world-space inertia square root
    comes from a symmetric ``3 x 3`` eigendecomposition, and the normalized
    minimum eigenvalue uses twelve fixed cyclic Jacobi sweeps. This bounded
    device work is suitable for CUDA graph capture.

    Finite asymmetric matrices are symmetrized. Positive mass below
    ``mass_floor``, inertia eigenvalues below ``inertia_floor``, and normalized
    smooth eigenvalues below ``eta_floor`` are clamped and reported with
    ``BODY_WEIGHT_STATUS_REGULARIZED``. Non-finite input, nonpositive mass, or
    invalid policy parameters return ``BODY_WEIGHT_STATUS_INVALID`` and zero
    matrices.

    Args:
        smooth_diagonal: Symmetric ``6 x 6`` body block ``A_ii`` in
            linear-first twist order.
        mass: Positive body mass [kg].
        inertia_world: Symmetric body inertia about its center of mass in
            world axes [kg m^2].
        sigma: Lower spectral fraction in the weight clamp.
        beta: Nominal weight transition threshold. The ``sigma`` floor may
            exceed it for sufficiently stiff modes.
        mass_floor: Minimum accepted positive mass [kg].
        inertia_floor: Minimum principal moment [kg m^2].
        eta_floor: Minimum dimensionless normalized smooth eigenvalue.
        symmetry_tolerance: Relative tolerance before symmetrization is
            reported as regularization.

    Returns:
        The weight, inverse weight, spectral values, and validation status.
    """
    if (
        not wp.isfinite(mass)
        or not _is_finite_mat33(inertia_world)
        or not _is_finite_mat66(smooth_diagonal)
        or not wp.isfinite(sigma)
        or not wp.isfinite(beta)
        or not wp.isfinite(mass_floor)
        or not wp.isfinite(inertia_floor)
        or not wp.isfinite(eta_floor)
        or not wp.isfinite(symmetry_tolerance)
        or mass <= 0.0
        or sigma <= 0.0
        or sigma > 1.0
        or beta <= 0.0
        or mass_floor <= 0.0
        or inertia_floor <= 0.0
        or eta_floor <= 0.0
        or symmetry_tolerance < 0.0
    ):
        return _make_invalid_body_weight_result()

    status = wp.int32(BODY_WEIGHT_STATUS_VALID)
    regularized_mass = mass
    if regularized_mass < mass_floor:
        regularized_mass = mass_floor
        status = BODY_WEIGHT_STATUS_REGULARIZED

    if _requires_symmetry_regularization_mat33(inertia_world, symmetry_tolerance):
        status = BODY_WEIGHT_STATUS_REGULARIZED
    if _requires_symmetry_regularization_mat66(smooth_diagonal, symmetry_tolerance):
        status = BODY_WEIGHT_STATUS_REGULARIZED
    symmetric_inertia = _symmetrize_mat33(inertia_world)
    symmetric_smooth = _symmetrize_mat66(smooth_diagonal)

    inertia_axes, inertia_eigenvalues = wp.eig3(symmetric_inertia)
    regularized_inertia_eigenvalues = inertia_eigenvalues
    for index in range(3):
        if not wp.isfinite(regularized_inertia_eigenvalues[index]):
            return _make_invalid_body_weight_result()
        for row in range(3):
            if not wp.isfinite(inertia_axes[row, index]):
                return _make_invalid_body_weight_result()
        if regularized_inertia_eigenvalues[index] < inertia_floor:
            regularized_inertia_eigenvalues[index] = inertia_floor
            status = BODY_WEIGHT_STATUS_REGULARIZED

    regularized_inertia = wp.mat33f(0.0)
    inverse_inertia = wp.mat33f(0.0)
    inverse_sqrt_inertia = wp.mat33f(0.0)
    for row in range(3):
        for col in range(3):
            for index in range(3):
                axis_product = inertia_axes[row, index] * inertia_axes[col, index]
                principal_inertia = regularized_inertia_eigenvalues[index]
                regularized_inertia[row, col] += axis_product * principal_inertia
                inverse_inertia[row, col] += axis_product / principal_inertia
                inverse_sqrt_inertia[row, col] += axis_product / wp.sqrt(principal_inertia)

    inverse_sqrt_mass = 1.0 / wp.sqrt(regularized_mass)
    inverse_sqrt_spatial_mass = mat66f(0.0)
    for index in range(3):
        inverse_sqrt_spatial_mass[index, index] = inverse_sqrt_mass
    for row in range(3):
        for col in range(3):
            inverse_sqrt_spatial_mass[row + 3, col + 3] = inverse_sqrt_inertia[row, col]

    normalized_smooth = inverse_sqrt_spatial_mass @ symmetric_smooth @ inverse_sqrt_spatial_mass
    normalized_smooth = _symmetrize_mat66(normalized_smooth)
    eta = _compute_symmetric_eigenvalue_min_fixed(normalized_smooth)
    if not wp.isfinite(eta):
        return _make_invalid_body_weight_result()
    if eta < eta_floor:
        eta = eta_floor
        status = BODY_WEIGHT_STATUS_REGULARIZED

    alpha = wp.max(sigma * eta, wp.min(beta, eta))
    if not wp.isfinite(alpha) or alpha <= 0.0:
        return _make_invalid_body_weight_result()

    weight = mat66f(0.0)
    inverse_weight = mat66f(0.0)
    for index in range(3):
        weight[index, index] = alpha * regularized_mass
        inverse_weight[index, index] = 1.0 / (alpha * regularized_mass)
    for row in range(3):
        for col in range(3):
            weight[row + 3, col + 3] = alpha * regularized_inertia[row, col]
            inverse_weight[row + 3, col + 3] = inverse_inertia[row, col] / alpha

    result = BodyWeightResult()
    result.weight = weight
    result.inverse_weight = inverse_weight
    result.eta = eta
    result.alpha = alpha
    result.status = status
    return result
