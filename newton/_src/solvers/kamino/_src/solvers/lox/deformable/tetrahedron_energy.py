# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Spectral operations of the stable Neo-Hookean tetrahedra of the LOX deformables.

The Hessian of the stable Neo-Hookean energy of a tetrahedron has a semi-analytic
eigensystem (Smith et al., "Analytic Eigensystems for Isotropic Distortion Energies",
2019): with the rotation-variant singular value decomposition ``F = U diag(s) V^T``, three
scaling modes come from a 3-by-3 eigenproblem on the singular values, and the twist and flip
modes of each pair of axes have closed-form eigenvalues. One eigensystem per time step yields
both a projected positive semidefinite metric and a majorizing metric of the energy.
"""

import warp as wp

from .energy import mat99f, tet_stable_neo_hookean_differential, vec9f

###
# Module interface
###

__all__ = [
    "TetrahedronSpectralMetrics",
    "tet_stable_neo_hookean_eigensystem",
    "tet_stable_neo_hookean_hessian",
    "tet_stable_neo_hookean_spectral_metrics",
]

###
# Module configs
###

# Keep the 9-by-9 metric reconstruction loops compact in callers.
wp.set_module_options({"enable_backward": False, "max_unroll": 4})


###
# Constants
###


_EIGEN_TOLERANCE = 1.0e-6
"""Relative tolerance, against the largest eigenvalue magnitude, of the zero modes of the Hessian."""


###
# Types
###


@wp.struct
class TetrahedronSpectralMetrics:
    """Projected and majorizing metrics sharing one eigensystem."""

    projected: mat99f
    """Hessian with its eigenvalues clamped from below by the minimum eigenvalue."""
    majorizer: mat99f
    """Positive semidefinite metric that majorizes the Hessian, with a margin for negative curvature."""


###
# Functions
###


@wp.func
def _determinant_pressure(
    deformation: wp.mat33,
    k_mu: float,
    k_lambda: float,
    activation: float,
) -> float:
    """Evaluate ``lambda_hat * (J - alpha + activation)`` without cancellation."""
    lambda_hat = k_lambda + k_mu
    pressure = lambda_hat * (wp.determinant(deformation) - 1.0 + activation)
    if lambda_hat > 0.0:
        pressure -= lambda_hat * k_mu / wp.max(lambda_hat, 1.0e-6)
    elif lambda_hat < 0.0:
        pressure += lambda_hat * k_mu / wp.max(-lambda_hat, 1.0e-6)
    return pressure


@wp.func
def _column_major(value: wp.mat33) -> vec9f:
    """Return the column-major vector of a matrix, matching the Hessian index ``row + 3 column``."""
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
def _skew(value: wp.vec3) -> wp.mat33:
    """Return the cross-product matrix of a vector."""
    return wp.mat33(0.0, -value[2], value[1], value[2], 0.0, -value[0], -value[1], value[0], 0.0)


@wp.func
def tet_stable_neo_hookean_hessian(
    deformation: wp.mat33,
    rest_volume: float,
    k_mu: float,
    k_lambda: float,
    activation: float,
) -> mat99f:
    """Evaluate the full LOX stable Neo-Hookean Hessian.

    The determinant Hessian couples each pair of deformation columns through the cross-product
    matrix of the third column.
    """
    columns = wp.transpose(deformation)
    cofactor = _column_major(
        wp.matrix_from_cols(
            wp.cross(columns[1], columns[2]), wp.cross(columns[2], columns[0]), wp.cross(columns[0], columns[1])
        )
    )
    hessian = rest_volume * (k_mu * wp.identity(n=9, dtype=float) + (k_lambda + k_mu) * wp.outer(cofactor, cofactor))
    pressure = rest_volume * _determinant_pressure(deformation, k_mu, k_lambda, activation)
    for first in range(3):
        second = (first + 1) % 3
        third = (first + 2) % 3
        # The block of the columns (first, second) is -[f_third]x, and its transpose the block (second, first)
        block = pressure * _skew(columns[third])
        for row in range(3):
            for column in range(3):
                hessian[3 * first + row, 3 * second + column] -= block[row, column]
                hessian[3 * second + row, 3 * first + column] += block[row, column]
    return hessian


@wp.func
def tet_stable_neo_hookean_eigensystem(
    deformation: wp.mat33,
    rest_volume: float,
    k_mu: float,
    k_lambda: float,
    activation: float,
):
    """Return the eigenvalues and the eigenvectors, as rows, of the stable Neo-Hookean Hessian.

    The first three modes scale the singular values; modes ``3 + k`` and ``6 + k`` twist and flip the
    two axes other than ``k``, with eigenvalues ``mu V + p V s_k`` and ``mu V - p V s_k``, where ``p`` is
    the determinant pressure.
    """
    left = wp.mat33()
    singular = wp.vec3()
    right = wp.mat33()
    wp.svd3(deformation, left, singular, right)
    right_transpose = wp.transpose(right)
    shear = rest_volume * k_mu
    volume = rest_volume * (k_lambda + k_mu)
    pressure = rest_volume * _determinant_pressure(deformation, k_mu, k_lambda, activation)
    cross = wp.vec3(singular[1] * singular[2], singular[0] * singular[2], singular[0] * singular[1])
    off_diagonal = wp.mat33(0.0, singular[2], singular[1], singular[2], 0.0, singular[0], singular[1], singular[0], 0.0)
    scaling = shear * wp.identity(n=3, dtype=float) + volume * wp.outer(cross, cross) + pressure * off_diagonal
    scaling_vectors = wp.mat33()
    scaling_values = wp.vec3()
    wp.eig3(scaling, scaling_vectors, scaling_values)

    eigenvalues = vec9f(0.0)
    eigenvectors = mat99f(0.0)
    half_root = wp.sqrt(0.5)
    for k in range(3):
        coefficients = wp.mat33(0.0)
        for axis in range(3):
            coefficients[axis, axis] = scaling_vectors[axis, k]
        scaling_mode = _column_major(left * coefficients * right_transpose)

        first = (k + 1) % 3
        second = (k + 2) % 3
        twist = wp.mat33(0.0)
        twist[first, second] = half_root
        twist[second, first] = -half_root
        twist_mode = _column_major(left * twist * right_transpose)
        flip = wp.mat33(0.0)
        flip[first, second] = half_root
        flip[second, first] = half_root
        flip_mode = _column_major(left * flip * right_transpose)

        eigenvalues[k] = scaling_values[k]
        eigenvalues[3 + k] = shear + pressure * singular[k]
        eigenvalues[6 + k] = shear - pressure * singular[k]
        for coordinate in range(9):
            eigenvectors[k, coordinate] = scaling_mode[coordinate]
            eigenvectors[3 + k, coordinate] = twist_mode[coordinate]
            eigenvectors[6 + k, coordinate] = flip_mode[coordinate]
    return eigenvalues, eigenvectors


@wp.func
def tet_stable_neo_hookean_spectral_metrics(
    deformation: wp.mat33,
    rest_volume: float,
    k_mu: float,
    k_lambda: float,
    activation: float,
    minimum_eigenvalue: float,
    negative_curvature_margin: float,
) -> TetrahedronSpectralMetrics:
    """Build projected and majorizing metrics from the semi-analytic eigensystem."""
    result = TetrahedronSpectralMetrics()
    eigenvalues, eigenvectors = tet_stable_neo_hookean_eigensystem(deformation, rest_volume, k_mu, k_lambda, activation)
    curvature_margin = wp.max(0.0, negative_curvature_margin)
    scale = float(0.0)
    decomposition_is_finite = wp.bool(True)
    for mode in range(9):
        scale = wp.max(scale, wp.abs(eigenvalues[mode]))
        decomposition_is_finite = decomposition_is_finite and wp.isfinite(eigenvalues[mode])
        for coordinate in range(9):
            decomposition_is_finite = decomposition_is_finite and wp.isfinite(eigenvectors[mode, coordinate])
    if not decomposition_is_finite:
        _stress, gauss_newton = tet_stable_neo_hookean_differential(
            deformation,
            rest_volume,
            k_mu,
            k_lambda,
            activation,
        )
        gauss_newton += wp.max(0.0, minimum_eigenvalue - rest_volume * k_mu) * wp.identity(n=9, dtype=float)
        result.projected = gauss_newton
        hessian = tet_stable_neo_hookean_hessian(deformation, rest_volume, k_mu, k_lambda, activation)
        spectral_bound = float(0.0)
        hessian_is_finite = wp.bool(True)
        for row in range(9):
            row_sum = float(0.0)
            for column in range(9):
                row_sum += wp.abs(hessian[row, column])
                hessian_is_finite = hessian_is_finite and wp.isfinite(hessian[row, column])
            spectral_bound = wp.max(spectral_bound, row_sum)
        if not hessian_is_finite:
            spectral_bound = wp.max(0.0, rest_volume * k_mu)
        result.majorizer = (1.0 + curvature_margin) * spectral_bound * wp.identity(n=9, dtype=float)
        return result

    tolerance = _EIGEN_TOLERANCE * scale
    negative_curvature = float(0.0)
    for mode in range(9):
        if eigenvalues[mode] < -tolerance:
            negative_curvature = wp.max(negative_curvature, -eigenvalues[mode])

    projected = mat99f(0.0)
    majorizer = mat99f(0.0)
    for mode in range(9):
        projected_eigenvalue = wp.max(eigenvalues[mode], minimum_eigenvalue)
        metric_eigenvalue = curvature_margin * negative_curvature
        if wp.abs(eigenvalues[mode]) > tolerance:
            metric_eigenvalue += wp.abs(eigenvalues[mode])
        for row in range(9):
            for column in range(9):
                product = eigenvectors[mode, row] * eigenvectors[mode, column]
                projected[row, column] += projected_eigenvalue * product
                majorizer[row, column] += metric_eigenvalue * product
    result.projected = projected
    result.majorizer = majorizer
    return result
