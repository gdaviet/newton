# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""LOX-only spectral operations for stable Neo-Hookean tetrahedra."""

import warp as wp
from warp.fem.linalg import symmetric_eigenvalues_qr

from .deformable_energy import (
    mat99,
    tet_cofactor,
    tet_stable_neo_hookean_differential,
    vec9,
)

# Keep fixed-size spectral reconstruction loops compact in callers.
wp.set_module_options({"enable_backward": False, "max_unroll": 4})

_EIGEN_TOLERANCE = 1.0e-6


@wp.struct
class TetrahedronSpectralMetrics:
    """Projected and majorizing metrics sharing one eigendecomposition."""

    projected: mat99
    majorizer: mat99


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
def _levi_civita_3(first: int, second: int, third: int) -> float:
    if first == second or first == third or second == third:
        return 0.0
    if (
        (first == 0 and second == 1 and third == 2)
        or (first == 1 and second == 2 and third == 0)
        or (first == 2 and second == 0 and third == 1)
    ):
        return 1.0
    return -1.0


@wp.func
def tet_stable_neo_hookean_hessian(
    deformation: wp.mat33,
    rest_volume: float,
    k_mu: float,
    k_lambda: float,
    activation: float,
) -> mat99:
    """Evaluate the full LOX stable Neo-Hookean Hessian."""
    cofactor = tet_cofactor(deformation)
    cofactor_vector = vec9(
        cofactor[0, 0],
        cofactor[1, 0],
        cofactor[2, 0],
        cofactor[0, 1],
        cofactor[1, 1],
        cofactor[2, 1],
        cofactor[0, 2],
        cofactor[1, 2],
        cofactor[2, 2],
    )
    lambda_hat = k_lambda + k_mu
    hessian = rest_volume * (
        k_mu * wp.identity(n=9, dtype=float) + lambda_hat * wp.outer(cofactor_vector, cofactor_vector)
    )

    determinant_hessian_scale = rest_volume * _determinant_pressure(
        deformation,
        k_mu,
        k_lambda,
        activation,
    )
    for first in range(9):
        first_row = first % 3
        first_column = first // 3
        for second in range(9):
            second_row = second % 3
            second_column = second // 3
            determinant_hessian = 0.0
            for remaining_row in range(3):
                row_sign = _levi_civita_3(first_row, second_row, remaining_row)
                for remaining_column in range(3):
                    determinant_hessian += (
                        row_sign
                        * _levi_civita_3(first_column, second_column, remaining_column)
                        * deformation[remaining_row, remaining_column]
                    )
            hessian[first, second] += determinant_hessian_scale * determinant_hessian
    return hessian


@wp.func
def _scaled_eigendecomposition(hessian: mat99):
    hessian_scale = float(0.0)
    hessian_is_finite = wp.bool(True)
    for row in range(9):
        for column in range(9):
            hessian_scale = wp.max(hessian_scale, wp.abs(hessian[row, column]))
            hessian_is_finite = hessian_is_finite and wp.isfinite(hessian[row, column])

    eigenvalues = vec9(0.0)
    eigenvectors = mat99(0.0)
    decomposition_is_finite = wp.bool(False)
    if hessian_is_finite and hessian_scale != 0.0:
        eigenvalues, eigenvectors = symmetric_eigenvalues_qr(
            hessian / hessian_scale,
            _EIGEN_TOLERANCE,
        )
        decomposition_is_finite = wp.bool(True)
        for mode in range(9):
            decomposition_is_finite = decomposition_is_finite and wp.isfinite(eigenvalues[mode])
            for coordinate in range(9):
                decomposition_is_finite = decomposition_is_finite and wp.isfinite(eigenvectors[mode, coordinate])
    return hessian_scale, eigenvalues, eigenvectors, decomposition_is_finite


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
    """Build projected and majorizing metrics from one scaled eigensolve."""
    result = TetrahedronSpectralMetrics()
    hessian = tet_stable_neo_hookean_hessian(
        deformation,
        rest_volume,
        k_mu,
        k_lambda,
        activation,
    )
    hessian_scale, eigenvalues, eigenvectors, decomposition_is_finite = _scaled_eigendecomposition(hessian)
    curvature_margin = wp.max(0.0, negative_curvature_margin)
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

    negative_curvature = float(0.0)
    for mode in range(9):
        if eigenvalues[mode] < -_EIGEN_TOLERANCE:
            negative_curvature = wp.max(negative_curvature, -hessian_scale * eigenvalues[mode])

    projected = mat99(0.0)
    majorizer = mat99(0.0)
    for row in range(9):
        for column in range(9):
            projected_value = float(0.0)
            majorizer_value = float(0.0)
            for mode in range(9):
                normalized_eigenvalue = eigenvalues[mode]
                mode_product = eigenvectors[mode, row] * eigenvectors[mode, column]
                projected_value += mode_product * wp.max(
                    hessian_scale * normalized_eigenvalue,
                    minimum_eigenvalue,
                )
                metric_eigenvalue = curvature_margin * negative_curvature
                if normalized_eigenvalue > _EIGEN_TOLERANCE:
                    metric_eigenvalue += hessian_scale * normalized_eigenvalue
                elif normalized_eigenvalue < -_EIGEN_TOLERANCE:
                    metric_eigenvalue -= hessian_scale * normalized_eigenvalue
                majorizer_value += mode_product * metric_eigenvalue
            projected[row, column] = projected_value
            majorizer[row, column] = majorizer_value
    result.projected = projected
    result.majorizer = majorizer
    return result
