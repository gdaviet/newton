# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tetrahedral deformable energy differentials used by LOX."""

import warp as wp


class mat99(wp.types.matrix(shape=(9, 9), dtype=wp.float32)):
    pass


class vec9(wp.types.vector(length=9, dtype=wp.float32)):
    pass


@wp.func
def tet_cofactor(deformation: wp.mat33) -> wp.mat33:
    """Compute a deformation cofactor without an inverse."""
    f11, f21, f31 = deformation[0, 0], deformation[1, 0], deformation[2, 0]
    f12, f22, f32 = deformation[0, 1], deformation[1, 1], deformation[2, 1]
    f13, f23, f33 = deformation[0, 2], deformation[1, 2], deformation[2, 2]

    return wp.mat33(
        f22 * f33 - f23 * f32,
        f23 * f31 - f21 * f33,
        f21 * f32 - f22 * f31,
        f13 * f32 - f12 * f33,
        f11 * f33 - f13 * f31,
        f12 * f31 - f11 * f32,
        f12 * f23 - f13 * f22,
        f13 * f21 - f11 * f23,
        f11 * f22 - f12 * f21,
    )


@wp.func
def tet_vertex_coefficient(vertex_order: int, rest_pose: wp.mat33) -> wp.vec3:
    """Return the rest-coordinate coefficient for one tet vertex."""
    if vertex_order == 0:
        return wp.vec3(
            -(rest_pose[0, 0] + rest_pose[1, 0] + rest_pose[2, 0]),
            -(rest_pose[0, 1] + rest_pose[1, 1] + rest_pose[2, 1]),
            -(rest_pose[0, 2] + rest_pose[1, 2] + rest_pose[2, 2]),
        )
    if vertex_order == 1:
        return wp.vec3(rest_pose[0, 0], rest_pose[0, 1], rest_pose[0, 2])
    if vertex_order == 2:
        return wp.vec3(rest_pose[1, 0], rest_pose[1, 1], rest_pose[1, 2])
    return wp.vec3(rest_pose[2, 0], rest_pose[2, 1], rest_pose[2, 2])


@wp.func
def tet_stable_neo_hookean_differential(
    deformation: wp.mat33,
    rest_volume: float,
    k_mu: float,
    k_lambda: float,
    activation: float,
):
    """Evaluate the stable Neo-Hookean stress and PSD tangent."""
    deformation_vector = vec9(
        deformation[0, 0],
        deformation[1, 0],
        deformation[2, 0],
        deformation[0, 1],
        deformation[1, 1],
        deformation[2, 1],
        deformation[0, 2],
        deformation[1, 2],
        deformation[2, 2],
    )
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
    alpha = 1.0
    if lambda_hat > 0.0:
        alpha = 1.0 + k_mu / wp.max(lambda_hat, 1.0e-6)
    elif lambda_hat < 0.0:
        alpha = 1.0 - k_mu / wp.max(-lambda_hat, 1.0e-6)
    constraint = wp.determinant(deformation) - alpha + activation
    stress = rest_volume * (k_mu * deformation_vector + lambda_hat * constraint * cofactor_vector)
    tangent = rest_volume * (
        k_mu * wp.identity(n=9, dtype=float) + lambda_hat * wp.outer(cofactor_vector, cofactor_vector)
    )
    return stress, tangent


@wp.func
def contract_tet_force(stress: vec9, coefficient: wp.vec3) -> wp.vec3:
    """Contract deformation-space stress into one nodal force."""
    return wp.vec3(
        -(stress[0] * coefficient[0] + stress[3] * coefficient[1] + stress[6] * coefficient[2]),
        -(stress[1] * coefficient[0] + stress[4] * coefficient[1] + stress[7] * coefficient[2]),
        -(stress[2] * coefficient[0] + stress[5] * coefficient[1] + stress[8] * coefficient[2]),
    )


@wp.func
def contract_tet_pair_tangent(
    tangent: mat99,
    row_coefficient: wp.vec3,
    column_coefficient: wp.vec3,
) -> wp.mat33:
    """Contract a deformation-space tangent into one nodal pair block."""
    block = wp.mat33(0.0)
    for row_axis in range(3):
        for column_axis in range(3):
            value = 0.0
            for deformation_row in range(3):
                for deformation_column in range(3):
                    value += (
                        row_coefficient[deformation_row]
                        * tangent[3 * deformation_row + row_axis, 3 * deformation_column + column_axis]
                        * column_coefficient[deformation_column]
                    )
            block[row_axis, column_axis] = value
    return block


@wp.func
def _tet_metric_derivatives(
    deformation: wp.mat33,
    coefficient: wp.vec3,
):
    f0 = wp.vec3(deformation[0, 0], deformation[1, 0], deformation[2, 0])
    f1 = wp.vec3(deformation[0, 1], deformation[1, 1], deformation[2, 1])
    f2 = wp.vec3(deformation[0, 2], deformation[1, 2], deformation[2, 2])
    dc00 = 2.0 * coefficient[0] * f0
    dc01 = coefficient[0] * f1 + coefficient[1] * f0
    dc02 = coefficient[0] * f2 + coefficient[2] * f0
    dc11 = 2.0 * coefficient[1] * f1
    dc12 = coefficient[1] * f2 + coefficient[2] * f1
    dc22 = 2.0 * coefficient[2] * f2
    return dc00, dc01, dc02, dc11, dc12, dc22


@wp.func
def tet_objective_damping_differential(
    deformation_start: wp.mat33,
    deformation: wp.mat33,
    row_coefficient: wp.vec3,
    column_coefficient: wp.vec3,
    rest_volume: float,
    damping: float,
    dt: float,
):
    """Evaluate one nodal damping force and pair-block PSD tangent."""
    inverse_dt = 1.0 / dt
    f0 = wp.vec3(deformation[0, 0], deformation[1, 0], deformation[2, 0])
    f1 = wp.vec3(deformation[0, 1], deformation[1, 1], deformation[2, 1])
    f2 = wp.vec3(deformation[0, 2], deformation[1, 2], deformation[2, 2])
    f0_start = wp.vec3(deformation_start[0, 0], deformation_start[1, 0], deformation_start[2, 0])
    f1_start = wp.vec3(deformation_start[0, 1], deformation_start[1, 1], deformation_start[2, 1])
    f2_start = wp.vec3(deformation_start[0, 2], deformation_start[1, 2], deformation_start[2, 2])
    c00_rate = (wp.dot(f0, f0) - wp.dot(f0_start, f0_start)) * inverse_dt
    c01_rate = (wp.dot(f0, f1) - wp.dot(f0_start, f1_start)) * inverse_dt
    c02_rate = (wp.dot(f0, f2) - wp.dot(f0_start, f2_start)) * inverse_dt
    c11_rate = (wp.dot(f1, f1) - wp.dot(f1_start, f1_start)) * inverse_dt
    c12_rate = (wp.dot(f1, f2) - wp.dot(f1_start, f2_start)) * inverse_dt
    c22_rate = (wp.dot(f2, f2) - wp.dot(f2_start, f2_start)) * inverse_dt

    dr00, dr01, dr02, dr11, dr12, dr22 = _tet_metric_derivatives(deformation, row_coefficient)
    dc00, dc01, dc02, dc11, dc12, dc22 = _tet_metric_derivatives(deformation, column_coefficient)
    force = (
        -rest_volume
        * damping
        * (
            c00_rate * dr00
            + 2.0 * c01_rate * dr01
            + 2.0 * c02_rate * dr02
            + c11_rate * dr11
            + 2.0 * c12_rate * dr12
            + c22_rate * dr22
        )
    )
    tangent = (
        rest_volume
        * damping
        * inverse_dt
        * (
            wp.outer(dr00, dc00)
            + 2.0 * wp.outer(dr01, dc01)
            + 2.0 * wp.outer(dr02, dc02)
            + wp.outer(dr11, dc11)
            + 2.0 * wp.outer(dr12, dc12)
            + wp.outer(dr22, dc22)
        )
    )
    return force, tangent
