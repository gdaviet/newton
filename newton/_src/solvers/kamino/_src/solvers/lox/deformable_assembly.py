# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU kernels for the frozen LOX deformable smooth system."""

import warp as wp

from .deformable_energy import (
    contract_tet_force,
    contract_tet_pair_tangent,
    mat99,
    tet_objective_damping_differential,
    tet_stable_neo_hookean_differential,
    tet_vertex_coefficient,
)
from .deformable_tetrahedron_energy import tet_stable_neo_hookean_spectral_metrics

# Keep nested 9-by-9 eigensolver loops from expanding into multi-megabyte kernels.
wp.set_module_options({"enable_backward": False, "max_unroll": 4})

PARTICLE_FLAG_ACTIVE = 1
_GEOMETRY_EPSILON = 1.0e-6
# Add an isotropic margin only when the frozen Hessian has negative curvature.
_TETRAHEDRON_NEGATIVE_CURVATURE_MARGIN = 32.0

DEFORMABLE_WEIGHT_STATUS_INVALID = 0
DEFORMABLE_WEIGHT_STATUS_VALID = 1
DEFORMABLE_WEIGHT_STATUS_REGULARIZED = 2


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
def _triangle_coefficients(order: int, rest_pose: wp.mat22) -> wp.vec2:
    if order == 0:
        return wp.vec2(
            -rest_pose[0, 0] - rest_pose[1, 0],
            -rest_pose[0, 1] - rest_pose[1, 1],
        )
    if order == 1:
        return wp.vec2(rest_pose[0, 0], rest_pose[0, 1])
    return wp.vec2(rest_pose[1, 0], rest_pose[1, 1])


@wp.func
def _triangle_metric_derivatives(
    deformation_0: wp.vec3,
    deformation_1: wp.vec3,
    coefficients: wp.vec2,
):
    dc00 = 2.0 * coefficients[0] * deformation_0
    dc01 = coefficients[0] * deformation_1 + coefficients[1] * deformation_0
    dc11 = 2.0 * coefficients[1] * deformation_1
    return dc00, dc01, dc11


@wp.func
def _normalized_vector_derivative(
    vector_length: float,
    normalized_vector: wp.vec3,
    vector_derivative: wp.mat33,
) -> wp.mat33:
    projection = wp.identity(n=3, dtype=float) - wp.outer(normalized_vector, normalized_vector)
    return (1.0 / vector_length) * projection * vector_derivative


@wp.func
def _angle_derivative(
    normal_0: wp.vec3,
    normal_1: wp.vec3,
    edge: wp.vec3,
    normal_0_derivative: wp.mat33,
    normal_1_derivative: wp.mat33,
    sin_angle: float,
    cos_angle: float,
) -> wp.vec3:
    skew_normal_0 = wp.skew(normal_0)
    skew_normal_1 = wp.skew(normal_1)
    dsin = wp.transpose(skew_normal_0 * normal_1_derivative - skew_normal_1 * normal_0_derivative) * edge
    dcos = wp.transpose(normal_0_derivative) * normal_1 + wp.transpose(normal_1_derivative) * normal_0
    return dsin * cos_angle - dcos * sin_angle


@wp.func
def _select_gradient(
    order: int,
    gradient_0: wp.vec3,
    gradient_1: wp.vec3,
    gradient_2: wp.vec3,
    gradient_3: wp.vec3,
) -> wp.vec3:
    if order == 0:
        return gradient_0
    if order == 1:
        return gradient_1
    if order == 2:
        return gradient_2
    return gradient_3


@wp.kernel
def gather_particle_state(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    packed_to_newton: wp.array[wp.int32],
    packed_source_world: wp.array[wp.int32],
    position_start: wp.array[wp.vec3],
    velocity_start: wp.array[wp.vec3],
    external_force: wp.array[wp.vec3],
):
    """Gather step-start Newton particle data into the packed deformable layout."""
    packed_particle = wp.tid()
    particle = packed_to_newton[packed_particle]
    position_start[packed_particle] = particle_q[particle]
    velocity_start[packed_particle] = particle_qd[particle]

    if _particle_is_dynamic(packed_particle, packed_to_newton, particle_mass, particle_flags):
        world = packed_source_world[packed_particle]
        external_force[packed_particle] = particle_f[particle] + particle_mass[particle] * gravity[world]
    else:
        external_force[packed_particle] = wp.vec3(0.0)


@wp.kernel
def set_particle_linearization(
    position_start: wp.array[wp.vec3],
    linearization_velocity: wp.array[wp.vec3],
    external_force: wp.array[wp.vec3],
    packed_to_newton: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    particle_flags: wp.array[wp.int32],
    time_step: wp.array[wp.float32],
    position_linearized: wp.array[wp.vec3],
    velocity_linearized: wp.array[wp.vec3],
    smooth_force: wp.array[wp.vec3],
):
    """Set the packed deformable linearization while restoring external forces."""
    packed_particle = wp.tid()
    particle = packed_to_newton[packed_particle]
    dt = time_step[packed_world[packed_particle]]
    velocity = linearization_velocity[packed_particle]
    if (particle_flags[particle] & PARTICLE_FLAG_ACTIVE) == 0:
        velocity = wp.vec3(0.0)
    position_linearized[packed_particle] = position_start[packed_particle] + dt * velocity
    velocity_linearized[packed_particle] = velocity
    smooth_force[packed_particle] = external_force[packed_particle]


@wp.kernel
def assemble_particle_mass(
    packed_to_newton: wp.array[wp.int32],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    triplet_values: wp.array[wp.mat33],
):
    """Write mass or identity blocks into the particle diagonal triplets."""
    packed_particle = wp.tid()
    diagonal = 1.0
    if _particle_is_dynamic(packed_particle, packed_to_newton, particle_mass, particle_flags):
        particle = packed_to_newton[packed_particle]
        diagonal = particle_mass[particle]
    triplet_values[packed_particle] = diagonal * wp.identity(n=3, dtype=float)


@wp.kernel
def assemble_triangle_system(
    position_start: wp.array[wp.vec3],
    position_linearized: wp.array[wp.vec3],
    velocity_linearized: wp.array[wp.vec3],
    triangle_indices: wp.array2d[wp.int32],
    triangle_poses: wp.array[wp.mat22],
    triangle_areas: wp.array[float],
    triangle_materials: wp.array2d[float],
    triangle_activations: wp.array[float],
    packed_to_newton: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    time_step: wp.array[wp.float32],
    triplet_offset: int,
    triplet_values: wp.array[wp.mat33],
    smooth_force: wp.array[wp.vec3],
):
    """Assemble stable membrane forces and PSD pair-block tangents."""
    triangle = wp.tid()
    vertex_0 = triangle_indices[triangle, 0]
    vertex_1 = triangle_indices[triangle, 1]
    vertex_2 = triangle_indices[triangle, 2]
    dt = time_step[packed_world[vertex_0]]

    x_0 = position_linearized[vertex_0]
    x_1 = position_linearized[vertex_1]
    x_2 = position_linearized[vertex_2]
    x_10 = x_1 - x_0
    x_20 = x_2 - x_0
    rest_pose = triangle_poses[triangle]

    deformation_0 = x_10 * rest_pose[0, 0] + x_20 * rest_pose[1, 0]
    deformation_1 = x_10 * rest_pose[0, 1] + x_20 * rest_pose[1, 1]

    f00 = wp.dot(deformation_0, deformation_0)
    f11 = wp.dot(deformation_1, deformation_1)
    f01 = wp.dot(deformation_0, deformation_1)
    area_ratio_squared = wp.max(f00 * f11 - f01 * f01, 1.0e-20)
    area_ratio = wp.sqrt(area_ratio_squared)
    inverse_area_ratio = 1.0 / area_ratio
    area_gradient_0 = inverse_area_ratio * (f11 * deformation_0 - f01 * deformation_1)
    area_gradient_1 = inverse_area_ratio * (f00 * deformation_1 - f01 * deformation_0)

    mu = triangle_materials[triangle, 0]
    lmbd = triangle_materials[triangle, 1] + mu
    damping = triangle_materials[triangle, 2]
    alpha = 1.0
    if lmbd > 1.0e-6:
        alpha = 1.0 + mu / lmbd
    area_constraint = area_ratio - alpha + triangle_activations[triangle]

    stress_0 = mu * deformation_0 + lmbd * area_constraint * area_gradient_0
    stress_1 = mu * deformation_1 + lmbd * area_constraint * area_gradient_1
    rest_area = triangle_areas[triangle]

    x0_0 = position_start[vertex_0]
    x0_1 = position_start[vertex_1]
    x0_2 = position_start[vertex_2]
    x0_10 = x0_1 - x0_0
    x0_20 = x0_2 - x0_0
    deformation_start_0 = x0_10 * rest_pose[0, 0] + x0_20 * rest_pose[1, 0]
    deformation_start_1 = x0_10 * rest_pose[0, 1] + x0_20 * rest_pose[1, 1]
    inverse_dt = 1.0 / dt
    c00_rate = (f00 - wp.dot(deformation_start_0, deformation_start_0)) * inverse_dt
    c01_rate = (f01 - wp.dot(deformation_start_0, deformation_start_1)) * inverse_dt
    c11_rate = (f11 - wp.dot(deformation_start_1, deformation_start_1)) * inverse_dt

    for row in range(3):
        row_vertex = triangle_indices[triangle, row]
        row_coefficients = _triangle_coefficients(row, rest_pose)
        area_derivative = area_gradient_0 * row_coefficients[0] + area_gradient_1 * row_coefficients[1]
        force = -(stress_0 * row_coefficients[0] + stress_1 * row_coefficients[1])

        dc00_row, dc01_row, dc11_row = _triangle_metric_derivatives(
            deformation_0,
            deformation_1,
            row_coefficients,
        )
        if damping > 0.0:
            force -= damping * (c00_rate * dc00_row + 2.0 * c01_rate * dc01_row + c11_rate * dc11_row)

        if _particle_is_dynamic(row_vertex, packed_to_newton, particle_mass, particle_flags):
            wp.atomic_add(smooth_force, row_vertex, rest_area * force)

        for column in range(3):
            column_vertex = triangle_indices[triangle, column]
            block = wp.mat33(0.0)
            if _particle_is_dynamic(
                row_vertex, packed_to_newton, particle_mass, particle_flags
            ) and _particle_is_dynamic(column_vertex, packed_to_newton, particle_mass, particle_flags):
                column_coefficients = _triangle_coefficients(column, rest_pose)
                area_derivative_column = (
                    area_gradient_0 * column_coefficients[0] + area_gradient_1 * column_coefficients[1]
                )
                identity_scale = mu * (
                    row_coefficients[0] * column_coefficients[0] + row_coefficients[1] * column_coefficients[1]
                )
                block = identity_scale * wp.identity(n=3, dtype=float)
                block += lmbd * wp.outer(area_derivative, area_derivative_column)

                if damping > 0.0:
                    dc00_column, dc01_column, dc11_column = _triangle_metric_derivatives(
                        deformation_0,
                        deformation_1,
                        column_coefficients,
                    )
                    block += (
                        damping
                        * inverse_dt
                        * (
                            wp.outer(dc00_row, dc00_column)
                            + 2.0 * wp.outer(dc01_row, dc01_column)
                            + wp.outer(dc11_row, dc11_column)
                        )
                    )
                block *= rest_area * dt * dt

            triplet_values[triplet_offset + triangle * 9 + row * 3 + column] = block

    velocity_midpoint = (
        velocity_linearized[vertex_0] + velocity_linearized[vertex_1] + velocity_linearized[vertex_2]
    ) / 3.0
    normal_raw = wp.cross(x_10, x_20)
    current_area = 0.5 * wp.length(normal_raw)
    normal = wp.normalize(normal_raw)
    velocity_direction = wp.normalize(velocity_midpoint)
    drag = velocity_midpoint * (
        triangle_materials[triangle, 3] * current_area * wp.abs(wp.dot(normal, velocity_midpoint))
    )
    lift_angle = wp.HALF_PI - wp.acos(wp.clamp(wp.dot(normal, velocity_direction), -1.0, 1.0))
    lift = normal * (
        triangle_materials[triangle, 4] * current_area * lift_angle * wp.dot(velocity_midpoint, velocity_midpoint)
    )
    aerodynamic_force = -(drag + lift)
    for order in range(3):
        vertex = triangle_indices[triangle, order]
        if _particle_is_dynamic(vertex, packed_to_newton, particle_mass, particle_flags):
            wp.atomic_add(smooth_force, vertex, aerodynamic_force)


@wp.kernel
def assemble_tetrahedron_system(
    position_start: wp.array[wp.vec3],
    position_linearized: wp.array[wp.vec3],
    tetrahedron_indices: wp.array2d[wp.int32],
    tetrahedron_poses: wp.array[wp.mat33],
    tetrahedron_materials: wp.array2d[float],
    tetrahedron_activations: wp.array[float],
    packed_to_newton: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    time_step: wp.array[wp.float32],
    triplet_offset: int,
    tetrahedron_metric: wp.array[mat99],
    triplet_values: wp.array[wp.mat33],
    smooth_force: wp.array[wp.vec3],
):
    """Assemble volumetric Neo-Hookean forces and contracted majorizing metrics."""
    tetrahedron = wp.tid()
    vertex_0 = tetrahedron_indices[tetrahedron, 0]
    vertex_1 = tetrahedron_indices[tetrahedron, 1]
    vertex_2 = tetrahedron_indices[tetrahedron, 2]
    vertex_3 = tetrahedron_indices[tetrahedron, 3]
    dt = time_step[packed_world[vertex_0]]

    position_0 = position_linearized[vertex_0]
    position_1 = position_linearized[vertex_1]
    position_2 = position_linearized[vertex_2]
    position_3 = position_linearized[vertex_3]
    rest_pose = tetrahedron_poses[tetrahedron]
    deformation = (
        wp.matrix_from_cols(
            position_1 - position_0,
            position_2 - position_0,
            position_3 - position_0,
        )
        * rest_pose
    )

    position_start_0 = position_start[vertex_0]
    deformation_start = (
        wp.matrix_from_cols(
            position_start[vertex_1] - position_start_0,
            position_start[vertex_2] - position_start_0,
            position_start[vertex_3] - position_start_0,
        )
        * rest_pose
    )

    rest_volume = 1.0 / (6.0 * wp.determinant(rest_pose))
    k_mu = tetrahedron_materials[tetrahedron, 0]
    k_lambda = tetrahedron_materials[tetrahedron, 1]
    damping = tetrahedron_materials[tetrahedron, 2]
    stress, _gauss_newton_tangent = tet_stable_neo_hookean_differential(
        deformation,
        rest_volume,
        k_mu,
        k_lambda,
        tetrahedron_activations[tetrahedron],
    )
    spectral_metrics = tet_stable_neo_hookean_spectral_metrics(
        deformation,
        rest_volume,
        k_mu,
        k_lambda,
        tetrahedron_activations[tetrahedron],
        0.0,
        _TETRAHEDRON_NEGATIVE_CURVATURE_MARGIN,
    )
    tangent = spectral_metrics.majorizer
    tetrahedron_metric[tetrahedron] = tangent

    for row in range(4):
        row_vertex = tetrahedron_indices[tetrahedron, row]
        row_coefficient = tet_vertex_coefficient(row, rest_pose)
        force = contract_tet_force(stress, row_coefficient)
        if damping > 0.0:
            damping_force, _unused_diagonal_damping_block = tet_objective_damping_differential(
                deformation_start,
                deformation,
                row_coefficient,
                row_coefficient,
                rest_volume,
                damping,
                dt,
            )
            force += damping_force

        row_dynamic = _particle_is_dynamic(row_vertex, packed_to_newton, particle_mass, particle_flags)
        if row_dynamic:
            wp.atomic_add(smooth_force, row_vertex, force)

        for column in range(4):
            column_vertex = tetrahedron_indices[tetrahedron, column]
            block = wp.mat33(0.0)
            if row_dynamic and _particle_is_dynamic(
                column_vertex,
                packed_to_newton,
                particle_mass,
                particle_flags,
            ):
                column_coefficient = tet_vertex_coefficient(column, rest_pose)
                block = contract_tet_pair_tangent(tangent, row_coefficient, column_coefficient)
                if damping > 0.0:
                    _unused_damping_force, damping_block = tet_objective_damping_differential(
                        deformation_start,
                        deformation,
                        row_coefficient,
                        column_coefficient,
                        rest_volume,
                        damping,
                        dt,
                    )
                    block += damping_block
                block *= dt * dt
            triplet_values[triplet_offset + tetrahedron * 16 + row * 4 + column] = block


@wp.kernel
def assemble_bending_system(
    position_start: wp.array[wp.vec3],
    position_linearized: wp.array[wp.vec3],
    bending_indices: wp.array2d[wp.int32],
    source_edge_indices: wp.array[wp.int32],
    edge_rest_angle: wp.array[float],
    edge_rest_length: wp.array[float],
    edge_bending_properties: wp.array2d[float],
    packed_to_newton: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    time_step: wp.array[wp.float32],
    triplet_offset: int,
    triplet_values: wp.array[wp.mat33],
    smooth_force: wp.array[wp.vec3],
):
    """Assemble dihedral bending forces and Gauss-Newton pair blocks."""
    bending = wp.tid()
    source_edge = source_edge_indices[bending]
    vertex_0 = bending_indices[bending, 0]
    vertex_1 = bending_indices[bending, 1]
    vertex_2 = bending_indices[bending, 2]
    vertex_3 = bending_indices[bending, 3]
    dt = time_step[packed_world[vertex_0]]

    x_0 = position_linearized[vertex_0]
    x_1 = position_linearized[vertex_1]
    x_2 = position_linearized[vertex_2]
    x_3 = position_linearized[vertex_3]
    x_02 = x_2 - x_0
    x_03 = x_3 - x_0
    x_13 = x_3 - x_1
    x_12 = x_2 - x_1
    edge_vector = x_3 - x_2
    normal_raw_0 = wp.cross(x_02, x_03)
    normal_raw_1 = wp.cross(x_13, x_12)
    normal_length_0 = wp.length(normal_raw_0)
    normal_length_1 = wp.length(normal_raw_1)
    edge_length = wp.length(edge_vector)

    valid = (
        normal_length_0 >= _GEOMETRY_EPSILON
        and normal_length_1 >= _GEOMETRY_EPSILON
        and edge_length >= _GEOMETRY_EPSILON
    )
    if not valid:
        for row in range(4):
            for column in range(4):
                triplet_values[triplet_offset + bending * 16 + row * 4 + column] = wp.mat33(0.0)
        return

    normal_0 = normal_raw_0 / normal_length_0
    normal_1 = normal_raw_1 / normal_length_1
    edge = edge_vector / edge_length
    sin_angle = wp.dot(wp.cross(normal_0, normal_1), edge)
    cos_angle = wp.dot(normal_0, normal_1)
    angle = wp.atan2(sin_angle, cos_angle)

    normal_0_derivative_0 = _normalized_vector_derivative(normal_length_0, normal_0, wp.skew(edge_vector))
    normal_1_derivative_0 = wp.mat33(0.0)
    normal_0_derivative_1 = wp.mat33(0.0)
    normal_1_derivative_1 = _normalized_vector_derivative(normal_length_1, normal_1, -wp.skew(edge_vector))
    normal_0_derivative_2 = _normalized_vector_derivative(normal_length_0, normal_0, -wp.skew(x_03))
    normal_1_derivative_2 = _normalized_vector_derivative(normal_length_1, normal_1, wp.skew(x_13))
    normal_0_derivative_3 = _normalized_vector_derivative(normal_length_0, normal_0, wp.skew(x_02))
    normal_1_derivative_3 = _normalized_vector_derivative(normal_length_1, normal_1, -wp.skew(x_12))

    gradient_0 = _angle_derivative(
        normal_0,
        normal_1,
        edge,
        normal_0_derivative_0,
        normal_1_derivative_0,
        sin_angle,
        cos_angle,
    )
    gradient_1 = _angle_derivative(
        normal_0,
        normal_1,
        edge,
        normal_0_derivative_1,
        normal_1_derivative_1,
        sin_angle,
        cos_angle,
    )
    gradient_2 = _angle_derivative(
        normal_0,
        normal_1,
        edge,
        normal_0_derivative_2,
        normal_1_derivative_2,
        sin_angle,
        cos_angle,
    )
    gradient_3 = _angle_derivative(
        normal_0,
        normal_1,
        edge,
        normal_0_derivative_3,
        normal_1_derivative_3,
        sin_angle,
        cos_angle,
    )

    rest_length = edge_rest_length[source_edge]
    stiffness = edge_bending_properties[source_edge, 0]
    damping = edge_bending_properties[source_edge, 1]
    elastic_scale = stiffness * rest_length
    force_scale = elastic_scale * (angle - edge_rest_angle[source_edge])
    tangent_scale = elastic_scale

    if damping > 0.0:
        x0_0 = position_start[vertex_0]
        x0_1 = position_start[vertex_1]
        x0_2 = position_start[vertex_2]
        x0_3 = position_start[vertex_3]
        normal_start_raw_0 = wp.cross(x0_2 - x0_0, x0_3 - x0_0)
        normal_start_raw_1 = wp.cross(x0_3 - x0_1, x0_2 - x0_1)
        edge_start_raw = x0_3 - x0_2
        normal_start_length_0 = wp.length(normal_start_raw_0)
        normal_start_length_1 = wp.length(normal_start_raw_1)
        edge_start_length = wp.length(edge_start_raw)
        if (
            normal_start_length_0 >= _GEOMETRY_EPSILON
            and normal_start_length_1 >= _GEOMETRY_EPSILON
            and edge_start_length >= _GEOMETRY_EPSILON
        ):
            normal_start_0 = normal_start_raw_0 / normal_start_length_0
            normal_start_1 = normal_start_raw_1 / normal_start_length_1
            edge_start = edge_start_raw / edge_start_length
            angle_start = wp.atan2(
                wp.dot(wp.cross(normal_start_0, normal_start_1), edge_start),
                wp.dot(normal_start_0, normal_start_1),
            )
            angle_delta = angle - angle_start
            if angle_delta > wp.pi:
                angle_delta -= 2.0 * wp.pi
            elif angle_delta < -wp.pi:
                angle_delta += 2.0 * wp.pi
            force_scale += damping * rest_length * angle_delta / dt
            tangent_scale += damping * rest_length / dt

    for row in range(4):
        row_vertex = bending_indices[bending, row]
        row_gradient = _select_gradient(row, gradient_0, gradient_1, gradient_2, gradient_3)
        if _particle_is_dynamic(row_vertex, packed_to_newton, particle_mass, particle_flags):
            wp.atomic_add(smooth_force, row_vertex, -force_scale * row_gradient)

        for column in range(4):
            column_vertex = bending_indices[bending, column]
            block = wp.mat33(0.0)
            if _particle_is_dynamic(
                row_vertex, packed_to_newton, particle_mass, particle_flags
            ) and _particle_is_dynamic(column_vertex, packed_to_newton, particle_mass, particle_flags):
                column_gradient = _select_gradient(column, gradient_0, gradient_1, gradient_2, gradient_3)
                block = dt * dt * tangent_scale * wp.outer(row_gradient, column_gradient)
            triplet_values[triplet_offset + bending * 16 + row * 4 + column] = block


@wp.kernel
def finish_smooth_rhs(
    matrix_velocity: wp.array[wp.vec3],
    smooth_force: wp.array[wp.vec3],
    velocity_start: wp.array[wp.vec3],
    velocity_linearized: wp.array[wp.vec3],
    packed_to_newton: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    time_step: wp.array[wp.float32],
    smooth_rhs: wp.array[wp.vec3],
):
    """Finish the affine frozen-dynamics right-hand side."""
    packed_particle = wp.tid()
    dt = time_step[packed_world[packed_particle]]
    if _particle_is_dynamic(packed_particle, packed_to_newton, particle_mass, particle_flags):
        particle = packed_to_newton[packed_particle]
        inertial_correction = particle_mass[particle] * (
            velocity_start[packed_particle] - velocity_linearized[packed_particle]
        )
        smooth_rhs[packed_particle] = (
            matrix_velocity[packed_particle] + inertial_correction + dt * smooth_force[packed_particle]
        )
    else:
        smooth_rhs[packed_particle] = wp.vec3(0.0)


@wp.func
def _is_finite_mat33(value: wp.mat33) -> bool:
    finite = True
    for row in range(3):
        for column in range(3):
            finite = finite and wp.isfinite(value[row, column])
    return finite


@wp.kernel
def compute_consensus_weight(
    smooth_values: wp.array[wp.mat33],
    diagonal_slots: wp.array[wp.int32],
    packed_to_newton: wp.array[wp.int32],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    sigma: float,
    beta: float,
    weight: wp.array[float],
    inverse_weight: wp.array[float],
    weight_status: wp.array[wp.int32],
):
    """Compute one scalar consensus weight from each smooth diagonal block."""
    packed_particle = wp.tid()
    if not _particle_is_dynamic(packed_particle, packed_to_newton, particle_mass, particle_flags):
        weight[packed_particle] = 0.0
        inverse_weight[packed_particle] = 0.0
        weight_status[packed_particle] = DEFORMABLE_WEIGHT_STATUS_VALID
        return

    particle = packed_to_newton[packed_particle]
    mass = particle_mass[particle]
    diagonal = smooth_values[diagonal_slots[packed_particle]]
    if not _is_finite_mat33(diagonal):
        weight[packed_particle] = 0.0
        inverse_weight[packed_particle] = 0.0
        weight_status[packed_particle] = DEFORMABLE_WEIGHT_STATUS_INVALID
        return

    diagonal = 0.5 * (diagonal + wp.transpose(diagonal))
    eigenvectors, eigenvalues = wp.eig3(diagonal)
    finite = _is_finite_mat33(eigenvectors)
    for axis in range(3):
        finite = finite and wp.isfinite(eigenvalues[axis])
    if not finite:
        weight[packed_particle] = 0.0
        inverse_weight[packed_particle] = 0.0
        weight_status[packed_particle] = DEFORMABLE_WEIGHT_STATUS_INVALID
        return

    eta = wp.min(eigenvalues)
    eta_floor = wp.max(1.0e-8, 1.0e-6 * mass)
    status = wp.int32(DEFORMABLE_WEIGHT_STATUS_VALID)
    if eta < eta_floor:
        eta = eta_floor
        status = DEFORMABLE_WEIGHT_STATUS_REGULARIZED
    particle_weight = wp.max(sigma * eta, wp.min(beta * mass, eta))
    if not wp.isfinite(particle_weight) or particle_weight <= 0.0:
        weight[packed_particle] = 0.0
        inverse_weight[packed_particle] = 0.0
        weight_status[packed_particle] = DEFORMABLE_WEIGHT_STATUS_INVALID
        return

    weight[packed_particle] = particle_weight
    inverse_weight[packed_particle] = 1.0 / particle_weight
    weight_status[packed_particle] = status


@wp.kernel
def select_consensus_weight(
    full_weight: wp.array[float],
    full_inverse_weight: wp.array[float],
    unilateral_incidence: wp.array[wp.int32],
    selective: bool,
    consensus_enabled: wp.array[wp.int32],
    weight: wp.array[float],
    inverse_weight: wp.array[float],
):
    """Restrict consensus weights to dynamic nodes in the selected support."""
    particle = wp.tid()
    enabled = full_inverse_weight[particle] > 0.0 and (not selective or unilateral_incidence[particle] > 0)
    consensus_enabled[particle] = wp.where(enabled, 1, 0)
    weight[particle] = wp.where(enabled, full_weight[particle], 0.0)
    inverse_weight[particle] = wp.where(enabled, full_inverse_weight[particle], 0.0)


@wp.kernel
def add_consensus_weight(
    diagonal_slots: wp.array[wp.int32],
    weight: wp.array[float],
    system_values: wp.array[wp.mat33],
):
    """Add scalar consensus weights to system diagonal blocks."""
    packed_particle = wp.tid()
    diagonal_slot = diagonal_slots[packed_particle]
    system_values[diagonal_slot] += weight[packed_particle] * wp.identity(n=3, dtype=float)


@wp.kernel
def prepare_candidate_rhs(
    smooth_rhs: wp.array[wp.vec3],
    nonlinear_rhs: wp.array[wp.vec3],
    consensus_center: wp.array[wp.vec3],
    weight: wp.array[float],
    current_velocity: wp.array[wp.vec3],
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    candidate_rhs: wp.array[wp.vec3],
):
    """Form the LOX global candidate RHS with inactive-world retention."""
    particle = wp.tid()
    if world_active[packed_world[particle]] != 0:
        candidate_rhs[particle] = (
            smooth_rhs[particle] + nonlinear_rhs[particle] + weight[particle] * consensus_center[particle]
        )
    else:
        candidate_rhs[particle] = current_velocity[particle]


@wp.kernel
def retain_direct_candidate_rhs(
    current_velocity: wp.array[wp.vec3],
    packed_iterative: wp.array[wp.int32],
    candidate_rhs: wp.array[wp.vec3],
):
    """Replace direct-component right-hand sides for identity-masked CR batches."""
    particle = wp.tid()
    if packed_iterative[particle] == 0:
        candidate_rhs[particle] = current_velocity[particle]
