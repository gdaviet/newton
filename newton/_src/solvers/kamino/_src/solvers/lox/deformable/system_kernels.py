# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernels of the frozen LOX deformable smooth system and its candidate solves.

The assembly kernels linearize the particle system once per time step at the end-of-step
positions ``x0 + dt v_lin``: they write the particle masses and the element pair blocks into
fixed assembly triplets, scaled by ``dt^2``, and accumulate the element forces into the
smooth force. Elements write zero blocks for pairs with a prescribed particle. The
consensus kernels select and assemble the consensus weights of the particles, which
:mod:`.consensus_kernels` derives from the Schur complements of their patches of neighbors.
The candidate kernels form the right-hand side of each splitting iteration, apply
the system matrix to the conjugate-residual rows of the active worlds, and maintain the
recycled correction subspace of the conjugate-residual solve.
"""

import warp as wp

from .......geometry import ParticleFlags
from .energy import (
    contract_tet_force,
    contract_tet_pair_tangent,
    mat99f,
    tet_objective_damping_differential,
    tet_stable_neo_hookean_differential,
    tet_vertex_coefficient,
)
from .membrane import membrane_area_ratio_gradient
from .tetrahedron_energy import tet_stable_neo_hookean_spectral_metrics
from .types import RECYCLED_CR_DIRECTION_COUNT

###
# Module interface
###

__all__ = [
    "_apply_recycled_cr_basis",
    "_apply_system_matrix",
    "_assemble_bending_system",
    "_assemble_consensus_weight",
    "_assemble_particle_mass",
    "_assemble_tetrahedron_system",
    "_assemble_triangle_system",
    "_compute_candidate_linear_residual",
    "_compute_candidate_linear_residual_and_advance_recycling",
    "_compute_recycled_cr_coefficients",
    "_finalize_smooth_rhs",
    "_gather_particle_state",
    "_prepare_candidate_rhs",
    "_prepare_particle_linearization",
    "_reduce_recycled_cr_basis",
    "_retain_direct_candidate_rhs",
    "_rollback_unsafe_recycled_cr_projection",
    "_select_consensus_weight",
    "_validate_and_restart_recycled_cr_projection",
    "is_dynamic_particle",
    "triangle_vertex_coefficients",
]

###
# Module configs
###

# Keep nested 9-by-9 eigensolver loops from expanding into multi-megabyte kernels.
wp.set_module_options({"enable_backward": False, "max_unroll": 4})


###
# Constants
###


_GEOMETRY_EPSILON = 1.0e-6
"""Smallest normal and edge length of a valid bending element [m]."""


_TETRAHEDRON_NEGATIVE_CURVATURE_MARGIN = 32.0
"""Isotropic margin of the tetrahedron majorizer, relative to the negative curvature of the frozen Hessian.

The margin applies only when the frozen Hessian has negative curvature.
"""


###
# Functions
###


@wp.func
def is_dynamic_particle(
    packed_particle: int,
    packed_to_newton: wp.array[wp.int32],
    model_particle_mass: wp.array[float],
    model_particle_flags: wp.array[wp.int32],
) -> bool:
    """Return whether a packed particle is active and has a positive mass."""
    particle = packed_to_newton[packed_particle]
    return model_particle_mass[particle] > 0.0 and (model_particle_flags[particle] & ParticleFlags.ACTIVE) != 0


@wp.func
def triangle_vertex_coefficients(order: int, rest_pose: wp.mat22) -> wp.vec2:
    """Return the coefficients of one triangle vertex in the two deformation columns."""
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
    """Return the derivatives of the three metric entries ``F^T F`` with respect to one nodal position."""
    dc00 = 2.0 * coefficients[0] * deformation_0
    dc01 = coefficients[0] * deformation_1 + coefficients[1] * deformation_0
    dc11 = 2.0 * coefficients[1] * deformation_1
    return dc00, dc01, dc11


@wp.func
def _vector_is_finite(value: wp.vec3) -> bool:
    finite = True
    for axis in range(3):
        finite = finite and wp.isfinite(value[axis])
    return finite


@wp.func
def _matrix_is_finite(value: wp.mat33) -> bool:
    finite = True
    for row in range(3):
        for column in range(3):
            finite = finite and wp.isfinite(value[row, column])
    return finite


@wp.func
def _triangle_internal_force(
    coefficients: wp.vec2,
    stress_0: wp.vec3,
    stress_1: wp.vec3,
    deformation_0: wp.vec3,
    deformation_1: wp.vec3,
    damping: float,
    c00_rate: float,
    c01_rate: float,
    c11_rate: float,
    rest_area: float,
) -> wp.vec3:
    """Return the elastic and damping force of a triangle on one vertex."""
    force = -(stress_0 * coefficients[0] + stress_1 * coefficients[1])
    if damping > 0.0:
        dc00, dc01, dc11 = _triangle_metric_derivatives(
            deformation_0,
            deformation_1,
            coefficients,
        )
        force -= damping * (c00_rate * dc00 + 2.0 * c01_rate * dc01 + c11_rate * dc11)
    return rest_area * force


@wp.func
def _triangle_tangent_block(
    row_coefficients: wp.vec2,
    column_coefficients: wp.vec2,
    area_gradient_0: wp.vec3,
    area_gradient_1: wp.vec3,
    deformation_0: wp.vec3,
    deformation_1: wp.vec3,
    mu: float,
    lmbd: float,
    damping: float,
    inverse_dt: float,
    rest_area_dt_squared: float,
) -> wp.mat33:
    """Return the PSD tangent block of a triangle on one vertex pair, scaled by ``dt^2``."""
    area_derivative_row = area_gradient_0 * row_coefficients[0] + area_gradient_1 * row_coefficients[1]
    area_derivative_column = area_gradient_0 * column_coefficients[0] + area_gradient_1 * column_coefficients[1]
    identity_scale = mu * (row_coefficients[0] * column_coefficients[0] + row_coefficients[1] * column_coefficients[1])
    block = identity_scale * wp.identity(n=3, dtype=float)
    block += lmbd * wp.outer(area_derivative_row, area_derivative_column)
    if damping > 0.0:
        dc00_row, dc01_row, dc11_row = _triangle_metric_derivatives(
            deformation_0,
            deformation_1,
            row_coefficients,
        )
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
    return rest_area_dt_squared * block


@wp.func
def _normalized_vector_derivative(
    vector_length: float,
    normalized_vector: wp.vec3,
    vector_derivative: wp.mat33,
) -> wp.mat33:
    """Return the derivative of a normalized vector from the derivative of the vector."""
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
    """Return the gradient of the dihedral angle with respect to one vertex."""
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


@wp.func
def _restart_recycled_cr_component(
    component: wp.int32,
    basis_denominator: wp.array2d[wp.float32],
    orthogonalization: wp.array2d[wp.float32],
    projection: wp.array2d[wp.float32],
):
    """Drop every retained direction of one component."""
    for slot in range(RECYCLED_CR_DIRECTION_COUNT):
        basis_denominator[slot, component] = 0.0
        orthogonalization[slot, component] = 0.0
        projection[slot, component] = 0.0


###
# Kernels
###


@wp.kernel
def _gather_particle_state(
    # Inputs:
    state_particle_q: wp.array[wp.vec3],
    state_particle_qd: wp.array[wp.vec3],
    state_particle_f: wp.array[wp.vec3],
    model_particle_mass: wp.array[float],
    model_particle_flags: wp.array[wp.int32],
    model_gravity: wp.array[wp.vec3],
    packed_to_newton: wp.array[wp.int32],
    packed_source_world: wp.array[wp.int32],
    # Outputs:
    position_start: wp.array[wp.vec3],
    velocity_start: wp.array[wp.vec3],
    external_force: wp.array[wp.vec3],
):
    """Gather step-start Newton particle data into the packed deformable layout."""
    packed_particle = wp.tid()
    particle = packed_to_newton[packed_particle]
    position_start[packed_particle] = state_particle_q[particle]
    velocity_start[packed_particle] = state_particle_qd[particle]

    if is_dynamic_particle(packed_particle, packed_to_newton, model_particle_mass, model_particle_flags):
        wid = packed_source_world[packed_particle]
        external_force[packed_particle] = (
            state_particle_f[particle] + model_particle_mass[particle] * model_gravity[wid]
        )
    else:
        external_force[packed_particle] = wp.vec3(0.0)


@wp.kernel
def _prepare_particle_linearization(
    # Inputs:
    position_start: wp.array[wp.vec3],
    linearization_velocity: wp.array[wp.vec3],
    external_force: wp.array[wp.vec3],
    packed_to_newton: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    model_particle_flags: wp.array[wp.int32],
    model_time_dt: wp.array[wp.float32],
    # Outputs:
    position_linearized: wp.array[wp.vec3],
    velocity_linearized: wp.array[wp.vec3],
    smooth_force: wp.array[wp.vec3],
):
    """Set the packed deformable linearization while restoring external forces."""
    packed_particle = wp.tid()
    particle = packed_to_newton[packed_particle]
    dt = model_time_dt[packed_world[packed_particle]]
    velocity = linearization_velocity[packed_particle]
    if (model_particle_flags[particle] & ParticleFlags.ACTIVE) == 0:
        velocity = wp.vec3(0.0)
    position_linearized[packed_particle] = position_start[packed_particle] + dt * velocity
    velocity_linearized[packed_particle] = velocity
    smooth_force[packed_particle] = external_force[packed_particle]


@wp.kernel
def _assemble_particle_mass(
    # Inputs:
    packed_to_newton: wp.array[wp.int32],
    model_particle_mass: wp.array[float],
    model_particle_flags: wp.array[wp.int32],
    # Outputs:
    triplet_values: wp.array[wp.mat33],
):
    """Write mass or identity blocks into the particle diagonal triplets."""
    packed_particle = wp.tid()
    diagonal = 1.0
    if is_dynamic_particle(packed_particle, packed_to_newton, model_particle_mass, model_particle_flags):
        particle = packed_to_newton[packed_particle]
        diagonal = model_particle_mass[particle]
    triplet_values[packed_particle] = diagonal * wp.identity(n=3, dtype=float)


@wp.kernel
def _assemble_triangle_system(
    # Inputs:
    position_start: wp.array[wp.vec3],
    position_linearized: wp.array[wp.vec3],
    velocity_linearized: wp.array[wp.vec3],
    triangle_indices: wp.array2d[wp.int32],
    model_tri_poses: wp.array[wp.mat22],
    model_tri_areas: wp.array[float],
    model_tri_materials: wp.array2d[float],
    model_tri_activations: wp.array[float],
    packed_to_newton: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    model_particle_mass: wp.array[float],
    model_particle_flags: wp.array[wp.int32],
    model_time_dt: wp.array[wp.float32],
    triplet_offset: int,
    # Outputs:
    triplet_values: wp.array[wp.mat33],
    smooth_force: wp.array[wp.vec3],
    world_failed: wp.array[wp.bool],
):
    """Assemble stable membrane forces and PSD pair-block tangents."""
    triangle = wp.tid()
    vertex_0 = triangle_indices[triangle, 0]
    vertex_1 = triangle_indices[triangle, 1]
    vertex_2 = triangle_indices[triangle, 2]
    dt = model_time_dt[packed_world[vertex_0]]

    x_0 = position_linearized[vertex_0]
    x_1 = position_linearized[vertex_1]
    x_2 = position_linearized[vertex_2]
    x_10 = x_1 - x_0
    x_20 = x_2 - x_0
    rest_pose = model_tri_poses[triangle]

    deformation_0 = x_10 * rest_pose[0, 0] + x_20 * rest_pose[1, 0]
    deformation_1 = x_10 * rest_pose[0, 1] + x_20 * rest_pose[1, 1]

    f00 = wp.dot(deformation_0, deformation_0)
    f11 = wp.dot(deformation_1, deformation_1)
    f01 = wp.dot(deformation_0, deformation_1)
    area_ratio, area_gradient_0, area_gradient_1 = membrane_area_ratio_gradient(
        deformation_0,
        deformation_1,
    )

    mu = model_tri_materials[triangle, 0]
    lmbd = model_tri_materials[triangle, 1] + mu
    damping = model_tri_materials[triangle, 2]
    alpha = 1.0
    if lmbd > 1.0e-6:
        alpha = 1.0 + mu / lmbd
    area_constraint = area_ratio - alpha + model_tri_activations[triangle]

    stress_0 = mu * deformation_0 + lmbd * area_constraint * area_gradient_0
    stress_1 = mu * deformation_1 + lmbd * area_constraint * area_gradient_1
    rest_area = model_tri_areas[triangle]

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

    velocity_midpoint = (
        velocity_linearized[vertex_0] + velocity_linearized[vertex_1] + velocity_linearized[vertex_2]
    ) / 3.0
    normal_raw = wp.cross(x_10, x_20)
    current_area = 0.5 * wp.length(normal_raw)
    normal = wp.normalize(normal_raw)
    velocity_direction = wp.normalize(velocity_midpoint)
    drag = velocity_midpoint * (
        model_tri_materials[triangle, 3] * current_area * wp.abs(wp.dot(normal, velocity_midpoint))
    )
    lift_angle = wp.HALF_PI - wp.acos(wp.clamp(wp.dot(normal, velocity_direction), -1.0, 1.0))
    lift = normal * (
        model_tri_materials[triangle, 4] * current_area * lift_angle * wp.dot(velocity_midpoint, velocity_midpoint)
    )
    aerodynamic_force = -(drag + lift)

    # Preflight every contribution so a malformed element writes either all or
    # none of its shared particle forces and sparse blocks.
    element_finite = wp.bool(True)
    rest_area_dt_squared = rest_area * dt * dt
    for row in range(3):
        row_vertex = triangle_indices[triangle, row]
        row_dynamic = is_dynamic_particle(row_vertex, packed_to_newton, model_particle_mass, model_particle_flags)
        row_coefficients = triangle_vertex_coefficients(row, rest_pose)
        if row_dynamic:
            internal_force = _triangle_internal_force(
                row_coefficients,
                stress_0,
                stress_1,
                deformation_0,
                deformation_1,
                damping,
                c00_rate,
                c01_rate,
                c11_rate,
                rest_area,
            )
            element_finite = (
                element_finite
                and _vector_is_finite(internal_force)
                and _vector_is_finite(aerodynamic_force)
                and _vector_is_finite(internal_force + aerodynamic_force)
            )
        for column in range(3):
            column_vertex = triangle_indices[triangle, column]
            if row_dynamic and is_dynamic_particle(
                column_vertex,
                packed_to_newton,
                model_particle_mass,
                model_particle_flags,
            ):
                block = _triangle_tangent_block(
                    row_coefficients,
                    triangle_vertex_coefficients(column, rest_pose),
                    area_gradient_0,
                    area_gradient_1,
                    deformation_0,
                    deformation_1,
                    mu,
                    lmbd,
                    damping,
                    inverse_dt,
                    rest_area_dt_squared,
                )
                element_finite = element_finite and _matrix_is_finite(block)

    if not element_finite:
        world_failed[packed_world[vertex_0]] = True
        return

    for row in range(3):
        row_vertex = triangle_indices[triangle, row]
        row_dynamic = is_dynamic_particle(row_vertex, packed_to_newton, model_particle_mass, model_particle_flags)
        row_coefficients = triangle_vertex_coefficients(row, rest_pose)
        if row_dynamic:
            internal_force = _triangle_internal_force(
                row_coefficients,
                stress_0,
                stress_1,
                deformation_0,
                deformation_1,
                damping,
                c00_rate,
                c01_rate,
                c11_rate,
                rest_area,
            )
            wp.atomic_add(
                smooth_force,
                row_vertex,
                internal_force + aerodynamic_force,
            )
        for column in range(3):
            column_vertex = triangle_indices[triangle, column]
            block = wp.mat33(0.0)
            if row_dynamic and is_dynamic_particle(
                column_vertex,
                packed_to_newton,
                model_particle_mass,
                model_particle_flags,
            ):
                block = _triangle_tangent_block(
                    row_coefficients,
                    triangle_vertex_coefficients(column, rest_pose),
                    area_gradient_0,
                    area_gradient_1,
                    deformation_0,
                    deformation_1,
                    mu,
                    lmbd,
                    damping,
                    inverse_dt,
                    rest_area_dt_squared,
                )
            triplet_values[triplet_offset + triangle * 9 + row * 3 + column] = block


@wp.kernel
def _assemble_tetrahedron_system(
    # Inputs:
    position_start: wp.array[wp.vec3],
    position_linearized: wp.array[wp.vec3],
    tetrahedron_indices: wp.array2d[wp.int32],
    model_tet_poses: wp.array[wp.mat33],
    model_tet_materials: wp.array2d[float],
    model_tet_activations: wp.array[float],
    packed_to_newton: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    model_particle_mass: wp.array[float],
    model_particle_flags: wp.array[wp.int32],
    model_time_dt: wp.array[wp.float32],
    triplet_offset: int,
    # Outputs:
    tetrahedron_metric: wp.array[mat99f],
    triplet_values: wp.array[wp.mat33],
    smooth_force: wp.array[wp.vec3],
):
    """Assemble volumetric Neo-Hookean forces and contracted majorizing metrics."""
    tetrahedron = wp.tid()
    vertex_0 = tetrahedron_indices[tetrahedron, 0]
    vertex_1 = tetrahedron_indices[tetrahedron, 1]
    vertex_2 = tetrahedron_indices[tetrahedron, 2]
    vertex_3 = tetrahedron_indices[tetrahedron, 3]
    dt = model_time_dt[packed_world[vertex_0]]

    position_0 = position_linearized[vertex_0]
    position_1 = position_linearized[vertex_1]
    position_2 = position_linearized[vertex_2]
    position_3 = position_linearized[vertex_3]
    rest_pose = model_tet_poses[tetrahedron]
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
    k_mu = model_tet_materials[tetrahedron, 0]
    k_lambda = model_tet_materials[tetrahedron, 1]
    damping = model_tet_materials[tetrahedron, 2]
    stress, _gauss_newton_tangent = tet_stable_neo_hookean_differential(
        deformation,
        rest_volume,
        k_mu,
        k_lambda,
        model_tet_activations[tetrahedron],
    )
    spectral_metrics = tet_stable_neo_hookean_spectral_metrics(
        deformation,
        rest_volume,
        k_mu,
        k_lambda,
        model_tet_activations[tetrahedron],
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

        row_dynamic = is_dynamic_particle(row_vertex, packed_to_newton, model_particle_mass, model_particle_flags)
        if row_dynamic:
            wp.atomic_add(smooth_force, row_vertex, force)

        for column in range(4):
            column_vertex = tetrahedron_indices[tetrahedron, column]
            block = wp.mat33(0.0)
            if row_dynamic and is_dynamic_particle(
                column_vertex,
                packed_to_newton,
                model_particle_mass,
                model_particle_flags,
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
def _assemble_bending_system(
    # Inputs:
    position_start: wp.array[wp.vec3],
    position_linearized: wp.array[wp.vec3],
    bending_indices: wp.array2d[wp.int32],
    source_edge_indices: wp.array[wp.int32],
    model_edge_rest_angle: wp.array[float],
    model_edge_rest_length: wp.array[float],
    model_edge_bending_properties: wp.array2d[float],
    packed_to_newton: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    model_particle_mass: wp.array[float],
    model_particle_flags: wp.array[wp.int32],
    model_time_dt: wp.array[wp.float32],
    triplet_offset: int,
    # Outputs:
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
    dt = model_time_dt[packed_world[vertex_0]]

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

    rest_length = model_edge_rest_length[source_edge]
    stiffness = model_edge_bending_properties[source_edge, 0]
    damping = model_edge_bending_properties[source_edge, 1]
    elastic_scale = stiffness * rest_length
    force_scale = elastic_scale * (angle - model_edge_rest_angle[source_edge])
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
        if is_dynamic_particle(row_vertex, packed_to_newton, model_particle_mass, model_particle_flags):
            wp.atomic_add(smooth_force, row_vertex, -force_scale * row_gradient)

        for column in range(4):
            column_vertex = bending_indices[bending, column]
            block = wp.mat33(0.0)
            if is_dynamic_particle(
                row_vertex, packed_to_newton, model_particle_mass, model_particle_flags
            ) and is_dynamic_particle(column_vertex, packed_to_newton, model_particle_mass, model_particle_flags):
                column_gradient = _select_gradient(column, gradient_0, gradient_1, gradient_2, gradient_3)
                block = dt * dt * tangent_scale * wp.outer(row_gradient, column_gradient)
            triplet_values[triplet_offset + bending * 16 + row * 4 + column] = block


@wp.kernel
def _finalize_smooth_rhs(
    # Inputs:
    matrix_velocity: wp.array[wp.vec3],
    smooth_force: wp.array[wp.vec3],
    velocity_start: wp.array[wp.vec3],
    velocity_linearized: wp.array[wp.vec3],
    packed_to_newton: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    model_particle_mass: wp.array[float],
    model_particle_flags: wp.array[wp.int32],
    model_time_dt: wp.array[wp.float32],
    # Outputs:
    smooth_rhs: wp.array[wp.vec3],
):
    """Finish the affine frozen-dynamics right-hand side."""
    packed_particle = wp.tid()
    dt = model_time_dt[packed_world[packed_particle]]
    if is_dynamic_particle(packed_particle, packed_to_newton, model_particle_mass, model_particle_flags):
        particle = packed_to_newton[packed_particle]
        inertial_correction = model_particle_mass[particle] * (
            velocity_start[packed_particle] - velocity_linearized[packed_particle]
        )
        smooth_rhs[packed_particle] = (
            matrix_velocity[packed_particle] + inertial_correction + dt * smooth_force[packed_particle]
        )
    else:
        smooth_rhs[packed_particle] = wp.vec3(0.0)


@wp.kernel
def _select_consensus_weight(
    # Inputs:
    full_weight: wp.array[float],
    full_inverse_weight: wp.array[float],
    unilateral_incidence: wp.array[wp.int32],
    selective: bool,
    # Outputs:
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
def _assemble_consensus_weight(
    # Inputs:
    diagonal_slots: wp.array[wp.int32],
    weight: wp.array[float],
    fraction: float,
    # Outputs:
    system_values: wp.array[wp.mat33],
):
    """Add the scalar consensus weights, scaled by the consensus fraction, to the system diagonal blocks."""
    packed_particle = wp.tid()
    diagonal_slot = diagonal_slots[packed_particle]
    system_values[diagonal_slot] += (fraction * weight[packed_particle]) * wp.identity(n=3, dtype=float)


@wp.kernel
def _prepare_candidate_rhs(
    # Inputs:
    smooth_rhs: wp.array[wp.vec3],
    nonlinear_rhs: wp.array[wp.vec3],
    consensus_center: wp.array[wp.vec3],
    weight: wp.array[float],
    current_velocity: wp.array[wp.vec3],
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    # Outputs:
    candidate_rhs: wp.array[wp.vec3],
):
    """Form the LOX global candidate RHS with inactive-world retention."""
    particle = wp.tid()
    if world_active[packed_world[particle]]:
        candidate_rhs[particle] = (
            smooth_rhs[particle] + nonlinear_rhs[particle] + weight[particle] * consensus_center[particle]
        )
    else:
        candidate_rhs[particle] = current_velocity[particle]


@wp.kernel
def _retain_direct_candidate_rhs(
    # Inputs:
    current_velocity: wp.array[wp.vec3],
    packed_iterative: wp.array[wp.int32],
    # Outputs:
    candidate_rhs: wp.array[wp.vec3],
):
    """Replace direct-component right-hand sides for identity-masked CR batches."""
    particle = wp.tid()
    if packed_iterative[particle] == 0:
        candidate_rhs[particle] = current_velocity[particle]


# This kernel launches with its own block size, so a unique module compiles it alone for that size.
@wp.kernel(module="unique", enable_backward=False)
def _apply_system_matrix(
    # Inputs:
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array3d[wp.float32],
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
    packed_world: wp.array[wp.int32],
    packed_iterative: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    alpha: float,
    beta: float,
    # Outputs:
    result: wp.array[wp.float32],
):
    """Compute ``alpha A x + beta y`` on the CR rows of active worlds and ``alpha x + beta y`` elsewhere."""
    row, subrow = wp.tid()
    scalar_row = 3 * row + subrow
    value = x[scalar_row]
    if packed_iterative[row] != 0 and world_active[packed_world[row]]:
        value = wp.float32(0.0)
        slot = wp.int32(offsets[row])
        slot_end = wp.int32(offsets[row + 1])
        while slot < slot_end:
            scalar_column = 3 * columns[slot]
            for column in range(3):
                value += values[slot, subrow, column] * x[scalar_column + column]
            slot += 1
    value *= alpha
    if beta != 0.0:
        value += beta * y[scalar_row]
    result[scalar_row] = value


@wp.kernel
def _reduce_recycled_cr_basis(
    # Inputs:
    velocity: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
    right_hand_side: wp.array[wp.vec3],
    previous_velocity: wp.array[wp.vec3],
    previous_matrix_velocity: wp.array[wp.vec3],
    basis: wp.array2d[wp.vec3],
    basis_valid: wp.array[wp.int32],
    previous_projection: wp.array2d[wp.float32],
    replacement_cursor: wp.array[wp.int32],
    packed_component: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    packed_iterative: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    recycling_started: wp.array[wp.int32],
    # Outputs:
    reductions: wp.array2d[wp.float32],
):
    """Accumulate the per-component inner products of the new correction with itself, the residual, and the basis."""
    particle = wp.tid()
    if recycling_started[0] != 0 and packed_iterative[particle] != 0 and world_active[packed_world[particle]]:
        component = packed_component[particle]
        particle_residual = residual[particle]
        current_matrix_velocity = right_hand_side[particle] - particle_residual
        raw_direction = velocity[particle] - previous_velocity[particle]
        raw_matrix_direction = current_matrix_velocity - previous_matrix_velocity[particle]
        wp.atomic_add(reductions, 0, component, wp.dot(raw_direction, raw_matrix_direction))
        wp.atomic_add(reductions, 1, component, wp.dot(raw_direction, particle_residual))
        replacement_slot = replacement_cursor[0]
        previous_correction = wp.vec3(0.0)
        for slot in range(RECYCLED_CR_DIRECTION_COUNT):
            if basis_valid[slot] != 0:
                old_direction = basis[slot, particle]
                previous_correction += previous_projection[slot, component] * old_direction
                if slot != replacement_slot:
                    wp.atomic_add(
                        reductions,
                        3 + slot,
                        component,
                        wp.dot(old_direction, raw_matrix_direction),
                    )
                    wp.atomic_add(
                        reductions,
                        3 + RECYCLED_CR_DIRECTION_COUNT + slot,
                        component,
                        wp.dot(old_direction, particle_residual),
                    )
        wp.atomic_add(reductions, 2, component, wp.dot(raw_direction - previous_correction, raw_direction))


@wp.kernel
def _compute_recycled_cr_coefficients(
    # Inputs:
    reductions: wp.array2d[wp.float32],
    basis_valid: wp.array[wp.int32],
    replacement_cursor: wp.array[wp.int32],
    # Outputs:
    basis_denominator: wp.array2d[wp.float32],
    orthogonalization: wp.array2d[wp.float32],
    projection: wp.array2d[wp.float32],
):
    """Orthogonalize the new correction of each component against the basis and compute its projection."""
    component = wp.tid()
    replacement_slot = replacement_cursor[0]
    raw_energy = reductions[0, component]
    new_denominator = raw_energy
    new_numerator = reductions[1, component]
    for slot in range(RECYCLED_CR_DIRECTION_COUNT):
        orthogonalization[slot, component] = 0.0
        projection[slot, component] = 0.0
        denominator = basis_denominator[slot, component]
        if slot != replacement_slot and basis_valid[slot] != 0 and denominator > 1.0e-30:
            coupling = reductions[3 + slot, component]
            old_residual_dot = reductions[3 + RECYCLED_CR_DIRECTION_COUNT + slot, component]
            coefficient = coupling / denominator
            orthogonalization[slot, component] = coefficient
            projection[slot, component] = old_residual_dot / denominator
            new_denominator -= coupling * coefficient
            new_numerator -= coefficient * old_residual_dot

    minimum_denominator = wp.max(1.0e-6 * wp.abs(raw_energy), 1.0e-30)
    basis_denominator[replacement_slot, component] = 0.0
    restart_dot = reductions[2, component]
    restart = not wp.isfinite(restart_dot) or restart_dot <= 0.0
    if restart:
        _restart_recycled_cr_component(component, basis_denominator, orthogonalization, projection)
    elif new_denominator > minimum_denominator:
        coefficient = new_numerator / new_denominator
        if wp.isfinite(coefficient):
            basis_denominator[replacement_slot, component] = new_denominator
            projection[replacement_slot, component] = coefficient


@wp.kernel
def _apply_recycled_cr_basis(
    # Inputs:
    right_hand_side: wp.array[wp.vec3],
    basis_denominator: wp.array2d[wp.float32],
    basis_valid: wp.array[wp.int32],
    replacement_cursor: wp.array[wp.int32],
    orthogonalization: wp.array2d[wp.float32],
    projection: wp.array2d[wp.float32],
    packed_component: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    packed_iterative: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    recycling_started: wp.array[wp.int32],
    # Outputs:
    basis: wp.array2d[wp.vec3],
    matrix_basis: wp.array2d[wp.vec3],
    velocity: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
    previous_velocity: wp.array[wp.vec3],
    previous_matrix_velocity: wp.array[wp.vec3],
    projection_reductions: wp.array2d[wp.float32],
):
    """Store the orthogonalized correction in the basis and add the projected correction to the velocity."""
    particle = wp.tid()
    if packed_iterative[particle] != 0 and world_active[packed_world[particle]]:
        component = packed_component[particle]
        current_velocity = velocity[particle]
        particle_residual = residual[particle]
        current_matrix_velocity = right_hand_side[particle] - particle_residual
        raw_direction = current_velocity - previous_velocity[particle]
        raw_matrix_direction = current_matrix_velocity - previous_matrix_velocity[particle]
        previous_velocity[particle] = current_velocity
        previous_matrix_velocity[particle] = current_matrix_velocity
        if recycling_started[0] != 0:
            replacement_slot = replacement_cursor[0]
            new_direction = raw_direction
            new_matrix_direction = raw_matrix_direction
            correction = wp.vec3(0.0)
            matrix_correction = wp.vec3(0.0)
            for slot in range(RECYCLED_CR_DIRECTION_COUNT):
                if slot != replacement_slot and basis_valid[slot] != 0:
                    old_direction = basis[slot, particle]
                    old_matrix_direction = matrix_basis[slot, particle]
                    new_direction -= orthogonalization[slot, component] * old_direction
                    new_matrix_direction -= orthogonalization[slot, component] * old_matrix_direction
                    correction += projection[slot, component] * old_direction
                    matrix_correction += projection[slot, component] * old_matrix_direction
            if basis_denominator[replacement_slot, component] > 0.0:
                correction += projection[replacement_slot, component] * new_direction
                matrix_correction += projection[replacement_slot, component] * new_matrix_direction
            else:
                new_direction = wp.vec3(0.0)
                new_matrix_direction = wp.vec3(0.0)
            basis[replacement_slot, particle] = new_direction
            matrix_basis[replacement_slot, particle] = new_matrix_direction
            velocity[particle] = current_velocity + correction
            residual[particle] -= matrix_correction
            wp.atomic_add(projection_reductions, 0, component, wp.dot(particle_residual, correction))
            wp.atomic_add(projection_reductions, 1, component, wp.dot(correction, matrix_correction))


@wp.kernel
def _validate_and_restart_recycled_cr_projection(
    # Inputs:
    projection_reductions: wp.array2d[wp.float32],
    recycling_started: wp.array[wp.int32],
    # Outputs:
    basis_denominator: wp.array2d[wp.float32],
    orthogonalization: wp.array2d[wp.float32],
    projection: wp.array2d[wp.float32],
    projection_accepted: wp.array[wp.int32],
):
    """Reject a recycled projection that increases the linear quadratic."""
    component = wp.tid()
    accepted = wp.int32(1)
    if recycling_started[0] != 0:
        residual_dot_correction = projection_reductions[0, component]
        correction_energy = projection_reductions[1, component]
        scale = wp.max(wp.abs(residual_dot_correction), wp.abs(correction_energy))
        tolerance = wp.max(1.0e-5 * scale, 1.0e-30)
        accepted = wp.int32(
            wp.isfinite(residual_dot_correction)
            and wp.isfinite(correction_energy)
            and correction_energy >= -tolerance
            and residual_dot_correction + tolerance >= 0.5 * correction_energy
        )
        if accepted == 0:
            _restart_recycled_cr_component(component, basis_denominator, orthogonalization, projection)
    projection_accepted[component] = accepted


@wp.kernel
def _rollback_unsafe_recycled_cr_projection(
    # Inputs:
    right_hand_side: wp.array[wp.vec3],
    previous_velocity: wp.array[wp.vec3],
    previous_matrix_velocity: wp.array[wp.vec3],
    packed_component: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    packed_iterative: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    recycling_started: wp.array[wp.int32],
    projection_accepted: wp.array[wp.int32],
    # Outputs:
    velocity: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
):
    """Restore the unprojected velocity and residual of the components with a rejected projection."""
    particle = wp.tid()
    component = packed_component[particle]
    if (
        recycling_started[0] != 0
        and packed_iterative[particle] != 0
        and world_active[packed_world[particle]]
        and projection_accepted[component] == 0
    ):
        velocity[particle] = previous_velocity[particle]
        residual[particle] = right_hand_side[particle] - previous_matrix_velocity[particle]


@wp.kernel
def _compute_candidate_linear_residual(
    # Inputs:
    residual: wp.array[wp.vec3],
    inverse_scale: wp.array[wp.float32],
    packed_world: wp.array[wp.int32],
    packed_iterative: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    # Outputs:
    world_residual: wp.array[wp.float32],
):
    """Accumulate the largest weight-scaled CR residual of each active world."""
    particle = wp.tid()
    wid = packed_world[particle]
    if packed_iterative[particle] != 0 and world_active[wid]:
        scaled = inverse_scale[particle] * residual[particle]
        value = wp.max(wp.abs(scaled[0]), wp.abs(scaled[1]))
        value = wp.max(value, wp.abs(scaled[2]))
        if not wp.isfinite(value):
            value = wp.float32(3.402823e38)
        wp.atomic_max(world_residual, wid, value)


@wp.kernel
def _compute_candidate_linear_residual_and_advance_recycling(
    # Inputs:
    residual: wp.array[wp.vec3],
    inverse_scale: wp.array[wp.float32],
    packed_world: wp.array[wp.int32],
    packed_iterative: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    # Outputs:
    world_residual: wp.array[wp.float32],
    recycling_started: wp.array[wp.int32],
    replacement_cursor: wp.array[wp.int32],
    basis_valid: wp.array[wp.int32],
):
    """Accumulate the largest weight-scaled CR residual of each active world and advance the replacement slot."""
    particle = wp.tid()
    wid = packed_world[particle]
    if packed_iterative[particle] != 0 and world_active[wid]:
        scaled = inverse_scale[particle] * residual[particle]
        value = wp.max(wp.abs(scaled[0]), wp.abs(scaled[1]))
        value = wp.max(value, wp.abs(scaled[2]))
        if not wp.isfinite(value):
            value = wp.float32(3.402823e38)
        wp.atomic_max(world_residual, wid, value)
    if particle == 0:
        replacement_slot = replacement_cursor[0]
        basis_valid[replacement_slot] = recycling_started[0]
        recycling_started[0] = 1
        replacement_cursor[0] = (replacement_slot + 1) % RECYCLED_CR_DIRECTION_COUNT
