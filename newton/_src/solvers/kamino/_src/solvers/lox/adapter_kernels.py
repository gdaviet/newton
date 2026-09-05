# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels bridging Kamino containers with LOX solver storage.

Preparation kernels materialize compact, coalesced row data once per nonlinear
evaluation. LOX reuses those rows across its projection iterations; consumers
therefore avoid repeated sparse-Jacobian indirection and contact preprocessing.
"""

from functools import cache

import warp as wp

from ...core.joints import JointActuationType, JointCorrectionMode
from ...core.math import compute_body_pose_update_with_logmap, contact_wrench_matrix_from_points
from ...core.types import mat36f, mat66f, vec6f
from ...geometry.contacts import ContactMode
from ...kinematics.joints import compute_joint_pose_and_relative_motion, make_write_joint_data
from .bias import compute_contact_velocity_target, compute_limit_velocity_target
from .projection import PROJECTION_STATUS_VALID

wp.set_module_options({"enable_backward": False})


@wp.func
def _load_sparse_jacobian_row(index: wp.int32, jacobian_data: wp.array[vec6f]) -> vec6f:
    if index >= 0:
        return jacobian_data[index]
    return vec6f(0.0)


@wp.func
def _inverse_mass_quadratic_form(
    jacobian: vec6f,
    body: wp.int32,
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
) -> wp.float32:
    if body < 0:
        return 0.0
    linear = wp.vec3f(jacobian[0], jacobian[1], jacobian[2])
    angular = wp.vec3f(jacobian[3], jacobian[4], jacobian[5])
    return inverse_mass[body] * wp.dot(linear, linear) + wp.dot(angular, inverse_inertia_world[body] @ angular)


@wp.func
def _inverse_mass_bilinear_form(
    first: vec6f,
    second: vec6f,
    body: wp.int32,
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
) -> wp.float32:
    if body < 0:
        return 0.0
    first_linear = wp.vec3f(first[0], first[1], first[2])
    second_linear = wp.vec3f(second[0], second[1], second[2])
    first_angular = wp.vec3f(first[3], first[4], first[5])
    second_angular = wp.vec3f(second[3], second[4], second[5])
    return inverse_mass[body] * wp.dot(first_linear, second_linear) + wp.dot(
        first_angular, inverse_inertia_world[body] @ second_angular
    )


@wp.kernel
def _evaluate_lagged_scalar_velocity_consistency(
    row_world: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    world_active: wp.array[wp.bool],
    global_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    inverse_velocity_tolerance: wp.float32,
    world_required: wp.array[wp.int32],
    world_residual: wp.array[wp.float32],
):
    row = wp.tid()
    world = row_world[row]
    if not world_active[world]:
        return

    first = body_first[row]
    second = body_second[row]
    if first < 0 and second < 0:
        return
    value = wp.float32(0.0)
    if first >= 0:
        value += wp.dot(jacobian_first[row], global_twist[first] - projected_twist_previous[first])
    if second >= 0:
        value += wp.dot(jacobian_second[row], global_twist[second] - projected_twist_previous[second])
    wp.atomic_max(world_required, world, 1)
    wp.atomic_max(world_residual, world, wp.abs(value) * inverse_velocity_tolerance)


@wp.kernel
def _evaluate_lagged_contact_velocity_consistency(
    world_dynamic_offset: wp.array[wp.int32],
    world_dynamic_count: wp.array[wp.int32],
    dynamic_body_first: wp.array[wp.int32],
    dynamic_body_second: wp.array[wp.int32],
    dynamic_jacobian_first: wp.array[vec6f],
    dynamic_jacobian_second: wp.array[vec6f],
    world_structural_offset: wp.array[wp.int32],
    world_structural_count: wp.array[wp.int32],
    structural_body_first: wp.array[wp.int32],
    structural_body_second: wp.array[wp.int32],
    structural_jacobian_first: wp.array[vec6f],
    structural_jacobian_second: wp.array[vec6f],
    world_friction_offset: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    world_limit_offset: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    world_active: wp.array[wp.bool],
    global_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    inverse_velocity_tolerance: wp.float32,
    block_count: wp.int32,
    world_required: wp.array[wp.int32],
    world_residual: wp.array[wp.float32],
):
    world, block, lane = wp.tid()
    if not world_active[world]:
        return

    residual = wp.float32(0.0)
    if block == 0:
        scalar_local = lane
        while scalar_local < world_dynamic_count[world]:
            row = world_dynamic_offset[world] + scalar_local
            first = dynamic_body_first[row]
            second = dynamic_body_second[row]
            scalar_value = wp.float32(0.0)
            if first >= 0:
                scalar_value += wp.dot(
                    dynamic_jacobian_first[row], global_twist[first] - projected_twist_previous[first]
                )
            if second >= 0:
                scalar_value += wp.dot(
                    dynamic_jacobian_second[row], global_twist[second] - projected_twist_previous[second]
                )
            residual = wp.max(residual, wp.abs(scalar_value) * inverse_velocity_tolerance)
            scalar_local += wp.block_dim()
        scalar_local = lane
        while scalar_local < world_structural_count[world]:
            row = world_structural_offset[world] + scalar_local
            first = structural_body_first[row]
            second = structural_body_second[row]
            scalar_value = wp.float32(0.0)
            if first >= 0:
                scalar_value += wp.dot(
                    structural_jacobian_first[row], global_twist[first] - projected_twist_previous[first]
                )
            if second >= 0:
                scalar_value += wp.dot(
                    structural_jacobian_second[row], global_twist[second] - projected_twist_previous[second]
                )
            residual = wp.max(residual, wp.abs(scalar_value) * inverse_velocity_tolerance)
            scalar_local += wp.block_dim()
        scalar_local = lane
        while scalar_local < world_friction_count[world]:
            row = world_friction_offset[world] + scalar_local
            first = friction_body_first[row]
            second = friction_body_second[row]
            scalar_value = wp.float32(0.0)
            if first >= 0:
                scalar_value += wp.dot(
                    friction_jacobian_first[row], global_twist[first] - projected_twist_previous[first]
                )
            if second >= 0:
                scalar_value += wp.dot(
                    friction_jacobian_second[row], global_twist[second] - projected_twist_previous[second]
                )
            residual = wp.max(residual, wp.abs(scalar_value) * inverse_velocity_tolerance)
            scalar_local += wp.block_dim()
        scalar_local = lane
        while scalar_local < world_limit_count[world]:
            row = world_limit_offset[world] + scalar_local
            first = limit_body_first[row]
            second = limit_body_second[row]
            scalar_value = wp.float32(0.0)
            if first >= 0:
                scalar_value += wp.dot(limit_jacobian_first[row], global_twist[first] - projected_twist_previous[first])
            if second >= 0:
                scalar_value += wp.dot(
                    limit_jacobian_second[row], global_twist[second] - projected_twist_previous[second]
                )
            residual = wp.max(residual, wp.abs(scalar_value) * inverse_velocity_tolerance)
            scalar_local += wp.block_dim()

    contact_count = world_contact_count[world]
    local = block * wp.block_dim() + lane
    stride = block_count * wp.block_dim()
    while local < contact_count:
        contact = world_contact_offset[world] + local
        first = contact_body_first[contact]
        second = contact_body_second[contact]
        contact_value = wp.vec3f(0.0)
        if first >= 0:
            contact_value += contact_jacobian_first[contact] @ (global_twist[first] - projected_twist_previous[first])
        if second >= 0:
            contact_value += contact_jacobian_second[contact] @ (
                global_twist[second] - projected_twist_previous[second]
            )
        residual = wp.max(residual, wp.max(wp.abs(contact_value)) * inverse_velocity_tolerance)
        local += stride

    block_residual = wp.tile_max(wp.tile(residual))[0]
    if lane == 0:
        if block == 0:
            scalar_count = (
                world_dynamic_count[world]
                + world_structural_count[world]
                + world_friction_count[world]
                + world_limit_count[world]
            )
            if scalar_count + contact_count > 0:
                world_required[world] = 1
        if block * wp.block_dim() < contact_count or block == 0:
            wp.atomic_max(world_residual, world, block_residual)


@wp.kernel
def _prepare_dynamic_rows(
    row_world: wp.array[wp.int32],
    uses_dof_jacobian: wp.array[wp.bool],
    body_first_global: wp.array[wp.int32],
    body_second_global: wp.array[wp.int32],
    value_index: wp.array[wp.int32],
    dof_index: wp.array[wp.int32],
    sparse_first_index: wp.array[wp.int32],
    sparse_second_index: wp.array[wp.int32],
    sparse_jacobian_data: wp.array[vec6f],
    sparse_dof_jacobian_data: wp.array[vec6f],
    joint_inertia: wp.array[wp.float32],
    joint_free_velocity: wp.array[wp.float32],
    dynamic_effort_index: wp.array[wp.int32],
    effort_value_index: wp.array[wp.int32],
    effort_inverse_inertia: wp.array[wp.float32],
    effort_free_velocity: wp.array[wp.float32],
    joint_armature: wp.array[wp.float32],
    joint_position_stiffness: wp.array[wp.float32],
    joint_actuation_type: wp.array[wp.int32],
    joint_velocity: wp.array[wp.float32],
    joint_velocity_begin: wp.array[wp.float32],
    external_effort: wp.array[wp.float32],
    velocity_stiffness: wp.array[wp.float32],
    effort_limit: wp.array[wp.float32],
    linearization_twist: wp.array[vec6f],
    time_step: wp.array[wp.float32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    effective_inertia: wp.array[wp.float32],
    free_velocity: wp.array[wp.float32],
    effort_intercept: wp.array[wp.float32],
    effort_slope: wp.array[wp.float32],
    effort_impulse_bound: wp.array[wp.float32],
):
    row = wp.tid()
    world = row_world[row]
    dt = time_step[world]
    if uses_dof_jacobian[row]:
        jacobian_first[row] = _load_sparse_jacobian_row(sparse_first_index[row], sparse_dof_jacobian_data)
        jacobian_second[row] = _load_sparse_jacobian_row(sparse_second_index[row], sparse_dof_jacobian_data)
    else:
        jacobian_first[row] = _load_sparse_jacobian_row(sparse_first_index[row], sparse_jacobian_data)
        jacobian_second[row] = _load_sparse_jacobian_row(sparse_second_index[row], sparse_jacobian_data)
    source = value_index[row]
    dof = dof_index[row]
    inertia = wp.float32(0.0)
    velocity = wp.float32(0.0)
    if source >= 0:
        inertia = joint_inertia[source]
        velocity = joint_free_velocity[source]
    bounded = dynamic_effort_index[row]
    if bounded >= 0:
        effort_source = effort_value_index[bounded]
        actuator_inverse_inertia = effort_inverse_inertia[effort_source]
        if actuator_inverse_inertia > 0.0:
            actuator_inertia = 1.0 / actuator_inverse_inertia
            velocity = (inertia * velocity + actuator_inertia * effort_free_velocity[effort_source]) / (
                inertia + actuator_inertia
            )
            inertia += actuator_inertia
    effective_inertia[row] = inertia
    mode = joint_actuation_type[dof]
    if inertia > 0.0:
        velocity += joint_armature[dof] * (joint_velocity_begin[dof] - joint_velocity[dof]) / inertia
        if (
            mode == JointActuationType.POSITION
            or mode == JointActuationType.POSITION_VELOCITY
            or mode == JointActuationType.POSITION_VELOCITY_FORCE
        ):
            linearization_velocity = wp.float32(0.0)
            first = body_first_global[row]
            second = body_second_global[row]
            if first >= 0:
                linearization_velocity += wp.dot(jacobian_first[row], linearization_twist[first])
            if second >= 0:
                linearization_velocity += wp.dot(jacobian_second[row], linearization_twist[second])
            velocity += dt * dt * joint_position_stiffness[dof] * linearization_velocity / inertia
    free_velocity[row] = velocity
    if bounded >= 0:
        gradient = wp.float32(0.0)
        if mode == JointActuationType.VELOCITY:
            gradient = velocity_stiffness[dof]
        if (
            mode == JointActuationType.POSITION
            or mode == JointActuationType.POSITION_VELOCITY
            or mode == JointActuationType.POSITION_VELOCITY_FORCE
        ):
            gradient = velocity_stiffness[dof] + dt * joint_position_stiffness[dof]
        beta = inertia * velocity
        effort_intercept[bounded] = (beta - joint_armature[dof] * joint_velocity_begin[dof]) / dt - external_effort[dof]
        effort_slope[bounded] = gradient
        effort_impulse_bound[bounded] = dt * effort_limit[dof]


@wp.kernel
def _promote_effort_counters(
    effort_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    counter_next: wp.array[wp.float32],
    counter_applied: wp.array[wp.float32],
):
    effort = wp.tid()
    if world_active[effort_world[effort]]:
        counter_applied[effort] = counter_next[effort]


@wp.kernel
def _clear_active_effort_residuals(
    world_active: wp.array[wp.bool],
    residual_scaled: wp.array[wp.float32],
    residual_unscaled: wp.array[wp.float32],
):
    world = wp.tid()
    if world_active[world]:
        residual_scaled[world] = 0.0
        residual_unscaled[world] = 0.0


@wp.kernel
def _update_effort_counters(
    effort_world: wp.array[wp.int32],
    effort_dynamic_row: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    effective_inertia: wp.array[wp.float32],
    projected_twist: wp.array[vec6f],
    intercept: wp.array[wp.float32],
    slope: wp.array[wp.float32],
    impulse_bound: wp.array[wp.float32],
    counter_applied: wp.array[wp.float32],
    time_step: wp.array[wp.float32],
    velocity_tolerance: wp.float32,
    counter_next: wp.array[wp.float32],
    raw_impulse: wp.array[wp.float32],
    net_applied: wp.array[wp.float32],
    net_target: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
    residual: wp.array[wp.float32],
    world_residual_scaled: wp.array[wp.float32],
    world_residual_unscaled: wp.array[wp.float32],
):
    effort = wp.tid()
    world = effort_world[effort]
    if not world_active[world]:
        return

    dynamic_row = effort_dynamic_row[effort]
    current_velocity = wp.float32(0.0)
    first = body_first[dynamic_row]
    second = body_second[dynamic_row]
    if first >= 0:
        current_velocity += wp.dot(jacobian_first[dynamic_row], projected_twist[first])
    if second >= 0:
        current_velocity += wp.dot(jacobian_second[dynamic_row], projected_twist[second])

    raw = time_step[world] * (intercept[effort] - slope[effort] * current_velocity)
    bound = impulse_bound[effort]
    target = wp.clamp(raw, -bound, bound)
    next_counter = target - raw
    applied_counter = counter_applied[effort]
    defect = wp.abs(next_counter - applied_counter)
    scaled_defect = defect / (wp.max(1.0e-6, effective_inertia[dynamic_row]) * velocity_tolerance)

    counter_next[effort] = next_counter
    raw_impulse[effort] = raw
    net_applied[effort] = raw + applied_counter
    net_target[effort] = target
    velocity[effort] = current_velocity
    residual[effort] = defect
    wp.atomic_max(world_residual_scaled, world, scaled_defect)
    wp.atomic_max(world_residual_unscaled, world, defect)


@wp.kernel
def _prepare_joint_frictions(
    row_world: wp.array[wp.int32],
    body_first_global: wp.array[wp.int32],
    body_second_global: wp.array[wp.int32],
    dof_index: wp.array[wp.int32],
    sparse_first_index: wp.array[wp.int32],
    sparse_second_index: wp.array[wp.int32],
    sparse_jacobian_data: wp.array[vec6f],
    friction_force: wp.array[wp.float32],
    body_velocity_begin: wp.array[vec6f],
    time_step: wp.array[wp.float32],
    initialize: wp.bool,
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    impulse_bound: wp.array[wp.float32],
    reaction: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
):
    row = wp.tid()
    world = row_world[row]
    first_jacobian = _load_sparse_jacobian_row(sparse_first_index[row], sparse_jacobian_data)
    second_jacobian = _load_sparse_jacobian_row(sparse_second_index[row], sparse_jacobian_data)
    bound = time_step[world] * friction_force[dof_index[row]]
    jacobian_first[row] = first_jacobian
    jacobian_second[row] = second_jacobian
    impulse_bound[row] = bound
    reaction[row] = wp.clamp(reaction[row], -bound, bound)
    if initialize:
        value = wp.float32(0.0)
        first = body_first_global[row]
        second = body_second_global[row]
        if first >= 0:
            value += wp.dot(first_jacobian, body_velocity_begin[first])
        if second >= 0:
            value += wp.dot(second_jacobian, body_velocity_begin[second])
        velocity[row] = value


@wp.kernel
def _prepare_structural_rows(
    body_first_global: wp.array[wp.int32],
    body_second_global: wp.array[wp.int32],
    sparse_first_index: wp.array[wp.int32],
    sparse_second_index: wp.array[wp.int32],
    sparse_jacobian_data: wp.array[vec6f],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    effective_mass: wp.array[wp.float32],
):
    row = wp.tid()
    first_jacobian = _load_sparse_jacobian_row(sparse_first_index[row], sparse_jacobian_data)
    second_jacobian = _load_sparse_jacobian_row(sparse_second_index[row], sparse_jacobian_data)
    jacobian_first[row] = first_jacobian
    jacobian_second[row] = second_jacobian

    inverse_effective_mass = _inverse_mass_quadratic_form(
        first_jacobian, body_first_global[row], inverse_mass, inverse_inertia_world
    ) + _inverse_mass_quadratic_form(second_jacobian, body_second_global[row], inverse_mass, inverse_inertia_world)
    effective_mass[row] = 1.0 / inverse_effective_mass if inverse_effective_mass > 1.0e-12 else 0.0


@cache
def make_evaluate_candidate_structural_residual_kernel(correction: JointCorrectionMode):
    """Build a kernel that evaluates exact joint residuals at candidate poses."""

    @wp.kernel
    def _evaluate_candidate_structural_residual(
        time_step: wp.array[wp.float32],
        joint_world: wp.array[wp.int32],
        joint_dof_type: wp.array[wp.int32],
        joint_coords_offset: wp.array[wp.int32],
        joint_dofs_offset: wp.array[wp.int32],
        joint_kinematic_offset: wp.array[wp.int32],
        joint_body_first: wp.array[wp.int32],
        joint_body_second: wp.array[wp.int32],
        joint_first_position: wp.array[wp.vec3f],
        joint_second_position: wp.array[wp.vec3f],
        joint_first_orientation: wp.array[wp.mat33f],
        joint_second_orientation: wp.array[wp.mat33f],
        body_pose: wp.array[wp.transformf],
        candidate_twist: wp.array[vec6f],
        linearization_twist: wp.array[vec6f],
        previous_joint_coordinate: wp.array[wp.float32],
        world_active: wp.array[wp.bool],
        candidate_residual: wp.array[wp.float32],
        scratch_residual_velocity: wp.array[wp.float32],
        scratch_joint_coordinate: wp.array[wp.float32],
        scratch_joint_velocity: wp.array[wp.float32],
    ):
        joint = wp.tid()
        world = joint_world[joint]
        if not world_active[world]:
            return
        dt = time_step[world]

        first = joint_body_first[joint]
        second = joint_body_second[joint]
        first_pose = wp.transform_identity(dtype=wp.float32)
        if first >= 0:
            first_delta = candidate_twist[first] - linearization_twist[first]
            first_pose = compute_body_pose_update_with_logmap(
                dt,
                body_pose[first],
                wp.vec3f(first_delta[0], first_delta[1], first_delta[2]),
                wp.vec3f(first_delta[3], first_delta[4], first_delta[5]),
            )
        second_delta = candidate_twist[second] - linearization_twist[second]
        second_pose = compute_body_pose_update_with_logmap(
            dt,
            body_pose[second],
            wp.vec3f(second_delta[0], second_delta[1], second_delta[2]),
            wp.vec3f(second_delta[3], second_delta[4], second_delta[5]),
        )

        _, relative_position, relative_orientation, relative_twist = compute_joint_pose_and_relative_motion(
            first_pose,
            second_pose,
            wp.spatial_vectorf(0.0),
            wp.spatial_vectorf(0.0),
            joint_first_position[joint],
            joint_second_position[joint],
            joint_first_orientation[joint],
            joint_second_orientation[joint],
        )
        wp.static(make_write_joint_data(correction))(
            joint_dof_type[joint],
            joint_kinematic_offset[joint],
            joint_dofs_offset[joint],
            joint_coords_offset[joint],
            relative_position,
            relative_orientation,
            relative_twist,
            previous_joint_coordinate,
            candidate_residual,
            scratch_residual_velocity,
            scratch_joint_coordinate,
            scratch_joint_velocity,
        )

    return _evaluate_candidate_structural_residual


@wp.kernel
def _blend_structural_candidate_twist(
    global_twist: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
    projected_fraction: wp.float32,
    candidate_twist: wp.array[vec6f],
):
    body = wp.tid()
    candidate_twist[body] = global_twist[body] + projected_fraction * (projected_twist[body] - global_twist[body])


@wp.kernel
def _include_dynamic_compliance_in_structural_effective_mass(
    joint_body_first: wp.array[wp.int32],
    joint_body_second: wp.array[wp.int32],
    joint_structural_offset: wp.array[wp.int32],
    joint_structural_count: wp.array[wp.int32],
    joint_dynamic_offset: wp.array[wp.int32],
    joint_dynamic_count: wp.array[wp.int32],
    structural_jacobian_first: wp.array[vec6f],
    structural_jacobian_second: wp.array[vec6f],
    dynamic_jacobian_first: wp.array[vec6f],
    dynamic_jacobian_second: wp.array[vec6f],
    dynamic_effective_inertia: wp.array[wp.float32],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
    effective_mass: wp.array[wp.float32],
):
    joint = wp.tid()
    structural_count = joint_structural_count[joint]
    dynamic_count = joint_dynamic_count[joint]
    if structural_count == 0 or dynamic_count == 0:
        return

    first_body = joint_body_first[joint]
    second_body = joint_body_second[joint]
    dynamic_offset = joint_dynamic_offset[joint]

    # Factor the mixed dynamic-row Schur block
    #
    #   J_d M^-1 J_d^T + diag(m_j^-1).
    #
    # Keeping m_j^-1 here retains the full implicit drive compliance for
    # effort-limited drives independently of whether their multiplier is at
    # its bound.
    lower = mat66f(0.0)
    valid = wp.bool(True)
    for row in range(6):
        if row < dynamic_count:
            first_row = dynamic_jacobian_first[dynamic_offset + row]
            second_row = dynamic_jacobian_second[dynamic_offset + row]
            for col in range(6):
                if col <= row and col < dynamic_count:
                    first_col = dynamic_jacobian_first[dynamic_offset + col]
                    second_col = dynamic_jacobian_second[dynamic_offset + col]
                    value = _inverse_mass_bilinear_form(
                        first_row, first_col, first_body, inverse_mass, inverse_inertia_world
                    ) + _inverse_mass_bilinear_form(
                        second_row, second_col, second_body, inverse_mass, inverse_inertia_world
                    )
                    if row == col:
                        inertia = dynamic_effective_inertia[dynamic_offset + row]
                        if inertia > 0.0:
                            value += 1.0 / inertia
                    for inner in range(6):
                        if inner < col:
                            value -= lower[row, inner] * lower[col, inner]
                    if row == col:
                        if not wp.isfinite(value) or value <= 1.0e-12:
                            valid = False
                        elif valid:
                            lower[row, col] = wp.sqrt(value)
                    elif valid:
                        lower[row, col] = value / lower[col, col]

    if not valid:
        return

    structural_offset = joint_structural_offset[joint]
    for structural_local in range(6):
        if structural_local < structural_count:
            structural_row = structural_offset + structural_local
            first_structural = structural_jacobian_first[structural_row]
            second_structural = structural_jacobian_second[structural_row]
            inverse_effective_mass = _inverse_mass_bilinear_form(
                first_structural, first_structural, first_body, inverse_mass, inverse_inertia_world
            ) + _inverse_mass_bilinear_form(
                second_structural, second_structural, second_body, inverse_mass, inverse_inertia_world
            )

            # Subtract the response of the retained mixed drive rows:
            # J_s M^-1 J_d^T S_d^-1 J_d M^-1 J_s^T.
            forward = vec6f(0.0)
            for row in range(6):
                if row < dynamic_count:
                    coupling = _inverse_mass_bilinear_form(
                        first_structural,
                        dynamic_jacobian_first[dynamic_offset + row],
                        first_body,
                        inverse_mass,
                        inverse_inertia_world,
                    ) + _inverse_mass_bilinear_form(
                        second_structural,
                        dynamic_jacobian_second[dynamic_offset + row],
                        second_body,
                        inverse_mass,
                        inverse_inertia_world,
                    )
                    for inner in range(6):
                        if inner < row:
                            coupling -= lower[row, inner] * forward[inner]
                    forward[row] = coupling / lower[row, row]
                    inverse_effective_mass -= forward[row] * forward[row]

            effective_mass[structural_row] = 1.0 / inverse_effective_mass if inverse_effective_mass > 1.0e-12 else 0.0


@wp.kernel
def _accumulate_constraint_incidence(
    entity_world: wp.array[wp.int32],
    entity_local: wp.array[wp.int32],
    world_count: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    body_constraint_count: wp.array[wp.int32],
    body_has_unilateral: wp.array[wp.int32],
):
    entity = wp.tid()
    world = entity_world[entity]
    if entity_local[entity] >= world_count[world]:
        return
    first = body_first[entity]
    second = body_second[entity]
    if first >= 0:
        wp.atomic_add(body_constraint_count, first, 1)
        wp.atomic_max(body_has_unilateral, first, 1)
    if second >= 0 and second != first:
        wp.atomic_add(body_constraint_count, second, 1)
        wp.atomic_max(body_has_unilateral, second, 1)


@wp.kernel
def _clear_inactive_limits(
    entity_world: wp.array[wp.int32],
    entity_local: wp.array[wp.int32],
    world_count: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    reaction: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
):
    limit = wp.tid()
    if entity_local[limit] >= world_count[entity_world[limit]]:
        body_first[limit] = -1
        body_second[limit] = -1
        reaction[limit] = 0.0
        velocity[limit] = 0.0


@wp.kernel
def _clear_inactive_contacts(
    entity_world: wp.array[wp.int32],
    entity_local: wp.array[wp.int32],
    world_count: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    reaction: wp.array[wp.vec3f],
    velocity: wp.array[wp.vec3f],
):
    contact = wp.tid()
    if entity_local[contact] >= world_count[entity_world[contact]]:
        body_first[contact] = -1
        body_second[contact] = -1
        reaction[contact] = wp.vec3f(0.0)
        velocity[contact] = wp.vec3f(0.0)


@wp.kernel
def _copy_clamped_world_counts(
    source: wp.array[wp.int32],
    capacity: wp.array[wp.int32],
    destination: wp.array[wp.int32],
):
    world = wp.tid()
    destination[world] = wp.min(wp.max(source[world], 0), capacity[world])


@wp.kernel
def _mark_worlds_with_unilaterals(
    contact_count: wp.array[wp.int32],
    limit_count: wp.array[wp.int32],
    friction_count: wp.array[wp.int32],
    world_has_unilateral: wp.array[wp.bool],
):
    world = wp.tid()
    world_has_unilateral[world] = contact_count[world] > 0 or limit_count[world] > 0 or friction_count[world] > 0


@wp.kernel
def _prepare_limits(
    source_active: wp.array[wp.int32],
    source_capacity: wp.int32,
    source_world: wp.array[wp.int32],
    source_local: wp.array[wp.int32],
    source_bodies: wp.array[wp.vec2i],
    source_violation: wp.array[wp.float32],
    source_reaction: wp.array[wp.float32],
    body_velocity_begin: wp.array[vec6f],
    world_capacity: wp.array[wp.int32],
    world_offset: wp.array[wp.int32],
    body_vector_index: wp.array[wp.int32],
    sparse_jacobian_offsets: wp.array[wp.int32],
    sparse_jacobian_data: wp.array[vec6f],
    time_step: wp.array[wp.float32],
    stabilization_fraction: wp.float32,
    import_reactions: wp.bool,
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    bias: wp.array[wp.float32],
    reaction: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
):
    source = wp.tid()
    if source >= wp.min(source_active[0], source_capacity):
        return
    world = source_world[source]
    local = source_local[source]
    if world < 0 or world >= world_capacity.shape[0] or local < 0 or local >= world_capacity[world]:
        return

    destination = world_offset[world] + local
    bodies = source_bodies[source]
    first_global = bodies[0]
    second_global = bodies[1]
    first_dynamic = first_global >= 0 and body_vector_index[first_global] >= 0
    second_dynamic = second_global >= 0 and body_vector_index[second_global] >= 0
    if not first_dynamic and not second_dynamic:
        body_first[destination] = -1
        body_second[destination] = -1
        jacobian_first[destination] = vec6f(0.0)
        jacobian_second[destination] = vec6f(0.0)
        bias[destination] = 0.0
        reaction[destination] = 0.0
        velocity[destination] = 0.0
        return
    sparse_offset = sparse_jacobian_offsets[source]
    first_sparse_index = sparse_offset + 1 if first_global >= 0 else -1
    first_jacobian = _load_sparse_jacobian_row(first_sparse_index, sparse_jacobian_data)
    second_jacobian = _load_sparse_jacobian_row(sparse_offset, sparse_jacobian_data)
    body_first[destination] = first_global
    body_second[destination] = second_global
    jacobian_first[destination] = first_jacobian
    jacobian_second[destination] = second_jacobian
    velocity_previous = wp.float32(0.0)
    if first_global >= 0:
        velocity_previous += wp.dot(first_jacobian, body_velocity_begin[first_global])
    if second_global >= 0:
        velocity_previous += wp.dot(second_jacobian, body_velocity_begin[second_global])
    dt = time_step[world]
    target = compute_limit_velocity_target(source_violation[source], dt, stabilization_fraction)
    bias[destination] = -target
    if import_reactions:
        reaction[destination] = dt * source_reaction[source]
        velocity[destination] = velocity_previous


@wp.kernel
def _prepare_contacts(
    source_active: wp.array[wp.int32],
    source_capacity: wp.int32,
    source_world: wp.array[wp.int32],
    source_local: wp.array[wp.int32],
    source_bodies: wp.array[wp.vec2i],
    source_position_a: wp.array[wp.vec3f],
    source_position_b: wp.array[wp.vec3f],
    source_frame: wp.array[wp.quatf],
    source_gap: wp.array[wp.vec4f],
    source_material: wp.array[wp.vec2f],
    source_reaction: wp.array[wp.vec3f],
    body_pose: wp.array[wp.transformf],
    body_velocity_begin: wp.array[vec6f],
    world_capacity: wp.array[wp.int32],
    world_count: wp.array[wp.int32],
    world_offset: wp.array[wp.int32],
    body_vector_index: wp.array[wp.int32],
    time_step: wp.array[wp.float32],
    stabilization_fraction: wp.float32,
    dead_zone: wp.float32,
    impact_velocity_threshold: wp.float32,
    recoverable_response: wp.bool,
    import_reactions: wp.bool,
    compact_contacts: wp.bool,
    source_to_internal: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[mat36f],
    jacobian_second: wp.array[mat36f],
    bias: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    reaction: wp.array[wp.vec3f],
    velocity: wp.array[wp.vec3f],
):
    source = wp.tid()
    if source >= wp.min(source_active[0], source_capacity):
        return
    world = source_world[source]
    local = source_local[source]
    if world < 0 or world >= world_capacity.shape[0] or local < 0 or local >= world_capacity[world]:
        return

    bodies = source_bodies[source]
    first_global = bodies[0]
    second_global = bodies[1]
    first_dynamic = first_global >= 0 and body_vector_index[first_global] >= 0
    second_dynamic = second_global >= 0 and body_vector_index[second_global] >= 0
    if not first_dynamic and not second_dynamic:
        source_to_internal[source] = -1
        return
    destination = source_to_internal[source]
    if compact_contacts:
        destination = world_offset[world] + wp.atomic_add(world_count, world, 1)
        source_to_internal[source] = destination
    first_jacobian = mat36f(0.0)
    second_jacobian = mat36f(0.0)
    rotation = wp.quat_to_matrix(source_frame[source])
    body_position_b = wp.transform_get_translation(body_pose[second_global])
    jacobian_transpose_b = contact_wrench_matrix_from_points(source_position_b[source], body_position_b) @ rotation
    for component in range(3):
        for dof in range(6):
            second_jacobian[component, dof] = jacobian_transpose_b[dof, component]
    if first_global >= 0:
        body_position_a = wp.transform_get_translation(body_pose[first_global])
        jacobian_transpose_a = -contact_wrench_matrix_from_points(source_position_a[source], body_position_a) @ rotation
        for component in range(3):
            for dof in range(6):
                first_jacobian[component, dof] = jacobian_transpose_a[dof, component]

    velocity_previous = wp.vec3f(0.0)
    if first_global >= 0:
        velocity_previous += first_jacobian @ body_velocity_begin[first_global]
    if second_global >= 0:
        velocity_previous += second_jacobian @ body_velocity_begin[second_global]
    gap = source_gap[source]
    material = source_material[source]
    dt = time_step[world]
    target = compute_contact_velocity_target(
        gap[3],
        velocity_previous[2],
        material[1],
        dt,
        stabilization_fraction,
        dead_zone,
        impact_velocity_threshold,
        recoverable_response,
    )

    body_first[destination] = first_global
    body_second[destination] = second_global
    jacobian_first[destination] = first_jacobian
    jacobian_second[destination] = second_jacobian
    bias[destination] = wp.vec3f(0.0, 0.0, -target)
    friction[destination] = material[0]
    if import_reactions:
        reaction[destination] = dt * source_reaction[source]
        velocity[destination] = velocity_previous


@wp.kernel
def _reset_structural_multipliers_masked(
    row_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    reaction: wp.array[wp.float32],
):
    row = wp.tid()
    if world_mask[row_world[row]]:
        reaction[row] = 0.0


@wp.kernel
def _reset_effort_rows_masked(
    effort_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    effort_intercept: wp.array[wp.float32],
    effort_slope: wp.array[wp.float32],
    effort_impulse_bound: wp.array[wp.float32],
    effort_raw_impulse: wp.array[wp.float32],
    effort_counter_applied: wp.array[wp.float32],
    effort_counter_next: wp.array[wp.float32],
    effort_net_applied: wp.array[wp.float32],
    effort_net_target: wp.array[wp.float32],
    effort_velocity: wp.array[wp.float32],
    effort_residual: wp.array[wp.float32],
):
    effort = wp.tid()
    if world_mask[effort_world[effort]]:
        effort_intercept[effort] = 0.0
        effort_slope[effort] = 0.0
        effort_impulse_bound[effort] = 0.0
        effort_raw_impulse[effort] = 0.0
        effort_counter_applied[effort] = 0.0
        effort_counter_next[effort] = 0.0
        effort_net_applied[effort] = 0.0
        effort_net_target[effort] = 0.0
        effort_velocity[effort] = 0.0
        effort_residual[effort] = 0.0


@wp.kernel
def _reset_effort_worlds_masked(
    world_mask: wp.array[wp.bool],
    world_effort_residual_max: wp.array[wp.float32],
    world_effort_defect_max: wp.array[wp.float32],
):
    world = wp.tid()
    if world_mask[world]:
        world_effort_residual_max[world] = 0.0
        world_effort_defect_max[world] = 0.0


@wp.kernel
def _reset_friction_reactions_masked(
    friction_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    friction_reaction: wp.array[wp.float32],
):
    friction = wp.tid()
    if world_mask[friction_world[friction]]:
        friction_reaction[friction] = 0.0


@wp.kernel
def _scale_structural_reactions(
    scale: wp.float32,
    reaction: wp.array[wp.float32],
):
    row = wp.tid()
    reaction[row] *= scale


@wp.kernel
def _write_dynamic_outputs(
    inverse_time_step: wp.array[wp.float32],
    row_world: wp.array[wp.int32],
    multiplier_index: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    effective_inertia: wp.array[wp.float32],
    free_velocity: wp.array[wp.float32],
    body_velocity: wp.array[vec6f],
    destination: wp.array[wp.float32],
    destination_wrench: wp.array[vec6f],
):
    row = wp.tid()
    velocity = wp.float32(0.0)
    first = body_first[row]
    second = body_second[row]
    if first >= 0:
        velocity += wp.dot(jacobian_first[row], body_velocity[first])
    if second >= 0:
        velocity += wp.dot(jacobian_second[row], body_velocity[second])
    force = inverse_time_step[row_world[row]] * effective_inertia[row] * (free_velocity[row] - velocity)
    destination[multiplier_index[row]] = force
    if first >= 0:
        wp.atomic_add(destination_wrench, first, force * jacobian_first[row])
    if second >= 0:
        wp.atomic_add(destination_wrench, second, force * jacobian_second[row])


@wp.kernel
def _write_dynamic_outputs_with_effort(
    inverse_time_step: wp.array[wp.float32],
    row_world: wp.array[wp.int32],
    multiplier_index: wp.array[wp.int32],
    effort_index: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    effective_inertia: wp.array[wp.float32],
    free_velocity: wp.array[wp.float32],
    effort_counter_applied: wp.array[wp.float32],
    effort_net_applied: wp.array[wp.float32],
    effort_value_index: wp.array[wp.int32],
    body_velocity: wp.array[vec6f],
    destination: wp.array[wp.float32],
    effort_destination: wp.array[wp.float32],
    destination_wrench: wp.array[vec6f],
):
    row = wp.tid()
    velocity = wp.float32(0.0)
    first = body_first[row]
    second = body_second[row]
    if first >= 0:
        velocity += wp.dot(jacobian_first[row], body_velocity[first])
    if second >= 0:
        velocity += wp.dot(jacobian_second[row], body_velocity[second])
    inv_dt = inverse_time_step[row_world[row]]
    multiplier = inv_dt * effective_inertia[row] * (free_velocity[row] - velocity)
    bounded = effort_index[row]
    if bounded >= 0:
        multiplier += inv_dt * effort_counter_applied[bounded]
        effort_destination[effort_value_index[bounded]] = inv_dt * effort_net_applied[bounded]
    destination_index = multiplier_index[row]
    if destination_index >= 0:
        destination[destination_index] = multiplier
    elif bounded >= 0:
        multiplier = inv_dt * effort_net_applied[bounded]
    else:
        return
    if first >= 0:
        wp.atomic_add(destination_wrench, first, multiplier * jacobian_first[row])
    if second >= 0:
        wp.atomic_add(destination_wrench, second, multiplier * jacobian_second[row])


@wp.kernel
def _write_friction_outputs(
    inverse_time_step: wp.array[wp.float32],
    row_world: wp.array[wp.int32],
    multiplier_index: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    source: wp.array[wp.float32],
    destination: wp.array[wp.float32],
    destination_wrench: wp.array[vec6f],
):
    row = wp.tid()
    force = inverse_time_step[row_world[row]] * source[row]
    destination[multiplier_index[row]] = force
    first = body_first[row]
    second = body_second[row]
    if first >= 0:
        wp.atomic_add(destination_wrench, first, force * jacobian_first[row])
    if second >= 0:
        wp.atomic_add(destination_wrench, second, force * jacobian_second[row])


@wp.kernel
def _accumulate_aligned_joint_wrenches(
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    reaction: wp.array[wp.float32],
    destination: wp.array[vec6f],
):
    row = wp.tid()
    scale = reaction[row]
    first = body_first[row]
    second = body_second[row]
    if first >= 0:
        wp.atomic_add(destination, first, scale * jacobian_first[row])
    if second >= 0:
        wp.atomic_add(destination, second, scale * jacobian_second[row])


@wp.kernel
def _update_structural_multipliers_from_candidate_rows(
    time_step: wp.array[wp.float32],
    structural_tolerance: wp.float32,
    row_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    body_first_global: wp.array[wp.int32],
    body_second_global: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    candidate_residual: wp.array[wp.float32],
    proximal_defect: wp.array[wp.float32],
    frozen_residual: wp.array[wp.float32],
    penalty: wp.array[wp.float32],
    linearization_twist: wp.array[vec6f],
    candidate_twist: wp.array[vec6f],
    proximal_relaxation: wp.float32,
    body_vector_index: wp.array[wp.int32],
    reaction: wp.array[wp.float32],
    world_residual: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
):
    row = wp.tid()
    world = row_world[row]
    dt = time_step[world]
    if not world_active[world] or projection_status[world] != PROJECTION_STATUS_VALID:
        return

    first_global = body_first_global[row]
    second_global = body_second_global[row]
    first_dynamic = first_global >= 0 and body_vector_index[first_global] >= 0
    second_dynamic = second_global >= 0 and body_vector_index[second_global] >= 0
    if not first_dynamic and not second_dynamic:
        reaction[row] = 0.0
        return
    candidate_velocity = wp.float32(0.0)
    linearization_velocity = wp.float32(0.0)
    if first_global >= 0:
        candidate_velocity += wp.dot(jacobian_first[row], candidate_twist[first_global])
        linearization_velocity += wp.dot(jacobian_first[row], linearization_twist[first_global])
    if second_global >= 0:
        candidate_velocity += wp.dot(jacobian_second[row], candidate_twist[second_global])
        linearization_velocity += wp.dot(jacobian_second[row], linearization_twist[second_global])

    linear_residual = frozen_residual[row] + dt * (candidate_velocity - linearization_velocity)
    update_residual = linear_residual
    if proximal_relaxation > 0.0:
        defect = proximal_defect[row]
        defect += proximal_relaxation * (candidate_residual[row] - linear_residual - defect)
        proximal_defect[row] = defect
        update_residual += defect
    candidate_residual[row] = update_residual
    wp.atomic_max(world_residual, world, wp.abs(update_residual) / structural_tolerance)

    reaction_delta = -penalty[row] * update_residual
    reaction[row] += reaction_delta

    for axis in range(6):
        if first_global >= 0 and body_vector_index[first_global] >= 0:
            wp.atomic_add(
                right_hand_side,
                body_vector_index[first_global] + axis,
                dt * reaction_delta * jacobian_first[row][axis],
            )
        if second_global >= 0 and body_vector_index[second_global] >= 0:
            wp.atomic_add(
                right_hand_side,
                body_vector_index[second_global] + axis,
                dt * reaction_delta * jacobian_second[row][axis],
            )


@wp.kernel
def _reduce_structural_candidate_residual(
    structural_tolerance: wp.float32,
    row_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    body_first_global: wp.array[wp.int32],
    body_second_global: wp.array[wp.int32],
    candidate_residual: wp.array[wp.float32],
    body_vector_index: wp.array[wp.int32],
    world_residual: wp.array[wp.float32],
):
    row = wp.tid()
    world = row_world[row]
    if not world_active[world] or projection_status[world] != PROJECTION_STATUS_VALID:
        return

    first = body_first_global[row]
    second = body_second_global[row]
    first_dynamic = first >= 0 and body_vector_index[first] >= 0
    second_dynamic = second >= 0 and body_vector_index[second] >= 0
    if not first_dynamic and not second_dynamic:
        return
    wp.atomic_max(world_residual, world, wp.abs(candidate_residual[row]) / structural_tolerance)


@wp.kernel
def _write_limit_outputs(
    source_active: wp.array[wp.int32],
    source_capacity: wp.int32,
    source_world: wp.array[wp.int32],
    source_local: wp.array[wp.int32],
    world_capacity: wp.array[wp.int32],
    world_offset: wp.array[wp.int32],
    inverse_time_step: wp.array[wp.float32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    reaction: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
    destination_reaction: wp.array[wp.float32],
    destination_velocity: wp.array[wp.float32],
    destination_wrench: wp.array[vec6f],
):
    source = wp.tid()
    if source >= wp.min(source_active[0], source_capacity):
        return
    world = source_world[source]
    local = source_local[source]
    if world < 0 or world >= world_capacity.shape[0] or local < 0 or local >= world_capacity[world]:
        return
    internal = world_offset[world] + local
    force = inverse_time_step[world] * reaction[internal]
    destination_reaction[source] = force
    destination_velocity[source] = velocity[internal]
    first = body_first[internal]
    second = body_second[internal]
    if first >= 0:
        wp.atomic_add(destination_wrench, first, force * jacobian_first[internal])
    if second >= 0:
        wp.atomic_add(destination_wrench, second, force * jacobian_second[internal])


@wp.kernel
def _write_contact_outputs(
    source_active: wp.array[wp.int32],
    source_capacity: wp.int32,
    source_world: wp.array[wp.int32],
    source_to_internal: wp.array[wp.int32],
    inverse_time_step: wp.array[wp.float32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[mat36f],
    jacobian_second: wp.array[mat36f],
    reaction: wp.array[wp.vec3f],
    velocity: wp.array[wp.vec3f],
    destination_reaction: wp.array[wp.vec3f],
    destination_velocity: wp.array[wp.vec3f],
    destination_mode: wp.array[wp.int32],
    destination_wrench: wp.array[vec6f],
):
    source = wp.tid()
    if source >= wp.min(source_active[0], source_capacity):
        return
    internal = source_to_internal[source]
    if internal < 0:
        zero_velocity = wp.vec3f(0.0)
        destination_reaction[source] = wp.vec3f(0.0)
        destination_velocity[source] = zero_velocity
        destination_mode[source] = wp.static(ContactMode.make_compute_mode_func())(zero_velocity)
        return
    world = source_world[source]
    force = inverse_time_step[world] * reaction[internal]
    destination_reaction[source] = force
    destination_velocity[source] = velocity[internal]
    destination_mode[source] = wp.static(ContactMode.make_compute_mode_func())(velocity[internal])
    first = body_first[internal]
    second = body_second[internal]
    if first >= 0:
        wp.atomic_add(destination_wrench, first, wp.transpose(jacobian_first[internal]) @ force)
    if second >= 0:
        wp.atomic_add(destination_wrench, second, wp.transpose(jacobian_second[internal]) @ force)
