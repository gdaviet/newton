# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernels of the LOX rod material rows.

The loading kernel evaluates the strain, stress, tangent stiffness, and strain-rate
Jacobians of the six rows of each rod at the beginning of the time step. The proximal
kernel updates the bend and twist rows from the exact candidate-pose strain during the
iterations, and the output kernel adds the rod wrenches to the Kamino joint wrenches.
"""

import warp as wp

from ......solvers.vbd.rigid_vbd_kernels import (
    _assemble_geometric_rod_kappa_z,
    _measure_rod_bend_twist_z,
    _rod_bend_twist_jacobian_z_from_measure,
)
from ...core.math import compute_body_pose_update_with_logmap
from ...core.types import vec6f

###
# Module interface
###

__all__ = [
    "_accumulate_rod_wrenches",
    "_initialize_rod_rest_state",
    "_load_rod_rows",
    "_update_rod_proximal",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


@wp.func
def _joint_pose(
    bid: wp.int32,
    body_pose: wp.array[wp.transformf],
    position: wp.vec3f,
    orientation: wp.mat33f,
) -> wp.transformf:
    """Return the world pose of a joint frame on a body, or of a world-attached frame for ``bid < 0``."""
    local_pose = wp.transformf(position, wp.quat_from_matrix(orientation))
    if bid >= 0:
        return body_pose[bid] * local_pose
    return local_pose


@wp.func
def _candidate_joint_pose(
    bid: wp.int32,
    body_pose: wp.array[wp.transformf],
    body_twist: wp.array[vec6f],
    dt: wp.float32,
    position: wp.vec3f,
    orientation: wp.mat33f,
) -> wp.transformf:
    """Return the world pose of a joint frame after integrating the body twist over ``dt``."""
    local_pose = wp.transformf(position, wp.quat_from_matrix(orientation))
    if bid < 0:
        return local_pose
    twist = body_twist[bid]
    candidate_pose = compute_body_pose_update_with_logmap(
        dt,
        body_pose[bid],
        wp.vec3f(twist[0], twist[1], twist[2]),
        wp.vec3f(twist[3], twist[4], twist[5]),
    )
    return candidate_pose * local_pose


@wp.func
def _linear_rod_strain(pose_a: wp.transformf, pose_b: wp.transformf) -> wp.vec3f:
    """Return the stretch and shear strain: the joint separation in the joint frame of body A."""
    separation = wp.transform_get_translation(pose_b) - wp.transform_get_translation(pose_a)
    return wp.quat_rotate_inv(wp.transform_get_rotation(pose_a), separation)


@wp.func
def _jacobian_row(linear: wp.vec3f, angular: wp.vec3f) -> vec6f:
    return vec6f(linear[0], linear[1], linear[2], angular[0], angular[1], angular[2])


###
# Kernels
###


@wp.kernel
def _initialize_rod_rest_state(
    # Inputs:
    rod_joint: wp.array[wp.int32],
    joint_target_q_start: wp.array[wp.int32],
    joint_target_q: wp.array[wp.float32],
    # Outputs:
    rest_curvature: wp.array[wp.vec3f],
    rest_twist: wp.array[wp.float32],
):
    """Store the rest curvature and twist of each rod from its structural-rest rotation.

    The rest rotation of the child anchor relative to the parent anchor is the angular part
    of the joint target, a quaternion in the coordinate layout as in :class:`~newton.solvers.SolverVBD`.
    """
    rod = wp.tid()
    start = joint_target_q_start[rod_joint[rod]]
    rest_rotation = wp.quatf(
        joint_target_q[start + 3],
        joint_target_q[start + 4],
        joint_target_q[start + 5],
        joint_target_q[start + 6],
    )
    measure = _measure_rod_bend_twist_z(wp.quat_identity(), wp.normalize(rest_rotation))
    rest_curvature[rod] = measure.kb_world
    rest_twist[rod] = measure.twist


@wp.kernel
def _load_rod_rows(
    # Inputs:
    model_time_dt: wp.array[wp.float32],
    model_bodies_wid: wp.array[wp.int32],
    rod_joint: wp.array[wp.int32],
    rod_body_a: wp.array[wp.int32],
    rod_body_b: wp.array[wp.int32],
    rod_dof_offset: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    model_joints_B_r_Bj: wp.array[wp.vec3f],
    model_joints_F_r_Fj: wp.array[wp.vec3f],
    model_joints_X_Bj: wp.array[wp.mat33f],
    model_joints_X_Fj: wp.array[wp.mat33f],
    model_joints_k_p_j: wp.array[wp.float32],
    model_joints_k_d_j: wp.array[wp.float32],
    data_bodies_q_i: wp.array[wp.transformf],
    body_twist: wp.array[vec6f],
    rest_curvature: wp.array[wp.vec3f],
    rest_twist: wp.array[wp.float32],
    # Outputs:
    strain: wp.array[wp.float32],
    stress: wp.array[wp.float32],
    tangent: wp.array[wp.float32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
):
    """Evaluate the six material rows of each rod at the current body poses.

    The damping stress uses the strain rate of ``body_twist``, or zero when it is ``None``.
    Disabled rods write zero rows.
    """
    rod = wp.tid()
    jid = rod_joint[rod]
    row_offset = 6 * rod
    if not joint_enabled[jid]:
        for local_row in range(6):
            row = row_offset + local_row
            strain[row] = 0.0
            stress[row] = 0.0
            tangent[row] = 0.0
            jacobian_a[row] = vec6f(0.0)
            jacobian_b[row] = vec6f(0.0)
        return

    bid_a = rod_body_a[rod]
    bid_b = rod_body_b[rod]
    dt = model_time_dt[model_bodies_wid[bid_b]]
    pose_a = _joint_pose(bid_a, data_bodies_q_i, model_joints_B_r_Bj[jid], model_joints_X_Bj[jid])
    pose_b = _joint_pose(bid_b, data_bodies_q_i, model_joints_F_r_Fj[jid], model_joints_X_Fj[jid])
    orientation_a = wp.transform_get_rotation(pose_a)
    linear_strain = _linear_rod_strain(pose_a, pose_b)
    measure = _measure_rod_bend_twist_z(orientation_a, wp.transform_get_rotation(pose_b))
    angular_strain = _assemble_geometric_rod_kappa_z(
        orientation_a, measure.kb_world, measure.twist, rest_curvature[rod], rest_twist[rod]
    )

    # The rod DOFs hold the stretch, shear, bend, and twist coefficients; the two shear
    # and the two bend rows share one coefficient
    dof = rod_dof_offset[rod]
    # Material slots follow the canonical axes: shear x, shear y, stretch z, bend x, bend y, twist z.
    linear_stiffness = wp.vec3f(model_joints_k_p_j[dof], model_joints_k_p_j[dof + 1], model_joints_k_p_j[dof + 2])
    angular_stiffness = wp.vec3f(model_joints_k_p_j[dof + 3], model_joints_k_p_j[dof + 4], model_joints_k_p_j[dof + 5])
    linear_damping = wp.vec3f(model_joints_k_d_j[dof], model_joints_k_d_j[dof + 1], model_joints_k_d_j[dof + 2])
    angular_damping = wp.vec3f(model_joints_k_d_j[dof + 3], model_joints_k_d_j[dof + 4], model_joints_k_d_j[dof + 5])

    # Stretch and shear rows along the joint axes of body A, at the joint anchor of body B
    anchor = wp.transform_get_translation(pose_b)
    com_b = wp.transform_get_translation(data_bodies_q_i[bid_b])
    com_a = wp.vec3f(0.0)
    if bid_a >= 0:
        com_a = wp.transform_get_translation(data_bodies_q_i[bid_a])
    material_axes = wp.quat_to_matrix(orientation_a)
    for local_row in range(3):
        axis = wp.vec3f(material_axes[0, local_row], material_axes[1, local_row], material_axes[2, local_row])
        jacobian_a[row_offset + local_row] = _jacobian_row(-axis, wp.cross(axis, anchor - com_a))
        jacobian_b[row_offset + local_row] = _jacobian_row(axis, wp.cross(anchor - com_b, axis))

    # Bend and twist rows
    angular_jacobian_a = _rod_bend_twist_jacobian_z_from_measure(measure, True)
    angular_jacobian_b = _rod_bend_twist_jacobian_z_from_measure(measure, False)
    for local_row in range(3):
        row_a = wp.vec3f(
            angular_jacobian_a[local_row, 0], angular_jacobian_a[local_row, 1], angular_jacobian_a[local_row, 2]
        )
        row_b = wp.vec3f(
            angular_jacobian_b[local_row, 0], angular_jacobian_b[local_row, 1], angular_jacobian_b[local_row, 2]
        )
        jacobian_a[row_offset + 3 + local_row] = _jacobian_row(wp.vec3f(0.0), row_a)
        jacobian_b[row_offset + 3 + local_row] = _jacobian_row(wp.vec3f(0.0), row_b)

    inv_dt = 1.0 / dt
    for local_row in range(6):
        row = row_offset + local_row
        rate = wp.float32(0.0)
        if body_twist:
            rate = wp.dot(jacobian_b[row], body_twist[bid_b])
            if bid_a >= 0:
                rate += wp.dot(jacobian_a[row], body_twist[bid_a])
        value = wp.float32(0.0)
        stiffness = wp.float32(0.0)
        damping = wp.float32(0.0)
        if local_row < 3:
            value = linear_strain[local_row]
            stiffness = linear_stiffness[local_row]
            damping = linear_damping[local_row]
        else:
            value = angular_strain[local_row - 3]
            stiffness = angular_stiffness[local_row - 3]
            damping = angular_damping[local_row - 3]
        strain[row] = value
        stress[row] = stiffness * value + damping * rate
        tangent[row] = stiffness + damping * inv_dt


@wp.kernel
def _update_rod_proximal(
    # Inputs:
    model_time_dt: wp.array[wp.float32],
    model_bodies_wid: wp.array[wp.int32],
    rotation_tolerance: wp.float32,
    velocity_tolerance: wp.float32,
    proximal_relaxation: wp.float32,
    rod_joint: wp.array[wp.int32],
    rod_body_a: wp.array[wp.int32],
    rod_body_b: wp.array[wp.int32],
    rod_dof_offset: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    model_joints_B_r_Bj: wp.array[wp.vec3f],
    model_joints_F_r_Fj: wp.array[wp.vec3f],
    model_joints_X_Bj: wp.array[wp.mat33f],
    model_joints_X_Fj: wp.array[wp.mat33f],
    model_joints_k_p_j: wp.array[wp.float32],
    data_bodies_q_i: wp.array[wp.transformf],
    global_twist: wp.array[vec6f],
    world_active: wp.array[wp.bool],
    rest_curvature: wp.array[wp.vec3f],
    rest_twist: wp.array[wp.float32],
    frozen_strain: wp.array[wp.float32],
    frozen_stress: wp.array[wp.float32],
    tangent: wp.array[wp.float32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    # Outputs:
    proximal_coordinate: wp.array[wp.float32],
    multiplier: wp.array[wp.float32],
    body_impulse: wp.array[vec6f],
    world_residual: wp.array[wp.float32],
    world_failed: wp.array[wp.bool],
):
    """Update the bend and twist proximal rows of each rod and scatter the correction of the next candidate solve.

    The proximal coordinate of each row averages the exact candidate-pose strain and the
    frozen strain shifted by the stress mismatch; the multiplier relaxes towards the
    tangent stress at that coordinate. The stretch and shear rows keep their linearly
    implicit stress. The twist row follows the candidate twist continuously across the
    principal angle branches, as predicted by the frozen twist Jacobian. A non-finite
    update fails the world and leaves the rows of the rod unchanged.
    """
    rod = wp.tid()
    jid = rod_joint[rod]
    if not joint_enabled[jid]:
        return
    bid_a = rod_body_a[rod]
    bid_b = rod_body_b[rod]
    wid = model_bodies_wid[bid_b]
    if not world_active[wid] or world_failed[wid]:
        return
    dt = model_time_dt[wid]

    pose_a = _candidate_joint_pose(
        bid_a, data_bodies_q_i, global_twist, dt, model_joints_B_r_Bj[jid], model_joints_X_Bj[jid]
    )
    pose_b = _candidate_joint_pose(
        bid_b, data_bodies_q_i, global_twist, dt, model_joints_F_r_Fj[jid], model_joints_X_Fj[jid]
    )
    orientation_a = wp.transform_get_rotation(pose_a)
    linear_strain = _linear_rod_strain(pose_a, pose_b)
    measure = _measure_rod_bend_twist_z(orientation_a, wp.transform_get_rotation(pose_b))
    angular_strain = _assemble_geometric_rod_kappa_z(
        orientation_a, measure.kb_world, measure.twist, rest_curvature[rod], rest_twist[rod]
    )
    candidate_strain = vec6f(
        linear_strain[0], linear_strain[1], linear_strain[2], angular_strain[0], angular_strain[1], angular_strain[2]
    )

    # Lift the principal twist change to the branch of the twist predicted by the frozen Jacobian
    row_offset = 6 * rod
    twist_row = row_offset + 5
    predicted_twist_change = dt * wp.dot(jacobian_b[twist_row], global_twist[bid_b])
    if bid_a >= 0:
        predicted_twist_change += dt * wp.dot(jacobian_a[twist_row], global_twist[bid_a])
    principal_twist_change = candidate_strain[5] - frozen_strain[twist_row]
    lifted_twist_change = principal_twist_change + wp.tau * wp.round(
        (predicted_twist_change - principal_twist_change) / wp.tau
    )

    coordinate_new = vec6f(0.0)
    multiplier_new = vec6f(0.0)
    correction = vec6f(0.0)
    residual = wp.float32(0.0)
    finite = wp.isfinite(predicted_twist_change) and wp.isfinite(lifted_twist_change)
    for local_row in range(6):
        row = row_offset + local_row
        row_tangent = tangent[row]
        frozen = frozen_strain[row]
        center = candidate_strain[local_row]
        material_offset = wp.float32(0.0)
        if local_row == 5:
            center = frozen + lifted_twist_change
            # The elastic twist strain stays principal while the proximal coordinate follows
            # the lifted twist; this affine offset makes their stresses equal at consensus
            twist_stiffness = model_joints_k_p_j[rod_dof_offset[rod] + 5]
            material_offset = twist_stiffness * (candidate_strain[5] - center)
        coordinate = center
        row_multiplier = wp.float32(0.0)
        row_correction = wp.float32(0.0)
        # The stiff stretch and shear rows stay linearly implicit, which keeps a small
        # rotational mismatch of the frozen geometry out of their axial wrench
        if row_tangent > 0.0 and local_row >= 3:
            stress = frozen_stress[row]
            coordinate = 0.5 * (frozen + center) + 0.5 * (multiplier[row] - stress - material_offset) / row_tangent
            row_multiplier = multiplier[row] + proximal_relaxation * row_tangent * (center - coordinate)
            linearized_change = wp.dot(jacobian_b[row], global_twist[bid_b])
            if bid_a >= 0:
                linearized_change += wp.dot(jacobian_a[row], global_twist[bid_a])
            row_correction = stress + dt * row_tangent * linearized_change - row_multiplier
            residual = wp.max(residual, wp.abs(center - coordinate) / rotation_tolerance)
            residual = wp.max(residual, wp.abs(coordinate - proximal_coordinate[row]) / (dt * velocity_tolerance))
        coordinate_new[local_row] = coordinate
        multiplier_new[local_row] = row_multiplier
        correction[local_row] = row_correction
        finite = (
            finite
            and wp.isfinite(center)
            and wp.isfinite(coordinate)
            and wp.isfinite(row_multiplier)
            and wp.isfinite(row_correction)
        )
    if not finite or not wp.isfinite(residual):
        world_failed[wid] = True
        return

    impulse_a = vec6f(0.0)
    impulse_b = vec6f(0.0)
    for local_row in range(6):
        row = row_offset + local_row
        scale = dt * correction[local_row]
        impulse_a += scale * jacobian_a[row]
        impulse_b += scale * jacobian_b[row]
        proximal_coordinate[row] = coordinate_new[local_row]
        multiplier[row] = multiplier_new[local_row]
    if bid_a >= 0:
        wp.atomic_add(body_impulse, bid_a, impulse_a)
    wp.atomic_add(body_impulse, bid_b, impulse_b)
    wp.atomic_max(world_residual, wid, residual)


@wp.kernel
def _accumulate_rod_wrenches(
    # Inputs:
    model_bodies_wid: wp.array[wp.int32],
    rod_body_a: wp.array[wp.int32],
    rod_body_b: wp.array[wp.int32],
    stress: wp.array[wp.float32],
    jacobian_a: wp.array[vec6f],
    jacobian_b: wp.array[vec6f],
    world_failed: wp.array[wp.bool],
    # Outputs:
    data_bodies_w_j_i: wp.array[vec6f],
):
    """Add the material wrench of each rod of the non-failed worlds to the joint wrenches of its bodies."""
    rod = wp.tid()
    bid_a = rod_body_a[rod]
    bid_b = rod_body_b[rod]
    if world_failed[model_bodies_wid[bid_b]]:
        return
    wrench_a = vec6f(0.0)
    wrench_b = vec6f(0.0)
    row_offset = 6 * rod
    for local_row in range(6):
        row = row_offset + local_row
        wrench_a -= stress[row] * jacobian_a[row]
        wrench_b -= stress[row] * jacobian_b[row]
    if bid_a >= 0:
        wp.atomic_add(data_bodies_w_j_i, bid_a, wrench_a)
    wp.atomic_add(data_bodies_w_j_i, bid_b, wrench_b)
