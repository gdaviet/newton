# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smooth rod material elements for the LOX rigid-body backend."""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _assemble_geometric_rod_kappa_z,
    _measure_rod_bend_twist_z,
    _rod_bend_twist_jacobian_z_from_measure,
)

from ......core.types import MAXVAL
from ......sim import JointTargetMode, JointType, Model
from ...core.data import DataKamino
from ...core.joints import JointDoFType
from ...core.math import compute_body_pose_update_with_logmap
from ...core.model import ModelKamino
from ...core.types import vec6f
from .system import BatchedPrimalBodySystem
from .time import validate_world_time_step

__all__ = ["RodMaterialSystem", "validate_rod_model"]

wp.set_module_options({"enable_backward": False})


def validate_rod_model(model: Model, *, use_fk_solver: bool) -> None:
    """Validate the model subset supported by LOX rod materials."""
    if model.joint_count == 0:
        return

    rod_joints = np.flatnonzero(model.joint_type.numpy() == int(JointType.ROD))
    if rod_joints.size == 0:
        return
    if use_fk_solver:
        raise ValueError("SolverKamino rod joints do not support use_fk_solver=True.")

    joint_world = model.joint_world.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    body_world = model.body_world.numpy()
    joint_dof_dim = model.joint_dof_dim.numpy()
    joint_q_start = model.joint_q_start.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    armature = model.joint_armature.numpy()
    damping = model.joint_damping.numpy()
    friction = model.joint_friction.numpy()
    target_ke = model.joint_target_ke.numpy()
    target_kd = model.joint_target_kd.numpy()
    target_mode = model.joint_target_mode.numpy()
    limit_lower = model.joint_limit_lower.numpy()
    limit_upper = model.joint_limit_upper.numpy()

    def normalized_world(world: int) -> int:
        return 0 if model.world_count == 1 and world < 0 else world

    for rod_joint in rod_joints:
        joint = int(rod_joint)
        world = normalized_world(int(joint_world[joint]))
        parent = int(joint_parent[joint])
        child = int(joint_child[joint])
        q_count = int(joint_q_start[joint + 1] - joint_q_start[joint])
        dof_start = int(joint_qd_start[joint])
        dof_end = int(joint_qd_start[joint + 1])
        qd_count = dof_end - dof_start
        dof_dim = (int(joint_dof_dim[joint, 0]), int(joint_dof_dim[joint, 1]))

        if q_count != 4 or qd_count != 4 or dof_dim != (2, 2):
            raise ValueError(
                f"SolverKamino rod joint {joint} requires q_count=4, qd_count=4, "
                f"and dof_dim=(2, 2); got q_count={q_count}, qd_count={qd_count}, "
                f"and dof_dim={dof_dim}."
            )
        if child < 0 or child >= model.body_count:
            raise ValueError(f"SolverKamino rod joint {joint} has invalid child body {child}.")
        if world < 0 or world >= model.world_count or normalized_world(int(body_world[child])) != world:
            raise ValueError(f"SolverKamino rod joint {joint} and child body {child} must share a world.")
        if parent >= model.body_count or parent < -1:
            raise ValueError(f"SolverKamino rod joint {joint} has invalid parent body {parent}.")
        if parent >= 0 and normalized_world(int(body_world[parent])) != world:
            raise ValueError(f"SolverKamino rod joint {joint} and parent body {parent} must share a world.")

        rod_ke = target_ke[dof_start:dof_end]
        rod_kd = target_kd[dof_start:dof_end]
        if not np.isfinite(rod_ke).all() or np.any(rod_ke < 0.0):
            raise ValueError(f"SolverKamino rod joint {joint} stiffness values must be finite and nonnegative.")
        if not np.isfinite(rod_kd).all() or np.any(rod_kd < 0.0):
            raise ValueError(f"SolverKamino rod joint {joint} damping values must be finite and nonnegative.")

        unsupported = {
            "joint_armature": armature[dof_start:dof_end],
            "joint_damping": damping[dof_start:dof_end],
            "joint_friction": friction[dof_start:dof_end],
        }
        for name, values in unsupported.items():
            if not np.isfinite(values).all() or np.any(values != 0.0):
                raise ValueError(f"SolverKamino rod joint {joint} does not support nonzero {name}.")

        if np.any(limit_lower[dof_start:dof_end] > -MAXVAL) or np.any(limit_upper[dof_start:dof_end] < MAXVAL):
            raise ValueError(f"SolverKamino rod joint {joint} does not support finite joint limits.")
        if np.any(target_mode[dof_start:dof_end] == int(JointTargetMode.EFFORT)):
            raise ValueError(f"SolverKamino rod joint {joint} does not support force control.")


@wp.func
def _local_joint_pose(position: wp.vec3f, orientation: wp.mat33f) -> wp.transformf:
    return wp.transformf(position, wp.quat_from_matrix(orientation))


@wp.func
def _joint_pose(
    body: wp.int32,
    body_pose: wp.array[wp.transformf],
    position: wp.vec3f,
    orientation: wp.mat33f,
) -> wp.transformf:
    local_pose = _local_joint_pose(position, orientation)
    if body >= 0:
        return body_pose[body] * local_pose
    return local_pose


@wp.func
def _candidate_joint_pose(
    body: wp.int32,
    body_pose: wp.array[wp.transformf],
    candidate_velocity: wp.array[vec6f],
    linearization_velocity: wp.array[vec6f],
    dt: wp.float32,
    position: wp.vec3f,
    orientation: wp.mat33f,
) -> wp.transformf:
    local_pose = _local_joint_pose(position, orientation)
    if body < 0:
        return local_pose
    delta = candidate_velocity[body] - linearization_velocity[body]
    candidate_pose = compute_body_pose_update_with_logmap(
        dt,
        body_pose[body],
        wp.vec3f(delta[0], delta[1], delta[2]),
        wp.vec3f(delta[3], delta[4], delta[5]),
    )
    return candidate_pose * local_pose


@wp.func
def _linear_rod_strain(parent_pose: wp.transformf, child_pose: wp.transformf) -> wp.vec3f:
    parent_position = wp.transform_get_translation(parent_pose)
    child_position = wp.transform_get_translation(child_pose)
    parent_orientation = wp.transform_get_rotation(parent_pose)
    return wp.quat_rotate_inv(parent_orientation, child_position - parent_position)


@wp.func
def _make_linear_jacobian_row(linear: wp.vec3f, angular: wp.vec3f) -> vec6f:
    return vec6f(linear[0], linear[1], linear[2], angular[0], angular[1], angular[2])


@wp.kernel
def _initialize_rod_rest_state(
    rod_joint: wp.array[wp.int32],
    rod_body_first: wp.array[wp.int32],
    rod_body_second: wp.array[wp.int32],
    joint_first_position: wp.array[wp.vec3f],
    joint_second_position: wp.array[wp.vec3f],
    joint_first_orientation: wp.array[wp.mat33f],
    joint_second_orientation: wp.array[wp.mat33f],
    body_rest_pose: wp.array[wp.transformf],
    rest_curvature_local: wp.array[wp.vec3f],
    rest_twist: wp.array[wp.float32],
):
    rod = wp.tid()
    joint = rod_joint[rod]
    first_pose = _joint_pose(
        rod_body_first[rod],
        body_rest_pose,
        joint_first_position[joint],
        joint_first_orientation[joint],
    )
    second_pose = _joint_pose(
        rod_body_second[rod],
        body_rest_pose,
        joint_second_position[joint],
        joint_second_orientation[joint],
    )
    first_orientation = wp.transform_get_rotation(first_pose)
    second_orientation = wp.transform_get_rotation(second_pose)
    measure = _measure_rod_bend_twist_z(first_orientation, second_orientation)
    rest_curvature_local[rod] = wp.quat_rotate(wp.quat_inverse(first_orientation), measure.kb_world)
    rest_twist[rod] = measure.twist


@wp.kernel
def _evaluate_rod_materials(
    rod_joint: wp.array[wp.int32],
    rod_body_first: wp.array[wp.int32],
    rod_body_second: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    rod_dof_offset: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    joint_first_position: wp.array[wp.vec3f],
    joint_second_position: wp.array[wp.vec3f],
    joint_first_orientation: wp.array[wp.mat33f],
    joint_second_orientation: wp.array[wp.mat33f],
    joint_stiffness: wp.array[wp.float32],
    joint_damping: wp.array[wp.float32],
    body_pose: wp.array[wp.transformf],
    body_velocity: wp.array[vec6f],
    rest_curvature_local: wp.array[wp.vec3f],
    rest_twist: wp.array[wp.float32],
    time_step: wp.array[wp.float32],
    strain: wp.array[wp.float32],
    stress: wp.array[wp.float32],
    tangent_diagonal: wp.array[wp.float32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
):
    rod = wp.tid()
    joint = rod_joint[rod]
    row_offset = 6 * rod
    if not joint_enabled[joint]:
        for local_row in range(6):
            row = row_offset + local_row
            strain[row] = 0.0
            stress[row] = 0.0
            tangent_diagonal[row] = 0.0
            jacobian_first[row] = vec6f(0.0)
            jacobian_second[row] = vec6f(0.0)
        return

    first = rod_body_first[rod]
    second = rod_body_second[rod]
    dt = time_step[body_world[second]]
    first_pose = _joint_pose(
        first,
        body_pose,
        joint_first_position[joint],
        joint_first_orientation[joint],
    )
    second_pose = _joint_pose(
        second,
        body_pose,
        joint_second_position[joint],
        joint_second_orientation[joint],
    )
    first_orientation = wp.transform_get_rotation(first_pose)
    second_orientation = wp.transform_get_rotation(second_pose)

    linear_strain = _linear_rod_strain(first_pose, second_pose)
    measure = _measure_rod_bend_twist_z(first_orientation, second_orientation)
    angular_strain = _assemble_geometric_rod_kappa_z(
        first_orientation,
        measure.kb_world,
        measure.twist,
        rest_curvature_local[rod],
        rest_twist[rod],
    )
    dof_offset = rod_dof_offset[rod]
    stiffness = wp.vec3f(
        joint_stiffness[dof_offset + 1],
        joint_stiffness[dof_offset + 1],
        joint_stiffness[dof_offset],
    )
    angular_stiffness = wp.vec3f(
        joint_stiffness[dof_offset + 2],
        joint_stiffness[dof_offset + 2],
        joint_stiffness[dof_offset + 3],
    )
    damping = wp.vec3f(
        joint_damping[dof_offset + 1],
        joint_damping[dof_offset + 1],
        joint_damping[dof_offset],
    )
    angular_damping = wp.vec3f(
        joint_damping[dof_offset + 2],
        joint_damping[dof_offset + 2],
        joint_damping[dof_offset + 3],
    )
    child_position = wp.transform_get_translation(second_pose)
    child_com = wp.transform_get_translation(body_pose[second])
    first_com = wp.vec3f(0.0)
    if first >= 0:
        first_com = wp.transform_get_translation(body_pose[first])

    material_axes = wp.quat_to_matrix(first_orientation)
    for local_row in range(3):
        axis = wp.vec3f(
            material_axes[0, local_row],
            material_axes[1, local_row],
            material_axes[2, local_row],
        )
        first_angular = wp.cross(axis, child_position - first_com)
        second_angular = wp.cross(child_position - child_com, axis)
        jacobian_first[row_offset + local_row] = _make_linear_jacobian_row(-axis, first_angular)
        jacobian_second[row_offset + local_row] = _make_linear_jacobian_row(axis, second_angular)

    first_angular_jacobian = _rod_bend_twist_jacobian_z_from_measure(measure, True)
    second_angular_jacobian = _rod_bend_twist_jacobian_z_from_measure(measure, False)
    for local_row in range(3):
        first_row = wp.vec3f(
            first_angular_jacobian[local_row, 0],
            first_angular_jacobian[local_row, 1],
            first_angular_jacobian[local_row, 2],
        )
        second_row = wp.vec3f(
            second_angular_jacobian[local_row, 0],
            second_angular_jacobian[local_row, 1],
            second_angular_jacobian[local_row, 2],
        )
        jacobian_first[row_offset + 3 + local_row] = _make_linear_jacobian_row(wp.vec3f(0.0), first_row)
        jacobian_second[row_offset + 3 + local_row] = _make_linear_jacobian_row(wp.vec3f(0.0), second_row)

    inverse_time_step = 1.0 / dt
    for local_row in range(3):
        row = row_offset + local_row
        rate = wp.dot(jacobian_second[row], body_velocity[second])
        if first >= 0:
            rate += wp.dot(jacobian_first[row], body_velocity[first])
        value = linear_strain[local_row]
        strain[row] = value
        stress[row] = stiffness[local_row] * value + damping[local_row] * rate
        tangent_diagonal[row] = stiffness[local_row] + damping[local_row] * inverse_time_step
    for local_row in range(3):
        row = row_offset + 3 + local_row
        rate = wp.dot(jacobian_second[row], body_velocity[second])
        if first >= 0:
            rate += wp.dot(jacobian_first[row], body_velocity[first])
        value = angular_strain[local_row]
        strain[row] = value
        stress[row] = angular_stiffness[local_row] * value + angular_damping[local_row] * rate
        tangent_diagonal[row] = angular_stiffness[local_row] + angular_damping[local_row] * inverse_time_step


@wp.kernel
def _accumulate_rod_wrenches(
    rod_body_first: wp.array[wp.int32],
    rod_body_second: wp.array[wp.int32],
    stress: wp.array[wp.float32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    body_wrench: wp.array[wp.spatial_vectorf],
):
    rod = wp.tid()
    first_wrench = vec6f(0.0)
    second_wrench = vec6f(0.0)
    row_offset = 6 * rod
    for local_row in range(6):
        row = row_offset + local_row
        first_wrench -= stress[row] * jacobian_first[row]
        second_wrench -= stress[row] * jacobian_second[row]

    first = rod_body_first[rod]
    second = rod_body_second[rod]
    if first >= 0:
        wp.atomic_add(
            body_wrench,
            first,
            wp.spatial_vectorf(
                first_wrench[0],
                first_wrench[1],
                first_wrench[2],
                first_wrench[3],
                first_wrench[4],
                first_wrench[5],
            ),
        )
    if second >= 0:
        wp.atomic_add(
            body_wrench,
            second,
            wp.spatial_vectorf(
                second_wrench[0],
                second_wrench[1],
                second_wrench[2],
                second_wrench[3],
                second_wrench[4],
                second_wrench[5],
            ),
        )


@wp.kernel
def _update_rod_proximal(
    rod_joint: wp.array[wp.int32],
    rod_body_first: wp.array[wp.int32],
    rod_body_second: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    joint_first_position: wp.array[wp.vec3f],
    joint_second_position: wp.array[wp.vec3f],
    joint_first_orientation: wp.array[wp.mat33f],
    joint_second_orientation: wp.array[wp.mat33f],
    body_pose: wp.array[wp.transformf],
    candidate_velocity: wp.array[vec6f],
    linearization_velocity: wp.array[vec6f],
    rest_curvature_local: wp.array[wp.vec3f],
    rest_twist: wp.array[wp.float32],
    frozen_strain: wp.array[wp.float32],
    frozen_stress: wp.array[wp.float32],
    tangent_diagonal: wp.array[wp.float32],
    frozen_jacobian_first: wp.array[vec6f],
    frozen_jacobian_second: wp.array[vec6f],
    body_vector_index: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    time_step: wp.array[wp.float32],
    rotation_tolerance: wp.float32,
    velocity_tolerance: wp.float32,
    proximal_relaxation: wp.float32,
    proximal_coordinate: wp.array[wp.float32],
    multiplier: wp.array[wp.float32],
    nonlinear_right_hand_side: wp.array[wp.float32],
    world_residual: wp.array[wp.float32],
    world_failed: wp.array[wp.int32],
):
    """Update one fixed-metric bend/twist prox and scatter its next RHS correction."""
    rod = wp.tid()
    joint = rod_joint[rod]
    if not joint_enabled[joint]:
        return

    first = rod_body_first[rod]
    second = rod_body_second[rod]
    world = body_world[second]
    if not world_active[world]:
        return
    dt = time_step[world]

    first_pose = _candidate_joint_pose(
        first,
        body_pose,
        candidate_velocity,
        linearization_velocity,
        dt,
        joint_first_position[joint],
        joint_first_orientation[joint],
    )
    second_pose = _candidate_joint_pose(
        second,
        body_pose,
        candidate_velocity,
        linearization_velocity,
        dt,
        joint_second_position[joint],
        joint_second_orientation[joint],
    )
    first_orientation = wp.transform_get_rotation(first_pose)
    second_orientation = wp.transform_get_rotation(second_pose)
    linear_strain = _linear_rod_strain(first_pose, second_pose)
    measure = _measure_rod_bend_twist_z(first_orientation, second_orientation)
    angular_strain = _assemble_geometric_rod_kappa_z(
        first_orientation,
        measure.kb_world,
        measure.twist,
        rest_curvature_local[rod],
        rest_twist[rod],
    )
    candidate_strain = vec6f(
        linear_strain[0],
        linear_strain[1],
        linear_strain[2],
        angular_strain[0],
        angular_strain[1],
        angular_strain[2],
    )

    coordinate_new = vec6f(0.0)
    multiplier_new = vec6f(0.0)
    correction = vec6f(0.0)
    residual = wp.float32(0.0)
    row_offset = 6 * rod
    finite = wp.bool(True)
    for local_row in range(6):
        row = row_offset + local_row
        tangent = tangent_diagonal[row]
        center = candidate_strain[local_row]
        previous = proximal_coordinate[row]
        frozen = frozen_strain[row]
        stress = frozen_stress[row]
        dual = multiplier[row]
        coordinate = center
        dual_new = wp.float32(0.0)
        row_correction = wp.float32(0.0)
        # Keep the very stiff stretch/shear block linearly implicit. Feeding its
        # frozen-geometry defect back through the axial tangent can amplify a
        # small rotational mismatch into an unstable wrench.
        if tangent > 0.0 and local_row >= 3:
            coordinate = 0.5 * (frozen + center) + 0.5 * (dual - stress) / tangent
            dual_new = dual + proximal_relaxation * tangent * (center - coordinate)

            linearized_change = wp.float32(0.0)
            if first >= 0:
                linearized_change += wp.dot(
                    frozen_jacobian_first[row],
                    candidate_velocity[first] - linearization_velocity[first],
                )
            if second >= 0:
                linearized_change += wp.dot(
                    frozen_jacobian_second[row],
                    candidate_velocity[second] - linearization_velocity[second],
                )
            predicted_stress = stress + dt * tangent * linearized_change
            row_correction = predicted_stress - dual_new

            residual = wp.max(residual, wp.abs(center - coordinate) / rotation_tolerance)
            residual = wp.max(residual, wp.abs(coordinate - previous) / (dt * velocity_tolerance))

        coordinate_new[local_row] = coordinate
        multiplier_new[local_row] = dual_new
        correction[local_row] = row_correction
        finite = (
            finite
            and wp.isfinite(center)
            and wp.isfinite(coordinate)
            and wp.isfinite(dual_new)
            and wp.isfinite(row_correction)
        )

    if not finite or not wp.isfinite(residual):
        wp.atomic_max(world_failed, world, 1)
        return

    first_vector_offset = -1
    if first >= 0:
        first_vector_offset = body_vector_index[first]
    second_vector_offset = -1
    if second >= 0:
        second_vector_offset = body_vector_index[second]
    for local_row in range(6):
        row = row_offset + local_row
        scale = dt * correction[local_row]
        if first_vector_offset >= 0:
            first_jacobian = frozen_jacobian_first[row]
            for axis in range(6):
                wp.atomic_add(
                    nonlinear_right_hand_side,
                    first_vector_offset + axis,
                    scale * first_jacobian[axis],
                )
        if second_vector_offset >= 0:
            second_jacobian = frozen_jacobian_second[row]
            for axis in range(6):
                wp.atomic_add(
                    nonlinear_right_hand_side,
                    second_vector_offset + axis,
                    scale * second_jacobian[axis],
                )
        proximal_coordinate[row] = coordinate_new[local_row]
        multiplier[row] = multiplier_new[local_row]
    wp.atomic_max(world_residual, world, residual)


class RodMaterialSystem:
    """Allocated LOX material state for Newton rod joints."""

    def __init__(self, model: ModelKamino, data: DataKamino, *, proximal_relaxation: float = 1.0):
        if not np.isfinite(proximal_relaxation) or proximal_relaxation < 0.0 or proximal_relaxation > 1.0:
            raise ValueError(f"LOX rod proximal relaxation must be in [0, 1], got {proximal_relaxation}.")
        source_model = model._model
        joint_types = model.joints.dof_type.numpy().astype(np.int32, copy=False)
        rod_joints = np.flatnonzero(joint_types == int(JointDoFType.ROD)).astype(np.int32)
        if source_model is None and rod_joints.size:
            raise ValueError("Rod materials require a ModelKamino converted from Newton.")

        self.model = model
        self.data = data
        self.source_model = source_model
        self.device = wp.get_device(model.device)
        self.count = rod_joints.size
        self.proximal_relaxation = proximal_relaxation

        parent = model.joints.bid_B.numpy().astype(np.int32, copy=False)
        child = model.joints.bid_F.numpy().astype(np.int32, copy=False)
        dof_offset = model.joints.dofs_offset.numpy().astype(np.int32, copy=False)
        self.joint = wp.array(rod_joints, dtype=wp.int32, device=self.device)
        self.body_first = wp.array(parent[rod_joints], dtype=wp.int32, device=self.device)
        self.body_second = wp.array(child[rod_joints], dtype=wp.int32, device=self.device)
        self.dof_offset = wp.array(dof_offset[rod_joints], dtype=wp.int32, device=self.device)

        row_count = 6 * self.count
        self.rest_curvature_local = wp.zeros(self.count, dtype=wp.vec3f, device=self.device)
        self.rest_twist = wp.zeros(self.count, dtype=wp.float32, device=self.device)
        self.strain = wp.zeros(row_count, dtype=wp.float32, device=self.device)
        self.stress = wp.zeros(row_count, dtype=wp.float32, device=self.device)
        self.tangent_diagonal = wp.zeros(row_count, dtype=wp.float32, device=self.device)
        self.jacobian_first = wp.zeros(row_count, dtype=vec6f, device=self.device)
        self.jacobian_second = wp.zeros(row_count, dtype=vec6f, device=self.device)
        self.proximal_coordinate = wp.zeros(row_count, dtype=wp.float32, device=self.device)
        self.multiplier = wp.zeros(row_count, dtype=wp.float32, device=self.device)
        self.world_proximal_residual = wp.zeros(model.size.num_worlds, dtype=wp.float32, device=self.device)
        self.world_proximal_failed = wp.zeros(model.size.num_worlds, dtype=wp.int32, device=self.device)
        self.refresh_rest_state()

    def refresh_rest_state(self) -> None:
        """Recompute pre-curved rod rest invariants from model rest poses."""
        if self.count == 0:
            return
        wp.launch(
            _initialize_rod_rest_state,
            dim=self.count,
            inputs=[
                self.joint,
                self.body_first,
                self.body_second,
                self.model.joints.B_r_Bj,
                self.model.joints.F_r_Fj,
                self.model.joints.X_Bj,
                self.model.joints.X_Fj,
                self.model.bodies.q_i_0,
            ],
            outputs=[self.rest_curvature_local, self.rest_twist],
            device=self.device,
        )

    def evaluate(self, time_step: wp.array[wp.float32], body_velocity: wp.array[vec6f]) -> None:
        """Evaluate rod strain, stress, tangent, and endpoint Jacobians."""
        if self.count == 0:
            return
        validate_world_time_step(time_step, self.model.size.num_worlds, self.device)
        wp.launch(
            _evaluate_rod_materials,
            dim=self.count,
            inputs=[
                self.joint,
                self.body_first,
                self.body_second,
                self.model.bodies.wid,
                self.dof_offset,
                self.source_model.joint_enabled,
                self.model.joints.B_r_Bj,
                self.model.joints.F_r_Fj,
                self.model.joints.X_Bj,
                self.model.joints.X_Fj,
                self.model.joints.k_p_j,
                self.model.joints.k_d_j,
                self.data.bodies.q_i,
                body_velocity,
                self.rest_curvature_local,
                self.rest_twist,
                time_step,
            ],
            outputs=[
                self.strain,
                self.stress,
                self.tangent_diagonal,
                self.jacobian_first,
                self.jacobian_second,
            ],
            device=self.device,
        )

    def assemble(
        self,
        system: BatchedPrimalBodySystem,
        linearization_twist: wp.array[vec6f],
        time_step: wp.array[wp.float32],
        prescribed_twist: wp.array[vec6f] | None = None,
    ) -> None:
        """Evaluate and add rod material blocks to a LOX primal system."""
        if self.count == 0:
            return
        self.evaluate(time_step, linearization_twist)
        system.add_smooth_material_blocks(
            self.body_first,
            self.body_second,
            self.jacobian_first,
            self.jacobian_second,
            self.stress,
            self.tangent_diagonal,
            linearization_twist,
            time_step,
            prescribed_twist=prescribed_twist,
        )
        if self.proximal_relaxation > 0.0:
            wp.copy(self.proximal_coordinate, self.strain)
            wp.copy(self.multiplier, self.stress)
            self.world_proximal_residual.zero_()
            self.world_proximal_failed.zero_()

    def update_proximal(
        self,
        system: BatchedPrimalBodySystem,
        candidate_velocity: wp.array[vec6f],
        linearization_velocity: wp.array[vec6f],
        world_active: wp.array[wp.bool],
        time_step: wp.array[wp.float32],
        position_tolerance: float,
        rotation_tolerance: float,
        velocity_tolerance: float,
    ) -> None:
        """Update nonlinear rod bend/twist and assemble the next candidate RHS correction."""
        if self.count == 0 or self.proximal_relaxation <= 0.0:
            return
        body_count = self.model.size.sum_of_num_bodies
        if candidate_velocity.shape[0] != body_count or linearization_velocity.shape[0] != body_count:
            raise ValueError("Rod proximal velocities must contain one entry per body.")
        if world_active.shape[0] != self.model.size.num_worlds:
            raise ValueError("Rod proximal world mask must contain one entry per world.")
        validate_world_time_step(time_step, self.model.size.num_worlds, self.device)
        if position_tolerance <= 0.0 or rotation_tolerance <= 0.0 or velocity_tolerance <= 0.0:
            raise ValueError("Rod proximal convergence tolerances must be positive.")

        system.nonlinear_right_hand_side.zero_()
        self.world_proximal_residual.zero_()
        self.world_proximal_failed.zero_()
        wp.launch(
            _update_rod_proximal,
            dim=self.count,
            inputs=[
                self.joint,
                self.body_first,
                self.body_second,
                self.model.bodies.wid,
                self.source_model.joint_enabled,
                self.model.joints.B_r_Bj,
                self.model.joints.F_r_Fj,
                self.model.joints.X_Bj,
                self.model.joints.X_Fj,
                self.data.bodies.q_i,
                candidate_velocity,
                linearization_velocity,
                self.rest_curvature_local,
                self.rest_twist,
                self.strain,
                self.stress,
                self.tangent_diagonal,
                self.jacobian_first,
                self.jacobian_second,
                system.body_vector_index,
                world_active,
                time_step,
                rotation_tolerance,
                velocity_tolerance,
                self.proximal_relaxation,
            ],
            outputs=[
                self.proximal_coordinate,
                self.multiplier,
                system.nonlinear_right_hand_side,
                self.world_proximal_residual,
                self.world_proximal_failed,
            ],
            device=self.device,
        )

    def accumulate_wrenches(
        self,
        body_wrench: wp.array[wp.spatial_vectorf],
        time_step: wp.array[wp.float32],
        body_velocity: wp.array[vec6f],
    ) -> None:
        """Evaluate and add final rod material wrenches to body output."""
        if self.count == 0:
            return
        self.evaluate(time_step, body_velocity)
        wp.launch(
            _accumulate_rod_wrenches,
            dim=self.count,
            inputs=[
                self.body_first,
                self.body_second,
                self.stress,
                self.jacobian_first,
                self.jacobian_second,
            ],
            outputs=[body_wrench],
            device=self.device,
        )

    def reset(self) -> None:
        """Clear transient material buffers before the next step baseline."""
        self.strain.zero_()
        self.stress.zero_()
        self.tangent_diagonal.zero_()
        self.jacobian_first.zero_()
        self.jacobian_second.zero_()
        self.proximal_coordinate.zero_()
        self.multiplier.zero_()
        self.world_proximal_residual.zero_()
        self.world_proximal_failed.zero_()
