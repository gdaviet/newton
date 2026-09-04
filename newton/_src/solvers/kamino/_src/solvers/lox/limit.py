# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint-limit operations used by the LOX nonlinear loop."""

import warp as wp

from ...core.model import ModelKamino
from ...kinematics.limits import LimitsKamino, read_joint_coords_map_and_limits

__all__ = ["refresh_active_joint_configuration_limits"]

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.kernel
def _refresh_active_joint_configuration_limits(
    limits_model_num: wp.array[wp.int32],
    limits_jid: wp.array[wp.int32],
    limits_dof: wp.array[wp.int32],
    limits_side: wp.array[wp.float32],
    model_joint_dof_type: wp.array[wp.int32],
    model_joint_dofs_offset: wp.array[wp.int32],
    model_joint_coords_offset: wp.array[wp.int32],
    model_joint_q_j_min: wp.array[wp.float32],
    model_joint_q_j_max: wp.array[wp.float32],
    state_joints_q_j: wp.array[wp.float32],
    limits_r_q: wp.array[wp.float32],
):
    limit = wp.tid()
    if limit >= limits_model_num[0]:
        return

    joint = limits_jid[limit]
    dofs_offset = model_joint_dofs_offset[joint]
    local_dof = limits_dof[limit] - dofs_offset
    dof_count, q_min, q_max, q_mapped = read_joint_coords_map_and_limits(
        model_joint_dof_type[joint],
        dofs_offset,
        model_joint_coords_offset[joint],
        model_joint_q_j_min,
        model_joint_q_j_max,
        state_joints_q_j,
    )
    if local_dof < 0 or local_dof >= dof_count:
        return
    if limits_side[limit] > 0.0:
        limits_r_q[limit] = q_mapped[local_dof] - q_min[local_dof]
    else:
        limits_r_q[limit] = q_max[local_dof] - q_mapped[local_dof]


def refresh_active_joint_configuration_limits(
    model: ModelKamino,
    limits: LimitsKamino,
    joint_position: wp.array[wp.float32],
) -> None:
    """Refresh violations for LOX's active, frozen limit set."""
    if limits.model_max_limits_host <= 0:
        return

    wp.launch(
        kernel=_refresh_active_joint_configuration_limits,
        dim=limits.model_max_limits_host,
        inputs=[
            limits.model_active_limits,
            limits.jid,
            limits.dof,
            limits.side,
            model.joints.dof_type,
            model.joints.dofs_offset,
            model.joints.coords_offset,
            model.joints.q_j_min,
            model.joints.q_j_max,
            joint_position,
        ],
        outputs=[limits.r_q],
        device=model.device,
    )
