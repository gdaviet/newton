# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rod joints of the LOX body system.

Newton rod joints connect rigid bodies with stretch, shear, bend, and twist stiffness
and damping. LOX assembles each rod as six smooth material rows of the body system,
linearized at the beginning of the time step; optional proximal updates correct the bend
and twist rows towards their exact candidate-pose strain during the iterations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ......core.types import MAXVAL
from ......sim import JointTargetMode, JointType, Model
from ...core.data import DataKamino
from ...core.joints import JointDoFType
from ...core.model import ModelKamino
from ...core.types import vec6f
from .rod_kernels import _accumulate_rod_wrenches, _initialize_rod_rest_state, _load_rod_rows, _update_rod_proximal
from .types import RodRows

if TYPE_CHECKING:
    from .system import LOXSystem

###
# Module interface
###

__all__ = [
    "RodMaterials",
    "validate_rod_model",
]


###
# Functions
###


def validate_rod_model(model: Model) -> None:
    """Validate the rod joints of a Newton model for LOX.

    Raises:
        ValueError: If a rod joint has an unsupported layout, endpoints in different worlds,
            negative or non-finite coefficients, armature, joint damping or friction, finite
            limits, or force control.
    """
    if model.joint_count == 0:
        return
    rod_joints = np.flatnonzero(model.joint_type.numpy() == int(JointType.ROD))
    if rod_joints.size == 0:
        return

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
                f"LOX rod joint {joint} requires q_count=4, qd_count=4, and dof_dim=(2, 2); "
                f"got q_count={q_count}, qd_count={qd_count}, and dof_dim={dof_dim}."
            )
        if child < 0 or child >= model.body_count:
            raise ValueError(f"LOX rod joint {joint} has invalid child body {child}.")
        if world < 0 or world >= model.world_count or normalized_world(int(body_world[child])) != world:
            raise ValueError(f"LOX rod joint {joint} and child body {child} must share a world.")
        if parent >= model.body_count or parent < -1:
            raise ValueError(f"LOX rod joint {joint} has invalid parent body {parent}.")
        if parent >= 0 and normalized_world(int(body_world[parent])) != world:
            raise ValueError(f"LOX rod joint {joint} and parent body {parent} must share a world.")

        rod_ke = target_ke[dof_start:dof_end]
        rod_kd = target_kd[dof_start:dof_end]
        if not np.isfinite(rod_ke).all() or np.any(rod_ke < 0.0):
            raise ValueError(f"LOX rod joint {joint} stiffness values must be finite and nonnegative.")
        if not np.isfinite(rod_kd).all() or np.any(rod_kd < 0.0):
            raise ValueError(f"LOX rod joint {joint} damping values must be finite and nonnegative.")
        for name, values in (
            ("joint_armature", armature[dof_start:dof_end]),
            ("joint_damping", damping[dof_start:dof_end]),
            ("joint_friction", friction[dof_start:dof_end]),
        ):
            if not np.isfinite(values).all() or np.any(values != 0.0):
                raise ValueError(f"LOX rod joint {joint} requires zero {name}.")
        if np.any(limit_lower[dof_start:dof_end] > -MAXVAL) or np.any(limit_upper[dof_start:dof_end] < MAXVAL):
            raise ValueError(f"LOX rod joint {joint} requires unbounded joint limits.")
        if np.any(target_mode[dof_start:dof_end] == int(JointTargetMode.EFFORT)):
            raise ValueError(f"LOX rod joint {joint} requires a target mode other than force control.")


###
# Interfaces
###


class RodMaterials:
    """Smooth material rows of the rod joints in the LOX body system.

    The arrays live in :class:`RodRows`, exposed as :attr:`data`.

    Args:
        model: The Kamino model, converted from a Newton model.
        data: The Kamino data of the model.
        proximal_relaxation: Relaxation factor in ``[0, 1]`` of the bend and twist proximal
            updates; zero keeps the linearized rows.
    """

    def __init__(self, model: ModelKamino, data: DataKamino, proximal_relaxation: float = 0.0):
        self.model = model
        self.kamino_data = data
        """Kamino data of the model."""
        self.device = wp.get_device(model.device)
        self.proximal_relaxation = proximal_relaxation
        joint_dof_type = model.joints.dof_type.numpy().astype(np.int32, copy=False)
        rod_joint = np.flatnonzero(joint_dof_type == int(JointDoFType.ROD)).astype(np.int32)
        self._source_model = model._model
        if self._source_model is None and rod_joint.size > 0:
            raise ValueError("LOX rod joints require a Kamino model converted from a Newton model.")
        self._proximal = proximal_relaxation > 0.0 and rod_joint.size > 0
        self._data = RodRows(
            joint=rod_joint,
            body_a=model.joints.bid_B.numpy()[rod_joint],
            body_b=model.joints.bid_F.numpy()[rod_joint],
            dof_offset=model.joints.dofs_offset.numpy()[rod_joint],
            num_worlds=model.info.num_worlds,
            num_bodies=model.size.sum_of_num_bodies,
            proximal=self.proximal,
            device=self.device,
        )
        self.refresh_rest_state()

    ###
    # Properties
    ###

    @property
    def data(self) -> RodRows:
        """Returns the rod material rows."""
        return self._data

    @property
    def proximal(self) -> bool:
        """Whether there are rods with proximal updates of their bend and twist rows."""
        return self._proximal

    ###
    # Public API
    ###

    def refresh_rest_state(self) -> None:
        """Recompute the rest curvature and twist of the rods from the initial body poses."""
        rows = self._data
        if rows.count == 0:
            return
        joints = self.model.joints
        wp.launch(
            _initialize_rod_rest_state,
            dim=rows.count,
            inputs=[
                rows.joint,
                rows.body_a,
                rows.body_b,
                joints.B_r_Bj,
                joints.F_r_Fj,
                joints.X_Bj,
                joints.X_Fj,
                self.model.bodies.q_i_0,
            ],
            outputs=[rows.rest_curvature, rows.rest_twist],
            device=self.device,
        )

    def add_to_system(
        self,
        system: LOXSystem,
        time_step: wp.array[wp.float32],
        prescribed_twist: wp.array[vec6f],
    ) -> None:
        """Evaluate the rows at the beginning of the time step and add them to the smooth system.

        With proximal updates, also reset the proximal state to the frozen strain and stress.

        Args:
            system: The smooth body system.
            time_step: Time step of each world [s].
            prescribed_twist: Twist of the bodies outside the smooth system [m/s, rad/s].
        """
        rows = self._data
        if rows.count == 0:
            return
        self._load(time_step, body_twist=None)
        system.add_rod_rows(
            rows.body_a,
            rows.body_b,
            rows.jacobian_a,
            rows.jacobian_b,
            rows.stress,
            rows.tangent,
            time_step,
            prescribed_twist=prescribed_twist,
        )
        if self.proximal:
            wp.copy(rows.proximal_coordinate, rows.strain)
            wp.copy(rows.multiplier, rows.stress)
            rows.body_impulse.zero_()
            rows.world_residual.zero_()

    def update_proximal(
        self,
        time_step: wp.array[wp.float32],
        rotation_tolerance: float,
        velocity_tolerance: float,
        global_twist: wp.array[vec6f],
        world_active: wp.array[wp.bool],
        world_failed: wp.array[wp.bool],
    ) -> None:
        """Update the bend and twist proximal rows from the candidate twist.

        Rewrites :attr:`RodRows.body_impulse` for the next candidate solve, and accumulates the
        scaled proximal residual of each world into :attr:`RodRows.world_residual`, which the
        convergence check reads and clears. A non-finite update fails its world.

        Args:
            time_step: Time step of each world [s].
            rotation_tolerance: Bend and twist tolerance of the proximal residual [rad].
            velocity_tolerance: Rate tolerance of the proximal residual [rad/s].
            global_twist: Candidate twist of each body from the smooth solve [m/s, rad/s].
            world_active: Whether each world is still iterating.
            world_failed: Whether the solve of each world failed.
        """
        if not self.proximal:
            return
        rows = self._data
        joints = self.model.joints
        rows.body_impulse.zero_()
        wp.launch(
            _update_rod_proximal,
            dim=rows.count,
            inputs=[
                time_step,
                self.model.bodies.wid,
                rotation_tolerance,
                velocity_tolerance,
                self.proximal_relaxation,
                rows.joint,
                rows.body_a,
                rows.body_b,
                rows.dof_offset,
                self._source_model.joint_enabled,
                joints.B_r_Bj,
                joints.F_r_Fj,
                joints.X_Bj,
                joints.X_Fj,
                joints.k_p_j,
                self.kamino_data.bodies.q_i,
                global_twist,
                world_active,
                rows.rest_curvature,
                rows.rest_twist,
                rows.strain,
                rows.stress,
                rows.tangent,
                rows.jacobian_a,
                rows.jacobian_b,
            ],
            outputs=[
                rows.proximal_coordinate,
                rows.multiplier,
                rows.body_impulse,
                rows.world_residual,
                world_failed,
            ],
            device=self.device,
        )

    def accumulate_wrenches(
        self,
        joint_wrench: wp.array[vec6f],
        time_step: wp.array[wp.float32],
        body_twist: wp.array[vec6f],
        world_failed: wp.array[wp.bool],
    ) -> None:
        """Evaluate the rows at the accepted twist and add the rod wrenches of the non-failed worlds.

        Args:
            joint_wrench: Joint wrench of each body, which receives the rod wrenches [N, N m].
            time_step: Time step of each world [s].
            body_twist: Accepted twist of each body [m/s, rad/s].
            world_failed: Whether the solve of each world failed.
        """
        rows = self._data
        if rows.count == 0:
            return
        self._load(time_step, body_twist=body_twist)
        wp.launch(
            _accumulate_rod_wrenches,
            dim=rows.count,
            inputs=[
                self.model.bodies.wid,
                rows.body_a,
                rows.body_b,
                rows.stress,
                rows.jacobian_a,
                rows.jacobian_b,
                world_failed,
            ],
            outputs=[joint_wrench],
            device=self.device,
        )

    ###
    # Internals
    ###

    def _load(self, time_step: wp.array[wp.float32], body_twist: wp.array[vec6f] | None) -> None:
        """Evaluate the rows at the current body poses, with the damping rate of ``body_twist``."""
        rows = self._data
        joints = self.model.joints
        wp.launch(
            _load_rod_rows,
            dim=rows.count,
            inputs=[
                time_step,
                self.model.bodies.wid,
                rows.joint,
                rows.body_a,
                rows.body_b,
                rows.dof_offset,
                self._source_model.joint_enabled,
                joints.B_r_Bj,
                joints.F_r_Fj,
                joints.X_Bj,
                joints.X_Fj,
                joints.k_p_j,
                joints.k_d_j,
                self.kamino_data.bodies.q_i,
                body_twist,
                rows.rest_curvature,
                rows.rest_twist,
            ],
            outputs=[rows.strain, rows.stress, rows.tangent, rows.jacobian_a, rows.jacobian_b],
            device=self.device,
        )
