# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen-contact orchestration for the LOX rigid-body solve."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ......sim import ModelFlags
from ...core.bodies import update_body_wrenches
from ...core.types import vec6f
from .adapter import LOXKaminoAdapter
from .colored_gauss_seidel import ColoredGaussSeidelProjection
from .integration import _write_integrator_body_inputs
from .joint_delassus import BatchedStructuralDelassus
from .projection import PROJECTION_STATUS_VALID
from .rod import validate_rod_model
from .sweep import (
    compute_projection_residuals,
    prepare_jacobi_projection_data,
    prepare_physical_projection_data,
    project_constraints_jacobi,
)
from .time import validate_world_time_steps

if TYPE_CHECKING:
    from ......sim import Model
    from ....config import ConstraintStabilizationConfig, LOXSolverConfig
    from ...core.data import DataKamino
    from ...core.joints import JointCorrectionMode
    from ...core.model import ModelKamino
    from ...dynamics.dual import DualProblem
    from ...geometry.contacts import ContactsKamino
    from ...kinematics.jacobians import SparseSystemJacobians
    from ...kinematics.limits import LimitsKamino

__all__ = [
    "LOX_STATUS_ACTIVE",
    "LOX_STATUS_CONVERGED",
    "LOX_STATUS_FAILED",
    "LOX_STATUS_ITERATION_LIMIT",
    "LOXSolver",
]

LOX_STATUS_ACTIVE = 0
"""The world is active in the current splitting solve."""

LOX_STATUS_CONVERGED = 1
"""The world met the configured splitting tolerances."""

LOX_STATUS_FAILED = 2
"""A unilateral projection failed for the world."""

LOX_STATUS_ITERATION_LIMIT = 3
"""The world reached the configured splitting iteration limit."""

wp.set_module_options({"enable_backward": False})

_JOINT_PENALTY_SEED_PERCENTILE = 0.02


def _low_mode_eigenvalue(eigenvalues: np.ndarray) -> float:
    """Select the discrete low-mode percentile used for ALM scale seeding."""
    index = int(_JOINT_PENALTY_SEED_PERCENTILE * eigenvalues.size)
    return float(eigenvalues[index])


@wp.kernel
def _finalize_world_status(
    world_converged: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    world_iteration_limit: wp.array[wp.bool],
    world_accepted: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
):
    world = wp.tid()
    converged = world_converged[world] and not world_failed[world]
    world_accepted[world] = not world_failed[world]
    if world_failed[world]:
        world_status[world] = LOX_STATUS_FAILED
    elif converged:
        world_status[world] = LOX_STATUS_CONVERGED
    elif world_iteration_limit[world]:
        world_status[world] = LOX_STATUS_ITERATION_LIMIT
    else:
        world_status[world] = LOX_STATUS_ACTIVE


@wp.kernel
def _reset_solver_worlds_masked(
    world_mask: wp.array[wp.bool],
    world_active: wp.array[wp.bool],
    world_converged: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    world_iteration_limit: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
    world_accepted: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    contact_residual_max: wp.array[wp.float32],
    limit_residual_max: wp.array[wp.float32],
    friction_residual_max: wp.array[wp.float32],
):
    world = wp.tid()
    if world_mask[world]:
        world_active[world] = True
        world_converged[world] = False
        world_failed[world] = False
        world_iteration_limit[world] = False
        iteration_count[world] = 0
        world_accepted[world] = False
        world_status[world] = LOX_STATUS_ACTIVE
        contact_residual_max[world] = 0.0
        limit_residual_max[world] = 0.0
        friction_residual_max[world] = 0.0


@wp.kernel
def _update_iteration_condition(
    max_iterations: wp.int32,
    world_active: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
    condition: wp.array[wp.int32],
):
    world = wp.tid()
    if world_active[world] and iteration_count[world] < max_iterations:
        wp.atomic_max(condition, 0, 1)


@wp.kernel
def _mark_iteration_limit(
    world_active: wp.array[wp.bool],
    world_iteration_limit: wp.array[wp.bool],
):
    world = wp.tid()
    if world_active[world]:
        world_active[world] = False
        world_iteration_limit[world] = True


@wp.kernel
def _initialize_body_velocity_guess(
    body_block: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
    velocity_start: wp.array[vec6f],
    external_wrench: wp.array[wp.spatial_vectorf],
    gravity: wp.array[wp.vec3f],
    time_step: wp.array[wp.float32],
    fraction: float,
    velocity_guess: wp.array[vec6f],
):
    body = wp.tid()
    dt = time_step[body_world[body]]
    guess = velocity_start[body]
    if body_block[body] >= 0 and fraction > 0.0:
        wrench = external_wrench[body]
        inv_mass = inverse_mass[body]
        linear_acceleration = inv_mass * wp.vec3f(wrench[0], wrench[1], wrench[2])
        if inv_mass > 0.0:
            linear_acceleration += gravity[body_world[body]]
        angular_acceleration = inverse_inertia_world[body] @ wp.vec3f(wrench[3], wrench[4], wrench[5])
        for axis in range(3):
            guess[axis] += fraction * dt * linear_acceleration[axis]
            guess[axis + 3] += fraction * dt * angular_acceleration[axis]
    velocity_guess[body] = guess


class LOXSolver:
    """Run a fixed LOX splitting solve on one frozen linearization.

    The caller owns collision detection, Jacobian construction, pose
    integration, and nonlinear relinearization. This class owns the smooth
    system update, weight construction, blocked dense LLT solve, unilateral
    projections, convergence freezing, and output conversion for one such
    linearization. A nonfailed world that reaches the iteration limit is
    accepted using its last projected iterate.
    """

    def __init__(
        self,
        model: ModelKamino,
        data: DataKamino,
        jacobians: SparseSystemJacobians,
        limits: LimitsKamino | None,
        contacts: ContactsKamino | None,
        config: LOXSolverConfig,
        source_model: Model,
        constraints: ConstraintStabilizationConfig,
        rotation_correction: JointCorrectionMode,
    ):
        rigid_adapter = None
        if model.size.sum_of_num_bodies > 0:
            rigid_adapter = LOXKaminoAdapter(
                model=model,
                data=data,
                jacobians=jacobians,
                limits=limits,
                contacts=contacts,
                eliminate_fixed_world_islands=config.eliminate_fixed_world_islands,
                projection_method=config.projection_method,
                rotation_correction=rotation_correction,
                joint_proximal_relaxation=config.joint_proximal_relaxation,
                rod_proximal_relaxation=config.rod_proximal_relaxation,
            )
        if rigid_adapter is None:
            raise ValueError("LOX requires at least one rigid body.")
        self._config = config
        self._newton_model = source_model
        self.model = model
        self._constraints = constraints
        self.rigid_adapter = rigid_adapter
        self.device = model.device
        self.num_worlds = model.info.num_worlds
        self.max_iterations = config.max_iterations
        self.use_graph_conditionals = config.use_graph_conditionals
        self.fixed_iterations = config.fixed_iterations
        self.projection_iterations = config.projection_iterations
        self.projection_method = config.projection_method
        self.gauss_seidel_max_colors = config.gauss_seidel_max_colors
        self._colored_gauss_seidel = None
        self.inertial_warmstart_fraction = config.inertial_warmstart_fraction
        self.position_tolerance = config.position_tolerance
        self.rotation_tolerance = config.rotation_tolerance
        self.velocity_tolerance = config.velocity_tolerance
        self.weight_sigma = config.weight_sigma
        self.weight_beta = config.weight_beta
        self.joint_penalty_scale = wp.full(
            self.num_worlds, config.joint_penalty_scale, dtype=wp.float32, device=self.device
        )
        self.joint_multiplier_projected_fraction = config.joint_multiplier_projected_fraction
        self.joint_warmstart_factor = config.joint_warmstart_factor
        self._bind_rigid_topology()
        self.world_accepted = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        self.world_status = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self._iteration_condition = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._projection_theta = (
            wp.ones(self.num_worlds, dtype=wp.float32, device=self.device)
            if config.projection_method == "apgd"
            else None
        )
        self._projection_beta = (
            wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
            if config.projection_method == "apgd"
            else None
        )
        self._projection_restart_dot = (
            wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
            if config.projection_method == "apgd"
            else None
        )
        self.has_bounded_effort = rigid_adapter is not None and rigid_adapter.has_bounded_effort
        self._time_step: wp.array[wp.float32] | None = None
        self._inverse_time_step: wp.array[wp.float32] | None = None
        self._time_step_prepared = False

    def _bind_rigid_topology(self) -> None:
        """Bind solver views and scratch arrays to the current rigid topology."""
        rigid_adapter = self.rigid_adapter
        self.system = rigid_adapter.system if rigid_adapter is not None else None
        self._has_dynamic_rigid_bodies = self.system is not None and bool(self.system.dynamic_bodies)
        if self.system is not None:
            self.system.selective_body_weights = self._config.selective_weights
        self.splitting = rigid_adapter.splitting if rigid_adapter is not None else None
        self.projected_twist = (
            self.splitting.projected_twist
            if self.splitting is not None
            else wp.empty(0, dtype=vec6f, device=self.device)
        )
        self.world_active = (
            self.splitting.world_active
            if self.splitting is not None
            else wp.ones(self.num_worlds, dtype=wp.bool, device=self.device)
        )
        self.world_converged = (
            self.splitting.world_converged
            if self.splitting is not None
            else wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        )
        self.world_failed = (
            self.splitting.world_failed
            if self.splitting is not None
            else wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        )
        self.world_iteration_limit = (
            self.splitting.world_iteration_limit
            if self.splitting is not None
            else wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        )
        self.iteration_count = (
            self.splitting.iteration_count
            if self.splitting is not None
            else wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        )
        self.contact_residual_max = (
            rigid_adapter.world_contact_residual_max
            if rigid_adapter is not None
            else wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        )
        self.limit_residual_max = (
            rigid_adapter.world_limit_residual_max
            if rigid_adapter is not None
            else wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        )
        self.friction_residual_max = (
            rigid_adapter.world_friction_residual_max
            if rigid_adapter is not None
            else wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        )
        rigid_body_count = rigid_adapter.model.size.sum_of_num_bodies if rigid_adapter is not None else 0
        self._initial_twist = wp.zeros(rigid_body_count, dtype=vec6f, device=self.device)

    def _make_structural_delassus(self) -> BatchedStructuralDelassus:
        rigid_adapter = self.rigid_adapter
        return BatchedStructuralDelassus(
            body_system=self.system,
            body_first_global=rigid_adapter.structural_body_first_global,
            body_second_global=rigid_adapter.structural_body_second_global,
            jacobian_first=rigid_adapter.structural_jacobian_first,
            jacobian_second=rigid_adapter.structural_jacobian_second,
        )

    def rebuild_rigid_topology(self) -> None:
        """Rebuild rigid allocations after dynamic-body classification changes."""
        rigid_adapter = self.rigid_adapter
        if rigid_adapter is None:
            return
        rigid_adapter.rebuild_dynamic_body_topology()
        self._colored_gauss_seidel = None
        self._bind_rigid_topology()
        self.has_bounded_effort = rigid_adapter.has_bounded_effort
        self.reset()

    def joint_penalty_scale_seed(
        self,
        time_step: float,
    ) -> list[float]:
        """Estimate and apply a timestep-aware structural ALM scale.

        The estimate is the reciprocal of the discrete second-percentile
        positive eigenvalue of the effective-mass-normalized structural
        Delassus. Both the percentile and the resulting scale are evaluated
        independently for each world. Systems with fewer than 50 positive
        modes therefore retain the minimum-eigenvalue estimate. The smooth
        operator excludes the body contact-consensus metric, which is a
        separate splitting concern.

        This operation prepares the rigid adapter from the initial rigid state and
        performs a one-time dense structural assembly and host eigensolve, so
        it is intended for initialization rather than the captured simulation
        loop.

        Args:
            time_step: Uniform simulation time step [s].

        Returns:
            The estimated dimensionless structural ALM penalty scale for each
            world.
        """
        rigid_adapter = self.rigid_adapter
        if rigid_adapter is None:
            raise ValueError("joint penalty scale seeding requires a LOX rigid-body system.")
        rigid_adapter.prepare_joint_penalty_scale_seed(time_step)
        time_step_array = rigid_adapter.model.time.dt
        inverse_time_step = rigid_adapter.model.time.inv_dt
        validate_world_time_steps(time_step_array, inverse_time_step, self.num_worlds, self.device)
        if rigid_adapter.structural_row_count == 0:
            return self.joint_penalty_scale.numpy().tolist()

        self.reset()
        rigid_adapter.begin_time_step(time_step_array, inverse_time_step)
        try:
            rigid_adapter.body_linearization_twist.zero_()
            rigid_adapter.update(
                time_step_array,
                joint_penalty_scale=1.0,
                linearization_twist=rigid_adapter.body_linearization_twist,
                assemble_structural_penalty=False,
            )
            wp.copy(self.system.weighted_matrix, self.system.smooth_matrix)
            self.system.factorize()

            structural_delassus = self._make_structural_delassus()
            structural_delassus.assemble()
            matrix_values = structural_delassus.matrix.numpy()
            matrix_offsets = structural_delassus.info.mio.numpy()
            vector_offsets = structural_delassus.info.vio.numpy()
            vector_rows = structural_delassus.vector_row.numpy()
            effective_mass = rigid_adapter.structural_effective_mass.numpy()
            matrix_epsilon = np.finfo(matrix_values.dtype).eps

            seeds = self.joint_penalty_scale.numpy().astype(np.float64)
            positive_by_world: list[list[np.ndarray]] = [[] for _ in range(rigid_adapter.num_worlds)]
            for component, row_count in enumerate(structural_delassus.component_row_counts):
                if row_count == 0:
                    continue
                matrix_offset = int(matrix_offsets[component])
                vector_offset = int(vector_offsets[component])
                rows = vector_rows[vector_offset : vector_offset + row_count]
                row_metric = effective_mass[rows].astype(np.float64)
                metric_sqrt = np.sqrt(row_metric)
                matrix = (
                    matrix_values[matrix_offset : matrix_offset + row_count * row_count]
                    .reshape(row_count, row_count)
                    .astype(np.float64)
                )
                normalized = metric_sqrt[:, None] * matrix * metric_sqrt[None, :]
                eigenvalues = np.linalg.eigvalsh(normalized)
                positive_threshold = matrix_epsilon * row_count * float(eigenvalues[-1])
                positive = eigenvalues[eigenvalues > positive_threshold]
                if positive.size == 0:
                    world = structural_delassus.component_world_host[component]
                    raise RuntimeError(
                        f"World {world} component {component} structural Delassus has no resolvable positive eigenvalue."
                    )
                positive_by_world[structural_delassus.component_world_host[component]].append(positive)

            for world, component_eigenvalues in enumerate(positive_by_world):
                if not component_eigenvalues:
                    continue
                positive = np.sort(np.concatenate(component_eigenvalues))
                seed = 1.0 / _low_mode_eigenvalue(positive)
                if not math.isfinite(seed) or seed > np.finfo(np.float32).max:
                    raise RuntimeError(f"World {world} structural ALM penalty seed is not representable in float32.")
                seeds[world] = seed

            applied_seeds = seeds.astype(np.float32)
            self.joint_penalty_scale.assign(applied_seeds)
            return applied_seeds.tolist()
        finally:
            self.reset()

    def solve_forward_dynamics(
        self,
        _contacts: object | None = None,
    ) -> None:
        """Solve one LOX forward-dynamics step from prepared Kamino data."""
        constraints = self._constraints
        time_step = self.model.time.dt
        inverse_time_step = self.model.time.inv_dt
        rigid_adapter = self.rigid_adapter
        self.begin_time_step(
            time_step,
            inverse_time_step,
            limit_stabilization_fraction=constraints.beta,
            contact_stabilization_fraction=constraints.gamma,
            contact_dead_zone=constraints.delta,
            impact_velocity_threshold=self._config.impact_velocity_threshold,
            contact_recoverable_response=self._config.contact_recoverable_response,
        )

        linearization_twist = None
        if rigid_adapter is not None:
            rigid_adapter.body_linearization_twist.zero_()
            linearization_twist = rigid_adapter.body_linearization_twist
        self.solve(linearization_twist=linearization_twist, write_output=False)

        if rigid_adapter is not None:
            rigid_adapter.write_outputs(time_step, inverse_time_step, write_body_velocity=False)
        if rigid_adapter is not None:
            model = rigid_adapter.model
            data = rigid_adapter.data
            update_body_wrenches(model.bodies, data.bodies)
            # Expose the accepted LOX velocity as equivalent inputs to the
            # selected Kamino integrator.
            wp.launch(
                _write_integrator_body_inputs,
                dim=model.size.sum_of_num_bodies,
                inputs=[
                    rigid_adapter.system.body_vector_index,
                    model.bodies.wid,
                    self.world_accepted,
                    time_step,
                    model.bodies.m_i,
                    data.bodies.I_i,
                    model.bodies.inv_m_i,
                    model.bodies.inv_i_I_i,
                    model.gravity.vector,
                    rigid_adapter.body_velocity_begin,
                    self.projected_twist,
                ],
                outputs=[data.bodies.w_i, data.bodies.u_i],
                device=model.device,
            )

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Refresh or rebuild the rigid adapter after model changes."""
        rigid_adapter = self.rigid_adapter
        if rigid_adapter is None:
            return
        if flags & ModelFlags.BODY_PROPERTIES and rigid_adapter.dynamic_body_topology_changed():
            self.rebuild_rigid_topology()
        else:
            rigid_adapter.notify_model_changed()

    def validate_model_changed(self, *, use_fk_solver: bool) -> None:
        """Validate LOX-specific values derived from the Newton model."""
        # Host-side validation cannot synchronize aliased Newton arrays while a
        # CUDA graph is being captured. The same topology was validated when the
        # solver was built; captured property updates retain that topology.
        if self.device.is_cuda and self.device.is_capturing:
            return
        if self._newton_model is not None and self.rigid_adapter is not None and self.rigid_adapter.rods.count > 0:
            validate_rod_model(self._newton_model, use_fk_solver=use_fk_solver)
        if self.rigid_adapter is not None:
            self.rigid_adapter.validate_model_changed()

    def reset(
        self,
        problem: DualProblem | None = None,
        world_mask: wp.array[wp.bool] | None = None,
    ) -> None:
        """Clear structural/splitting warm starts and per-world diagnostics."""
        del problem
        if world_mask is not None:
            if world_mask.shape != (self.num_worlds,) or world_mask.dtype != wp.bool:
                raise ValueError(f"world_mask must have shape ({self.num_worlds},) and dtype bool.")
            if world_mask.device != self.device:
                raise ValueError(f"world_mask must be allocated on {self.device}, found {world_mask.device}.")
        if self.splitting is not None:
            self.splitting.reset(world_mask=world_mask)
            self.rigid_adapter.reset_structural_multipliers(world_mask=world_mask)
            if self.has_bounded_effort:
                self.rigid_adapter.reset_effort_counters(world_mask=world_mask)
            self.rigid_adapter.rods.reset()
            self.rigid_adapter.projection_status.zero_()
            self.rigid_adapter.contact_residual.zero_()
            self.rigid_adapter.limit_residual.zero_()
            self.rigid_adapter.friction_residual.zero_()
            self.rigid_adapter.reset_friction_reactions(world_mask=world_mask)
        elif world_mask is None:
            self.world_active.fill_(True)
            self.world_converged.zero_()
            self.world_failed.zero_()
            self.world_iteration_limit.zero_()
            self.iteration_count.zero_()
        if world_mask is None:
            self.world_accepted.zero_()
            self.world_status.fill_(LOX_STATUS_ACTIVE)
            self.contact_residual_max.zero_()
            self.limit_residual_max.zero_()
            self.friction_residual_max.zero_()
        else:
            wp.launch(
                _reset_solver_worlds_masked,
                dim=self.num_worlds,
                inputs=[world_mask],
                outputs=[
                    self.world_active,
                    self.world_converged,
                    self.world_failed,
                    self.world_iteration_limit,
                    self.iteration_count,
                    self.world_accepted,
                    self.world_status,
                    self.contact_residual_max,
                    self.limit_residual_max,
                    self.friction_residual_max,
                ],
                device=self.device,
            )
        self._initial_twist.zero_()
        self._time_step_prepared = False

    def load_state_dual_impulses(
        self,
        body_dual_impulse: wp.array[wp.spatial_vector] | None,
    ) -> None:
        """Load consensus impulse warm starts from the input Newton state."""
        if self.splitting is not None:
            if body_dual_impulse is None:
                raise ValueError("LOX rigid bodies require State.body_lox_dual_impulse.")
            if body_dual_impulse.shape != (self.rigid_adapter.model.size.sum_of_num_bodies,):
                raise ValueError("State.body_lox_dual_impulse must contain one entry per body.")
            if body_dual_impulse.dtype != wp.spatial_vectorf or body_dual_impulse.device != self.device:
                raise ValueError("State.body_lox_dual_impulse has an incompatible dtype or device.")
            wp.copy(self.splitting.splitting_dual_impulse, body_dual_impulse.view(dtype=vec6f))

    def write_state_dual_impulses(
        self,
        body_dual_impulse: wp.array[wp.spatial_vector] | None,
    ) -> None:
        """Write consensus impulse warm starts to the output Newton state."""
        if self.splitting is not None:
            if body_dual_impulse is None:
                raise ValueError("LOX rigid bodies require State.body_lox_dual_impulse.")
            if body_dual_impulse.shape != (self.rigid_adapter.model.size.sum_of_num_bodies,):
                raise ValueError("State.body_lox_dual_impulse must contain one entry per body.")
            if body_dual_impulse.dtype != wp.spatial_vectorf or body_dual_impulse.device != self.device:
                raise ValueError("State.body_lox_dual_impulse has an incompatible dtype or device.")
            wp.copy(body_dual_impulse.view(dtype=vec6f), self.splitting.splitting_dual_impulse)

    def begin_time_step(
        self,
        time_step: wp.array[wp.float32],
        inverse_time_step: wp.array[wp.float32],
        limit_stabilization_fraction: float = 0.01,
        contact_stabilization_fraction: float = 0.01,
        contact_dead_zone: float = 1.0e-6,
        impact_velocity_threshold: float = 1.0e-3,
        contact_recoverable_response: bool = False,
        reset_dual: bool = False,
    ) -> None:
        """Freeze inertial velocities and import damped constraint warm starts.

        The body consensus warm start is loaded from the Newton input state as
        the generalized impulse ``W u`` and converted back to the scaled dual
        after the solve builds its current body weight.
        """
        validate_world_time_steps(time_step, inverse_time_step, self.num_worlds, self.device)
        self._time_step = time_step
        self._inverse_time_step = inverse_time_step
        if self.rigid_adapter is not None:
            self.rigid_adapter.scale_structural_multipliers(self.joint_warmstart_factor)
            self.rigid_adapter.begin_time_step(
                time_step,
                inverse_time_step,
                limit_stabilization_fraction=limit_stabilization_fraction,
                contact_stabilization_fraction=contact_stabilization_fraction,
                contact_dead_zone=contact_dead_zone,
                impact_velocity_threshold=impact_velocity_threshold,
                contact_recoverable_response=contact_recoverable_response,
            )
            wp.launch(
                _initialize_body_velocity_guess,
                dim=self.rigid_adapter.model.size.sum_of_num_bodies,
                inputs=[
                    self.system.body_block,
                    self.rigid_adapter.model.bodies.wid,
                    self.rigid_adapter.model.bodies.inv_m_i,
                    self.rigid_adapter.data.bodies.inv_I_i,
                    self.rigid_adapter.body_velocity_begin,
                    self.rigid_adapter.data.bodies.w_e_i,
                    self.rigid_adapter.model.gravity.vector,
                    time_step,
                    self.inertial_warmstart_fraction,
                ],
                outputs=[self._initial_twist],
                device=self.device,
            )
            if reset_dual:
                self.splitting.splitting_dual.zero_()
                self.splitting.splitting_dual_impulse.zero_()
        self._time_step_prepared = True

    def _prepare_colored_gauss_seidel_projection(self) -> None:
        projection_adapter = self.rigid_adapter
        if self._colored_gauss_seidel is None or self._colored_gauss_seidel.rigid_adapter is not projection_adapter:
            self._colored_gauss_seidel = ColoredGaussSeidelProjection(
                projection_adapter,
                self.gauss_seidel_max_colors,
            )
        prepared_status = self.rigid_adapter.world_jacobi_projection_status
        inverse_weight = self.system.inverse_weight
        self._colored_gauss_seidel.prepare(inverse_weight, prepared_status)

    def _prepare_body_space_projection(self) -> None:
        rigid_adapter = self.rigid_adapter
        if not self._has_dynamic_rigid_bodies:
            rigid_adapter.projection_status.fill_(PROJECTION_STATUS_VALID)
            return
        prepare_physical_projection_data(
            rigid_adapter.friction_world,
            rigid_adapter.friction_local,
            rigid_adapter.world_friction_count,
            rigid_adapter.friction_body_first,
            rigid_adapter.friction_body_second,
            rigid_adapter.friction_jacobian_first,
            rigid_adapter.friction_jacobian_second,
            rigid_adapter.contact_world,
            rigid_adapter.contact_local,
            rigid_adapter.world_contact_count,
            rigid_adapter.contact_body_first,
            rigid_adapter.contact_body_second,
            rigid_adapter.contact_jacobian_first,
            rigid_adapter.contact_jacobian_second,
            rigid_adapter.contact_bias,
            rigid_adapter.contact_friction,
            rigid_adapter.limit_world,
            rigid_adapter.limit_local,
            rigid_adapter.world_limit_count,
            rigid_adapter.limit_body_first,
            rigid_adapter.limit_body_second,
            rigid_adapter.limit_jacobian_first,
            rigid_adapter.limit_jacobian_second,
            self.system.inverse_weight,
            rigid_adapter.friction_physical_delassus,
            rigid_adapter.contact_physical_delassus,
            rigid_adapter.contact_prepared_delassus,
            rigid_adapter.limit_physical_delassus,
            rigid_adapter.world_physical_projection_status,
        )
        if self.projection_method in ("jacobi", "apgd") or (
            self.projection_method == "gauss_seidel" and self.gauss_seidel_max_colors == 1
        ):
            prepare_jacobi_projection_data(
                rigid_adapter.friction_world,
                rigid_adapter.friction_local,
                rigid_adapter.world_friction_count,
                rigid_adapter.friction_body_first,
                rigid_adapter.friction_body_second,
                rigid_adapter.friction_jacobian_first,
                rigid_adapter.friction_jacobian_second,
                rigid_adapter.contact_world,
                rigid_adapter.contact_local,
                rigid_adapter.world_contact_count,
                rigid_adapter.contact_body_first,
                rigid_adapter.contact_body_second,
                rigid_adapter.contact_jacobian_first,
                rigid_adapter.contact_jacobian_second,
                rigid_adapter.contact_bias,
                rigid_adapter.contact_friction,
                rigid_adapter.limit_world,
                rigid_adapter.limit_local,
                rigid_adapter.world_limit_count,
                rigid_adapter.limit_body_first,
                rigid_adapter.limit_body_second,
                rigid_adapter.limit_jacobian_first,
                rigid_adapter.limit_jacobian_second,
                rigid_adapter.body_constraint_count,
                rigid_adapter.static_body_constraint_count,
                self.system.inverse_weight,
                rigid_adapter.friction_projection_delassus,
                rigid_adapter.contact_projection_delassus,
                rigid_adapter.limit_projection_delassus,
                rigid_adapter.world_jacobi_projection_status,
            )
        elif self.projection_method == "gauss_seidel" and self.gauss_seidel_max_colors > 1:
            self._prepare_colored_gauss_seidel_projection()

    def _project_body_space_constraints(self) -> None:
        if not self._has_dynamic_rigid_bodies:
            return
        rigid_adapter = self.rigid_adapter
        splitting = self.splitting
        if self.projection_method in ("jacobi", "apgd") or (
            self.projection_method == "gauss_seidel" and self.gauss_seidel_max_colors == 1
        ):
            project_constraints_jacobi(
                self.projection_iterations,
                splitting.world_active,
                splitting.body_world,
                rigid_adapter.friction_world,
                rigid_adapter.friction_local,
                rigid_adapter.world_friction_count,
                rigid_adapter.friction_body_first,
                rigid_adapter.friction_body_second,
                rigid_adapter.friction_jacobian_first,
                rigid_adapter.friction_jacobian_second,
                rigid_adapter.friction_impulse_bound,
                rigid_adapter.friction_projection_delassus,
                rigid_adapter.contact_world,
                rigid_adapter.contact_local,
                rigid_adapter.world_contact_count,
                rigid_adapter.contact_body_first,
                rigid_adapter.contact_body_second,
                rigid_adapter.contact_jacobian_first,
                rigid_adapter.contact_jacobian_second,
                rigid_adapter.contact_bias,
                rigid_adapter.contact_friction,
                rigid_adapter.contact_projection_delassus,
                rigid_adapter.limit_world,
                rigid_adapter.limit_local,
                rigid_adapter.world_limit_count,
                rigid_adapter.limit_body_first,
                rigid_adapter.limit_body_second,
                rigid_adapter.limit_jacobian_first,
                rigid_adapter.limit_jacobian_second,
                rigid_adapter.limit_bias,
                rigid_adapter.limit_projection_delassus,
                self.system.inverse_weight,
                splitting.projected_twist,
                rigid_adapter.projection_twist_delta,
                rigid_adapter.contact_reaction,
                rigid_adapter.limit_reaction,
                rigid_adapter.friction_reaction,
                rigid_adapter.world_jacobi_projection_status,
                rigid_adapter.projection_status,
                world_body_offset=rigid_adapter.model.info.bodies_offset,
                world_body_count=rigid_adapter.model.info.num_bodies,
                world_friction_offset=rigid_adapter.world_friction_offset,
                world_contact_offset=rigid_adapter.world_contact_offset,
                world_limit_offset=rigid_adapter.world_limit_offset,
                accelerated=self.projection_method == "apgd",
                theta=self._projection_theta,
                beta=self._projection_beta,
                restart_dot=self._projection_restart_dot,
                friction_trial=rigid_adapter.friction_acceleration_trial,
                friction_previous=rigid_adapter.friction_acceleration_previous,
                contact_trial=rigid_adapter.contact_acceleration_trial,
                contact_previous=rigid_adapter.contact_acceleration_previous,
                limit_trial=rigid_adapter.limit_acceleration_trial,
                limit_previous=rigid_adapter.limit_acceleration_previous,
            )
        elif self.projection_method == "gauss_seidel" and self.gauss_seidel_max_colors > 1:
            self._colored_gauss_seidel.project(
                self.projection_iterations,
                splitting.world_active,
                splitting.body_world,
                self.system.inverse_weight,
                splitting.projected_twist,
                rigid_adapter.projection_twist_delta,
                rigid_adapter.world_jacobi_projection_status,
                rigid_adapter.projection_status,
            )

    def _update_conditional_iteration(self) -> None:
        self._iteration_condition.zero_()
        wp.launch(
            _update_iteration_condition,
            dim=self.num_worlds,
            inputs=[
                self.max_iterations,
                self.world_active,
                self.iteration_count,
            ],
            outputs=[self._iteration_condition],
            device=self.device,
        )

    def _prepare_body_space_candidate(
        self,
        time_step: wp.array[wp.float32],
        linearization_twist: wp.array[vec6f],
    ) -> None:
        """Solve and prepare the rigid candidate without unilateral projection."""
        rigid_adapter = self.rigid_adapter
        system = self.system
        splitting = self.splitting
        if self.has_bounded_effort:
            rigid_adapter.promote_effort_counters(splitting.world_active)
            system.solve_candidate_with_effort(
                splitting.projected_twist,
                splitting.splitting_dual,
                rigid_adapter.body_effort_offset,
                rigid_adapter.body_effort_index,
                rigid_adapter.body_effort_side,
                rigid_adapter.effort_dynamic_row_index,
                rigid_adapter.dynamic_jacobian_first,
                rigid_adapter.dynamic_jacobian_second,
                rigid_adapter.effort_counter_applied,
                rigid_adapter.body_velocity_begin,
            )
        else:
            system.solve_candidate(
                splitting.projected_twist,
                splitting.splitting_dual,
                rigid_adapter.body_velocity_begin,
            )
        rigid_adapter.rods.update_proximal(
            system,
            system.body_solution,
            linearization_twist,
            splitting.world_active,
            time_step,
            self.position_tolerance,
            self.rotation_tolerance,
            self.velocity_tolerance,
        )
        splitting.prepare_projection(system.body_solution)

    def _finish_body_space_iteration(
        self, time_step: wp.array[wp.float32], linearization_twist: wp.array[vec6f]
    ) -> None:
        """Update structural state and finish the rigid splitting iteration."""
        rigid_adapter = self.rigid_adapter
        splitting = self.splitting
        rigid_adapter.update_structural_multipliers_from_twist(
            time_step,
            min(self.position_tolerance, self.rotation_tolerance),
            linearization_twist,
            splitting.global_twist,
            splitting.projected_twist,
            splitting.world_active,
            projected_fraction=self.joint_multiplier_projected_fraction,
        )
        if self.has_bounded_effort:
            rigid_adapter.update_effort_counters(
                time_step,
                self.velocity_tolerance,
                splitting.projected_twist,
                splitting.world_active,
            )
        if not self.fixed_iterations:
            rigid_adapter.evaluate_lagged_velocity_consistency(
                self.velocity_tolerance,
                splitting.global_twist,
                splitting.projected_twist_previous,
                splitting.world_active,
            )
            splitting.finish_iteration(
                rigid_adapter.projection_status,
                time_step,
                self.position_tolerance,
                self.rotation_tolerance,
                self.velocity_tolerance,
                effort_residual=(rigid_adapter.world_effort_residual_max if self.has_bounded_effort else None),
                structural_residual=rigid_adapter.world_structural_residual,
                projected_structural_residual=rigid_adapter.world_projected_structural_residual,
                lagged_velocity_residual=rigid_adapter.world_lagged_velocity_residual,
                lagged_velocity_required=rigid_adapter.world_lagged_velocity_required,
                proximal_residual=rigid_adapter.rods.world_proximal_residual,
                proximal_failed=rigid_adapter.rods.world_proximal_failed,
            )
        else:
            splitting.finish_fixed_iteration(
                rigid_adapter.projection_status,
                proximal_failed=rigid_adapter.rods.world_proximal_failed,
            )

    def _body_space_iteration(
        self,
        time_step: wp.array[wp.float32],
        linearization_twist: wp.array[vec6f],
        conditional: bool,
    ) -> None:
        self._prepare_body_space_candidate(time_step, linearization_twist)
        self._project_body_space_constraints()
        self._finish_body_space_iteration(time_step, linearization_twist)
        if conditional:
            self._update_conditional_iteration()

    def solve(
        self,
        *,
        initial_twist: wp.array[vec6f] | None = None,
        linearization_twist: wp.array[vec6f] | None = None,
        write_output: bool = True,
    ) -> None:
        """Assemble and solve one frozen-contact smooth linearization.

        Args:
            initial_twist: Optional initial projected body twist.
            linearization_twist: Optional body twist used to linearize joint constraints.
            write_output: Whether to write the solution to the Kamino containers.
        """
        if not self._time_step_prepared:
            raise RuntimeError("begin_time_step() must be called before solving LOX.")
        time_step = self._time_step
        inverse_time_step = self._inverse_time_step
        if time_step is None or inverse_time_step is None:
            raise RuntimeError("LOX per-world timestep arrays are not prepared.")
        if self.splitting is not None:
            if initial_twist is None:
                initial_twist = self._initial_twist
            self.splitting.begin(initial_twist, reset_dual=False)
        self.world_accepted.zero_()
        self.world_status.fill_(LOX_STATUS_ACTIVE)
        if self.rigid_adapter is not None:
            self.rigid_adapter.contact_residual.zero_()
            self.rigid_adapter.limit_residual.zero_()
            self.rigid_adapter.friction_residual.zero_()
        self.contact_residual_max.zero_()
        self.limit_residual_max.zero_()
        self.friction_residual_max.zero_()
        rigid_adapter = self.rigid_adapter
        system = self.system
        splitting = self.splitting
        if rigid_adapter is not None:
            rigid_adapter.update(
                time_step,
                joint_penalty_scale=self.joint_penalty_scale,
                linearization_twist=linearization_twist,
                assemble_structural_penalty=True,
            )
            if linearization_twist is None:
                linearization_twist = rigid_adapter.body_linearization_twist
            system.build_weighted_matrix(
                body_has_unilateral=rigid_adapter.body_has_unilateral,
                sigma=self.weight_sigma,
                beta=self.weight_beta,
            )
            splitting.restore_dual_from_impulse(system.inverse_weight, rigid_adapter.body_has_unilateral)
            system.factorize()
            self._prepare_body_space_projection()

        use_conditional_loop = (
            not self.fixed_iterations
            and self.use_graph_conditionals
            and (not self.device.is_cuda or not self.device.is_capturing or wp.is_conditional_graph_supported())
        )
        if use_conditional_loop:
            self._iteration_condition.fill_(1)
            wp.capture_while(
                self._iteration_condition,
                self._body_space_iteration,
                time_step=time_step,
                linearization_twist=linearization_twist,
                conditional=True,
            )
        else:
            for _iteration in range(self.max_iterations):
                self._body_space_iteration(time_step, linearization_twist, conditional=False)

        if splitting is not None:
            splitting.store_dual_impulse(system.weight, rigid_adapter.body_has_unilateral)
            splitting.mark_iteration_limit()
        else:
            wp.launch(
                _mark_iteration_limit,
                dim=self.num_worlds,
                inputs=[],
                outputs=[self.world_active, self.world_iteration_limit],
                device=self.device,
            )
        wp.launch(
            _finalize_world_status,
            dim=self.num_worlds,
            inputs=[self.world_converged, self.world_failed, self.world_iteration_limit],
            outputs=[self.world_accepted, self.world_status],
            device=self.device,
        )
        if rigid_adapter is not None:
            compute_projection_residuals(
                self.world_accepted,
                rigid_adapter.projection_status,
                rigid_adapter.friction_world,
                rigid_adapter.friction_local,
                rigid_adapter.world_friction_count,
                rigid_adapter.friction_body_first,
                rigid_adapter.friction_body_second,
                rigid_adapter.friction_jacobian_first,
                rigid_adapter.friction_jacobian_second,
                rigid_adapter.friction_impulse_bound,
                rigid_adapter.friction_reaction,
                rigid_adapter.friction_physical_delassus,
                rigid_adapter.contact_world,
                rigid_adapter.contact_local,
                rigid_adapter.world_contact_count,
                rigid_adapter.contact_body_first,
                rigid_adapter.contact_body_second,
                rigid_adapter.contact_jacobian_first,
                rigid_adapter.contact_jacobian_second,
                rigid_adapter.contact_bias,
                rigid_adapter.contact_friction,
                rigid_adapter.contact_reaction,
                rigid_adapter.contact_physical_delassus,
                rigid_adapter.limit_world,
                rigid_adapter.limit_local,
                rigid_adapter.world_limit_count,
                rigid_adapter.limit_body_first,
                rigid_adapter.limit_body_second,
                rigid_adapter.limit_jacobian_first,
                rigid_adapter.limit_jacobian_second,
                rigid_adapter.limit_bias,
                rigid_adapter.limit_reaction,
                rigid_adapter.limit_physical_delassus,
                splitting.projected_twist,
                rigid_adapter.friction_velocity,
                rigid_adapter.contact_velocity,
                rigid_adapter.limit_velocity,
                rigid_adapter.contact_residual,
                rigid_adapter.limit_residual,
                rigid_adapter.friction_residual,
                rigid_adapter.world_contact_residual_max,
                rigid_adapter.world_limit_residual_max,
                rigid_adapter.world_friction_residual_max,
            )
            if write_output:
                rigid_adapter.write_outputs(time_step, inverse_time_step, body_velocity=splitting.projected_twist)
