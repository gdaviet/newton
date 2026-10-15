# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deformable bodies of a LOX problem.

:class:`DeformableProblem` owns the smooth finite-element system of the particles of the
deformable bodies and their contact rows, the particle counterparts of the body system and
the unilateral rows of :class:`..problem.LOXProblem`. The LOX solver drives both through the
phases of each time step: the frozen linearization and the contacts at its beginning, the
consensus weights with the body weights, and the candidate solves and the proximal
updates of the splitting iterations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ....core.materials import MaterialMixMode
from ..types import ParticleContactRows
from .contact import DeformableContacts
from .penetration import DeformablePenetrationLimiter
from .self_contact import DeformableSelfContacts
from .system import DeformableSystem

if TYPE_CHECKING:
    from .......sim import Contacts, Model, State
    from .....config import LOXSolverConfig, MaterialManagerConfig
    from ....core.data import DataKamino
    from ....core.model import ModelKamino
    from ..types import LOXProblemData
    from .types import DeformableSystemData, DeformableTopology

###
# Module interface
###

__all__ = ["DeformableProblem"]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Interfaces
###


class DeformableProblem:
    """Own the smooth system and the contact rows of the particles of the deformable bodies.

    The rigid-soft contact rows are sized by the capacity of the Newton contacts of the first
    time step, see :meth:`ensure_contact_capacity`.

    Args:
        model: Newton model of the deformable bodies.
        kamino_model: Kamino model of the rigid bodies.
        kamino_data: Kamino data of the rigid bodies.
        config: LOX solver configuration.
        materials: Material mixing configuration of the contacts, or ``None`` for the defaults.
    """

    def __init__(
        self,
        model: Model,
        kamino_model: ModelKamino,
        kamino_data: DataKamino,
        config: LOXSolverConfig,
        materials: MaterialManagerConfig | None = None,
    ):
        self.model = model
        self.kamino_model = kamino_model
        self.kamino_data = kamino_data
        self.device = wp.get_device(model.device)
        self.num_worlds = kamino_model.info.num_worlds
        self.system = DeformableSystem(
            model,
            cr_iterations=config.deformable_cr_iterations,
            preconditioner=config.deformable_preconditioner,
            preconditioner_regularization=config.deformable_hessian_regularization,
            direct_max_particles=config.deformable_direct_max_particles,
            linear_solver=config.deformable_linear_solver,
            proximal_iterations=config.deformable_proximal_iterations,
            proximal_relaxation=config.deformable_proximal_relaxation,
            recycle_cr=config.deformable_recycle_cr,
            coarse_initial_guess=config.deformable_coarse_initial_guess,
            consensus_fraction=config.deformable_consensus_fraction,
            weight_beta=config.deformable_weight_beta or 0.0,
            weight_sigma=config.weight_sigma,
        )
        """Smooth finite-element system of the particles."""
        self.system.selective_consensus = config.selective_weights
        self.num_particles = self.system.particle_count
        """Number of packed particles."""
        self._friction_mix_mode = MaterialMixMode.AVERAGE
        self._restitution_mix_mode = MaterialMixMode.MIN
        if materials is not None:
            self._friction_mix_mode = MaterialMixMode.from_string(materials.friction_mix_mode)
            self._restitution_mix_mode = MaterialMixMode.from_string(materials.restitution_mix_mode)
        self._rigid_normal_cone_filtering = config.deformable_enable_rigid_contact_normal_cone_filtering
        self._normal_cone_filtering_min_distance = config.deformable_normal_cone_filtering_min_distance
        self.particle_unilateral_count = wp.zeros(self.num_particles, dtype=wp.int32, device=self.device)
        """Number of rows that couple each dynamic particle with a significant coefficient."""
        self.self_contacts: DeformableSelfContacts | None = None
        """Detector of the self-contact candidates of the deformable surfaces, with self-contacts or
        penetration-free contacts."""
        self._load_self_contacts = config.deformable_enable_self_contact
        if self._load_self_contacts and model.tri_count < 1:
            raise ValueError("LOX deformable self-contacts require a triangulated surface.")
        if model.tri_count > 0 and (
            config.deformable_enable_self_contact or config.deformable_enable_penetration_free_contact
        ):
            self.self_contacts = DeformableSelfContacts(
                model,
                margin=config.deformable_self_contact_margin,
                gap=config.deformable_self_contact_gap,
                vertex_buffer_size=config.deformable_self_contact_vertex_buffer_size,
                edge_buffer_size=config.deformable_self_contact_edge_buffer_size,
                topological_filter_threshold=config.deformable_self_contact_topological_filter_threshold,
                rest_exclusion_radius=config.deformable_self_contact_rest_exclusion_radius,
                edge_parallel_epsilon=config.deformable_self_contact_edge_parallel_epsilon,
                normal_cone_filtering=config.deformable_enable_normal_cone_filtering,
                normal_cone_filtering_min_distance=config.deformable_normal_cone_filtering_min_distance,
            )
        self.penetration_limiter: DeformablePenetrationLimiter | None = None
        """Step limiter of the penetration-free contacts, or ``None``."""
        if config.deformable_enable_penetration_free_contact and self.self_contacts is not None:
            # The limiter can truncate every dynamic particle, beyond the particles of the contact rows
            self.system.selective_consensus = False
            self.penetration_limiter = DeformablePenetrationLimiter(
                self.self_contacts,
                self.system.topology,
                self.system.data.inverse_weight,
                config.deformable_penetration_free_contact_relaxation,
            )
        self.contacts: DeformableContacts | None = None
        """Loader of the particle contact rows."""
        self._allocate_contacts(0)

    ###
    # Properties
    ###

    @property
    def topology(self) -> DeformableTopology:
        """Returns the packed particle layout."""
        return self.system.topology

    @property
    def system_data(self) -> DeformableSystemData:
        """Returns the arrays of the smooth system."""
        return self.system.data

    @property
    def contact_rows(self) -> ParticleContactRows:
        """Returns the particle contact rows."""
        return self.contacts.data

    @property
    def particle_world(self) -> wp.array[wp.int32]:
        """Returns the world of each packed particle."""
        return self.system.topology.packed_solve_world

    @property
    def preconditioner_status(self) -> wp.array[wp.int32] | None:
        """Returns the :class:`.types.DeformablePreconditionerStatus` of each world, or ``None`` without a
        factorized preconditioner."""
        return getattr(self.system.preconditioner, "world_status", None)

    ###
    # Public API
    ###

    def ensure_contact_capacity(self, contacts: Contacts | None) -> bool:
        """Size the rigid-soft contact rows for the soft contact capacity of the Newton contacts.

        Returns:
            Whether the rows were reallocated, which invalidates the projection schedule.
        """
        rigid_capacity = contacts.soft_contact_max if contacts is not None else 0
        if rigid_capacity == self.contact_rows.rigid_capacity:
            return False
        if self.device.is_cuda and self.device.is_capturing and not self.device.is_mempool_enabled:
            raise RuntimeError(
                "LOX particle contact rows cannot be allocated during CUDA graph capture without a stream-ordered "
                "memory pool. Enable the Warp memory pool or run one step before the capture."
            )
        self._allocate_contacts(rigid_capacity)
        return True

    def begin_time_step(self, state: State, time_step: wp.array[wp.float32]) -> None:
        """Assemble the frozen smooth system of the time step at the begin-step particle state, and detect
        the self-contact candidates."""
        self.system.assemble(state, time_step, finalize_consensus=False)
        if self.self_contacts is not None:
            self.self_contacts.detect(state.particle_q)
        if self.penetration_limiter is not None:
            self.penetration_limiter.begin_time_step(state.particle_q)

    def load_contacts(
        self,
        contacts: Contacts | None,
        state: State,
        time_step: wp.array[wp.float32],
        stabilization_fraction: float,
        dead_zone: float,
        problem_data: LOXProblemData,
        body_vector_index: wp.array[wp.int32],
    ) -> None:
        """Load the particle contact rows of the time step and count their incidences.

        Args:
            contacts: Newton contacts of the time step, or ``None``.
            state: Begin-step Newton state.
            time_step: Time step of each world [s].
            stabilization_fraction: Fraction of the contact penetration corrected in one step.
            dead_zone: Symmetric contact distance dead zone of the stabilization [m].
            problem_data: Rows and body data of the LOX problem, whose body incidence the rows extend.
            body_vector_index: First entry of each body in the packed body system vectors, or ``-1``.
        """
        full_inverse_weight = self.system.data.full_inverse_weight
        self.contacts.load(
            contacts,
            state.particle_q,
            state.particle_qd,
            full_inverse_weight,
            time_step,
            stabilization_fraction,
            dead_zone,
        )
        if self._load_self_contacts:
            topology = self.system.topology
            self.self_contacts.load(
                self.contact_rows,
                state.particle_q,
                state.particle_qd,
                topology.newton_to_packed,
                topology.packed_solve_world,
                full_inverse_weight,
                time_step,
                stabilization_fraction,
                dead_zone,
            )
        self.contacts.finish(problem_data, body_vector_index, full_inverse_weight, self.particle_unilateral_count)

    def build_consensus_weights(self) -> None:
        """Select the consensus weights of the particles touched by contact rows and factorize the candidate
        matrix ``A + alpha W``."""
        self.system.set_unilateral_incidence(self.particle_unilateral_count)

    def solve_candidate(
        self,
        consensus_center: wp.array[wp.vec3f],
        world_active: wp.array[wp.bool],
        use_cudss: bool | None = None,
    ) -> None:
        """Solve the smooth system for the candidate particle velocities of one iteration."""
        self.system.solve_candidate(consensus_center, world_active=world_active, use_cudss=use_cudss)

    def update_proximal(self, time_step: wp.array[wp.float32]) -> None:
        """Update the nonlinear element corrections of the next candidate solve."""
        self.system.update_proximal(time_step)

    def limit_penetration(
        self,
        projected_velocity: wp.array[wp.vec3f],
        world_active: wp.array[wp.bool],
        time_step: wp.array[wp.float32],
    ) -> None:
        """Truncate the projected particle velocities within the penetration-free step bounds, if enabled."""
        if self.penetration_limiter is not None:
            self.penetration_limiter.truncate(projected_velocity, world_active, time_step)

    ###
    # Internals
    ###

    def _allocate_contacts(self, rigid_capacity: int) -> None:
        self_capacity = self.self_contacts.capacity if self._load_self_contacts else 0
        rows = ParticleContactRows(rigid_capacity, self_capacity, self.num_worlds, self.device)
        self.contacts = DeformableContacts(
            self.model,
            self.kamino_model,
            self.kamino_data,
            rows,
            self.system.topology.newton_to_packed,
            self.system.topology.packed_solve_world,
            friction_mix_mode=self._friction_mix_mode,
            restitution_mix_mode=self._restitution_mix_mode,
            rigid_normal_cone_filtering=self._rigid_normal_cone_filtering,
            normal_cone_filtering_min_distance=self._normal_cone_filtering_min_distance,
        )
