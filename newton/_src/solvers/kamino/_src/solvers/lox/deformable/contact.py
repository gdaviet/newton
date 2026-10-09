# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Particle contacts of the LOX deformable bodies.

:class:`DeformableContacts` loads the soft contacts that Newton's collision pipeline
detects between the particle features of the deformable bodies and the collider shapes
into the :class:`..types.ParticleContactRows` of the LOX problem, at the beginning of
each time step. The projection of :mod:`..projection` then sweeps them together with
the rigid unilateral rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ....core.materials import MaterialMixMode
from ....core.types import vec6f
from .contact_filter import _compute_edge_normal_cones
from .contact_kernels import _finish_particle_contacts, _load_particle_contacts

if TYPE_CHECKING:
    from .......sim import Contacts, Model
    from ....core.data import DataKamino
    from ....core.model import ModelKamino
    from ..types import LOXProblemData, ParticleContactRows

###
# Module interface
###

__all__ = ["DeformableContacts"]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Interfaces
###


class DeformableContacts:
    """Load the soft contacts of the deformable bodies into particle contact rows.

    Args:
        model: Newton model of the deformable bodies.
        kamino_model: Kamino model of the rigid bodies.
        kamino_data: Kamino data of the rigid bodies, whose begin-step body poses and twists
            define the collider motion.
        rows: Particle contact rows of the LOX problem.
        newton_to_packed: Packed particle of each Newton particle.
        particle_world: World of each packed particle.
        friction_mix_mode: Mixing of the soft contact and shape friction coefficients.
        restitution_mix_mode: Mixing of the soft contact and shape restitution coefficients.
        rigid_normal_cone_filtering: Whether to keep only the full-surface rigid-soft contacts whose
            direction lies in the normal cone of their surface feature.
        normal_cone_filtering_min_distance: Separation below which the contacts bypass the normal-cone
            filter [m].
    """

    def __init__(
        self,
        model: Model,
        kamino_model: ModelKamino,
        kamino_data: DataKamino,
        rows: ParticleContactRows,
        newton_to_packed: wp.array[wp.int32],
        particle_world: wp.array[wp.int32],
        friction_mix_mode: MaterialMixMode = MaterialMixMode.AVERAGE,
        restitution_mix_mode: MaterialMixMode = MaterialMixMode.MIN,
        rigid_normal_cone_filtering: bool = False,
        normal_cone_filtering_min_distance: float = 1.0e-4,
    ):
        self.model = model
        self.kamino_model = kamino_model
        self.kamino_data = kamino_data
        self.device = wp.get_device(model.device)
        self._rows = rows
        self.newton_to_packed = newton_to_packed
        self.particle_world = particle_world
        self.friction_mix_mode = int(friction_mix_mode)
        self.restitution_mix_mode = int(restitution_mix_mode)
        self.rigid_normal_cone_filtering = rigid_normal_cone_filtering
        self.normal_cone_filtering_min_distance = normal_cone_filtering_min_distance
        self.soft_friction = float(model.soft_contact_mu)
        self.soft_restitution = float(model.soft_contact_restitution)
        if not np.isfinite(self.soft_friction) or self.soft_friction < 0.0:
            raise ValueError("The soft contact friction of the model must be finite and nonnegative.")
        if not np.isfinite(self.soft_restitution) or self.soft_restitution < 0.0:
            raise ValueError("The soft contact restitution of the model must be finite and nonnegative.")
        edge_count = model.edge_count
        self.edge_cone_axis = wp.zeros(edge_count, dtype=wp.vec3f, device=self.device)
        """Normal-cone axis of each surface edge."""
        self.edge_cone_cosine = wp.full(edge_count, -1.0, dtype=wp.float32, device=self.device)
        """Cosine of the normal-cone half angle of each surface edge, or ``-1`` for an unavailable cone."""
        self._empty_edge_indices = wp.empty((0, 4), dtype=wp.int32, device=self.device)
        self._empty_body_com = wp.empty(0, dtype=wp.vec3f, device=self.device)

    ###
    # Properties
    ###

    @property
    def data(self) -> ParticleContactRows:
        """Returns the particle contact rows."""
        return self._rows

    ###
    # Public API
    ###

    def load(
        self,
        contacts: Contacts | None,
        particle_q: wp.array[wp.vec3f],
        particle_qd: wp.array[wp.vec3f],
        particle_inverse_weight: wp.array[wp.float32],
        time_step: wp.array[wp.float32],
        stabilization_fraction: float,
        dead_zone: float,
    ) -> None:
        """Load the soft contacts of one time step into the rigid-soft rows.

        Args:
            contacts: Newton contacts with the soft contacts of the time step, or ``None``.
            particle_q: Begin-step particle positions [m].
            particle_qd: Begin-step particle velocities [m/s].
            particle_inverse_weight: Inverse weight of each packed particle; zero for prescribed particles.
            time_step: Time step of each world [s].
            stabilization_fraction: Fraction of the contact penetration corrected in one step.
            dead_zone: Symmetric contact distance dead zone of the stabilization [m].
        """
        rows = self._rows
        if rows.rigid_capacity == 0:
            return
        if contacts is None or contacts.soft_contact_max == 0:
            # No detected contacts leaves every rigid-soft row unused
            rows.status[: rows.rigid_capacity].zero_()
            return
        model = self.model
        filter_surface_contacts = bool(
            self.rigid_normal_cone_filtering
            and contacts._enable_rigid_soft_full_surface_contact
            and model.tri_count > 0
            and model.edge_count > 0
        )
        if filter_surface_contacts:
            wp.launch(
                _compute_edge_normal_cones,
                dim=model.edge_count,
                inputs=[particle_q, model.edge_indices],
                outputs=[self.edge_cone_axis, self.edge_cone_cosine],
                device=self.device,
            )
        capacity = min(rows.rigid_capacity, contacts.soft_contact_max)
        if capacity < rows.rigid_capacity:
            rows.status[capacity : rows.rigid_capacity].zero_()
        wp.launch(
            _load_particle_contacts,
            dim=capacity,
            inputs=[
                contacts.soft_contact_count,
                contacts.soft_contact_indices,
                contacts.soft_contact_barycentric,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                model.edge_indices if model.edge_indices is not None else self._empty_edge_indices,
                model.soft_mesh_adjacency_device,
                self.edge_cone_axis,
                self.edge_cone_cosine,
                filter_surface_contacts,
                self.normal_cone_filtering_min_distance,
                particle_q,
                particle_qd,
                model.particle_radius,
                model.particle_flags,
                self.newton_to_packed,
                self.particle_world,
                particle_inverse_weight,
                model.shape_body,
                model.shape_world,
                model.shape_margin,
                model.shape_material_mu,
                model.shape_material_restitution,
                self.kamino_model.bodies.wid,
                model.body_com if model.body_count > 0 else self._empty_body_com,
                self.kamino_data.bodies.q_i,
                self.kamino_data.bodies.u_i.view(dtype=vec6f),
                time_step,
                stabilization_fraction,
                dead_zone,
                self.soft_friction,
                self.soft_restitution,
                self.friction_mix_mode,
                self.restitution_mix_mode,
            ],
            outputs=[
                rows.world,
                rows.status,
                rows.particle,
                rows.coefficient,
                rows.body,
                rows.shape,
                rows.frame,
                rows.body_jacobian,
                rows.gap,
                rows.bias,
                rows.friction,
                rows.reaction,
            ],
            device=self.device,
        )

    def finish(
        self,
        problem_data: LOXProblemData,
        body_vector_index: wp.array[wp.int32],
        particle_inverse_weight: wp.array[wp.float32],
        particle_unilateral_count: wp.array[wp.int32],
    ) -> None:
        """Retire the rows between prescribed endpoints and count the active rows of each body and particle.

        The body counts add to :attr:`LOXProblemData.body_incidence`, which the rigid rows of the time step
        have already filled.

        Args:
            problem_data: Rows and body data of the LOX problem.
            body_vector_index: First entry of each body in the packed body system vectors, or ``-1``.
            particle_inverse_weight: Inverse weight of each packed particle; zero for prescribed particles.
            particle_unilateral_count: Number of rows that couple each dynamic particle with a significant
                coefficient, which enables its consensus weight.
        """
        rows = self._rows
        rows.world_count.zero_()
        problem_data.particle_incidence.zero_()
        problem_data.particle_weight_sum.zero_()
        particle_unilateral_count.zero_()
        if rows.capacity == 0:
            return
        wp.launch(
            _finish_particle_contacts,
            dim=rows.capacity,
            inputs=[rows.world, rows.particle, rows.coefficient, rows.body, body_vector_index, particle_inverse_weight],
            outputs=[
                rows.status,
                rows.world_count,
                problem_data.body_incidence,
                problem_data.particle_incidence,
                problem_data.particle_weight_sum,
                particle_unilateral_count,
            ],
            device=self.device,
        )
