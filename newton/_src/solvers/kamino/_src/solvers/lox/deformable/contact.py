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

from .......geometry.flags import ShapeFlags
from .......geometry.types import GeoType
from .......utils.mesh import MeshAdjacency
from ....core.materials import MaterialMixMode
from ....core.types import vec6f
from .contact_kernels import _finish_particle_contacts, _load_particle_contacts

if TYPE_CHECKING:
    from .......sim import Contacts, Model
    from ....core.data import DataKamino
    from ....core.model import ModelKamino
    from ..types import LOXProblemData, ParticleContactRows

###
# Module interface
###

__all__ = ["ColliderMeshTopology", "DeformableContacts"]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Interfaces
###


class ColliderMeshTopology:
    """Scaled triangles and adjacency of the collider meshes, which classify the rigid side of the contacts.

    The meshes of the particle-colliding mesh shapes are concatenated in their shape frames, with the
    shape scale applied; each shape addresses its triangles from an offset, and the other shapes have
    the offset ``-1``.

    Args:
        model: Newton model whose mesh shapes collide with the particles.
        include_shapes: Whether to gather the mesh shapes, which reads the shape arrays on the host;
            otherwise only the placeholder of the kernel arguments is built.
    """

    def __init__(self, model: Model, include_shapes: bool = True):
        device = wp.get_device(model.device)
        shape_count = int(model.shape_count)
        tri_offset = np.full(max(shape_count, 1), -1, dtype=np.int32)
        positions = []
        triangles = []
        vertex_count = 0
        triangle_count = 0
        if include_shapes and shape_count > 0:
            meshes = {int(mesh.id): mesh for mesh in getattr(model, "_mesh_keep_alive", [])}
            shape_type = model.shape_type.numpy()
            shape_flags = model.shape_flags.numpy()
            shape_source = model.shape_source_ptr.numpy()
            shape_scale = model.shape_scale.numpy()
            for shape in range(shape_count):
                if shape_type[shape] not in (int(GeoType.MESH), int(GeoType.CONVEX_MESH)):
                    continue
                if (int(shape_flags[shape]) & int(ShapeFlags.COLLIDE_PARTICLES)) == 0:
                    continue
                mesh = meshes.get(int(shape_source[shape]))
                if mesh is None:
                    continue
                points = mesh.points.numpy().astype(np.float32) * shape_scale[shape].astype(np.float32)
                indices = mesh.indices.numpy().astype(np.int32).reshape(-1, 3)
                tri_offset[shape] = triangle_count
                positions.append(points)
                triangles.append(indices + vertex_count)
                vertex_count += len(points)
                triangle_count += len(indices)
        if not triangles:
            # A placeholder triangle keeps the arrays valid; no shape addresses it
            positions.append(np.eye(3, dtype=np.float32))
            triangles.append(np.array([[0, 1, 2]], dtype=np.int32))
        positions = np.concatenate(positions)
        triangles = np.concatenate(triangles)
        adjacency = MeshAdjacency(tri_indices=triangles).init_vertex_adjacency(len(positions))
        self.position = wp.array(positions, dtype=wp.vec3f, device=device)
        """Vertex positions of the concatenated collider meshes in their scaled shape frames."""
        self.tri_indices = wp.array(triangles, dtype=wp.int32, device=device)
        """Triangles of the concatenated collider meshes."""
        self.edge_indices = wp.array(adjacency.edge_indices, dtype=wp.int32, device=device)
        """Edges of the concatenated collider meshes as ``(opposite_0, opposite_1, vertex_0, vertex_1)``."""
        self.adjacency = adjacency.to(device)
        """Vertex, edge, and triangle adjacency of the concatenated collider meshes."""
        self.shape_tri_offset = wp.array(tri_offset, dtype=wp.int32, device=device)
        """First triangle of each collider mesh shape, or ``-1`` for the other shapes."""
        self.face_side = wp.ones(len(triangles), dtype=wp.float32, device=device)
        """Side of each collider triangle; the collider meshes are wound outwards."""


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
            direction lies in the normal cones of the features of both primitives, with the face
            normal of a concave feature.
        normal_cone_filtering_min_distance: Separation below which the contacts bypass the normal-cone
            filter [m].
        collider_topology: Collider mesh topology of the rigid side of the filter, or ``None`` to classify
            the particle side only.
        face_side: Side of each particle surface triangle (zero for cloth, outward sign for a soft body
            boundary), or ``None`` for two-sided triangles.
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
        collider_topology: ColliderMeshTopology | None = None,
        face_side: wp.array[wp.float32] | None = None,
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
        if collider_topology is None:
            collider_topology = ColliderMeshTopology(model, include_shapes=rigid_normal_cone_filtering)
        self.collider_topology = collider_topology
        if face_side is None:
            face_side = wp.zeros(max(int(model.tri_count), 1), dtype=wp.float32, device=self.device)
        self.face_side = face_side
        self._empty_edge_indices = wp.empty((0, 4), dtype=wp.int32, device=self.device)
        self._empty_tri_indices = wp.empty((0, 3), dtype=wp.int32, device=self.device)
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
        # The cone filter needs every feature record of the surfaces, which full-surface contacts provide
        filter_surface_contacts = bool(
            self.rigid_normal_cone_filtering
            and contacts._enable_rigid_soft_full_surface_contact
            and model.tri_count > 0
            and model.edge_count > 0
        )
        topology = self.collider_topology
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
                model.tri_indices if model.tri_indices is not None else self._empty_tri_indices,
                model.edge_indices if model.edge_indices is not None else self._empty_edge_indices,
                model.soft_mesh_adjacency_device,
                self.face_side,
                topology.position,
                topology.tri_indices,
                topology.edge_indices,
                topology.adjacency,
                topology.face_side,
                topology.shape_tri_offset,
                model.shape_source_ptr,
                model.shape_transform,
                model.shape_scale,
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
