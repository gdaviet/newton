# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Containers of the LOX deformable particle systems.

Particles are packed by solve world, then by structural component, then by Newton
index. Every per-particle array below uses this packed order unless stated otherwise.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import warp as wp
import warp.sparse as wps

from ....core.types import to_warp_int32_array
from .energy import mat99f

###
# Module interface
###

__all__ = [
    "PATCH_BLOCK_COUNT",
    "PATCH_PARTICLE_COUNT",
    "RECYCLED_CR_DIRECTION_COUNT",
    "DeformableElementIncidence",
    "DeformableMembraneProximalData",
    "DeformablePreconditionerStatus",
    "DeformableRecyclingData",
    "DeformableSystemData",
    "DeformableTetrahedronProximalData",
    "DeformableTopology",
    "DeformableWeightStatus",
    "vec8i",
]


###
# Constants
###


RECYCLED_CR_DIRECTION_COUNT = 4
"""Number of recent candidate-solve corrections retained by the recycled conjugate-residual solve."""


_RECYCLED_CR_REDUCTION_COUNT = 3 + 2 * RECYCLED_CR_DIRECTION_COUNT
"""Number of per-component reductions of the recycled correction basis."""


_RECYCLED_CR_PROJECTION_REDUCTION_COUNT = 2
"""Number of per-component reductions of the recycled projection check."""


PATCH_PARTICLE_COUNT = 8
"""Largest number of particles of the patch of a patch consensus weight: the particle and its neighbors."""


PATCH_BLOCK_COUNT = PATCH_PARTICLE_COUNT * (PATCH_PARTICLE_COUNT + 1) // 2
"""Number of ``3 x 3`` blocks of the lower block triangle of a patch matrix."""


_COMPONENT_CHUNK_PARTICLE_COUNT = 1024
"""Largest number of packed particles of a component chunk, the unit of the component mass sum."""


###
# Types
###


class vec8i(wp.types.vector(length=PATCH_PARTICLE_COUNT, dtype=wp.int32)):
    """Packed particles of the slots of a consensus weight patch."""


###
# Enumerations
###


class DeformableWeightStatus(IntEnum):
    """Status of the consensus weight of a particle."""

    INVALID = 0
    """The smooth diagonal block was non-finite; the weight is zero."""

    VALID = 1
    """The weight was finite and required no regularization, or the particle is prescribed."""

    REGULARIZED = 2
    """A patch matrix was not positive definite, and its bound took the particle mass for the free patch, or the
    mean diagonal of the particle for the pinned patch."""


class DeformablePreconditionerStatus(IntEnum):
    """Status of the factorization of a world in a deformable preconditioner or direct solver."""

    VALID = 0
    """The factorization completed without pivot modification."""

    REGULARIZED = 1
    """At least one diagonal pivot was regularized."""

    FAILED = 2
    """The factorization encountered non-finite matrix data."""


###
# Containers
###


class DeformableTopology:
    """Immutable packed topology of the deformable particles and elements.

    ``packed_source_world`` preserves Newton ownership, including the global world
    ``-1``. ``packed_solve_world`` stores the nonnegative LOX partition used for
    batching, status, time steps, and rigid/deformable coupling arrays.

    Args:
        newton_to_packed: Packed index of each Newton particle.
        packed_to_newton: Newton index of each packed particle.
        packed_source_world: Newton world of each packed particle.
        packed_solve_world: LOX solve world of each packed particle.
        packed_component: Structural component of each packed particle.
        component_dof_offsets: First scalar degree of freedom of each component, followed by the total.
        triangle_indices: Packed particles of each triangle.
        bending_indices: Packed particles of each valid bending edge.
        source_edge_indices: Newton edge of each valid bending edge.
        bending_rest_scale: Discrete-shell scale ``|e|^2 / (A_0 + A_1)`` of each valid bending edge.
        tetrahedron_indices: Packed particles of each tetrahedron.
        device: Device of the arrays.
    """

    def __init__(
        self,
        newton_to_packed: np.ndarray,
        packed_to_newton: np.ndarray,
        packed_source_world: np.ndarray,
        packed_solve_world: np.ndarray,
        packed_component: np.ndarray,
        component_dof_offsets: np.ndarray,
        triangle_indices: np.ndarray,
        bending_indices: np.ndarray,
        source_edge_indices: np.ndarray,
        bending_rest_scale: np.ndarray,
        tetrahedron_indices: np.ndarray,
        device: wp.DeviceLike,
    ):
        self.component_count = len(component_dof_offsets) - 1
        """Number of structural components."""
        self.newton_to_packed = to_warp_int32_array(newton_to_packed, device)
        """Packed index of each Newton particle."""
        self.packed_to_newton = to_warp_int32_array(packed_to_newton, device)
        """Newton index of each packed particle."""
        self.packed_source_world = to_warp_int32_array(packed_source_world, device)
        """Newton world of each packed particle, or ``-1`` for a global particle."""
        self.packed_solve_world = to_warp_int32_array(packed_solve_world, device)
        """Nonnegative LOX solve world of each packed particle."""
        self.packed_component = to_warp_int32_array(packed_component, device)
        """Structural component of each packed particle: the particles connected by elements."""
        self.component_dof_offsets = to_warp_int32_array(component_dof_offsets, device)
        """First scalar degree of freedom of each component, followed by the total; the batches of the
        conjugate-residual solve."""
        self.triangle_indices = to_warp_int32_array(triangle_indices, device)
        """Packed particles of each triangle, shape ``[triangle_count, 3]``."""
        self.bending_indices = to_warp_int32_array(bending_indices, device)
        """Packed particles of each valid bending edge, shape ``[bending_count, 4]``."""
        self.source_edge_indices = to_warp_int32_array(source_edge_indices, device)
        """Newton edge of each valid bending edge."""
        self.bending_rest_scale = wp.array(np.asarray(bending_rest_scale, dtype=np.float32), dtype=float, device=device)
        """Discrete-shell scale ``|e|^2 / (A_0 + A_1)`` of each valid bending edge, from its rest length and the
        rest areas of its two triangles [dimensionless]."""
        self.tetrahedron_indices = to_warp_int32_array(tetrahedron_indices, device)
        """Packed particles of each tetrahedron, shape ``[tetrahedron_count, 4]``."""


class DeformableElementIncidence:
    """Elements of each packed particle, and the patches that assemble the patch consensus weights.

    The elements are the triangles, then the valid bending edges, then the tetrahedra. The
    pair blocks of each element are stored row-major in the assembly triplets from its first
    triplet. The patch arrays are rebuilt with the weights of each linearization.

    Args:
        particle_count: Number of particles.
        triangle_triplet_offset: First assembly triplet of the triangles.
        bending_triplet_offset: First assembly triplet of the bending edges.
        tetrahedron_triplet_offset: First assembly triplet of the tetrahedra.
        triangle_indices: Packed particles of each triangle.
        bending_indices: Packed particles of each valid bending edge.
        tetrahedron_indices: Packed particles of each tetrahedron.
        component_particle_offsets: First packed particle of each structural component, followed by the total.
        device: Device of the arrays.
    """

    def __init__(
        self,
        particle_count: int,
        triangle_triplet_offset: int,
        bending_triplet_offset: int,
        tetrahedron_triplet_offset: int,
        triangle_indices: np.ndarray,
        bending_indices: np.ndarray,
        tetrahedron_indices: np.ndarray,
        component_particle_offsets: np.ndarray,
        device: wp.DeviceLike,
    ):
        triangle_count = len(triangle_indices)
        bending_count = len(bending_indices)
        tetrahedron_count = len(tetrahedron_indices)
        triangle_vertices = np.full((triangle_count, 4), -1, dtype=np.int32)
        triangle_vertices[:, :3] = triangle_indices
        vertices = np.concatenate(
            (triangle_vertices, np.reshape(bending_indices, (-1, 4)), np.reshape(tetrahedron_indices, (-1, 4)))
        ).astype(np.int32)
        size = np.concatenate(
            (np.full(triangle_count, 3), np.full(bending_count, 4), np.full(tetrahedron_count, 4))
        ).astype(np.int32)
        triplet = np.concatenate(
            (
                triangle_triplet_offset + 9 * np.arange(triangle_count),
                bending_triplet_offset + 16 * np.arange(bending_count),
                tetrahedron_triplet_offset + 16 * np.arange(tetrahedron_count),
            )
        ).astype(np.int32)
        corner_vertex = vertices.reshape(-1)
        corner_element = np.repeat(np.arange(len(vertices), dtype=np.int32), 4)
        valid = corner_vertex >= 0
        corner_vertex = corner_vertex[valid]
        order = np.argsort(corner_vertex, kind="stable")
        offset = np.zeros(particle_count + 1, dtype=np.int32)
        offset[1:] = np.cumsum(np.bincount(corner_vertex, minlength=particle_count))

        self.element_vertices = wp.array(
            vertices if len(vertices) else np.full((1, 4), -1), dtype=wp.vec4i, device=device
        )
        """Packed particles of each element, ``-1`` past the three vertices of a triangle."""
        self.element_size = to_warp_int32_array(size if len(size) else np.zeros(1), device)
        """Number of particles of each element."""
        self.element_triplet = to_warp_int32_array(triplet if len(triplet) else np.zeros(1), device)
        """First assembly triplet of the pair blocks of each element."""
        self.particle_element_offset = to_warp_int32_array(offset, device)
        """First entry of each particle in :attr:`particle_element`, followed by the total."""
        particle_element = corner_element[valid][order]
        self.particle_element = to_warp_int32_array(particle_element if len(particle_element) else np.zeros(1), device)
        """Elements of each particle, grouped by particle."""
        self.particle_dynamic = wp.zeros(max(particle_count, 1), dtype=wp.int32, device=device)
        """Whether each particle is dynamic, refreshed with the patch weights of each linearization."""
        component_count = len(component_particle_offsets) - 1
        component_counts = np.diff(component_particle_offsets)
        component_chunk_counts = np.maximum(-(-component_counts // _COMPONENT_CHUNK_PARTICLE_COUNT), 1)
        component_chunk_offsets = np.concatenate(([0], np.cumsum(component_chunk_counts))).astype(np.int32)
        chunk_component = np.repeat(np.arange(component_count), component_chunk_counts)
        chunk_rank = np.arange(int(component_chunk_offsets[-1])) - component_chunk_offsets[chunk_component]
        chunk_first = component_particle_offsets[chunk_component] + _COMPONENT_CHUNK_PARTICLE_COUNT * chunk_rank
        self.chunk_count = int(component_chunk_offsets[-1])
        """Number of component chunks, the contiguous ranges of at most ``_COMPONENT_CHUNK_PARTICLE_COUNT``
        packed particles inside one structural component."""
        self.chunk_particle_offsets = to_warp_int32_array(
            np.concatenate((chunk_first, component_particle_offsets[-1:])).astype(np.int32), device
        )
        """First packed particle of each component chunk, followed by the particle count."""
        self.component_chunk_offsets = to_warp_int32_array(component_chunk_offsets, device)
        """First chunk of each structural component, followed by the chunk count."""
        self.chunk_mass = wp.zeros(max(self.chunk_count, 1), dtype=wp.float32, device=device)
        """Dynamic particle mass of each component chunk [kg]."""
        self.chunk_fixed_count = wp.zeros(max(self.chunk_count, 1), dtype=wp.int32, device=device)
        """Number of fixed particles of each component chunk."""
        self.component_mass = wp.zeros(max(component_count, 1), dtype=wp.float32, device=device)
        """Dynamic particle mass of each structural component, refreshed with the patch weights [kg]."""
        self.component_fixed_count = wp.zeros(max(component_count, 1), dtype=wp.int32, device=device)
        """Number of fixed particles of each structural component, refreshed with the patch weights."""
        self.patch_particles = wp.zeros(max(particle_count, 1), dtype=vec8i, device=device)
        """Patch of each particle, with the particle in the last slot."""
        self.patch_count = wp.zeros(max(particle_count, 1), dtype=wp.int32, device=device)
        """Number of particles of each patch, zero for a particle without a patch weight."""
        self.patch_blocks = wp.zeros(
            (2 * 9 * PATCH_BLOCK_COUNT, max(particle_count, 1)), dtype=wp.float32, device=device
        )
        """Lower block triangles of the free, then the pinned patch matrix of each particle, one column per
        particle: rows ``9 b`` to ``9 b + 8`` hold the row-major entries of block ``b = row (row + 1) / 2 + column``
        of the free matrix."""
        self.patch_smallest = wp.zeros(2 * max(particle_count, 1), dtype=wp.float32, device=device)
        """Smallest eigenvalue of the Schur complement of each particle in its free, then its pinned patch
        matrix, ``-1`` for a matrix that is not positive definite."""


class DeformableMembraneProximalData:
    """Fixed-metric proximal state of the membrane triangles.

    Each triangle holds its two deformation columns, which are the local coordinates of
    the proximal update.

    Args:
        triangle_count: Number of triangles.
        device: Device of the arrays.
    """

    def __init__(self, triangle_count: int, device: wp.DeviceLike):
        shape = (triangle_count, 2)
        self.frozen_coordinate = wp.zeros(shape, dtype=wp.vec3, device=device)
        """Deformation columns of each triangle at the linearization of the time step."""
        self.frozen_gradient = wp.zeros(shape, dtype=wp.vec3, device=device)
        """Energy gradient of each triangle at the linearization of the time step [N]."""
        self.frozen_area_gradient = wp.zeros(shape, dtype=wp.vec3, device=device)
        """Area-ratio gradient of each triangle at the linearization; it defines the proximal metric."""
        self.proximal_coordinate = wp.zeros(shape, dtype=wp.vec3, device=device)
        """Local proximal deformation columns of each triangle."""
        self.multiplier = wp.zeros(shape, dtype=wp.vec3, device=device)
        """Local force multiplier of each triangle [N]."""


class DeformableTetrahedronProximalData:
    """Fixed-majorizer proximal state of the tetrahedra.

    Each tetrahedron holds its deformation gradient, which is the local coordinate of the
    proximal update. The proximal metric is the majorizing metric of the assembly,
    :attr:`DeformableSystemData.tetrahedron_metric`.

    Args:
        tetrahedron_count: Number of tetrahedra.
        device: Device of the arrays.
    """

    def __init__(self, tetrahedron_count: int, device: wp.DeviceLike):
        self.frozen_coordinate = wp.zeros(tetrahedron_count, dtype=wp.mat33, device=device)
        """Deformation gradient of each tetrahedron at the linearization of the time step."""
        self.frozen_gradient = wp.zeros(tetrahedron_count, dtype=wp.mat33, device=device)
        """Energy gradient of each tetrahedron at the linearization of the time step [N]."""
        self.frozen_factor = wp.zeros(tetrahedron_count, dtype=mat99f, device=device)
        """Lower Cholesky factor of the frozen Gauss-Newton base matrix of each tetrahedron."""
        self.proximal_coordinate = wp.zeros(tetrahedron_count, dtype=wp.mat33, device=device)
        """Local proximal deformation gradient of each tetrahedron."""
        self.multiplier = wp.zeros(tetrahedron_count, dtype=wp.mat33, device=device)
        """Local force multiplier of each tetrahedron [N]."""


class DeformableRecyclingData:
    """Correction subspace of the recycled conjugate-residual candidate solve.

    The solve retains the last :data:`RECYCLED_CR_DIRECTION_COUNT` candidate-solve
    corrections and their matrix images, and projects each new candidate solve onto
    them before the conjugate-residual iterations.

    Args:
        particle_count: Number of particles.
        component_count: Number of structural components.
        device: Device of the arrays.
    """

    def __init__(self, particle_count: int, component_count: int, device: wp.DeviceLike):
        direction_shape = (RECYCLED_CR_DIRECTION_COUNT, component_count)
        self.previous_velocity = wp.zeros(particle_count, dtype=wp.vec3, device=device)
        """Candidate velocity of each particle after the previous projection [m/s]."""
        self.previous_matrix_velocity = wp.zeros(particle_count, dtype=wp.vec3, device=device)
        """Matrix image of the previous candidate velocity of each particle."""
        self.basis = wp.zeros((RECYCLED_CR_DIRECTION_COUNT, particle_count), dtype=wp.vec3, device=device)
        """Retained correction directions."""
        self.matrix_basis = wp.zeros((RECYCLED_CR_DIRECTION_COUNT, particle_count), dtype=wp.vec3, device=device)
        """Matrix images of the retained correction directions."""
        self.basis_denominator = wp.zeros(direction_shape, dtype=wp.float32, device=device)
        """Energy of each retained direction in each component, or zero for an unused direction."""
        self.basis_valid = wp.zeros(RECYCLED_CR_DIRECTION_COUNT, dtype=wp.int32, device=device)
        """Whether each retained direction slot holds a direction."""
        self.started = wp.zeros(1, dtype=wp.int32, device=device)
        """Whether a previous candidate solve of the time step provides a correction direction."""
        self.replacement_cursor = wp.zeros(1, dtype=wp.int32, device=device)
        """Direction slot replaced by the next correction."""
        self.reductions = wp.zeros((_RECYCLED_CR_REDUCTION_COUNT, component_count), dtype=wp.float32, device=device)
        """Per-component inner products of the new correction with itself, the residual, and the retained basis."""
        self.orthogonalization = wp.zeros(direction_shape, dtype=wp.float32, device=device)
        """Coefficients orthogonalizing the new correction against the retained directions."""
        self.projection = wp.zeros(direction_shape, dtype=wp.float32, device=device)
        """Coefficients of the projected correction along each direction."""
        self.projection_reductions = wp.zeros(
            (_RECYCLED_CR_PROJECTION_REDUCTION_COUNT, component_count), dtype=wp.float32, device=device
        )
        """Per-component residual and energy inner products of the projected correction."""
        self.projection_accepted = wp.ones(component_count, dtype=wp.int32, device=device)
        """Whether the projected correction of each component decreased the linear quadratic."""


class DeformableSystemData:
    """Arrays of the frozen smooth deformable system and its candidate solves.

    The smooth matrix ``A`` holds one ``3 x 3`` block per particle pair of each element.
    The system matrix adds the selected consensus weights ``alpha W`` to its diagonal blocks,
    and the preconditioner matrix the full consensus weights ``alpha W``, with ``alpha`` the
    consensus fraction.

    Args:
        particle_count: Number of particles.
        world_count: Number of solve worlds.
        component_count: Number of structural components.
        triangle_count: Number of triangles.
        tetrahedron_count: Number of tetrahedra.
        triplet_rows: Packed block row of each assembly triplet.
        triplet_columns: Packed block column of each assembly triplet.
        smooth_matrix: Block sparse smooth matrix, with its fixed topology.
        system_matrix: Block sparse system matrix, with the topology of the smooth matrix.
        preconditioner_matrix: Block sparse preconditioner matrix, with the topology of the smooth matrix.
        diagonal_slots: Value slot of the diagonal block of each particle in the block sparse matrices.
        packed_iterative: Whether each particle belongs to a component solved by conjugate residuals.
        membrane_proximal: Whether to allocate the membrane proximal state.
        tetrahedron_proximal: Whether to allocate the tetrahedron proximal state.
        recycling: Whether to allocate the correction subspace of the recycled conjugate-residual solve.
        device: Device of the arrays.
    """

    def __init__(
        self,
        particle_count: int,
        world_count: int,
        component_count: int,
        triangle_count: int,
        tetrahedron_count: int,
        triplet_rows: np.ndarray,
        triplet_columns: np.ndarray,
        smooth_matrix: wps.BsrMatrix,
        system_matrix: wps.BsrMatrix,
        preconditioner_matrix: wps.BsrMatrix,
        diagonal_slots: np.ndarray,
        packed_iterative: np.ndarray,
        membrane_proximal: bool,
        tetrahedron_proximal: bool,
        recycling: bool,
        device: wp.DeviceLike,
    ):
        self.triplet_rows = to_warp_int32_array(triplet_rows, device)
        """Packed block row of each assembly triplet."""
        self.triplet_columns = to_warp_int32_array(triplet_columns, device)
        """Packed block column of each assembly triplet."""
        self.triplet_values = wp.zeros(len(triplet_rows), dtype=wp.mat33, device=device)
        """Block of each assembly triplet: the particle masses, then the element pair blocks."""
        self.tetrahedron_metric = wp.zeros(tetrahedron_count, dtype=mat99f, device=device)
        """Majorizing metric of each tetrahedron at the linearization of the time step."""
        self.smooth_matrix = smooth_matrix
        """Frozen smooth matrix ``A``."""
        self.system_matrix = system_matrix
        """Candidate system matrix ``A + W`` with the selected consensus weights."""
        self.preconditioner_matrix = preconditioner_matrix
        """Preconditioner matrix ``A + W`` with the full consensus weights."""
        self.diagonal_slots = to_warp_int32_array(diagonal_slots, device)
        """Value slot of the diagonal block of each particle in the block sparse matrices."""
        self.packed_iterative = to_warp_int32_array(packed_iterative, device)
        """Whether each particle belongs to a component solved by conjugate residuals; other particles
        belong to directly solved components."""

        self.position_start = wp.zeros(particle_count, dtype=wp.vec3, device=device)
        """Position of each particle at the beginning of the time step [m]."""
        self.velocity_start = wp.zeros(particle_count, dtype=wp.vec3, device=device)
        """Velocity of each particle at the beginning of the time step [m/s]."""
        self.external_force = wp.zeros(particle_count, dtype=wp.vec3, device=device)
        """External and gravity force of each dynamic particle [N]."""
        self.position_linearized = wp.zeros(particle_count, dtype=wp.vec3, device=device)
        """End-of-step position of each particle at the linearization velocity [m]."""
        self.velocity_linearized = wp.zeros(particle_count, dtype=wp.vec3, device=device)
        """Linearization velocity of each particle [m/s]."""
        self.smooth_force = wp.zeros(particle_count, dtype=wp.vec3, device=device)
        """External, elastic, damping, and aerodynamic force of each particle at the linearization [N]."""
        self.matrix_velocity = wp.zeros(particle_count, dtype=wp.vec3, device=device)
        """Product of the smooth matrix with the linearization velocity [N s]."""
        self.smooth_rhs = wp.zeros(particle_count, dtype=wp.vec3, device=device)
        """Frozen smooth right-hand side ``f`` [N s]."""

        self.full_weight = wp.zeros(particle_count, dtype=wp.float32, device=device)
        """Consensus weight of each dynamic particle [kg]."""
        self.full_inverse_weight = wp.zeros(particle_count, dtype=wp.float32, device=device)
        """Inverse consensus weight of each dynamic particle [1/kg]."""
        self.weight = wp.zeros(particle_count, dtype=wp.float32, device=device)
        """Consensus weight of each particle in the selected support, or zero [kg]."""
        self.inverse_weight = wp.zeros(particle_count, dtype=wp.float32, device=device)
        """Inverse consensus weight of each particle in the selected support, or zero [1/kg]."""
        self.weight_status = wp.zeros(particle_count, dtype=wp.int32, device=device)
        """:class:`DeformableWeightStatus` of each particle."""
        self.unilateral_incidence = wp.zeros(particle_count, dtype=wp.int32, device=device)
        """Number of unilateral rows of each particle, frozen for the time step."""
        self.consensus_enabled = wp.ones(particle_count, dtype=wp.int32, device=device)
        """Whether each particle has a nonzero consensus weight."""
        self.world_active = wp.ones(world_count, dtype=wp.bool, device=device)
        """Whether each world is still iterating; inactive worlds keep their candidate velocity."""

        self.candidate_rhs = wp.zeros(particle_count, dtype=wp.vec3, device=device)
        """Right-hand side of the candidate solve of the current splitting iteration [N s]."""
        self.smooth_velocity = wp.zeros(particle_count, dtype=wp.vec3, device=device)
        """Candidate velocity of each particle from the smooth solve [m/s]."""
        self.nonlinear_rhs = wp.zeros(particle_count, dtype=wp.vec3, device=device)
        """Right-hand-side correction of the element proximal updates [N s]."""
        self.proximal_position_residual = wp.zeros(world_count, dtype=wp.float32, device=device)
        """Largest scaled position residual of the element proximal updates of each world [m]."""
        self.proximal_velocity_residual = wp.zeros(world_count, dtype=wp.float32, device=device)
        """Largest scaled velocity residual of the element proximal updates of each world [m/s]."""
        self.proximal_failed = wp.zeros(world_count, dtype=wp.bool, device=device)
        """Whether the assembly or a candidate deformation of the element proximal updates of each world failed."""
        self.assembly_failed = wp.zeros(world_count, dtype=wp.bool, device=device)
        """Whether the assembly of each world failed: a non-finite element contribution."""
        self.candidate_linear_residual = wp.zeros(world_count, dtype=wp.float32, device=device)
        """Largest weight-scaled residual of the conjugate-residual candidate solve of each world [m/s]."""

        self.membrane_proximal = DeformableMembraneProximalData(triangle_count, device) if membrane_proximal else None
        """Proximal state of the membrane triangles, with membrane proximal updates."""
        self.tetrahedron_proximal = (
            DeformableTetrahedronProximalData(tetrahedron_count, device) if tetrahedron_proximal else None
        )
        """Proximal state of the tetrahedra, with tetrahedron proximal updates."""
        self.recycling = DeformableRecyclingData(particle_count, component_count, device) if recycling else None
        """Correction subspace of the recycled conjugate-residual solve, with recycling."""
