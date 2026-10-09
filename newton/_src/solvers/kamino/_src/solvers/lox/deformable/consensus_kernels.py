# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernels of the patch consensus weights of the LOX deformable particles.

Each dynamic particle ``i`` takes one scalar consensus weight from its patch: the particle and
the dynamic particles of its stiffest elements, at most :data:`~.types.PATCH_PARTICLE_COUNT`
particles. Two patch matrices bound the inverse Delassus ``[(A^-1)_ii]^-1`` of the particle. In
the free patch, the patch particles keep their masses and the pair blocks of the elements whose
dynamic particles all lie in the patch; with the element tangents positive semi-definite, the
Schur complement of ``i`` satisfies ``m_i I <= S_free <= [(A^-1)_ii]^-1``. In the pinned patch,
they keep their full rows of ``A``, the particles outside the patch held fixed, so ``S_pinned``
bounds it from above. The weight is the geometric mean of the smallest eigenvalues of both
complements, which stays within the bounds. An optional cap at ``beta`` times the free bound
bounds it like the rigid body weights.

Four launches compute the weights, spreading the element loads of each patch over one thread per
block row: :func:`_select_consensus_patch` grows the patch of each particle,
:func:`_assemble_consensus_patch_rows` assembles each block row of both patch matrices,
:func:`_eliminate_consensus_patch` reduces each matrix to the Schur complement of its particle,
and :func:`_finish_consensus_weight` combines the smallest eigenvalues of both complements.
"""

import warp as wp

from .system_kernels import _matrix_is_finite, is_dynamic_particle
from .types import PATCH_BLOCK_COUNT, PATCH_PARTICLE_COUNT, DeformableWeightStatus, vec8i

###
# Module interface
###

__all__ = [
    "_assemble_consensus_patch_rows",
    "_eliminate_consensus_patch",
    "_finish_consensus_weight",
    "_mark_dynamic_particles",
    "_select_consensus_patch",
]


###
# Constants
###


_PATCH_CANDIDATE_COUNT = 2 * PATCH_PARTICLE_COUNT
"""Number of the stiffest elements of a particle that may grow its patch."""


_patch_candidates = wp.types.vector(length=_PATCH_CANDIDATE_COUNT, dtype=wp.int32)
_patch_candidate_stiffness = wp.types.vector(length=_PATCH_CANDIDATE_COUNT, dtype=wp.float32)
_patch_row = wp.types.vector(length=9 * PATCH_PARTICLE_COUNT, dtype=wp.float32)


###
# Module configs
###

# The fixed-size loops over the patch slots and the candidates unroll, so the per-thread patch
# arrays take constant indices and stay in registers.
wp.set_module_options({"enable_backward": False, "max_unroll": _PATCH_CANDIDATE_COUNT})


###
# Functions
###


@wp.func
def _patch_slot(patch: vec8i, count: wp.int32, particle: wp.int32) -> wp.int32:
    """Return the slot of a particle in a patch, or ``-1`` outside of it."""
    slot = wp.int32(-1)
    for index in range(PATCH_PARTICLE_COUNT):
        if index < count and patch[index] == particle:
            slot = index
    return slot


@wp.func
def _patch_particle(patch: vec8i, slot: wp.int32) -> wp.int32:
    """Return the particle of a patch slot."""
    particle = wp.int32(-1)
    for index in range(PATCH_PARTICLE_COUNT):
        if index == slot:
            particle = patch[index]
    return particle


@wp.func
def _set_patch_particle(patch: vec8i, slot: wp.int32, particle: wp.int32) -> vec8i:
    """Return the patch with a particle in one slot."""
    result = patch
    for index in range(PATCH_PARTICLE_COUNT):
        if index == slot:
            result[index] = particle
    return result


@wp.func
def _element_stiffness(
    element: wp.int32,
    particle: wp.int32,
    triplet_values: wp.array[wp.mat33],
    element_vertices: wp.array[wp.vec4i],
    element_triplet: wp.array[wp.int32],
    element_size: wp.array[wp.int32],
) -> wp.float32:
    """Return the squared norm of the diagonal pair block of an element on one of its particles."""
    vertices = element_vertices[element]
    size = element_size[element]
    corner = wp.int32(-1)
    for index in range(4):
        if index < size and vertices[index] == particle and corner < 0:
            corner = index
    if corner < 0:
        return 0.0
    block = triplet_values[element_triplet[element] + corner * size + corner]
    return wp.ddot(block, block)


@wp.func
def _insert_candidate(
    candidates: _patch_candidates,
    candidate_stiffness: _patch_candidate_stiffness,
    element: wp.int32,
    stiffness: wp.float32,
) -> tuple[_patch_candidates, _patch_candidate_stiffness]:
    """Insert a positive-stiffness element into candidates sorted by decreasing stiffness.

    The element follows the candidates of equal stiffness, and the last candidate of a full list
    drops out. Empty entries hold the element ``-1`` with zero stiffness.
    """
    inserted = candidates
    inserted_stiffness = candidate_stiffness
    if candidate_stiffness[0] < stiffness:
        inserted[0] = element
        inserted_stiffness[0] = stiffness
    for position in range(1, _PATCH_CANDIDATE_COUNT):
        if candidate_stiffness[position] < stiffness:
            if candidate_stiffness[position - 1] < stiffness:
                inserted[position] = candidates[position - 1]
                inserted_stiffness[position] = candidate_stiffness[position - 1]
            else:
                inserted[position] = element
                inserted_stiffness[position] = stiffness
    return inserted, inserted_stiffness


@wp.func
def _block_index(row: wp.int32, column: wp.int32) -> wp.int32:
    """Return the index of the block ``(row, column)``, ``row >= column``, of a packed lower block triangle."""
    return row * (row + 1) // 2 + column


@wp.func
def _mirror_lower(block: wp.mat33) -> wp.mat33:
    """Return the symmetric matrix of the lower triangle of a block."""
    return wp.mat33(
        block[0, 0],
        block[1, 0],
        block[2, 0],
        block[1, 0],
        block[1, 1],
        block[2, 1],
        block[2, 0],
        block[2, 1],
        block[2, 2],
    )


@wp.func
def _add_row_block(row: _patch_row, column_slot: wp.int32, block: wp.mat33) -> _patch_row:
    """Return a patch block row with a block added at one column slot."""
    result = row
    for slot in range(PATCH_PARTICLE_COUNT):
        if slot == column_slot:
            for block_row in range(3):
                for block_column in range(3):
                    result[9 * slot + 3 * block_row + block_column] += block[block_row, block_column]
    return result


@wp.func
def _cholesky_inverse_transpose(block: wp.mat33) -> tuple[wp.mat33, wp.bool]:
    """Return ``L^-T`` of the Cholesky factor ``L`` of the lower triangle of a block, and whether the block is
    positive definite."""
    pivot = block[0, 0]
    if not (pivot > 0.0) or not wp.isfinite(pivot):
        return wp.mat33(0.0), False
    l00 = wp.sqrt(pivot)
    l10 = block[1, 0] / l00
    l20 = block[2, 0] / l00
    pivot = block[1, 1] - l10 * l10
    if not (pivot > 0.0) or not wp.isfinite(pivot):
        return wp.mat33(0.0), False
    l11 = wp.sqrt(pivot)
    l21 = (block[2, 1] - l20 * l10) / l11
    pivot = block[2, 2] - l20 * l20 - l21 * l21
    if not (pivot > 0.0) or not wp.isfinite(pivot):
        return wp.mat33(0.0), False
    l22 = wp.sqrt(pivot)
    inverse00 = 1.0 / l00
    inverse11 = 1.0 / l11
    inverse22 = 1.0 / l22
    inverse10 = -l10 * inverse00 * inverse11
    inverse21 = -l21 * inverse11 * inverse22
    inverse20 = -(l20 * inverse00 + l21 * inverse10) * inverse22
    return wp.mat33(inverse00, inverse10, inverse20, 0.0, inverse11, inverse21, 0.0, 0.0, inverse22), True


@wp.func
def _read_patch_block(patch_blocks: wp.array2d[wp.float32], first: wp.int32, particle: wp.int32) -> wp.mat33:
    """Return the block of a particle whose nine entries start at row ``first`` of the patch blocks."""
    block = wp.mat33(0.0)
    for row in range(3):
        for column in range(3):
            block[row, column] = patch_blocks[first + 3 * row + column, particle]
    return block


@wp.func
def _write_patch_block(patch_blocks: wp.array2d[wp.float32], first: wp.int32, particle: wp.int32, block: wp.mat33):
    """Store the block of a particle whose nine entries start at row ``first`` of the patch blocks."""
    for row in range(3):
        for column in range(3):
            patch_blocks[first + 3 * row + column, particle] = block[row, column]


###
# Kernels
###


@wp.kernel
def _mark_dynamic_particles(
    # Inputs:
    packed_to_newton: wp.array[wp.int32],
    model_particle_mass: wp.array[float],
    model_particle_flags: wp.array[wp.int32],
    # Outputs:
    particle_dynamic: wp.array[wp.int32],
):
    """Mark the dynamic packed particles, see :func:`.system_kernels.is_dynamic_particle`."""
    pid = wp.tid()
    particle_dynamic[pid] = wp.where(
        is_dynamic_particle(pid, packed_to_newton, model_particle_mass, model_particle_flags), 1, 0
    )


@wp.kernel
def _select_consensus_patch(
    # Inputs:
    smooth_values: wp.array[wp.mat33],
    diagonal_slots: wp.array[wp.int32],
    triplet_values: wp.array[wp.mat33],
    particle_element_offset: wp.array[wp.int32],
    particle_element: wp.array[wp.int32],
    element_vertices: wp.array[wp.vec4i],
    element_triplet: wp.array[wp.int32],
    element_size: wp.array[wp.int32],
    particle_dynamic: wp.array[wp.int32],
    # Outputs:
    weight: wp.array[float],
    inverse_weight: wp.array[float],
    weight_status: wp.array[wp.int32],
    patch_particles: wp.array[vec8i],
    patch_count: wp.array[wp.int32],
):
    """Grow the patch of each dynamic particle, and set the weights of the other particles.

    The patch grows from the particle ``i`` with the dynamic particles of its stiffest elements, by
    decreasing stiffness of the element on ``i``, while they fit in ``PATCH_PARTICLE_COUNT``
    particles. The particle takes the last slot, whose elimination the other slots precede. A
    prescribed particle takes a zero weight, a particle with a non-finite diagonal block an invalid
    zero weight, and both an empty patch.
    """
    pid = wp.tid()
    if particle_dynamic[pid] == 0:
        weight[pid] = 0.0
        inverse_weight[pid] = 0.0
        weight_status[pid] = DeformableWeightStatus.VALID
        patch_count[pid] = 0
        return
    if not _matrix_is_finite(smooth_values[diagonal_slots[pid]]):
        weight[pid] = 0.0
        inverse_weight[pid] = 0.0
        weight_status[pid] = DeformableWeightStatus.INVALID
        patch_count[pid] = 0
        return

    # Rank the stiffest elements of the particle by decreasing stiffness on it
    candidates = _patch_candidates(-1)
    candidate_stiffness = _patch_candidate_stiffness(0.0)
    for incident in range(particle_element_offset[pid], particle_element_offset[pid + 1]):
        element = particle_element[incident]
        stiffness = _element_stiffness(element, pid, triplet_values, element_vertices, element_triplet, element_size)
        if stiffness > 0.0:
            candidates, candidate_stiffness = _insert_candidate(candidates, candidate_stiffness, element, stiffness)

    # Grow the patch with the dynamic particles of each candidate that fits
    patch = vec8i(-1)
    patch[0] = pid
    count = wp.int32(1)
    for candidate in range(_PATCH_CANDIDATE_COUNT):
        element = candidates[candidate]
        if element >= 0 and count < PATCH_PARTICLE_COUNT:
            vertices = element_vertices[element]
            size = element_size[element]
            fresh = wp.vec4i(0)
            added = wp.int32(0)
            for corner in range(4):
                if corner < size:
                    vertex = vertices[corner]
                    if particle_dynamic[vertex] != 0 and _patch_slot(patch, count, vertex) < 0:
                        fresh[corner] = 1
                        added += 1
            if count + added <= PATCH_PARTICLE_COUNT:
                for corner in range(4):
                    if fresh[corner] != 0:
                        patch = _set_patch_particle(patch, count, vertices[corner])
                        count += 1

    last = _patch_particle(patch, count - 1)
    patch = _set_patch_particle(patch, count - 1, pid)
    patch[0] = last
    patch_particles[pid] = patch
    patch_count[pid] = count


@wp.kernel
def _assemble_consensus_patch_rows(
    # Inputs:
    smooth_offsets: wp.array[wp.int32],
    smooth_columns: wp.array[wp.int32],
    smooth_values: wp.array[wp.mat33],
    triplet_values: wp.array[wp.mat33],
    particle_element_offset: wp.array[wp.int32],
    particle_element: wp.array[wp.int32],
    element_vertices: wp.array[wp.vec4i],
    element_triplet: wp.array[wp.int32],
    element_size: wp.array[wp.int32],
    particle_dynamic: wp.array[wp.int32],
    packed_to_newton: wp.array[wp.int32],
    model_particle_mass: wp.array[float],
    patch_particles: wp.array[vec8i],
    patch_count: wp.array[wp.int32],
    # Outputs:
    patch_blocks: wp.array2d[wp.float32],
):
    """Assemble one block row of the lower block triangle of a patch matrix of each particle.

    Thread ``(matrix, slot, i)`` of the ``2 x PATCH_PARTICLE_COUNT x particle_count`` threads writes
    the blocks of row ``slot`` of the free (``matrix = 0``) or pinned (``matrix = 1``) patch matrix
    of particle ``i``, up to the diagonal block. The free row holds the mass of the slot particle and
    the pair blocks of its elements whose dynamic particles all lie in the patch; the pinned row holds
    the blocks of its row of ``A`` within the patch. The diagonal block holds the lower triangle of
    its terms, mirrored.
    """
    tid = wp.tid()
    particle_count = patch_count.shape[0]
    pid = tid % particle_count
    slot = (tid // particle_count) % PATCH_PARTICLE_COUNT
    matrix = tid // (particle_count * PATCH_PARTICLE_COUNT)
    count = patch_count[pid]
    if slot >= count:
        return
    patch = patch_particles[pid]
    particle = _patch_particle(patch, slot)
    row = _patch_row(0.0)
    if matrix == 0:
        particle_mass = model_particle_mass[packed_to_newton[particle]]
        row = _add_row_block(row, slot, wp.diag(wp.vec3(particle_mass)))
        for incident in range(particle_element_offset[particle], particle_element_offset[particle + 1]):
            element = particle_element[incident]
            vertices = element_vertices[element]
            size = element_size[element]
            slots = wp.vec4i(-1)
            inside = wp.bool(True)
            own_corner = wp.int32(-1)
            for corner in range(4):
                if corner < size:
                    vertex = vertices[corner]
                    vertex_slot = _patch_slot(patch, count, vertex)
                    if vertex_slot < 0:
                        inside = inside and particle_dynamic[vertex] == 0
                    elif vertex_slot == slot:
                        own_corner = corner
                    slots[corner] = vertex_slot
            if inside and own_corner >= 0:
                triplet = element_triplet[element] + own_corner * size
                for corner in range(4):
                    column_slot = slots[corner]
                    if corner < size and column_slot >= 0 and column_slot <= slot:
                        block = triplet_values[triplet + corner]
                        if column_slot == slot:
                            block = _mirror_lower(block)
                        row = _add_row_block(row, column_slot, block)
    else:
        for entry in range(smooth_offsets[particle], smooth_offsets[particle + 1]):
            column_slot = _patch_slot(patch, count, smooth_columns[entry])
            if column_slot >= 0 and column_slot <= slot:
                block = smooth_values[entry]
                if column_slot == slot:
                    block = _mirror_lower(block)
                row = _add_row_block(row, column_slot, block)

    first = 9 * (matrix * PATCH_BLOCK_COUNT + _block_index(slot, 0))
    for column_slot in range(PATCH_PARTICLE_COUNT):
        if column_slot <= slot:
            for entry in range(9):
                patch_blocks[first + 9 * column_slot + entry, pid] = row[9 * column_slot + entry]


@wp.kernel
def _eliminate_consensus_patch(
    # Inputs:
    patch_count: wp.array[wp.int32],
    # Outputs:
    patch_blocks: wp.array2d[wp.float32],
    patch_smallest: wp.array[wp.float32],
):
    """Reduce each patch matrix to the Schur complement of its particle and store its smallest eigenvalue.

    Thread ``(matrix, i)`` of the ``2 x particle_count`` threads eliminates the patch slots before the
    last from patch matrix ``matrix`` of particle ``i``, in place: each eliminated block column ``k``
    takes ``L_kk^-T`` from the right, and the trailing blocks lose its outer products, which leaves
    the Schur complement of the particle in the last diagonal block. The smallest eigenvalue is ``-1``
    for a matrix that is not positive definite.
    """
    tid = wp.tid()
    particle_count = patch_count.shape[0]
    pid = tid % particle_count
    matrix = tid // particle_count
    count = patch_count[pid]
    if count == 0:
        return
    first = 9 * matrix * PATCH_BLOCK_COUNT
    valid = wp.bool(True)
    for k in range(count - 1):
        inverse_transpose, pivot_valid = _cholesky_inverse_transpose(
            _read_patch_block(patch_blocks, first + 9 * _block_index(k, k), pid)
        )
        if not pivot_valid:
            valid = False
            break
        for i in range(k + 1, count):
            lower = first + 9 * _block_index(i, k)
            _write_patch_block(
                patch_blocks, lower, pid, _read_patch_block(patch_blocks, lower, pid) @ inverse_transpose
            )
        for i in range(k + 1, count):
            row_block = _read_patch_block(patch_blocks, first + 9 * _block_index(i, k), pid)
            for j in range(k + 1, i + 1):
                column_block = _read_patch_block(patch_blocks, first + 9 * _block_index(j, k), pid)
                target = first + 9 * _block_index(i, j)
                updated = _read_patch_block(patch_blocks, target, pid) - row_block @ wp.transpose(column_block)
                _write_patch_block(patch_blocks, target, pid, updated)
    smallest = wp.float32(-1.0)
    if valid:
        schur = _mirror_lower(_read_patch_block(patch_blocks, first + 9 * _block_index(count - 1, count - 1), pid))
        _eigenvectors, eigenvalues = wp.eig3(schur)
        least = wp.min(eigenvalues)
        if least > 0.0 and wp.isfinite(least):
            smallest = least
    patch_smallest[tid] = smallest


@wp.kernel
def _finish_consensus_weight(
    # Inputs:
    packed_to_newton: wp.array[wp.int32],
    model_particle_mass: wp.array[float],
    patch_count: wp.array[wp.int32],
    patch_smallest: wp.array[wp.float32],
    beta: wp.float32,
    sigma: wp.float32,
    # Outputs:
    weight: wp.array[float],
    inverse_weight: wp.array[float],
    weight_status: wp.array[wp.int32],
):
    """Set the weight of each particle with a patch to the geometric mean of the smallest eigenvalues of its
    free and pinned Schur complements, at least the particle mass.

    A positive ``beta`` caps the weight at ``beta`` times the free bound, and ``sigma`` keeps at least that
    fraction of the mean. A patch matrix that is not positive definite regularizes the weight to the
    particle mass.
    """
    pid = wp.tid()
    if patch_count[pid] == 0:
        return
    free_smallest = patch_smallest[pid]
    pinned_smallest = patch_smallest[patch_count.shape[0] + pid]
    mass = model_particle_mass[packed_to_newton[pid]]
    particle_weight = mass
    status = wp.int32(DeformableWeightStatus.REGULARIZED)
    if free_smallest > 0.0 and pinned_smallest > 0.0:
        # Round-off can only move the bounds out of their order m <= free <= pinned
        free_bound = wp.max(free_smallest, mass)
        particle_weight = wp.sqrt(free_bound * wp.max(pinned_smallest, free_bound))
        if beta > 0.0:
            particle_weight = wp.max(sigma * particle_weight, wp.min(particle_weight, beta * free_bound))
        status = wp.int32(DeformableWeightStatus.VALID)
    weight[pid] = particle_weight
    inverse_weight[pid] = 1.0 / particle_weight
    weight_status[pid] = status
