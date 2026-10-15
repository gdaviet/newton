# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Adaptive factorized sparse approximate inverse (FSAI) preconditioning of the LOX deformable system.

The preconditioner applies ``G^T G``, where ``G`` is a lower-triangular scalar factor with at
most :data:`_SUPPORT_SIZE` entries per row. The support of each row grows greedily from its
diagonal on the graph of the diagonally equilibrated system matrix, adding the lower entry
with the largest local residual. For a support ``S`` of row ``i``, the row solves
``A[S, S] z = e_i`` and stores ``z / sqrt(z_i)``. Every row is independent, so the setup,
refits, and both sparse products are parallel; no triangular solve is applied.

The supports are selected once, from the first factorized matrix, together with a
deterministic transposed index of the factor. Later factorizations refit the values on
these supports, since the system topology is fixed. The selection runs on the device behind
a ready flag, so that the first factorization may be captured in a CUDA graph.

See Kolotilina and Yeremin, "Factorized sparse approximate inverse preconditionings I. Theory",
SIAM Journal on Matrix Analysis and Applications 14(1), 1993, and the adaptive FSAI of hypre,
https://hypre.readthedocs.io/en/latest/solvers-fsai.html, for the pattern growth.
"""

from __future__ import annotations

import functools

import numpy as np
import warp as wp
import warp.sparse as wps

# Private Warp API: the scalar views of vector arrays, which make the factor gathers four-byte loads.
from warp._src import sparse as _wps_internal
from warp.optim.linear import LinearOperator

from .preconditioner import _initialize_factorization_status
from .types import DeformablePreconditionerStatus

###
# Module interface
###

__all__ = ["DeformableFSAI"]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


_SUPPORT_SIZE = 8
"""Default largest scalar entry count of a factor row, including its diagonal."""


_SUPPORT_SIZE_LIMIT = 16
"""Largest supported entry count of a factor row; setup keeps a dense local factor of that dimension."""


_GROWTH_TOLERANCE = 1.0e-3
"""Relative reduction of the local Kaporin bound below which a support stops growing."""


###
# Functions
###


@wp.func
def _find_block_slot(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    row: wp.int32,
    column: wp.int32,
) -> wp.int32:
    """Return the BSR value slot of block ``(row, column)``, or -1 if it is not stored."""
    lower = offsets[row]
    upper = offsets[row + 1]
    while lower < upper:
        middle = (lower + upper) // 2
        middle_column = columns[middle]
        if middle_column == column:
            return middle
        if middle_column < column:
            lower = middle + 1
        else:
            upper = middle
    return wp.int32(-1)


@wp.func
def _scalar_entry(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.mat33],
    row: wp.int32,
    column: wp.int32,
) -> float:
    """Return scalar entry ``(row, column)`` of a 3-by-3 BSR matrix, or zero if its block is not stored."""
    slot = _find_block_slot(offsets, columns, row // 3, column // 3)
    value = float(0.0)
    if slot >= 0:
        value = values[slot][row % 3, column % 3]
    return value


@wp.func
def _scalar_diagonal(
    diagonal_slots: wp.array[wp.int32],
    values: wp.array[wp.mat33],
    row: wp.int32,
) -> float:
    """Return the diagonal entry of a scalar row of a 3-by-3 BSR matrix."""
    return values[diagonal_slots[row // 3]][row % 3, row % 3]


###
# Kernels
###


@functools.cache
def _make_select_kernel(support_size: int):
    """Create the adaptive support-selection kernel for a static support size."""
    local_vector = wp.types.vector(length=support_size, dtype=wp.float32)
    local_indices = wp.types.vector(length=support_size, dtype=wp.int32)
    local_matrix = wp.types.matrix(shape=(support_size, support_size), dtype=wp.float32)

    @wp.kernel(module="unique")
    def select_fsai_support(
        # Inputs:
        offsets: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array[wp.mat33],
        diagonal_slots: wp.array[wp.int32],
        row_active: wp.array[wp.int32],
        pattern_ready: wp.array[wp.int32],
        regularization: float,
        # Outputs:
        factor_columns: wp.array2d[wp.int32],
    ):
        """Grow the support of one scalar factor row by the largest equilibrated local residual."""
        row = wp.tid()
        if pattern_ready[0] != 0:
            return
        for slot in range(wp.static(support_size)):
            factor_columns[row, slot] = -1
        factor_columns[row, 0] = row
        if row_active[row // 3] == 0:
            return
        row_diagonal = _scalar_diagonal(diagonal_slots, values, row)
        if not (wp.isfinite(row_diagonal) and row_diagonal > 0.0):
            return

        # Bordered Cholesky factor of the equilibrated support matrix, its forward solve of e_0,
        # and the support solution z, which all start from the diagonal entry.
        support = local_indices(-1)
        factor = local_matrix()
        forward = local_vector()
        solution = local_vector()
        support[0] = row
        factor[0, 0] = 1.0
        forward[0] = 1.0
        solution[0] = 1.0
        size = wp.int32(1)
        while size < wp.static(support_size):
            best = wp.int32(-1)
            best_score = float(0.0)
            for position in range(size):
                member = support[position]
                block_row = member // 3
                component = member % 3
                for entry in range(offsets[block_row], offsets[block_row + 1]):
                    block_column = columns[entry]
                    if row_active[block_column] != 0 and 3 * block_column < row:
                        for axis in range(3):
                            candidate = 3 * block_column + axis
                            selected = bool(candidate >= row)
                            for other in range(size):
                                if support[other] == candidate:
                                    selected = True
                            if candidate == best:
                                selected = True
                            if not selected and values[entry][component, axis] != 0.0:
                                candidate_diagonal = _scalar_diagonal(diagonal_slots, values, candidate)
                                if wp.isfinite(candidate_diagonal) and candidate_diagonal > 0.0:
                                    residual = float(0.0)
                                    for other in range(size):
                                        neighbor = support[other]
                                        residual += (
                                            _scalar_entry(offsets, columns, values, candidate, neighbor)
                                            * solution[other]
                                            / wp.sqrt(_scalar_diagonal(diagonal_slots, values, neighbor))
                                        )
                                    score = wp.abs(residual) / wp.sqrt(candidate_diagonal)
                                    if score > best_score or (score == best_score and candidate < best):
                                        best = candidate
                                        best_score = score
            if best < 0:
                break

            # Border the factor with the equilibrated column of the selected entry.
            best_scale = 1.0 / wp.sqrt(_scalar_diagonal(diagonal_slots, values, best))
            border = local_vector()
            pivot = float(1.0)
            for position in range(size):
                neighbor = support[position]
                value = (
                    _scalar_entry(offsets, columns, values, best, neighbor)
                    * best_scale
                    / wp.sqrt(_scalar_diagonal(diagonal_slots, values, neighbor))
                )
                for k in range(position):
                    value -= factor[position, k] * border[k]
                border[position] = value / factor[position, position]
                pivot -= border[position] * border[position]
            if not (wp.isfinite(pivot) and pivot > regularization):
                break
            pivot = wp.sqrt(pivot)
            support[size] = best
            value = float(0.0)
            for position in range(size):
                factor[size, position] = border[position]
                value -= border[position] * forward[position]
            factor[size, size] = pivot
            forward[size] = value / pivot
            previous = solution[0]
            size += 1
            for reverse in range(size):
                position = size - 1 - reverse
                value = forward[position]
                for k in range(position + 1, size):
                    value -= factor[k, position] * solution[k]
                solution[position] = value / factor[position, position]
            # The Kaporin bound is 1 / z_0; stop once a step barely reduces it.
            if 1.0 - previous / solution[0] <= wp.static(_GROWTH_TOLERANCE):
                break

        for position in range(size):
            factor_columns[row, position] = support[position]

    return select_fsai_support


@wp.kernel
def _count_transpose_entries(
    # Inputs:
    factor_columns: wp.array2d[wp.int32],
    pattern_ready: wp.array[wp.int32],
    # Outputs:
    transpose_counts: wp.array[wp.int32],
):
    """Count the factor entries in every scalar column."""
    row = wp.tid()
    if pattern_ready[0] != 0:
        return
    for slot in range(factor_columns.shape[1]):
        column = factor_columns[row, slot]
        if column >= 0:
            wp.atomic_add(transpose_counts, column + 1, 1)


@wp.kernel
def _scan_transpose_offsets(
    # Inputs:
    transpose_counts: wp.array[wp.int32],
    pattern_ready: wp.array[wp.int32],
    # Outputs:
    transpose_offsets: wp.array[wp.int32],
):
    """Accumulate the column offsets of the transposed factor index serially, once."""
    if pattern_ready[0] != 0:
        return
    total = wp.int32(0)
    for column in range(transpose_offsets.shape[0]):
        total += transpose_counts[column]
        transpose_offsets[column] = total


@wp.kernel
def _fill_transpose_entries(
    # Inputs:
    factor_columns: wp.array2d[wp.int32],
    transpose_offsets: wp.array[wp.int32],
    pattern_ready: wp.array[wp.int32],
    # Outputs:
    transpose_cursor: wp.array[wp.int32],
    transpose_entries: wp.array[wp.int32],
):
    """Scatter the flat factor slot of every entry into its column segment."""
    row = wp.tid()
    if pattern_ready[0] != 0:
        return
    support_size = factor_columns.shape[1]
    for slot in range(support_size):
        column = factor_columns[row, slot]
        if column >= 0:
            position = transpose_offsets[column] + wp.atomic_add(transpose_cursor, column, 1)
            transpose_entries[position] = row * support_size + slot


@wp.kernel
def _sort_transpose_entries(
    # Inputs:
    transpose_offsets: wp.array[wp.int32],
    pattern_ready: wp.array[wp.int32],
    # Outputs:
    transpose_entries: wp.array[wp.int32],
):
    """Sort every column segment by factor row, so the transposed product sums in a fixed order."""
    column = wp.tid()
    if pattern_ready[0] != 0:
        return
    begin = transpose_offsets[column]
    end = transpose_offsets[column + 1]
    for i in range(begin + 1, end):
        key = transpose_entries[i]
        j = i - 1
        while j >= begin and transpose_entries[j] > key:
            transpose_entries[j + 1] = transpose_entries[j]
            j -= 1
        transpose_entries[j + 1] = key


@wp.kernel
def _extract_transpose_rows(
    # Inputs:
    transpose_entries: wp.array[wp.int32],
    support_size: wp.int32,
    pattern_ready: wp.array[wp.int32],
    # Outputs:
    transpose_rows: wp.array[wp.int32],
):
    """Store the factor row of every transposed entry."""
    position = wp.tid()
    if pattern_ready[0] != 0:
        return
    transpose_rows[position] = transpose_entries[position] // support_size


@wp.kernel
def _gather_transpose_values(
    # Inputs:
    factor_values: wp.array2d[wp.float32],
    transpose_entries: wp.array[wp.int32],
    transpose_offsets: wp.array[wp.int32],
    # Outputs:
    transpose_values: wp.array[wp.float32],
):
    """Copy the factor values in transposed order, so that the transposed product streams them."""
    position = wp.tid()
    if position >= transpose_offsets[transpose_offsets.shape[0] - 1]:
        return
    support_size = factor_values.shape[1]
    entry = transpose_entries[position]
    row = entry // support_size
    transpose_values[position] = factor_values[row, entry - row * support_size]


@wp.kernel
def _mark_pattern_ready(
    # Outputs:
    pattern_ready: wp.array[wp.int32],
):
    """Mark the supports and the transposed factor index as built."""
    pattern_ready[0] = 1


@functools.cache
def _make_refit_kernel(support_size: int):
    """Create the factor refit kernel for a static support size."""
    local_vector = wp.types.vector(length=support_size, dtype=wp.float32)
    local_matrix = wp.types.matrix(shape=(support_size, support_size), dtype=wp.float32)

    @wp.kernel(module="unique")
    def refit_fsai(
        # Inputs:
        offsets: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array[wp.mat33],
        factor_columns: wp.array2d[wp.int32],
        packed_world: wp.array[wp.int32],
        row_active: wp.array[wp.int32],
        regularization: float,
        # Outputs:
        factor_values: wp.array2d[wp.float32],
        world_status: wp.array[wp.int32],
    ):
        """Solve ``A[S, S] z = e_0`` on the support of one scalar row and store ``z / sqrt(z_0)``."""
        row = wp.tid()
        for slot in range(wp.static(support_size)):
            factor_values[row, slot] = 0.0
        if row_active[row // 3] == 0:
            factor_values[row, 0] = 1.0
            return

        # Left-looking Cholesky of the support matrix, padded with identity rows.
        matrix = local_matrix()
        size = wp.int32(0)
        for slot in range(wp.static(support_size)):
            if factor_columns[row, slot] >= 0:
                size = slot + 1
        for i in range(wp.static(support_size)):
            if i < size:
                row_i = factor_columns[row, i]
                for j in range(i + 1):
                    matrix[i, j] = _scalar_entry(offsets, columns, values, row_i, factor_columns[row, j])
            else:
                matrix[i, i] = 1.0
        status = wp.int32(DeformablePreconditionerStatus.VALID)
        for j in range(wp.static(support_size)):
            original = matrix[j, j]
            pivot = original
            for k in range(j):
                pivot -= matrix[j, k] * matrix[j, k]
            if not (wp.isfinite(pivot) and original > 0.0):
                status = wp.int32(DeformablePreconditionerStatus.FAILED)
                pivot = wp.max(wp.abs(original), 1.0)
            floor = regularization * original
            if pivot < floor:
                pivot = floor
                if status == DeformablePreconditionerStatus.VALID:
                    status = wp.int32(DeformablePreconditionerStatus.REGULARIZED)
            pivot = wp.sqrt(pivot)
            matrix[j, j] = pivot
            for i in range(j + 1, wp.static(support_size)):
                value = matrix[i, j]
                for k in range(j):
                    value -= matrix[i, k] * matrix[j, k]
                matrix[i, j] = value / pivot

        # z = L^-T L^-1 e_0.
        solution = local_vector()
        solution[0] = 1.0 / matrix[0, 0]
        for i in range(1, wp.static(support_size)):
            value = float(0.0)
            for k in range(i):
                value -= matrix[i, k] * solution[k]
            solution[i] = value / matrix[i, i]
        for reverse in range(wp.static(support_size)):
            i = wp.static(support_size) - 1 - reverse
            value = solution[i]
            for k in range(i + 1, wp.static(support_size)):
                value -= matrix[k, i] * solution[k]
            solution[i] = value / matrix[i, i]
        diagonal = solution[0]
        if not (wp.isfinite(diagonal) and diagonal > 0.0):
            status = wp.int32(DeformablePreconditionerStatus.FAILED)
            factor_values[row, 0] = 1.0
        else:
            norm = 1.0 / wp.sqrt(diagonal)
            for slot in range(wp.static(support_size)):
                if slot < size:
                    factor_values[row, slot] = solution[slot] * norm
        if status != DeformablePreconditionerStatus.VALID:
            wp.atomic_max(world_status, packed_world[row // 3], status)

    return refit_fsai


@wp.kernel
def _apply_fsai_factor(
    # Inputs:
    factor_columns: wp.array2d[wp.int32],
    factor_values: wp.array2d[wp.float32],
    right_hand_side: wp.array[wp.float32],
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    row_active: wp.array[wp.int32],
    # Outputs:
    factor_product: wp.array[wp.float32],
):
    """Write ``G x`` on the scalar conjugate-residual rows of active worlds and zero elsewhere."""
    row = wp.tid()
    particle = row // 3
    wid = packed_world[particle]
    value = float(0.0)
    if row_active[particle] != 0 and world_active[wid] and world_status[wid] != DeformablePreconditionerStatus.FAILED:
        for slot in range(factor_columns.shape[1]):
            column = factor_columns[row, slot]
            if column >= 0:
                value += factor_values[row, slot] * right_hand_side[column]
    factor_product[row] = value


@wp.kernel
def _apply_fsai_transpose(
    # Inputs:
    transpose_offsets: wp.array[wp.int32],
    transpose_rows: wp.array[wp.int32],
    transpose_values: wp.array[wp.float32],
    factor_product: wp.array[wp.float32],
    right_hand_side: wp.array[wp.float32],
    addend: wp.array[wp.float32],
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    row_active: wp.array[wp.int32],
    alpha: float,
    beta: float,
    # Outputs:
    result: wp.array[wp.float32],
):
    """Write ``alpha G^T (G x) + beta y`` on the scalar conjugate-residual rows of active worlds."""
    column = wp.tid()
    particle = column // 3
    wid = packed_world[particle]
    value = right_hand_side[column]
    if row_active[particle] != 0 and world_active[wid] and world_status[wid] != DeformablePreconditionerStatus.FAILED:
        value = float(0.0)
        for position in range(transpose_offsets[column], transpose_offsets[column + 1]):
            value += transpose_values[position] * factor_product[transpose_rows[position]]
    value *= alpha
    if beta != 0.0:
        value += beta * addend[column]
    result[column] = value


###
# Interfaces
###


class DeformableFSAI:
    """Apply an adaptive factorized sparse approximate inverse ``G^T G`` of the system matrix.

    Args:
        system_matrix: Symmetric 3-by-3 block system matrix with both triangles stored.
        diagonal_slots: BSR value slot for every diagonal block.
        packed_world: World index for every packed block row.
        world_active: Mutable active flag for every world.
        batch_offsets: Scalar degree-of-freedom offsets for Warp batching.
        regularization: Relative positive pivot floor of the local Cholesky factors.
        row_active: Optional nonzero flag for rows assigned to CR.
        support_size: Largest scalar entry count of a factor row, including its diagonal.
            One gives scalar Jacobi.
    """

    def __init__(
        self,
        system_matrix: wps.BsrMatrix,
        diagonal_slots: wp.array[wp.int32],
        packed_world: wp.array[wp.int32],
        world_active: wp.array[wp.bool],
        batch_offsets: wp.array[wp.int32],
        regularization: float = 1.0e-6,
        row_active: wp.array[wp.int32] | None = None,
        *,
        support_size: int = _SUPPORT_SIZE,
    ):
        if system_matrix.block_shape != (3, 3) or system_matrix.nrow != system_matrix.ncol:
            raise ValueError("LOX deformable FSAI requires a square 3-by-3 BSR matrix.")
        if diagonal_slots.shape != (system_matrix.nrow,) or diagonal_slots.dtype != wp.int32:
            raise ValueError("LOX deformable FSAI requires one int32 diagonal slot per block row.")
        if not np.isfinite(regularization) or regularization <= 0.0 or regularization >= 1.0:
            raise ValueError(f"LOX deformable FSAI regularization must be in (0, 1), got {regularization}.")
        if (
            not isinstance(support_size, int)
            or isinstance(support_size, bool)
            or not 1 <= support_size <= _SUPPORT_SIZE_LIMIT
        ):
            raise ValueError(
                f"LOX deformable FSAI support size must be an integer in [1, {_SUPPORT_SIZE_LIMIT}], got {support_size}."
            )
        self.system_matrix = system_matrix
        self.device = system_matrix.device
        self.row_count = int(system_matrix.nrow)
        self.diagonal_slots = diagonal_slots
        self.packed_world = packed_world
        self.world_active = world_active
        self.regularization = float(regularization)
        self.support_size = support_size
        """Largest scalar entry count of a factor row, including its diagonal."""
        if row_active is None:
            row_active = wp.ones(self.row_count, dtype=wp.int32, device=self.device)
        elif row_active.shape != (self.row_count,) or row_active.dtype != wp.int32:
            raise ValueError("LOX deformable FSAI requires one int32 active flag per row.")
        self.row_active = row_active
        scalar_count = 3 * self.row_count
        self.world_status = wp.zeros(world_active.shape[0], dtype=wp.int32, device=self.device)
        """:class:`DeformablePreconditionerStatus` of each world after the last factorization."""
        self.factor_columns = wp.empty((scalar_count, support_size), dtype=wp.int32, device=self.device)
        """Scalar column of each factor slot, with the diagonal in slot 0 and -1 for unused slots."""
        self.factor_values = wp.zeros((scalar_count, support_size), dtype=wp.float32, device=self.device)
        """Value of each factor slot."""
        self.transpose_offsets = wp.zeros(scalar_count + 1, dtype=wp.int32, device=self.device)
        """Offset of the factor entries of every scalar column in :attr:`transpose_entries`."""
        self.transpose_entries = wp.zeros(scalar_count * support_size, dtype=wp.int32, device=self.device)
        """Flat factor slot ``row * support_size + slot`` of every entry, by column and then row."""
        self.transpose_rows = wp.zeros(scalar_count * support_size, dtype=wp.int32, device=self.device)
        """Factor row of every entry of :attr:`transpose_entries`."""
        self.transpose_values = wp.zeros(scalar_count * support_size, dtype=wp.float32, device=self.device)
        """Factor value of every entry of :attr:`transpose_entries`, refreshed by every factorization."""
        self.pattern_ready = wp.zeros(1, dtype=wp.int32, device=self.device)
        """Nonzero once the supports and the transposed index are built."""
        self.factor_product = wp.zeros(scalar_count, dtype=wp.float32, device=self.device)
        """Scratch product ``G x`` of the last application."""
        self._transpose_counts = wp.zeros(scalar_count + 1, dtype=wp.int32, device=self.device)
        self._transpose_cursor = wp.zeros(scalar_count, dtype=wp.int32, device=self.device)
        self._pattern_built = False
        self._scalar_views: dict[tuple[int, int], wp.array[wp.float32]] = {}
        self._select_kernel = _make_select_kernel(support_size)
        self._refit_kernel = _make_refit_kernel(support_size)
        self.linear_operator = LinearOperator(
            shape=system_matrix.shape,
            dtype=system_matrix.dtype,
            device=self.device,
            matvec=self._matvec,
            batch_offsets=batch_offsets,
        )
        """Warp linear operator applying the preconditioner."""

    ###
    # Public API
    ###

    def factorize(self) -> None:
        """Refit the factor to the current system matrix, selecting the supports on the first call."""
        if not self._pattern_built:
            self._build_pattern()
        wp.launch(
            _initialize_factorization_status,
            dim=self.world_status.shape[0],
            outputs=[self.world_status],
            device=self.device,
        )
        matrix = self.system_matrix
        wp.launch(
            self._refit_kernel,
            dim=3 * self.row_count,
            inputs=[
                matrix.offsets,
                matrix.columns,
                matrix.values,
                self.factor_columns,
                self.packed_world,
                self.row_active,
                self.regularization,
            ],
            outputs=[self.factor_values, self.world_status],
            device=self.device,
        )
        wp.launch(
            _gather_transpose_values,
            dim=self.transpose_values.shape[0],
            inputs=[self.factor_values, self.transpose_entries, self.transpose_offsets],
            outputs=[self.transpose_values],
            device=self.device,
        )

    ###
    # Internals
    ###

    def _build_pattern(self) -> None:
        """Launch the flag-gated support selection and transposed index construction."""
        matrix = self.system_matrix
        scalar_count = 3 * self.row_count
        wp.launch(
            self._select_kernel,
            dim=scalar_count,
            inputs=[
                matrix.offsets,
                matrix.columns,
                matrix.values,
                self.diagonal_slots,
                self.row_active,
                self.pattern_ready,
                self.regularization,
            ],
            outputs=[self.factor_columns],
            device=self.device,
        )
        wp.launch(
            _count_transpose_entries,
            dim=scalar_count,
            inputs=[self.factor_columns, self.pattern_ready],
            outputs=[self._transpose_counts],
            device=self.device,
        )
        wp.launch(
            _scan_transpose_offsets,
            dim=1,
            inputs=[self._transpose_counts, self.pattern_ready],
            outputs=[self.transpose_offsets],
            device=self.device,
        )
        wp.launch(
            _fill_transpose_entries,
            dim=scalar_count,
            inputs=[self.factor_columns, self.transpose_offsets, self.pattern_ready],
            outputs=[self._transpose_cursor, self.transpose_entries],
            device=self.device,
        )
        wp.launch(
            _sort_transpose_entries,
            dim=scalar_count,
            inputs=[self.transpose_offsets, self.pattern_ready],
            outputs=[self.transpose_entries],
            device=self.device,
        )
        wp.launch(
            _extract_transpose_rows,
            dim=self.transpose_rows.shape[0],
            inputs=[self.transpose_entries, self.support_size, self.pattern_ready],
            outputs=[self.transpose_rows],
            device=self.device,
        )
        wp.launch(_mark_pattern_ready, dim=1, outputs=[self.pattern_ready], device=self.device)
        # A captured graph replays the gated launches; an eager call completes the build.
        if not self.device.is_capturing:
            self._pattern_built = True

    def _matvec(
        self,
        x: wp.array[wp.vec3],
        y: wp.array[wp.vec3],
        z: wp.array[wp.vec3],
        alpha: float,
        beta: float,
    ) -> None:
        """Write ``z = alpha G^T G x + beta y``."""
        x_scalar = self._scalar_view(x)
        wp.launch(
            _apply_fsai_factor,
            dim=3 * self.row_count,
            inputs=[
                self.factor_columns,
                self.factor_values,
                x_scalar,
                self.packed_world,
                self.world_active,
                self.world_status,
                self.row_active,
            ],
            outputs=[self.factor_product],
            device=self.device,
        )
        wp.launch(
            _apply_fsai_transpose,
            dim=3 * self.row_count,
            inputs=[
                self.transpose_offsets,
                self.transpose_rows,
                self.transpose_values,
                self.factor_product,
                x_scalar,
                self._scalar_view(y),
                self.packed_world,
                self.world_active,
                self.world_status,
                self.row_active,
                alpha,
                beta,
            ],
            outputs=[self._scalar_view(z)],
            device=self.device,
        )

    def _scalar_view(self, vector: wp.array[wp.vec3]) -> wp.array[wp.float32]:
        """Return a cached flat ``float32`` view of a particle vector array."""
        key = (vector.ptr, vector.shape[0])
        view = self._scalar_views.get(key)
        if view is None:
            view = _wps_internal._vec_array_view(vector, wp.float32, 3 * self.row_count)
            self._scalar_views[key] = view
        return view
