# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Block incomplete LDLT preconditioning for the LOX cloth system."""

from __future__ import annotations

import numpy as np
import warp as wp
import warp.sparse as wps
from warp.optim.linear import LinearOperator

__all__ = [
    "DEFORMABLE_PRECONDITIONER_STATUS_FAILED",
    "DEFORMABLE_PRECONDITIONER_STATUS_REGULARIZED",
    "DEFORMABLE_PRECONDITIONER_STATUS_VALID",
    "DeformableIncompleteLDLT",
]

DEFORMABLE_PRECONDITIONER_STATUS_VALID = 0
"""The incomplete factorization completed without pivot modification."""

DEFORMABLE_PRECONDITIONER_STATUS_REGULARIZED = 1
"""At least one diagonal pivot was regularized."""

DEFORMABLE_PRECONDITIONER_STATUS_FAILED = 2
"""The factorization encountered non-finite matrix data."""

_PERSISTENT_ROW_LIMIT = 256
"""Largest world eligible for eager persistent application.

The factor vectors remain in preallocated global storage, so this is a
parallelism crossover rather than a shared-memory capacity limit. Larger
worlds expose enough level parallelism to favor scheduled kernels.
"""

_PERSISTENT_BLOCK_DIM_LIMIT = 512
"""Largest thread block used by persistent application."""

_LEVEL_BLOCK_DIM = 32
"""Thread block size used by level-scheduled triangular solves."""


@wp.func
def _is_finite_mat33(value: wp.mat33) -> bool:
    finite = True
    for row in range(3):
        for column in range(3):
            finite = finite and wp.isfinite(value[row, column])
    return finite


@wp.func
def _max_abs_mat33(value: wp.mat33) -> float:
    scale = 0.0
    for row in range(3):
        for column in range(3):
            scale = wp.max(scale, wp.abs(value[row, column]))
    return scale


@wp.func
def _diagonal_mat33(value: wp.vec3) -> wp.mat33:
    result = wp.mat33(0.0)
    result[0, 0] = value[0]
    result[1, 1] = value[1]
    result[2, 2] = value[2]
    return result


@wp.kernel
def _begin_factorization(
    world_status: wp.array[wp.int32],
):
    world = wp.tid()
    world_status[world] = DEFORMABLE_PRECONDITIONER_STATUS_VALID


@wp.kernel
def _factorize_level(
    level_rows: wp.array[wp.int32],
    level_start: int,
    lower_offsets: wp.array[wp.int32],
    lower_columns: wp.array[wp.int32],
    lower_system_slots: wp.array[wp.int32],
    common_offsets: wp.array[wp.int32],
    common_row_slots: wp.array[wp.int32],
    common_column_slots: wp.array[wp.int32],
    common_diagonals: wp.array[wp.int32],
    system_diagonal_slots: wp.array[wp.int32],
    system_values: wp.array[wp.mat33],
    packed_world: wp.array[wp.int32],
    row_active: wp.array[wp.int32],
    regularization: float,
    lower_values: wp.array[wp.mat33],
    diagonal_values: wp.array[wp.mat33],
    inverse_diagonal_values: wp.array[wp.mat33],
    world_status: wp.array[wp.int32],
):
    level_index = wp.tid()
    row = level_rows[level_start + level_index]
    if row_active[row] == 0:
        lower_slot = lower_offsets[row]
        while lower_slot < lower_offsets[row + 1]:
            lower_values[lower_slot] = wp.mat33(0.0)
            lower_slot += 1
        diagonal_values[row] = wp.identity(n=3, dtype=float)
        inverse_diagonal_values[row] = wp.identity(n=3, dtype=float)
        return
    status = wp.int32(DEFORMABLE_PRECONDITIONER_STATUS_VALID)

    lower_start = lower_offsets[row]
    lower_end = lower_offsets[row + 1]
    lower_slot = lower_start
    while lower_slot < lower_end:
        column = lower_columns[lower_slot]
        system_slot = lower_system_slots[lower_slot]
        schur_block = wp.mat33(0.0)
        if system_slot >= 0:
            schur_block = system_values[system_slot]
        common = common_offsets[lower_slot]
        common_end = common_offsets[lower_slot + 1]
        while common < common_end:
            row_slot = common_row_slots[common]
            column_slot = common_column_slots[common]
            diagonal = common_diagonals[common]
            schur_block -= lower_values[row_slot] * diagonal_values[diagonal] * wp.transpose(lower_values[column_slot])
            common += 1

        if _is_finite_mat33(schur_block):
            lower_values[lower_slot] = schur_block * inverse_diagonal_values[column]
        else:
            lower_values[lower_slot] = wp.mat33(0.0)
            status = DEFORMABLE_PRECONDITIONER_STATUS_FAILED
        lower_slot += 1

    pivot = system_values[system_diagonal_slots[row]]
    lower_slot = lower_start
    while lower_slot < lower_end:
        column = lower_columns[lower_slot]
        lower_block = lower_values[lower_slot]
        pivot -= lower_block * diagonal_values[column] * wp.transpose(lower_block)
        lower_slot += 1
    pivot = 0.5 * (pivot + wp.transpose(pivot))

    if not _is_finite_mat33(pivot):
        pivot = wp.identity(n=3, dtype=float)
        diagonal_values[row] = pivot
        inverse_diagonal_values[row] = pivot
        status = DEFORMABLE_PRECONDITIONER_STATUS_FAILED
    else:
        eigenvectors, eigenvalues = wp.eig3(pivot)
        scale = wp.max(_max_abs_mat33(pivot), 1.0e-12)
        eigenvalue_floor = wp.max(regularization * scale, 1.0e-12)
        finite_eigendecomposition = _is_finite_mat33(eigenvectors)
        for axis in range(3):
            finite_eigendecomposition = finite_eigendecomposition and wp.isfinite(eigenvalues[axis])

        if not finite_eigendecomposition:
            pivot = scale * wp.identity(n=3, dtype=float)
            diagonal_values[row] = pivot
            inverse_diagonal_values[row] = (1.0 / scale) * wp.identity(n=3, dtype=float)
            status = DEFORMABLE_PRECONDITIONER_STATUS_FAILED
        else:
            inverse_eigenvalues = wp.vec3(0.0)
            for axis in range(3):
                if eigenvalues[axis] < eigenvalue_floor:
                    eigenvalues[axis] = eigenvalue_floor
                    if status != DEFORMABLE_PRECONDITIONER_STATUS_FAILED:
                        status = DEFORMABLE_PRECONDITIONER_STATUS_REGULARIZED
                inverse_eigenvalues[axis] = 1.0 / eigenvalues[axis]
            diagonal_values[row] = eigenvectors * _diagonal_mat33(eigenvalues) * wp.transpose(eigenvectors)
            inverse_diagonal_values[row] = (
                eigenvectors * _diagonal_mat33(inverse_eigenvalues) * wp.transpose(eigenvectors)
            )

    if status != DEFORMABLE_PRECONDITIONER_STATUS_VALID:
        world = packed_world[row]
        wp.atomic_max(world_status, world, status)


@wp.kernel
def _forward_level(
    level_rows: wp.array[wp.int32],
    level_start: int,
    lower_offsets: wp.array[wp.int32],
    lower_columns: wp.array[wp.int32],
    lower_values: wp.array[wp.mat33],
    row_active: wp.array[wp.int32],
    right_hand_side: wp.array[wp.vec3],
    forward_solution: wp.array[wp.vec3],
):
    level_index = wp.tid()
    row = level_rows[level_start + level_index]
    value = right_hand_side[row]
    if row_active[row] == 0:
        forward_solution[row] = value
        return
    lower_slot = lower_offsets[row]
    lower_end = lower_offsets[row + 1]
    while lower_slot < lower_end:
        value -= lower_values[lower_slot] * forward_solution[lower_columns[lower_slot]]
        lower_slot += 1
    forward_solution[row] = value


@wp.kernel
def _refresh_upper_values(
    upper_lower_slots: wp.array[wp.int32],
    lower_values: wp.array[wp.mat33],
    upper_values: wp.array[wp.mat33],
):
    upper_slot = wp.tid()
    upper_values[upper_slot] = wp.transpose(lower_values[upper_lower_slots[upper_slot]])


@wp.kernel
def _backward_level(
    level_rows: wp.array[wp.int32],
    level_start: int,
    upper_offsets: wp.array[wp.int32],
    upper_rows: wp.array[wp.int32],
    upper_values: wp.array[wp.mat33],
    inverse_diagonal_values: wp.array[wp.mat33],
    forward_solution: wp.array[wp.vec3],
    right_hand_side: wp.array[wp.vec3],
    addend: wp.array[wp.vec3],
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    row_active: wp.array[wp.int32],
    alpha: float,
    beta: float,
    backward_solution: wp.array[wp.vec3],
    result: wp.array[wp.vec3],
):
    level_index = wp.tid()
    row = level_rows[level_start + level_index]
    world = packed_world[row]
    value = right_hand_side[row]
    if (
        row_active[row] != 0
        and world_active[world] != 0
        and world_status[world] != DEFORMABLE_PRECONDITIONER_STATUS_FAILED
    ):
        value = inverse_diagonal_values[row] * forward_solution[row]
        upper_slot = upper_offsets[row]
        upper_end = upper_offsets[row + 1]
        while upper_slot < upper_end:
            source_row = upper_rows[upper_slot]
            value -= upper_values[upper_slot] * backward_solution[source_row]
            upper_slot += 1
    backward_solution[row] = value
    value *= alpha
    if beta != 0.0:
        value += beta * addend[row]
    result[row] = value


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


@wp.kernel
def _persistent_apply(
    world_row_offsets: wp.array[wp.int32],
    world_level_offsets: wp.array[wp.int32],
    world_level_rows: wp.array[wp.int32],
    level_count: int,
    lower_offsets: wp.array[wp.int32],
    lower_columns: wp.array[wp.int32],
    lower_values: wp.array[wp.mat33],
    inverse_diagonal_values: wp.array[wp.mat33],
    upper_offsets: wp.array[wp.int32],
    upper_rows: wp.array[wp.int32],
    upper_lower_slots: wp.array[wp.int32],
    right_hand_side: wp.array[wp.vec3],
    addend: wp.array[wp.vec3],
    world_active: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    row_active: wp.array[wp.int32],
    alpha: float,
    beta: float,
    forward_solution: wp.array[wp.vec3],
    result: wp.array[wp.vec3],
):
    thread = wp.tid()
    block_dim = wp.block_dim()
    world = thread // block_dim
    lane = thread - world * block_dim
    row_start = world_row_offsets[world]
    row_end = world_row_offsets[world + 1]
    solve = world_active[world] != 0 and world_status[world] != DEFORMABLE_PRECONDITIONER_STATUS_FAILED

    if not solve:
        row = row_start + lane
        while row < row_end:
            value = alpha * right_hand_side[row]
            if beta != 0.0:
                value += beta * addend[row]
            result[row] = value
            row += block_dim
        return

    level_stride = level_count + 1
    level = wp.int32(0)
    while level < level_count:
        level_start = world_level_offsets[world * level_stride + level]
        level_end = world_level_offsets[world * level_stride + level + 1]
        level_pass = wp.int32(0)
        while level_start + level_pass < level_end:
            level_index = level_start + level_pass + lane
            if level_index < level_end:
                row = world_level_rows[level_index]
                value = right_hand_side[row]
                if row_active[row] != 0:
                    lower_slot = lower_offsets[row]
                    lower_end = lower_offsets[row + 1]
                    while lower_slot < lower_end:
                        column = lower_columns[lower_slot]
                        value -= lower_values[lower_slot] * forward_solution[column]
                        lower_slot += 1
                forward_solution[row] = value
            _sync_threads()
            level_pass += block_dim
        level += 1

    level = level_count - 1
    while level >= 0:
        level_start = world_level_offsets[world * level_stride + level]
        level_end = world_level_offsets[world * level_stride + level + 1]
        level_pass = wp.int32(0)
        while level_start + level_pass < level_end:
            level_index = level_start + level_pass + lane
            active = level_index < level_end
            row = row_start
            value = wp.vec3(0.0)
            if active:
                row = world_level_rows[level_index]
                value = forward_solution[row]
                if row_active[row] != 0:
                    value = inverse_diagonal_values[row] * value
                    upper_slot = upper_offsets[row]
                    upper_end = upper_offsets[row + 1]
                    while upper_slot < upper_end:
                        source_row = upper_rows[upper_slot]
                        lower_slot = upper_lower_slots[upper_slot]
                        value -= wp.transpose(lower_values[lower_slot]) * forward_solution[source_row]
                        upper_slot += 1
                forward_solution[row] = value
            if active:
                value *= alpha
                if beta != 0.0:
                    value += beta * addend[row]
                result[row] = value
            _sync_threads()
            level_pass += block_dim
        level -= 1


class DeformableIncompleteLDLT:
    """Apply an IC(k)-pattern block incomplete LDLT preconditioner."""

    def __init__(
        self,
        system_matrix: wps.BsrMatrix,
        packed_world: wp.array[wp.int32],
        world_active: wp.array[wp.int32],
        batch_offsets: wp.array[wp.int32],
        regularization: float = 1.0e-6,
        row_active: wp.array[wp.int32] | None = None,
    ):
        """Construct symbolic IC(0) data from an immutable BSR topology.

        Args:
            system_matrix: Symmetric 3-by-3 block system matrix.
            packed_world: World index for every packed block row.
            world_active: Mutable active flag for every world.
            batch_offsets: Scalar degree-of-freedom offsets for Warp batching.
            regularization: Relative eigenvalue floor applied to pivots.
            row_active: Optional nonzero flag for rows assigned to CR.
        """
        if system_matrix.block_shape != (3, 3) or system_matrix.nrow != system_matrix.ncol:
            raise ValueError("LOX cloth incomplete LDLT requires a square 3-by-3 BSR matrix.")
        if not np.isfinite(regularization) or regularization <= 0.0:
            raise ValueError(
                f"LOX cloth incomplete LDLT regularization must be finite and positive, got {regularization}."
            )
        self.system_matrix = system_matrix
        self.device = system_matrix.device
        self.row_count = int(system_matrix.nrow)
        self.regularization = float(regularization)
        self.packed_world = packed_world
        self.world_active = world_active
        if row_active is None:
            row_active = wp.ones(self.row_count, dtype=wp.int32, device=self.device)
        elif row_active.shape != (self.row_count,) or row_active.dtype != wp.int32:
            raise ValueError("LOX cloth incomplete LDLT requires one int32 active flag per row.")
        self.row_active = row_active

        system_offsets = system_matrix.offsets.numpy()
        system_columns = system_matrix.columns.numpy()
        packed_world_np = packed_world.numpy().astype(np.int32, copy=False)
        batch_offsets_np = batch_offsets.numpy().astype(np.int64, copy=False)
        if (
            batch_offsets_np.ndim != 1
            or batch_offsets_np.shape[0] < 2
            or batch_offsets_np[0] != 0
            or batch_offsets_np[-1] != 3 * self.row_count
            or np.any(batch_offsets_np % 3 != 0)
        ):
            raise ValueError("LOX cloth incomplete LDLT requires valid scalar batch ranges.")
        batch_row_offsets_np = (batch_offsets_np // 3).astype(np.int32)
        if np.any(np.diff(batch_row_offsets_np) <= 0):
            raise ValueError("LOX cloth incomplete LDLT requires nonempty scalar batch ranges.")
        world_counts = np.bincount(packed_world_np, minlength=world_active.shape[0])
        world_row_offsets_np = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(world_counts, dtype=np.int32)))
        expected_packed_world = np.repeat(
            np.arange(world_active.shape[0], dtype=np.int32),
            np.diff(world_row_offsets_np),
        )
        if not np.array_equal(packed_world_np, expected_packed_world):
            raise ValueError("LOX cloth incomplete LDLT requires rows packed contiguously by world.")
        system_slot_by_coordinate: dict[tuple[int, int], int] = {}
        original_lower_coordinates: list[tuple[int, int]] = []
        diagonal_system_slots = np.full(self.row_count, -1, dtype=np.int32)
        for row in range(self.row_count):
            for slot in range(int(system_offsets[row]), int(system_offsets[row + 1])):
                column = int(system_columns[slot])
                system_slot_by_coordinate[(row, column)] = slot
                if column < row:
                    original_lower_coordinates.append((row, column))
                elif column == row:
                    diagonal_system_slots[row] = slot

        if np.any(diagonal_system_slots < 0):
            raise ValueError("LOX cloth incomplete LDLT requires one diagonal block per row.")

        for row, column in system_slot_by_coordinate:
            if row != column and (column, row) not in system_slot_by_coordinate:
                raise ValueError("LOX cloth incomplete LDLT requires a structurally symmetric system matrix.")

        lower_columns_by_row: list[list[int]] = [[] for _ in range(self.row_count)]
        for row, column in original_lower_coordinates:
            lower_columns_by_row[row].append(column)

        lower_coordinates = [
            (row, column) for row, row_columns in enumerate(lower_columns_by_row) for column in sorted(row_columns)
        ]
        lower_coordinate_array = np.asarray(lower_coordinates, dtype=np.int32).reshape((-1, 2))
        lower_rows = wp.array(lower_coordinate_array[:, 0], dtype=wp.int32, device=self.device)
        lower_columns = wp.array(lower_coordinate_array[:, 1], dtype=wp.int32, device=self.device)
        lower_initial_values = wp.zeros(lower_coordinate_array.shape[0], dtype=wp.mat33, device=self.device)
        self.lower_matrix = wps.bsr_zeros(
            self.row_count,
            self.row_count,
            wp.mat33,
            device=self.device,
        )
        wps.bsr_set_from_triplets(
            self.lower_matrix,
            lower_rows,
            lower_columns,
            lower_initial_values,
            prune_numerical_zeros=False,
            topology="compact",
        )

        lower_offsets_np = self.lower_matrix.offsets.numpy()
        lower_columns_np = self.lower_matrix.columns.numpy()
        lower_slot_count = int(lower_offsets_np[-1])
        lower_rows_np = np.empty(lower_slot_count, dtype=np.int32)
        lower_slot_by_coordinate: dict[tuple[int, int], int] = {}
        lower_system_slots_np = np.empty(lower_slot_count, dtype=np.int32)
        for row in range(self.row_count):
            for slot in range(int(lower_offsets_np[row]), int(lower_offsets_np[row + 1])):
                column = int(lower_columns_np[slot])
                lower_rows_np[slot] = row
                lower_slot_by_coordinate[(row, column)] = slot
                lower_system_slots_np[slot] = system_slot_by_coordinate.get((row, column), -1)

        common_offsets_np = np.zeros(lower_slot_count + 1, dtype=np.int32)
        common_row_slots: list[int] = []
        common_column_slots: list[int] = []
        common_diagonals: list[int] = []
        for lower_slot in range(lower_slot_count):
            row = int(lower_rows_np[lower_slot])
            column = int(lower_columns_np[lower_slot])
            for row_neighbor_slot in range(int(lower_offsets_np[row]), int(lower_offsets_np[row + 1])):
                neighbor = int(lower_columns_np[row_neighbor_slot])
                if neighbor >= column:
                    break
                column_neighbor_slot = lower_slot_by_coordinate.get((column, neighbor))
                if column_neighbor_slot is not None:
                    common_row_slots.append(row_neighbor_slot)
                    common_column_slots.append(column_neighbor_slot)
                    common_diagonals.append(neighbor)
            common_offsets_np[lower_slot + 1] = len(common_row_slots)

        levels_np = np.zeros(self.row_count, dtype=np.int32)
        for row in range(self.row_count):
            row_columns = lower_columns_np[int(lower_offsets_np[row]) : int(lower_offsets_np[row + 1])]
            if row_columns.size > 0:
                levels_np[row] = 1 + int(np.max(levels_np[row_columns]))
        self.level_count = int(np.max(levels_np)) + 1
        level_rows_np = np.argsort(levels_np, kind="stable").astype(np.int32)
        level_counts = np.bincount(levels_np, minlength=self.level_count)
        level_offsets_np = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(level_counts, dtype=np.int32)))

        world_level_rows: list[int] = []
        world_level_offsets_np = np.zeros(
            world_active.shape[0] * (self.level_count + 1),
            dtype=np.int32,
        )
        max_world_level_width = 0
        for world in range(world_active.shape[0]):
            world_level_base = world * (self.level_count + 1)
            world_start = int(world_row_offsets_np[world])
            world_end = int(world_row_offsets_np[world + 1])
            world_level_offsets_np[world_level_base] = len(world_level_rows)
            for level in range(self.level_count):
                rows = np.flatnonzero(levels_np[world_start:world_end] == level) + world_start
                world_level_rows.extend(int(row) for row in rows)
                max_world_level_width = max(max_world_level_width, int(rows.shape[0]))
                world_level_offsets_np[world_level_base + level + 1] = len(world_level_rows)

        upper_rows: list[int] = []
        upper_lower_slots: list[int] = []
        upper_offsets_np = np.zeros(self.row_count + 1, dtype=np.int32)
        upper_by_row: list[list[tuple[int, int]]] = [[] for _ in range(self.row_count)]
        for lower_slot in range(lower_slot_count):
            source_row = int(lower_rows_np[lower_slot])
            column = int(lower_columns_np[lower_slot])
            upper_by_row[column].append((source_row, lower_slot))
        for row in range(self.row_count):
            for source_row, lower_slot in upper_by_row[row]:
                upper_rows.append(source_row)
                upper_lower_slots.append(lower_slot)
            upper_offsets_np[row + 1] = len(upper_rows)

        self.lower_system_slots = wp.array(lower_system_slots_np, dtype=wp.int32, device=self.device)
        self.common_offsets = wp.array(common_offsets_np, dtype=wp.int32, device=self.device)
        self.common_row_slots = wp.array(common_row_slots, dtype=wp.int32, device=self.device)
        self.common_column_slots = wp.array(common_column_slots, dtype=wp.int32, device=self.device)
        self.common_diagonals = wp.array(common_diagonals, dtype=wp.int32, device=self.device)
        self.system_diagonal_slots = wp.array(diagonal_system_slots, dtype=wp.int32, device=self.device)
        self.level_rows = wp.array(level_rows_np, dtype=wp.int32, device=self.device)
        self.level_offsets = tuple(int(value) for value in level_offsets_np)
        self.world_row_offsets = wp.array(world_row_offsets_np, dtype=wp.int32, device=self.device)
        self.world_level_offsets = wp.array(world_level_offsets_np, dtype=wp.int32, device=self.device)
        self.world_level_rows = wp.array(world_level_rows, dtype=wp.int32, device=self.device)
        self.upper_offsets = wp.array(upper_offsets_np, dtype=wp.int32, device=self.device)
        self.upper_rows = wp.array(upper_rows, dtype=wp.int32, device=self.device)
        self.upper_lower_slots = wp.array(upper_lower_slots, dtype=wp.int32, device=self.device)
        self.upper_values = wp.empty(len(upper_rows), dtype=wp.mat33, device=self.device)

        max_world_rows = int(np.max(np.diff(world_row_offsets_np), initial=0))
        self.uses_persistent_apply = self.device.is_cuda and max_world_rows <= _PERSISTENT_ROW_LIMIT
        self._persistent_block_dim = min(
            _PERSISTENT_BLOCK_DIM_LIMIT,
            max(32, 1 << max(0, max_world_level_width - 1).bit_length()),
        )

        self.diagonal_values = wp.empty(self.row_count, dtype=wp.mat33, device=self.device)
        self.inverse_diagonal_values = wp.empty(self.row_count, dtype=wp.mat33, device=self.device)
        self.world_status = wp.zeros(world_active.shape[0], dtype=wp.int32, device=self.device)
        self.forward_solution = wp.empty(self.row_count, dtype=wp.vec3, device=self.device)
        self.backward_solution = wp.empty(self.row_count, dtype=wp.vec3, device=self.device)

        self.linear_operator = LinearOperator(
            shape=system_matrix.shape,
            dtype=system_matrix.dtype,
            device=self.device,
            matvec=self._matvec,
            batch_offsets=batch_offsets,
        )

    def factorize(self) -> None:
        """Numerically factor the current system matrix once."""
        wp.launch(
            _begin_factorization,
            dim=self.world_status.shape[0],
            inputs=[self.world_status],
            device=self.device,
        )
        for level in range(self.level_count):
            level_start = self.level_offsets[level]
            level_end = self.level_offsets[level + 1]
            wp.launch(
                _factorize_level,
                dim=level_end - level_start,
                inputs=[
                    self.level_rows,
                    level_start,
                    self.lower_matrix.offsets,
                    self.lower_matrix.columns,
                    self.lower_system_slots,
                    self.common_offsets,
                    self.common_row_slots,
                    self.common_column_slots,
                    self.common_diagonals,
                    self.system_diagonal_slots,
                    self.system_matrix.values,
                    self.packed_world,
                    self.row_active,
                    self.regularization,
                ],
                outputs=[
                    self.lower_matrix.values,
                    self.diagonal_values,
                    self.inverse_diagonal_values,
                    self.world_status,
                ],
                device=self.device,
            )
        if self.upper_values.shape[0] > 0:
            wp.launch(
                _refresh_upper_values,
                dim=self.upper_values.shape[0],
                inputs=[self.upper_lower_slots, self.lower_matrix.values],
                outputs=[self.upper_values],
                device=self.device,
            )

    def _matvec(
        self,
        x: wp.array[wp.vec3],
        y: wp.array[wp.vec3],
        z: wp.array[wp.vec3],
        alpha: float,
        beta: float,
    ) -> None:
        if self.uses_persistent_apply:
            wp.launch(
                _persistent_apply,
                dim=self.world_active.shape[0] * self._persistent_block_dim,
                inputs=[
                    self.world_row_offsets,
                    self.world_level_offsets,
                    self.world_level_rows,
                    self.level_count,
                    self.lower_matrix.offsets,
                    self.lower_matrix.columns,
                    self.lower_matrix.values,
                    self.inverse_diagonal_values,
                    self.upper_offsets,
                    self.upper_rows,
                    self.upper_lower_slots,
                    x,
                    y,
                    self.world_active,
                    self.world_status,
                    self.row_active,
                    alpha,
                    beta,
                ],
                outputs=[self.forward_solution, z],
                block_dim=self._persistent_block_dim,
                device=self.device,
            )
            return

        for level in range(self.level_count):
            level_start = self.level_offsets[level]
            level_end = self.level_offsets[level + 1]
            wp.launch(
                _forward_level,
                dim=level_end - level_start,
                inputs=[
                    self.level_rows,
                    level_start,
                    self.lower_matrix.offsets,
                    self.lower_matrix.columns,
                    self.lower_matrix.values,
                    self.row_active,
                    x,
                ],
                outputs=[self.forward_solution],
                block_dim=_LEVEL_BLOCK_DIM,
                device=self.device,
            )
        for level in range(self.level_count - 1, -1, -1):
            level_start = self.level_offsets[level]
            level_end = self.level_offsets[level + 1]
            wp.launch(
                _backward_level,
                dim=level_end - level_start,
                inputs=[
                    self.level_rows,
                    level_start,
                    self.upper_offsets,
                    self.upper_rows,
                    self.upper_values,
                    self.inverse_diagonal_values,
                    self.forward_solution,
                    x,
                    y,
                    self.packed_world,
                    self.world_active,
                    self.world_status,
                    self.row_active,
                    alpha,
                    beta,
                ],
                outputs=[self.backward_solution, z],
                block_dim=_LEVEL_BLOCK_DIM,
                device=self.device,
            )
