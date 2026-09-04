# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shallow two-level preconditioning for LOX deformables."""

from __future__ import annotations

import numpy as np
import warp as wp
import warp.sparse as wps
from warp.optim.linear import LinearOperator

from ...linalg import DenseLinearOperatorData, DenseSquareMultiLinearInfo, LLTBlockedSolver
from .deformable_jacobi import DeformableJacobi
from .deformable_preconditioner import DEFORMABLE_PRECONDITIONER_STATUS_FAILED

__all__ = ["DeformableTwoLevel"]

_AGGREGATE_PARTICLE_COUNT = 64
"""Target particle count for deterministic graph aggregates."""

_AGGREGATE_COUNT_LIMIT = 128
"""Largest coarse vertex count allocated for one iterative component."""

_COARSE_BLOCK_AGGREGATE_COUNT = 8
"""Maximum aggregate count in each default dense coarse block."""

_AGGREGATE_BLOCK_DIM = 64
"""Thread count used to restrict one graph aggregate."""


def _aggregate_component(
    component_rows: np.ndarray,
    adjacency_offsets: np.ndarray,
    adjacency_columns: np.ndarray,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flat aggregate particles and offsets in deterministic BFS order."""
    remaining = np.zeros(adjacency_offsets.size - 1, dtype=bool)
    remaining[component_rows] = True
    traversal = np.empty(component_rows.size, dtype=np.int32)
    queue = np.empty(component_rows.size, dtype=np.int32)
    traversal_end = 0
    for seed in component_rows:
        if not remaining[seed]:
            continue
        queue_begin = 0
        queue_end = 1
        queue[0] = seed
        remaining[seed] = False
        while queue_begin < queue_end:
            row = queue[queue_begin]
            queue_begin += 1
            traversal[traversal_end] = row
            traversal_end += 1
            neighbors = adjacency_columns[adjacency_offsets[row] : adjacency_offsets[row + 1]]
            new_neighbors = neighbors[remaining[neighbors]]
            remaining[new_neighbors] = False
            queue[queue_end : queue_end + new_neighbors.size] = new_neighbors
            queue_end += new_neighbors.size

    aggregate_count = (component_rows.size + target_size - 1) // target_size
    aggregate_offsets = np.minimum(
        np.arange(aggregate_count + 1, dtype=np.int32) * target_size,
        component_rows.size,
    )
    return traversal, aggregate_offsets


@wp.kernel
def _assemble_coarse_matrix(
    fine_values: wp.array[wp.mat33],
    coarse_slot_base: wp.array[wp.int32],
    coarse_slot_dimension: wp.array[wp.int32],
    coarse_matrix: wp.array[wp.float32],
):
    slot = wp.tid()
    base = coarse_slot_base[slot]
    if base < 0:
        return
    dimension = coarse_slot_dimension[slot]
    value = fine_values[slot]
    for row in range(3):
        for column in range(3):
            wp.atomic_add(coarse_matrix, base + row * dimension + column, value[row, column])


@wp.kernel
def _restrict_residual_by_aggregate(
    right_hand_side: wp.array[wp.vec3],
    aggregate_offsets: wp.array[wp.int32],
    aggregate_particles: wp.array[wp.int32],
    aggregate_coarse_offset: wp.array[wp.int32],
    aggregate_world: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    coarse_right_hand_side: wp.array[wp.float32],
):
    thread = wp.tid()
    aggregate = thread // _AGGREGATE_BLOCK_DIM
    lane = thread - aggregate * _AGGREGATE_BLOCK_DIM
    value = wp.vec3(0.0)
    world = aggregate_world[aggregate]
    if world_active[world] != 0 and world_status[world] != DEFORMABLE_PRECONDITIONER_STATUS_FAILED:
        slot = wp.int32(aggregate_offsets[aggregate] + lane)
        slot_end = aggregate_offsets[aggregate + 1]
        while slot < slot_end:
            value += right_hand_side[aggregate_particles[slot]]
            slot += _AGGREGATE_BLOCK_DIM
    value = wp.tile_reduce(wp.add, wp.tile(value, preserve_type=True))[0]
    if lane == 0:
        coarse_offset = aggregate_coarse_offset[aggregate]
        for axis in range(3):
            coarse_right_hand_side[coarse_offset + axis] = value[axis]


@wp.kernel
def _restrict_residual_by_aggregate_cpu(
    right_hand_side: wp.array[wp.vec3],
    aggregate_offsets: wp.array[wp.int32],
    aggregate_particles: wp.array[wp.int32],
    aggregate_coarse_offset: wp.array[wp.int32],
    aggregate_world: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    coarse_right_hand_side: wp.array[wp.float32],
):
    aggregate = wp.tid()
    value = wp.vec3(0.0)
    world = aggregate_world[aggregate]
    if world_active[world] != 0 and world_status[world] != DEFORMABLE_PRECONDITIONER_STATUS_FAILED:
        for slot in range(aggregate_offsets[aggregate], aggregate_offsets[aggregate + 1]):
            value += right_hand_side[aggregate_particles[slot]]
    coarse_offset = aggregate_coarse_offset[aggregate]
    for axis in range(3):
        coarse_right_hand_side[coarse_offset + axis] = value[axis]


@wp.kernel
def _apply_two_level(
    inverse_diagonal: wp.array[wp.mat33],
    coarse_solution: wp.array[wp.float32],
    particle_coarse_offset: wp.array[wp.int32],
    right_hand_side: wp.array[wp.vec3],
    addend: wp.array[wp.vec3],
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    alpha: float,
    beta: float,
    result: wp.array[wp.vec3],
):
    particle = wp.tid()
    coarse_offset = particle_coarse_offset[particle]
    world = packed_world[particle]
    value = right_hand_side[particle]
    if (
        coarse_offset >= 0
        and world_active[world] != 0
        and world_status[world] != DEFORMABLE_PRECONDITIONER_STATUS_FAILED
    ):
        value = inverse_diagonal[particle] * value
        value += wp.vec3(
            coarse_solution[coarse_offset],
            coarse_solution[coarse_offset + 1],
            coarse_solution[coarse_offset + 2],
        )
    value *= alpha
    if beta != 0.0:
        value += beta * addend[particle]
    result[particle] = value


class DeformableTwoLevel:
    """Apply block Jacobi plus block-diagonal aggregate coarse correction."""

    def __init__(
        self,
        system_matrix: wps.BsrMatrix,
        diagonal_slots: wp.array[wp.int32],
        packed_component: np.ndarray,
        packed_world: wp.array[wp.int32],
        world_active: wp.array[wp.int32],
        batch_offsets: wp.array[wp.int32],
        regularization: float = 1.0e-6,
        row_active: wp.array[wp.int32] | None = None,
    ):
        """Cache deterministic graph aggregates and dense coarse storage.

        Args:
            system_matrix: Symmetric 3-by-3 block system matrix.
            diagonal_slots: BSR value slot for every diagonal block.
            packed_component: Structural component for every packed particle.
            packed_world: World index for every packed particle.
            world_active: Mutable active flag for every world.
            batch_offsets: Scalar degree-of-freedom offsets for Warp batching.
            regularization: Relative positive fine-diagonal floor.
            row_active: Optional nonzero flag for rows assigned to CR.
        """
        if system_matrix.block_shape != (3, 3) or system_matrix.nrow != system_matrix.ncol:
            raise ValueError("LOX deformable two-level preconditioning requires a square 3-by-3 BSR matrix.")
        row_count = int(system_matrix.nrow)
        if packed_component.shape != (row_count,) or packed_component.dtype != np.int32:
            raise ValueError("LOX deformable two-level preconditioning requires one int32 component per row.")
        if row_active is None:
            row_active = wp.ones(row_count, dtype=wp.int32, device=system_matrix.device)
        elif row_active.shape != (row_count,) or row_active.dtype != wp.int32:
            raise ValueError("LOX deformable two-level preconditioning requires one int32 active flag per row.")
        self.system_matrix = system_matrix
        self.device = system_matrix.device
        self.row_count = row_count
        self.packed_world = packed_world
        self.world_active = world_active
        self.fine_preconditioner = DeformableJacobi(
            system_matrix,
            diagonal_slots,
            packed_world,
            world_active,
            batch_offsets,
            regularization=regularization,
            row_active=row_active,
        )
        self.world_status = self.fine_preconditioner.world_status

        offsets = system_matrix.offsets.numpy().astype(np.int32, copy=False)
        columns = system_matrix.columns.numpy().astype(np.int32, copy=False)
        slot_rows = np.repeat(np.arange(row_count, dtype=np.int32), np.diff(offsets))
        component = packed_component
        active = row_active.numpy().astype(bool, copy=False)
        if np.any(component[slot_rows] != component[columns]):
            raise ValueError("LOX deformable two-level matrix entries cannot span structural components.")

        adjacency_mask = (columns != slot_rows) & active[columns]
        adjacency_rows = slot_rows[adjacency_mask]
        adjacency_columns = columns[adjacency_mask]
        adjacency_keys = adjacency_rows.astype(np.int64) * row_count + adjacency_columns
        adjacency_order = np.argsort(adjacency_keys, kind="stable")
        adjacency_rows = adjacency_rows[adjacency_order]
        adjacency_columns = adjacency_columns[adjacency_order]
        adjacency_offsets = np.empty(row_count + 1, dtype=np.int32)
        adjacency_offsets[0] = 0
        np.cumsum(np.bincount(adjacency_rows, minlength=row_count), out=adjacency_offsets[1:])

        iterative_rows = np.flatnonzero(active & (component >= 0)).astype(np.int32)
        if iterative_rows.size == 0:
            raise ValueError("LOX deformable two-level preconditioning requires at least one iterative component.")
        _, component_inverse = np.unique(component[iterative_rows], return_inverse=True)
        component_particle_counts = np.bincount(component_inverse).astype(np.int32)
        target_sizes = np.maximum(
            _AGGREGATE_PARTICLE_COUNT,
            (component_particle_counts + _AGGREGATE_COUNT_LIMIT - 1) // _AGGREGATE_COUNT_LIMIT,
        )
        aggregate_counts = (component_particle_counts + target_sizes - 1) // target_sizes
        aggregate_count = int(np.sum(aggregate_counts))
        particle_aggregate = np.full(row_count, -1, dtype=np.int32)
        aggregate_offsets = np.empty(aggregate_count + 1, dtype=np.int32)
        aggregate_offsets[0] = 0
        aggregate_particles = np.empty(iterative_rows.size, dtype=np.int32)
        next_aggregate = 0
        next_particle = 0
        for component_position in range(component_particle_counts.size):
            rows = iterative_rows[component_inverse == component_position]
            traversal, local_offsets = _aggregate_component(
                rows,
                adjacency_offsets,
                adjacency_columns,
                int(target_sizes[component_position]),
            )
            component_aggregate_count = int(aggregate_counts[component_position])
            component_particle_count = int(component_particle_counts[component_position])
            aggregate_particles[next_particle : next_particle + component_particle_count] = traversal
            aggregate_offsets[next_aggregate + 1 : next_aggregate + component_aggregate_count + 1] = (
                next_particle + local_offsets[1:]
            )
            particle_aggregate[traversal] = np.repeat(
                np.arange(next_aggregate, next_aggregate + component_aggregate_count, dtype=np.int32),
                np.diff(local_offsets),
            )
            next_aggregate += component_aggregate_count
            next_particle += component_particle_count

        self.aggregate_count = aggregate_count
        slot_row_aggregate = particle_aggregate[slot_rows]
        slot_column_aggregate = particle_aggregate[columns]
        coarse_block_counts = np.concatenate(
            tuple(
                np.minimum(
                    _COARSE_BLOCK_AGGREGATE_COUNT,
                    count
                    - np.arange(
                        (count + _COARSE_BLOCK_AGGREGATE_COUNT - 1) // _COARSE_BLOCK_AGGREGATE_COUNT,
                        dtype=np.int32,
                    )
                    * _COARSE_BLOCK_AGGREGATE_COUNT,
                )
                for count in aggregate_counts
            )
        )
        coarse_block_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int32), np.cumsum(coarse_block_counts, dtype=np.int32))
        )
        aggregate_block = np.repeat(np.arange(coarse_block_counts.size, dtype=np.int32), coarse_block_counts)
        aggregate_local = np.arange(aggregate_count, dtype=np.int32)
        aggregate_local -= np.repeat(coarse_block_offsets[:-1], coarse_block_counts)
        info = DenseSquareMultiLinearInfo()
        info.finalize(
            dimensions=(3 * coarse_block_counts).tolist(),
            dtype=wp.float32,
            itype=wp.int32,
            device=self.device,
        )
        matrix_offsets = np.asarray(info.mio.numpy(), dtype=np.int64)
        vector_offsets = np.asarray(info.vio.numpy(), dtype=np.int64)

        particle_coarse_offset = np.full(row_count, -1, dtype=np.int32)
        particle_aggregates = particle_aggregate[iterative_rows]
        particle_blocks = aggregate_block[particle_aggregates]
        particle_coarse_offset[iterative_rows] = (
            vector_offsets[particle_blocks] + 3 * aggregate_local[particle_aggregates]
        )

        aggregate_first_particles = aggregate_particles[aggregate_offsets[:-1]]
        aggregate_coarse_offset = particle_coarse_offset[aggregate_first_particles]
        aggregate_world = packed_world.numpy().astype(np.int32, copy=False)[aggregate_first_particles]

        coarse_slot_base = np.full(columns.shape[0], -1, dtype=np.int32)
        coarse_slot_dimension = np.zeros(columns.shape[0], dtype=np.int32)
        valid_slot_mask = (slot_row_aggregate >= 0) & (slot_column_aggregate >= 0)
        valid_slots = np.flatnonzero(valid_slot_mask)
        valid_row_aggregates = slot_row_aggregate[valid_slots]
        valid_column_aggregates = slot_column_aggregate[valid_slots]
        valid_blocks = aggregate_block[valid_row_aggregates]
        retained_slot_mask = aggregate_block[valid_column_aggregates] == valid_blocks
        retained_slots = valid_slots[retained_slot_mask]
        retained_row_aggregates = valid_row_aggregates[retained_slot_mask]
        retained_column_aggregates = valid_column_aggregates[retained_slot_mask]
        retained_blocks = valid_blocks[retained_slot_mask]
        retained_dimensions = 3 * coarse_block_counts[retained_blocks]
        coarse_slot_base[retained_slots] = (
            matrix_offsets[retained_blocks]
            + 3 * aggregate_local[retained_row_aggregates] * retained_dimensions
            + 3 * aggregate_local[retained_column_aggregates]
        )
        coarse_slot_dimension[retained_slots] = retained_dimensions

        self.particle_coarse_offset = wp.array(particle_coarse_offset, dtype=wp.int32, device=self.device)
        self.aggregate_offsets = wp.array(aggregate_offsets, dtype=wp.int32, device=self.device)
        self.aggregate_particles = wp.array(aggregate_particles, dtype=wp.int32, device=self.device)
        self.aggregate_coarse_offset = wp.array(aggregate_coarse_offset, dtype=wp.int32, device=self.device)
        self.aggregate_world = wp.array(aggregate_world, dtype=wp.int32, device=self.device)
        self.coarse_slot_base = wp.array(coarse_slot_base, dtype=wp.int32, device=self.device)
        self.coarse_slot_dimension = wp.array(coarse_slot_dimension, dtype=wp.int32, device=self.device)
        self.coarse_matrix = wp.zeros(info.total_mat_size, dtype=wp.float32, device=self.device)
        self.coarse_right_hand_side = wp.zeros(info.total_vec_size, dtype=wp.float32, device=self.device)
        self.coarse_solution = wp.zeros(info.total_vec_size, dtype=wp.float32, device=self.device)
        coarse_operator = DenseLinearOperatorData(info=info, mat=self.coarse_matrix)
        self.coarse_solver = LLTBlockedSolver(
            operator=coarse_operator,
            factorize_block_size=24,
            solve_block_size=24,
            solve_block_dim=256,
            dtype=wp.float32,
            device=self.device,
        )
        self.linear_operator = LinearOperator(
            shape=system_matrix.shape,
            dtype=system_matrix.dtype,
            device=self.device,
            matvec=self._matvec,
            batch_offsets=batch_offsets,
        )

    def factorize(self) -> None:
        """Factor the fine diagonal and block-diagonal Galerkin coarse matrix."""
        self.fine_preconditioner.factorize()
        self.coarse_matrix.zero_()
        wp.launch(
            _assemble_coarse_matrix,
            dim=self.system_matrix.values.shape[0],
            inputs=[
                self.system_matrix.values,
                self.coarse_slot_base,
                self.coarse_slot_dimension,
            ],
            outputs=[self.coarse_matrix],
            device=self.device,
        )
        self.coarse_solver.compute(self.coarse_matrix)

    def _matvec(
        self,
        x: wp.array[wp.vec3],
        y: wp.array[wp.vec3],
        z: wp.array[wp.vec3],
        alpha: float,
        beta: float,
    ) -> None:
        if self.device.is_cuda:
            wp.launch(
                _restrict_residual_by_aggregate,
                dim=self.aggregate_count * _AGGREGATE_BLOCK_DIM,
                inputs=[
                    x,
                    self.aggregate_offsets,
                    self.aggregate_particles,
                    self.aggregate_coarse_offset,
                    self.aggregate_world,
                    self.world_active,
                    self.world_status,
                ],
                outputs=[self.coarse_right_hand_side],
                block_dim=_AGGREGATE_BLOCK_DIM,
                device=self.device,
            )
        else:
            wp.launch(
                _restrict_residual_by_aggregate_cpu,
                dim=self.aggregate_count,
                inputs=[
                    x,
                    self.aggregate_offsets,
                    self.aggregate_particles,
                    self.aggregate_coarse_offset,
                    self.aggregate_world,
                    self.world_active,
                    self.world_status,
                ],
                outputs=[self.coarse_right_hand_side],
                device=self.device,
            )
        self.coarse_solver.solve(self.coarse_right_hand_side, self.coarse_solution)
        wp.launch(
            _apply_two_level,
            dim=self.row_count,
            inputs=[
                self.fine_preconditioner.inverse_diagonal,
                self.coarse_solution,
                self.particle_coarse_offset,
                x,
                y,
                self.packed_world,
                self.world_active,
                self.world_status,
                alpha,
                beta,
            ],
            outputs=[z],
            device=self.device,
        )
