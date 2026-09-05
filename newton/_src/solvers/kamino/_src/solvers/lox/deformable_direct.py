# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Blocked direct solves for disconnected LOX deformables."""

from __future__ import annotations

import numpy as np
import warp as wp
import warp.sparse as wps

from ...linalg import DenseLinearOperatorData, DenseSquareMultiLinearInfo, HybridLLTBlockedSolver
from .deformable_preconditioner import DEFORMABLE_PRECONDITIONER_STATUS_VALID

__all__ = ["DeformableBlockLLT"]

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _pack_component_matrix(
    matrix_slot_row: wp.array[wp.int32],
    matrix_slot_column: wp.array[wp.int32],
    matrix_system_slot: wp.array[wp.int32],
    matrix_values: wp.array[wp.mat33],
    particle_direct_block: wp.array[wp.int32],
    particle_component_local: wp.array[wp.int32],
    component_dimensions: wp.array[wp.int32],
    component_matrix_offsets: wp.array[wp.int32],
    component_matrix: wp.array[wp.float32],
):
    slot = wp.tid()
    row = matrix_slot_row[slot]
    column = matrix_slot_column[slot]
    component = particle_direct_block[row]
    dimension = component_dimensions[component]
    matrix_offset = component_matrix_offsets[component]
    row_offset = 3 * particle_component_local[row]
    column_offset = 3 * particle_component_local[column]
    value = matrix_values[matrix_system_slot[slot]]
    for local_row in range(3):
        for local_column in range(3):
            dense_index = matrix_offset + (row_offset + local_row) * dimension + column_offset + local_column
            component_matrix[dense_index] = value[local_row, local_column]


@wp.kernel
def _pack_component_vector(
    value: wp.array[wp.vec3],
    particle_direct_block: wp.array[wp.int32],
    particle_component_local: wp.array[wp.int32],
    component_vector_offsets: wp.array[wp.int32],
    packed_value: wp.array[wp.float32],
):
    particle = wp.tid()
    component = particle_direct_block[particle]
    if component < 0:
        return
    vector_offset = component_vector_offsets[component]
    particle_offset = vector_offset + 3 * particle_component_local[particle]
    particle_value = value[particle]
    for axis in range(3):
        packed_value[particle_offset + axis] = particle_value[axis]


@wp.kernel
def _unpack_component_vector(
    packed_value: wp.array[wp.float32],
    particle_direct_block: wp.array[wp.int32],
    particle_component_local: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    component_vector_offsets: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    value: wp.array[wp.vec3],
):
    particle = wp.tid()
    component = particle_direct_block[particle]
    if component < 0:
        return
    if world_active[packed_world[particle]] == 0:
        return
    vector_offset = component_vector_offsets[component]
    particle_offset = vector_offset + 3 * particle_component_local[particle]
    value[particle] = wp.vec3(
        packed_value[particle_offset],
        packed_value[particle_offset + 1],
        packed_value[particle_offset + 2],
    )


class DeformableBlockLLT:
    """Reuse the rigid-body blocked LLT solver over deformable components."""

    def __init__(
        self,
        system_matrix: wps.BsrMatrix,
        packed_component: np.ndarray,
        packed_iterative: np.ndarray,
        packed_world: wp.array[wp.int32],
        world_active: wp.array[wp.int32],
        component_count: int,
    ):
        """Allocate dense factor blocks for immutable structural components.

        Args:
            system_matrix: Symmetric 3-by-3 block deformable matrix.
            packed_component: Structural component for every packed particle.
            packed_iterative: Nonzero for particles assigned to CR.
            packed_world: World for every packed particle.
            world_active: Mutable active flag for every world.
            component_count: Number of structural components.
        """
        if system_matrix.block_shape != (3, 3) or system_matrix.nrow != system_matrix.ncol:
            raise ValueError("LOX deformable block LLT requires a square 3-by-3 BSR matrix.")
        if packed_component.shape != (system_matrix.nrow,) or packed_component.dtype != np.int32:
            raise ValueError("LOX deformable block LLT requires one int32 component per particle.")
        if packed_iterative.shape != (system_matrix.nrow,) or packed_iterative.dtype != np.int32:
            raise ValueError("LOX deformable block LLT requires one int32 solve mode per particle.")
        if packed_world.shape != (system_matrix.nrow,) or packed_world.dtype != wp.int32:
            raise ValueError("LOX deformable block LLT requires one int32 world per particle.")
        if not isinstance(component_count, int) or isinstance(component_count, bool) or component_count < 1:
            raise ValueError("LOX deformable block LLT requires at least one component.")

        self.system_matrix = system_matrix
        self.device = system_matrix.device
        self.row_count = int(system_matrix.nrow)
        self.packed_world = packed_world
        self.world_active = world_active

        packed_component_np = packed_component
        packed_iterative_np = packed_iterative
        if np.any(packed_component_np < 0) or np.any(packed_component_np >= component_count):
            raise ValueError("LOX deformable block LLT component indices are out of range.")
        all_component_particle_counts = np.bincount(packed_component_np, minlength=component_count)
        if np.any(all_component_particle_counts == 0):
            raise ValueError("LOX deformable block LLT components must not be empty.")
        component_iterative = np.zeros(component_count, dtype=bool)
        for component in range(component_count):
            values = packed_iterative_np[packed_component_np == component]
            if np.any(values != values[0]):
                raise ValueError("LOX deformable block LLT solve modes must be constant within each component.")
            component_iterative[component] = values[0] != 0
        direct_components = np.flatnonzero(~component_iterative).astype(np.int32)
        if direct_components.size == 0:
            raise ValueError("LOX deformable block LLT requires at least one direct component.")
        component_to_direct = np.full(component_count, -1, dtype=np.int32)
        component_to_direct[direct_components] = np.arange(direct_components.size, dtype=np.int32)
        particle_direct_block_np = component_to_direct[packed_component_np]
        component_particle_counts = all_component_particle_counts[direct_components]

        component_local_np = np.empty(self.row_count, dtype=np.int32)
        next_local = np.zeros(direct_components.size, dtype=np.int32)
        for particle, component in enumerate(particle_direct_block_np):
            if component >= 0:
                component_local_np[particle] = next_local[component]
                next_local[component] += 1
            else:
                component_local_np[particle] = -1

        matrix_offsets_np = system_matrix.offsets.numpy().astype(np.int32, copy=False)
        matrix_columns_np = system_matrix.columns.numpy().astype(np.int32, copy=False)
        matrix_slot_rows_np = np.repeat(
            np.arange(self.row_count, dtype=np.int32),
            np.diff(matrix_offsets_np),
        )
        if np.any(packed_component_np[matrix_slot_rows_np] != packed_component_np[matrix_columns_np]):
            raise ValueError("LOX deformable block LLT matrix entries cannot span structural components.")
        direct_slot_mask = particle_direct_block_np[matrix_slot_rows_np] >= 0
        direct_system_slots_np = np.flatnonzero(direct_slot_mask).astype(np.int32)
        direct_matrix_rows_np = matrix_slot_rows_np[direct_slot_mask]
        direct_matrix_columns_np = matrix_columns_np[direct_slot_mask]

        self.info = DenseSquareMultiLinearInfo()
        self.info.finalize(
            dimensions=[3 * int(count) for count in component_particle_counts],
            dtype=wp.float32,
            itype=wp.int32,
            device=self.device,
        )
        self.matrix = wp.zeros(self.info.total_mat_size, dtype=wp.float32, device=self.device)
        self.right_hand_side = wp.empty(self.info.total_vec_size, dtype=wp.float32, device=self.device)
        self.solution = wp.empty(self.info.total_vec_size, dtype=wp.float32, device=self.device)
        self.matrix_slot_row = wp.array(direct_matrix_rows_np, dtype=wp.int32, device=self.device)
        self.matrix_slot_column = wp.array(direct_matrix_columns_np, dtype=wp.int32, device=self.device)
        self.matrix_system_slot = wp.array(direct_system_slots_np, dtype=wp.int32, device=self.device)
        self.particle_direct_block = wp.array(particle_direct_block_np, dtype=wp.int32, device=self.device)
        self.particle_component_local = wp.array(component_local_np, dtype=wp.int32, device=self.device)

        operator = DenseLinearOperatorData(info=self.info, mat=self.matrix)
        self.linear_solver = HybridLLTBlockedSolver(
            operator=operator,
            factorize_block_size=64,
            solve_block_dim=256,
            dtype=wp.float32,
            device=self.device,
        )
        self.world_status = wp.full(
            world_active.shape[0],
            DEFORMABLE_PRECONDITIONER_STATUS_VALID,
            dtype=wp.int32,
            device=self.device,
        )

    def factorize(self) -> None:
        """Pack and factor the current component matrices once."""
        wp.launch(
            _pack_component_matrix,
            dim=self.matrix_system_slot.shape[0],
            inputs=[
                self.matrix_slot_row,
                self.matrix_slot_column,
                self.matrix_system_slot,
                self.system_matrix.values,
                self.particle_direct_block,
                self.particle_component_local,
                self.info.dim,
                self.info.mio,
            ],
            outputs=[self.matrix],
            device=self.device,
        )
        self.linear_solver.compute(self.matrix)

    def solve(self, right_hand_side: wp.array[wp.vec3], solution: wp.array[wp.vec3]) -> None:
        """Solve all component blocks and update active worlds."""
        if right_hand_side.shape != (self.row_count,) or right_hand_side.dtype != wp.vec3:
            raise ValueError("LOX deformable block LLT right-hand side must contain one vec3 per particle.")
        if solution.shape != (self.row_count,) or solution.dtype != wp.vec3:
            raise ValueError("LOX deformable block LLT solution must contain one vec3 per particle.")
        wp.launch(
            _pack_component_vector,
            dim=self.row_count,
            inputs=[
                right_hand_side,
                self.particle_direct_block,
                self.particle_component_local,
                self.info.vio,
            ],
            outputs=[self.right_hand_side],
            device=self.device,
        )
        self.linear_solver.solve(self.right_hand_side, self.solution)
        wp.launch(
            _unpack_component_vector,
            dim=self.row_count,
            inputs=[
                self.solution,
                self.particle_direct_block,
                self.particle_component_local,
                self.packed_world,
                self.info.vio,
                self.world_active,
            ],
            outputs=[solution],
            device=self.device,
        )
