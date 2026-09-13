# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""cuDSS direct solves for LOX deformables."""

from __future__ import annotations

import numpy as np
import warp as wp
import warp.sparse as wps

from .deformable_preconditioner import DEFORMABLE_PRECONDITIONER_STATUS_VALID

__all__ = ["DeformableCuDSS"]

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _pack_scalar_lower_values(
    block_values: wp.array[wp.mat33],
    scalar_block_slots: wp.array[wp.int32],
    scalar_block_rows: wp.array[wp.int32],
    scalar_block_columns: wp.array[wp.int32],
    scalar_values: wp.array[wp.float32],
):
    scalar_slot = wp.tid()
    block = block_values[scalar_block_slots[scalar_slot]]
    scalar_values[scalar_slot] = block[
        scalar_block_rows[scalar_slot],
        scalar_block_columns[scalar_slot],
    ]


@wp.kernel
def _copy_active_solution(
    source: wp.array[wp.vec3],
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    destination: wp.array[wp.vec3],
):
    particle = wp.tid()
    if world_active[packed_world[particle]] != 0:
        destination[particle] = source[particle]


def _build_scalar_lower_structure(
    block_offsets: np.ndarray,
    block_columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand a 3-by-3 BSR structure into lower-triangular scalar CSR."""
    block_row_count = block_offsets.shape[0] - 1
    scalar_offsets = [0]
    scalar_columns: list[int] = []
    scalar_block_slots: list[int] = []
    scalar_block_rows: list[int] = []
    scalar_block_columns: list[int] = []

    for block_row in range(block_row_count):
        row_start = int(block_offsets[block_row])
        row_end = int(block_offsets[block_row + 1])
        row_blocks = sorted(
            ((int(block_columns[block_slot]), block_slot) for block_slot in range(row_start, row_end)),
            key=lambda entry: entry[0],
        )
        for local_row in range(3):
            for block_column, block_slot in row_blocks:
                if block_column > block_row:
                    break
                local_column_end = local_row + 1 if block_column == block_row else 3
                for local_column in range(local_column_end):
                    scalar_columns.append(3 * block_column + local_column)
                    scalar_block_slots.append(block_slot)
                    scalar_block_rows.append(local_row)
                    scalar_block_columns.append(local_column)
            scalar_offsets.append(len(scalar_columns))

    return tuple(
        np.asarray(values, dtype=np.int32)
        for values in (
            scalar_offsets,
            scalar_columns,
            scalar_block_slots,
            scalar_block_rows,
            scalar_block_columns,
        )
    )


def _load_cudss_bindings():
    """Load the optional cuDSS bindings only when the backend is selected."""
    try:
        import nvmath  # noqa: PLC0415
        from nvmath.bindings import cudss  # noqa: PLC0415
        from nvmath.internal.memory import get_device_memory_resource  # noqa: PLC0415
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "LOX deformable cuDSS solves require nvmath-python and a matching cuDSS runtime. "
            "Install 'nvmath-python[cu12]' or 'nvmath-python[cu13]' for the CUDA "
            "major version in use."
        ) from error
    if not hasattr(cudss, "set_async_workspace_allocator"):
        raise RuntimeError("LOX deformable cuDSS solves require nvmath-python 1.0 or newer.")
    return nvmath, cudss, get_device_memory_resource


class DeformableCuDSS:
    """Reuse one cuDSS analysis and numerical factorization across candidate solves."""

    def __init__(
        self,
        system_matrix: wps.BsrMatrix,
        packed_world: wp.array[wp.int32],
        world_active: wp.array[wp.int32],
    ):
        """Create persistent cuDSS descriptors and analyze the fixed sparsity.

        Args:
            system_matrix: Symmetric 3-by-3 block deformable matrix.
            packed_world: World for every packed particle.
            world_active: Mutable active flag for every world.
        """
        if system_matrix.block_shape != (3, 3) or system_matrix.nrow != system_matrix.ncol:
            raise ValueError("LOX deformable cuDSS requires a square 3-by-3 BSR matrix.")
        if not system_matrix.device.is_cuda:
            raise ValueError("LOX deformable cuDSS requires a CUDA device.")
        if packed_world.shape != (system_matrix.nrow,) or packed_world.dtype != wp.int32:
            raise ValueError("LOX deformable cuDSS requires one int32 world per particle.")
        if world_active.ndim != 1 or world_active.dtype != wp.int32:
            raise ValueError("LOX deformable cuDSS requires a one-dimensional int32 active-world mask.")
        if packed_world.device != system_matrix.device or world_active.device != system_matrix.device:
            raise ValueError(f"LOX deformable cuDSS metadata must reside on {system_matrix.device}.")

        self.system_matrix = system_matrix
        self.device = system_matrix.device
        self.row_count = int(system_matrix.nrow)
        self.packed_world = packed_world
        self.world_active = world_active
        self._nvmath, self._cudss, self._get_device_memory_resource = _load_cudss_bindings()

        structure = _build_scalar_lower_structure(
            system_matrix.offsets.numpy().astype(np.int32, copy=False),
            system_matrix.columns.numpy().astype(np.int32, copy=False),
        )
        offsets_np, columns_np, block_slots_np, block_rows_np, block_columns_np = structure
        self.scalar_offsets = wp.array(offsets_np, dtype=wp.int32, device=self.device)
        self.scalar_columns = wp.array(columns_np, dtype=wp.int32, device=self.device)
        self.scalar_block_slots = wp.array(block_slots_np, dtype=wp.int32, device=self.device)
        self.scalar_block_rows = wp.array(block_rows_np, dtype=wp.int32, device=self.device)
        self.scalar_block_columns = wp.array(block_columns_np, dtype=wp.int32, device=self.device)
        scalar_rows_np = np.repeat(
            np.arange(3 * self.row_count, dtype=np.int32),
            np.diff(offsets_np),
        )
        analysis_values_np = (scalar_rows_np == columns_np).astype(np.float32)
        self.scalar_values = wp.array(analysis_values_np, dtype=wp.float32, device=self.device)
        self.solution = wp.empty(self.row_count, dtype=wp.vec3, device=self.device)
        self.world_status = wp.full(
            world_active.shape[0],
            DEFORMABLE_PRECONDITIONER_STATUS_VALID,
            dtype=wp.int32,
            device=self.device,
        )

        self._handle = 0
        self._config = 0
        self._data = 0
        self._matrix = 0
        self._solution = 0
        self._right_hand_side = 0
        self._workspace_pool = None
        self._async_workspace_allocator = False
        try:
            self._create_descriptors()
            self._execute(self._cudss.Phase.ANALYSIS)
        except Exception:
            self.close()
            raise

    def _create_descriptors(self) -> None:
        cudss = self._cudss
        value_type = self._nvmath.CudaDataType.CUDA_R_32F
        index_type = self._nvmath.CudaDataType.CUDA_R_32I
        dimension = 3 * self.row_count
        with wp.ScopedDevice(self.device):
            self._handle = cudss.create()
            if self.device.is_mempool_supported:
                self._workspace_pool = self._get_device_memory_resource(self.device.ordinal)
                self._async_workspace_allocator = cudss.set_async_workspace_allocator(
                    self._handle,
                    int(self._workspace_pool.handle),
                )
            self._config = cudss.config_create()
            self._data = cudss.data_create(self._handle)
            self._matrix = cudss.matrix_create_csr(
                dimension,
                dimension,
                self.scalar_values.shape[0],
                self.scalar_offsets.ptr,
                0,
                self.scalar_columns.ptr,
                self.scalar_values.ptr,
                index_type,
                index_type,
                value_type,
                cudss.MatrixType.SPD,
                cudss.MatrixViewType.LOWER,
                cudss.IndexBase.ZERO,
            )
            self._solution = cudss.matrix_create_dn(
                dimension,
                1,
                dimension,
                self.solution.ptr,
                value_type,
                cudss.Layout.COL_MAJOR,
            )
            self._right_hand_side = cudss.matrix_create_dn(
                dimension,
                1,
                dimension,
                self.solution.ptr,
                value_type,
                cudss.Layout.COL_MAJOR,
            )

    def _execute(self, phase) -> None:
        if self.device.is_capturing:
            if phase == self._cudss.Phase.ANALYSIS:
                raise RuntimeError("LOX deformable cuDSS analysis cannot execute during CUDA graph capture.")
            if not self._async_workspace_allocator:
                raise RuntimeError(
                    "LOX deformable cuDSS factorization and solve capture require CUDA memory-pool support."
                )
        with wp.ScopedDevice(self.device):
            self._cudss.set_stream(self._handle, self.device.stream.cuda_stream or 0)
            self._cudss.execute(
                self._handle,
                phase,
                self._config,
                self._data,
                self._matrix,
                self._solution,
                self._right_hand_side,
            )

    def factorize(self) -> None:
        """Pack and numerically factor the current matrix values."""
        wp.launch(
            _pack_scalar_lower_values,
            dim=self.scalar_values.shape[0],
            inputs=[
                self.system_matrix.values,
                self.scalar_block_slots,
                self.scalar_block_rows,
                self.scalar_block_columns,
            ],
            outputs=[self.scalar_values],
            device=self.device,
        )
        with wp.ScopedDevice(self.device):
            self._cudss.matrix_set_values(self._matrix, self.scalar_values.ptr)
        self._execute(self._cudss.Phase.FACTORIZATION)

    def solve(self, right_hand_side: wp.array[wp.vec3], solution: wp.array[wp.vec3]) -> None:
        """Solve with the retained factorization and copy active-world results."""
        if right_hand_side.shape != (self.row_count,) or right_hand_side.dtype != wp.vec3:
            raise ValueError("LOX deformable cuDSS right-hand side must contain one vec3 per particle.")
        if solution.shape != (self.row_count,) or solution.dtype != wp.vec3:
            raise ValueError("LOX deformable cuDSS solution must contain one vec3 per particle.")
        if right_hand_side.device != self.device or solution.device != self.device:
            raise ValueError(f"LOX deformable cuDSS vectors must reside on {self.device}.")
        with wp.ScopedDevice(self.device):
            self._cudss.matrix_set_values(self._right_hand_side, right_hand_side.ptr)
        self._execute(self._cudss.Phase.SOLVE)
        wp.launch(
            _copy_active_solution,
            dim=self.row_count,
            inputs=[self.solution, self.packed_world, self.world_active],
            outputs=[solution],
            device=self.device,
        )

    def close(self) -> None:
        """Release the cuDSS descriptors owned by this solver."""
        cudss = getattr(self, "_cudss", None)
        if cudss is None:
            return
        with wp.ScopedDevice(self.device):
            for name in ("_right_hand_side", "_solution", "_matrix"):
                descriptor = getattr(self, name, 0)
                if descriptor:
                    cudss.matrix_destroy(descriptor)
                    setattr(self, name, 0)
            if self._data:
                cudss.data_destroy(self._handle, self._data)
                self._data = 0
            if self._config:
                cudss.config_destroy(self._config)
                self._config = 0
            if self._handle:
                cudss.destroy(self._handle)
                self._handle = 0
            self._workspace_pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
