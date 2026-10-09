# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preallocated block-Jacobi preconditioning of the LOX deformable system.

Each ``3 x 3`` diagonal block is symmetrized and inverted through its eigendecomposition,
with its eigenvalues floored relative to its scale.
"""

from __future__ import annotations

import numpy as np
import warp as wp
import warp.sparse as wps
from warp.optim.linear import LinearOperator

from .preconditioner import _initialize_factorization_status, diagonal_mat33, is_finite_mat33, max_abs_mat33
from .types import DeformablePreconditionerStatus

###
# Module interface
###

__all__ = ["DeformableJacobi"]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _factorize_block_jacobi(
    # Inputs:
    diagonal_slots: wp.array[wp.int32],
    system_values: wp.array[wp.mat33],
    packed_world: wp.array[wp.int32],
    row_active: wp.array[wp.int32],
    regularization: float,
    # Outputs:
    inverse_diagonal: wp.array[wp.mat33],
    world_status: wp.array[wp.int32],
):
    """Invert the regularized diagonal block of each conjugate-residual row."""
    row = wp.tid()
    if row_active[row] == 0:
        inverse_diagonal[row] = wp.identity(n=3, dtype=float)
        return
    block = system_values[diagonal_slots[row]]
    block = 0.5 * (block + wp.transpose(block))
    status = wp.int32(DeformablePreconditionerStatus.VALID)
    if not is_finite_mat33(block):
        inverse_diagonal[row] = wp.identity(n=3, dtype=float)
        status = wp.int32(DeformablePreconditionerStatus.FAILED)
    else:
        eigenvectors, eigenvalues = wp.eig3(block)
        scale = wp.max(max_abs_mat33(block), 1.0e-12)
        eigenvalue_floor = wp.max(regularization * scale, 1.0e-12)
        finite_eigendecomposition = is_finite_mat33(eigenvectors)
        for axis in range(3):
            finite_eigendecomposition = finite_eigendecomposition and wp.isfinite(eigenvalues[axis])
        if not finite_eigendecomposition:
            inverse_diagonal[row] = (1.0 / scale) * wp.identity(n=3, dtype=float)
            status = wp.int32(DeformablePreconditionerStatus.FAILED)
        else:
            inverse_eigenvalues = wp.vec3(0.0)
            for axis in range(3):
                if eigenvalues[axis] < eigenvalue_floor:
                    eigenvalues[axis] = eigenvalue_floor
                    status = wp.int32(DeformablePreconditionerStatus.REGULARIZED)
                inverse_eigenvalues[axis] = 1.0 / eigenvalues[axis]
            inverse_diagonal[row] = eigenvectors * diagonal_mat33(inverse_eigenvalues) * wp.transpose(eigenvectors)
    if status != DeformablePreconditionerStatus.VALID:
        wp.atomic_max(world_status, packed_world[row], status)


@wp.kernel
def _apply_block_jacobi(
    # Inputs:
    inverse_diagonal: wp.array[wp.mat33],
    right_hand_side: wp.array[wp.vec3],
    addend: wp.array[wp.vec3],
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    row_active: wp.array[wp.int32],
    alpha: float,
    beta: float,
    # Outputs:
    result: wp.array[wp.vec3],
):
    """Write ``alpha D^-1 x + beta y`` on the conjugate-residual rows of active worlds."""
    row = wp.tid()
    wid = packed_world[row]
    value = right_hand_side[row]
    if row_active[row] != 0 and world_active[wid] and world_status[wid] != DeformablePreconditionerStatus.FAILED:
        value = inverse_diagonal[row] * value
    value *= alpha
    if beta != 0.0:
        value += beta * addend[row]
    result[row] = value


###
# Interfaces
###


class DeformableJacobi:
    """Apply a preallocated 3-by-3 block-Jacobi preconditioner.

    Args:
        system_matrix: Symmetric 3-by-3 block system matrix.
        diagonal_slots: BSR value slot for every diagonal block.
        packed_world: World index for every packed block row.
        world_active: Mutable active flag for every world.
        batch_offsets: Scalar degree-of-freedom offsets for Warp batching.
        regularization: Relative positive diagonal floor.
        row_active: Optional nonzero flag for rows assigned to CR.
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
    ):
        if system_matrix.block_shape != (3, 3) or system_matrix.nrow != system_matrix.ncol:
            raise ValueError("LOX cloth Jacobi requires a square 3-by-3 BSR matrix.")
        if diagonal_slots.shape != (system_matrix.nrow,) or diagonal_slots.dtype != wp.int32:
            raise ValueError("LOX cloth Jacobi requires one int32 diagonal slot per block row.")
        if not np.isfinite(regularization) or regularization <= 0.0:
            raise ValueError(f"LOX cloth Jacobi regularization must be finite and positive, got {regularization}.")
        self.system_matrix = system_matrix
        self.device = system_matrix.device
        self.row_count = int(system_matrix.nrow)
        self.diagonal_slots = diagonal_slots
        self.packed_world = packed_world
        self.world_active = world_active
        self.regularization = float(regularization)
        if row_active is None:
            row_active = wp.ones(self.row_count, dtype=wp.int32, device=self.device)
        elif row_active.shape != (self.row_count,) or row_active.dtype != wp.int32:
            raise ValueError("LOX cloth Jacobi requires one int32 active flag per row.")
        self.row_active = row_active
        self.world_status = wp.zeros(world_active.shape[0], dtype=wp.int32, device=self.device)
        """:class:`DeformablePreconditionerStatus` of each world after the last factorization."""
        self.inverse_diagonal = wp.empty(self.row_count, dtype=wp.mat33, device=self.device)
        """Inverse regularized diagonal block of each row."""
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
        """Extract the inverse diagonal of the current system matrix."""
        wp.launch(
            _initialize_factorization_status,
            dim=self.world_status.shape[0],
            outputs=[self.world_status],
            device=self.device,
        )
        wp.launch(
            _factorize_block_jacobi,
            dim=self.row_count,
            inputs=[
                self.diagonal_slots,
                self.system_matrix.values,
                self.packed_world,
                self.row_active,
                self.regularization,
            ],
            outputs=[self.inverse_diagonal, self.world_status],
            device=self.device,
        )

    ###
    # Internals
    ###

    def _matvec(
        self,
        x: wp.array[wp.vec3],
        y: wp.array[wp.vec3],
        z: wp.array[wp.vec3],
        alpha: float,
        beta: float,
    ) -> None:
        """Write ``z = alpha D^-1 x + beta y``."""
        wp.launch(
            _apply_block_jacobi,
            dim=self.row_count,
            inputs=[
                self.inverse_diagonal,
                x,
                y,
                self.packed_world,
                self.world_active,
                self.world_status,
                self.row_active,
                alpha,
                beta,
            ],
            outputs=[z],
            device=self.device,
        )
