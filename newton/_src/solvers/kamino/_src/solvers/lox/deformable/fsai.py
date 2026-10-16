# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Adaptive factorized sparse approximate inverse (FSAI) preconditioning of the LOX deformable system.

The preconditioner applies ``G^T G``, where ``G`` is a lower-triangular scalar factor with at
most ``support_size`` entries per row, built by :class:`warp.optim.linear.FSAI`. The support of
each row grows greedily from its diagonal on the graph of the diagonally equilibrated system
matrix. For a support ``S`` of row ``i``, the row solves ``A[S, S] z = e_i`` and stores
``z / sqrt(z_i)``. Every row is independent, so the setup, refits, and both sparse products are
parallel; no triangular solve is applied.

The supports are selected on the device at the first factorization, behind a ready flag, so
that the first factorization may be captured in a CUDA graph. Later factorizations refit the
values on these supports, since the system topology is fixed. A row whose local factorization
is poorly conditioned is regularized, or falls back to its scaled diagonal, without blocking
the refit of the other rows.

Warp provides :class:`warp.optim.linear.FSAI` from version 1.19; with older versions, this module
uses the vendored copy in :mod:`._warp_fsai`.

See Kolotilina and Yeremin, "Factorized sparse approximate inverse preconditionings I. Theory",
SIAM Journal on Matrix Analysis and Applications 14(1), 1993, and the adaptive FSAI of hypre,
https://hypre.readthedocs.io/en/latest/solvers-fsai.html, for the pattern growth.
"""

from __future__ import annotations

import warp as wp
import warp.sparse as wps
from warp.optim.linear import LinearOperator

try:
    from warp.optim.linear import FSAI
except ImportError:  # Warp < 1.19
    from ._warp_fsai import FSAI

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


###
# Kernels
###


@wp.kernel
def _mask_fsai_product(
    # Inputs:
    product: wp.array[wp.vec3],
    right_hand_side: wp.array[wp.vec3],
    addend: wp.array[wp.vec3],
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    row_active: wp.array[wp.int32],
    alpha: float,
    beta: float,
    # Outputs:
    result: wp.array[wp.vec3],
):
    """Write ``alpha G^T G x + beta y`` on the conjugate-residual rows of active worlds and ``alpha x + beta y`` elsewhere."""
    row = wp.tid()
    value = right_hand_side[row]
    if row_active[row] != 0 and world_active[packed_world[row]]:
        value = product[row]
    value *= alpha
    if beta != 0.0:
        value += beta * addend[row]
    result[row] = value


###
# Interfaces
###


class DeformableFSAI:
    """Apply an adaptive factorized sparse approximate inverse ``G^T G`` of the system matrix.

    Args:
        system_matrix: Symmetric 3-by-3 block system matrix with both triangles stored, in
            compact storage with a fixed topology.
        packed_world: World index for every packed block row.
        world_active: Mutable active flag for every world.
        batch_offsets: Scalar degree-of-freedom offsets for Warp batching.
        regularization: Smallest pivot of the diagonally equilibrated local Cholesky factors, in
            ``(0, 1)``; refits raise smaller pivots to it.
        row_active: Optional nonzero flag for rows assigned to CR. The preconditioner is the
            identity on the other rows, which share no matrix entries with CR rows.
        support_size: Largest scalar entry count of a factor row, including its diagonal.
            One gives scalar Jacobi.
    """

    def __init__(
        self,
        system_matrix: wps.BsrMatrix,
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
        self.device = system_matrix.device
        self.row_count = int(system_matrix.nrow)
        self.packed_world = packed_world
        self.world_active = world_active
        if row_active is None:
            row_active = wp.ones(self.row_count, dtype=wp.int32, device=self.device)
        elif row_active.shape != (self.row_count,) or row_active.dtype != wp.int32:
            raise ValueError("LOX deformable FSAI requires one int32 active flag per row.")
        self.row_active = row_active
        self.fsai = FSAI(
            system_matrix,
            max_row_size=support_size,
            pivot_floor=regularization,
            reuse_pattern=True,
            defer_pattern=True,
            regularize_refit=True,
            validate_topology=False,
        )
        """Warp FSAI factor of the system matrix, refit by :meth:`factorize`."""
        self.product = wp.zeros(self.row_count, dtype=wp.vec3, device=self.device)
        """Scratch product ``G^T G x`` of the last application."""
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

    @property
    def pattern_ready(self) -> wp.array[wp.int32]:
        """Nonzero once the supports are selected, which the first factorization does."""
        return self.fsai.pattern_ready

    @property
    def update_status(self) -> wp.array[wp.int32]:
        """Counts of the last factorization: topology mismatches, invalid diagonals, failed rows, regularized rows."""
        return self.fsai.update_status

    def factorize(self) -> None:
        """Refit the factor to the current system matrix, selecting the supports on the first call."""
        self.fsai.update()

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
        """Write ``z = alpha G^T G x + beta y``."""
        self.fsai.matvec(x, self.product, self.product, 1.0, 0.0)
        wp.launch(
            _mask_fsai_product,
            dim=self.row_count,
            inputs=[self.product, x, y, self.packed_world, self.world_active, self.row_active, alpha, beta],
            outputs=[z],
            device=self.device,
        )
