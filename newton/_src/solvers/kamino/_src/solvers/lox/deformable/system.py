# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Static topology and frozen smooth-system assembly of the LOX deformables.

Each particle contributes three velocity unknowns. Particles are packed by solve world,
then by structural component (the particles connected by triangles, bending edges, or
tetrahedra), then by Newton index, so that every component is one contiguous batch of the
conjugate-residual candidate solve. Small components may be solved by dense ``LLT``
factorization instead, and every component by cuDSS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp
import warp.sparse as wps

# Private Warp APIs: the scalar views of vector arrays for the masked block product, and the
# conjugate-residual internals reused by the recycled solve.
from warp._src import sparse as _wps_internal
from warp._src.optim import linear as _wpl_internal
from warp.optim import linear as wpl

from .......core.types import override
from .......geometry import ParticleFlags
from .consensus_kernels import (
    _assemble_consensus_patch_rows,
    _eliminate_consensus_patch,
    _finish_consensus_weight,
    _mark_dynamic_particles,
    _select_consensus_patch,
    _sum_chunk_mass,
    _sum_component_mass,
)
from .cudss import DeformableCuDSS
from .direct import DeformableBlockLLT
from .fsai import DeformableFSAI
from .jacobi import DeformableJacobi
from .preconditioner import DeformableIncompleteLDLT
from .proximal import DeformableMembraneProximal
from .system_kernels import (
    _apply_recycled_cr_basis,
    _apply_system_matrix,
    _assemble_bending_system,
    _assemble_consensus_weight,
    _assemble_particle_mass,
    _assemble_tetrahedron_system,
    _assemble_triangle_system,
    _compute_candidate_linear_residual,
    _compute_candidate_linear_residual_and_advance_recycling,
    _compute_recycled_cr_coefficients,
    _finalize_smooth_rhs,
    _gather_particle_state,
    _prepare_candidate_rhs,
    _prepare_particle_linearization,
    _reduce_recycled_cr_basis,
    _retain_direct_candidate_rhs,
    _rollback_unsafe_recycled_cr_projection,
    _select_consensus_weight,
    _validate_and_restart_recycled_cr_projection,
)
from .tetrahedron_proximal import DeformableTetrahedronProximal
from .two_level import DeformableMultilevel, DeformableTwoLevel
from .types import (
    PATCH_PARTICLE_COUNT,
    DeformableElementIncidence,
    DeformableRecyclingData,
    DeformableSystemData,
    DeformableTopology,
)

if TYPE_CHECKING:
    from .......sim import Model, State

###
# Module interface
###

__all__ = [
    "DeformableSystem",
    "validate_deformable_model",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


_SYSTEM_MATVEC_BLOCK_DIM = 64
"""Thread block size of the masked block sparse system product."""

_COMPONENT_MASS_BLOCK_DIM = 64
"""Thread block size of the component chunk mass sum, one block per chunk."""


###
# Functions
###


def _require_array(model: Model, name: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    """Return a Newton model array on the host after checking its presence and shape."""
    value = getattr(model, name, None)
    if value is None:
        raise ValueError(f"LOX deformables require Model.{name}.")
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"LOX deformables expected Model.{name} shape {expected_shape}, found {value.shape}.")
    return value.numpy()


def _particle_solve_worlds(model: Model, particle_source_world: np.ndarray) -> np.ndarray:
    """Map Newton particle ownership to nonnegative LOX solve partitions."""
    world_count = int(model.world_count)
    if world_count < 1:
        raise ValueError("LOX deformables require at least one Newton world.")
    if np.any(particle_source_world < -1) or np.any(particle_source_world >= world_count):
        raise ValueError("LOX deformable particle world indices must be -1 or a valid Newton world index.")
    particle_solve_world = particle_source_world
    if np.any(particle_source_world == -1):
        if world_count != 1:
            raise ValueError("LOX deformables do not support global particles shared by multiple worlds.")
        particle_solve_world = particle_source_world.copy()
        particle_solve_world[particle_source_world == -1] = 0
    return particle_solve_world.astype(np.int32, copy=False)


def _deformable_particle_components(
    particle_count: int,
    particle_world: np.ndarray,
    triangle_indices: np.ndarray,
    bending_indices: np.ndarray,
    tetrahedron_indices: np.ndarray,
) -> np.ndarray:
    """Return deterministic structural-component labels."""
    parents = np.arange(particle_count, dtype=np.int32)

    def find(particle: int) -> int:
        root = particle
        while parents[root] != root:
            root = int(parents[root])
        while parents[particle] != particle:
            parent = int(parents[particle])
            parents[particle] = root
            particle = parent
        return root

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        lower = min(first_root, second_root)
        higher = max(first_root, second_root)
        parents[higher] = lower

    for elements in (triangle_indices, bending_indices, tetrahedron_indices):
        for vertices in elements:
            first = int(vertices[0])
            for vertex in vertices[1:]:
                union(first, int(vertex))

    roots = np.asarray([find(particle) for particle in range(particle_count)], dtype=np.int32)
    unique_roots = sorted({int(root) for root in roots}, key=lambda root: (int(particle_world[root]), root))
    root_to_component = {root: component for component, root in enumerate(unique_roots)}
    return np.asarray([root_to_component[int(root)] for root in roots], dtype=np.int32)


def validate_deformable_model(model: Model) -> None:
    """Validate the Newton subset supported by the LOX deformable path.

    :class:`SolverKamino` applies this validation when LOX is selected for a
    model containing particles.

    Args:
        model: Newton model to validate.

    Raises:
        ValueError: If the model contains unsupported or inconsistent deformable data.
    """
    if model.particle_count <= 0:
        raise ValueError("LOX deformables require at least one particle.")
    if model.tri_count <= 0 and model.tet_count <= 0:
        raise ValueError("LOX deformables require at least one triangle or tetrahedron.")

    unsupported_counts = {
        "springs": model.spring_count,
        "muscles": model.muscle_count,
    }
    unsupported = [f"{name} (found {count})" for name, count in unsupported_counts.items() if count > 0]
    if unsupported:
        raise ValueError("LOX deformables do not support " + ", ".join(unsupported) + ".")

    particle_count = int(model.particle_count)
    triangle_count = int(model.tri_count)
    tetrahedron_count = int(model.tet_count)
    edge_count = int(model.edge_count)

    particle_mass = _require_array(model, "particle_mass", (particle_count,))
    particle_flags = _require_array(model, "particle_flags", (particle_count,))
    particle_source_world = _require_array(model, "particle_world", (particle_count,))
    _particle_solve_worlds(model, particle_source_world)

    if not np.all(np.isfinite(particle_mass)) or np.any(particle_mass < 0.0):
        raise ValueError("LOX deformable particle masses must be finite and non-negative.")
    if np.any((particle_flags & int(ParticleFlags.PROXY)) != 0):
        raise ValueError("LOX deformables do not support solver-coupling proxy particles.")
    triangle_indices = np.empty((0, 3), dtype=np.int32)
    if triangle_count > 0:
        triangle_indices = _require_array(model, "tri_indices", (triangle_count, 3))
        triangle_poses = _require_array(model, "tri_poses", (triangle_count,))
        triangle_areas = _require_array(model, "tri_areas", (triangle_count,))
        triangle_materials = _require_array(model, "tri_materials", (triangle_count, 5))
        triangle_activations = _require_array(model, "tri_activations", (triangle_count,))

        if np.any(triangle_indices < 0) or np.any(triangle_indices >= particle_count):
            raise ValueError("LOX deformable triangle indices must reference valid particles.")
        if np.any(
            (triangle_indices[:, 0] == triangle_indices[:, 1])
            | (triangle_indices[:, 0] == triangle_indices[:, 2])
            | (triangle_indices[:, 1] == triangle_indices[:, 2])
        ):
            raise ValueError("LOX deformable triangles must reference three distinct particles.")
        triangle_world = particle_source_world[triangle_indices]
        if np.any(triangle_world != triangle_world[:, :1]):
            raise ValueError("LOX deformable triangles cannot span Newton worlds.")
        if not np.all(np.isfinite(triangle_areas)) or np.any(triangle_areas <= 0.0):
            raise ValueError("LOX deformable triangle rest areas must be finite and positive.")
        if not np.all(np.isfinite(triangle_poses)):
            raise ValueError("LOX deformable triangle rest poses must be finite.")
        if np.any(np.abs(np.linalg.det(triangle_poses)) <= 1.0e-12):
            raise ValueError("LOX deformable triangle rest poses must be non-singular.")
        if not np.all(np.isfinite(triangle_materials)) or np.any(triangle_materials < 0.0):
            raise ValueError("LOX deformable triangle material coefficients must be finite and non-negative.")
        if not np.all(np.isfinite(triangle_activations)):
            raise ValueError("LOX deformable triangle activations must be finite.")

    tetrahedron_indices = np.empty((0, 4), dtype=np.int32)
    if tetrahedron_count > 0:
        tetrahedron_indices = _require_array(model, "tet_indices", (tetrahedron_count, 4))
        tetrahedron_poses = _require_array(model, "tet_poses", (tetrahedron_count,))
        tetrahedron_materials = _require_array(model, "tet_materials", (tetrahedron_count, 3))
        tetrahedron_activations = _require_array(model, "tet_activations", (tetrahedron_count,))

        if np.any(tetrahedron_indices < 0) or np.any(tetrahedron_indices >= particle_count):
            raise ValueError("LOX deformable tetrahedron indices must reference valid particles.")
        for vertices in tetrahedron_indices:
            if np.unique(vertices).size != 4:
                raise ValueError("LOX deformable tetrahedra must reference four distinct particles.")
        tetrahedron_world = particle_source_world[tetrahedron_indices]
        if np.any(tetrahedron_world != tetrahedron_world[:, :1]):
            raise ValueError("LOX deformable tetrahedra cannot span Newton worlds.")
        if not np.all(np.isfinite(tetrahedron_poses)):
            raise ValueError("LOX deformable tetrahedron rest poses must be finite.")
        pose_determinants = np.linalg.det(tetrahedron_poses)
        if not np.all(np.isfinite(pose_determinants)) or np.any(pose_determinants <= 0.0):
            raise ValueError("LOX deformable tetrahedron rest poses must have finite positive determinants.")
        if not np.all(np.isfinite(tetrahedron_materials)) or np.any(tetrahedron_materials < 0.0):
            raise ValueError("LOX deformable tetrahedron material coefficients must be finite and non-negative.")
        if not np.all(np.isfinite(tetrahedron_activations)):
            raise ValueError("LOX deformable tetrahedron activations must be finite.")

    attached = np.zeros(particle_count, dtype=bool)
    attached[triangle_indices.reshape(-1)] = True
    attached[tetrahedron_indices.reshape(-1)] = True
    if not np.all(attached):
        unattached_count = int(np.count_nonzero(~attached))
        raise ValueError(f"LOX deformables do not support unattached particles (found {unattached_count}).")

    if edge_count > 0:
        edge_indices = _require_array(model, "edge_indices", (edge_count, 4))
        edge_rest_angle = _require_array(model, "edge_rest_angle", (edge_count,))
        edge_rest_length = _require_array(model, "edge_rest_length", (edge_count,))
        edge_properties = _require_array(model, "edge_bending_properties", (edge_count, 2))

        if np.any(edge_indices < -1) or np.any(edge_indices >= particle_count):
            raise ValueError("LOX deformable bending indices must be -1 or reference valid particles.")
        for edge_vertices in edge_indices:
            referenced_vertices = edge_vertices[edge_vertices >= 0]
            if referenced_vertices.size > 0 and np.any(
                particle_source_world[referenced_vertices] != particle_source_world[referenced_vertices[0]]
            ):
                raise ValueError("LOX deformable bending edges cannot span Newton worlds.")
        valid_edges = np.all(edge_indices >= 0, axis=1)
        if np.any(valid_edges):
            if np.any(edge_rest_length[valid_edges] <= 0.0):
                raise ValueError("LOX deformable valid bending edges must have positive rest length.")
        if not np.all(np.isfinite(edge_rest_angle)) or not np.all(np.isfinite(edge_rest_length)):
            raise ValueError("LOX deformable bending rest data must be finite.")
        if not np.all(np.isfinite(edge_properties)) or np.any(edge_properties < 0.0):
            raise ValueError("LOX deformable bending properties must be finite and non-negative.")


###
# Interfaces
###


class _RecyclingCR(wpl.CR):
    """Fixed-count CR augmented by recent candidate-solve corrections.

    Args:
        recycling: Correction subspace of the recycled solve.
        packed_component: Structural component of each packed particle.
        packed_world: Solve world of each packed particle.
        packed_iterative: Whether each packed particle belongs to a CR component.
        world_active: Whether each world is still iterating.
    """

    def __init__(
        self,
        *args,
        recycling: DeformableRecyclingData,
        packed_component: wp.array[wp.int32],
        packed_world: wp.array[wp.int32],
        packed_iterative: wp.array[wp.int32],
        world_active: wp.array[wp.bool],
        **kwargs,
    ):
        self._recycling = recycling
        self._packed_component = packed_component
        self._packed_world = packed_world
        self._packed_iterative = packed_iterative
        self._world_active = world_active
        super().__init__(*args, **kwargs)

    ###
    # Implementation
    ###

    @override
    def _run(self, A, b, x, M):
        # This is Warp's fixed-count preconditioned recurrence with one
        # correction-space projection after the true initial residual is built.
        assert M is not None
        recycling = self._recycling
        batch_offsets = A.batch_offsets
        tiled_dot = self._tiled_dot
        r, z = self._r_and_z_buf[0], self._r_and_z_buf[1]
        r_copy, az = self._r_and_Az[0], self._r_and_Az[1]
        y, ap = self._y_and_Ap_buf[0], self._y_and_Ap_buf[1]
        p = self._p
        r_norm_sq = tiled_dot.col(0)
        z_az_new = tiled_dot.col(1)
        z_az_old, atol_sq = self._residuals[0], self._residuals[1]

        _wpl_internal._initialize_absolute_tolerance(b, self._tol, self._atol, tiled_dot, atol_sq)
        A.matvec(x, b, r, alpha=-1.0, beta=1.0)
        recycling.reductions.zero_()
        wp.launch(
            _reduce_recycled_cr_basis,
            dim=x.shape[0],
            inputs=[
                x,
                r,
                b,
                recycling.previous_velocity,
                recycling.previous_matrix_velocity,
                recycling.basis,
                recycling.basis_valid,
                recycling.projection,
                recycling.replacement_cursor,
                self._packed_component,
                self._packed_world,
                self._packed_iterative,
                self._world_active,
                recycling.started,
            ],
            outputs=[recycling.reductions],
            device=self._device,
        )
        wp.launch(
            _compute_recycled_cr_coefficients,
            dim=batch_offsets.shape[0] - 1,
            inputs=[
                recycling.reductions,
                recycling.basis_valid,
                recycling.replacement_cursor,
            ],
            outputs=[recycling.basis_denominator, recycling.orthogonalization, recycling.projection],
            device=self._device,
        )
        recycling.projection_reductions.zero_()
        wp.launch(
            _apply_recycled_cr_basis,
            dim=x.shape[0],
            inputs=[
                b,
                recycling.basis_denominator,
                recycling.basis_valid,
                recycling.replacement_cursor,
                recycling.orthogonalization,
                recycling.projection,
                self._packed_component,
                self._packed_world,
                self._packed_iterative,
                self._world_active,
                recycling.started,
            ],
            outputs=[
                recycling.basis,
                recycling.matrix_basis,
                x,
                r,
                recycling.previous_velocity,
                recycling.previous_matrix_velocity,
                recycling.projection_reductions,
            ],
            device=self._device,
        )
        wp.launch(
            _validate_and_restart_recycled_cr_projection,
            dim=batch_offsets.shape[0] - 1,
            inputs=[
                recycling.projection_reductions,
                recycling.started,
            ],
            outputs=[
                recycling.basis_denominator,
                recycling.orthogonalization,
                recycling.projection,
                recycling.projection_accepted,
            ],
            device=self._device,
        )
        wp.launch(
            _rollback_unsafe_recycled_cr_projection,
            dim=x.shape[0],
            inputs=[
                b,
                recycling.previous_velocity,
                recycling.previous_matrix_velocity,
                self._packed_component,
                self._packed_world,
                self._packed_iterative,
                self._world_active,
                recycling.started,
                recycling.projection_accepted,
            ],
            outputs=[x, r],
            device=self._device,
        )
        self._y_and_Ap_buf.zero_()

        def update_residual_direction_products():
            A.matvec(z, az, az, alpha=1.0, beta=0.0)
            r_copy.assign(r)
            tiled_dot.compute(self._r_and_z_buf, self._r_and_Az)

        z.zero_()
        M.matvec(r, z, z, alpha=1.0, beta=0.0)
        update_residual_direction_products()
        p.assign(z)
        ap.assign(az)

        for iteration in range(self._maxiter):
            z_az_old.assign(z_az_new)
            M.matvec(ap, y, y, alpha=1.0, beta=0.0)
            tiled_dot.compute(ap, y, col_offset=1)
            y_ap = tiled_dot.col(1)
            wp.launch(
                kernel=_wpl_internal._cr_kernel_1,
                dim=x.shape[0],
                device=self._device,
                inputs=[
                    atol_sq,
                    r_norm_sq,
                    z_az_old,
                    y_ap,
                    x,
                    r,
                    z,
                    p,
                    ap,
                    y,
                    batch_offsets,
                    self._dofs_per_entry,
                ],
            )
            if iteration + 1 < self._maxiter:
                update_residual_direction_products()
                wp.launch(
                    kernel=_wpl_internal._cr_kernel_2,
                    dim=z.shape[0],
                    device=self._device,
                    inputs=[
                        atol_sq,
                        r_norm_sq,
                        z_az_old,
                        z_az_new,
                        z,
                        p,
                        az,
                        ap,
                        batch_offsets,
                        self._dofs_per_entry,
                    ],
                )

        # Refresh the final residual norm and skip the unused next A*z product.
        r_copy.assign(r)
        tiled_dot.compute(self._r_and_z_buf, self._r_and_Az)
        return self._cur_iter_and_condition[0:1], r_norm_sq, atol_sq


class DeformableSystem:
    """Frozen smooth system of the LOX deformable particles and its candidate solves.

    Each time step, :meth:`assemble` linearizes the particle dynamics at the end-of-step
    positions of the step-start velocities into the smooth matrix ``A`` and right-hand side
    ``f``, and derives a scalar consensus weight per dynamic particle from the Schur
    complements of its patch of at most :data:`~.types.PATCH_PARTICLE_COUNT`
    particles, optionally capped at ``weight_beta`` times its free bound, see
    :mod:`.consensus_kernels`. Each splitting iteration, :meth:`solve_candidate` solves ``(A + alpha W) v = f + r_nl + W c`` for the
    candidate velocity ``v`` from the consensus center ``c = u + alpha p`` and the
    right-hand-side correction ``r_nl`` of the element proximal updates
    (:meth:`update_proximal`), with ``alpha`` the consensus fraction.

    The arrays live in :class:`DeformableSystemData`, exposed as :attr:`data`, and the packed
    topology in :class:`DeformableTopology`, exposed as :attr:`topology`.

    Args:
        model: Newton model containing the supported deformable subset.
        cr_iterations: Fixed number of preconditioned CR iterations per candidate solve.
        preconditioner: Deformable preconditioner type: ``"fsai"``, ``"incomplete_ldlt"``,
            ``"multilevel"``, ``"two_level"``, or ``"block_jacobi"``.
        preconditioner_regularization: Relative pivot or diagonal floor of the preconditioner.
        direct_max_particles: Largest component solved with blocked Cholesky.
        linear_solver: Deformable global linear solver type: ``"cr"``, ``"cudss"``, or
            ``"cudss_then_cr"``.
        proximal_iterations: Fixed local Gauss-Newton iterations per elastic element prox.
        proximal_relaxation: Relaxation factor for the local multiplier update.
        recycle_cr: Whether to augment CR with recent candidate-solve corrections.
        coarse_initial_guess: Whether to correct the starting guess of each CR candidate solve with the
            component-global Galerkin coarse solve of the ``"multilevel"`` preconditioner.
        consensus_fraction: Fraction ``alpha`` of the consensus weights in the matrix of the candidate solves,
            in [0, 1].
        weight_beta: Largest particle weight relative to the free bound of the particle, or zero for no cap.
        weight_sigma: Smallest fraction of the geometric mean of the patch bounds that the ``weight_beta`` cap
            keeps, in [0, 1].
    """

    def __init__(
        self,
        model: Model,
        *,
        cr_iterations: int = 4,
        preconditioner: str = "fsai",
        preconditioner_regularization: float = 1.0e-6,
        direct_max_particles: int = 0,
        linear_solver: str = "cr",
        proximal_iterations: int = 4,
        proximal_relaxation: float = 1.0,
        recycle_cr: bool = False,
        coarse_initial_guess: bool = False,
        consensus_fraction: float = 0.5,
        weight_beta: float = 0.0,
        weight_sigma: float = 0.0,
    ):
        validate_deformable_model(model)
        if not isinstance(cr_iterations, int) or isinstance(cr_iterations, bool) or cr_iterations < 1:
            raise ValueError(
                f"LOX deformable CR iterations must be an integer greater than or equal to one, got {cr_iterations}."
            )
        if preconditioner not in ("fsai", "incomplete_ldlt", "multilevel", "two_level", "block_jacobi"):
            raise ValueError(
                "LOX deformable preconditioner must be 'fsai', 'incomplete_ldlt', 'multilevel', "
                f"'two_level', or 'block_jacobi', got {preconditioner!r}."
            )
        if (
            not isinstance(direct_max_particles, int)
            or isinstance(direct_max_particles, bool)
            or direct_max_particles < 0
        ):
            raise ValueError(
                f"LOX deformable direct-solve particle limit must be a non-negative integer, got {direct_max_particles}."
            )
        if linear_solver not in ("cr", "cudss", "cudss_then_cr"):
            raise ValueError(
                f"LOX deformable linear solver must be 'cr', 'cudss', or 'cudss_then_cr', got {linear_solver!r}."
            )
        if not isinstance(proximal_iterations, int) or isinstance(proximal_iterations, bool) or proximal_iterations < 0:
            raise ValueError(
                f"LOX deformable proximal iterations must be a non-negative integer, got {proximal_iterations}."
            )
        if not np.isfinite(proximal_relaxation) or proximal_relaxation < 0.0 or proximal_relaxation > 1.0:
            raise ValueError(f"LOX deformable proximal relaxation must be in [0, 1], got {proximal_relaxation}.")
        if not isinstance(recycle_cr, bool):
            raise ValueError(f"LOX deformable CR recycling must be a boolean, got {recycle_cr}.")
        if not isinstance(coarse_initial_guess, bool):
            raise ValueError(f"LOX deformable coarse initial guess must be a boolean, got {coarse_initial_guess}.")
        if not np.isfinite(consensus_fraction) or consensus_fraction < 0.0 or consensus_fraction > 1.0:
            raise ValueError(f"LOX deformable consensus fraction must be in [0, 1], got {consensus_fraction}.")
        if not np.isfinite(weight_beta) or not (weight_beta == 0.0 or weight_beta >= 1.0):
            raise ValueError(f"LOX deformable weight beta must be zero or at least one, got {weight_beta}.")
        if not np.isfinite(weight_sigma) or weight_sigma < 0.0 or weight_sigma > 1.0:
            raise ValueError(f"LOX deformable weight sigma must be in [0, 1], got {weight_sigma}.")

        self.model = model
        """Newton model of the deformables."""
        self.device = model.device
        self.particle_count = int(model.particle_count)
        """Number of particles."""
        self.triangle_count = int(model.tri_count)
        """Number of membrane triangles."""
        self.tetrahedron_count = int(model.tet_count)
        """Number of tetrahedra."""
        self.recycle_cr = recycle_cr
        """Whether the CR candidate solve recycles recent corrections."""
        self.linear_solver = linear_solver
        """Global linear solver type of the candidate solve."""
        self.selective_consensus = False
        """Whether the consensus weights are restricted to the particles with unilateral rows."""
        self.consensus_fraction = float(consensus_fraction)
        """Fraction ``alpha`` of the consensus weights in the matrix of the candidate solves."""
        self.weight_beta = float(weight_beta)
        """Largest particle weight relative to the free bound of the particle, or zero for no cap."""
        self.weight_sigma = float(weight_sigma)
        """Smallest fraction of the geometric mean that the :attr:`weight_beta` cap keeps."""

        particle_source_world = model.particle_world.numpy().astype(np.int32, copy=False)
        particle_solve_world = _particle_solve_worlds(model, particle_source_world)
        triangle_indices_np = (
            model.tri_indices.numpy().astype(np.int32, copy=False)
            if model.tri_count > 0
            else np.empty((0, 3), dtype=np.int32)
        )
        tetrahedron_indices_np = (
            model.tet_indices.numpy().astype(np.int32, copy=False)
            if model.tet_count > 0
            else np.empty((0, 4), dtype=np.int32)
        )
        edge_indices_np = (
            model.edge_indices.numpy().astype(np.int32, copy=False)
            if model.edge_count > 0
            else np.empty((0, 4), dtype=np.int32)
        )
        valid_edge_mask = np.all(edge_indices_np >= 0, axis=1)
        source_edge_indices_np = np.flatnonzero(valid_edge_mask).astype(np.int32)
        bending_indices_np = edge_indices_np[valid_edge_mask]
        particle_component = _deformable_particle_components(
            self.particle_count,
            particle_solve_world,
            triangle_indices_np,
            bending_indices_np,
            tetrahedron_indices_np,
        )
        component_count = int(np.max(particle_component)) + 1
        component_particle_counts_np = np.bincount(particle_component, minlength=component_count).astype(np.int32)
        if linear_solver == "cudss":
            component_direct_np = np.ones(component_count, dtype=bool)
        elif linear_solver == "cudss_then_cr":
            component_direct_np = np.zeros(component_count, dtype=bool)
        else:
            component_direct_np = component_particle_counts_np <= direct_max_particles
            if direct_max_particles == 0:
                component_direct_np[:] = False
        has_direct_components = bool(np.any(component_direct_np))
        self.has_iterative_components = bool(np.any(~component_direct_np))
        """Whether any component is solved by preconditioned CR."""

        original_indices = np.arange(self.particle_count, dtype=np.int32)
        packed_to_newton_np = np.lexsort(
            (original_indices, particle_component, particle_solve_world),
        ).astype(np.int32)
        newton_to_packed_np = np.empty(self.particle_count, dtype=np.int32)
        newton_to_packed_np[packed_to_newton_np] = original_indices
        packed_source_world_np = particle_source_world[packed_to_newton_np]
        packed_solve_world_np = particle_solve_world[packed_to_newton_np]
        packed_component_np = particle_component[packed_to_newton_np]
        packed_iterative_np = (~component_direct_np[packed_component_np]).astype(np.int32)

        component_dof_offsets_np = 3 * np.concatenate(
            (np.zeros(1, dtype=np.int32), np.cumsum(component_particle_counts_np, dtype=np.int32))
        )

        packed_triangle_indices_np = newton_to_packed_np[triangle_indices_np]
        packed_bending_indices_np = (
            newton_to_packed_np[bending_indices_np] if np.any(valid_edge_mask) else np.empty((0, 4), dtype=np.int32)
        )
        packed_tetrahedron_indices_np = newton_to_packed_np[tetrahedron_indices_np]
        self.bending_count = int(source_edge_indices_np.shape[0])
        """Number of valid bending edges."""
        # Discrete-shell scale of each bending edge (Grinspun et al. 2003): the hinge energy
        # 1/2 k |e|^2 / (A_0 + A_1) theta^2 converges as the mesh is refined, and on near-equilateral
        # meshes bends a cylinder with the plate bending stiffness k [N m].
        bending_rest_scale_np = np.empty(0, dtype=np.float64)
        if self.bending_count > 0:
            rest_positions = model.particle_q.numpy().astype(np.float64)
            rest_length = model.edge_rest_length.numpy().astype(np.float64)[source_edge_indices_np]
            opposite_0 = rest_positions[bending_indices_np[:, 0]]
            opposite_1 = rest_positions[bending_indices_np[:, 1]]
            edge_start = rest_positions[bending_indices_np[:, 2]]
            edge_end = rest_positions[bending_indices_np[:, 3]]
            edge_vector = edge_end - edge_start
            area_0 = 0.5 * np.linalg.norm(np.cross(edge_vector, opposite_0 - edge_start), axis=1)
            area_1 = 0.5 * np.linalg.norm(np.cross(edge_vector, opposite_1 - edge_start), axis=1)
            bending_rest_scale_np = rest_length * rest_length / np.maximum(area_0 + area_1, 1.0e-20)

        self._topology = DeformableTopology(
            newton_to_packed=newton_to_packed_np,
            packed_to_newton=packed_to_newton_np,
            packed_source_world=packed_source_world_np,
            packed_solve_world=packed_solve_world_np,
            packed_component=packed_component_np,
            component_dof_offsets=component_dof_offsets_np,
            triangle_indices=packed_triangle_indices_np,
            bending_indices=packed_bending_indices_np,
            source_edge_indices=source_edge_indices_np,
            bending_rest_scale=bending_rest_scale_np,
            tetrahedron_indices=packed_tetrahedron_indices_np,
            device=self.device,
        )
        topology = self._topology

        # The assembly triplets hold the particle masses, then the 3 x 3 pair blocks of the
        # triangles, the bending edges, and the tetrahedra.
        self.triangle_triplet_offset = self.particle_count
        self.bending_triplet_offset = self.triangle_triplet_offset + 9 * self.triangle_count
        self.tetrahedron_triplet_offset = self.bending_triplet_offset + 16 * self.bending_count
        self.triplet_count = self.tetrahedron_triplet_offset + 16 * self.tetrahedron_count
        self.element_incidence = DeformableElementIncidence(
            self.particle_count,
            self.triangle_triplet_offset,
            self.bending_triplet_offset,
            self.tetrahedron_triplet_offset,
            packed_triangle_indices_np,
            packed_bending_indices_np,
            packed_tetrahedron_indices_np,
            component_dof_offsets_np // 3,
            self.device,
        )
        """Elements of each particle, which assemble the patch consensus weights."""

        triplet_rows_np = np.empty(self.triplet_count, dtype=np.int32)
        triplet_columns_np = np.empty(self.triplet_count, dtype=np.int32)
        triplet_rows_np[: self.particle_count] = original_indices
        triplet_columns_np[: self.particle_count] = original_indices

        cursor = self.triangle_triplet_offset
        for vertices in packed_triangle_indices_np:
            for row in vertices:
                for column in vertices:
                    triplet_rows_np[cursor] = row
                    triplet_columns_np[cursor] = column
                    cursor += 1
        for vertices in packed_bending_indices_np:
            for row in vertices:
                for column in vertices:
                    triplet_rows_np[cursor] = row
                    triplet_columns_np[cursor] = column
                    cursor += 1
        for vertices in packed_tetrahedron_indices_np:
            for row in vertices:
                for column in vertices:
                    triplet_rows_np[cursor] = row
                    triplet_columns_np[cursor] = column
                    cursor += 1

        # Build the fixed block sparse topology shared by the smooth, system, and preconditioner matrices.
        structural_coordinates = np.unique(
            np.column_stack((triplet_rows_np, triplet_columns_np)),
            axis=0,
        ).astype(np.int32, copy=False)
        structural_rows = wp.array(structural_coordinates[:, 0], dtype=wp.int32, device=self.device)
        structural_columns = wp.array(structural_coordinates[:, 1], dtype=wp.int32, device=self.device)
        structural_values = wp.zeros(structural_coordinates.shape[0], dtype=wp.mat33, device=self.device)
        smooth_matrix = wps.bsr_zeros(
            self.particle_count,
            self.particle_count,
            wp.mat33,
            device=self.device,
        )
        wps.bsr_set_from_triplets(
            smooth_matrix,
            structural_rows,
            structural_columns,
            structural_values,
            prune_numerical_zeros=False,
            topology="compact",
        )

        matrix_offsets = smooth_matrix.offsets.numpy()
        matrix_columns = smooth_matrix.columns.numpy()
        diagonal_slots_np = np.empty(self.particle_count, dtype=np.int32)
        for row in range(self.particle_count):
            row_start = int(matrix_offsets[row])
            row_end = int(matrix_offsets[row + 1])
            matching = np.flatnonzero(matrix_columns[row_start:row_end] == row)
            if matching.size != 1:
                raise RuntimeError(f"LOX deformable topology has no unique diagonal block in row {row}.")
            diagonal_slots_np[row] = row_start + int(matching[0])
        self.matrix_nnz = int(matrix_offsets[-1])
        """Number of blocks of the block sparse matrices."""
        system_matrix = wps.bsr_copy(
            smooth_matrix,
            structure_only=True,
            topology="compact",
        )
        preconditioner_matrix = wps.bsr_copy(
            smooth_matrix,
            structure_only=True,
            topology="compact",
        )

        proximal_enabled = proximal_iterations > 0 and proximal_relaxation > 0.0
        self._data = DeformableSystemData(
            particle_count=self.particle_count,
            world_count=int(model.world_count),
            component_count=component_count,
            triangle_count=self.triangle_count,
            tetrahedron_count=self.tetrahedron_count,
            triplet_rows=triplet_rows_np,
            triplet_columns=triplet_columns_np,
            smooth_matrix=smooth_matrix,
            system_matrix=system_matrix,
            preconditioner_matrix=preconditioner_matrix,
            diagonal_slots=diagonal_slots_np,
            packed_iterative=packed_iterative_np,
            membrane_proximal=self.triangle_count > 0 and proximal_enabled,
            tetrahedron_proximal=self.tetrahedron_count > 0 and proximal_enabled,
            recycling=self.recycle_cr and self.has_iterative_components,
            device=self.device,
        )
        data = self._data
        self._system_matvec_scalar_views: dict[int, wp.array] = {}

        self.membrane_proximal = (
            DeformableMembraneProximal(
                model, topology, data, data.membrane_proximal, proximal_iterations, proximal_relaxation
            )
            if data.membrane_proximal is not None
            else None
        )
        """Membrane proximal updates, with membrane triangles and proximal iterations."""
        self.tetrahedron_proximal = (
            DeformableTetrahedronProximal(
                model, topology, data, data.tetrahedron_proximal, proximal_iterations, proximal_relaxation
            )
            if data.tetrahedron_proximal is not None
            else None
        )
        """Tetrahedron proximal updates, with tetrahedra and proximal iterations."""

        self.system_operator = wpl.LinearOperator(
            shape=data.system_matrix.shape,
            dtype=data.system_matrix.dtype,
            device=self.device,
            matvec=self._system_matvec,
            batch_offsets=topology.component_dof_offsets,
        )
        """Warp linear operator of the system matrix on the CR rows of the active worlds."""
        self.cudss_solver = (
            DeformableCuDSS(
                data.system_matrix,
                topology.packed_solve_world,
                data.world_active,
            )
            if linear_solver in ("cudss", "cudss_then_cr")
            else None
        )
        """cuDSS solver of the whole system, with the cuDSS linear solvers."""
        if linear_solver == "cudss":
            self.direct_solver = self.cudss_solver
        elif has_direct_components:
            self.direct_solver = DeformableBlockLLT(
                data.system_matrix,
                packed_component_np,
                packed_iterative_np,
                topology.packed_solve_world,
                data.world_active,
                component_count,
            )
        else:
            self.direct_solver = None
        if self.has_iterative_components and preconditioner == "fsai":
            self.preconditioner = DeformableFSAI(
                data.preconditioner_matrix,
                topology.packed_solve_world,
                data.world_active,
                topology.component_dof_offsets,
                regularization=preconditioner_regularization,
                row_active=data.packed_iterative,
            )
        elif self.has_iterative_components and preconditioner == "incomplete_ldlt":
            self.preconditioner = DeformableIncompleteLDLT(
                data.preconditioner_matrix,
                topology.packed_solve_world,
                data.world_active,
                topology.component_dof_offsets,
                regularization=preconditioner_regularization,
                row_active=data.packed_iterative,
            )
        elif self.has_iterative_components and preconditioner in ("multilevel", "two_level"):
            preconditioner_type = DeformableMultilevel if preconditioner == "multilevel" else DeformableTwoLevel
            self.preconditioner = preconditioner_type(
                data.preconditioner_matrix,
                data.diagonal_slots,
                packed_component_np,
                topology.packed_solve_world,
                data.world_active,
                topology.component_dof_offsets,
                regularization=preconditioner_regularization,
                row_active=data.packed_iterative,
            )
        elif self.has_iterative_components:
            self.preconditioner = DeformableJacobi(
                data.preconditioner_matrix,
                data.diagonal_slots,
                topology.packed_solve_world,
                data.world_active,
                topology.component_dof_offsets,
                regularization=preconditioner_regularization,
                row_active=data.packed_iterative,
            )
        else:
            assert self.direct_solver is not None
            self.preconditioner = self.direct_solver
        self.coarse_corrector: DeformableMultilevel | None = None
        """Multilevel preconditioner whose component-global coarse solve corrects the starting guess of each
        CR candidate solve, or ``None``."""
        if self.has_iterative_components and coarse_initial_guess:
            if isinstance(self.preconditioner, DeformableMultilevel):
                self.coarse_corrector = self.preconditioner
            else:
                self.coarse_corrector = DeformableMultilevel(
                    data.preconditioner_matrix,
                    data.diagonal_slots,
                    packed_component_np,
                    topology.packed_solve_world,
                    data.world_active,
                    topology.component_dof_offsets,
                    regularization=preconditioner_regularization,
                    row_active=data.packed_iterative,
                )
        self._coarse_residual = (
            wp.zeros(self.particle_count, dtype=wp.vec3, device=self.device)
            if self.coarse_corrector is not None
            else None
        )
        if not self.has_iterative_components:
            self.cr_state = None
        elif data.recycling is not None:
            self.cr_state = _RecyclingCR(
                self.system_operator,
                data.candidate_rhs,
                data.smooth_velocity,
                tol=0.0,
                atol=0.0,
                maxiter=cr_iterations,
                M=self.preconditioner.linear_operator,
                check_every=0,
                use_cuda_graph=False,
                recycling=data.recycling,
                packed_component=topology.packed_component,
                packed_world=topology.packed_solve_world,
                packed_iterative=data.packed_iterative,
                world_active=data.world_active,
            )
        else:
            self.cr_state = wpl.CR(
                self.system_operator,
                data.candidate_rhs,
                data.smooth_velocity,
                tol=0.0,
                atol=0.0,
                maxiter=cr_iterations,
                M=self.preconditioner.linear_operator,
                check_every=0,
                use_cuda_graph=False,
            )
        # Warp's stateful CR API keeps its final recursive residual private;
        # retaining this allocated view saves a redundant A*x.
        self.cr_residual = None if self.cr_state is None else self.cr_state._r_and_z_buf[0]

    ###
    # Properties
    ###

    @property
    def data(self) -> DeformableSystemData:
        """Returns the arrays of the deformable system."""
        return self._data

    @property
    def topology(self) -> DeformableTopology:
        """Returns the packed topology of the deformables."""
        return self._topology

    ###
    # Public API
    ###

    def assemble(
        self,
        state: State,
        time_step: wp.array[wp.float32],
        *,
        finalize_consensus: bool = True,
    ) -> None:
        """Assemble the frozen smooth matrix and affine right-hand side.

        Args:
            state: Beginning-of-step Newton particle state.
            time_step: Time step of each world [s].
            finalize_consensus: Whether to build and factor the weighted system immediately.

        Raises:
            ValueError: If the particle state is invalid.
        """
        if state.particle_q is None or state.particle_qd is None or state.particle_f is None:
            raise ValueError("LOX deformables require particle positions, velocities, and forces in the input state.")
        expected_shape = (self.particle_count,)
        for name in ("particle_q", "particle_qd", "particle_f"):
            value = getattr(state, name)
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"LOX deformables expected State.{name} shape {expected_shape}, found {value.shape}.")
            if value.device != self.device:
                raise ValueError(f"LOX deformables expected State.{name} on {self.device}, found {value.device}.")

        data = self._data
        wp.launch(
            _gather_particle_state,
            dim=self.particle_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                state.particle_f,
                self.model.particle_mass,
                self.model.particle_flags,
                self.model.gravity,
                self._topology.packed_to_newton,
                self._topology.packed_source_world,
            ],
            outputs=[
                data.position_start,
                data.velocity_start,
                data.external_force,
            ],
            device=self.device,
        )
        self._assemble_linearization(data.velocity_start, time_step, finalize_consensus)

    def set_unilateral_incidence(self, incidence: wp.array[wp.int32] | None) -> None:
        """Freeze nodal unilateral support and rebuild the weighted linear system.

        Args:
            incidence: Number of unilateral rows of each packed particle, or ``None`` for none.
        """
        data = self._data
        if incidence is None:
            data.unilateral_incidence.zero_()
        else:
            if incidence.shape != (self.particle_count,) or incidence.dtype != wp.int32:
                raise ValueError("LOX deformable incidence must contain one int32 entry per packed particle.")
            if incidence.device != self.device:
                raise ValueError(f"LOX deformable expected incidence on {self.device}, found {incidence.device}.")
            wp.copy(data.unilateral_incidence, incidence)
        self._finalize_consensus_system()

    def update_proximal(self, time_step: wp.array[wp.float32]) -> None:
        """Evaluate nonlinear elastic proxes for the current global velocity.

        Rewrites :attr:`DeformableSystemData.nonlinear_rhs` for the next candidate solve, and
        the proximal residuals and failures of each world.

        Args:
            time_step: Time step of each world [s].
        """
        if self.membrane_proximal is None and self.tetrahedron_proximal is None:
            return
        data = self._data
        data.nonlinear_rhs.zero_()
        data.proximal_position_residual.zero_()
        data.proximal_velocity_residual.zero_()
        wp.copy(data.proximal_failed, data.assembly_failed)
        if self.membrane_proximal is not None:
            self.membrane_proximal.update(time_step)
        if self.tetrahedron_proximal is not None:
            self.tetrahedron_proximal.update(time_step)

    def solve_candidate(
        self,
        consensus_center: wp.array[wp.vec3],
        *,
        world_active: wp.array[wp.bool] | None = None,
        use_cudss: bool | None = None,
    ) -> None:
        """Run the fixed-count warm-started preconditioned CR candidate solve.

        The candidate velocity is written to :attr:`DeformableSystemData.smooth_velocity`; the
        particles of inactive worlds keep their velocity. With :attr:`coarse_corrector`, the
        component-global Galerkin coarse solve first corrects the warm start.

        Args:
            consensus_center: Splitting target ``c = u + alpha p`` of each particle [m/s].
            world_active: Optional per-world active mask copied into stable storage.
            use_cudss: Whether to use the retained cuDSS factorization for this
                candidate. Defaults to true for the pure cuDSS mode and false
                otherwise.
        """
        data = self._data
        if consensus_center.shape != (self.particle_count,) or consensus_center.dtype != wp.vec3:
            raise ValueError("LOX deformable consensus center must be a vec3 array with one entry per packed particle.")
        if consensus_center.device != self.device:
            raise ValueError(
                f"LOX deformable expected consensus center on {self.device}, found {consensus_center.device}."
            )
        if world_active is not None:
            if world_active.shape != data.world_active.shape or world_active.dtype != wp.bool:
                raise ValueError(
                    f"LOX deformable world active mask must have shape {data.world_active.shape} and dtype bool."
                )
            if world_active.device != self.device:
                raise ValueError(
                    f"LOX deformable expected world active mask on {self.device}, found {world_active.device}."
                )
            wp.copy(dest=data.world_active, src=world_active, count=data.world_active.shape[0])

        wp.launch(
            _prepare_candidate_rhs,
            dim=self.particle_count,
            inputs=[
                data.smooth_rhs,
                data.nonlinear_rhs,
                consensus_center,
                data.weight,
                data.smooth_velocity,
                self._topology.packed_solve_world,
                data.world_active,
            ],
            outputs=[data.candidate_rhs],
            device=self.device,
        )
        if use_cudss is None:
            use_cudss = self.linear_solver == "cudss"
        if use_cudss:
            if self.cudss_solver is None:
                raise RuntimeError("LOX deformable cuDSS candidate solve was not initialized.")
            self.cudss_solver.solve(data.candidate_rhs, data.smooth_velocity)
            data.candidate_linear_residual.zero_()
            return
        if self.direct_solver is not None:
            self.direct_solver.solve(data.candidate_rhs, data.smooth_velocity)
            wp.launch(
                _retain_direct_candidate_rhs,
                dim=self.particle_count,
                inputs=[data.smooth_velocity, data.packed_iterative],
                outputs=[data.candidate_rhs],
                device=self.device,
            )
        data.candidate_linear_residual.zero_()
        if not self.has_iterative_components:
            return

        assert self.cr_state is not None
        if self.coarse_corrector is not None:
            # Remove the smooth global error of the warm start, which the local preconditioner resolves slowly.
            self._system_matvec(data.smooth_velocity, data.candidate_rhs, self._coarse_residual, -1.0, 1.0)
            self.coarse_corrector.add_global_coarse_correction(self._coarse_residual, data.smooth_velocity)
        self.cr_state(M=self.preconditioner.linear_operator)
        assert self.cr_residual is not None

        residual_inputs = [
            self.cr_residual,
            data.full_inverse_weight,
            self._topology.packed_solve_world,
            data.packed_iterative,
            data.world_active,
        ]
        if data.recycling is None:
            wp.launch(
                _compute_candidate_linear_residual,
                dim=self.particle_count,
                inputs=residual_inputs,
                outputs=[data.candidate_linear_residual],
                device=self.device,
            )
        else:
            wp.launch(
                _compute_candidate_linear_residual_and_advance_recycling,
                dim=self.particle_count,
                inputs=residual_inputs,
                outputs=[
                    data.candidate_linear_residual,
                    data.recycling.started,
                    data.recycling.replacement_cursor,
                    data.recycling.basis_valid,
                ],
                device=self.device,
            )

    ###
    # Internals
    ###

    def _assemble_linearization(
        self,
        linearization_velocity: wp.array[wp.vec3],
        time_step: wp.array[wp.float32],
        finalize_consensus: bool = True,
    ) -> None:
        """Assemble and factor one deformable linearization with fixed step-start data."""
        model = self.model
        topology = self._topology
        data = self._data
        data.assembly_failed.zero_()
        wp.launch(
            _prepare_particle_linearization,
            dim=self.particle_count,
            inputs=[
                data.position_start,
                linearization_velocity,
                data.external_force,
                topology.packed_to_newton,
                topology.packed_solve_world,
                model.particle_flags,
                time_step,
            ],
            outputs=[
                data.position_linearized,
                data.velocity_linearized,
                data.smooth_force,
            ],
            device=self.device,
        )
        wp.launch(
            _assemble_particle_mass,
            dim=self.particle_count,
            inputs=[
                topology.packed_to_newton,
                model.particle_mass,
                model.particle_flags,
            ],
            outputs=[data.triplet_values],
            device=self.device,
        )
        if self.triangle_count > 0:
            wp.launch(
                _assemble_triangle_system,
                dim=self.triangle_count,
                inputs=[
                    data.position_start,
                    data.position_linearized,
                    data.velocity_linearized,
                    topology.triangle_indices,
                    model.tri_poses,
                    model.tri_areas,
                    model.tri_materials,
                    model.tri_activations,
                    topology.packed_to_newton,
                    topology.packed_solve_world,
                    model.particle_mass,
                    model.particle_flags,
                    time_step,
                    self.triangle_triplet_offset,
                ],
                outputs=[data.triplet_values, data.smooth_force, data.assembly_failed],
                device=self.device,
            )
        if self.bending_count > 0:
            wp.launch(
                _assemble_bending_system,
                dim=self.bending_count,
                inputs=[
                    data.position_start,
                    data.position_linearized,
                    topology.bending_indices,
                    topology.source_edge_indices,
                    topology.bending_rest_scale,
                    model.edge_rest_angle,
                    model.edge_bending_properties,
                    topology.packed_to_newton,
                    topology.packed_solve_world,
                    model.particle_mass,
                    model.particle_flags,
                    time_step,
                    self.bending_triplet_offset,
                ],
                outputs=[data.triplet_values, data.smooth_force],
                device=self.device,
            )
        if self.tetrahedron_count > 0:
            wp.launch(
                _assemble_tetrahedron_system,
                dim=self.tetrahedron_count,
                inputs=[
                    data.position_start,
                    data.position_linearized,
                    topology.tetrahedron_indices,
                    model.tet_poses,
                    model.tet_materials,
                    model.tet_activations,
                    topology.packed_to_newton,
                    topology.packed_solve_world,
                    model.particle_mass,
                    model.particle_flags,
                    time_step,
                    self.tetrahedron_triplet_offset,
                ],
                outputs=[data.tetrahedron_metric, data.triplet_values, data.smooth_force],
                device=self.device,
            )

        wps.bsr_set_from_triplets(
            data.smooth_matrix,
            data.triplet_rows,
            data.triplet_columns,
            data.triplet_values,
            prune_numerical_zeros=False,
            topology="masked",
        )
        wps.bsr_mv(
            data.smooth_matrix,
            x=data.velocity_linearized,
            y=data.matrix_velocity,
            alpha=1.0,
            beta=0.0,
        )
        wp.launch(
            _finalize_smooth_rhs,
            dim=self.particle_count,
            inputs=[
                data.matrix_velocity,
                data.smooth_force,
                data.velocity_start,
                data.velocity_linearized,
                topology.packed_to_newton,
                topology.packed_solve_world,
                model.particle_mass,
                model.particle_flags,
                time_step,
            ],
            outputs=[data.smooth_rhs],
            device=self.device,
        )
        incidence = self.element_incidence
        wp.launch(
            _mark_dynamic_particles,
            dim=self.particle_count,
            inputs=[topology.packed_to_newton, model.particle_mass, model.particle_flags],
            outputs=[incidence.particle_dynamic],
            device=self.device,
        )
        wp.launch_tiled(
            _sum_chunk_mass,
            dim=incidence.chunk_count,
            inputs=[
                incidence.chunk_particle_offsets,
                topology.packed_to_newton,
                model.particle_mass,
                incidence.particle_dynamic,
            ],
            outputs=[incidence.chunk_mass, incidence.chunk_fixed_count],
            device=self.device,
            block_dim=_COMPONENT_MASS_BLOCK_DIM,
        )
        wp.launch(
            _sum_component_mass,
            dim=topology.component_count,
            inputs=[incidence.component_chunk_offsets, incidence.chunk_mass, incidence.chunk_fixed_count],
            outputs=[incidence.component_mass, incidence.component_fixed_count],
            device=self.device,
        )
        wp.launch(
            _select_consensus_patch,
            dim=self.particle_count,
            inputs=[
                data.smooth_matrix.values,
                data.diagonal_slots,
                data.triplet_values,
                incidence.particle_element_offset,
                incidence.particle_element,
                incidence.element_vertices,
                incidence.element_triplet,
                incidence.element_size,
                incidence.particle_dynamic,
            ],
            outputs=[
                data.full_weight,
                data.full_inverse_weight,
                data.weight_status,
                incidence.patch_particles,
                incidence.patch_count,
            ],
            device=self.device,
        )
        # One thread per block row of the free and pinned patch matrices of each particle
        wp.launch(
            _assemble_consensus_patch_rows,
            dim=2 * PATCH_PARTICLE_COUNT * self.particle_count,
            inputs=[
                data.smooth_matrix.offsets,
                data.smooth_matrix.columns,
                data.smooth_matrix.values,
                data.triplet_values,
                incidence.particle_element_offset,
                incidence.particle_element,
                incidence.element_vertices,
                incidence.element_triplet,
                incidence.element_size,
                incidence.particle_dynamic,
                topology.packed_to_newton,
                model.particle_mass,
                incidence.patch_particles,
                incidence.patch_count,
            ],
            outputs=[incidence.patch_blocks],
            device=self.device,
        )
        wp.launch(
            _eliminate_consensus_patch,
            dim=2 * self.particle_count,
            inputs=[incidence.patch_count],
            outputs=[incidence.patch_blocks, incidence.patch_smallest],
            device=self.device,
        )
        wp.launch(
            _finish_consensus_weight,
            dim=self.particle_count,
            inputs=[
                topology.packed_to_newton,
                topology.packed_component,
                model.particle_mass,
                data.smooth_matrix.values,
                data.diagonal_slots,
                incidence.component_mass,
                incidence.component_fixed_count,
                incidence.patch_count,
                incidence.patch_smallest,
                self.weight_beta,
                self.weight_sigma,
            ],
            outputs=[data.full_weight, data.full_inverse_weight, data.weight_status],
            device=self.device,
        )
        if finalize_consensus:
            self._finalize_consensus_system()
        wp.copy(
            dest=data.smooth_velocity,
            src=data.velocity_linearized,
            count=self.particle_count,
        )
        data.nonlinear_rhs.zero_()
        data.proximal_position_residual.zero_()
        data.proximal_velocity_residual.zero_()
        wp.copy(data.proximal_failed, data.assembly_failed)
        if self.membrane_proximal is not None:
            self.membrane_proximal.initialize()
        if self.tetrahedron_proximal is not None:
            self.tetrahedron_proximal.initialize()

    def _finalize_consensus_system(self) -> None:
        """Apply the consensus support and refactor the operator and preconditioner.

        A positive consensus fraction ``alpha`` adds the selected weights ``alpha W`` to the system matrix and
        the full weights ``alpha W`` to the preconditioner matrix.
        """
        data = self._data
        if data.recycling is not None:
            data.recycling.started.zero_()
            data.recycling.basis_valid.zero_()
            data.recycling.basis_denominator.zero_()
            data.recycling.replacement_cursor.zero_()
        wp.launch(
            _select_consensus_weight,
            dim=self.particle_count,
            inputs=[
                data.full_weight,
                data.full_inverse_weight,
                data.unilateral_incidence,
                self.selective_consensus,
            ],
            outputs=[data.consensus_enabled, data.weight, data.inverse_weight],
            device=self.device,
        )
        wp.copy(
            dest=data.system_matrix.values,
            src=data.smooth_matrix.values,
            count=self.matrix_nnz,
        )
        wp.copy(
            dest=data.preconditioner_matrix.values,
            src=data.smooth_matrix.values,
            count=self.matrix_nnz,
        )
        if self.consensus_fraction > 0.0:
            wp.launch(
                _assemble_consensus_weight,
                dim=self.particle_count,
                inputs=[data.diagonal_slots, data.weight, self.consensus_fraction],
                outputs=[data.system_matrix.values],
                device=self.device,
            )
            wp.launch(
                _assemble_consensus_weight,
                dim=self.particle_count,
                inputs=[data.diagonal_slots, data.full_weight, self.consensus_fraction],
                outputs=[data.preconditioner_matrix.values],
                device=self.device,
            )
        if self.cudss_solver is not None:
            self.cudss_solver.factorize()
        if self.direct_solver is not None and self.direct_solver is not self.cudss_solver:
            self.direct_solver.factorize()
        if self.has_iterative_components:
            self.preconditioner.factorize()
        if self.coarse_corrector is not None and self.coarse_corrector is not self.preconditioner:
            self.coarse_corrector.factorize_global_coarse()

    def _system_matvec(
        self,
        x: wp.array[wp.vec3],
        y: wp.array[wp.vec3],
        z: wp.array[wp.vec3],
        alpha: float,
        beta: float,
    ) -> None:
        """Write ``z = alpha A x + beta y`` on the CR rows of the active worlds and ``alpha x + beta y`` elsewhere."""
        data = self._data
        matrix = data.system_matrix
        x_scalar = self._system_matvec_scalar_views.get(x.ptr)
        if x_scalar is None:
            x_scalar = _wps_internal._vec_array_view(x, wp.float32, matrix.ncol * 3)
            self._system_matvec_scalar_views[x.ptr] = x_scalar
        y_scalar = self._system_matvec_scalar_views.get(y.ptr)
        if y_scalar is None:
            y_scalar = _wps_internal._vec_array_view(y, wp.float32, matrix.nrow * 3)
            self._system_matvec_scalar_views[y.ptr] = y_scalar
        z_scalar = self._system_matvec_scalar_views.get(z.ptr)
        if z_scalar is None:
            z_scalar = _wps_internal._vec_array_view(z, wp.float32, matrix.nrow * 3)
            self._system_matvec_scalar_views[z.ptr] = z_scalar
        wp.launch(
            kernel=_apply_system_matrix,
            dim=(matrix.nrow, 3),
            inputs=[
                matrix.offsets,
                matrix.columns,
                matrix.scalar_values,
                x_scalar,
                y_scalar,
                self._topology.packed_solve_world,
                data.packed_iterative,
                data.world_active,
                alpha,
                beta,
            ],
            outputs=[z_scalar],
            block_dim=_SYSTEM_MATVEC_BLOCK_DIM,
            device=self.device,
        )
