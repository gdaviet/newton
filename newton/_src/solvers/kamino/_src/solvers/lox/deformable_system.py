# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Static topology and frozen smooth-system assembly for LOX deformables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
import warp.sparse as wps
from warp._src import sparse as _wps_internal
from warp.optim import linear as wpl

from .deformable_assembly import (
    DEFORMABLE_WEIGHT_STATUS_INVALID,
    DEFORMABLE_WEIGHT_STATUS_REGULARIZED,
    DEFORMABLE_WEIGHT_STATUS_VALID,
    add_consensus_weight,
    assemble_bending_system,
    assemble_particle_mass,
    assemble_tetrahedron_system,
    assemble_triangle_system,
    compute_consensus_weight,
    finish_smooth_rhs,
    gather_particle_state,
    prepare_candidate_rhs,
    retain_direct_candidate_rhs,
    select_consensus_weight,
    set_particle_linearization,
)
from .deformable_direct import DeformableBlockLLT
from .deformable_energy import mat99
from .deformable_jacobi import DeformableJacobi
from .deformable_preconditioner import DeformableIncompleteLDLT
from .deformable_proximal import DeformableMembraneProximal
from .deformable_tetrahedron_proximal import DeformableTetrahedronProximal
from .deformable_two_level import DeformableTwoLevel
from .time import validate_world_time_step
from .weight import BODY_WEIGHT_SIGMA_DEFAULT, DEFORMABLE_WEIGHT_BETA_DEFAULT

if TYPE_CHECKING:
    from ......sim import Model, State

__all__ = [
    "DEFORMABLE_WEIGHT_STATUS_INVALID",
    "DEFORMABLE_WEIGHT_STATUS_REGULARIZED",
    "DEFORMABLE_WEIGHT_STATUS_VALID",
    "DeformableFEMSystem",
    "DeformableTopology",
    "validate_deformable_model",
]

_PARTICLE_FLAG_ACTIVE = 1
_PARTICLE_FLAG_PROXY = 2
_SYSTEM_MATVEC_BLOCK_DIM = 64


@wp.kernel
def _masked_system_matvec(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array3d[wp.float32],
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
    packed_world: wp.array[wp.int32],
    packed_iterative: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    alpha: float,
    beta: float,
    result: wp.array[wp.float32],
):
    row, subrow = wp.tid()
    scalar_row = 3 * row + subrow
    value = x[scalar_row]
    if packed_iterative[row] != 0 and world_active[packed_world[row]] != 0:
        value = wp.float32(0.0)
        slot = wp.int32(offsets[row])
        slot_end = wp.int32(offsets[row + 1])
        while slot < slot_end:
            scalar_column = 3 * columns[slot]
            for column in range(3):
                value += values[slot, subrow, column] * x[scalar_column + column]
            slot += 1
    value *= alpha
    if beta != 0.0:
        value += beta * y[scalar_row]
    result[scalar_row] = value


def _require_array(model: Model, name: str, expected_shape: tuple[int, ...]) -> np.ndarray:
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
    if np.any((particle_flags & _PARTICLE_FLAG_PROXY) != 0):
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


@dataclass(frozen=True)
class DeformableTopology:
    """Host and device mappings for one immutable deformable topology.

    ``packed_source_world`` preserves Newton ownership, including global world
    ``-1``. ``packed_solve_world`` stores the nonnegative LOX partition used
    for batching, status, timestep, and rigid/deformable coupling arrays.
    """

    newton_to_packed: wp.array[wp.int32]
    packed_to_newton: wp.array[wp.int32]
    packed_source_world: wp.array[wp.int32]
    packed_solve_world: wp.array[wp.int32]
    component_dof_offsets: wp.array[wp.int32]
    triangle_indices: wp.array2d[wp.int32]
    bending_indices: wp.array2d[wp.int32]
    source_edge_indices: wp.array[wp.int32]
    tetrahedron_indices: wp.array2d[wp.int32]


class DeformableFEMSystem:
    """Own the static BSR topology and frozen smooth deformable assembly buffers."""

    def __init__(
        self,
        model: Model,
        cr_iterations: int = 4,
        weight_sigma: float = BODY_WEIGHT_SIGMA_DEFAULT,
        weight_beta: float = DEFORMABLE_WEIGHT_BETA_DEFAULT,
        preconditioner: str = "two_level",
        preconditioner_regularization: float = 1.0e-6,
        direct_max_particles: int = 0,
        proximal_iterations: int = 1,
        proximal_relaxation: float = 1.0,
    ):
        """Construct the immutable packed deformable topology.

        Args:
            model: Newton model containing the supported deformable subset.
            cr_iterations: Fixed number of preconditioned CR iterations per candidate solve.
            weight_sigma: Relative lower smooth-scale clamp for nodal weights.
            weight_beta: Upper mass-relative clamp for nodal weights.
            preconditioner: Deformable preconditioner type.
            preconditioner_regularization: Relative incomplete-factor pivot floor.
            direct_max_particles: Largest component solved with blocked Cholesky.
            proximal_iterations: Fixed local Gauss-Newton iterations per elastic element prox.
            proximal_relaxation: Relaxation factor for the local multiplier update.
        """
        validate_deformable_model(model)
        if not isinstance(cr_iterations, int) or isinstance(cr_iterations, bool) or cr_iterations < 1:
            raise ValueError(
                f"LOX deformable CR iterations must be an integer greater than or equal to one, got {cr_iterations}."
            )
        if not np.isfinite(weight_sigma) or weight_sigma <= 0.0 or weight_sigma > 1.0:
            raise ValueError(f"LOX deformable weight sigma must be in (0, 1], got {weight_sigma}.")
        if not np.isfinite(weight_beta) or weight_beta < 1.0:
            raise ValueError(f"LOX deformable weight beta must be at least one, got {weight_beta}.")
        if preconditioner not in ("incomplete_ldlt", "two_level", "block_jacobi"):
            raise ValueError(
                "LOX deformable preconditioner must be 'incomplete_ldlt', 'two_level', "
                f"or 'block_jacobi', got {preconditioner!r}."
            )
        if (
            not isinstance(direct_max_particles, int)
            or isinstance(direct_max_particles, bool)
            or direct_max_particles < 0
        ):
            raise ValueError(
                f"LOX deformable direct-solve particle limit must be a non-negative integer, got {direct_max_particles}."
            )
        if not isinstance(proximal_iterations, int) or isinstance(proximal_iterations, bool) or proximal_iterations < 0:
            raise ValueError(
                f"LOX deformable proximal iterations must be a non-negative integer, got {proximal_iterations}."
            )
        if not np.isfinite(proximal_relaxation) or proximal_relaxation < 0.0 or proximal_relaxation > 1.0:
            raise ValueError(f"LOX deformable proximal relaxation must be in [0, 1], got {proximal_relaxation}.")

        self.model = model
        self.device = model.device
        self.particle_count = int(model.particle_count)
        self.triangle_count = int(model.tri_count)
        self.tetrahedron_count = int(model.tet_count)
        self.weight_sigma = float(weight_sigma)
        self.weight_beta = float(weight_beta)
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
        component_direct_np = component_particle_counts_np <= direct_max_particles
        if direct_max_particles == 0:
            component_direct_np[:] = False
        has_direct_components = bool(np.any(component_direct_np))
        self.has_iterative_components = bool(np.any(~component_direct_np))

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

        self.topology = DeformableTopology(
            newton_to_packed=wp.array(newton_to_packed_np, dtype=wp.int32, device=self.device),
            packed_to_newton=wp.array(packed_to_newton_np, dtype=wp.int32, device=self.device),
            packed_source_world=wp.array(packed_source_world_np, dtype=wp.int32, device=self.device),
            packed_solve_world=wp.array(packed_solve_world_np, dtype=wp.int32, device=self.device),
            component_dof_offsets=wp.array(component_dof_offsets_np, dtype=wp.int32, device=self.device),
            triangle_indices=wp.array(packed_triangle_indices_np, dtype=wp.int32, device=self.device),
            bending_indices=wp.array(packed_bending_indices_np, dtype=wp.int32, device=self.device),
            source_edge_indices=wp.array(source_edge_indices_np, dtype=wp.int32, device=self.device),
            tetrahedron_indices=wp.array(packed_tetrahedron_indices_np, dtype=wp.int32, device=self.device),
        )

        self.triangle_triplet_offset = self.particle_count
        self.bending_triplet_offset = self.triangle_triplet_offset + 9 * self.triangle_count
        self.tetrahedron_triplet_offset = self.bending_triplet_offset + 16 * self.bending_count
        self.triplet_count = self.tetrahedron_triplet_offset + 16 * self.tetrahedron_count

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

        self.triplet_rows = wp.array(triplet_rows_np, dtype=wp.int32, device=self.device)
        self.triplet_columns = wp.array(triplet_columns_np, dtype=wp.int32, device=self.device)
        self.triplet_values = wp.zeros(self.triplet_count, dtype=wp.mat33, device=self.device)
        self.tetrahedron_metric = wp.zeros(self.tetrahedron_count, dtype=mat99, device=self.device)

        structural_coordinates = np.unique(
            np.column_stack((triplet_rows_np, triplet_columns_np)),
            axis=0,
        ).astype(np.int32, copy=False)
        structural_rows = wp.array(structural_coordinates[:, 0], dtype=wp.int32, device=self.device)
        structural_columns = wp.array(structural_coordinates[:, 1], dtype=wp.int32, device=self.device)
        structural_values = wp.zeros(structural_coordinates.shape[0], dtype=wp.mat33, device=self.device)
        self.smooth_matrix = wps.bsr_zeros(
            self.particle_count,
            self.particle_count,
            wp.mat33,
            device=self.device,
        )
        wps.bsr_set_from_triplets(
            self.smooth_matrix,
            structural_rows,
            structural_columns,
            structural_values,
            prune_numerical_zeros=False,
            topology="compact",
        )

        matrix_offsets = self.smooth_matrix.offsets.numpy()
        matrix_columns = self.smooth_matrix.columns.numpy()
        diagonal_slots_np = np.empty(self.particle_count, dtype=np.int32)
        for row in range(self.particle_count):
            row_start = int(matrix_offsets[row])
            row_end = int(matrix_offsets[row + 1])
            matching = np.flatnonzero(matrix_columns[row_start:row_end] == row)
            if matching.size != 1:
                raise RuntimeError(f"LOX deformable topology has no unique diagonal block in row {row}.")
            diagonal_slots_np[row] = row_start + int(matching[0])
        self.diagonal_slots = wp.array(diagonal_slots_np, dtype=wp.int32, device=self.device)
        self.matrix_nnz = int(matrix_offsets[-1])
        self.system_matrix = wps.bsr_copy(
            self.smooth_matrix,
            structure_only=True,
            topology="compact",
        )
        self.preconditioner_matrix = wps.bsr_copy(
            self.smooth_matrix,
            structure_only=True,
            topology="compact",
        )

        self.position_start = wp.empty(self.particle_count, dtype=wp.vec3, device=self.device)
        self.velocity_start = wp.empty(self.particle_count, dtype=wp.vec3, device=self.device)
        self.external_force = wp.empty(self.particle_count, dtype=wp.vec3, device=self.device)
        self.position_linearized = wp.empty(self.particle_count, dtype=wp.vec3, device=self.device)
        self.velocity_linearized = wp.empty(self.particle_count, dtype=wp.vec3, device=self.device)
        self.smooth_force = wp.empty(self.particle_count, dtype=wp.vec3, device=self.device)
        self.matrix_velocity = wp.empty(self.particle_count, dtype=wp.vec3, device=self.device)
        self.smooth_rhs = wp.empty(self.particle_count, dtype=wp.vec3, device=self.device)
        self.full_weight = wp.empty(self.particle_count, dtype=wp.float32, device=self.device)
        self.full_inverse_weight = wp.empty(self.particle_count, dtype=wp.float32, device=self.device)
        self.weight = wp.empty(self.particle_count, dtype=wp.float32, device=self.device)
        self.inverse_weight = wp.empty(self.particle_count, dtype=wp.float32, device=self.device)
        self.weight_status = wp.empty(self.particle_count, dtype=wp.int32, device=self.device)
        self.unilateral_incidence = wp.zeros(self.particle_count, dtype=wp.int32, device=self.device)
        self.consensus_enabled = wp.ones(self.particle_count, dtype=wp.int32, device=self.device)
        self.selective_consensus = False
        self.world_active = wp.ones(model.world_count, dtype=wp.int32, device=self.device)
        self.candidate_rhs = wp.empty(self.particle_count, dtype=wp.vec3, device=self.device)
        self.smooth_velocity = wp.empty(self.particle_count, dtype=wp.vec3, device=self.device)
        self._system_matvec_scalar_views: dict[int, wp.array] = {}
        self.nonlinear_rhs = wp.zeros(self.particle_count, dtype=wp.vec3, device=self.device)
        self.proximal_position_residual = wp.zeros(model.world_count, dtype=wp.float32, device=self.device)
        self.proximal_velocity_residual = wp.zeros(model.world_count, dtype=wp.float32, device=self.device)
        self.proximal_failed = wp.zeros(model.world_count, dtype=wp.int32, device=self.device)
        self.membrane_proximal = (
            DeformableMembraneProximal(self, proximal_iterations, proximal_relaxation)
            if self.triangle_count > 0 and proximal_iterations > 0 and proximal_relaxation > 0.0
            else None
        )
        self.tetrahedron_proximal = (
            DeformableTetrahedronProximal(self, proximal_iterations, proximal_relaxation)
            if self.tetrahedron_count > 0 and proximal_iterations > 0 and proximal_relaxation > 0.0
            else None
        )

        self.system_operator = wpl.LinearOperator(
            shape=self.system_matrix.shape,
            dtype=self.system_matrix.dtype,
            device=self.device,
            matvec=self._system_matvec,
            batch_offsets=self.topology.component_dof_offsets,
        )
        self.packed_iterative = wp.array(packed_iterative_np, dtype=wp.int32, device=self.device)
        self.direct_solver = (
            DeformableBlockLLT(
                self.system_matrix,
                packed_component_np,
                packed_iterative_np,
                self.topology.packed_solve_world,
                self.world_active,
                component_count,
            )
            if has_direct_components
            else None
        )
        if self.has_iterative_components and preconditioner == "incomplete_ldlt":
            self.preconditioner = DeformableIncompleteLDLT(
                self.preconditioner_matrix,
                self.topology.packed_solve_world,
                self.world_active,
                self.topology.component_dof_offsets,
                regularization=preconditioner_regularization,
                row_active=self.packed_iterative,
            )
        elif self.has_iterative_components and preconditioner == "two_level":
            self.preconditioner = DeformableTwoLevel(
                self.preconditioner_matrix,
                self.diagonal_slots,
                packed_component_np,
                self.topology.packed_solve_world,
                self.world_active,
                self.topology.component_dof_offsets,
                regularization=preconditioner_regularization,
                row_active=self.packed_iterative,
            )
        elif self.has_iterative_components:
            self.preconditioner = DeformableJacobi(
                self.preconditioner_matrix,
                self.diagonal_slots,
                self.topology.packed_solve_world,
                self.world_active,
                self.topology.component_dof_offsets,
                regularization=preconditioner_regularization,
                row_active=self.packed_iterative,
            )
        else:
            assert self.direct_solver is not None
            self.preconditioner = self.direct_solver
        self.cr_state = (
            None
            if not self.has_iterative_components
            else wpl.CR(
                self.system_operator,
                self.candidate_rhs,
                self.smooth_velocity,
                tol=0.0,
                atol=0.0,
                maxiter=cr_iterations,
                M=self.preconditioner.linear_operator,
                check_every=0,
                use_cuda_graph=False,
            )
        )

    def assemble(
        self,
        state: State,
        time_step: wp.array[wp.float32],
        finalize_consensus: bool = True,
    ) -> None:
        """Assemble the frozen smooth matrix and affine right-hand side.

        Args:
            state: Beginning-of-step Newton particle state.
            time_step: Per-world simulation time steps [s].
            finalize_consensus: Whether to build and factor the weighted system immediately.

        Raises:
            ValueError: If the time step or particle state is invalid.
        """
        validate_world_time_step(time_step, self.model.world_count, self.device)
        if state.particle_q is None or state.particle_qd is None or state.particle_f is None:
            raise ValueError("LOX deformables require particle positions, velocities, and forces in the input state.")
        expected_shape = (self.particle_count,)
        for name in ("particle_q", "particle_qd", "particle_f"):
            value = getattr(state, name)
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"LOX deformables expected State.{name} shape {expected_shape}, found {value.shape}.")
            if value.device != self.device:
                raise ValueError(f"LOX deformables expected State.{name} on {self.device}, found {value.device}.")

        wp.launch(
            gather_particle_state,
            dim=self.particle_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                state.particle_f,
                self.model.particle_mass,
                self.model.particle_flags,
                self.model.gravity,
                self.topology.packed_to_newton,
                self.topology.packed_source_world,
            ],
            outputs=[
                self.position_start,
                self.velocity_start,
                self.external_force,
            ],
            device=self.device,
        )
        self._assemble_linearization(self.velocity_start, time_step, finalize_consensus)

    def _assemble_linearization(
        self,
        linearization_velocity: wp.array[wp.vec3],
        time_step: wp.array[wp.float32],
        finalize_consensus: bool = True,
    ) -> None:
        """Assemble and factor one deformable linearization with fixed step-start data."""
        wp.launch(
            set_particle_linearization,
            dim=self.particle_count,
            inputs=[
                self.position_start,
                linearization_velocity,
                self.external_force,
                self.topology.packed_to_newton,
                self.topology.packed_solve_world,
                self.model.particle_flags,
                time_step,
            ],
            outputs=[
                self.position_linearized,
                self.velocity_linearized,
                self.smooth_force,
            ],
            device=self.device,
        )
        wp.launch(
            assemble_particle_mass,
            dim=self.particle_count,
            inputs=[
                self.topology.packed_to_newton,
                self.model.particle_mass,
                self.model.particle_flags,
            ],
            outputs=[self.triplet_values],
            device=self.device,
        )
        if self.triangle_count > 0:
            wp.launch(
                assemble_triangle_system,
                dim=self.triangle_count,
                inputs=[
                    self.position_start,
                    self.position_linearized,
                    self.velocity_linearized,
                    self.topology.triangle_indices,
                    self.model.tri_poses,
                    self.model.tri_areas,
                    self.model.tri_materials,
                    self.model.tri_activations,
                    self.topology.packed_to_newton,
                    self.topology.packed_solve_world,
                    self.model.particle_mass,
                    self.model.particle_flags,
                    time_step,
                    self.triangle_triplet_offset,
                ],
                outputs=[self.triplet_values, self.smooth_force],
                device=self.device,
            )
        if self.bending_count > 0:
            wp.launch(
                assemble_bending_system,
                dim=self.bending_count,
                inputs=[
                    self.position_start,
                    self.position_linearized,
                    self.topology.bending_indices,
                    self.topology.source_edge_indices,
                    self.model.edge_rest_angle,
                    self.model.edge_rest_length,
                    self.model.edge_bending_properties,
                    self.topology.packed_to_newton,
                    self.topology.packed_solve_world,
                    self.model.particle_mass,
                    self.model.particle_flags,
                    time_step,
                    self.bending_triplet_offset,
                ],
                outputs=[self.triplet_values, self.smooth_force],
                device=self.device,
            )
        if self.tetrahedron_count > 0:
            wp.launch(
                assemble_tetrahedron_system,
                dim=self.tetrahedron_count,
                inputs=[
                    self.position_start,
                    self.position_linearized,
                    self.topology.tetrahedron_indices,
                    self.model.tet_poses,
                    self.model.tet_materials,
                    self.model.tet_activations,
                    self.topology.packed_to_newton,
                    self.topology.packed_solve_world,
                    self.model.particle_mass,
                    self.model.particle_flags,
                    time_step,
                    self.tetrahedron_triplet_offset,
                ],
                outputs=[self.tetrahedron_metric, self.triplet_values, self.smooth_force],
                device=self.device,
            )

        wps.bsr_set_from_triplets(
            self.smooth_matrix,
            self.triplet_rows,
            self.triplet_columns,
            self.triplet_values,
            prune_numerical_zeros=False,
            topology="masked",
        )
        wps.bsr_mv(
            self.smooth_matrix,
            x=self.velocity_linearized,
            y=self.matrix_velocity,
            alpha=1.0,
            beta=0.0,
        )
        wp.launch(
            finish_smooth_rhs,
            dim=self.particle_count,
            inputs=[
                self.matrix_velocity,
                self.smooth_force,
                self.velocity_start,
                self.velocity_linearized,
                self.topology.packed_to_newton,
                self.topology.packed_solve_world,
                self.model.particle_mass,
                self.model.particle_flags,
                time_step,
            ],
            outputs=[self.smooth_rhs],
            device=self.device,
        )
        wp.launch(
            compute_consensus_weight,
            dim=self.particle_count,
            inputs=[
                self.smooth_matrix.values,
                self.diagonal_slots,
                self.topology.packed_to_newton,
                self.model.particle_mass,
                self.model.particle_flags,
                self.weight_sigma,
                self.weight_beta,
            ],
            outputs=[self.full_weight, self.full_inverse_weight, self.weight_status],
            device=self.device,
        )
        if finalize_consensus:
            self._finalize_consensus_system()
        wp.copy(
            dest=self.smooth_velocity,
            src=self.velocity_linearized,
            count=self.particle_count,
        )
        self.nonlinear_rhs.zero_()
        self.proximal_position_residual.zero_()
        self.proximal_velocity_residual.zero_()
        self.proximal_failed.zero_()
        if self.membrane_proximal is not None:
            self.membrane_proximal.initialize()
        if self.tetrahedron_proximal is not None:
            self.tetrahedron_proximal.initialize()

    def _finalize_consensus_system(self) -> None:
        """Apply the consensus support and refactor the operator and preconditioner."""
        wp.launch(
            select_consensus_weight,
            dim=self.particle_count,
            inputs=[
                self.full_weight,
                self.full_inverse_weight,
                self.unilateral_incidence,
                self.selective_consensus,
            ],
            outputs=[self.consensus_enabled, self.weight, self.inverse_weight],
            device=self.device,
        )
        wp.copy(
            dest=self.system_matrix.values,
            src=self.smooth_matrix.values,
            count=self.matrix_nnz,
        )
        wp.launch(
            add_consensus_weight,
            dim=self.particle_count,
            inputs=[self.diagonal_slots, self.weight],
            outputs=[self.system_matrix.values],
            device=self.device,
        )
        wp.copy(
            dest=self.preconditioner_matrix.values,
            src=self.smooth_matrix.values,
            count=self.matrix_nnz,
        )
        wp.launch(
            add_consensus_weight,
            dim=self.particle_count,
            inputs=[self.diagonal_slots, self.full_weight],
            outputs=[self.preconditioner_matrix.values],
            device=self.device,
        )
        if self.direct_solver is not None:
            self.direct_solver.factorize()
        if self.has_iterative_components:
            self.preconditioner.factorize()

    def set_unilateral_incidence(self, incidence: wp.array[wp.int32] | None) -> None:
        """Freeze nodal unilateral support and rebuild the weighted linear system."""
        if incidence is None:
            self.unilateral_incidence.zero_()
        else:
            if incidence.shape != (self.particle_count,) or incidence.dtype != wp.int32:
                raise ValueError("LOX deformable incidence must contain one int32 entry per packed particle.")
            if incidence.device != self.device:
                raise ValueError(f"LOX deformable expected incidence on {self.device}, found {incidence.device}.")
            wp.copy(self.unilateral_incidence, incidence)
        self._finalize_consensus_system()

    def update_proximal(self, time_step: wp.array[wp.float32]) -> None:
        """Evaluate nonlinear elastic proxes for the current global velocity."""
        if self.membrane_proximal is None and self.tetrahedron_proximal is None:
            return
        self.nonlinear_rhs.zero_()
        self.proximal_position_residual.zero_()
        self.proximal_velocity_residual.zero_()
        self.proximal_failed.zero_()
        if self.membrane_proximal is not None:
            self.membrane_proximal.update(time_step)
        if self.tetrahedron_proximal is not None:
            self.tetrahedron_proximal.update(time_step)

    def _system_matvec(
        self,
        x: wp.array[wp.vec3],
        y: wp.array[wp.vec3],
        z: wp.array[wp.vec3],
        alpha: float,
        beta: float,
    ) -> None:
        matrix = self.system_matrix
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
            kernel=_masked_system_matvec,
            dim=(matrix.nrow, 3),
            inputs=[
                matrix.offsets,
                matrix.columns,
                matrix.scalar_values,
                x_scalar,
                y_scalar,
                self.topology.packed_solve_world,
                self.packed_iterative,
                self.world_active,
                alpha,
                beta,
            ],
            outputs=[z_scalar],
            block_dim=_SYSTEM_MATVEC_BLOCK_DIM,
            device=self.device,
        )

    def solve_candidate(
        self,
        consensus_center: wp.array[wp.vec3],
        world_active: wp.array[wp.int32] | None = None,
    ) -> None:
        """Run the fixed-count warm-started preconditioned CR candidate solve.

        Args:
            consensus_center: Current ``p + lambda`` nodal center [m/s].
            world_active: Optional per-world active mask copied into stable storage.

        """
        if consensus_center.shape != (self.particle_count,) or consensus_center.dtype != wp.vec3:
            raise ValueError("LOX deformable consensus center must be a vec3 array with one entry per packed particle.")
        if consensus_center.device != self.device:
            raise ValueError(
                f"LOX deformable expected consensus center on {self.device}, found {consensus_center.device}."
            )
        if world_active is not None:
            if world_active.shape != self.world_active.shape or world_active.dtype != wp.int32:
                raise ValueError(
                    f"LOX deformable world active mask must have shape {self.world_active.shape} and dtype int32."
                )
            if world_active.device != self.device:
                raise ValueError(
                    f"LOX deformable expected world active mask on {self.device}, found {world_active.device}."
                )
            wp.copy(dest=self.world_active, src=world_active, count=self.world_active.shape[0])

        wp.launch(
            prepare_candidate_rhs,
            dim=self.particle_count,
            inputs=[
                self.smooth_rhs,
                self.nonlinear_rhs,
                consensus_center,
                self.weight,
                self.smooth_velocity,
                self.topology.packed_solve_world,
                self.world_active,
            ],
            outputs=[self.candidate_rhs],
            device=self.device,
        )
        if self.direct_solver is not None:
            self.direct_solver.solve(self.candidate_rhs, self.smooth_velocity)
            wp.launch(
                retain_direct_candidate_rhs,
                dim=self.particle_count,
                inputs=[self.smooth_velocity, self.packed_iterative],
                outputs=[self.candidate_rhs],
                device=self.device,
            )
        if not self.has_iterative_components:
            return

        assert self.cr_state is not None
        self.cr_state(M=self.preconditioner.linear_operator)
