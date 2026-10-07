# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Experimental smooth density measurements for the Baumgarte controller."""

from enum import Enum

import warp as wp
import warp.fem as fem
from warp.fem import cache

import newton

from .rasterized_collisions import Collider, collision_sdf

wp.set_module_options({"enable_backward": False, "max_unroll": 0})


class DensityMeasurement(Enum):
    """Experimental particle-volume measurement used for divergence feedback.

    Access through ``SolverImplicitMPM.DensityMeasurement``. Smooth modes
    require ordinary PIC integration and use the selected strain basis.
    They affect only the density controller; mass, strain and collision
    operators retain their usual quadrature. These modes may change without
    prior notice.
    """

    CELL = "cell"
    BOX = "box"
    BSPLINE = "bspline"
    PARTICLE = "particle"


@wp.func
def quadratic_bspline(x: float):
    x = wp.abs(x)
    if x < 0.5:
        return 0.75 - x * x
    if x < 1.5:
        return 0.5 * (1.5 - x) * (1.5 - x)
    return 0.0


@wp.func
def box_overlap(x: float, radius: float, center: float, h: float):
    return wp.max(0.0, wp.min(x + radius, center + 0.5 * h) - wp.max(x - radius, center - 0.5 * h))


def make_deposition_kernel(grid):
    """Scatter reference volumes; halo cells do not renormalize the weights."""

    @cache.dynamic_kernel(suffix=grid.name)
    def deposit(
        geo: grid.CellArg,
        positions: wp.array[wp.vec3],
        radii: wp.array[float],
        volumes: wp.array[float],
        flags: wp.array[int],
        environments: wp.array[int],
        h: float,
        mode: int,
        output: wp.array[float],
    ):
        p = wp.tid()
        if (flags[p] & newton.ParticleFlags.ACTIVE) == 0 or volumes[p] == 0.0:
            return
        environment = int(0)
        if environments:
            environment = environments[p]
        x = positions[p]
        local_cell = wp.vec3i(0)
        containing_center = wp.vec3(0.0)
        if wp.static(isinstance(grid, fem.Nanogrid)):
            # Deposition needs exact cell membership, not a nearest-cell search.
            uvw = wp.volume_world_to_index(geo.cell_grid, x) + wp.vec3(0.5)
            local_cell = wp.vec3i(int(wp.floor(uvw[0])), int(wp.floor(uvw[1])), int(wp.floor(uvw[2])))
            containing_center = wp.volume_index_to_world(geo.cell_grid, wp.vec3(local_cell))
        else:
            sample = grid.cell_lookup(geo, x, 0.0, environment)
            if sample.element_index < 0:
                return
            sample.element_coords = wp.vec3(0.5)
            containing_center = grid.cell_position(geo, sample)
        for a in range(-1, 2):
            for b in range(-1, 2):
                for c in range(-1, 2):
                    center = containing_center + h * wp.vec3(float(a), float(b), float(c))
                    weight = float(0.0)
                    if mode == 1:
                        overlap = box_overlap(x[0], radii[p], center[0], h)
                        overlap *= box_overlap(x[1], radii[p], center[1], h)
                        overlap *= box_overlap(x[2], radii[p], center[2], h)
                        weight = overlap / (8.0 * radii[p] * radii[p] * radii[p])
                    else:
                        q = (x - center) / h
                        weight = quadratic_bspline(q[0]) * quadratic_bspline(q[1]) * quadratic_bspline(q[2])
                    if weight > 0.0:
                        if wp.static(isinstance(grid, fem.Nanogrid)):
                            ijk = local_cell + wp.vec3i(a, b, c) + geo.env_offsets[environment]
                            cell = wp.volume_lookup_index(geo.cell_grid, ijk[0], ijk[1], ijk[2])
                            if cell >= 0 and cell < output.shape[0]:
                                if geo.cell_env[cell] == environment:
                                    wp.atomic_add(output, cell, weight * volumes[p] / (h * h * h))
                        else:
                            target = grid.cell_lookup(geo, center, 0.0, environment)
                            if target.element_index >= 0:
                                actual = grid.cell_position(geo, target)
                                # Grid3D lookups clamp at their boundary: reject clamped cells.
                                if wp.length_sq(actual - center) < 1.0e-8 * h * h:
                                    wp.atomic_add(output, target.element_index, weight * volumes[p] / (h * h * h))

    return deposit


def make_capacity_kernel(grid, domain):
    """Integrate solid exclusion through full B2 support, including empty cells.

    Midpoint quadrature uses the initial eight-particles-per-cell spacing:
    six points per axis across the three-cell support. It reproduces the
    reference lattice at grid-aligned planar walls. Static capacities are
    reused by cell position across sparse-grid rebuilds.
    """

    @cache.dynamic_kernel(suffix=f"{grid.name}_{domain.name}")
    def capacity(
        geo: grid.CellArg,
        previous_geo: grid.CellArg,
        domain_indices: domain.ElementIndexArg,
        previous_capacity: wp.array[float],
        collider: Collider,
        body_q: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector],
        body_q_prev: wp.array[wp.transform],
        separate_worlds: bool,
        reuse: bool,
        h: float,
        output: wp.array[float],
    ):
        cell = domain.element_index(domain_indices, wp.tid())
        if wp.static(isinstance(grid, fem.Nanogrid)):
            if cell < 0 or cell >= wp.volume_voxel_count(geo.cell_grid):
                return
        sample = fem.make_free_sample(cell, wp.vec3(0.5))
        center = grid.cell_position(geo, sample)
        environment = grid.cell_environment_index(geo, sample)
        collider_environment = wp.where(separate_worlds, environment, -2)
        if reuse:
            old = grid.cell_lookup(previous_geo, center, 0.0, environment)
            if old.element_index >= 0:
                actual = grid.cell_position(previous_geo, old)
                if wp.length_sq(actual - center) < 1.0e-8 * h * h:
                    cached = previous_capacity[old.element_index]
                    if cached >= 0.0:
                        output[cell] = cached
                        return
        sdf, _grad, _vel, _collider_id, _material_id = collision_sdf(
            center, collider_environment, collider, body_q, body_qd, body_q_prev, 1.0
        )
        support_radius = 2.598077 * h
        if sdf > support_radius:
            output[cell] = 1.0
            return
        if sdf < -support_radius:
            output[cell] = 0.0
            return
        available = float(0.0)
        for a in range(6):
            for b in range(6):
                for c in range(6):
                    q = wp.vec3(0.5 * float(a) - 1.25, 0.5 * float(b) - 1.25, 0.5 * float(c) - 1.25)
                    phi, _n, _v, _cid, _mid = collision_sdf(
                        center + h * q, collider_environment, collider, body_q, body_qd, body_q_prev, 1.0
                    )
                    if phi >= 0.0:
                        available += 0.125 * quadratic_bspline(q[0]) * quadratic_bspline(q[1]) * quadratic_bspline(q[2])
        output[cell] = available

    return capacity


@wp.kernel
def add_available_density_offset(
    node_volume: wp.array[float],
    collider_volume: wp.array[float],
    fraction: float,
    output: wp.array[float],
):
    node = wp.tid()
    output[node] += fraction * (node_volume[node] - collider_volume[node])


@fem.integrand
def cell_density_offset(
    s: fem.Sample,
    tau: fem.Field,
    measured_volume: wp.array[float],
    available_volume: wp.array[float],
    fraction: float,
    mode: int,
    inv_cell_volume: float,
):
    """Project the cell density signal into the actual strain test space."""
    available = float(0.0)
    if mode == 2:
        available = available_volume[s.element_index]
    error = available - measured_volume[s.element_index]
    if mode == 2:
        error = wp.min(error, 0.0)
    return fraction * error * tau(s) * inv_cell_volume


@wp.func
def reference_cell_mask(
    center: wp.vec3,
    collider_environment: int,
    collider: Collider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    h: float,
):
    """Evaluate the existing eight-site wall quadrature for one cell."""
    sdf, _n, _v, _cid, _mid = collision_sdf(center, collider_environment, collider, body_q, body_qd, body_q_prev, 1.0)
    if sdf > 0.866026 * h:
        return wp.uint32(255)
    if sdf < -0.866026 * h:
        return wp.uint32(0)
    mask = wp.uint32(0)
    for k in range(8):
        q = wp.vec3(float(k // 4) - 0.5, float((k // 2) % 2) - 0.5, float(k % 2) - 0.5)
        phi, _normal, _velocity, _collider, _material = collision_sdf(
            center + 0.5 * h * q, collider_environment, collider, body_q, body_qd, body_q_prev, 1.0
        )
        if phi >= 0.0:
            mask |= wp.uint32(1) << wp.uint32(k)
    return mask


def make_reference_mask_kernel(grid, *, center_only=False):
    """Cache solid exclusion at eight fixed geometric quadrature sites per cell."""

    @cache.dynamic_kernel(suffix=f"{grid.name}_{center_only}")
    def reference_mask(
        geo: grid.CellArg,
        previous_geo: grid.CellArg,
        previous_mask: wp.array[wp.uint32],
        collider: Collider,
        body_q: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector],
        body_q_prev: wp.array[wp.transform],
        separate_worlds: bool,
        reuse: bool,
        h: float,
        output: wp.array[wp.uint32],
    ):
        cell = wp.tid()
        if wp.static(isinstance(grid, fem.Nanogrid)):
            if cell >= wp.volume_voxel_count(geo.cell_grid):
                output[cell] = wp.uint32(0)
                return
        sample = fem.make_free_sample(cell, wp.vec3(0.5))
        center = grid.cell_position(geo, sample)
        environment = grid.cell_environment_index(geo, sample)
        if reuse:
            old = grid.cell_lookup(previous_geo, center, 0.0, environment)
            if old.element_index >= 0:
                actual = grid.cell_position(previous_geo, old)
                if wp.length_sq(actual - center) < 1.0e-8 * h * h:
                    output[cell] = previous_mask[old.element_index]
                    return
        collider_environment = wp.where(separate_worlds, environment, -2)
        if wp.static(center_only):
            sdf, _n, _v, _cid, _mid = collision_sdf(
                center, collider_environment, collider, body_q, body_qd, body_q_prev, 1.0
            )
            output[cell] = wp.uint32(256)
            if sdf > 0.866026 * h:
                output[cell] = wp.uint32(255)
            elif sdf < -0.866026 * h:
                output[cell] = wp.uint32(0)
        else:
            output[cell] = reference_cell_mask(center, collider_environment, collider, body_q, body_qd, body_q_prev, h)

    return reference_mask


def make_reference_samples_kernel(grid):
    """Evaluate the same eight wall sites concurrently for boundary cells."""

    @cache.dynamic_kernel(suffix=grid.name)
    def reference_samples(
        geo: grid.CellArg,
        collider: Collider,
        body_q: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector],
        body_q_prev: wp.array[wp.transform],
        separate_worlds: bool,
        h: float,
        masks: wp.array[wp.uint32],
    ):
        cell, k = wp.tid()
        # The marker remains set until all eight sites have finished.
        if masks[cell] & wp.uint32(256):
            sample = fem.make_free_sample(cell, wp.vec3(0.5))
            center = grid.cell_position(geo, sample)
            environment = grid.cell_environment_index(geo, sample)
            q = wp.vec3(float(k // 4) - 0.5, float((k // 2) % 2) - 0.5, float(k % 2) - 0.5)
            phi, _n, _v, _cid, _mid = collision_sdf(
                center + 0.5 * h * q,
                wp.where(separate_worlds, environment, -2),
                collider,
                body_q,
                body_qd,
                body_q_prev,
                1.0,
            )
            if phi >= 0.0:
                wp.atomic_or(masks, cell, wp.uint32(1) << wp.uint32(k))

    return reference_samples


@wp.kernel
def finalize_reference_masks(masks: wp.array[wp.uint32]):
    i = wp.tid()
    masks[i] = masks[i] & wp.uint32(255)


def make_density_support_points_kernel(grid):
    """Emit the two-cell reference halo around occupied sparse cells."""

    @cache.dynamic_kernel(suffix=grid.name)
    def support_points(
        geo: grid.CellArg,
        points: wp.array[wp.vec3i],
        environments: wp.array[int],
        valid: wp.array[int],
    ):
        i = wp.tid()
        cell = i // 125
        if cell >= wp.volume_voxel_count(geo.cell_grid):
            valid[i] = 0
            return
        environment = geo.cell_env[cell]
        k = i % 125
        offset = wp.vec3i(k // 25 - 2, (k // 5) % 5 - 2, k % 5 - 2)
        points[i] = geo.cell_ijk[cell] - geo.env_offsets[environment] + offset
        environments[i] = environment
        valid[i] = 1

    return support_points


def make_density_neighbor_kernel(pic, support_grid):
    """Resolve the shared particle bins and wall masks once per sparse cell."""
    domain = pic.domain
    grid = domain.geometry
    domain_indices = pic._use_domain_element_indices

    @cache.dynamic_kernel(suffix=f"{grid.name}_{domain.name}_{domain_indices}")
    def density_neighbors(
        geo: grid.CellArg,
        domain_arg: domain.ElementIndexArg,
        support_geo: grid.CellArg,
        reference_mask: wp.array[wp.uint32],
        neighbor_cells: wp.array2d[int],
        neighbor_masks: wp.array2d[wp.uint32],
    ):
        i = wp.tid()
        cell, k = i // 125, i % 125
        if cell >= wp.volume_voxel_count(geo.cell_grid):
            return
        environment = geo.cell_env[cell]
        offset = wp.vec3i(k // 25 - 2, (k // 5) % 5 - 2, k % 5 - 2)
        local = geo.cell_ijk[cell] - geo.env_offsets[environment] + offset
        ijk = local + geo.env_offsets[environment]
        neighbor = wp.volume_lookup_index(geo.cell_grid, ijk[0], ijk[1], ijk[2])
        if neighbor >= 0:
            if geo.cell_env[neighbor] != environment:
                neighbor = -1
            elif wp.static(domain_indices):
                neighbor = domain.element_partition_index(domain_arg, neighbor)
        neighbor_cells[cell, k] = neighbor
        ijk = local + support_geo.env_offsets[environment]
        reference = wp.volume_lookup_index(support_geo.cell_grid, ijk[0], ijk[1], ijk[2])
        mask = wp.uint32(0)
        if reference >= 0:
            mask = reference_mask[reference]
        neighbor_masks[cell, k] = mask

    return density_neighbors


def make_particle_density_kernel(pic, support_grid=None, *, cache_neighbors=False):
    """Gather neighbors directly from the existing MPM PicQuadrature cell lists."""
    domain = pic.domain
    grid = domain.geometry
    domain_indices = pic._use_domain_element_indices

    @cache.dynamic_kernel(
        suffix=f"{grid.name}_{domain.name}_{domain_indices}_{support_grid is not None}_{cache_neighbors}"
    )
    def particle_density(
        geo: grid.CellArg,
        domain_arg: domain.ElementIndexArg,
        particle_offsets: wp.array[int],
        particle_indices: wp.array[int],
        positions: wp.array[wp.vec3],
        volumes: wp.array[float],
        flags: wp.array[int],
        environments: wp.array[int],
        reference_mask: wp.array[wp.uint32],
        h: float,
        collider: Collider,
        body_q: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector],
        body_q_prev: wp.array[wp.transform],
        separate_worlds: bool,
        support_geo: grid.CellArg,
        neighbor_cells: wp.array2d[int],
        neighbor_masks: wp.array2d[wp.uint32],
        density: wp.array[float],
        capacity: wp.array[float],
        error: wp.array[float],
    ):
        p = wp.tid()
        if (flags[p] & newton.ParticleFlags.ACTIVE) == 0 or volumes[p] == 0.0:
            density[p] = 0.0
            capacity[p] = 0.0
            error[p] = 0.0
            return
        environment = int(0)
        if environments:
            environment = environments[p]
        x = positions[p]
        local_cell = wp.vec3i(0)
        containing_center = wp.vec3(0.0)
        if wp.static(isinstance(grid, fem.Nanogrid)):
            uvw = wp.volume_world_to_index(geo.cell_grid, x) + wp.vec3(0.5)
            local_cell = wp.vec3i(int(wp.floor(uvw[0])), int(wp.floor(uvw[1])), int(wp.floor(uvw[2])))
            containing_center = wp.volume_index_to_world(geo.cell_grid, wp.vec3(local_cell))
        else:
            sample = grid.cell_lookup(geo, x, 0.0, environment)
            if sample.element_index < 0:
                density[p] = 0.0
                capacity[p] = 0.0
                error[p] = 0.0
                return
            sample.element_coords = wp.vec3(0.5)
            containing_center = grid.cell_position(geo, sample)
        base_cell = int(-1)
        if wp.static(cache_neighbors):
            base_ijk = local_cell + geo.env_offsets[environment]
            base_cell = wp.volume_lookup_index(geo.cell_grid, base_ijk[0], base_ijk[1], base_ijk[2])
        coordinates = (x - containing_center) / h + wp.vec3(0.5)
        lower = wp.vec3i(
            int(wp.floor(coordinates[0] - 1.5)),
            int(wp.floor(coordinates[1] - 1.5)),
            int(wp.floor(coordinates[2] - 1.5)),
        )
        rho = float(0.0)
        available = float(0.0)
        inv_volume = 1.0 / (h * h * h)
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    center = containing_center + h * wp.vec3(
                        float(lower[0] + a), float(lower[1] + b), float(lower[2] + c)
                    )
                    cell = int(-1)
                    qp_cell = int(-1)
                    neighbor_index = ((lower[0] + a + 2) * 5 + lower[1] + b + 2) * 5 + lower[2] + c + 2
                    if wp.static(cache_neighbors):
                        if base_cell >= 0:
                            qp_cell = neighbor_cells[base_cell, neighbor_index]
                    else:
                        cell = int(-1)
                        if wp.static(isinstance(grid, fem.Nanogrid)):
                            ijk = local_cell + lower + wp.vec3i(a, b, c) + geo.env_offsets[environment]
                            cell = wp.volume_lookup_index(geo.cell_grid, ijk[0], ijk[1], ijk[2])
                            if cell >= 0:
                                if geo.cell_env[cell] != environment:
                                    cell = -1
                        else:
                            target = grid.cell_lookup(geo, center, 0.0, environment)
                            if target.element_index < 0:
                                continue
                            actual = grid.cell_position(geo, target)
                            if wp.length_sq(actual - center) >= 1.0e-8 * h * h:
                                continue
                            cell = target.element_index
                        qp_cell = cell
                        if wp.static(domain_indices):
                            if cell >= 0:
                                qp_cell = domain.element_partition_index(domain_arg, cell)
                    # Halo cells may have no bin in the active PIC partition.
                    if qp_cell >= 0 and qp_cell + 1 < particle_offsets.shape[0]:
                        for entry in range(particle_offsets[qp_cell], particle_offsets[qp_cell + 1]):
                            q = particle_indices[entry]
                            if q >= 0 and q < volumes.shape[0]:
                                if flags[q] & newton.ParticleFlags.ACTIVE and volumes[q] > 0.0:
                                    offset = (x - positions[q]) / h
                                    rho += (
                                        volumes[q]
                                        * inv_volume
                                        * quadratic_bspline(offset[0])
                                        * quadratic_bspline(offset[1])
                                        * quadratic_bspline(offset[2])
                                    )
                    mask = wp.uint32(0)
                    if wp.static(cache_neighbors):
                        if base_cell >= 0:
                            mask = neighbor_masks[base_cell, neighbor_index]
                    elif wp.static(support_grid is not None):
                        ijk = local_cell + lower + wp.vec3i(a, b, c) + support_geo.env_offsets[environment]
                        reference_cell = wp.volume_lookup_index(support_geo.cell_grid, ijk[0], ijk[1], ijk[2])
                        if reference_cell >= 0:
                            mask = reference_mask[reference_cell]
                    elif cell >= 0:
                        mask = reference_mask[cell]
                    else:
                        # Missing sparse cells contain no particles, but their
                        # wall quadrature still contributes to kernel capacity.
                        mask = reference_cell_mask(
                            center,
                            wp.where(separate_worlds, environment, -2),
                            collider,
                            body_q,
                            body_qd,
                            body_q_prev,
                            h,
                        )
                    offset = (x - center) / h
                    if mask == wp.uint32(255):
                        wx = quadratic_bspline(offset[0] - 0.25) + quadratic_bspline(offset[0] + 0.25)
                        wy = quadratic_bspline(offset[1] - 0.25) + quadratic_bspline(offset[1] + 0.25)
                        wz = quadratic_bspline(offset[2] - 0.25) + quadratic_bspline(offset[2] + 0.25)
                        available += 0.125 * wx * wy * wz
                    elif mask != wp.uint32(0):
                        for k in range(8):
                            if mask & (wp.uint32(1) << wp.uint32(k)):
                                location = wp.vec3(float(k // 4) - 0.5, float((k // 2) % 2) - 0.5, float(k % 2) - 0.5)
                                delta = offset - 0.5 * location
                                available += (
                                    0.125
                                    * quadratic_bspline(delta[0])
                                    * quadratic_bspline(delta[1])
                                    * quadratic_bspline(delta[2])
                                )
        density[p] = rho
        capacity[p] = available
        # Integrating Vp*(A/rho - 1) reproduces beta*(Acell - Vcell)
        # for a uniform density. Only the measurement becomes particle-centered.
        error[p] = wp.min(available / wp.max(rho, 1.0e-12) - 1.0, 0.0)

    return particle_density


@fem.integrand
def particle_density_offset(
    s: fem.Sample,
    tau: fem.Field,
    error: wp.array[float],
    flags: wp.array[int],
    fraction: float,
    inv_cell_volume: float,
):
    if (flags[s.qp_index] & newton.ParticleFlags.ACTIVE) == 0:
        return 0.0
    return fraction * error[s.qp_index] * tau(s) * inv_cell_volume


@wp.kernel
def add_overpacking_offset(measured: wp.array[float], output: wp.array[float]):
    """Add only the expansive part of the assembled BOX density correction."""
    i = wp.tid()
    output[i] += wp.min(measured[i], 0.0)
