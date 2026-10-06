# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed per-cell quadrature and transfers for the implicit MPM solver.

Strain-space integrals (strain operator, compliance, elastic right-hand side,
strain node volumes and yield parameters) are evaluated at points with fixed
positions relative to the grid cells, so that velocity shape-function gradients
are only sampled where they are smooth. The points and their weights form a
:class:`warp.fem.ExplicitQuadrature`.

Particles carry history only. Their data is moved to the points with trilinear
hat weights defined on the lattice formed by all points, which are continuous
in particle position, and each point's quadrature weight is its transferred
particle volume. Results return to particles with the same weights. Mass,
momentum, particle advection and particle-based contact rows may also go
through the points ("two-hop" transfers), so that velocity modes invisible at
the points stay invisible to particles.

With one point per cell, the lattice is the dual grid of cell centers, as in
MPM Lite (https://arxiv.org/abs/2602.07853). With two points per axis, the
points are the 2x2x2 Gauss-Legendre points of each cell, as in PQMPM
(https://doi.org/10.1002/nme.6588).
"""

import warp as wp
import warp.fem as fem
import warp.sparse as wps

import newton

from .implicit_mpm_model import MaterialParameters
from .implicit_mpm_solver_kernels import ElasticityInputs
from .material_kernels import (
    extract_elastic_parameters,
    get_elastic_parameters,
    get_yield_parameters,
    hencky_strain,
    stress_strain_relationship,
    update_particle_history,
)
from .rheology_solver_kernels import YieldParamVec

wp.set_module_options({"enable_backward": False})

POINT_STENCIL_SIZE = wp.constant(8)
"""Number of lattice points supporting each particle's trilinear weights."""

_POINT_COORDS_TOLERANCE = wp.constant(1.0e-3)
"""Tolerance on recovered cell coordinates when validating a lattice point lookup."""


def cell_quadrature_abscissae(points_per_axis: int) -> list[float]:
    """Return the sorted per-axis Gauss-Legendre cell coordinates of the fixed quadrature points."""
    if points_per_axis not in (1, 2):
        raise ValueError(f"Unsupported number of cell quadrature points per axis: {points_per_axis}")
    coords, _weights = fem.polynomial.quadrature_1d(points_per_axis, fem.Polynomial.GAUSS_LEGENDRE)
    return sorted(float(c) for c in coords)


@wp.kernel
def transfer_particles_to_points(
    point_index: wp.array2d[int],
    point_weight: wp.array2d[float],
    particle_flags: wp.array[wp.int32],
    particle_volume: wp.array[float],
    elastic_strain: wp.array[wp.mat33],
    particle_stress: wp.array[wp.mat33],
    particle_Jp: wp.array[float],
    material_parameters: MaterialParameters,
    dt: float,
    point_volume: wp.array[float],
    point_stiffness: wp.array[float],
    point_elastic_strain: wp.array[wp.mat33],
    point_elastic_parameters: wp.array[wp.vec3],
    point_yield_parameters: wp.array[YieldParamVec],
    point_stress: wp.array[wp.mat33],
):
    """Accumulate extensive particle history at the lattice points.

    The Hencky strain is accumulated weighted by volume times Young's modulus,
    which preserves the transferred Kirchhoff stress ``V * tau`` for a uniform
    Poisson ratio.
    """
    p = wp.tid()
    if ~particle_flags[p] & newton.ParticleFlags.ACTIVE:
        return

    volume = particle_volume[p]
    if volume <= 0.0:
        return

    elastic_parameters = get_elastic_parameters(p, material_parameters)
    yield_parameters = get_yield_parameters(p, material_parameters, particle_Jp[p], dt)
    stiffness = volume * elastic_parameters[0]
    hencky = hencky_strain(elastic_strain[p])

    for k in range(POINT_STENCIL_SIZE):
        q = point_index[p, k]
        if q < 0:
            continue
        w = point_weight[p, k]
        wv = w * volume
        ws = w * stiffness
        wp.atomic_add(point_volume, q, wv)
        wp.atomic_add(point_stiffness, q, ws)
        wp.atomic_add(point_elastic_strain, q, ws * hencky)
        wp.atomic_add(point_elastic_parameters, q, wv * elastic_parameters)
        wp.atomic_add(point_yield_parameters, q, wv * yield_parameters)
        wp.atomic_add(point_stress, q, wv * particle_stress[p])


@wp.kernel
def normalize_point_data(
    inv_cell_volume: float,
    point_volume: wp.array[float],
    point_stiffness: wp.array[float],
    point_flags: wp.array[wp.int32],
    point_fraction: wp.array[float],
    point_elastic_strain: wp.array[wp.mat33],
    point_elastic_parameters: wp.array[wp.vec3],
    point_yield_parameters: wp.array[YieldParamVec],
    point_stress: wp.array[wp.mat33],
):
    """Convert accumulated extensive point data to volume averages and quadrature weights."""
    q = wp.tid()
    volume = point_volume[q]
    point_fraction[q] = volume * inv_cell_volume
    if volume <= 0.0:
        point_flags[q] = 0
        return

    point_flags[q] = newton.ParticleFlags.ACTIVE
    inv_volume = 1.0 / volume
    stiffness = point_stiffness[q]
    if stiffness > 0.0:
        point_elastic_strain[q] = point_elastic_strain[q] / stiffness
    point_elastic_parameters[q] = point_elastic_parameters[q] * inv_volume
    point_yield_parameters[q] = wp.max(YieldParamVec(0.0), point_yield_parameters[q] * inv_volume)
    point_stress[q] = point_stress[q] * inv_volume


@fem.integrand
def cell_strain_rhs(
    s: fem.Sample,
    tau: fem.Field,
    point_elastic_strain: wp.array[wp.mat33],
    point_elastic_parameters: wp.array[wp.vec3],
    point_flags: wp.array[wp.int32],
    inv_cell_volume: float,
    dt: float,
):
    """Elastic right-hand side from the transferred Hencky strain at fixed points."""
    q = s.qp_index
    if ~point_flags[q] & newton.ParticleFlags.ACTIVE:
        return 0.0

    _compliance, _poisson, damping = extract_elastic_parameters(point_elastic_parameters[q])
    alpha = 1.0 / (1.0 + damping / dt)
    return alpha * wp.ddot(tau(s), point_elastic_strain[q]) * inv_cell_volume


@fem.integrand
def cell_compliance_form(
    s: fem.Sample,
    tau: fem.Field,
    sig: fem.Field,
    point_elastic_parameters: wp.array[wp.vec3],
    point_flags: wp.array[wp.int32],
    inv_cell_volume: float,
    dt: float,
):
    """Isotropic Hencky compliance at fixed points.

    The elastic rotation cancels out of the isotropic Hencky compliance, so the
    transferred strain magnitude is the only history this form depends on.
    """
    q = s.qp_index
    if ~point_flags[q] & newton.ParticleFlags.ACTIVE:
        return 0.0

    compliance, poisson, damping = extract_elastic_parameters(point_elastic_parameters[q])
    gamma = compliance / (1.0 + damping / dt)
    return wp.ddot(tau(s), stress_strain_relationship(sig(s), gamma, poisson)) * inv_cell_volume


@fem.integrand
def integrate_point_yield_parameters(
    s: fem.Sample,
    u: fem.Field,
    point_yield_parameters: wp.array[YieldParamVec],
    point_flags: wp.array[wp.int32],
    inv_cell_volume: float,
):
    q = s.qp_index
    if ~point_flags[q] & newton.ParticleFlags.ACTIVE:
        return 0.0
    return wp.dot(u(s), point_yield_parameters[q]) * inv_cell_volume


@fem.integrand
def sample_point_strain_fields(
    s: fem.Sample,
    grid_vel: fem.Field,
    elastic_strain_delta: fem.Field,
    plastic_strain_delta: fem.Field,
    stress: fem.Field,
    point_elastic_strain_delta: wp.array[wp.mat33],
    point_plastic_strain_delta: wp.array[wp.mat33],
    point_stress: wp.array[wp.mat33],
    point_velocity_gradient: wp.array[wp.mat33],
):
    """Store strain-space solver results and the velocity gradient at each fixed point."""
    q = s.qp_index
    point_elastic_strain_delta[q] = elastic_strain_delta(s)
    point_plastic_strain_delta[q] = plastic_strain_delta(s)
    point_stress[q] = stress(s)
    point_velocity_gradient[q] = fem.grad(grid_vel, s)


@wp.kernel
def update_particle_strains_from_points(
    dt: float,
    kinematic_update: int,
    point_index: wp.array2d[int],
    point_weight: wp.array2d[float],
    particle_flags: wp.array[wp.int32],
    particle_density: wp.array[float],
    material_parameters: MaterialParameters,
    point_elastic_strain_delta: wp.array[wp.mat33],
    point_plastic_strain_delta: wp.array[wp.mat33],
    point_stress: wp.array[wp.mat33],
    point_velocity_gradient: wp.array[wp.mat33],
    elastic_strain_prev: wp.array[wp.mat33],
    particle_Jp_prev: wp.array[float],
    elastic_strain: wp.array[wp.mat33],
    particle_Jp: wp.array[float],
    particle_stress: wp.array[wp.mat33],
):
    """Update particle history from point results interpolated with the transfer weights.

    With ``kinematic_update == 0``, the elastic strain follows the solver's
    stress-consistent elastic strain increment plus the interpolated spin.
    Otherwise it follows ``F <- (I + dt G - D_p) F`` with the interpolated
    velocity gradient ``G`` and plastic strain increment ``D_p``, which is the
    MPM Lite update for elastic materials.
    """
    p = wp.tid()
    F_prev = elastic_strain_prev[p]
    Jp_prev = particle_Jp_prev[p]

    is_active = (particle_flags[p] & newton.ParticleFlags.ACTIVE) != 0
    if not is_active or particle_density[p] == 0.0:
        elastic_strain[p] = F_prev
        particle_Jp[p] = Jp_prev
        particle_stress[p] = wp.mat33(0.0)
        return

    elastic_delta = wp.mat33(0.0)
    plastic_delta = wp.mat33(0.0)
    stress = wp.mat33(0.0)
    vel_grad = wp.mat33(0.0)
    for k in range(POINT_STENCIL_SIZE):
        q = point_index[p, k]
        if q < 0:
            continue
        w = point_weight[p, k]
        elastic_delta += w * point_elastic_strain_delta[q]
        plastic_delta += w * point_plastic_strain_delta[q]
        stress += w * point_stress[q]
        vel_grad += w * point_velocity_gradient[q]

    F_new, Jp_new, stress_new = update_particle_history(
        p, dt, kinematic_update, material_parameters, F_prev, Jp_prev, elastic_delta, plastic_delta, stress, vel_grad
    )
    elastic_strain[p] = F_new
    particle_Jp[p] = Jp_new
    particle_stress[p] = stress_new


@wp.kernel
def transfer_particle_momentum_to_points(
    apic: int,
    dt: float,
    point_index: wp.array2d[int],
    point_weight: wp.array2d[float],
    point_offset: wp.array2d[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_volume: wp.array[float],
    particle_density: wp.array[float],
    velocities: wp.array[wp.vec3],
    velocity_gradients: wp.array[wp.mat33],
    particle_world: wp.array[int],
    gravity: wp.array[wp.vec3],
    point_mass: wp.array[float],
    point_transfer_volume: wp.array[float],
    point_velocity: wp.array[wp.vec3],
    point_velocity_gradient: wp.array[wp.mat33],
):
    """Accumulate particle mass and affine momentum at the lattice points."""
    p = wp.tid()
    if ~particle_flags[p] & newton.ParticleFlags.ACTIVE:
        return

    volume = particle_volume[p]
    mass = particle_density[p] * volume
    # Average each particle's world gravity together with its momentum.
    velocity = velocities[p] + dt * gravity[particle_world[p]]
    velocity_gradient = wp.mat33(0.0)
    if apic != 0:
        velocity_gradient = velocity_gradients[p]

    for k in range(POINT_STENCIL_SIZE):
        q = point_index[p, k]
        if q < 0:
            continue
        w = point_weight[p, k]
        wm = w * mass
        wp.atomic_add(point_mass, q, wm)
        wp.atomic_add(point_transfer_volume, q, w * volume)
        wp.atomic_add(point_velocity, q, wm * (velocity + velocity_gradient @ point_offset[p, k]))
        wp.atomic_add(point_velocity_gradient, q, wm * velocity_gradient)


@wp.kernel
def normalize_point_momentum(
    inv_cell_volume: float,
    point_mass: wp.array[float],
    point_transfer_volume: wp.array[float],
    point_transfer_flags: wp.array[wp.int32],
    point_transfer_fraction: wp.array[float],
    point_density: wp.array[float],
    point_velocity: wp.array[wp.vec3],
    point_velocity_gradient: wp.array[wp.mat33],
):
    """Convert accumulated point momentum to mass-averaged velocities and densities."""
    q = wp.tid()
    mass = point_mass[q]
    volume = point_transfer_volume[q]
    point_transfer_fraction[q] = volume * inv_cell_volume
    if mass <= 0.0 or volume <= 0.0:
        point_transfer_flags[q] = 0
        point_density[q] = 0.0
        return

    point_transfer_flags[q] = newton.ParticleFlags.ACTIVE
    point_density[q] = mass / volume
    point_velocity[q] = point_velocity[q] / mass
    point_velocity_gradient[q] = point_velocity_gradient[q] / mass


@fem.integrand
def sample_point_velocity(
    s: fem.Sample,
    grid_vel: fem.Field,
    point_velocity: wp.array[wp.vec3],
    point_velocity_gradient: wp.array[wp.mat33],
):
    q = s.qp_index
    point_velocity[q] = grid_vel(s)
    point_velocity_gradient[q] = fem.grad(grid_vel, s)


@wp.kernel
def advect_particles_from_points(
    dt: float,
    max_vel: float,
    point_index: wp.array2d[int],
    point_weight: wp.array2d[float],
    particle_flags: wp.array[wp.int32],
    point_velocity: wp.array[wp.vec3],
    point_velocity_gradient: wp.array[wp.mat33],
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    vel_grad: wp.array[wp.mat33],
):
    """Advect particles with velocities and gradients interpolated from the lattice points."""
    p = wp.tid()
    if ~particle_flags[p] & newton.ParticleFlags.ACTIVE:
        return

    velocity = wp.vec3(0.0)
    velocity_gradient = wp.mat33(0.0)
    for k in range(POINT_STENCIL_SIZE):
        q = point_index[p, k]
        if q < 0:
            continue
        w = point_weight[p, k]
        velocity += w * point_velocity[q]
        velocity_gradient += w * point_velocity_gradient[q]

    speed_sq = wp.length_sq(velocity)
    if speed_sq > max_vel * max_vel:
        velocity = velocity * (max_vel / wp.sqrt(speed_sq))

    pos[p] = pos[p] + dt * velocity
    vel[p] = velocity
    vel_grad[p] = velocity_gradient


@wp.kernel
def cell_stencil_points(
    positions: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    offset: float,
    points: wp.array[wp.vec3],
    point_flags: wp.array[wp.int32],
):
    """Emit points whose cells cover every cell of each particle's lattice stencil."""
    p, corner = wp.tid()
    sign = wp.vec3(
        wp.where((corner & 4) != 0, 1.0, -1.0),
        wp.where((corner & 2) != 0, 1.0, -1.0),
        wp.where((corner & 1) != 0, 1.0, -1.0),
    )
    index = p * POINT_STENCIL_SIZE + corner
    points[index] = positions[p] + offset * sign
    point_flags[index] = particle_flags[p]


@fem.integrand
def point_trial_value(s: fem.Sample, trial: fem.Field):
    return trial(s)


_COLLIDER_TRIPLETS_PER_POINT = wp.constant(8)
"""Velocity nodes per point for the Q1 velocity basis."""


@wp.kernel
def two_hop_collider_triplets(
    collider_space_node: wp.array[int],
    evaluation_particle: wp.array[int],
    point_index: wp.array2d[int],
    point_weight: wp.array2d[float],
    point_node_offsets: wp.array[int],
    point_node_columns: wp.array[int],
    point_node_values: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    rows: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[float],
):
    """Write collider-node rows that interpolate node velocities through the lattice points."""
    n = wp.tid()
    base = n * POINT_STENCIL_SIZE * _COLLIDER_TRIPLETS_PER_POINT
    for t in range(POINT_STENCIL_SIZE * _COLLIDER_TRIPLETS_PER_POINT):
        rows[base + t] = -1
        columns[base + t] = -1
        values[base + t] = 0.0

    space_node = collider_space_node[n]
    if space_node < 0:
        return
    # A zero normal disables contact at this collider node
    if wp.length_sq(collider_normals[n]) == 0.0:
        return

    p = evaluation_particle[space_node]
    for k in range(POINT_STENCIL_SIZE):
        q = point_index[p, k]
        if q < 0:
            continue
        w = point_weight[p, k]
        node_begin = point_node_offsets[q]
        node_end = wp.min(point_node_offsets[q + 1], node_begin + _COLLIDER_TRIPLETS_PER_POINT)
        for b in range(node_begin, node_end):
            t = base + k * _COLLIDER_TRIPLETS_PER_POINT + (b - node_begin)
            rows[t] = wp.int32(n)
            columns[t] = wp.int32(point_node_columns[b])
            values[t] = w * point_node_values[b]


@wp.kernel
def fill_point_coords(abscissae: wp.array[float], point_coords: wp.array2d[fem.Coords]):
    """Fill the tensor-product point coordinates shared by every cell."""
    cell, local = wp.tid()
    n = abscissae.shape[0]
    point_coords[cell, local] = fem.Coords(
        abscissae[local // (n * n)], abscissae[(local // n) % n], abscissae[local % n]
    )


class CellQuadrature:
    """Fixed per-cell quadrature points and the particle transfer weights onto them.

    The points and their transferred-volume weights define a
    :class:`warp.fem.ExplicitQuadrature`; this class owns the particle-to-point
    transfer weights and the per-point history arrays indexed like that
    quadrature's ``s.qp_index``.

    Args:
        points_per_axis: Number of points per cell along each axis; ``1`` uses
            cell centers and ``2`` uses the 2x2x2 Gauss-Legendre points.
        device: Device of the persistent point coordinates. Construct this
            object outside graph capture, since it copies them from the host.
    """

    def __init__(self, points_per_axis: int, device: wp.DeviceLike = None):
        self.points_per_axis = points_per_axis
        self.points_per_cell = points_per_axis**3
        self.abscissae = cell_quadrature_abscissae(points_per_axis)
        self._abscissae_array = wp.array(self.abscissae, dtype=float, device=device)
        self._temporaries = []

        self.quadrature: fem.ExplicitQuadrature | None = None
        self.transfer_quadrature: fem.ExplicitQuadrature | None = None
        self.pic: fem.PicQuadrature | None = None
        self.point_index = None
        self.point_weight = None
        self.point_offset = None
        self.point_coords = None
        self.point_flags = None
        self.point_volume = None
        self.point_stiffness = None
        self.point_elastic_strain = None
        self.point_elastic_parameters = None
        self.point_yield_parameters = None
        self.point_stress = None

    def _borrow(self, temporary_store: fem.TemporaryStore | None, shape, dtype, device):
        temporary = fem.borrow_temporary(temporary_store, shape=shape, dtype=dtype, device=device)
        self._temporaries.append(temporary)
        return temporary

    def release(self):
        """Release the per-step temporaries."""
        for temporary in self._temporaries:
            temporary.release()
        self._temporaries = []
        self.quadrature = None
        self.transfer_quadrature = None
        self.pic = None

    @property
    def point_count(self) -> int:
        return self.point_coords.size

    def compute_weights(
        self,
        pic: fem.PicQuadrature,
        positions: wp.array[wp.vec3],
        voxel_size: float,
        temporary_store: fem.TemporaryStore | None,
    ):
        """Compute each particle's trilinear lattice weights onto the fixed points.

        Each particle's own cell and cell coordinates come from the particle
        quadrature ``pic``, which must use domain element indices. Lattice
        points falling in cells outside the domain partition are skipped and
        the remaining weights are renormalized.
        """
        domain = pic.domain
        device = positions.device
        self.pic = pic
        if self._abscissae_array.device != device:
            raise ValueError(
                f"CellQuadrature was created on {self._abscissae_array.device}, got positions on {device}."
            )

        # Match the cell indexing that ExplicitQuadrature selects for its point indices
        cell_count = domain.element_count()
        use_geometry_index = cell_count == domain.geometry_element_count()

        particle_count = positions.shape[0]
        self.point_index = self._borrow(temporary_store, (particle_count, POINT_STENCIL_SIZE), int, device)
        self.point_weight = self._borrow(temporary_store, (particle_count, POINT_STENCIL_SIZE), float, device)
        self.point_offset = self._borrow(temporary_store, (particle_count, POINT_STENCIL_SIZE), wp.vec3, device)
        self.point_coords = self._borrow(temporary_store, (cell_count, self.points_per_cell), fem.Coords, device)

        wp.launch(
            _make_point_weight_kernel(domain),
            dim=particle_count,
            inputs=[
                domain.element_arg_value(device=device),
                domain.element_index_arg_value(device=device),
                positions,
                pic.cell_indices,
                pic.particle_coords,
                voxel_size,
                self._abscissae_array,
                self.points_per_cell,
                use_geometry_index,
                self.point_index,
                self.point_weight,
                self.point_offset,
            ],
            device=device,
        )
        wp.launch(
            fill_point_coords,
            dim=self.point_coords.shape,
            inputs=[self._abscissae_array, self.point_coords],
            device=device,
        )

    def transfer_particles(
        self,
        domain: fem.GeometryDomain,
        particle_flags: wp.array[wp.int32],
        particle_volume: wp.array[float],
        elastic_strain: wp.array[wp.mat33],
        particle_stress: wp.array[wp.mat33],
        particle_Jp: wp.array[float],
        material_parameters: MaterialParameters,
        dt: float,
        inv_cell_volume: float,
        temporary_store: fem.TemporaryStore | None,
    ):
        """Transfer particle history to the points and build the point quadrature."""
        device = particle_volume.device
        point_count = self.point_count

        self.point_flags = self._borrow(temporary_store, point_count, wp.int32, device)
        self.point_volume = self._borrow(temporary_store, point_count, float, device)
        self.point_stiffness = self._borrow(temporary_store, point_count, float, device)
        self.point_elastic_strain = self._borrow(temporary_store, point_count, wp.mat33, device)
        self.point_elastic_parameters = self._borrow(temporary_store, point_count, wp.vec3, device)
        self.point_yield_parameters = self._borrow(temporary_store, point_count, YieldParamVec, device)
        self.point_stress = self._borrow(temporary_store, point_count, wp.mat33, device)
        point_fraction = self._borrow(temporary_store, self.point_coords.shape, float, device)

        for array in (
            self.point_volume,
            self.point_stiffness,
            self.point_elastic_strain,
            self.point_elastic_parameters,
            self.point_yield_parameters,
            self.point_stress,
        ):
            array.zero_()

        wp.launch(
            transfer_particles_to_points,
            dim=particle_volume.shape[0],
            inputs=[
                self.point_index,
                self.point_weight,
                particle_flags,
                particle_volume,
                elastic_strain,
                particle_stress,
                particle_Jp,
                material_parameters,
                dt,
                self.point_volume,
                self.point_stiffness,
                self.point_elastic_strain,
                self.point_elastic_parameters,
                self.point_yield_parameters,
                self.point_stress,
            ],
            device=device,
        )
        wp.launch(
            normalize_point_data,
            dim=point_count,
            inputs=[
                inv_cell_volume,
                self.point_volume,
                self.point_stiffness,
                self.point_flags,
                point_fraction.flatten(),
                self.point_elastic_strain,
                self.point_elastic_parameters,
                self.point_yield_parameters,
                self.point_stress,
            ],
            device=device,
        )

        self.quadrature = fem.ExplicitQuadrature(domain, points=self.point_coords, weights=point_fraction)

    def transfer_momentum(
        self,
        domain: fem.GeometryDomain,
        apic: bool,
        dt: float,
        particle_flags: wp.array[wp.int32],
        particle_volume: wp.array[float],
        particle_density: wp.array[float],
        velocities: wp.array[wp.vec3],
        velocity_gradients: wp.array[wp.mat33],
        particle_world: wp.array[int],
        gravity: wp.array[wp.vec3],
        inv_cell_volume: float,
        temporary_store: fem.TemporaryStore | None,
    ):
        """Transfer particle mass and affine momentum to the points and build their quadrature.

        Grid nodes then receive mass and momentum from the points, as in the
        two-hop transfers of MPM Lite.
        """
        device = particle_volume.device
        point_count = self.point_count

        point_mass = self._borrow(temporary_store, point_count, float, device)
        point_transfer_volume = self._borrow(temporary_store, point_count, float, device)
        point_transfer_fraction = self._borrow(temporary_store, self.point_coords.shape, float, device)
        self.point_transfer_flags = self._borrow(temporary_store, point_count, wp.int32, device)
        self.point_density = self._borrow(temporary_store, point_count, float, device)
        self.point_velocity = self._borrow(temporary_store, point_count, wp.vec3, device)
        self.point_velocity_gradient = self._borrow(temporary_store, point_count, wp.mat33, device)
        self.point_world = self._borrow(temporary_store, point_count, wp.int32, device)

        for array in (point_mass, point_transfer_volume, self.point_velocity, self.point_velocity_gradient):
            array.zero_()
        self.point_world.zero_()

        wp.launch(
            transfer_particle_momentum_to_points,
            dim=particle_volume.shape[0],
            inputs=[
                int(apic),
                dt,
                self.point_index,
                self.point_weight,
                self.point_offset,
                particle_flags,
                particle_volume,
                particle_density,
                velocities,
                velocity_gradients,
                particle_world,
                gravity,
                point_mass,
                point_transfer_volume,
                self.point_velocity,
                self.point_velocity_gradient,
            ],
            device=device,
        )
        wp.launch(
            normalize_point_momentum,
            dim=point_count,
            inputs=[
                inv_cell_volume,
                point_mass,
                point_transfer_volume,
                self.point_transfer_flags,
                point_transfer_fraction.flatten(),
                self.point_density,
                self.point_velocity,
                self.point_velocity_gradient,
            ],
            device=device,
        )

        self.transfer_quadrature = fem.ExplicitQuadrature(
            domain, points=self.point_coords, weights=point_transfer_fraction
        )

    def build_collider_matrix(
        self,
        collider_matrix: wps.BsrMatrix,
        collider_partition: fem.SpacePartition,
        velocity_trial: fem.field.TrialField,
        velocity_node_count: int,
        collider_normals: wp.array[wp.vec3],
        temporary_store: fem.TemporaryStore | None,
    ):
        """Map node velocities to particle-based collider nodes through the lattice points.

        Collider rows then constrain the same two-hop velocity that advects
        the particles. Requires a particle-based collider basis built on the
        evaluation points of the particle quadrature passed to :meth:`compute_weights`.
        """
        device = collider_normals.device
        collider_node_count = collider_partition.node_count()

        point_nodes = wps.bsr_zeros(self.point_count, velocity_node_count, block_type=float, device=device)
        fem.interpolate(
            point_trial_value,
            at=self.transfer_quadrature,
            dest=point_nodes,
            fields={"trial": velocity_trial},
            temporary_store=temporary_store,
        )

        triplet_count = collider_node_count * POINT_STENCIL_SIZE * _COLLIDER_TRIPLETS_PER_POINT
        rows = self._borrow(temporary_store, triplet_count, wp.int32, device)
        columns = self._borrow(temporary_store, triplet_count, wp.int32, device)
        values = self._borrow(temporary_store, triplet_count, float, device)
        wp.launch(
            two_hop_collider_triplets,
            dim=collider_node_count,
            inputs=[
                collider_partition.space_node_indices(),
                self.pic.cell_particle_indices,
                self.point_index,
                self.point_weight,
                point_nodes.offsets,
                point_nodes.columns,
                point_nodes.values,
                collider_normals,
                rows,
                columns,
                values,
            ],
            device=device,
        )

        wps.bsr_set_zero(collider_matrix, rows_of_blocks=collider_node_count, cols_of_blocks=velocity_node_count)
        wps.bsr_set_from_triplets(collider_matrix, rows, columns, values, prune_numerical_zeros=True)

    def advect_particles(
        self,
        dt: float,
        max_vel: float,
        grid_vel: fem.DiscreteField,
        particle_flags: wp.array[wp.int32],
        pos: wp.array[wp.vec3],
        vel: wp.array[wp.vec3],
        vel_grad: wp.array[wp.mat33],
        temporary_store: fem.TemporaryStore | None,
    ):
        """Advect particles with the grid velocity sampled at the points (two-hop G2P)."""
        device = pos.device
        point_count = self.point_count
        point_velocity = self._borrow(temporary_store, point_count, wp.vec3, device)
        point_velocity_gradient = self._borrow(temporary_store, point_count, wp.mat33, device)

        fem.interpolate(
            sample_point_velocity,
            at=self.transfer_quadrature,
            fields={"grid_vel": grid_vel},
            values={"point_velocity": point_velocity, "point_velocity_gradient": point_velocity_gradient},
            temporary_store=temporary_store,
        )
        wp.launch(
            advect_particles_from_points,
            dim=pos.shape[0],
            inputs=[
                dt,
                max_vel,
                self.point_index,
                self.point_weight,
                particle_flags,
                point_velocity,
                point_velocity_gradient,
                pos,
                vel,
                vel_grad,
            ],
            device=device,
        )

    def elasticity_inputs(self, dt: float, inv_cell_volume: float) -> ElasticityInputs:
        """Bind transferred cell-point material data to the shared elastic assembly."""
        values = {
            "point_elastic_parameters": self.point_elastic_parameters,
            "point_flags": self.point_flags,
            "inv_cell_volume": inv_cell_volume,
            "dt": dt,
        }
        return ElasticityInputs(
            quadrature=self.quadrature,
            strain_rhs=cell_strain_rhs,
            compliance=cell_compliance_form,
            fields={},
            values={**values, "point_elastic_strain": self.point_elastic_strain},
            compliance_values=values,
        )

    def yield_parameters_inputs(self, inv_cell_volume: float) -> tuple[fem.Integrand, dict]:
        """Bind cell-point material parameters to the shared yield-parameter assembly."""
        return integrate_point_yield_parameters, {
            "point_yield_parameters": self.point_yield_parameters,
            "point_flags": self.point_flags,
            "inv_cell_volume": inv_cell_volume,
        }

    def update_particles(
        self,
        dt: float,
        kinematic_update: bool,
        grid_vel: fem.DiscreteField,
        elastic_strain_delta: fem.DiscreteField,
        plastic_strain_delta: fem.DiscreteField,
        stress: fem.DiscreteField,
        particle_flags: wp.array[wp.int32],
        particle_density: wp.array[float],
        material_parameters: MaterialParameters,
        elastic_strain_prev: wp.array[wp.mat33],
        particle_Jp_prev: wp.array[float],
        elastic_strain: wp.array[wp.mat33],
        particle_Jp: wp.array[float],
        particle_stress: wp.array[wp.mat33],
        temporary_store: fem.TemporaryStore | None,
    ):
        """Sample solver results at the points and interpolate them back to particles."""
        device = particle_density.device
        point_count = self.point_count

        point_elastic_strain_delta = self._borrow(temporary_store, point_count, wp.mat33, device)
        point_plastic_strain_delta = self._borrow(temporary_store, point_count, wp.mat33, device)
        point_stress = self._borrow(temporary_store, point_count, wp.mat33, device)
        point_velocity_gradient = self._borrow(temporary_store, point_count, wp.mat33, device)

        fem.interpolate(
            sample_point_strain_fields,
            at=self.quadrature,
            fields={
                "grid_vel": grid_vel,
                "elastic_strain_delta": elastic_strain_delta,
                "plastic_strain_delta": plastic_strain_delta,
                "stress": stress,
            },
            values={
                "point_elastic_strain_delta": point_elastic_strain_delta,
                "point_plastic_strain_delta": point_plastic_strain_delta,
                "point_stress": point_stress,
                "point_velocity_gradient": point_velocity_gradient,
            },
            temporary_store=temporary_store,
        )

        wp.launch(
            update_particle_strains_from_points,
            dim=particle_density.shape[0],
            inputs=[
                dt,
                int(kinematic_update),
                self.point_index,
                self.point_weight,
                particle_flags,
                particle_density,
                material_parameters,
                point_elastic_strain_delta,
                point_plastic_strain_delta,
                point_stress,
                point_velocity_gradient,
                elastic_strain_prev,
                particle_Jp_prev,
                elastic_strain,
                particle_Jp,
                particle_stress,
            ],
            device=device,
        )


def _make_point_weight_kernel(domain: fem.GeometryDomain):
    cell_lookup = domain.element_partition_lookup
    partition_index = domain.element_partition_index
    element_index = domain.element_index

    @fem.cache.dynamic_kernel(suffix=domain.name)
    def compute_point_weights(
        cell_arg_value: domain.ElementArg,
        domain_index_arg_value: domain.ElementIndexArg,
        positions: wp.array[wp.vec3],
        particle_cell_index: wp.array[fem.ElementIndex],
        particle_coords: wp.array[fem.Coords],
        voxel_size: float,
        abscissae: wp.array[float],
        points_per_cell: int,
        use_geometry_index: bool,
        point_index: wp.array2d[int],
        point_weight: wp.array2d[float],
        point_offset: wp.array2d[wp.vec3],
    ):
        p = wp.tid()
        domain_arg = domain.DomainArg(cell_arg_value, domain_index_arg_value)

        for k in range(POINT_STENCIL_SIZE):
            point_index[p, k] = -1
            point_weight[p, k] = 0.0
            point_offset[p, k] = wp.vec3(0.0)

        own_cell = particle_cell_index[p]
        if own_cell == fem.NULL_ELEMENT_INDEX:
            return
        if use_geometry_index:
            own_cell = element_index(domain_index_arg_value, own_cell)
        x = positions[p]
        u = particle_coords[p]

        n = abscissae.shape[0]
        first = abscissae[0]
        last = abscissae[n - 1]

        # Bracket each coordinate between two consecutive lattice abscissae,
        # which may belong to neighboring cells.
        lo_offset = wp.vec3i(0)
        lo_index = wp.vec3i(0)
        t = wp.vec3(0.0)
        for a in range(3):
            ua = u[a]
            if ua < first:
                lo_offset[a] = -1
                lo_index[a] = n - 1
                t[a] = (ua - (last - 1.0)) / (first + 1.0 - last)
            elif ua >= last:
                lo_offset[a] = 0
                lo_index[a] = n - 1
                t[a] = (ua - last) / (first + 1.0 - last)
            else:
                j = int(0)
                while j < n - 2 and ua >= abscissae[j + 1]:
                    j += 1
                lo_offset[a] = 0
                lo_index[a] = j
                t[a] = (ua - abscissae[j]) / (abscissae[j + 1] - abscissae[j])

        weight_sum = float(0.0)
        for corner in range(POINT_STENCIL_SIZE):
            w = float(1.0)
            offset = wp.vec3i(0)
            index = wp.vec3i(0)
            target = wp.vec3(0.0)
            for a in range(3):
                if ((corner >> (2 - a)) & 1) == 0:
                    w *= 1.0 - t[a]
                    offset[a] = lo_offset[a]
                    index[a] = lo_index[a]
                elif lo_index[a] == n - 1:
                    w *= t[a]
                    offset[a] = lo_offset[a] + 1
                    index[a] = 0
                else:
                    w *= t[a]
                    offset[a] = lo_offset[a]
                    index[a] = lo_index[a] + 1
                target[a] = abscissae[index[a]]

            if w <= 0.0:
                continue

            cell = own_cell
            point_delta = voxel_size * (wp.vec3(float(offset[0]), float(offset[1]), float(offset[2])) + target - u)
            if offset[0] != 0 or offset[1] != 0 or offset[2] != 0:
                point_pos = x + point_delta
                point_sample = cell_lookup(domain_arg, point_pos)
                if point_sample.element_index == fem.NULL_ELEMENT_INDEX:
                    continue
                # Lookups of positions in missing cells may return a neighbor's closest point
                coords_error = point_sample.element_coords - target
                if wp.max(wp.abs(coords_error)) > _POINT_COORDS_TOLERANCE:
                    continue

                cell = partition_index(domain_index_arg_value, point_sample.element_index)
                if cell == fem.NULL_ELEMENT_INDEX:
                    continue
                if use_geometry_index:
                    cell = point_sample.element_index

            local = (index[0] * n + index[1]) * n + index[2]
            point_index[p, corner] = cell * points_per_cell + local
            point_weight[p, corner] = w
            point_offset[p, corner] = point_delta
            weight_sum += w

        if weight_sum > 0.0:
            for k in range(POINT_STENCIL_SIZE):
                point_weight[p, k] = point_weight[p, k] / weight_sum

    return compute_point_weights
