# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed per-cell quadrature and transfers for the implicit MPM solver.

Strain-space integrals (strain operator, compliance, elastic right-hand side,
strain node volumes and yield parameters) are evaluated at points with fixed
positions relative to the grid cells, so that velocity shape-function gradients
are only sampled where they are smooth. The points and their weights form a
:class:`warp.fem.ExplicitQuadrature`.

Particles carry history only. Their history and the point quadrature weights
come from a generalized moving least-squares (GMLS) fit centered at each cell,
with a quadratic B-spline kernel of 1.5-voxel radius and a per-particle
partition of unity over the existing cells. The affine fit is evaluated at the
cell's points; it is continuous in particle position and reproduces linear
fields.

Solver results return to particles with trilinear hat weights defined on the
lattice formed by all points, which are also continuous in particle position.
For strain bases that are affine in each cell (P1d), each cell's strain results
instead form an affine frame about its center, which particles evaluate with
the GMLS kernel weights; the points then only integrate frame fields.

Mass, momentum, particle advection and particle-based contact rows may also go
through the cell centers with trilinear weights on their lattice ("two-hop"
transfers), so that velocity modes invisible at the centers stay invisible to
particles.

With one point per cell, the points are the cell centers and their lattice is
the dual grid, as in MPM Lite (https://arxiv.org/abs/2602.07853). With two
points per axis, the points are the 2x2x2 Gauss-Legendre points of each cell,
as in PQMPM (https://doi.org/10.1002/nme.6588).
"""

import warp as wp
import warp.fem as fem
import warp.sparse as wps

import newton

from .implicit_mpm_model import MaterialParameters
from .implicit_mpm_solver_kernels import ElasticityInputs
from .material_kernels import (
    INFINITY,
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
    """Accumulate particle mass and affine momentum at the lattice points.

    Particles with zero density are kinematic and contribute a quasi-infinite
    mass, so that points and then grid nodes near them follow their velocity.
    """
    p = wp.tid()
    if ~particle_flags[p] & newton.ParticleFlags.ACTIVE:
        return

    volume = particle_volume[p]
    density = particle_density[p]
    mass = INFINITY * volume
    velocity = velocities[p]
    if density > 0.0:
        mass = density * volume
        # Average each particle's world gravity together with its momentum.
        velocity += dt * gravity[particle_world[p]]
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
def interpolate_particle_velocity_gradient(
    point_index: wp.array2d[int],
    point_weight: wp.array2d[float],
    particle_flags: wp.array[wp.int32],
    point_velocity_gradient: wp.array[wp.mat33],
    velocity_gradient: wp.array[wp.mat33],
):
    """Interpolate the velocity gradient at the lattice points to particles, as in particle advection."""
    p = wp.tid()
    gradient = wp.mat33(0.0)
    if particle_flags[p] & newton.ParticleFlags.ACTIVE:
        for k in range(POINT_STENCIL_SIZE):
            q = point_index[p, k]
            if q >= 0:
                gradient += point_weight[p, k] * point_velocity_gradient[q]
    velocity_gradient[p] = gradient


@wp.kernel
def cell_stencil_points(
    positions: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_environment: wp.array[int],
    offset: float,
    points: wp.array[wp.vec3],
    point_flags: wp.array[wp.int32],
    point_environment: wp.array[int],
):
    """Emit points whose cells cover every cell of each particle's lattice stencil.

    With a non-empty ``particle_environment``, each point inherits its particle's environment.
    """
    p, corner = wp.tid()
    sign = wp.vec3(
        wp.where((corner & 4) != 0, 1.0, -1.0),
        wp.where((corner & 2) != 0, 1.0, -1.0),
        wp.where((corner & 1) != 0, 1.0, -1.0),
    )
    index = p * POINT_STENCIL_SIZE + corner
    points[index] = positions[p] + offset * sign
    point_flags[index] = particle_flags[p]
    if particle_environment:
        point_environment[index] = particle_environment[p]


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
        strain_frames: Return solver strain results to particles through
            per-cell affine frames, for strain bases that are affine in each
            cell (``"P1d"``). Requires two points per axis; the points then
            only integrate the frame fields.
    """

    def __init__(self, points_per_axis: int, device: wp.DeviceLike = None, *, strain_frames: bool = False):
        if strain_frames and points_per_axis < 2:
            raise ValueError("Strain frames require two points per axis.")
        self.strain_frames = strain_frames
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
        self._positions = None
        self.voxel_size = None
        self._cell_center = None
        self._cell_neighbors = None
        self._cell_point_row = None
        self._particle_normalization = None

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
        """Compute each particle's trilinear lattice weights onto the fixed points, and the GMLS cell neighborhoods.

        Each particle's own cell and cell coordinates come from the particle
        quadrature ``pic``, which must use domain element indices. Lattice
        points falling in cells outside the domain partition are skipped and
        the remaining weights are renormalized. Each cell's existing 3x3x3
        neighbors and each particle's GMLS kernel normalization over them are
        shared by the GMLS history fits and the strain frames.
        """
        domain = pic.domain
        device = positions.device
        self.pic = pic
        self._positions = positions
        self.voxel_size = voxel_size
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

        self._cell_center = self._borrow(temporary_store, cell_count, wp.vec3, device)
        self._cell_neighbors = self._borrow(temporary_store, (cell_count, _CELL_NEIGHBOR_COUNT), int, device)
        cell_neighbor_mask = self._borrow(temporary_store, cell_count, int, device)
        self._cell_point_row = self._borrow(temporary_store, cell_count, int, device)
        self._particle_normalization = self._borrow(temporary_store, particle_count, float, device)
        wp.launch(
            _make_cell_neighbor_kernel(domain),
            dim=cell_count,
            inputs=[
                domain.element_arg_value(device=device),
                domain.element_index_arg_value(device=device),
                use_geometry_index,
                voxel_size,
                self._cell_center,
                self._cell_neighbors,
                cell_neighbor_mask,
                self._cell_point_row,
            ],
            device=device,
        )
        wp.launch(
            gmls_particle_normalization,
            dim=particle_count,
            inputs=[pic.cell_indices, pic.particle_coords, cell_neighbor_mask, self._particle_normalization],
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
        *,
        transfer_elastic_history: bool = True,
    ):
        """Fit particle history at the points by GMLS around each cell center and build the point quadrature."""
        device = particle_volume.device
        particle_count = particle_volume.shape[0]
        point_count = self.point_count
        shape = self.point_coords.shape

        self.point_flags = self._borrow(temporary_store, point_count, wp.int32, device)
        self.point_volume = self._borrow(temporary_store, point_count, float, device)
        self.point_stiffness = None
        self.point_elastic_strain = None
        self.point_elastic_parameters = None
        particle_hencky = None
        particle_elastic_parameters = None
        if transfer_elastic_history:
            self.point_stiffness = self._borrow(temporary_store, point_count, float, device)
            self.point_elastic_strain = self._borrow(temporary_store, point_count, wp.mat33, device)
            self.point_elastic_parameters = self._borrow(temporary_store, point_count, wp.vec3, device)
            particle_hencky = self._borrow(temporary_store, particle_count, wp.mat33, device)
            particle_elastic_parameters = self._borrow(temporary_store, particle_count, wp.vec3, device)
        self.point_yield_parameters = self._borrow(temporary_store, point_count, YieldParamVec, device)
        self.point_stress = self._borrow(temporary_store, point_count, wp.mat33, device)
        point_fraction = self._borrow(temporary_store, shape, float, device)
        particle_yield_parameters = self._borrow(temporary_store, particle_count, YieldParamVec, device)
        # Rows of geometry cells outside the partition stay empty
        for array in (self.point_flags, self.point_volume, point_fraction):
            array.zero_()

        def per_cell(array):
            return None if array is None else array.reshape(shape)

        wp.launch(
            gmls_particle_data,
            dim=particle_count,
            inputs=[
                transfer_elastic_history,
                particle_flags,
                particle_volume,
                elastic_strain,
                particle_Jp,
                material_parameters,
                dt,
                particle_hencky,
                particle_elastic_parameters,
                particle_yield_parameters,
            ],
            device=device,
        )
        wp.launch(
            gmls_cell_values,
            dim=shape[0],
            inputs=[
                self._cell_center,
                self._cell_neighbors,
                self._cell_point_row,
                self.point_coords,
                self._positions,
                self.pic.cell_particle_offsets,
                self.pic.cell_particle_indices,
                particle_volume,
                self._particle_normalization,
                particle_hencky,
                particle_elastic_parameters,
                particle_yield_parameters,
                particle_stress,
                self.voxel_size,
                inv_cell_volume,
                transfer_elastic_history,
                # The stress warm start follows the polynomial order of the strain basis
                self.points_per_axis - 1,
                per_cell(self.point_volume),
                point_fraction,
                per_cell(self.point_flags),
                per_cell(self.point_stiffness),
                per_cell(self.point_elastic_strain),
                per_cell(self.point_elastic_parameters),
                per_cell(self.point_yield_parameters),
                per_cell(self.point_stress),
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

    def particle_velocity_gradient(
        self,
        grid_vel: fem.DiscreteField,
        particle_flags: wp.array[wp.int32],
        temporary_store: fem.TemporaryStore | None,
    ) -> wp.array[wp.mat33]:
        """Return the grid velocity gradient sampled at the transfer points and interpolated to particles.

        This is the velocity gradient that :meth:`advect_particles` gives to the particles.
        """
        device = particle_flags.device
        point_velocity = self._borrow(temporary_store, self.point_count, wp.vec3, device)
        point_velocity_gradient = self._borrow(temporary_store, self.point_count, wp.mat33, device)
        velocity_gradient = self._borrow(temporary_store, particle_flags.shape[0], wp.mat33, device)
        fem.interpolate(
            sample_point_velocity,
            at=self.transfer_quadrature,
            fields={"grid_vel": grid_vel},
            values={"point_velocity": point_velocity, "point_velocity_gradient": point_velocity_gradient},
            temporary_store=temporary_store,
        )
        wp.launch(
            interpolate_particle_velocity_gradient,
            dim=particle_flags.shape[0],
            inputs=[self.point_index, self.point_weight, particle_flags, point_velocity_gradient, velocity_gradient],
            device=device,
        )
        return velocity_gradient

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
        velocity_gradient: wp.array[wp.mat33] | None = None,
    ):
        """Sample solver results at the points and interpolate them back to particles.

        With strain frames, the strain results at each cell's points give its
        affine frame, and particles evaluate the frames around them with the
        GMLS kernel weights. The per-particle ``velocity_gradient``, required
        then, drives the kinematic update; pass the gradient that advects the
        particles, from :meth:`particle_velocity_gradient` of the transfer
        quadrature.
        """
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

        if self.strain_frames:
            if velocity_gradient is None:
                raise ValueError("Strain frames require the particle velocity gradient.")
            cell_count = self._cell_center.shape[0]
            shape = self.point_coords.shape
            frames = [self._borrow(temporary_store, (cell_count, 4), wp.mat33, device) for _ in range(3)]
            wp.launch(
                strain_frames_from_points,
                dim=cell_count,
                inputs=[
                    self._cell_point_row,
                    self.point_coords,
                    self.voxel_size,
                    point_elastic_strain_delta.reshape(shape),
                    point_plastic_strain_delta.reshape(shape),
                    point_stress.reshape(shape),
                    *frames,
                ],
                device=device,
            )
            wp.launch(
                update_particle_strains_from_frames,
                dim=particle_density.shape[0],
                inputs=[
                    dt,
                    int(kinematic_update),
                    self._cell_center,
                    self._cell_neighbors,
                    self.pic.cell_indices,
                    self._particle_normalization,
                    self._positions,
                    self.voxel_size,
                    particle_flags,
                    particle_density,
                    material_parameters,
                    *frames,
                    velocity_gradient,
                    elastic_strain_prev,
                    particle_Jp_prev,
                    elastic_strain,
                    particle_Jp,
                    particle_stress,
                ],
                device=device,
            )
            return

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


def _make_environment_cell_lookup(domain: fem.GeometryDomain):
    """Partition cell lookup within the environment (world) of a reference geometry cell."""
    cell_lookup = domain.element_partition_lookup
    environment_index = domain.element_environment_index
    multiple_environments = domain.geometry.environment_count() > 1

    @fem.cache.dynamic_func(suffix=domain.name)
    def environment_cell_lookup(domain_arg: domain.DomainArg, geometry_cell: int, pos: wp.vec3):
        if wp.static(multiple_environments):
            return cell_lookup(domain_arg, pos, environment_index(domain_arg.geo, geometry_cell))
        else:
            return cell_lookup(domain_arg, pos)

    return environment_cell_lookup


def _make_point_weight_kernel(domain: fem.GeometryDomain):
    cell_lookup = _make_environment_cell_lookup(domain)
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
        geometry_cell = element_index(domain_index_arg_value, own_cell)
        if use_geometry_index:
            own_cell = geometry_cell
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
                point_sample = cell_lookup(domain_arg, geometry_cell, point_pos)
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


_GMLS_KERNEL_VARIANCE = wp.constant(0.25)
"""Per-axis variance of the quadratic B-spline GMLS kernel, in squared voxels."""

_GMLS_FULL_SLOPE_VARIANCE = wp.constant(0.02)
"""Kernel-weighted particle offset variance, in squared voxels, above which GMLS uses the full slope.

Slopes fade out smoothly below this value and vanish at half of it.
"""

_GMLS_LOOKUP_TOLERANCE = wp.constant(1.0e-3)
"""Tolerance on recovered cell-center coordinates when validating a neighbor cell lookup."""

_GMLS_FULL_NEIGHBOR_MASK = wp.constant((1 << 27) - 1)
"""Neighbor mask of a cell whose 3x3x3 neighborhood is complete."""

_CELL_NEIGHBOR_COUNT = wp.constant(27)
"""Cells in the 3x3x3 neighborhood of a cell, including itself."""


@wp.func
def _quadratic_bspline(r: float):
    a = wp.abs(r)
    if a < 0.5:
        return 0.75 - a * a
    if a < 1.5:
        return 0.5 * (1.5 - a) * (1.5 - a)
    return 0.0


@wp.func
def _gmls_kernel_weight(d: wp.vec3):
    """Tensor-product quadratic B-spline at offset ``d``, in voxels."""
    return _quadratic_bspline(d[0]) * _quadratic_bspline(d[1]) * _quadratic_bspline(d[2])


@wp.func
def _gmls_affine_operator(weight: float, m1: wp.vec3, m2: wp.mat33, offset: wp.vec3):
    """Map from cross moments to the fitted value correction at ``offset`` from the kernel center.

    Returns ``c`` such that the order-1 value at ``offset`` (in voxels) is
    ``average + sum_a c[a] X_a``, with ``X_a`` the centered value-offset cross
    moments. Each eigen-direction of the offset covariance contributes with a
    C1 fade of its variance, so the fit blends continuously into the
    kernel-weighted average.
    """
    mean = m1 / weight
    covariance = m2 / weight - wp.outer(mean, mean)
    Q, d = wp.eig3(covariance)
    local = wp.transpose(Q) @ (offset - mean)
    inv_d = wp.vec3(0.0)
    for k in range(3):
        t = wp.clamp((2.0 * d[k] - _GMLS_FULL_SLOPE_VARIANCE) / _GMLS_FULL_SLOPE_VARIANCE, 0.0, 1.0)
        if t > 0.0:
            inv_d[k] = t * t * (3.0 - 2.0 * t) / d[k]
    return Q @ wp.cw_mul(inv_d, local)


@wp.kernel
def gmls_particle_data(
    transfer_elastic_history: bool,
    particle_flags: wp.array[wp.int32],
    particle_volume: wp.array[float],
    elastic_strain: wp.array[wp.mat33],
    particle_Jp: wp.array[float],
    material_parameters: MaterialParameters,
    dt: float,
    particle_hencky: wp.array[wp.mat33],
    particle_elastic_parameters: wp.array[wp.vec3],
    particle_yield_parameters: wp.array[YieldParamVec],
):
    """Per-particle values gathered by the GMLS fits, computed once."""
    p = wp.tid()
    if (~particle_flags[p] & newton.ParticleFlags.ACTIVE) or particle_volume[p] <= 0.0:
        return
    particle_yield_parameters[p] = get_yield_parameters(p, material_parameters, particle_Jp[p], dt)
    if transfer_elastic_history:
        particle_hencky[p] = hencky_strain(elastic_strain[p])
        particle_elastic_parameters[p] = get_elastic_parameters(p, material_parameters)


@wp.kernel
def gmls_particle_normalization(
    particle_cell_index: wp.array[fem.ElementIndex],
    particle_coords: wp.array[fem.Coords],
    cell_neighbor_mask: wp.array[int],
    particle_normalization: wp.array[float],
):
    """Sum of each particle's kernel weights over the existing cell centers, for a per-particle partition of unity.

    The cell-center kernels sum to one over the full lattice, so the sum is one
    minus the weights of the missing neighbor centers. Particles outside the
    domain get zero.
    """
    p = wp.tid()
    particle_normalization[p] = 0.0
    cell = particle_cell_index[p]
    if cell == fem.NULL_ELEMENT_INDEX:
        return

    u = particle_coords[p]
    missing = ~cell_neighbor_mask[cell] & _GMLS_FULL_NEIGHBOR_MASK
    total = float(1.0)
    while missing != 0:
        k = int(0)
        while ((missing >> k) & 1) == 0:
            k += 1
        missing = missing & ~(1 << k)
        c = wp.vec3(float(k // 9 - 1), float((k // 3) % 3 - 1), float(k % 3 - 1))
        total -= _gmls_kernel_weight(c + wp.vec3(0.5) - u)
    particle_normalization[p] = total


def _make_cell_neighbor_kernel(domain: fem.GeometryDomain):
    """Kernel computing cell centers and each cell's existing 3x3x3 neighbors in its world."""
    cell_lookup = _make_environment_cell_lookup(domain)
    partition_index = domain.element_partition_index
    element_index = domain.element_index
    element_position = domain.element_position

    @fem.cache.dynamic_func(suffix=domain.name)
    def cell_neighbor(
        domain_arg: domain.DomainArg,
        domain_index_arg_value: domain.ElementIndexArg,
        geometry_cell: int,
        center: wp.vec3,
    ):
        """Partition index of the cell centered at ``center`` in the world of ``geometry_cell``, or ``NULL_ELEMENT_INDEX``."""
        sample = cell_lookup(domain_arg, geometry_cell, center)
        if sample.element_index == fem.NULL_ELEMENT_INDEX:
            return fem.NULL_ELEMENT_INDEX
        if wp.max(wp.abs(sample.element_coords - wp.vec3(0.5))) > _GMLS_LOOKUP_TOLERANCE:
            return fem.NULL_ELEMENT_INDEX
        return partition_index(domain_index_arg_value, sample.element_index)

    @fem.cache.dynamic_kernel(suffix=domain.name)
    def gmls_cell_neighbors(
        cell_arg_value: domain.ElementArg,
        domain_index_arg_value: domain.ElementIndexArg,
        use_geometry_index: bool,
        voxel_size: float,
        cell_center: wp.array[wp.vec3],
        cell_neighbors: wp.array2d[int],
        cell_neighbor_mask: wp.array[int],
        cell_point_row: wp.array[int],
    ):
        """Store each partition cell's center, point row, and the partition indices of its 3x3x3 neighbors.

        Neighbor ``(ci + 1) * 9 + (cj + 1) * 3 + (ck + 1)`` is at offset
        ``(ci, cj, ck)``. Missing neighbors are ``NULL_ELEMENT_INDEX`` and leave
        their bit of ``cell_neighbor_mask`` unset. The point row indexes the
        point arrays, by geometry cell when ``use_geometry_index`` is set. Unused
        partition slots have no neighbors, including themselves.
        """
        cell = wp.tid()
        domain_arg = domain.DomainArg(cell_arg_value, domain_index_arg_value)
        geometry_cell = element_index(domain_index_arg_value, cell)
        if geometry_cell == fem.NULL_ELEMENT_INDEX:
            cell_center[cell] = wp.vec3(0.0)
            cell_neighbor_mask[cell] = 0
            cell_point_row[cell] = wp.where(use_geometry_index, fem.NULL_ELEMENT_INDEX, cell)
            for k in range(_CELL_NEIGHBOR_COUNT):
                cell_neighbors[cell, k] = fem.NULL_ELEMENT_INDEX
            return
        cell_point_row[cell] = wp.where(use_geometry_index, geometry_cell, cell)
        x_c = element_position(cell_arg_value, fem.make_free_sample(geometry_cell, fem.Coords(0.5, 0.5, 0.5)))
        cell_center[cell] = x_c
        mask = int(0)
        for ci in range(-1, 2):
            for cj in range(-1, 2):
                for ck in range(-1, 2):
                    k = (ci + 1) * 9 + (cj + 1) * 3 + (ck + 1)
                    neighbor = cell
                    if ci != 0 or cj != 0 or ck != 0:
                        c = wp.vec3(float(ci), float(cj), float(ck))
                        neighbor = cell_neighbor(
                            domain_arg, domain_index_arg_value, geometry_cell, x_c + voxel_size * c
                        )
                    cell_neighbors[cell, k] = neighbor
                    if neighbor != fem.NULL_ELEMENT_INDEX:
                        mask = mask | (1 << k)
        cell_neighbor_mask[cell] = mask

    return gmls_cell_neighbors


@wp.kernel
def gmls_cell_values(
    cell_center: wp.array[wp.vec3],
    cell_neighbors: wp.array2d[int],
    cell_point_row: wp.array[int],
    point_coords: wp.array2d[fem.Coords],
    positions: wp.array[wp.vec3],
    cell_particle_offsets: wp.array[int],
    cell_particle_indices: wp.array[int],
    particle_volume: wp.array[float],
    particle_normalization: wp.array[float],
    particle_hencky: wp.array[wp.mat33],
    particle_elastic_parameters: wp.array[wp.vec3],
    particle_yield_parameters: wp.array[YieldParamVec],
    particle_stress: wp.array[wp.mat33],
    voxel_size: float,
    inv_cell_volume: float,
    transfer_elastic_history: bool,
    stress_order: int,
    point_volume: wp.array2d[float],
    point_fraction: wp.array2d[float],
    point_flags: wp.array2d[wp.int32],
    point_stiffness: wp.array2d[float],
    point_elastic_strain: wp.array2d[wp.mat33],
    point_elastic_parameters: wp.array2d[wp.vec3],
    point_yield_parameters: wp.array2d[YieldParamVec],
    point_stress: wp.array2d[wp.mat33],
):
    """Gather kernel-weighted particle moments around a cell center, then evaluate the fits at its points.

    Point volumes split the cell's kernel volume along the affine expansion
    of the kernel volume density about the center, whose gradient is the
    first moment over the kernel variance; clamped to stay non-negative.
    The strain fit is the ratio of the fits of ``E h`` and ``E``, exact for
    a uniform Young's modulus ``E``. The stress fit is affine for
    ``stress_order`` 1 and the kernel-weighted average for 0.
    """
    cell = wp.tid()
    row = cell_point_row[cell]
    if row == fem.NULL_ELEMENT_INDEX:
        return
    x_c = cell_center[cell]
    inv_voxel_size = 1.0 / voxel_size
    points_per_cell = point_coords.shape[1]

    weight = float(0.0)
    m1 = wp.vec3(0.0)
    m2 = wp.mat33(0.0)
    stiffness = float(0.0)
    stiffness_m1 = wp.vec3(0.0)
    strain_m0 = wp.mat33(0.0)
    strain_m1_x = wp.mat33(0.0)
    strain_m1_y = wp.mat33(0.0)
    strain_m1_z = wp.mat33(0.0)
    stress_m0 = wp.mat33(0.0)
    stress_m1_x = wp.mat33(0.0)
    stress_m1_y = wp.mat33(0.0)
    stress_m1_z = wp.mat33(0.0)
    elastic_parameters = wp.vec3(0.0)
    yield_parameters = YieldParamVec(0.0)

    # The kernel support around the cell center lies within its 3x3x3 neighborhood
    for n in range(_CELL_NEIGHBOR_COUNT):
        neighbor = cell_neighbors[cell, n]
        if neighbor == fem.NULL_ELEMENT_INDEX:
            continue
        for k in range(cell_particle_offsets[neighbor], cell_particle_offsets[neighbor + 1]):
            p = cell_particle_indices[k]
            volume = particle_volume[p]
            normalization = particle_normalization[p]
            if volume <= 0.0 or normalization <= 0.0:
                continue
            d = (positions[p] - x_c) * inv_voxel_size
            w = _gmls_kernel_weight(d)
            if w <= 0.0:
                continue
            w *= volume / normalization
            stress = particle_stress[p]

            weight += w
            m1 += w * d
            m2 += w * wp.outer(d, d)
            yield_parameters += w * particle_yield_parameters[p]
            stress_m0 += w * stress
            if stress_order > 0:
                stress_m1_x += (w * d[0]) * stress
                stress_m1_y += (w * d[1]) * stress
                stress_m1_z += (w * d[2]) * stress
            if transfer_elastic_history:
                params = particle_elastic_parameters[p]
                ws = w * params[0]
                hencky = particle_hencky[p]
                stiffness += ws
                stiffness_m1 += ws * d
                strain_m0 += ws * hencky
                strain_m1_x += (ws * d[0]) * hencky
                strain_m1_y += (ws * d[1]) * hencky
                strain_m1_z += (ws * d[2]) * hencky
                elastic_parameters += w * params

    inv_weight = 0.0
    mean = wp.vec3(0.0)
    if weight > 0.0:
        inv_weight = 1.0 / weight
        mean = m1 * inv_weight

    for local in range(points_per_cell):
        offset = point_coords[row, local] - wp.vec3(0.5)
        share = weight / float(points_per_cell)
        for a in range(3):
            share *= wp.clamp(1.0 + mean[a] * offset[a] / _GMLS_KERNEL_VARIANCE, 0.0, 2.0)
        point_volume[row, local] = share
        point_fraction[row, local] = share * inv_cell_volume
        if share <= 0.0:
            point_flags[row, local] = 0
            point_yield_parameters[row, local] = YieldParamVec(0.0)
            point_stress[row, local] = wp.mat33(0.0)
            if transfer_elastic_history:
                point_stiffness[row, local] = 0.0
                point_elastic_strain[row, local] = wp.mat33(0.0)
                point_elastic_parameters[row, local] = wp.vec3(0.0)
            continue

        point_flags[row, local] = newton.ParticleFlags.ACTIVE
        point_yield_parameters[row, local] = wp.max(YieldParamVec(0.0), yield_parameters * inv_weight)
        c = _gmls_affine_operator(weight, m1, m2, offset)

        stress_c = wp.vec3(0.0)
        if stress_order > 0:
            stress_c = c
        average = stress_m0 * inv_weight
        point_stress[row, local] = (
            average
            + stress_c[0] * (stress_m1_x * inv_weight - mean[0] * average)
            + stress_c[1] * (stress_m1_y * inv_weight - mean[1] * average)
            + stress_c[2] * (stress_m1_z * inv_weight - mean[2] * average)
        )

        if transfer_elastic_history:
            point_stiffness[row, local] = stiffness * share * inv_weight
            point_elastic_parameters[row, local] = elastic_parameters * inv_weight
            average_stiffness = stiffness * inv_weight
            fitted_stiffness = average_stiffness + wp.dot(c, stiffness_m1 * inv_weight - mean * average_stiffness)
            fitted_stiffness = wp.max(fitted_stiffness, 0.5 * average_stiffness)
            average = strain_m0 * inv_weight
            point_elastic_strain[row, local] = (
                average
                + c[0] * (strain_m1_x * inv_weight - mean[0] * average)
                + c[1] * (strain_m1_y * inv_weight - mean[1] * average)
                + c[2] * (strain_m1_z * inv_weight - mean[2] * average)
            ) / wp.max(fitted_stiffness, 1.0e-30)


@wp.func
def _store_frame(
    frame: wp.array2d[wp.mat33],
    cell: int,
    value: wp.mat33,
    moment_x: wp.mat33,
    moment_y: wp.mat33,
    moment_z: wp.mat33,
    gradient_scale: wp.vec3,
):
    frame[cell, 0] = value
    frame[cell, 1] = moment_x * gradient_scale[0]
    frame[cell, 2] = moment_y * gradient_scale[1]
    frame[cell, 3] = moment_z * gradient_scale[2]


@wp.kernel
def strain_frames_from_points(
    cell_point_row: wp.array[int],
    point_coords: wp.array2d[fem.Coords],
    voxel_size: float,
    point_elastic_strain_delta: wp.array2d[wp.mat33],
    point_plastic_strain_delta: wp.array2d[wp.mat33],
    point_stress: wp.array2d[wp.mat33],
    elastic_frame: wp.array2d[wp.mat33],
    plastic_frame: wp.array2d[wp.mat33],
    stress_frame: wp.array2d[wp.mat33],
):
    """Fit each cell's affine frames of the solver strain fields from their values at its points.

    Frames store the value at the cell center, then the derivatives along each
    axis. The strain fields are affine in each cell, so the least-squares fit
    over the symmetric points is exact: the value is the point mean and each
    derivative the first moment over the second moment of the point offsets.
    """
    cell = wp.tid()
    elastic = wp.mat33(0.0)
    elastic_x = wp.mat33(0.0)
    elastic_y = wp.mat33(0.0)
    elastic_z = wp.mat33(0.0)
    plastic = wp.mat33(0.0)
    plastic_x = wp.mat33(0.0)
    plastic_y = wp.mat33(0.0)
    plastic_z = wp.mat33(0.0)
    stress = wp.mat33(0.0)
    stress_x = wp.mat33(0.0)
    stress_y = wp.mat33(0.0)
    stress_z = wp.mat33(0.0)
    gradient_scale = wp.vec3(0.0)

    row = cell_point_row[cell]
    points_per_cell = point_coords.shape[1]
    if row != fem.NULL_ELEMENT_INDEX:
        second_moment = wp.vec3(0.0)
        for local in range(points_per_cell):
            offset = point_coords[row, local] - wp.vec3(0.5)
            second_moment += wp.cw_mul(offset, offset)
            e = point_elastic_strain_delta[row, local]
            pl = point_plastic_strain_delta[row, local]
            st = point_stress[row, local]
            elastic += e
            elastic_x += offset[0] * e
            elastic_y += offset[1] * e
            elastic_z += offset[2] * e
            plastic += pl
            plastic_x += offset[0] * pl
            plastic_y += offset[1] * pl
            plastic_z += offset[2] * pl
            stress += st
            stress_x += offset[0] * st
            stress_y += offset[1] * st
            stress_z += offset[2] * st
        inv_count = 1.0 / float(points_per_cell)
        elastic *= inv_count
        plastic *= inv_count
        stress *= inv_count
        gradient_scale = wp.cw_div(wp.vec3(1.0 / voxel_size), second_moment)

    _store_frame(elastic_frame, cell, elastic, elastic_x, elastic_y, elastic_z, gradient_scale)
    _store_frame(plastic_frame, cell, plastic, plastic_x, plastic_y, plastic_z, gradient_scale)
    _store_frame(stress_frame, cell, stress, stress_x, stress_y, stress_z, gradient_scale)


@wp.func
def _evaluate_frame(frame: wp.array2d[wp.mat33], cell: int, offset: wp.vec3):
    """Affine frame value at ``offset`` from its cell center."""
    return frame[cell, 0] + offset[0] * frame[cell, 1] + offset[1] * frame[cell, 2] + offset[2] * frame[cell, 3]


@wp.kernel
def update_particle_strains_from_frames(
    dt: float,
    kinematic_update: int,
    cell_center: wp.array[wp.vec3],
    cell_neighbors: wp.array2d[int],
    particle_cell_index: wp.array[fem.ElementIndex],
    particle_normalization: wp.array[float],
    positions: wp.array[wp.vec3],
    voxel_size: float,
    particle_flags: wp.array[wp.int32],
    particle_density: wp.array[float],
    material_parameters: MaterialParameters,
    elastic_frame: wp.array2d[wp.mat33],
    plastic_frame: wp.array2d[wp.mat33],
    stress_frame: wp.array2d[wp.mat33],
    velocity_gradient: wp.array[wp.mat33],
    elastic_strain_prev: wp.array[wp.mat33],
    particle_Jp_prev: wp.array[float],
    elastic_strain: wp.array[wp.mat33],
    particle_Jp: wp.array[float],
    particle_stress: wp.array[wp.mat33],
):
    """Update particle history from the affine frames of the cells around each particle.

    Each frame is evaluated at the particle and weighted by the normalized GMLS
    kernel. The particle velocity gradient drives the kinematic update.
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
    cell = particle_cell_index[p]
    normalization = particle_normalization[p]
    if cell != fem.NULL_ELEMENT_INDEX and normalization > 0.0:
        x = positions[p]
        inv_voxel_size = 1.0 / voxel_size
        for n in range(_CELL_NEIGHBOR_COUNT):
            frame = cell_neighbors[cell, n]
            if frame == fem.NULL_ELEMENT_INDEX:
                continue
            offset = x - cell_center[frame]
            w = _gmls_kernel_weight(offset * inv_voxel_size) / normalization
            if w <= 0.0:
                continue
            elastic_delta += w * _evaluate_frame(elastic_frame, frame, offset)
            plastic_delta += w * _evaluate_frame(plastic_frame, frame, offset)
            stress += w * _evaluate_frame(stress_frame, frame, offset)

    F_new, Jp_new, stress_new = update_particle_history(
        p,
        dt,
        kinematic_update,
        material_parameters,
        F_prev,
        Jp_prev,
        elastic_delta,
        plastic_delta,
        stress,
        velocity_gradient[p],
    )
    elastic_strain[p] = F_new
    particle_Jp[p] = Jp_new
    particle_stress[p] = stress_new
