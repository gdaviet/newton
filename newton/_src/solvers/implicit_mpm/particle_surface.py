# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Particle surface extraction using anisotropic kernels and marching cubes.

Implements the method from Yu & Turk, "Reconstructing Surfaces of
Particle-Based Fluids Using Anisotropic Kernels", Eurographics/ACM SIGGRAPH
Symposium on Computer Animation, 2010.

The pipeline computes per-particle anisotropy matrices via Weighted PCA,
then evaluates a smooth scalar field on a regular grid using oriented
ellipsoidal kernels, and extracts the isosurface with
:class:`warp.MarchingCubes`.

Typical usage::

    surface_ctx = ParticleSurface(voxel_size=0.01)
    verts, indices, normals = surface_ctx.extract(
        state.particle_q, model.particle_radius,
    )
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp
import warp.fem as fem

from newton._src.utils.mesh import compute_vertex_normals

__all__ = ["ParticleSurface", "extract_particle_surface"]

wp.set_module_options({"enable_backward": False})


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _compute_aabb(
    positions: wp.array[wp.vec3],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
):
    i = wp.tid()
    p = positions[i]
    wp.atomic_min(lower, 0, p)
    wp.atomic_max(upper, 0, p)


@wp.kernel
def _blur_axis_x(src: wp.array3d[wp.float32], dst: wp.array3d[wp.float32], weights: wp.array[float], hw: int):
    i, j, k = wp.tid()
    val = float(0.0)
    for di in range(-hw, hw + 1):
        ii = wp.clamp(i + di, 0, src.shape[0] - 1)
        val += src[ii, j, k] * weights[wp.abs(di)]
    dst[i, j, k] = val


@wp.kernel
def _blur_axis_y(src: wp.array3d[wp.float32], dst: wp.array3d[wp.float32], weights: wp.array[float], hw: int):
    i, j, k = wp.tid()
    val = float(0.0)
    for dj in range(-hw, hw + 1):
        jj = wp.clamp(j + dj, 0, src.shape[1] - 1)
        val += src[i, jj, k] * weights[wp.abs(dj)]
    dst[i, j, k] = val


@wp.kernel
def _blur_axis_z(src: wp.array3d[wp.float32], dst: wp.array3d[wp.float32], weights: wp.array[float], hw: int):
    i, j, k = wp.tid()
    val = float(0.0)
    for dk in range(-hw, hw + 1):
        kk = wp.clamp(k + dk, 0, src.shape[2] - 1)
        val += src[i, j, kk] * weights[wp.abs(dk)]
    dst[i, j, k] = val


@wp.func
def _weight(dist: float, radius: float) -> float:
    """Cubic falloff weight: w = (1 - (d/r)^3) for d < r."""
    if dist >= radius:
        return 0.0
    q = dist / radius
    return 1.0 - q * q * q


@wp.func
def _cubic_bspline(q: float) -> float:
    """Cubic B-spline kernel with compact support at q=2."""
    if q < 1.0:
        return 1.0 - 1.5 * q * q + 0.75 * q * q * q
    elif q < 2.0:
        t = 2.0 - q
        return 0.25 * t * t * t
    return 0.0


# ---------------------------------------------------------------------------
# Pass 1: Smooth particle centers (Eq. 6 of Yu & Turk 2010)
# ---------------------------------------------------------------------------


@wp.kernel
def _smooth_positions(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    search_radius: float,
    smooth_lambda: float,
    smoothed: wp.array[wp.vec3],
):
    i = wp.tid()
    xi = positions[i]

    avg = wp.vec3(0.0)
    w_sum = float(0.0)

    query = wp.hash_grid_query(grid, xi, search_radius)
    idx = int(0)
    while wp.hash_grid_query_next(query, idx):
        dist = wp.length(xi - positions[idx])
        w = _weight(dist, search_radius)
        avg += w * positions[idx]
        w_sum += w

    if w_sum > 0.0:
        avg = avg / w_sum
        smoothed[i] = (1.0 - smooth_lambda) * xi + smooth_lambda * avg
    else:
        smoothed[i] = xi


# ---------------------------------------------------------------------------
# Pass 2: Per-particle anisotropy via Weighted PCA (Eqs. 9-16)
# ---------------------------------------------------------------------------


@wp.kernel
def _compute_anisotropy(
    grid: wp.uint64,
    smoothed: wp.array[wp.vec3],
    search_radius: float,
    k_r: float,
    k_s: float,
    k_n: float,
    n_epsilon: int,
    G_out: wp.array[wp.mat33],
    det_G_out: wp.array[float],
):
    i = wp.tid()
    xi = smoothed[i]
    h = search_radius

    x_w = wp.vec3(0.0)
    w_sum = float(0.0)
    count = int(0)

    query = wp.hash_grid_query(grid, xi, search_radius)
    idx = int(0)
    while wp.hash_grid_query_next(query, idx):
        dist = wp.length(xi - smoothed[idx])
        w = _weight(dist, search_radius)
        x_w += w * smoothed[idx]
        w_sum += w
        if w > 0.0:
            count += 1

    inv_h = 1.0 / h
    G = wp.identity(n=3, dtype=float) * inv_h
    det_g = inv_h * inv_h * inv_h

    if count > n_epsilon and w_sum > 0.0:
        x_w = x_w / w_sum

        C = wp.mat33(0.0)
        query2 = wp.hash_grid_query(grid, xi, search_radius)
        idx2 = int(0)
        while wp.hash_grid_query_next(query2, idx2):
            dist2 = wp.length(xi - smoothed[idx2])
            w2 = _weight(dist2, search_radius)
            if w2 > 0.0:
                d = smoothed[idx2] - x_w
                C += w2 * wp.outer(d, d)
        C = C / w_sum

        U = wp.mat33()
        sigma = wp.vec3()
        V = wp.mat33()
        wp.svd3(C, U, sigma, V)

        # Erratum fix: covariance eigenvalues are variances; sqrt for axis lengths
        s1 = wp.sqrt(wp.max(sigma[0], 1.0e-10))
        s2 = wp.sqrt(wp.max(sigma[1], 1.0e-10))
        s3 = wp.sqrt(wp.max(sigma[2], 1.0e-10))

        # Clamp minimum eigenvalue ratio (Eq. 14)
        s2 = wp.max(s2, s1 / k_r)
        s3 = wp.max(s3, s1 / k_r)

        # Auto-calibrate k_s so that a uniform neighborhood (s1≈s2≈s3)
        # produces det(G) matching the isotropic fallback det = (1/(k_n*h))^3.
        # For isotropic C: det(G_aniso) = (inv_h / (k_s*s))^3 = det(G_iso) = (1/(k_n*h))^3
        # => k_s_auto = k_n / s_geometric_mean.
        s_geo = wp.pow(s1 * s2 * s3, 1.0 / 3.0)
        k_s_eff = k_n / wp.max(s_geo, 1.0e-10)

        inv_s1 = 1.0 / (k_s_eff * s1)
        inv_s2 = 1.0 / (k_s_eff * s2)
        inv_s3 = 1.0 / (k_s_eff * s3)

        S_inv = wp.diag(wp.vec3(inv_s1, inv_s2, inv_s3))
        G_aniso = inv_h * U @ S_inv @ wp.transpose(U)
        det_aniso = inv_h * inv_h * inv_h * inv_s1 * inv_s2 * inv_s3

        # Blend: use anisotropic G only when there's significant directional
        # variation (s1/s3 > threshold).  Otherwise fall back to isotropic
        # to avoid noisy G from jitter in uniform regions.
        aniso_ratio = s1 / wp.max(s3, 1.0e-10)
        blend = wp.clamp((aniso_ratio - 1.2) / 0.8, 0.0, 1.0)  # ramp from 1.2 to 2.0

        iso_scale = 1.0 / (k_n * h)
        G_iso = wp.identity(n=3, dtype=float) * iso_scale
        det_iso = iso_scale * iso_scale * iso_scale

        G = (1.0 - blend) * G_iso + blend * G_aniso
        det_g = (1.0 - blend) * det_iso + blend * det_aniso
    elif count <= n_epsilon and count > 0:
        scale = 1.0 / (k_n * h)
        G = wp.identity(n=3, dtype=float) * scale
        det_g = scale * scale * scale

    G_out[i] = G
    det_G_out[i] = det_g


@wp.kernel
def _fill_isotropic_G(
    kernel_radius: float,
    k_n: float,
    G_out: wp.array[wp.mat33],
    det_G_out: wp.array[float],
):
    """Fill all particles with the same isotropic G = (1/(k_n*h)) * I."""
    i = wp.tid()
    scale = 1.0 / (k_n * kernel_radius)
    G_out[i] = wp.identity(n=3, dtype=float) * scale
    det_G_out[i] = scale * scale * scale


# ---------------------------------------------------------------------------
# Pass 3: Scalar field evaluation (Eq. 8)
# ---------------------------------------------------------------------------


@wp.kernel
def _eval_scalar_field(
    smoothed: wp.array[wp.vec3],
    radii: wp.array[float],
    G_matrices: wp.array[wp.mat33],
    det_G: wp.array[float],
    grid_origin: wp.vec3,
    inv_voxel_size: float,
    nx: int,
    ny: int,
    nz: int,
    field: wp.array3d[wp.float32],
):
    """Particle-centric scalar field evaluation.

    Each particle splatts its contribution onto nearby grid nodes using
    atomic adds, avoiding hash-grid queries from grid nodes entirely.
    """
    pid = wp.tid()
    x_p = smoothed[pid]
    r_p = radii[pid]
    volume = 8.0 * r_p * r_p * r_p
    # SPH normalization: integral of P(||u||) over 3D = pi, so sigma = 1/pi
    sigma = wp.static(1.0 / math.pi)
    G = G_matrices[pid]
    dG = det_G[pid]
    weight = volume * sigma * dG

    # Axis-aligned bounding box of the kernel support (||G*r|| < 2).
    G_inv = wp.inverse(G)
    reach_x = 2.0 * wp.length(wp.vec3(G_inv[0, 0], G_inv[1, 0], G_inv[2, 0]))
    reach_y = 2.0 * wp.length(wp.vec3(G_inv[0, 1], G_inv[1, 1], G_inv[2, 1]))
    reach_z = 2.0 * wp.length(wp.vec3(G_inv[0, 2], G_inv[1, 2], G_inv[2, 2]))

    lo_x = wp.max(int(wp.ceil((x_p[0] - reach_x - grid_origin[0]) * inv_voxel_size)), 0)
    lo_y = wp.max(int(wp.ceil((x_p[1] - reach_y - grid_origin[1]) * inv_voxel_size)), 0)
    lo_z = wp.max(int(wp.ceil((x_p[2] - reach_z - grid_origin[2]) * inv_voxel_size)), 0)
    hi_x = wp.min(int(wp.floor((x_p[0] + reach_x - grid_origin[0]) * inv_voxel_size)), nx - 1)
    hi_y = wp.min(int(wp.floor((x_p[1] + reach_y - grid_origin[1]) * inv_voxel_size)), ny - 1)
    hi_z = wp.min(int(wp.floor((x_p[2] + reach_z - grid_origin[2]) * inv_voxel_size)), nz - 1)

    voxel_size = 1.0 / inv_voxel_size
    for i in range(lo_x, hi_x + 1):
        for j in range(lo_y, hi_y + 1):
            for k in range(lo_z, hi_z + 1):
                x_node = grid_origin + voxel_size * wp.vec3(float(i), float(j), float(k))
                Gr = G * (x_node - x_p)
                q = wp.length(Gr)
                val = weight * _cubic_bspline(q)
                if val > 0.0:
                    wp.atomic_add(field, i, j, k, val)


@wp.kernel
def _flip_winding(indices: wp.array[wp.int32]):
    """Swap first and second vertex of each triangle to flip face normals."""
    tid = wp.tid()
    base = tid * 3
    tmp = indices[base]
    indices[base] = indices[base + 1]
    indices[base + 1] = tmp


# ---------------------------------------------------------------------------
# Mesh smoothing kernels (Laplacian)
# ---------------------------------------------------------------------------


@wp.kernel
def _laplacian_scatter(
    indices: wp.array[wp.int32],
    verts: wp.array[wp.vec3],
    neighbor_sum: wp.array[wp.vec3],
    valence: wp.array[wp.int32],
):
    tid = wp.tid()
    tri = tid // 3
    local = tid - tri * 3
    base = tri * 3

    i0 = indices[base + local]
    i1 = indices[base + (local + 1) % 3]
    i2 = indices[base + (local + 2) % 3]

    wp.atomic_add(neighbor_sum, i0, verts[i1] + verts[i2])
    wp.atomic_add(valence, i0, 2)


@wp.kernel
def _laplacian_apply(
    verts: wp.array[wp.vec3],
    neighbor_sum: wp.array[wp.vec3],
    valence: wp.array[wp.int32],
    smoothed: wp.array[wp.vec3],
    factor: float,
):
    i = wp.tid()
    v = valence[i]
    if v > 0:
        avg = neighbor_sum[i] / float(v)
        smoothed[i] = verts[i] + factor * (avg - verts[i])
    else:
        smoothed[i] = verts[i]


# ---------------------------------------------------------------------------
# Density → SDF conversion and redistancing
# ---------------------------------------------------------------------------


@wp.kernel
def _density_to_sdf_3d(
    field: wp.array3d[wp.float32],
    threshold: float,
):
    """Convert density field to SDF in-place: sdf = threshold - density."""
    i, j, k = wp.tid()
    field[i, j, k] = threshold - field[i, j, k]


@wp.kernel
def _redistance_step(
    sdf: wp.array3d[wp.float32],
    sdf_out: wp.array3d[wp.float32],
    inv_dx: float,
):
    """One step of the fast-sweeping Eikonal redistancing.

    Enforces |∇d| = 1 by shifting each node toward the signed distance
    implied by its upwind neighbors, using Godunov's scheme.
    """
    i, j, k = wp.tid()
    nx = sdf.shape[0]
    ny = sdf.shape[1]
    nz = sdf.shape[2]

    d = sdf[i, j, k]
    s = wp.sign(d)

    # Upwind finite differences (Godunov)
    dx_m = sdf[wp.max(i - 1, 0), j, k]
    dx_p = sdf[wp.min(i + 1, nx - 1), j, k]
    dy_m = sdf[i, wp.max(j - 1, 0), k]
    dy_p = sdf[i, wp.min(j + 1, ny - 1), k]
    dz_m = sdf[i, j, wp.max(k - 1, 0)]
    dz_p = sdf[i, j, wp.min(k + 1, nz - 1)]

    # Godunov upwind: pick the derivative that "looks" toward the interface
    ax = wp.max(wp.max(s * (d - dx_m), 0.0), wp.max(-s * (dx_p - d), 0.0)) * inv_dx
    ay = wp.max(wp.max(s * (d - dy_m), 0.0), wp.max(-s * (dy_p - d), 0.0)) * inv_dx
    az = wp.max(wp.max(s * (d - dz_m), 0.0), wp.max(-s * (dz_p - d), 0.0)) * inv_dx

    grad_mag = wp.sqrt(ax * ax + ay * ay + az * az)

    # PDE: d_t + sign(d0) * (|∇d| - 1) = 0
    # Explicit Euler with CFL-limited dt = 0.5 * dx
    dt = 0.5 / inv_dx
    sdf_out[i, j, k] = d - dt * s * (grad_mag - 1.0)


# ---------------------------------------------------------------------------
# ParticleSurface context
# ---------------------------------------------------------------------------


class ParticleSurface:
    """Reusable context for extracting a triangle mesh from particle data.

    Uses the Yu & Turk (2010) anisotropic kernel method: per-particle
    Weighted PCA determines oriented ellipsoidal kernels that produce a
    smooth scalar field whose isosurface tightly wraps the particles.

    Args:
        voxel_size: Edge length of each grid voxel [m].
        kernel_radius: Search radius for neighbor queries [m].
            Defaults to ``3 * voxel_size``.
        threshold: Isosurface level for marching cubes.  The scalar field
            is approximately 1.0 inside dense particle regions.
        smooth_lambda: Blending factor for position smoothing [0, 1].
            Higher values produce smoother surfaces.
        anisotropic: Enable per-particle WPCA anisotropic kernels.
            When ``False`` (default), all particles use isotropic kernels.
        k_r: Maximum eigenvalue ratio for anisotropy clamping.
        k_s: Covariance scaling factor.
        k_n: Isotropic fallback scale for isolated particles.
        n_epsilon: Minimum neighbor count for anisotropic kernels.
        padding: Extra voxels added around the particle bounding box.
        field_smooth_iterations: Number of separable Gaussian blur passes
            applied to the scalar field before marching cubes.  Smooths
            the transition zone to reduce MC staircase artifacts.
        field_smooth_radius: Half-width of the Gaussian blur in voxels.
        redistance_iterations: Number of Eikonal redistancing iterations
            applied after converting the density field to a signed
            distance field.  Improves SDF quality (|∇d| ≈ 1) away from
            the surface.  Set to 0 to skip.
        mesh_smooth_iterations: Number of Laplacian smoothing passes
            applied to the extracted mesh.  Set to 0 to disable.
        mesh_smooth_lambda: Laplacian step size [0, 1].
        device: Warp device for computation.
    """

    def __init__(
        self,
        voxel_size: float,
        kernel_radius: float | None = None,
        threshold: float = 0.5,
        smooth_lambda: float = 0.9,
        anisotropic: bool = False,
        k_r: float = 4.0,
        k_s: float = 10.0,
        k_n: float = 0.5,
        n_epsilon: int = 25,
        padding: int = 2,
        field_smooth_iterations: int = 1,
        field_smooth_radius: int = 2,
        redistance_iterations: int = 0,
        mesh_smooth_iterations: int = 0,
        mesh_smooth_lambda: float = 1.0,
        device: wp.DeviceLike = None,
    ):
        self.voxel_size = voxel_size
        self.kernel_radius = kernel_radius if kernel_radius is not None else 3.0 * voxel_size
        self.anisotropic = anisotropic
        self.threshold = threshold
        self.smooth_lambda = smooth_lambda
        self.k_r = k_r
        self.k_s = k_s
        self.k_n = k_n
        self.n_epsilon = n_epsilon
        self.padding = padding
        self.field_smooth_iterations = field_smooth_iterations
        self.field_smooth_radius = field_smooth_radius
        self.redistance_iterations = redistance_iterations
        self.mesh_smooth_iterations = mesh_smooth_iterations
        self.mesh_smooth_lambda = mesh_smooth_lambda

        self._device = wp.get_device() if device is None else wp.get_device(device)

        # Cached objects (allocated lazily)
        self._mc: wp.MarchingCubes | None = None
        self._hash_grid: wp.HashGrid | None = None
        self._blur_temp: wp.array | None = None
        self._blur_weights: wp.array | None = None
        self._field: wp.array | None = None
        self._grid_dims: tuple[int, int, int] | None = None
        self._grid_origin: wp.vec3 | None = None
        self._hash_grid_dim: int = 0

        # Per-particle temporaries
        self._smoothed: wp.array | None = None
        self._G: wp.array | None = None
        self._det_G: wp.array | None = None
        self._n_particles: int = 0

        # Last extraction results
        self._verts: wp.array | None = None
        self._indices: wp.array | None = None
        self._normals: wp.array | None = None

    # -- Public properties --

    @property
    def verts(self) -> wp.array | None:
        """Vertex positions from the last extraction."""
        return self._verts

    @property
    def indices(self) -> wp.array | None:
        """Triangle indices from the last extraction."""
        return self._indices

    @property
    def normals(self) -> wp.array | None:
        """Per-vertex normals from the last extraction."""
        return self._normals

    @property
    def field(self) -> wp.array | None:
        """Dense scalar field from the last extraction, shape ``(nx, ny, nz)``."""
        return self._field

    @property
    def grid_origin(self) -> wp.vec3 | None:
        """World-space position of grid node ``(0, 0, 0)``."""
        return self._grid_origin

    @property
    def grid_dims(self) -> tuple[int, int, int] | None:
        """Grid node counts ``(nx, ny, nz)``."""
        return self._grid_dims

    @property
    def smoothed_positions(self) -> wp.array | None:
        """Smoothed particle positions from the last extraction."""
        return self._smoothed

    @property
    def anisotropy_matrices(self) -> wp.array | None:
        """Per-particle anisotropy matrices G from the last extraction."""
        return self._G

    @property
    def anisotropy_det(self) -> wp.array | None:
        """Per-particle ``det(G)`` from the last extraction."""
        return self._det_G

    def fem_field(self) -> fem.DiscreteField:
        """Return the signed distance field as a :class:`warp.fem.DiscreteField`.

        The field is a signed distance function (negative inside, positive
        outside, zero at the surface) living on a Q1 (trilinear) function
        space over a :class:`warp.fem.Grid3D` matching the extraction grid.

        It can be used with :func:`warp.fem.interpolate` or
        :func:`warp.fem.integrate` to evaluate smooth values, gradients,
        and curvature at arbitrary positions.

        Must be called after :meth:`extract`.

        Returns:
            A :class:`warp.fem.DiscreteField` with scalar ``float`` DOFs
            representing signed distance values.
        """
        nx, ny, nz = self._grid_dims
        grid = fem.Grid3D(
            bounds_lo=self._grid_origin,
            bounds_hi=wp.vec3(
                self._grid_origin[0] + (nx - 1) * self.voxel_size,
                self._grid_origin[1] + (ny - 1) * self.voxel_size,
                self._grid_origin[2] + (nz - 1) * self.voxel_size,
            ),
            res=wp.vec3i(nx - 1, ny - 1, nz - 1),
        )
        space = fem.make_polynomial_space(grid, degree=1, dtype=float)
        discrete_field = fem.make_discrete_field(space)
        discrete_field.dof_values = self._field.flatten()
        return discrete_field

    # -- Core extraction --

    def extract(
        self,
        positions: wp.array,
        radii: wp.array,
        compute_normals: bool = True,
    ) -> tuple[wp.array | None, wp.array | None, wp.array | None]:
        """Extract a triangle mesh from particle positions.

        Args:
            positions: Particle positions, shape ``(N,)``, dtype ``wp.vec3``.
            radii: Per-particle radius [m].
            compute_normals: Whether to compute per-vertex normals.

        Returns:
            Tuple of ``(vertices, indices, normals)``.  All ``None`` when no
            surface can be extracted.
        """
        n = positions.shape[0]
        if n == 0:
            self._verts = self._indices = self._normals = None
            return None, None, None

        device = positions.device

        # Step 1: Compute AABB
        lower = wp.array([wp.vec3(1e30, 1e30, 1e30)], dtype=wp.vec3, device=device)
        upper = wp.array([wp.vec3(-1e30, -1e30, -1e30)], dtype=wp.vec3, device=device)
        wp.launch(_compute_aabb, dim=n, inputs=[positions, lower, upper], device=device)

        aabb_min = lower.numpy()[0]
        aabb_max = upper.numpy()[0]

        pad = self.kernel_radius + self.voxel_size * self.padding
        grid_min = np.floor((aabb_min - pad) / self.voxel_size) * self.voxel_size
        grid_max = np.ceil((aabb_max + pad) / self.voxel_size) * self.voxel_size
        dims = np.round((grid_max - grid_min) / self.voxel_size).astype(int) + 1

        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
        grid_origin = wp.vec3(float(grid_min[0]), float(grid_min[1]), float(grid_min[2]))
        grid_end = wp.vec3(float(grid_max[0]), float(grid_max[1]), float(grid_max[2]))

        # Step 2: Allocate / resize cached objects
        self._ensure_resources(nx, ny, nz, grid_origin, grid_end, n, device)

        # Step 3: Smooth particle positions (skip hash grid build if lambda ≈ 0)
        if self.smooth_lambda > 1e-6:
            self._hash_grid.build(positions, 1.5 * self.kernel_radius)
            wp.launch(
                _smooth_positions,
                dim=n,
                inputs=[self._hash_grid.id, positions, self.kernel_radius, self.smooth_lambda, self._smoothed],
                device=device,
            )
        else:
            wp.copy(self._smoothed, positions)

        # Step 4: Compute per-particle anisotropy
        if self.anisotropic:
            self._hash_grid.build(self._smoothed, 1.5 * self.kernel_radius)
            wp.launch(
                _compute_anisotropy,
                dim=n,
                inputs=[
                    self._hash_grid.id,
                    self._smoothed,
                    self.kernel_radius,
                    self.k_r,
                    self.k_s,
                    self.k_n,
                    self.n_epsilon,
                    self._G,
                    self._det_G,
                ],
                device=device,
            )
        else:
            wp.launch(
                _fill_isotropic_G,
                dim=n,
                inputs=[self.kernel_radius, self.k_n, self._G, self._det_G],
                device=device,
            )

        # Step 5: Evaluate scalar field — particle-centric splatting
        self._field.zero_()
        wp.launch(
            _eval_scalar_field,
            dim=n,
            inputs=[
                self._smoothed,
                radii,
                self._G,
                self._det_G,
                grid_origin,
                1.0 / self.voxel_size,
                nx,
                ny,
                nz,
                self._field,
            ],
            device=device,
        )

        # Step 5b: Gaussian blur on density field
        if self.field_smooth_iterations > 0 and self.field_smooth_radius > 0:
            self._apply_field_blur(nx, ny, nz, device)

        # Step 5c: Convert density → SDF (negative inside, positive outside)
        wp.launch(
            _density_to_sdf_3d, dim=(nx, ny, nz),
            inputs=[self._field, self.threshold], device=device,
        )

        # Step 5d: Redistancing (enforce |∇d| ≈ 1)
        if self.redistance_iterations > 0:
            self._apply_redistancing(nx, ny, nz, device)

        # Step 6: Marching cubes on the SDF (surface at d = 0).
        effective_threshold = 0.0
        if self.mesh_smooth_iterations > 0:
            shrink = 0.15 * math.sqrt(float(self.mesh_smooth_iterations)) * self.mesh_smooth_lambda * self.voxel_size
            effective_threshold = -shrink / self.kernel_radius

        self._mc.surface(self._field, effective_threshold)
        verts = self._mc.verts
        indices = self._mc.indices

        # MC orients normals from low→high. For SDF (low = inside),
        # that means normals point outward — correct convention, no flip needed.

        if verts is None or verts.shape[0] == 0:
            self._verts = self._indices = self._normals = None
            return None, None, None

        # Step 7: Laplacian smoothing
        if self.mesh_smooth_iterations > 0 and indices.shape[0] > 0:
            num_verts = verts.shape[0]
            num_tri_verts = indices.shape[0]
            smoothed = wp.empty(num_verts, dtype=wp.vec3, device=device)
            neighbor_sum = wp.zeros(num_verts, dtype=wp.vec3, device=device)
            valence = wp.zeros(num_verts, dtype=wp.int32, device=device)

            for _ in range(self.mesh_smooth_iterations):
                neighbor_sum.zero_()
                valence.zero_()
                wp.launch(_laplacian_scatter, dim=num_tri_verts, inputs=[indices, verts, neighbor_sum, valence], device=device)
                wp.launch(_laplacian_apply, dim=num_verts, inputs=[verts, neighbor_sum, valence, smoothed, self.mesh_smooth_lambda], device=device)
                verts, smoothed = smoothed, verts

        # Step 8: Vertex normals
        normals = None
        if compute_normals:
            normals = compute_vertex_normals(verts, indices)

        self._verts = verts
        self._indices = indices
        self._normals = normals
        return verts, indices, normals

    # -- Internal helpers --

    def _ensure_resources(
        self,
        nx: int,
        ny: int,
        nz: int,
        grid_origin: wp.vec3,
        grid_end: wp.vec3,
        n_particles: int,
        device: wp.DeviceLike,
    ):
        new_dims = (nx, ny, nz)

        if self._grid_dims != new_dims:
            self._mc = wp.MarchingCubes(nx, ny, nz)
            self._field = wp.empty((nx, ny, nz), dtype=wp.float32, device=device)
            if self.field_smooth_iterations > 0 and self.field_smooth_radius > 0:
                self._blur_temp = wp.empty((nx, ny, nz), dtype=wp.float32, device=device)
            self._grid_dims = new_dims

        self._mc.domain_bounds_lower_corner = grid_origin
        self._mc.domain_bounds_upper_corner = grid_end
        self._grid_origin = grid_origin

        extent = max(
            float(grid_end[0] - grid_origin[0]),
            float(grid_end[1] - grid_origin[1]),
            float(grid_end[2] - grid_origin[2]),
        )
        hash_dim = max(16, int(math.ceil(extent / self.kernel_radius)))
        if self._hash_grid is None or self._hash_grid_dim != hash_dim:
            self._hash_grid = wp.HashGrid(hash_dim, hash_dim, hash_dim, device=device)
            self._hash_grid_dim = hash_dim

        if self._n_particles != n_particles:
            self._smoothed = wp.empty(n_particles, dtype=wp.vec3, device=device)
            self._G = wp.empty(n_particles, dtype=wp.mat33, device=device)
            self._det_G = wp.empty(n_particles, dtype=float, device=device)
            self._n_particles = n_particles

    def _apply_field_blur(self, nx: int, ny: int, nz: int, device: wp.DeviceLike):
        """Separable Gaussian blur on the scalar field."""
        hw = self.field_smooth_radius
        if self._blur_weights is None:
            sigma = max(hw / 2.0, 0.5)
            w = np.array([math.exp(-0.5 * (d / sigma) ** 2) for d in range(hw + 1)], dtype=np.float32)
            w /= w[0] + 2.0 * np.sum(w[1:])
            self._blur_weights = wp.array(w, dtype=float, device=device)

        src = self._field
        dst = self._blur_temp
        w = self._blur_weights
        for _ in range(self.field_smooth_iterations):
            wp.launch(_blur_axis_x, dim=(nx, ny, nz), inputs=[src, dst, w, hw], device=device)
            wp.launch(_blur_axis_y, dim=(nx, ny, nz), inputs=[dst, src, w, hw], device=device)
            wp.launch(_blur_axis_z, dim=(nx, ny, nz), inputs=[src, dst, w, hw], device=device)
            src, dst = dst, src
        if src is not self._field:
            self._field, self._blur_temp = src, dst

    def _apply_redistancing(self, nx: int, ny: int, nz: int, device: wp.DeviceLike):
        """Iterative Eikonal redistancing to enforce |∇d| ≈ 1."""
        if self._blur_temp is None or self._blur_temp.shape != self._field.shape:
            self._blur_temp = wp.empty((nx, ny, nz), dtype=wp.float32, device=device)

        inv_dx = 1.0 / self.voxel_size
        src = self._field
        dst = self._blur_temp
        for _ in range(self.redistance_iterations):
            wp.launch(_redistance_step, dim=(nx, ny, nz), inputs=[src, dst, inv_dx], device=device)
            src, dst = dst, src
        if src is not self._field:
            self._field, self._blur_temp = src, dst


def extract_particle_surface(
    positions: wp.array,
    radii: wp.array,
    voxel_size: float,
    kernel_radius: float | None = None,
    threshold: float = 0.5,
    smooth_lambda: float = 0.9,
    k_s: float = 10.0,
    mesh_smooth_iterations: int = 0,
    compute_normals: bool = True,
) -> tuple[wp.array | None, wp.array | None, wp.array | None]:
    """Extract a triangle mesh from particle positions (one-shot convenience).

    Args:
        positions: Particle positions, shape ``(N,)``, dtype ``wp.vec3``.
        radii: Per-particle radius [m].
        voxel_size: Edge length of each grid voxel [m].
        kernel_radius: Search radius [m].  Defaults to ``3 * voxel_size``.
        threshold: Isosurface level.
        smooth_lambda: Position smoothing blend factor [0, 1].
        k_s: Covariance scaling factor.
        mesh_smooth_iterations: Laplacian mesh smoothing passes.
        compute_normals: Whether to compute per-vertex normals.

    Returns:
        Tuple of ``(vertices, indices, normals)``.
    """
    ctx = ParticleSurface(
        voxel_size=voxel_size,
        kernel_radius=kernel_radius,
        threshold=threshold,
        smooth_lambda=smooth_lambda,
        k_s=k_s,
        mesh_smooth_iterations=mesh_smooth_iterations,
        device=positions.device,
    )
    return ctx.extract(positions, radii=radii, compute_normals=compute_normals)
