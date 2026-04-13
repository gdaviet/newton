# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example-side implicit MPM solver with surface tension.

This module intentionally keeps the surface extraction and surface-tension
implementation outside Newton's solver internals. It subclasses the public
:class:`newton.solvers.SolverImplicitMPM` and inserts the same surface-tension
pipeline that previously lived in the solver branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import warp as wp
import warp.fem as fem

import newton
from newton import geometry as nwgeo

from .particle_surface import ParticleSurface

__all__ = ["SolverImplicitMPMSurface"]

wp.set_module_options({"enable_backward": False})

INFINITY = wp.constant(1.0e12)
EPSILON = wp.constant(1.0e-12)
DEFAULT_CONTACT_ANGLE = np.pi / 2.0

GEO_SPHERE = wp.constant(int(newton.GeoType.SPHERE))
GEO_BOX = wp.constant(int(newton.GeoType.BOX))
GEO_CAPSULE = wp.constant(int(newton.GeoType.CAPSULE))
GEO_CYLINDER = wp.constant(int(newton.GeoType.CYLINDER))
GEO_ELLIPSOID = wp.constant(int(newton.GeoType.ELLIPSOID))
GEO_CONE = wp.constant(int(newton.GeoType.CONE))
GEO_MESH = wp.constant(int(newton.GeoType.MESH))
GEO_CONVEX_MESH = wp.constant(int(newton.GeoType.CONVEX_MESH))
GEO_PLANE = wp.constant(int(newton.GeoType.PLANE))
AXIS_Z = wp.constant(int(newton.Axis.Z))
COLLIDE_PARTICLES = wp.constant(int(newton.ShapeFlags.COLLIDE_PARTICLES))


@wp.func
def _sdf_to_indicator_value(d: float, interface_width: float):
    return wp.clamp(0.5 - d / (2.0 * interface_width), 0.0, 1.0)


@wp.func
def _sample_sdf_trilinear(
    field: wp.array3d[wp.float32],
    grid_origin: wp.vec3,
    inv_voxel_size: float,
    pos: wp.vec3,
):
    p = (pos - grid_origin) * inv_voxel_size
    i0 = int(wp.floor(p[0]))
    j0 = int(wp.floor(p[1]))
    k0 = int(wp.floor(p[2]))

    fx = p[0] - float(i0)
    fy = p[1] - float(j0)
    fz = p[2] - float(k0)

    i0 = wp.clamp(i0, 0, field.shape[0] - 2)
    j0 = wp.clamp(j0, 0, field.shape[1] - 2)
    k0 = wp.clamp(k0, 0, field.shape[2] - 2)

    c00 = field[i0, j0, k0] * (1.0 - fx) + field[i0 + 1, j0, k0] * fx
    c10 = field[i0, j0 + 1, k0] * (1.0 - fx) + field[i0 + 1, j0 + 1, k0] * fx
    c01 = field[i0, j0, k0 + 1] * (1.0 - fx) + field[i0 + 1, j0, k0 + 1] * fx
    c11 = field[i0, j0 + 1, k0 + 1] * (1.0 - fx) + field[i0 + 1, j0 + 1, k0 + 1] * fx

    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy

    return c0 * (1.0 - fz) + c1 * fz


@wp.func
def _world_shape_transform(
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    shape_id: int,
):
    body_id = shape_body[shape_id]
    X_ws = shape_transform[shape_id]
    if body_id >= 0:
        if body_q:
            X_ws = wp.transform_multiply(body_q[body_id], X_ws)
    return X_ws


@wp.func
def _transform_point_inverse(X: wp.transform, x: wp.vec3):
    return wp.quat_rotate_inv(wp.transform_get_rotation(X), x - wp.transform_get_translation(X))


@wp.func
def _sdf_ellipsoid(point: wp.vec3, radii: wp.vec3):
    q0 = wp.vec3(point[0] / radii[0], point[1] / radii[1], point[2] / radii[2])
    q1 = wp.vec3(
        point[0] / (radii[0] * radii[0]),
        point[1] / (radii[1] * radii[1]),
        point[2] / (radii[2] * radii[2]),
    )
    k0 = wp.length(q0)
    k1 = wp.length(q1)
    return k0 * (k0 - 1.0) / wp.max(k1, 1.0e-12)


@wp.func
def _shape_sdf_local(shape_type: int, shape_source_ptr: wp.uint64, scale: wp.vec3, point: wp.vec3):
    d = float(INFINITY)
    if shape_type == GEO_SPHERE:
        d = nwgeo.sdf_sphere(point, scale[0])
    elif shape_type == GEO_BOX:
        d = nwgeo.sdf_box(point, scale[0], scale[1], scale[2])
    elif shape_type == GEO_CAPSULE:
        d = nwgeo.sdf_capsule(point, scale[0], scale[1], int(AXIS_Z))
    elif shape_type == GEO_CYLINDER:
        d = nwgeo.sdf_cylinder(point, scale[0], scale[1], int(AXIS_Z))
    elif shape_type == GEO_ELLIPSOID:
        d = _sdf_ellipsoid(point, scale)
    elif shape_type == GEO_CONE:
        d = nwgeo.sdf_cone(point, scale[0], scale[1], int(AXIS_Z))
    elif shape_type == GEO_PLANE:
        d = point[2]
    elif shape_type == GEO_MESH or shape_type == GEO_CONVEX_MESH:
        d = nwgeo.sdf_mesh(shape_source_ptr, point, 1.0e6)
    return d


@wp.func
def _shape_sdf_world(
    shape_type: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_id: int,
    x: wp.vec3,
):
    X_ws = _world_shape_transform(shape_transform, shape_body, body_q, shape_id)
    x_local = _transform_point_inverse(X_ws, x)
    return (
        _shape_sdf_local(shape_type[shape_id], shape_source_ptr[shape_id], shape_scale[shape_id], x_local)
        - shape_margin[shape_id]
    )


@wp.func
def _shape_normal_world(
    shape_type: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_id: int,
    x: wp.vec3,
):
    eps = float(1.0e-4)
    ex = wp.vec3(eps, 0.0, 0.0)
    ey = wp.vec3(0.0, eps, 0.0)
    ez = wp.vec3(0.0, 0.0, eps)
    g = wp.vec3(
        _shape_sdf_world(
            shape_type, shape_transform, shape_body, shape_scale, shape_source_ptr, shape_margin, body_q, shape_id, x + ex
        )
        - _shape_sdf_world(
            shape_type, shape_transform, shape_body, shape_scale, shape_source_ptr, shape_margin, body_q, shape_id, x - ex
        ),
        _shape_sdf_world(
            shape_type, shape_transform, shape_body, shape_scale, shape_source_ptr, shape_margin, body_q, shape_id, x + ey
        )
        - _shape_sdf_world(
            shape_type, shape_transform, shape_body, shape_scale, shape_source_ptr, shape_margin, body_q, shape_id, x - ey
        ),
        _shape_sdf_world(
            shape_type, shape_transform, shape_body, shape_scale, shape_source_ptr, shape_margin, body_q, shape_id, x + ez
        )
        - _shape_sdf_world(
            shape_type, shape_transform, shape_body, shape_scale, shape_source_ptr, shape_margin, body_q, shape_id, x - ez
        ),
    )
    g_len = wp.length(g)
    return wp.where(g_len > 1.0e-12, g / g_len, wp.vec3(0.0, 0.0, 1.0))


@wp.func
def _model_shape_query(
    x: wp.vec3,
    shape_type: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_flags: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    shape_contact_angle: wp.array[float],
    body_q: wp.array[wp.transform],
):
    min_sdf = float(INFINITY)
    closest_shape = int(-1)
    contact_angle = float(DEFAULT_CONTACT_ANGLE)

    for shape_id in range(shape_type.shape[0]):
        if (shape_flags[shape_id] & int(COLLIDE_PARTICLES)) == 0:
            continue

        d = _shape_sdf_world(
            shape_type,
            shape_transform,
            shape_body,
            shape_scale,
            shape_source_ptr,
            shape_margin,
            body_q,
            shape_id,
            x,
        )
        if d < min_sdf:
            min_sdf = d
            closest_shape = shape_id
            contact_angle = shape_contact_angle[shape_id]

    n = wp.vec3(0.0)
    if closest_shape >= 0:
        n = _shape_normal_world(
            shape_type,
            shape_transform,
            shape_body,
            shape_scale,
            shape_source_ptr,
            shape_margin,
            body_q,
            closest_shape,
            x,
        )

    return min_sdf, n, closest_shape, contact_angle


@fem.integrand
def integrate_fraction(s: fem.Sample, phi: fem.Field, domain: fem.Domain, inv_cell_volume: float):
    return phi(s) * inv_cell_volume


@wp.kernel
def _sdf_to_indicator(sdf: wp.array[float], interface_width: float):
    i = wp.tid()
    sdf[i] = _sdf_to_indicator_value(sdf[i], interface_width)


@wp.kernel
def normalize_gradient(
    gradient: wp.array[wp.vec3],
    node_volume: wp.array[float],
    normal: wp.array[wp.vec3],
):
    i = wp.tid()
    vol = wp.max(node_volume[i], EPSILON)
    g = gradient[i] / vol
    g_len = wp.length(g)
    normal[i] = wp.where(g_len > EPSILON, g / g_len, wp.vec3(0.0))


@wp.kernel
def compute_curvature_from_divergence(
    divergence: wp.array[float],
    node_volume: wp.array[float],
    curvature: wp.array[float],
):
    i = wp.tid()
    vol = wp.max(node_volume[i], EPSILON)
    curvature[i] = divergence[i] / vol


@fem.integrand
def recover_gradient_integrand(
    s: fem.Sample,
    u: fem.Field,
    indicator: fem.Field,
):
    return wp.dot(u(s), fem.grad(indicator, s))


@fem.integrand
def divergence_normal_integrand(
    s: fem.Sample,
    phi: fem.Field,
    normal: fem.Field,
):
    return phi(s) * wp.trace(fem.grad(normal, s))


@wp.kernel
def apply_contact_angle_bc(
    field: wp.array3d[wp.float32],
    field_orig: wp.array3d[wp.float32],
    grid_origin: wp.vec3,
    voxel_size: float,
    max_depth: float,
    shape_type: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_flags: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    shape_contact_angle: wp.array[float],
    body_q: wp.array[wp.transform],
):
    i, j, k = wp.tid()
    x = grid_origin + voxel_size * wp.vec3(float(i), float(j), float(k))
    d_coll, n_coll, _shape_id, contact_angle = _model_shape_query(
        x,
        shape_type,
        shape_transform,
        shape_body,
        shape_flags,
        shape_scale,
        shape_source_ptr,
        shape_margin,
        shape_contact_angle,
        body_q,
    )

    if d_coll >= 0.0:
        return

    depth = -d_coll
    if depth > max_depth:
        return

    x_mirror = x - 2.0 * d_coll * n_coll
    phi_mirror = _sample_sdf_trilinear(field_orig, grid_origin, 1.0 / voxel_size, x_mirror)
    phi_reflected = phi_mirror + 2.0 * d_coll * wp.cos(contact_angle)

    blend_start = 0.5 * max_depth
    if depth <= blend_start:
        field[i, j, k] = phi_reflected
    else:
        t = (depth - blend_start) / (max_depth - blend_start)
        blend = t * t * (3.0 - 2.0 * t)
        field[i, j, k] = (1.0 - blend) * phi_reflected + blend * field_orig[i, j, k]


@wp.kernel
def union_particle_collider_sdf(
    field: wp.array3d[wp.float32],
    field_orig: wp.array3d[wp.float32],
    grid_origin: wp.vec3,
    voxel_size: float,
    onset: float,
    max_depth: float,
    shape_type: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_flags: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    shape_contact_angle: wp.array[float],
    body_q: wp.array[wp.transform],
):
    i, j, k = wp.tid()
    x = grid_origin + voxel_size * wp.vec3(float(i), float(j), float(k))
    d_coll, n_coll, _shape_id, contact_angle = _model_shape_query(
        x,
        shape_type,
        shape_transform,
        shape_body,
        shape_flags,
        shape_scale,
        shape_source_ptr,
        shape_margin,
        shape_contact_angle,
        body_q,
    )

    depth = onset - d_coll
    if depth > max_depth or depth < 0.0:
        return

    x_mirror = x + 2.0 * (onset - d_coll) * n_coll
    phi_mirror = _sample_sdf_trilinear(field_orig, grid_origin, 1.0 / voxel_size, x_mirror)
    theta = contact_angle
    phi_reflected = phi_mirror * wp.sin(theta) + 2.0 * (d_coll - onset) * wp.cos(theta)

    blend_start = 0.5 * max_depth
    if depth <= blend_start:
        field[i, j, k] = phi_reflected
    else:
        t = (depth - blend_start) / (max_depth - blend_start)
        blend = t * t * (3.0 - 2.0 * t)
        field[i, j, k] = (1.0 - blend) * phi_reflected + blend * field_orig[i, j, k]


@wp.kernel
def apply_virtual_surface_bc(
    field: wp.array3d[wp.float32],
    field_orig: wp.array3d[wp.float32],
    grid_origin: wp.vec3,
    voxel_size: float,
    max_depth: float,
    shape_type: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_flags: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    shape_contact_angle: wp.array[float],
    body_q: wp.array[wp.transform],
):
    i, j, k = wp.tid()
    x = grid_origin + voxel_size * wp.vec3(float(i), float(j), float(k))
    d_coll, n_coll, _shape_id, contact_angle = _model_shape_query(
        x,
        shape_type,
        shape_transform,
        shape_body,
        shape_flags,
        shape_scale,
        shape_source_ptr,
        shape_margin,
        shape_contact_angle,
        body_q,
    )

    if d_coll >= 0.0:
        return

    depth = -d_coll
    if depth > max_depth:
        return

    x_wall = x - d_coll * n_coll
    phi_wall = _sample_sdf_trilinear(field_orig, grid_origin, 1.0 / voxel_size, x_wall)
    phi_virtual = phi_wall - d_coll * wp.cos(contact_angle)

    blend_start = 0.5 * max_depth
    if depth <= blend_start:
        field[i, j, k] = phi_virtual
    else:
        t = (depth - blend_start) / (max_depth - blend_start)
        blend = t * t * (3.0 - 2.0 * t)
        field[i, j, k] = (1.0 - blend) * phi_virtual + blend * field_orig[i, j, k]


@fem.integrand
def integrate_csf_force(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    curvature: fem.Field,
    indicator: fem.Field,
    dt: float,
    inv_cell_volume: float,
    surface_tension_coefficient: float,
):
    kappa = curvature(s)
    grad_c = fem.grad(indicator, s)
    f_st = surface_tension_coefficient * kappa * grad_c
    return wp.dot(u(s), dt * f_st) * inv_cell_volume


@fem.integrand
def integrate_csf_force_contact_angle(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    curvature: fem.Field,
    indicator: fem.Field,
    particle_q: wp.array[wp.vec3],
    shape_type: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_flags: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    shape_contact_angle: wp.array[float],
    body_q: wp.array[wp.transform],
    dt: float,
    inv_cell_volume: float,
    surface_tension_coefficient: float,
    activation_dist: float,
):
    kappa = curvature(s)
    grad_c = fem.grad(indicator, s)
    x = particle_q[s.qp_index]
    d_coll, _n, _shape_id, contact_angle = _model_shape_query(
        x,
        shape_type,
        shape_transform,
        shape_body,
        shape_flags,
        shape_scale,
        shape_source_ptr,
        shape_margin,
        shape_contact_angle,
        body_q,
    )

    sigma_lg = surface_tension_coefficient
    sigma_sl = -sigma_lg * wp.cos(contact_angle)
    t = wp.clamp(d_coll / activation_dist, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    sigma = (1.0 - t) * sigma_sl + t * sigma_lg

    f_st = sigma * kappa * grad_c
    return wp.dot(u(s), dt * f_st) * inv_cell_volume


@fem.integrand
def integrate_csf_force_angle_mask(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    curvature: fem.Field,
    indicator: fem.Field,
    sdf_normal: fem.Field,
    particle_q: wp.array[wp.vec3],
    shape_type: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_flags: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    shape_contact_angle: wp.array[float],
    body_q: wp.array[wp.transform],
    dt: float,
    inv_cell_volume: float,
    surface_tension_coefficient: float,
    activation_dist: float,
    falloff: float,
):
    kappa = curvature(s)
    grad_c = fem.grad(indicator, s)
    f_st = surface_tension_coefficient * kappa * grad_c

    x = particle_q[s.qp_index]
    d_coll, n_coll, _shape_id, contact_angle = _model_shape_query(
        x,
        shape_type,
        shape_transform,
        shape_body,
        shape_flags,
        shape_scale,
        shape_source_ptr,
        shape_margin,
        shape_contact_angle,
        body_q,
    )

    if d_coll < activation_dist:
        n_fluid = sdf_normal(s)
        cos_angle = wp.dot(n_fluid, n_coll)
        t = wp.clamp((cos_angle - wp.cos(contact_angle)) / falloff + 0.5, 0.0, 1.0)
        mask = t * t * (3.0 - 2.0 * t)
        f_st = mask * f_st

    return wp.dot(u(s), dt * f_st) * inv_cell_volume


@wp.kernel
def add_velocity_increment(
    force_int: wp.array[wp.vec3],
    inv_mass_matrix: wp.array[float],
    velocity: wp.array[wp.vec3],
):
    i = wp.tid()
    velocity[i] += force_int[i] * inv_mass_matrix[i]


class SolverImplicitMPMSurface(newton.solvers.SolverImplicitMPM):
    """Implicit MPM solver with example-side surface extraction and tension."""

    @dataclass
    class Config(newton.solvers.SolverImplicitMPM.Config):
        """Surface solver configuration."""

        contact_angle_mode: Literal["force", "sdf", "union", "virtual"] = "virtual"
        """Contact-angle mode for wetting."""
        contact_angle_force: Literal["angle_mask", "dual_coeff"] = "angle_mask"
        """Contact-angle force model."""
        surface_voxel_size: float | None = None
        """Voxel size used by the surface SDF grid [m]."""
        surface_kernel_radius: float | None = None
        """Particle splatting kernel radius for the surface SDF [m]."""
        surface_threshold: float = 0.15
        """Surface SDF density threshold."""
        surface_padding: int | None = None
        """Extra voxels added around the particle bounding box."""
        surface_field_smooth_radius: int = 1
        """Gaussian blur half-width for the surface field."""
        surface_field_smooth_iterations: int = 2
        """Gaussian blur iterations for the surface field."""
        surface_redistance_iterations: int = 10
        """Redistancing iterations for the surface SDF."""

    @classmethod
    def register_custom_attributes(cls, builder: newton.ModelBuilder) -> None:
        """Register upstream implicit MPM attributes and ``mpm:surface_tension``."""
        newton.solvers.SolverImplicitMPM.register_custom_attributes(builder)
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="surface_tension",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mpm",
            )
        )

    def __init__(
        self,
        model: newton.Model,
        config: Config,
        temporary_store=None,
        verbose: bool | None = None,
        enable_timers: bool = False,
    ):
        super().__init__(
            model,
            config,
            temporary_store=temporary_store,
            verbose=verbose,
            enable_timers=enable_timers,
        )
        self.config = config
        self._options = config
        self._surface_collider_model = model
        self._shape_contact_angle = wp.full(
            shape=model.shape_count,
            value=float(DEFAULT_CONTACT_ANGLE),
            dtype=float,
            device=model.device,
        )
        self._has_contact_angle_value = False
        self._st_surface: ParticleSurface | None = None
        self._st_field_orig: wp.array3d[wp.float32] | None = None
        self._last_pic = None

    @property
    def particle_surface(self) -> ParticleSurface | None:
        """Surface extraction context used by the surface-tension solve."""
        return self._st_surface

    def setup_collider(
        self,
        collider_meshes: list[wp.Mesh] | None = None,
        collider_body_ids: list[int] | None = None,
        collider_margins: list[float] | None = None,
        collider_friction: list[float] | None = None,
        collider_adhesion: list[float] | None = None,
        collider_projection_threshold: list[float] | None = None,
        collider_contact_angle: list[float] | None = None,
        model: newton.Model | None = None,
        body_com: wp.array[wp.vec3] | None = None,
        body_mass: wp.array[float] | None = None,
        body_inv_inertia: wp.array[wp.mat33] | None = None,
        body_q: wp.array[wp.transform] | None = None,
    ) -> None:
        """Configure upstream MPM colliders and optional contact angles."""
        super().setup_collider(
            collider_meshes=collider_meshes,
            collider_body_ids=collider_body_ids,
            collider_margins=collider_margins,
            collider_friction=collider_friction,
            collider_adhesion=collider_adhesion,
            collider_projection_threshold=collider_projection_threshold,
            model=model,
            body_com=body_com,
            body_mass=body_mass,
            body_inv_inertia=body_inv_inertia,
            body_q=body_q,
        )

        self._surface_collider_model = model or self.model
        self._shape_contact_angle = self._make_shape_contact_angle(collider_contact_angle)

    def create_particle_surface(self, voxel_size: float | None = None, **kwargs) -> ParticleSurface:
        """Create a reusable particle surface extraction context."""
        if voxel_size is None:
            voxel_size = self.config.voxel_size * 0.5
        if kwargs.get("kernel_radius") is None:
            kwargs["kernel_radius"] = 1.5 * self.config.voxel_size
        return ParticleSurface(voxel_size=voxel_size, **kwargs)

    def extract_particle_surface(
        self,
        state: newton.State,
        surface: ParticleSurface,
        compute_normals: bool = True,
    ) -> tuple[wp.array[wp.vec3] | None, wp.array[wp.int32] | None, wp.array[wp.vec3] | None]:
        """Extract a particle surface mesh from ``state``."""
        return surface.extract(state.particle_q, radii=self._mpm_model.particle_radius, compute_normals=compute_normals)

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        dt: float,
    ) -> None:
        """Advance the simulation and keep the PIC quadrature for field gathers."""
        model = self.model
        with wp.ScopedDevice(model.device):
            pic = self._particles_to_cells(state_in.particle_q)
            scratch = self._rebuild_scratchpad(pic)
            self._last_pic = pic
            self._step_impl(state_in, state_out, dt, pic, scratch)
            scratch.release_temporaries()

    def _compute_unconstrained_velocity(
        self,
        state_in: newton.State,
        dt: float,
        pic,
        scratch,
        inv_cell_volume: float,
    ):
        if self._has_surface_tension():
            self._build_indicator_field(state_in, pic, scratch, inv_cell_volume)

        super()._compute_unconstrained_velocity(state_in, dt, pic, scratch, inv_cell_volume)

        if self._has_surface_tension():
            self._apply_surface_tension_force(state_in, dt, pic, scratch, inv_cell_volume)

    def gather_surface_tension_fields(self) -> dict[str, wp.array[float]]:
        """Gather surface tension indicator and curvature at particle positions."""
        scratch = self._scratchpad
        if not self._has_surface_tension() or self._last_pic is None or not hasattr(scratch, "_st_curvature_field"):
            return {}

        n = self.model.particle_count
        indicator_vals = wp.zeros(n, dtype=float, device=self.model.device)
        curvature_vals = wp.zeros(n, dtype=float, device=self.model.device)

        fem.interpolate(scratch.fraction_field, dest=indicator_vals, at=self._last_pic)
        fem.interpolate(scratch._st_curvature_field, dest=curvature_vals, at=self._last_pic)

        return {"indicator": indicator_vals, "curvature": curvature_vals}

    def _make_shape_contact_angle(self, collider_contact_angle: list[float] | None) -> wp.array[float]:
        collider_model = self._surface_collider_model
        angles = np.full(collider_model.shape_count, float(DEFAULT_CONTACT_ANGLE), dtype=np.float32)
        if collider_contact_angle is not None:
            values = np.asarray(collider_contact_angle, dtype=np.float32)
            flags = collider_model.shape_flags.numpy()
            shape_ids = np.flatnonzero((flags & int(newton.ShapeFlags.COLLIDE_PARTICLES)) != 0)
            if len(values) == 1:
                angles[shape_ids] = values[0]
            elif len(values) == collider_model.shape_count:
                angles[:] = values
            else:
                angles[shape_ids[: len(values)]] = values
        self._has_contact_angle_value = bool(np.any(np.abs(angles - float(DEFAULT_CONTACT_ANGLE)) > 1.0e-6))
        return wp.array(angles, dtype=float, device=collider_model.device)

    def _has_surface_tension(self) -> bool:
        return hasattr(self.model, "mpm") and hasattr(self.model.mpm, "surface_tension") and bool(
            np.any(self.model.mpm.surface_tension.numpy() > 0.0)
        )

    def _surface_tension_coeff(self) -> float:
        if not hasattr(self.model, "mpm") or not hasattr(self.model.mpm, "surface_tension"):
            return 0.0
        return float(np.max(self.model.mpm.surface_tension.numpy()))

    def _has_contact_angle(self) -> bool:
        return self._has_contact_angle_value

    def _build_indicator_field(self, state_in: newton.State, pic, scratch, inv_cell_volume: float) -> None:
        vel_node_count = scratch.fraction_field.space_partition.node_count()
        mpm_model = self._mpm_model

        with self._timer("Indicator field"):
            if self._st_surface is None:
                padding = self.config.surface_padding
                if padding is None:
                    padding = 8 if self.config.contact_angle_mode in ("union", "virtual") else 4
                self._st_surface = ParticleSurface(
                    voxel_size=self.config.surface_voxel_size or 0.5 * mpm_model.voxel_size,
                    kernel_radius=self.config.surface_kernel_radius or 1.5 * mpm_model.voxel_size,
                    threshold=self.config.surface_threshold,
                    padding=padding,
                    field_smooth_radius=self.config.surface_field_smooth_radius,
                    field_smooth_iterations=self.config.surface_field_smooth_iterations,
                    redistance_iterations=self.config.surface_redistance_iterations,
                    device=self.model.device,
                )

            self._st_surface.extract(state_in.particle_q, radii=mpm_model.particle_radius, compute_normals=False)

            if self._has_contact_angle():
                if self.config.contact_angle_mode == "sdf":
                    self._apply_contact_angle_bc(state_in)
                    self._st_surface.resurface()
                elif self.config.contact_angle_mode == "union":
                    self._union_collider_sdf(state_in)
                    self._st_surface.resurface()
                elif self.config.contact_angle_mode == "virtual":
                    self._apply_virtual_surface_bc(state_in)
                    self._st_surface.resurface()

            sdf_field = self._st_surface.fem_field()
            sdf_grid = sdf_field.space.geometry
            sdf_domain = fem.Cells(sdf_grid)
            sdf_node_count = sdf_field.space.node_count()

            sdf_scalar_space = sdf_field.space
            sdf_vec3_space = fem.make_collocated_function_space(sdf_scalar_space.basis, dtype=wp.vec3)

            sdf_scalar_test = fem.make_test(sdf_scalar_space, domain=sdf_domain)
            sdf_vec3_test = fem.make_test(sdf_vec3_space, domain=sdf_domain)

            sdf_node_volume = fem.integrate(
                integrate_fraction,
                fields={"phi": sdf_scalar_test},
                values={"inv_cell_volume": 1.0},
                assembly="nodal",
                output_dtype=float,
                temporary_store=self.temporary_store,
            )

            gradient_int = fem.integrate(
                recover_gradient_integrand,
                fields={"u": sdf_vec3_test, "indicator": sdf_field},
                assembly="nodal",
                output_dtype=wp.vec3,
                temporary_store=self.temporary_store,
            )

            sdf_normal = sdf_vec3_space.make_field()
            wp.launch(normalize_gradient, dim=sdf_node_count, inputs=[gradient_int, sdf_node_volume, sdf_normal.dof_values])

            divergence_int = fem.integrate(
                divergence_normal_integrand,
                fields={"phi": sdf_scalar_test, "normal": sdf_normal},
                assembly="nodal",
                output_dtype=float,
                temporary_store=self.temporary_store,
            )

            sdf_curvature = fem.make_discrete_field(sdf_scalar_space)
            wp.launch(
                compute_curvature_from_divergence,
                dim=sdf_node_count,
                inputs=[divergence_int, sdf_node_volume, sdf_curvature.dof_values],
            )

            scratch._st_curvature_field = fem.NonconformingField(domain=pic.domain, field=sdf_curvature, background=0.0)
            scratch._st_normal_field = fem.NonconformingField(domain=pic.domain, field=sdf_normal, background=wp.vec3(0.0))

            sdf_nc = fem.NonconformingField(domain=pic.domain, field=sdf_field, background=3.0 * mpm_model.voxel_size)
            fem.interpolate(sdf_nc, dest=scratch.fraction_field)
            wp.launch(_sdf_to_indicator, dim=vel_node_count, inputs=[scratch.fraction_field.dof_values, mpm_model.voxel_size])

    def _apply_contact_angle_bc(self, state_in: newton.State) -> None:
        surface = self._st_surface
        nx, ny, nz = surface.grid_dims
        self._copy_surface_field_orig(surface)

        wp.launch(
            apply_contact_angle_bc,
            dim=(nx, ny, nz),
            inputs=[
                surface.field,
                self._st_field_orig,
                surface.grid_origin,
                surface.voxel_size,
                8.0 * surface.voxel_size,
                self._surface_collider_model.shape_type,
                self._surface_collider_model.shape_transform,
                self._surface_collider_model.shape_body,
                self._surface_collider_model.shape_flags,
                self._surface_collider_model.shape_scale,
                self._surface_collider_model.shape_source_ptr,
                self._surface_collider_model.shape_margin,
                self._shape_contact_angle,
                state_in.body_q,
            ],
            device=self.model.device,
        )

    def _union_collider_sdf(self, state_in: newton.State) -> None:
        surface = self._st_surface
        nx, ny, nz = surface.grid_dims
        self._copy_surface_field_orig(surface)

        wp.launch(
            union_particle_collider_sdf,
            dim=(nx, ny, nz),
            inputs=[
                surface.field,
                self._st_field_orig,
                surface.grid_origin,
                surface.voxel_size,
                0.0 * self._mpm_model.voxel_size,
                8.0 * surface.voxel_size,
                self._surface_collider_model.shape_type,
                self._surface_collider_model.shape_transform,
                self._surface_collider_model.shape_body,
                self._surface_collider_model.shape_flags,
                self._surface_collider_model.shape_scale,
                self._surface_collider_model.shape_source_ptr,
                self._surface_collider_model.shape_margin,
                self._shape_contact_angle,
                state_in.body_q,
            ],
            device=self.model.device,
        )

        surface._apply_redistancing(nx, ny, nz, surface._device)
        surface._apply_field_blur(nx, ny, nz, surface._device)
        surface._apply_redistancing(nx, ny, nz, surface._device)

    def _apply_virtual_surface_bc(self, state_in: newton.State) -> None:
        surface = self._st_surface
        nx, ny, nz = surface.grid_dims
        self._copy_surface_field_orig(surface)

        wp.launch(
            apply_virtual_surface_bc,
            dim=(nx, ny, nz),
            inputs=[
                surface.field,
                self._st_field_orig,
                surface.grid_origin,
                surface.voxel_size,
                8.0 * surface.voxel_size,
                self._surface_collider_model.shape_type,
                self._surface_collider_model.shape_transform,
                self._surface_collider_model.shape_body,
                self._surface_collider_model.shape_flags,
                self._surface_collider_model.shape_scale,
                self._surface_collider_model.shape_source_ptr,
                self._surface_collider_model.shape_margin,
                self._shape_contact_angle,
                state_in.body_q,
            ],
            device=self.model.device,
        )

        if surface.redistance_iterations > 0:
            surface._apply_redistancing(nx, ny, nz, surface._device)

    def _copy_surface_field_orig(self, surface: ParticleSurface) -> None:
        if self._st_field_orig is None or self._st_field_orig.shape != surface.field.shape:
            self._st_field_orig = wp.empty_like(surface.field)
        wp.copy(self._st_field_orig, surface.field)

    def _apply_surface_tension_force(self, state_in: newton.State, dt: float, pic, scratch, inv_cell_volume: float) -> None:
        with self._timer("Surface tension"):
            surface_tension_coeff = self._surface_tension_coeff()
            if surface_tension_coeff <= 0.0:
                return

            output = fem.integrate(
                integrate_csf_force,
                quadrature=pic,
                fields={
                    "u": scratch.velocity_test,
                    "curvature": scratch._st_curvature_field,
                    "indicator": scratch.fraction_field,
                },
                values={
                    "dt": dt,
                    "inv_cell_volume": inv_cell_volume,
                    "surface_tension_coefficient": surface_tension_coeff,
                },
                output_dtype=wp.vec3,
                temporary_store=self.temporary_store,
            )

            if self._has_contact_angle() and self.config.contact_angle_mode in ("force", "union", "virtual"):
                if self.config.contact_angle_force == "dual_coeff":
                    output = fem.integrate(
                        integrate_csf_force_contact_angle,
                        quadrature=pic,
                        fields={
                            "u": scratch.velocity_test,
                            "curvature": scratch._st_curvature_field,
                            "indicator": scratch.fraction_field,
                        },
                        values={
                            "particle_q": state_in.particle_q,
                            "shape_type": self._surface_collider_model.shape_type,
                            "shape_transform": self._surface_collider_model.shape_transform,
                            "shape_body": self._surface_collider_model.shape_body,
                            "shape_flags": self._surface_collider_model.shape_flags,
                            "shape_scale": self._surface_collider_model.shape_scale,
                            "shape_source_ptr": self._surface_collider_model.shape_source_ptr,
                            "shape_margin": self._surface_collider_model.shape_margin,
                            "shape_contact_angle": self._shape_contact_angle,
                            "body_q": state_in.body_q,
                            "dt": dt,
                            "inv_cell_volume": inv_cell_volume,
                            "surface_tension_coefficient": surface_tension_coeff,
                            "activation_dist": 1.5 * self._mpm_model.voxel_size,
                        },
                        output_dtype=wp.vec3,
                        temporary_store=self.temporary_store,
                    )
                else:
                    output = fem.integrate(
                        integrate_csf_force_angle_mask,
                        quadrature=pic,
                        fields={
                            "u": scratch.velocity_test,
                            "curvature": scratch._st_curvature_field,
                            "indicator": scratch.fraction_field,
                            "sdf_normal": scratch._st_normal_field,
                        },
                        values={
                            "particle_q": state_in.particle_q,
                            "shape_type": self._surface_collider_model.shape_type,
                            "shape_transform": self._surface_collider_model.shape_transform,
                            "shape_body": self._surface_collider_model.shape_body,
                            "shape_flags": self._surface_collider_model.shape_flags,
                            "shape_scale": self._surface_collider_model.shape_scale,
                            "shape_source_ptr": self._surface_collider_model.shape_source_ptr,
                            "shape_margin": self._surface_collider_model.shape_margin,
                            "shape_contact_angle": self._shape_contact_angle,
                            "body_q": state_in.body_q,
                            "dt": dt,
                            "inv_cell_volume": inv_cell_volume,
                            "surface_tension_coefficient": surface_tension_coeff,
                            "activation_dist": 1.5 * self._mpm_model.voxel_size,
                            "falloff": 0.2,
                        },
                        output_dtype=wp.vec3,
                        temporary_store=self.temporary_store,
                    )

            wp.launch(
                add_velocity_increment,
                dim=scratch.velocity_node_count,
                inputs=[output, scratch.inv_mass_matrix, scratch.velocity_field.dof_values],
                device=self.model.device,
            )
