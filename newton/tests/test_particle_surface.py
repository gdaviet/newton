# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp
import warp.fem as fem

from newton._src.solvers.implicit_mpm.particle_surface import ParticleSurface, _redistance_step, extract_particle_surface
from newton.tests.unittest_utils import add_function_test


def _make_sphere_sdf(N, dx, R):
    """Create an exact sphere SDF on an NxNxN grid centered in the domain."""
    center = (N - 1) * dx / 2.0
    coords = np.arange(N, dtype=np.float32) * dx
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    sdf = np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2) - R
    return sdf.astype(np.float32), center


def _gradient_magnitude(sdf_np, dx):
    """Central finite differences, returns |grad| on interior [1:-1,1:-1,1:-1]."""
    gx = (sdf_np[2:, 1:-1, 1:-1] - sdf_np[:-2, 1:-1, 1:-1]) / (2.0 * dx)
    gy = (sdf_np[1:-1, 2:, 1:-1] - sdf_np[1:-1, :-2, 1:-1]) / (2.0 * dx)
    gz = (sdf_np[1:-1, 1:-1, 2:] - sdf_np[1:-1, 1:-1, :-2]) / (2.0 * dx)
    return np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)


def _make_sphere_particles(n=10000, seed=42, device=None):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-1, 1, size=(n * 2, 3))
        pts.append(p[np.linalg.norm(p, axis=1) < 1.0])
    pts = np.concatenate(pts)[:n].astype(np.float32)
    positions = wp.array(pts, dtype=wp.vec3, device=device)
    radii = wp.full(n, value=0.05, dtype=float, device=device)
    return positions, radii


def test_one_shot(test, device):
    positions, radii = _make_sphere_particles(device=device)
    verts, indices, normals = extract_particle_surface(
        positions, radii, voxel_size=0.05, kernel_radius=0.15,
    )
    test.assertIsNotNone(verts)
    test.assertGreater(verts.shape[0], 0)
    test.assertGreater(indices.shape[0], 0)
    test.assertEqual(indices.shape[0] % 3, 0)
    test.assertEqual(normals.shape[0], verts.shape[0])


def test_reusable_context(test, device):
    positions, radii = _make_sphere_particles(device=device)
    ctx = ParticleSurface(voxel_size=0.05, kernel_radius=0.15, device=device)
    verts, _, _ = ctx.extract(positions, radii=radii)
    test.assertIsNotNone(verts)

    verts2, _, _ = ctx.extract(positions, radii=radii)
    test.assertIsNotNone(verts2)


def test_mesh_smoothing(test, device):
    positions, radii = _make_sphere_particles(device=device)
    ctx = ParticleSurface(voxel_size=0.05, kernel_radius=0.15, mesh_smooth_iterations=3, device=device)
    verts, _, _ = ctx.extract(positions, radii=radii)
    test.assertIsNotNone(verts)


def test_empty_particles(test, device):
    positions = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    radii = wp.array(np.zeros(0, dtype=np.float32), dtype=float, device=device)
    ctx = ParticleSurface(voxel_size=0.05, device=device)
    verts, indices, normals = ctx.extract(positions, radii=radii)
    test.assertIsNone(verts)
    test.assertIsNone(indices)
    test.assertIsNone(normals)


def test_fem_field(test, device):
    positions, radii = _make_sphere_particles(device=device)
    ctx = ParticleSurface(voxel_size=0.1, kernel_radius=0.3, mesh_smooth_iterations=0, device=device)
    ctx.extract(positions, radii=radii)

    sdf = ctx.fem_field()
    field_np = ctx.field.numpy()
    origin = np.array([ctx.grid_origin[i] for i in range(3)])
    dx = ctx.voxel_size
    dims = ctx.grid_dims

    # Query at exact grid node positions — Q1 interpolation must reproduce DOF values.
    node_pts = []
    node_vals = []
    for ix in range(1, dims[0] - 1, 3):
        for iy in range(1, dims[1] - 1, 3):
            for iz in range(1, dims[2] - 1, 3):
                node_pts.append(origin + np.array([ix, iy, iz]) * dx)
                node_vals.append(field_np[ix, iy, iz])
    node_pts = np.array(node_pts, dtype=np.float32)
    node_vals = np.array(node_vals, dtype=np.float32)

    query_wp = wp.array(node_pts, dtype=wp.vec3, device=device)
    domain = fem.Cells(sdf.space.geometry)
    pic = fem.PicQuadrature(domain, positions=query_wp)
    fem_values = wp.zeros(len(node_pts), dtype=float, device=device)
    fem.interpolate(sdf, dest=fem_values, at=pic)

    diff = np.abs(fem_values.numpy() - node_vals)
    test.assertLess(
        diff.max(), 1e-4, f"FEM interpolation at grid nodes should be exact, got max_diff={diff.max():.6f}"
    )


def test_anisotropic(test, device):
    positions, radii = _make_sphere_particles(device=device)
    ctx = ParticleSurface(
        voxel_size=0.05, kernel_radius=0.15, anisotropic=True, mesh_smooth_iterations=0, device=device,
    )
    verts, indices, normals = ctx.extract(positions, radii=radii)
    test.assertIsNotNone(verts, "Anisotropic extraction produced no surface")
    test.assertGreater(verts.shape[0], 0)

    # Field should be close to 1.0 in the interior
    field_np = ctx.field.numpy()
    test.assertGreater(field_np.max(), 0.5, "Anisotropic field max should be significant")

    # G matrices should not all be identity (anisotropy should be active)
    G_np = ctx.anisotropy_matrices.numpy()
    off_diag = np.abs(G_np[:, 0, 1])  # sample one off-diagonal
    test.assertGreater(off_diag.max(), 1e-6, "G matrices should have off-diagonal entries (anisotropy)")


def _find_zero_crossings_1d(values, dx, origin=0.0):
    """Find zero crossing positions via linear interpolation along a 1D array."""
    crossings = []
    for i in range(len(values) - 1):
        a, b = values[i], values[i + 1]
        if a * b < 0:
            t = a / (a - b)
            crossings.append(origin + (i + t) * dx)
    return np.array(crossings)


def test_redistance_kernel(test, device):
    """Direct test of _redistance_step with a known perturbed sphere SDF."""
    N = 64
    dx = 0.05
    R = 1.0
    n_iters = 20

    sdf_exact, center = _make_sphere_sdf(N, dx, R)

    # Perturb: multiply by spatially varying factor to destroy |grad|=1
    coords = np.arange(N, dtype=np.float32)
    ii, jj, kk = np.meshgrid(coords, coords, coords, indexing="ij")
    factor = 1.0 + 0.5 * np.sin(2.0 * np.pi * ii / N) * np.sin(2.0 * np.pi * jj / N)
    sdf_perturbed = (sdf_exact * factor).astype(np.float32)

    # Pre-redistancing gradient error
    grad_mag_pre = _gradient_magnitude(sdf_perturbed, dx)
    err_pre = np.mean((grad_mag_pre - 1.0) ** 2)
    test.assertGreater(err_pre, 0.01, f"Perturbation should make gradient deviate from 1, got MSE={err_pre:.6f}")

    # --- Zero-level-set stability: a single iteration must not shift the
    # zero crossing by more than 0.05 * dx (smeared sign function test).
    src_1 = wp.array(sdf_perturbed, dtype=wp.float32, device=device)
    dst_1 = wp.empty((N, N, N), dtype=wp.float32, device=device)
    inv_dx = 1.0 / dx
    wp.launch(_redistance_step, dim=(N, N, N), inputs=[src_1, dst_1, inv_dx], device=device)
    result_1 = dst_1.numpy()

    # Compare zero crossings along an axis-aligned slice through the center
    mid = N // 2
    crossings_pre = _find_zero_crossings_1d(sdf_perturbed[:, mid, mid], dx)
    crossings_post = _find_zero_crossings_1d(result_1[:, mid, mid], dx)
    if len(crossings_pre) > 0 and len(crossings_pre) == len(crossings_post):
        max_shift = np.max(np.abs(crossings_post - crossings_pre))
        test.assertLess(
            max_shift,
            0.05 * dx,
            f"Single redistancing step shifted zero crossing by {max_shift / dx:.4f} voxels "
            f"(limit: 0.05 voxels)",
        )

    # --- Full redistancing run
    src = wp.array(sdf_perturbed, dtype=wp.float32, device=device)
    dst = wp.empty((N, N, N), dtype=wp.float32, device=device)
    for _ in range(n_iters):
        wp.launch(_redistance_step, dim=(N, N, N), inputs=[src, dst, inv_dx], device=device)
        src, dst = dst, src
    result = src.numpy()

    # No NaN or Inf
    test.assertFalse(np.any(np.isnan(result)), "Redistanced SDF contains NaN")
    test.assertFalse(np.any(np.isinf(result)), "Redistanced SDF contains Inf")

    # Values bounded (no divergence)
    test.assertLess(
        np.max(np.abs(result)),
        np.max(np.abs(sdf_perturbed)) * 2.0,
        "Redistanced values diverged",
    )

    # Sign preservation away from zero crossing
    far_from_surface = np.abs(sdf_exact) > 3.0 * dx
    signs_pre = np.sign(sdf_perturbed[far_from_surface])
    signs_post = np.sign(result[far_from_surface])
    sign_changes = np.sum(signs_pre != signs_post)
    test.assertEqual(sign_changes, 0, f"Sign changed at {sign_changes} voxels far from the surface")

    # Gradient magnitude improvement
    grad_mag_post = _gradient_magnitude(result, dx)
    err_post = np.mean((grad_mag_post - 1.0) ** 2)
    test.assertLess(
        err_post,
        err_pre,
        f"Gradient error should decrease after redistancing: pre={err_pre:.6f}, post={err_post:.6f}",
    )

    # Near the surface (within 5*dx), gradient should be close to 1
    sdf_exact_interior = sdf_exact[1:-1, 1:-1, 1:-1]
    near_surface = np.abs(sdf_exact_interior) < 5.0 * dx
    if np.sum(near_surface) > 100:
        grad_near_post = grad_mag_post[near_surface]
        near_err = np.mean((grad_near_post - 1.0) ** 2)
        test.assertLess(
            near_err,
            0.1,
            f"Gradient near surface should be close to 1: MSE={near_err:.6f}, "
            f"mean |grad|={np.mean(grad_near_post):.4f}",
        )


def test_redistance_via_api(test, device):
    """Integration test: redistancing through ParticleSurface.extract()."""
    positions, radii = _make_sphere_particles(device=device)

    ctx_no_redist = ParticleSurface(
        voxel_size=0.1, kernel_radius=0.3, redistance_iterations=0,
        field_smooth_iterations=1, mesh_smooth_iterations=0, device=device,
    )
    ctx_redist = ParticleSurface(
        voxel_size=0.1, kernel_radius=0.3, redistance_iterations=10,
        field_smooth_iterations=1, mesh_smooth_iterations=0, device=device,
    )

    verts0, indices0, _ = ctx_no_redist.extract(positions, radii=radii)
    verts1, indices1, _ = ctx_redist.extract(positions, radii=radii)

    # Both should produce valid meshes
    test.assertIsNotNone(verts0, "Non-redistanced extraction produced no mesh")
    test.assertIsNotNone(verts1, "Redistanced extraction produced no mesh")

    # No NaN/Inf in redistanced field
    field_redist = ctx_redist.field.numpy()
    test.assertFalse(np.any(np.isnan(field_redist)), "Redistanced field contains NaN")
    test.assertFalse(np.any(np.isinf(field_redist)), "Redistanced field contains Inf")

    # Gradient quality comparison
    dx = ctx_no_redist.voxel_size
    field_no_redist = ctx_no_redist.field.numpy()
    grad_no = _gradient_magnitude(field_no_redist, dx)
    grad_re = _gradient_magnitude(field_redist, dx)

    # Only compare in a band near the surface where gradient is meaningful
    sdf_interior = field_no_redist[1:-1, 1:-1, 1:-1]
    near_surface = np.abs(sdf_interior) < 5.0 * dx
    if np.sum(near_surface) > 100:
        err_no = np.mean((grad_no[near_surface] - 1.0) ** 2)
        err_re = np.mean((grad_re[near_surface] - 1.0) ** 2)
        test.assertLess(
            err_re,
            err_no,
            f"Redistancing should improve gradient near surface: no_redist={err_no:.6f}, redist={err_re:.6f}",
        )


class TestParticleSurface(unittest.TestCase):
    pass


devices = wp.get_cuda_devices()

add_function_test(TestParticleSurface, "test_one_shot", test_one_shot, devices=devices)
add_function_test(TestParticleSurface, "test_reusable_context", test_reusable_context, devices=devices)
add_function_test(TestParticleSurface, "test_mesh_smoothing", test_mesh_smoothing, devices=devices)
add_function_test(TestParticleSurface, "test_empty_particles", test_empty_particles, devices=devices)
# fem_field test uses Grid3D which doesn't support multi-GPU partitioning;
# run only on the default CUDA device.
add_function_test(TestParticleSurface, "test_fem_field", test_fem_field, devices=devices[:1])
add_function_test(TestParticleSurface, "test_anisotropic", test_anisotropic, devices=devices)
add_function_test(TestParticleSurface, "test_redistance_kernel", test_redistance_kernel, devices=devices)
add_function_test(TestParticleSurface, "test_redistance_via_api", test_redistance_via_api, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
