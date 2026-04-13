# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp
import warp.fem as fem

from newton._src.solvers.implicit_mpm.particle_surface import ParticleSurface, extract_particle_surface
from newton.tests.unittest_utils import add_function_test


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


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
