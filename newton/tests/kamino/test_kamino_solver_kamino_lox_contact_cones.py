# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the normal-cone filter of the LOX deformable contacts.

The scenes run Newton's collision pipeline with full-surface rigid-soft contacts and one LOX
step, then check which rows LOX keeps and along which normal: a contact keeps its row when its
direction lies in the normal cones of both primitives, and a concave feature replaces the
closest-point direction with its face normal.
"""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.solvers.lox.types import ParticleContactStatus
from newton.solvers import SolverKamino
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.test_kamino_solver_kamino_lox_deformables import _lox, _make_lox_config

###
# Helpers
###


def _grid(cells: int, half: float, height) -> tuple[list[tuple[float, float, float]], list[int]]:
    """Triangulate a square cloth patch with the given height field."""
    vertices = []
    for j in range(cells + 1):
        for i in range(cells + 1):
            x = -half + 2.0 * half * i / cells
            y = -half + 2.0 * half * j / cells
            vertices.append((x, y, height(x, y)))
    indices = []
    for j in range(cells):
        for i in range(cells):
            v00 = j * (cells + 1) + i
            indices += [v00, v00 + 1, v00 + cells + 2, v00, v00 + cells + 2, v00 + cells + 1]
    return vertices, indices


def _convex_mesh(vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]) -> newton.Mesh:
    """Build a collider mesh of a convex solid with outward faces."""
    points = np.asarray(vertices, dtype=np.float32)
    center = points.mean(axis=0)
    oriented = []
    for a, b, c in faces:
        normal = np.cross(points[b] - points[a], points[c] - points[a])
        oriented.append((a, b, c) if normal @ (points[[a, b, c]].mean(axis=0) - center) > 0.0 else (a, c, b))
    return newton.Mesh(points, np.asarray(oriented, dtype=np.int32).reshape(-1))


def _downward_pyramid(apex: tuple[float, float, float]) -> newton.Mesh:
    """Square pyramid whose apex points down."""
    ax, ay, az = apex
    vertices = [apex] + [(ax + sx * 0.05, ay + sy * 0.05, az + 0.08) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return _convex_mesh(vertices, [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (1, 3, 2), (1, 4, 3)])


def _valley_block() -> newton.Mesh:
    """Closed prism whose top is a floor (z=0, x<0) meeting a 60 deg wall (x>0) at a concave crease along y."""
    rise = 0.2 * math.tan(math.radians(60.0))
    # Counter-clockwise section in (x, z); vertices 0..5 at y = -0.2 and 6..11 at y = +0.2
    section = [(-0.3, -0.1), (0.3, -0.1), (0.3, rise), (0.2, rise), (0.0, 0.0), (-0.3, 0.0)]
    vertices = [(x, y, z) for y in (-0.2, 0.2) for x, z in section]
    faces = []
    for a, b, c in ((0, 1, 4), (1, 2, 3), (1, 3, 4), (0, 4, 5)):
        faces += [(a, b, c), (6 + a, 6 + c, 6 + b)]
    for k in range(6):
        a, b = k, (k + 1) % 6
        faces += [(a, 6 + b, b), (a, 6 + a, 6 + b)]
    return newton.Mesh(np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32).reshape(-1))


def _run_scene(
    device,
    cloths,
    collider: newton.Mesh | None,
    self_contact: bool = False,
    gap: float = 0.03,
    soft: dict | None = None,
    self_contact_gap: float = 0.03,
):
    """Collide the cloth patches and an optional soft grid with a static collider mesh and run one filtered LOX step.

    Returns:
        The model, the Newton contacts, and the LOX deformable problem.
    """
    builder = newton.ModelBuilder()
    builder.begin_world()
    if soft is not None:
        builder.add_soft_grid(
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            density=100.0,
            k_mu=1.0e3,
            k_lambda=1.0e3,
            k_damp=0.0,
            particle_radius=0.001,
            **soft,
        )
    for vertices, indices in cloths:
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=[wp.vec3(*v) for v in vertices],
            indices=indices,
            density=1.0,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.004,
        )
    if collider is not None:
        builder.add_shape_mesh(-1, mesh=collider)
    builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_gap=gap, enable_rigid_soft_full_surface_contact=True
    )
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    solver = SolverKamino(
        model,
        config=_make_lox_config(
            max_iterations=1,
            deformable_enable_self_contact=self_contact,
            deformable_enable_normal_cone_filtering=True,
            deformable_enable_rigid_contact_normal_cone_filtering=True,
            deformable_self_contact_margin=0.005,
            deformable_self_contact_gap=self_contact_gap,
        ),
    )
    solver.step(state_in, state_out, None, contacts, 0.01)
    return model, contacts, _lox(solver).deformables


def _rigid_records(contacts, rows, particle: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the status, row normal, and rigid point of the vertex records of one particle."""
    count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
    indices = contacts.soft_contact_indices.numpy()[:count]
    vertex_records = np.nonzero((indices[:, 0] == particle) & (indices[:, 1] < 0))[0]
    status = rows.status.numpy()[vertex_records]
    normals = rows.frame.numpy()[vertex_records][:, :, 2]
    points = contacts.soft_contact_body_pos.numpy()[vertex_records]
    return status, normals, points


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(np.clip(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)))


###
# Tests
###


class TestSolverKaminoLOXContactCones(unittest.TestCase):
    """Filter the deformable contacts with the normal cones of both primitives."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def test_cloth_pit_keeps_face_normal(self):
        """Keep the contact of a concave cloth vertex with a downward apex along a cloth face normal."""
        pit = 12
        cloth = _grid(4, 0.1, lambda x, y: -0.02 if abs(x) < 1e-9 and abs(y) < 1e-9 else 0.0)
        _model, contacts, deformables = _run_scene(self.default_device, [cloth], _downward_pyramid((0.0, 0.0, -0.008)))

        status, normals, _points = _rigid_records(contacts, deformables.contact_rows, pit)
        self.assertGreater(len(status), 0)
        self.assertTrue(np.all(status == ParticleContactStatus.ACTIVE))
        # The pit walls slope at atan(0.02 / 0.05) from the horizontal
        wall_tilt = math.degrees(math.atan(0.4))
        for normal in normals:
            self.assertAlmostEqual(_angle(normal, np.array([0.0, 0.0, 1.0])), wall_tilt, delta=1.0)
            self.assertLess(normal[2], 0.0)

    def test_concave_collider_crease_keeps_wall_normal(self):
        """Keep a cloth boundary contact at a concave collider crease along the wall face normal."""
        cloth = _grid(2, 0.05, lambda x, y: 0.012)
        # Shift the patch so its boundary column ends 40 mm short of the crease
        vertices = [(x - 0.09, y, z) for x, y, z in cloth[0]]
        _model, contacts, deformables = _run_scene(
            self.default_device, [(vertices, cloth[1])], _valley_block(), gap=0.05
        )
        rows = deformables.contact_rows

        wall_normal = np.array([-math.sin(math.radians(60.0)), 0.0, math.cos(math.radians(60.0))])
        crease_rows = 0
        for particle in (2, 5, 8):  # boundary column at x = -0.04
            status, normals, points = _rigid_records(contacts, rows, particle)
            on_crease = np.abs(points[:, 0]) + np.abs(points[:, 2]) < 1.0e-5
            for row_status, normal in zip(status[on_crease], normals[on_crease], strict=True):
                self.assertEqual(int(row_status), ParticleContactStatus.ACTIVE)
                self.assertLess(_angle(normal, wall_normal), 1.0)
                crease_rows += 1
        self.assertGreater(crease_rows, 0)

    def test_saddle_vertex_with_rising_ridge_is_dropped(self):
        """Drop the vertex contact of a saddle point whose rising edges are ridges, and keep the ridge records."""
        saddle = 12
        cloth = _grid(4, 0.1, lambda x, y: 4.0 * (x * x - y * y))
        _model, contacts, deformables = _run_scene(self.default_device, [cloth], _downward_pyramid((0.0, 0.0, 0.012)))
        rows = deformables.contact_rows

        status, _normals, _points = _rigid_records(contacts, rows, saddle)
        self.assertGreater(len(status), 0)
        self.assertTrue(np.all(status == ParticleContactStatus.UNUSED))
        count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
        indices = contacts.soft_contact_indices.numpy()[:count]
        near_saddle = np.nonzero((indices == saddle).any(axis=1) & (indices[:, 1] >= 0))[0]
        self.assertTrue(np.any(rows.status.numpy()[near_saddle] == ParticleContactStatus.ACTIVE))

    def test_self_contact_on_concave_crease_edge_keeps_face_normal(self):
        """Keep an edge-edge self-contact on a concave crease edge along a crease face normal."""
        slope = 0.15
        lower = _grid(4, 0.1, lambda x, y: slope * abs(x))
        # The upper edge along x crosses 12 mm above the crease, midway between two crease vertices
        upper = ([(-0.03, 0.025, 0.012), (0.03, 0.025, 0.012), (0.0, 0.06, 0.012)], [0, 1, 2])
        model, _contacts, deformables = _run_scene(self.default_device, [lower, upper], None, self_contact=True)
        rows = deformables.contact_rows

        upper_edge = {25, 26}
        packed_to_newton = deformables.topology.packed_to_newton.numpy()
        particle = np.where(rows.particle.numpy() >= 0, packed_to_newton[rows.particle.numpy()], -1)
        coefficient = rows.coefficient.numpy()
        status = rows.status.numpy()
        normals = rows.frame.numpy()[:, :, 2]
        q = model.particle_q.numpy()
        crease_rows = 0
        for row in range(rows.rigid_capacity, rows.rigid_capacity + rows.self_capacity):
            if status[row] != ParticleContactStatus.ACTIVE:
                continue
            members = {
                int(p) for p, c in zip(particle[row], coefficient[row], strict=True) if p >= 0 and abs(c) > 1.0e-6
            }
            if not upper_edge <= members:
                continue
            others = members - upper_edge
            if len(others) == 2 and all(abs(q[p][0]) < 1.0e-9 for p in others):
                self.assertAlmostEqual(
                    _angle(normals[row], np.array([0.0, 0.0, 1.0])), math.degrees(math.atan(slope)), delta=1.0
                )
                crease_rows += 1
        self.assertGreater(crease_rows, 0)

    def test_collider_inside_soft_body_recovers_along_outward_normal(self):
        """Push a collider apex out of a soft body along the outward face normal, with a negative gap."""
        apex = np.array([0.0125, 0.006, 0.048])
        soft = {"pos": wp.vec3(-0.05, -0.05, 0.0), "dim_x": 4, "dim_y": 4, "dim_z": 2, "cell_x": 0.025,
                "cell_y": 0.025, "cell_z": 0.025}  # fmt: skip
        _model, contacts, deformables = _run_scene(self.default_device, [], _downward_pyramid(tuple(apex)), soft=soft)
        rows = deformables.contact_rows

        count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
        indices = contacts.soft_contact_indices.numpy()[:count]
        points = contacts.soft_contact_body_pos.numpy()[:count]
        # Face records of the soft top surface against the apex, which sits 2 mm under it
        apex_records = np.nonzero((np.linalg.norm(points - apex, axis=1) < 1.0e-5) & (indices[:, 2] >= 0))[0]
        self.assertGreater(len(apex_records), 0)
        status = rows.status.numpy()[apex_records]
        normals = rows.frame.numpy()[apex_records][:, :, 2]
        gaps = rows.gap.numpy()[apex_records]
        self.assertTrue(np.all(status == ParticleContactStatus.ACTIVE))
        for normal, gap in zip(normals, gaps, strict=True):
            # The normal points from the collider towards the soft surface, against the outward normal
            self.assertLess(_angle(normal, np.array([0.0, 0.0, -1.0])), 1.0)
            self.assertLess(normal[2], 0.0)
            self.assertLess(gap, 0.0)

    def test_soft_body_interior_pairs_are_not_contacts(self):
        """Leave no self-contact row between the opposite faces of a thin soft slab."""
        thickness = 0.02
        soft = {"pos": wp.vec3(-0.04, -0.04, 0.0), "dim_x": 4, "dim_y": 4, "dim_z": 1, "cell_x": 0.02,
                "cell_y": 0.02, "cell_z": thickness}  # fmt: skip
        model, _contacts, deformables = _run_scene(
            self.default_device, [], None, self_contact=True, soft=soft, self_contact_gap=0.02
        )
        rows = deformables.contact_rows
        self_contacts = deformables.self_contacts

        levels = np.round(model.particle_q.numpy()[:, 2] / thickness).astype(int)
        tri_indices = model.tri_indices.numpy()
        records = self_contacts.detector.vertex_colliding_triangles.numpy().reshape(-1, 2)
        status = rows.status.numpy()[rows.rigid_capacity : rows.rigid_capacity + self_contacts.vertex_capacity]
        across = 0
        for candidate, (vertex, triangle) in enumerate(records):
            if vertex < 0 or triangle < 0:
                continue
            triangle_levels = set(levels[tri_indices[triangle]].tolist())
            # A top vertex over a bottom triangle, or the reverse, faces across the interior
            if len(triangle_levels) == 1 and levels[vertex] not in triangle_levels:
                across += 1
                self.assertNotEqual(int(status[candidate]), ParticleContactStatus.ACTIVE)
        self.assertGreater(across, 0)


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
