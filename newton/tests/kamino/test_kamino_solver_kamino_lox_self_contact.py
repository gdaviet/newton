# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the deformable self-contacts of the LOX Kamino dynamics backend.

The tests detect and load the vertex-triangle and edge-edge self-contacts of cloth and
tetrahedral surfaces, filter them with the normal cones of their surface features, project
them together with rigid-soft contacts, and truncate the particle motion with the
penetration-free step limiter.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.solvers.lox.projection_kernels import PARTICLE_COEFFICIENT_TOLERANCE
from newton._src.solvers.kamino._src.solvers.lox.types import ParticleContactStatus
from newton.solvers import SolverKamino
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.test_kamino_solver_kamino_lox_deformables import (
    _build_contact_model,
    _lox,
    _make_lox_config,
    _make_particle_contact,
    _world_time_steps,
)

###
# Helpers
###


def _build_cloth_mesh_model(
    device: wp.DeviceLike,
    vertices: list[tuple[float, float, float]],
    *,
    ground_plane: bool = False,
) -> newton.Model:
    """Build two disconnected cloth triangles from six vertices without gravity."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(*vertex) for vertex in vertices],
        indices=[0, 1, 2, 3, 4, 5],
        density=1.0,
        tri_ke=100.0,
        tri_ka=80.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
    )
    if ground_plane:
        builder.add_shape_plane()
    builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _build_self_contact_model(device: wp.DeviceLike) -> newton.Model:
    """Build two disconnected cloth triangles within self-contact range above a static plane."""
    model = _build_cloth_mesh_model(
        device,
        [
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.05),
            (0.8, 0.8, 0.05),
            (-0.8, 0.8, 0.05),
        ],
        ground_plane=True,
    )
    model.soft_contact_mu = 0.5
    model.soft_contact_restitution = 0.0
    return model


def _build_boundary_corner_self_contact_model(device: wp.DeviceLike) -> newton.Model:
    """Build two coplanar triangles whose closest pair is two boundary corners."""
    return _build_cloth_mesh_model(
        device,
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-0.05, -0.05, 0.0),
            (-1.0, -0.05, 0.0),
            (-0.05, -1.0, 0.0),
        ],
    )


def _build_close_normal_cone_self_contact_model(device: wp.DeviceLike, separation: float) -> newton.Model:
    """Build a vertex-triangle pair whose exact direction leaves the normal cone of the vertex."""
    return _build_cloth_mesh_model(
        device,
        [
            (separation, 0.0, 0.0),
            (separation, 1.0, 0.0),
            (separation, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 0.0, -1.0),
        ],
    )


def _build_edge_crossing_model(device: wp.DeviceLike) -> newton.Model:
    """Build two disconnected triangles with nearby crossing surface edges."""
    return _build_cloth_mesh_model(
        device,
        [
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, -0.5, 0.05),
            (0.0, 0.5, 0.05),
            (1.0, 0.0, 0.05),
        ],
    )


def _build_tetrahedral_surface_without_edges(device: wp.DeviceLike) -> newton.Model:
    """Build one tetrahedral grid with collision triangles but no bending edges."""
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=1.0,
        cell_y=1.0,
        cell_z=1.0,
        density=1.0,
        k_mu=100.0,
        k_lambda=80.0,
        k_damp=0.0,
        add_surface_mesh_edges=False,
    )
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _begin_deformable_time_step(solver: SolverKamino, state: newton.State, time_step: float) -> wp.array[wp.float32]:
    """Assemble the deformable system, detect the self-contact candidates, and weight the particles.

    Returns:
        The time step of each solve world.
    """
    deformables = _lox(solver).deformables
    world_time_step = _world_time_steps(solver, time_step)
    deformables.begin_time_step(state, world_time_step)
    deformables.build_consensus_weights()
    return world_time_step


def _set_self_contact_candidates(
    solver: SolverKamino,
    *,
    vertex_triangle: tuple[int, int] | None = None,
    edge_edge: tuple[int, int] | None = None,
) -> None:
    """Replace the detected self-contact candidates with at most one vertex-triangle and one edge-edge pair."""
    detector = _lox(solver).deformables.self_contacts.detector
    vertex_records = np.full(detector.vertex_colliding_triangles.shape, -1, dtype=np.int32)
    edge_records = np.full(detector.edge_colliding_edges.shape, -1, dtype=np.int32)
    if vertex_triangle is not None:
        vertex_records[:2] = vertex_triangle
    if edge_edge is not None:
        edge_records[:2] = edge_edge
    detector.vertex_colliding_triangles.assign(vertex_records)
    detector.edge_colliding_edges.assign(edge_records)


def _load_deformable_contacts(solver: SolverKamino, state: newton.State, time_step: wp.array[wp.float32]) -> None:
    """Load the particle contact rows of the frozen self-contact candidates."""
    lox = _lox(solver)
    lox.deformables.load_contacts(
        None,
        state,
        time_step,
        1.0,
        0.0,
        lox.problem.data,
        lox.problem.system.data.body_vector_index,
    )


def _truncate_velocity(solver: SolverKamino, velocity: np.ndarray, time_step: wp.array[wp.float32]) -> np.ndarray:
    """Truncate Newton-ordered particle velocities with the penetration-free limiter and return them in Newton order."""
    lox = _lox(solver)
    deformables = lox.deformables
    topology = deformables.topology
    projected_velocity = lox.data.particles.projected_velocity
    projected_velocity.assign(velocity[topology.packed_to_newton.numpy()])
    deformables.penetration_limiter.truncate(projected_velocity, lox.data.state.world_active, time_step)
    return projected_velocity.numpy()[topology.newton_to_packed.numpy()]


###
# Tests
###


class TestSolverKaminoLOXSelfContact(unittest.TestCase):
    """Detect, filter, project, and limit the self-contacts of LOX deformable surfaces."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def test_step_pure_cloth_self_contact(self):
        """Resolve filtered four-node contacts between cloth surface primitives."""
        model = _build_self_contact_model(self.default_device)
        state_in = model.state()
        state_out = model.state()
        solver = SolverKamino(
            model,
            config=_make_lox_config(
                deformable_enable_self_contact=True,
                deformable_self_contact_margin=0.1,
                deformable_self_contact_gap=0.05,
            ),
        )

        solver.step(state_in, state_out, None, None, 0.01)

        rows = _lox(solver).deformables.contact_rows
        active = rows.status.numpy() == ParticleContactStatus.ACTIVE
        self.assertGreater(int(np.count_nonzero(active)), 0)
        coefficients = rows.coefficient.numpy()[active]
        self.assertTrue(np.any(coefficients > 0.0))
        self.assertTrue(np.any(coefficients < 0.0))
        self.assertGreater(float(np.linalg.norm(state_out.particle_qd.numpy())), 0.0)

    def test_deformable_projection_uses_weighted_mass_split_majorizer(self):
        """Weight the shared Jacobi and APGD majorizer by the contact coefficients."""
        model = _build_self_contact_model(self.default_device)
        solver = SolverKamino(
            model,
            config=_make_lox_config(
                max_iterations=1,
                projection_sweeps_per_iteration=1,
                gauss_seidel_max_colors=1,
                deformable_enable_self_contact=True,
                deformable_self_contact_margin=0.1,
                deformable_self_contact_gap=0.05,
            ),
        )

        solver.step(model.state(), model.state(), None, None, 0.01)

        lox = _lox(solver)
        rows = lox.deformables.contact_rows
        active = rows.status.numpy() == ParticleContactStatus.ACTIVE
        self.assertGreater(int(np.count_nonzero(active)), 0)
        particles = rows.particle.numpy()
        coefficients = rows.coefficient.numpy()
        inverse_weight = lox.deformables.system_data.inverse_weight.numpy()
        particle_weight_sum = np.zeros_like(inverse_weight)
        particle_multiplicity = np.zeros_like(inverse_weight)
        for slot in range(particles.shape[1]):
            particle = particles[:, slot]
            significant = active & (particle >= 0) & (np.abs(coefficients[:, slot]) > PARTICLE_COEFFICIENT_TOLERANCE)
            np.add.at(particle_weight_sum, particle[significant], np.abs(coefficients[significant, slot]))
            np.add.at(particle_multiplicity, particle[significant], 1.0)

        weighted = np.zeros(len(active), dtype=np.float32)
        equal = np.zeros(len(active), dtype=np.float32)
        for slot in range(particles.shape[1]):
            particle = particles[:, slot]
            significant = active & (particle >= 0) & (np.abs(coefficients[:, slot]) > PARTICLE_COEFFICIENT_TOLERANCE)
            local_particle = particle[significant]
            weighted[significant] += (
                np.abs(coefficients[significant, slot])
                * particle_weight_sum[local_particle]
                * inverse_weight[local_particle]
            )
            equal[significant] += (
                coefficients[significant, slot] ** 2
                * particle_multiplicity[local_particle]
                * inverse_weight[local_particle]
            )

        self.assertGreater(float(np.max(np.abs(weighted[active] - equal[active]))), 1.0e-4)
        # The self-contact rows have no body, so their local block is the isotropic particle term
        delassus = lox._projection.data.particle_delassus.numpy()[active]
        np.testing.assert_allclose(
            delassus, weighted[active, None, None] * np.eye(3, dtype=np.float32), rtol=2.0e-6, atol=2.0e-6
        )

    def test_step_self_contact_with_alternative_projections(self):
        """Resolve deformable self-contact with colored Gauss--Seidel and APGD."""
        for max_colors, acceleration in ((4, "none"), (1, "apgd")):
            with self.subTest(max_colors=max_colors, acceleration=acceleration):
                model = _build_self_contact_model(self.default_device)
                state_out = model.state()
                solver = SolverKamino(
                    model,
                    config=_make_lox_config(
                        max_iterations=40,
                        projection_sweeps_per_iteration=8,
                        projection_acceleration=acceleration,
                        gauss_seidel_max_colors=max_colors,
                        deformable_enable_self_contact=True,
                        deformable_self_contact_margin=0.1,
                        deformable_self_contact_gap=0.05,
                    ),
                )

                solver.step(model.state(), state_out, None, None, 0.01)

                rows = _lox(solver).deformables.contact_rows
                np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])
                self.assertGreater(int(np.count_nonzero(rows.status.numpy() == ParticleContactStatus.ACTIVE)), 0)
                self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))

    def test_keep_boundary_corner_self_contact(self):
        """Keep an exterior closest-point pair between two cloth boundary corners."""
        model = _build_boundary_corner_self_contact_model(self.default_device)
        state = model.state()
        solver = SolverKamino(
            model,
            config=_make_lox_config(
                deformable_enable_self_contact=True,
                deformable_enable_normal_cone_filtering=True,
                deformable_self_contact_margin=0.1,
                deformable_self_contact_gap=0.01,
                deformable_self_contact_topological_filter_threshold=0,
            ),
        )
        time_step = _begin_deformable_time_step(solver, state, 0.01)
        _set_self_contact_candidates(solver, vertex_triangle=(3, 0))

        _load_deformable_contacts(solver, state, time_step)

        rows = _lox(solver).deformables.contact_rows
        self.assertEqual(int(rows.status.numpy()[0]), ParticleContactStatus.ACTIVE)
        np.testing.assert_allclose(
            rows.frame.numpy()[0, :, 2],
            (-np.sqrt(0.5), -np.sqrt(0.5), 0.0),
            atol=1.0e-6,
        )

    def test_bypass_normal_cone_filter_for_close_self_contact(self):
        """Keep a close self-contact despite an unreliable cone-rejected direction."""
        threshold = 1.0e-4
        for separation, expected_status in (
            (0.5 * threshold, ParticleContactStatus.ACTIVE),
            (2.0 * threshold, ParticleContactStatus.UNUSED),
        ):
            with self.subTest(separation=separation):
                model = _build_close_normal_cone_self_contact_model(self.default_device, separation)
                state = model.state()
                solver = SolverKamino(
                    model,
                    config=_make_lox_config(
                        deformable_enable_self_contact=True,
                        deformable_enable_normal_cone_filtering=True,
                        deformable_normal_cone_filtering_min_distance=threshold,
                        deformable_self_contact_margin=0.01,
                        deformable_self_contact_gap=0.01,
                        deformable_self_contact_topological_filter_threshold=0,
                    ),
                )
                time_step = _begin_deformable_time_step(solver, state, 0.01)
                _set_self_contact_candidates(solver, vertex_triangle=(3, 0))

                _load_deformable_contacts(solver, state, time_step)

                rows = _lox(solver).deformables.contact_rows
                self.assertEqual(int(rows.status.numpy()[0]), expected_status)

    def test_penetration_free_contact_uses_cone_pruned_candidate(self):
        """Limit motion with a detected candidate that the normal-cone filter rejects as a contact row."""
        separation = 2.0e-4
        model = _build_close_normal_cone_self_contact_model(self.default_device, separation)
        state = model.state()
        solver = SolverKamino(
            model,
            config=_make_lox_config(
                max_iterations=1,
                deformable_enable_self_contact=True,
                deformable_enable_penetration_free_contact=True,
                deformable_enable_normal_cone_filtering=True,
                deformable_normal_cone_filtering_min_distance=1.0e-4,
                deformable_self_contact_margin=0.01,
                deformable_self_contact_gap=0.01,
                deformable_self_contact_topological_filter_threshold=0,
            ),
        )
        time_step = _begin_deformable_time_step(solver, state, 0.1)
        _set_self_contact_candidates(solver, vertex_triangle=(3, 0))

        _load_deformable_contacts(solver, state, time_step)
        rows = _lox(solver).deformables.contact_rows
        self.assertEqual(int(rows.status.numpy()[0]), ParticleContactStatus.UNUSED)

        input_velocity = 5.0e-3
        velocity = np.zeros((model.particle_count, 3), dtype=np.float32)
        velocity[3, 0] = input_velocity
        truncated = _truncate_velocity(solver, velocity, time_step)

        self.assertGreater(float(truncated[3, 0]), 0.0)
        self.assertLess(float(truncated[3, 0]), input_velocity)

    def test_penetration_free_contact_truncates_frozen_candidates(self):
        """Prevent a frozen vertex-triangle candidate from crossing without rerunning detection."""
        model = _build_self_contact_model(self.default_device)
        state = model.state()
        time_step = 0.1
        solver = SolverKamino(
            model,
            config=_make_lox_config(
                max_iterations=1,
                deformable_enable_penetration_free_contact=True,
                deformable_self_contact_margin=0.1,
                deformable_self_contact_gap=0.05,
                deformable_self_contact_topological_filter_threshold=0,
            ),
        )
        world_time_step = _begin_deformable_time_step(solver, state, time_step)
        detector = _lox(solver).deformables.self_contacts.detector
        vertex_records = detector.vertex_colliding_triangles.numpy().copy()
        edge_records = detector.edge_colliding_edges.numpy().copy()

        velocity = np.zeros((model.particle_count, 3), dtype=np.float32)
        velocity[3, 2] = -1.0
        truncated = _truncate_velocity(solver, velocity, world_time_step)

        endpoint = state.particle_q.numpy() + time_step * truncated
        self.assertGreater(endpoint[3, 2], 0.0)
        self.assertLess(abs(float(truncated[3, 2])), 1.0)

        separating_velocity = np.zeros_like(velocity)
        separating_velocity[3, 2] = 0.1
        separating_result = _truncate_velocity(solver, separating_velocity, world_time_step)
        np.testing.assert_allclose(separating_result, separating_velocity, rtol=0.0, atol=1.0e-7)
        np.testing.assert_array_equal(detector.vertex_colliding_triangles.numpy(), vertex_records)
        np.testing.assert_array_equal(detector.edge_colliding_edges.numpy(), edge_records)

    def test_penetration_free_contact_truncates_edge_crossing(self):
        """Prevent a frozen edge-edge candidate from crossing."""
        model = _build_edge_crossing_model(self.default_device)
        state = model.state()
        time_step = 0.1
        solver = SolverKamino(
            model,
            config=_make_lox_config(
                max_iterations=1,
                deformable_enable_penetration_free_contact=True,
                deformable_self_contact_margin=0.1,
                deformable_self_contact_gap=0.2,
                deformable_self_contact_topological_filter_threshold=0,
            ),
        )
        world_time_step = _begin_deformable_time_step(solver, state, time_step)

        surface_edges = _lox(solver).deformables.self_contacts.edge_indices.numpy()
        crossing_edges = []
        for endpoints in ({0, 1}, {3, 4}):
            matches = [edge for edge, row in enumerate(surface_edges) if {int(row[2]), int(row[3])} == endpoints]
            self.assertEqual(len(matches), 1)
            crossing_edges.append(matches[0])
        crossing_edges.sort()
        _set_self_contact_candidates(solver, edge_edge=tuple(crossing_edges))

        velocity = np.zeros((model.particle_count, 3), dtype=np.float32)
        velocity[3:6, 2] = -1.0
        truncated = _truncate_velocity(solver, velocity, world_time_step)

        endpoint = state.particle_q.numpy() + time_step * truncated
        self.assertGreater(float(np.min(endpoint[3:5, 2])), 0.0)
        self.assertLess(float(np.max(np.abs(truncated[3:5, 2]))), 1.0)

    def test_penetration_free_contact_derives_tetrahedral_surface_edges(self):
        """Derive collision edges for a tetrahedral boundary without bending stencils."""
        model = _build_tetrahedral_surface_without_edges(self.default_device)
        self.assertEqual(model.edge_count, 0)
        solver = SolverKamino(
            model,
            config=_make_lox_config(
                deformable_enable_penetration_free_contact=True,
                deformable_self_contact_margin=0.1,
                deformable_self_contact_gap=0.1,
            ),
        )
        _begin_deformable_time_step(solver, model.state(), 0.01)

        self_contacts = _lox(solver).deformables.self_contacts
        self.assertIsNotNone(_lox(solver).deformables.penetration_limiter)
        self.assertGreater(self_contacts.edge_indices.shape[0], 0)
        self.assertEqual(self_contacts.detector.model.edge_count, self_contacts.edge_indices.shape[0])

    def test_step_applies_penetration_free_isotropic_bound(self):
        """Apply the penetration-free displacement cap inside every LOX splitting iteration."""
        model, _, _ = _build_contact_model(device=self.default_device)
        state_in = model.state()
        state_out = model.state()
        velocity = np.full((model.particle_count, 3), (10.0, 0.0, 0.0), dtype=np.float32)
        state_in.particle_qd.assign(velocity)
        time_step = 0.1
        margin = 0.1
        gap = 0.1
        query_radius = margin + gap
        relaxation = 0.85
        solver = SolverKamino(
            model,
            config=_make_lox_config(
                max_iterations=2,
                deformable_enable_penetration_free_contact=True,
                deformable_self_contact_margin=margin,
                deformable_self_contact_gap=gap,
                deformable_penetration_free_contact_relaxation=relaxation,
            ),
        )

        solver.step(state_in, state_out, None, None, time_step)

        displacement = state_out.particle_q.numpy() - state_in.particle_q.numpy()
        displacement_norm = np.linalg.norm(displacement, axis=1)
        self.assertLessEqual(float(np.max(displacement_norm)), 0.5 * query_radius * relaxation + 1.0e-6)

    def test_step_combined_rigid_and_cloth_self_contact(self):
        """Resolve imported rigid-cloth and generated cloth self-contacts together."""
        model = _build_self_contact_model(self.default_device)
        state_in = model.state()
        state_out = model.state()
        contacts = _make_particle_contact(model, state_in, 0, gap=0.0)
        solver = SolverKamino(
            model,
            config=_make_lox_config(
                deformable_enable_self_contact=True,
                deformable_self_contact_margin=0.1,
                deformable_self_contact_gap=0.05,
            ),
        )

        solver.step(state_in, state_out, None, contacts, 0.01)

        rows = _lox(solver).deformables.contact_rows
        self.assertEqual(rows.rigid_capacity, 1)
        self.assertGreater(rows.self_capacity, 0)
        status = rows.status.numpy()
        self.assertEqual(int(status[0]), ParticleContactStatus.ACTIVE)
        self.assertGreater(int(np.count_nonzero(status[1:] == ParticleContactStatus.ACTIVE)), 0)
        self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
