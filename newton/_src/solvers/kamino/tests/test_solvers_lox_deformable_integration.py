# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for cloth in the SolverKamino LOX backend."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.solvers.lox import (
    LOX_STATUS_CONVERGED,
    LOX_STATUS_FAILED,
)
from newton._src.solvers.kamino.config import LOXSolverConfig
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.lox_test_utils import (
    build_contact_model as _build_contact_model,
)
from newton._src.solvers.kamino.tests.lox_test_utils import (
    make_contacts as _make_contacts,
)
from newton._src.solvers.kamino.tests.lox_test_utils import (
    surface_position_for_gap as _surface_position_for_gap,
)


def _make_lox_config(**kwargs) -> newton.solvers.SolverKamino.Config:
    """Create a LOX config with optional cloth-specific overrides."""
    return newton.solvers.SolverKamino.Config(
        dynamics_solver="lox",
        lox=LOXSolverConfig(**kwargs),
    )


def _world_time_steps(
    model: newton.Model,
    value: float,
) -> tuple[wp.array[wp.float32], wp.array[wp.float32]]:
    """Construct explicit uniform per-world time-step arrays."""
    return (
        wp.full(model.world_count, value, dtype=wp.float32, device=model.device),
        wp.full(model.world_count, 1.0 / value, dtype=wp.float32, device=model.device),
    )


def _make_particle_contact(
    model: newton.Model,
    state: newton.State,
    shape: int,
    *,
    gap: float,
) -> newton.Contacts:
    """Create one manual particle-plane contact at the requested effective gap."""
    normal = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    body_position = _surface_position_for_gap(
        model,
        state.particle_q.numpy(),
        (0, -1, -1),
        (1.0, 0.0, 0.0),
        shape,
        gap,
        normal,
    )
    return _make_contacts(
        device=model.device,
        capacity=1,
        records=[
            {
                "indices": (0, -1, -1),
                "barycentric": (1.0, 0.0, 0.0),
                "shape": shape,
                "body_position": body_position,
                "normal": normal,
            }
        ],
    )


def _build_full_surface_dynamic_contact_model(device: wp.DeviceLike) -> tuple[newton.Model, int]:
    """Build a dynamic box crossing a cloth face while missing every particle."""
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform_identity())
    builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
    builder.add_cloth_grid(
        pos=wp.vec3(-1.0, -1.0, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1,
        dim_y=1,
        cell_x=2.0,
        cell_y=2.0,
        mass=0.1,
        tri_ke=100.0,
        tri_ka=80.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=2.0,
        edge_kd=0.0,
        particle_radius=0.0,
    )
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    model.soft_contact_mu = 0.5
    model.soft_contact_restitution = 0.0
    return model, body


def _build_global_cloth_local_rigid_model(device: wp.DeviceLike) -> tuple[newton.Model, int, int]:
    """Build global and local cloth coupled to one local dynamic plane."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 0.5),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1,
        dim_y=1,
        cell_x=1.0,
        cell_y=1.0,
        mass=1.0,
        tri_ke=100.0,
        tri_ka=80.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=2.0,
        edge_kd=0.0,
    )
    builder.begin_world(gravity=(0.0, 0.0, 0.0))
    builder.add_cloth_grid(
        pos=wp.vec3(10.0, 0.0, 0.5),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1,
        dim_y=1,
        cell_x=1.0,
        cell_y=1.0,
        mass=1.0,
        tri_ke=100.0,
        tri_ka=80.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=2.0,
        edge_kd=0.0,
    )
    body = builder.add_body(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    shape = builder.add_shape_plane(body=body)
    builder.end_world()
    model = builder.finalize(device=device)
    model.soft_contact_mu = 0.5
    model.soft_contact_restitution = 0.0
    return model, shape, body


def _build_self_contact_model(device: wp.DeviceLike) -> newton.Model:
    """Build two disconnected cloth triangles within self-contact range."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[
            wp.vec3(-1.0, -1.0, 0.0),
            wp.vec3(1.0, -1.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            wp.vec3(0.0, 0.0, 0.05),
            wp.vec3(0.8, 0.8, 0.05),
            wp.vec3(-0.8, 0.8, 0.05),
        ],
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
    builder.add_shape_plane()
    builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    model.soft_contact_mu = 0.5
    model.soft_contact_restitution = 0.0
    return model


def _build_boundary_corner_self_contact_model(device: wp.DeviceLike) -> newton.Model:
    """Build two coplanar triangles whose closest pair is two boundary corners."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            wp.vec3(-0.05, -0.05, 0.0),
            wp.vec3(-1.0, -0.05, 0.0),
            wp.vec3(-0.05, -1.0, 0.0),
        ],
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
    builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _build_close_normal_cone_self_contact_model(device: wp.DeviceLike, separation: float) -> newton.Model:
    """Build a vertex-triangle pair whose exact direction fails one vertex cone."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[
            wp.vec3(separation, 0.0, 0.0),
            wp.vec3(separation, 1.0, 0.0),
            wp.vec3(separation, 0.0, 1.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, -1.0, 0.0),
            wp.vec3(1.0, 0.0, -1.0),
        ],
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
    builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model


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


def _build_edge_crossing_model(device: wp.DeviceLike) -> newton.Model:
    """Build two disconnected triangles with nearby crossing surface edges."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[
            wp.vec3(-1.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, -1.0, 0.0),
            wp.vec3(0.0, -0.5, 0.05),
            wp.vec3(0.0, 0.5, 0.05),
            wp.vec3(1.0, 0.0, 0.05),
        ],
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
    builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model


class TestLOXDeformableIntegration(unittest.TestCase):
    """Test public SolverKamino cloth plumbing and mixed-world status."""

    def setUp(self):
        """Select the configured Newton test device."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_step_hanging_stiff_cloth_preserves_pins(self):
        """Keep pinned nodes fixed while a stiff bent cloth responds finitely."""
        model, _, _ = _build_contact_model(device=self.device, fix_left=True)
        model.set_gravity((0.0, 0.0, -9.81))
        triangle_materials = model.tri_materials.numpy()
        triangle_materials[:, :2] *= 100.0
        model.tri_materials.assign(triangle_materials)
        bending_properties = model.edge_bending_properties.numpy()
        bending_properties[:, 0] = 1.0e3
        model.edge_bending_properties.assign(bending_properties)
        state_in = model.state()
        positions = state_in.particle_q.numpy()
        positions[-1, 2] += 0.1
        state_in.particle_q.assign(positions)
        active = (model.particle_flags.numpy() & 1) != 0
        velocities = state_in.particle_qd.numpy()
        velocities[~active] = np.array((0.5, -0.25, 0.75), dtype=np.float32)
        state_in.particle_qd.assign(velocities)
        for selective_weights in (False, True):
            with self.subTest(selective_weights=selective_weights):
                state_out = model.state()
                solver = newton.solvers.SolverKamino(
                    model,
                    config=_make_lox_config(selective_weights=selective_weights),
                )

                solver.step(state_in, state_out, None, None, 0.005)

                np.testing.assert_array_equal(
                    state_out.particle_q.numpy()[~active],
                    state_in.particle_q.numpy()[~active],
                )
                np.testing.assert_array_equal(state_out.particle_qd.numpy()[~active], 0.0)
                self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))
                self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))
                self.assertGreater(np.linalg.norm(state_out.particle_qd.numpy()[active]), 0.0)

    def test_step_static_contact_applies_isotropic_friction(self):
        """Project a penetrating sliding particle against a static plane."""
        model, shapes, _ = _build_contact_model(device=self.device)
        state_in = model.state()
        state_in.particle_qd.fill_((1.0, 0.0, -1.0))
        contacts = _make_particle_contact(model, state_in, shapes[0], gap=-0.01)
        state_out = model.state()
        solver = newton.solvers.SolverKamino(model, config=_make_lox_config(max_iterations=40))
        dt = 0.01

        solver.step(state_in, state_out, None, contacts, dt)

        velocity = state_out.particle_qd.numpy()[0]
        self.assertGreaterEqual(float(velocity[2]), -1.0e-5)
        self.assertLess(float(np.linalg.norm(velocity[:2])), 1.0)
        np.testing.assert_allclose(
            state_out.particle_q.numpy(),
            state_in.particle_q.numpy() + dt * state_out.particle_qd.numpy(),
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        lox = solver._solver_kamino._solver_fd
        self.assertEqual(int(lox.world_status.numpy()[0]), LOX_STATUS_CONVERGED)
        self.assertTrue(bool(lox.world_accepted.numpy()[0]))

    def test_step_static_contact_with_alternative_projections(self):
        """Advance pure deformable contact with every alternative projection."""
        variants = (("gauss_seidel", 1), ("gauss_seidel", 4), ("apgd", 0))
        for projection_method, max_colors in variants:
            with self.subTest(projection_method=projection_method, max_colors=max_colors):
                model, shapes, _ = _build_contact_model(device=self.device)
                state_in = model.state()
                state_in.particle_qd.fill_((0.5, 0.0, -1.0))
                contacts = _make_particle_contact(model, state_in, shapes[0], gap=-0.01)
                state_out = model.state()
                solver = newton.solvers.SolverKamino(
                    model,
                    config=_make_lox_config(
                        max_iterations=40,
                        projection_iterations=8,
                        projection_method=projection_method,
                        gauss_seidel_max_colors=max_colors,
                    ),
                )

                solver.step(state_in, state_out, None, contacts, 0.01)

                lox = solver._solver_kamino._solver_fd
                self.assertNotEqual(int(lox.world_status.numpy()[0]), LOX_STATUS_FAILED)
                self.assertTrue(bool(lox.world_accepted.numpy()[0]))
                self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))

    def test_first_apgd_step_matches_mixed_contact_jacobi(self):
        """Match one mixed-contact Jacobi sweep with the first APGD block step."""
        results = {}
        for projection_method in ("jacobi", "apgd"):
            model, shapes, _ = _build_contact_model(device=self.device, collider="dynamic")
            state_in = model.state()
            state_in.particle_qd.fill_((0.5, 0.25, -1.0))
            contacts = _make_particle_contact(model, state_in, shapes[0], gap=-0.01)
            state_out = model.state()
            solver = newton.solvers.SolverKamino(
                model,
                config=_make_lox_config(
                    max_iterations=1,
                    projection_iterations=1,
                    projection_method=projection_method,
                ),
            )

            solver.step(state_in, state_out, None, contacts, 0.01)

            contact_system = solver._solver_kamino._solver_fd.deformable_contacts
            results[projection_method] = (
                state_out.particle_qd.numpy(),
                state_out.body_qd.numpy(),
                contact_system.reaction.numpy(),
            )

        for apgd_value, jacobi_value in zip(results["apgd"], results["jacobi"], strict=True):
            np.testing.assert_allclose(apgd_value, jacobi_value, rtol=2.0e-5, atol=2.0e-6)

    def test_first_apgd_step_matches_static_contact_jacobi(self):
        """Match one pure-deformable Jacobi sweep with the first APGD step."""
        results = {}
        for projection_method in ("jacobi", "apgd"):
            model, shapes, _ = _build_contact_model(device=self.device)
            state_in = model.state()
            state_in.particle_qd.fill_((0.5, 0.25, -1.0))
            contacts = _make_particle_contact(model, state_in, shapes[0], gap=-0.01)
            state_out = model.state()
            solver = newton.solvers.SolverKamino(
                model,
                config=_make_lox_config(
                    max_iterations=1,
                    projection_iterations=1,
                    projection_method=projection_method,
                ),
            )

            solver.step(state_in, state_out, None, contacts, 0.01)

            contact_system = solver._solver_kamino._solver_fd.deformable_contacts
            results[projection_method] = (
                state_out.particle_qd.numpy(),
                contact_system.reaction.numpy(),
            )

        for apgd_value, jacobi_value in zip(results["apgd"], results["jacobi"], strict=True):
            np.testing.assert_allclose(apgd_value, jacobi_value, rtol=2.0e-5, atol=2.0e-6)

    def test_one_color_gauss_seidel_matches_mixed_contact_jacobi(self):
        """Dispatch one-color Gauss--Seidel through the existing Jacobi path."""
        results = {}
        variants = (("jacobi", 0), ("gauss_seidel", 1))
        for projection_method, max_colors in variants:
            model, shapes, _ = _build_contact_model(device=self.device, collider="dynamic")
            state_in = model.state()
            state_in.particle_qd.fill_((0.5, 0.25, -1.0))
            contacts = _make_particle_contact(model, state_in, shapes[0], gap=-0.01)
            state_out = model.state()
            solver = newton.solvers.SolverKamino(
                model,
                config=_make_lox_config(
                    max_iterations=1,
                    projection_iterations=1,
                    projection_method=projection_method,
                    gauss_seidel_max_colors=max_colors,
                ),
            )

            solver.step(state_in, state_out, None, contacts, 0.01)

            lox = solver._solver_kamino._solver_fd
            results[projection_method] = (
                state_out.particle_qd.numpy(),
                state_out.body_qd.numpy(),
                lox.deformable_contacts.reaction.numpy(),
                lox.rigid_adapter.contact_reaction.numpy(),
            )

        for colored_value, jacobi_value in zip(results["gauss_seidel"], results["jacobi"], strict=True):
            np.testing.assert_allclose(colored_value, jacobi_value, rtol=0.0, atol=0.0)

    def test_one_color_gauss_seidel_matches_static_contact_jacobi(self):
        """Match the colored and parallel specializations of the particle map."""
        results = {}
        for projection_method, max_colors in (("jacobi", 0), ("gauss_seidel", 1)):
            model, shapes, _ = _build_contact_model(device=self.device)
            state_in = model.state()
            state_in.particle_qd.fill_((0.5, 0.25, -1.0))
            contacts = _make_particle_contact(model, state_in, shapes[0], gap=-0.01)
            state_out = model.state()
            solver = newton.solvers.SolverKamino(
                model,
                config=_make_lox_config(
                    max_iterations=1,
                    projection_iterations=3,
                    projection_method=projection_method,
                    gauss_seidel_max_colors=max_colors,
                ),
            )

            solver.step(state_in, state_out, None, contacts, 0.01)

            contact_system = solver._solver_kamino._solver_fd.deformable_contacts
            results[projection_method] = (
                state_out.particle_qd.numpy(),
                contact_system.reaction.numpy(),
            )

        for colored_value, jacobi_value in zip(results["gauss_seidel"], results["jacobi"], strict=True):
            np.testing.assert_allclose(colored_value, jacobi_value, rtol=2.0e-5, atol=2.0e-6)

    def test_step_mixed_contact_with_alternative_projections(self):
        """Advance two-way rigid-deformable contact with every alternative projection."""
        variants = (("gauss_seidel", 1), ("gauss_seidel", 4), ("apgd", 0))
        for projection_method, max_colors in variants:
            with self.subTest(projection_method=projection_method, max_colors=max_colors):
                model, shapes, bodies = _build_contact_model(device=self.device, collider="dynamic")
                state_in = model.state()
                state_in.particle_qd.fill_((0.0, 0.0, -1.0))
                contacts = _make_particle_contact(model, state_in, shapes[0], gap=0.0)
                state_out = model.state()
                solver = newton.solvers.SolverKamino(
                    model,
                    config=_make_lox_config(
                        max_iterations=50,
                        projection_iterations=8,
                        projection_method=projection_method,
                        gauss_seidel_max_colors=max_colors,
                    ),
                )

                solver.step(state_in, state_out, None, contacts, 0.01)

                lox = solver._solver_kamino._solver_fd
                self.assertNotEqual(int(lox.world_status.numpy()[0]), LOX_STATUS_FAILED)
                self.assertTrue(bool(lox.world_accepted.numpy()[0]))
                self.assertGreater(float(state_out.particle_qd.numpy()[0, 2]), -1.0)
                self.assertLess(float(state_out.body_qd.numpy()[bodies[0], 2]), 0.0)

    def test_step_dynamic_rigid_contact_updates_both_endpoints(self):
        """Exchange contact momentum in one coupled rigid-cloth solve."""
        model, shapes, bodies = _build_contact_model(
            device=self.device,
            collider="dynamic",
            body_com=(0.0, 0.0, 0.25),
        )
        state_in = model.state()
        state_in.particle_qd.fill_((0.0, 0.0, -1.0))
        contacts = _make_particle_contact(model, state_in, shapes[0], gap=0.0)
        state_out = model.state()
        solver = newton.solvers.SolverKamino(
            model,
            config=_make_lox_config(max_iterations=50),
        )
        particle_mass = model.particle_mass.numpy()
        body_mass = float(model.body_mass.numpy()[bodies[0]])
        momentum_before = np.sum(particle_mass[:, None] * state_in.particle_qd.numpy(), axis=0)
        momentum_before += body_mass * state_in.body_qd.numpy()[bodies[0], :3]

        solver.step(state_in, state_out, None, contacts, 0.01)

        self.assertGreater(float(state_out.particle_qd.numpy()[0, 2]), -1.0)
        self.assertLess(float(state_out.body_qd.numpy()[bodies[0], 2]), 0.0)
        momentum_after = np.sum(particle_mass[:, None] * state_out.particle_qd.numpy(), axis=0)
        momentum_after += body_mass * state_out.body_qd.numpy()[bodies[0], :3]
        np.testing.assert_allclose(momentum_after, momentum_before, rtol=2.0e-3, atol=2.0e-3)

    def _check_step_in_place_matches_ping_pong_with_nonzero_body_com(self, *, capture: bool) -> None:
        """Compare in-place LOX stepping with a ping-pong reference."""
        model, shapes, _ = _build_contact_model(
            device=self.device,
            collider="dynamic",
            body_com=(0.1, -0.05, 0.25),
        )
        state_ping = model.state()
        state_ping.particle_qd.fill_((0.0, 0.0, -1.0))
        state_pong = model.state()
        state_in_place = model.state()
        state_in_place.particle_qd.fill_((0.0, 0.0, -1.0))
        contacts_ping = _make_particle_contact(model, state_ping, shapes[0], gap=0.0)
        contacts_in_place = _make_particle_contact(model, state_in_place, shapes[0], gap=0.0)
        solver_ping_pong = newton.solvers.SolverKamino(
            model,
            config=_make_lox_config(max_iterations=50),
        )
        solver_in_place = newton.solvers.SolverKamino(
            model,
            config=_make_lox_config(max_iterations=50),
        )

        graph = None
        if capture:
            solver_in_place.step(state_in_place, state_in_place, None, contacts_in_place, 0.01)
            solver_in_place.reset(state_in_place)
            state_in_place.particle_qd.fill_((0.0, 0.0, -1.0))
            with wp.ScopedCapture(device=self.device) as captured:
                solver_in_place.step(state_in_place, state_in_place, None, contacts_in_place, 0.01)
            graph = captured.graph

        for _ in range(3):
            solver_ping_pong.step(state_ping, state_pong, None, contacts_ping, 0.01)
            state_ping, state_pong = state_pong, state_ping
            if graph is None:
                solver_in_place.step(state_in_place, state_in_place, None, contacts_in_place, 0.01)
            else:
                wp.capture_launch(graph)

        self.assertGreater(float(np.linalg.norm(state_ping.body_lox_dual_impulse.numpy())), 0.0)
        self.assertGreater(float(np.linalg.norm(state_ping.particle_lox_dual_impulse.numpy())), 0.0)
        for name in ("body_q", "body_qd", "particle_q", "particle_qd"):
            np.testing.assert_allclose(
                getattr(state_in_place, name).numpy(),
                getattr(state_ping, name).numpy(),
                rtol=1.0e-6,
                atol=1.0e-6,
            )
        np.testing.assert_allclose(
            state_in_place.body_lox_dual_impulse.numpy(),
            state_ping.body_lox_dual_impulse.numpy(),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            state_in_place.particle_lox_dual_impulse.numpy(),
            state_ping.particle_lox_dual_impulse.numpy(),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    def test_step_in_place_matches_ping_pong_with_nonzero_body_com(self):
        """Match ping-pong LOX stepping in place with persistent warm starts."""
        self._check_step_in_place_matches_ping_pong_with_nonzero_body_com(capture=False)

    def test_capture_step_in_place_matches_ping_pong_with_nonzero_body_com(self):
        """Replay in-place LOX capture in origin coordinates with persistent warm starts."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        self._check_step_in_place_matches_ping_pong_with_nonzero_body_com(capture=True)

    def test_step_dynamic_rigid_contact_against_pinned_particle(self):
        """Resolve a dynamic contact whose cloth endpoint has zero inverse mass."""
        model, shapes, bodies = _build_contact_model(
            device=self.device,
            fix_left=True,
            collider="dynamic",
        )
        state_in = model.state()
        body_velocity = np.zeros((1, 6), dtype=np.float32)
        body_velocity[0, 2] = 1.0
        state_in.body_qd.assign(body_velocity)
        contacts = _make_particle_contact(model, state_in, shapes[0], gap=0.0)
        state_out = model.state()
        solver = newton.solvers.SolverKamino(model, config=_make_lox_config())

        solver.step(state_in, state_out, None, contacts, 0.01)

        lox = solver._solver_kamino._solver_fd
        self.assertTrue(bool(lox.world_accepted.numpy()[0]))
        self.assertGreater(float(lox.deformable_contacts.rigid_delassus.numpy()[0, 2, 2]), 0.0)
        np.testing.assert_array_equal(state_out.particle_qd.numpy()[0], 0.0)
        self.assertLess(float(state_out.body_qd.numpy()[bodies[0], 2]), 1.0)

    def test_step_multiworld_cloth_independently(self):
        """Advance multiple cloth worlds with independent external forces."""
        model, _, _ = _build_contact_model(device=self.device, world_count=2)
        state_in = model.state()
        forces = np.zeros((model.particle_count, 3), dtype=np.float32)
        particle_world = model.particle_world.numpy()
        forces[particle_world == 0, 0] = 1.0
        forces[particle_world == 1, 0] = -2.0
        state_in.particle_f.assign(forces)
        state_out = model.state()
        solver = newton.solvers.SolverKamino(model, config=_make_lox_config())

        solver.step(state_in, state_out, None, None, 0.01)

        velocity = state_out.particle_qd.numpy()
        self.assertGreater(float(np.mean(velocity[particle_world == 0, 0])), 0.0)
        self.assertLess(float(np.mean(velocity[particle_world == 1, 0])), 0.0)
        np.testing.assert_array_equal(
            solver._solver_kamino._solver_fd.world_status.numpy(),
            [LOX_STATUS_CONVERGED, LOX_STATUS_CONVERGED],
        )

    def test_step_multiworld_cloth_uses_each_world_time_step(self):
        """Advance deformable worlds with distinct device-side time steps."""
        model, _, _ = _build_contact_model(device=self.device, world_count=2)
        state_in = model.state()
        particle_world = model.particle_world.numpy()
        forces = np.zeros((model.particle_count, 3), dtype=np.float32)
        forces[:, 0] = 1.0
        state_in.particle_f.assign(forces)
        state_out = model.state()
        solver = newton.solvers.SolverKamino(model, config=_make_lox_config())
        time_step = np.asarray([0.01, 0.025], dtype=np.float32)
        solver._solver_kamino._model.time.dt.assign(time_step)
        solver._solver_kamino._model.time.inv_dt.assign(1.0 / time_step)

        solver.step(state_in, state_out, None, None, None)

        velocity = state_out.particle_qd.numpy()[:, 0]
        mean_velocity = np.asarray([np.mean(velocity[particle_world == world]) for world in range(model.world_count)])
        self.assertGreater(mean_velocity[0], 0.0)
        np.testing.assert_allclose(mean_velocity[1] / mean_velocity[0], time_step[1] / time_step[0], rtol=1.0e-4)

    def test_step_pure_cloth_with_collision_pipeline_contacts(self):
        """Consume collision-pipeline soft contacts without rigid-body state."""
        model, _, _ = _build_contact_model(device=self.device)
        state_in = model.state()
        state_out = model.state()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverKamino(model, config=_make_lox_config())

        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, 0.01)

        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))

    def test_step_ignores_fully_prescribed_rigid_contacts(self):
        """Advance cloth when an unrelated static box contacts the ground."""
        builder = newton.ModelBuilder()
        builder.begin_world()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 2.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            cell_x=1.0,
            cell_y=1.0,
            mass=1.0,
            tri_ke=100.0,
            tri_ka=80.0,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
            edge_ke=2.0,
            edge_kd=0.0,
        )
        body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.45), wp.quat_identity()))
        builder.add_shape_box(
            body,
            hx=0.5,
            hy=0.5,
            hz=0.5,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )
        builder.add_ground_plane()
        builder.end_world()
        model = builder.finalize(device=self.device)
        model.set_gravity((0.0, 0.0, -9.81))
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = pipeline.contacts()
        state_in = model.state()
        state_out = model.state()
        solver = newton.solvers.SolverKamino(model, config=_make_lox_config())

        pipeline.collide(state_in, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        solver.step(state_in, state_out, None, contacts, 0.01)

        lox = solver._solver_kamino._solver_fd
        self.assertTrue(bool(lox.world_accepted.numpy()[0]))
        self.assertTrue(np.all(state_out.particle_qd.numpy()[:, 2] < 0.0))

    def test_step_pure_cloth_self_contact(self):
        """Resolve filtered four-node contacts between cloth surface primitives."""
        model = _build_self_contact_model(self.device)
        state_in = model.state()
        state_out = model.state()
        solver = newton.solvers.SolverKamino(
            model,
            config=_make_lox_config(
                deformable_enable_self_contact=True,
                deformable_self_contact_margin=0.1,
                deformable_self_contact_gap=0.05,
            ),
        )

        solver.step(state_in, state_out, None, None, 0.01)

        contacts = solver._solver_kamino._solver_fd.deformable_contacts
        self.assertIsNotNone(contacts)
        valid = contacts.status.numpy() == 1
        self.assertGreater(int(np.count_nonzero(valid)), 0)
        coefficients = contacts.coefficients.numpy()[valid]
        self.assertTrue(np.any(coefficients > 0.0))
        self.assertTrue(np.any(coefficients < 0.0))
        self.assertGreater(float(np.linalg.norm(state_out.particle_qd.numpy())), 0.0)

    def test_deformable_projection_uses_weighted_mass_split_majorizer(self):
        """Weight the shared Jacobi/APGD majorizer by contact coefficients."""
        model = _build_self_contact_model(self.device)
        solver = newton.solvers.SolverKamino(
            model,
            config=_make_lox_config(
                max_iterations=1,
                projection_iterations=1,
                deformable_enable_self_contact=True,
                deformable_self_contact_margin=0.1,
                deformable_self_contact_gap=0.05,
            ),
        )

        solver.step(model.state(), model.state(), None, None, 0.01)

        lox = solver._solver_kamino._solver_fd
        contacts = lox.deformable_contacts
        valid = contacts.status.numpy() == 1
        self.assertGreater(int(np.count_nonzero(valid)), 0)
        particle_indices = contacts.particle_indices.numpy()
        coefficients = contacts.coefficients.numpy()
        inverse_weight = contacts.cloth_system.inverse_weight.numpy()
        particle_weight_sum = np.zeros_like(inverse_weight)
        particle_multiplicity = np.zeros_like(inverse_weight)
        for slot in range(particle_indices.shape[1]):
            particle = particle_indices[:, slot]
            active = (
                valid
                & (particle >= 0)
                & (np.abs(coefficients[:, slot]) > 1.0e-5)
                & (inverse_weight[np.maximum(particle, 0)] > 0.0)
            )
            np.add.at(particle_weight_sum, particle[active], np.abs(coefficients[active, slot]))
            np.add.at(particle_multiplicity, particle[active], 1.0)

        weighted = np.zeros(len(valid), dtype=np.float32)
        equal = np.zeros(len(valid), dtype=np.float32)
        for slot in range(particle_indices.shape[1]):
            particle = particle_indices[:, slot]
            active = valid & (particle >= 0) & (np.abs(coefficients[:, slot]) > 1.0e-5)
            local_particle = particle[active]
            weighted[active] += (
                np.abs(coefficients[active, slot])
                * particle_weight_sum[local_particle]
                * inverse_weight[local_particle]
            )
            equal[active] += (
                coefficients[active, slot] ** 2 * particle_multiplicity[local_particle] * inverse_weight[local_particle]
            )

        self.assertGreater(float(np.max(np.abs(weighted[valid] - equal[valid]))), 1.0e-4)
        np.testing.assert_allclose(contacts.delassus.numpy()[valid], weighted[valid], rtol=2.0e-6, atol=2.0e-6)

    def test_step_self_contact_with_alternative_projections(self):
        """Resolve deformable self-contact with Gauss-Seidel and APGD."""
        variants = (("gauss_seidel", 4), ("apgd", 0))
        for projection_method, max_colors in variants:
            with self.subTest(projection_method=projection_method, max_colors=max_colors):
                model = _build_self_contact_model(self.device)
                state_out = model.state()
                solver = newton.solvers.SolverKamino(
                    model,
                    config=_make_lox_config(
                        max_iterations=40,
                        projection_iterations=8,
                        projection_method=projection_method,
                        gauss_seidel_max_colors=max_colors,
                        deformable_enable_self_contact=True,
                        deformable_self_contact_margin=0.1,
                        deformable_self_contact_gap=0.05,
                    ),
                )

                solver.step(model.state(), state_out, None, None, 0.01)

                lox = solver._solver_kamino._solver_fd
                self.assertNotEqual(int(lox.world_status.numpy()[0]), LOX_STATUS_FAILED)
                self.assertTrue(bool(lox.world_accepted.numpy()[0]))
                self.assertGreater(int(np.count_nonzero(lox.deformable_contacts.status.numpy() == 1)), 0)
                self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))

    def test_keep_boundary_corner_self_contact(self):
        """Keep an exterior closest-point pair between two cloth boundary corners."""
        model = _build_boundary_corner_self_contact_model(self.device)
        state = model.state()
        solver = newton.solvers.SolverKamino(
            model,
            config=_make_lox_config(
                deformable_enable_self_contact=True,
                deformable_enable_normal_cone_filtering=True,
                deformable_self_contact_margin=0.1,
                deformable_self_contact_gap=0.01,
                deformable_self_contact_topological_filter_threshold=0,
            ),
        )
        lox = solver._solver_kamino._solver_fd
        lox.begin_deformable_time_step(state, None, *_world_time_steps(model, 0.01))

        detector = lox.deformable_self_contact_detector
        vertex_records = np.full(detector.detector.vertex_colliding_triangles.shape, -1, dtype=np.int32)
        edge_records = np.full(detector.detector.edge_colliding_edges.shape, -1, dtype=np.int32)
        vertex_records[:2] = (3, 0)
        detector.detector.vertex_colliding_triangles.assign(vertex_records)
        detector.detector.edge_colliding_edges.assign(edge_records)

        contacts = lox.deformable_contacts
        contacts.prepare(None, state, _world_time_steps(model, 0.01)[0], self_contact_detector=detector)

        self.assertEqual(int(contacts.status.numpy()[0]), 1)
        np.testing.assert_allclose(
            contacts.frame.numpy()[0, :, 2],
            (-np.sqrt(0.5), -np.sqrt(0.5), 0.0),
            atol=1.0e-6,
        )

    def test_bypass_normal_cone_filter_for_close_self_contact(self):
        """Keep a close self-contact despite an unreliable cone-rejected direction."""
        threshold = 1.0e-4
        for separation, expected_status in (
            (0.5 * threshold, 1),
            (2.0 * threshold, 0),
        ):
            with self.subTest(separation=separation):
                model = _build_close_normal_cone_self_contact_model(self.device, separation)
                state = model.state()
                solver = newton.solvers.SolverKamino(
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
                lox = solver._solver_kamino._solver_fd
                lox.begin_deformable_time_step(state, None, *_world_time_steps(model, 0.01))

                detector = lox.deformable_self_contact_detector
                vertex_records = np.full(detector.detector.vertex_colliding_triangles.shape, -1, dtype=np.int32)
                edge_records = np.full(detector.detector.edge_colliding_edges.shape, -1, dtype=np.int32)
                vertex_records[:2] = (3, 0)
                detector.detector.vertex_colliding_triangles.assign(vertex_records)
                detector.detector.edge_colliding_edges.assign(edge_records)

                contacts = lox.deformable_contacts
                contacts.prepare(None, state, _world_time_steps(model, 0.01)[0], self_contact_detector=detector)

                self.assertEqual(int(contacts.status.numpy()[0]), expected_status)

    def test_penetration_free_contact_uses_cone_pruned_candidate(self):
        """Limit motion with a detected candidate rejected by the normal-cone filter."""
        separation = 2.0e-4
        model = _build_close_normal_cone_self_contact_model(self.device, separation)
        state = model.state()
        time_step = 0.1
        solver = newton.solvers.SolverKamino(
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
        lox = solver._solver_kamino._solver_fd
        lox.begin_deformable_time_step(state, None, *_world_time_steps(model, time_step))

        detector = lox.deformable_self_contact_detector
        vertex_records = np.full(detector.detector.vertex_colliding_triangles.shape, -1, dtype=np.int32)
        edge_records = np.full(detector.detector.edge_colliding_edges.shape, -1, dtype=np.int32)
        vertex_records[:2] = (3, 0)
        detector.detector.vertex_colliding_triangles.assign(vertex_records)
        detector.detector.edge_colliding_edges.assign(edge_records)

        contacts = lox.deformable_contacts
        contacts.prepare(None, state, _world_time_steps(model, time_step)[0], self_contact_detector=detector)
        self.assertEqual(int(contacts.status.numpy()[0]), 0)

        input_velocity = 5.0e-3
        velocity = np.zeros((model.particle_count, 3), dtype=np.float32)
        velocity[3, 0] = input_velocity
        packed_to_newton = lox.deformable_system.topology.packed_to_newton.numpy()
        lox.deformable_splitting.projected_velocity.assign(velocity[packed_to_newton])
        lox.deformable_penetration_free_limiter.truncate(
            lox.deformable_splitting.projected_velocity,
            lox.deformable_splitting.world_active,
            _world_time_steps(model, time_step)[0],
        )

        newton_to_packed = lox.deformable_system.topology.newton_to_packed.numpy()
        truncated = lox.deformable_splitting.projected_velocity.numpy()[newton_to_packed]
        self.assertGreater(float(truncated[3, 0]), 0.0)
        self.assertLess(float(truncated[3, 0]), input_velocity)

    def test_penetration_free_contact_truncates_frozen_candidates(self):
        """Prevent a frozen vertex-triangle candidate from crossing without rerunning detection."""
        model = _build_self_contact_model(self.device)
        state = model.state()
        time_step = 0.1
        solver = newton.solvers.SolverKamino(
            model,
            config=_make_lox_config(
                max_iterations=1,
                deformable_enable_penetration_free_contact=True,
                deformable_self_contact_margin=0.1,
                deformable_self_contact_gap=0.05,
                deformable_self_contact_topological_filter_threshold=0,
            ),
        )
        lox = solver._solver_kamino._solver_fd
        lox.begin_deformable_time_step(state, None, *_world_time_steps(model, time_step))
        limiter = lox.deformable_penetration_free_limiter
        detector = lox.deformable_self_contact_detector.detector
        vertex_records = detector.vertex_colliding_triangles.numpy().copy()
        edge_records = detector.edge_colliding_edges.numpy().copy()

        velocity = np.zeros((model.particle_count, 3), dtype=np.float32)
        velocity[3, 2] = -1.0
        packed_to_newton = lox.deformable_system.topology.packed_to_newton.numpy()
        lox.deformable_splitting.projected_velocity.assign(velocity[packed_to_newton])
        limiter.truncate(
            lox.deformable_splitting.projected_velocity,
            lox.deformable_splitting.world_active,
            _world_time_steps(model, time_step)[0],
        )

        newton_to_packed = lox.deformable_system.topology.newton_to_packed.numpy()
        truncated = lox.deformable_splitting.projected_velocity.numpy()[newton_to_packed]
        endpoint = state.particle_q.numpy() + time_step * truncated
        self.assertGreater(endpoint[3, 2], 0.0)
        self.assertLess(abs(float(truncated[3, 2])), 1.0)

        separating_velocity = np.zeros_like(velocity)
        separating_velocity[3, 2] = 0.1
        lox.deformable_splitting.projected_velocity.assign(separating_velocity[packed_to_newton])
        limiter.truncate(
            lox.deformable_splitting.projected_velocity,
            lox.deformable_splitting.world_active,
            _world_time_steps(model, time_step)[0],
        )
        separating_result = lox.deformable_splitting.projected_velocity.numpy()[newton_to_packed]
        np.testing.assert_allclose(separating_result, separating_velocity, rtol=0.0, atol=1.0e-7)
        np.testing.assert_array_equal(detector.vertex_colliding_triangles.numpy(), vertex_records)
        np.testing.assert_array_equal(detector.edge_colliding_edges.numpy(), edge_records)

    def test_penetration_free_contact_truncates_edge_crossing(self):
        """Prevent a frozen edge-edge candidate from crossing."""
        model = _build_edge_crossing_model(self.device)
        state = model.state()
        time_step = 0.1
        solver = newton.solvers.SolverKamino(
            model,
            config=_make_lox_config(
                max_iterations=1,
                deformable_enable_penetration_free_contact=True,
                deformable_self_contact_margin=0.1,
                deformable_self_contact_gap=0.2,
                deformable_self_contact_topological_filter_threshold=0,
            ),
        )
        lox = solver._solver_kamino._solver_fd
        lox.begin_deformable_time_step(state, None, *_world_time_steps(model, time_step))
        limiter = lox.deformable_penetration_free_limiter
        detector = lox.deformable_self_contact_detector

        surface_edges = detector.edge_indices.numpy()
        crossing_edges = []
        for endpoints in ({0, 1}, {3, 4}):
            matches = [edge for edge, row in enumerate(surface_edges) if {int(row[2]), int(row[3])} == endpoints]
            self.assertEqual(len(matches), 1)
            crossing_edges.append(matches[0])
        crossing_edges.sort()

        vertex_records = np.full(detector.detector.vertex_colliding_triangles.shape, -1, dtype=np.int32)
        edge_records = np.full(detector.detector.edge_colliding_edges.shape, -1, dtype=np.int32)
        edge_records[:2] = crossing_edges
        detector.detector.vertex_colliding_triangles.assign(vertex_records)
        detector.detector.edge_colliding_edges.assign(edge_records)

        velocity = np.zeros((model.particle_count, 3), dtype=np.float32)
        velocity[3:6, 2] = -1.0
        packed_to_newton = lox.deformable_system.topology.packed_to_newton.numpy()
        lox.deformable_splitting.projected_velocity.assign(velocity[packed_to_newton])
        limiter.truncate(
            lox.deformable_splitting.projected_velocity,
            lox.deformable_splitting.world_active,
            _world_time_steps(model, time_step)[0],
        )

        newton_to_packed = lox.deformable_system.topology.newton_to_packed.numpy()
        truncated = lox.deformable_splitting.projected_velocity.numpy()[newton_to_packed]
        endpoint = state.particle_q.numpy() + time_step * truncated
        self.assertGreater(float(np.min(endpoint[3:5, 2])), 0.0)
        self.assertLess(float(np.max(np.abs(truncated[3:5, 2]))), 1.0)

    def test_penetration_free_contact_derives_tetrahedral_surface_edges(self):
        """Derive collision edges for a tetrahedral boundary without bending stencils."""
        model = _build_tetrahedral_surface_without_edges(self.device)
        self.assertEqual(model.edge_count, 0)
        solver = newton.solvers.SolverKamino(
            model,
            config=_make_lox_config(
                deformable_enable_penetration_free_contact=True,
                deformable_self_contact_margin=0.1,
                deformable_self_contact_gap=0.1,
            ),
        )
        lox = solver._solver_kamino._solver_fd
        lox.begin_deformable_time_step(model.state(), None, *_world_time_steps(model, 0.01))

        detector = lox.deformable_self_contact_detector
        self.assertGreater(detector.edge_indices.shape[0], 0)
        self.assertEqual(detector.detector.model.edge_count, detector.edge_indices.shape[0])

    def test_step_applies_penetration_free_isotropic_bound(self):
        """Apply the penetration-free displacement cap inside every LOX splitting iteration."""
        model, _, _ = _build_contact_model(device=self.device)
        state_in = model.state()
        state_out = model.state()
        velocity = np.full((model.particle_count, 3), (10.0, 0.0, 0.0), dtype=np.float32)
        state_in.particle_qd.assign(velocity)
        time_step = 0.1
        margin = 0.1
        gap = 0.1
        query_radius = margin + gap
        relaxation = 0.85
        solver = newton.solvers.SolverKamino(
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
        model = _build_self_contact_model(self.device)
        state_in = model.state()
        state_out = model.state()
        contacts = _make_particle_contact(model, state_in, 0, gap=0.0)
        solver = newton.solvers.SolverKamino(
            model,
            config=_make_lox_config(
                deformable_enable_self_contact=True,
                deformable_self_contact_margin=0.1,
                deformable_self_contact_gap=0.05,
            ),
        )

        solver.step(state_in, state_out, None, contacts, 0.01)

        contact_system = solver._solver_kamino._solver_fd.deformable_contacts
        self.assertEqual(contact_system.rigid_contact_capacity, 1)
        self.assertGreater(contact_system.self_contact_capacity, 0)
        status = contact_system.status.numpy()
        self.assertEqual(int(status[0]), 1)
        self.assertGreater(int(np.count_nonzero(status[1:] == 1)), 0)
        self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))

    def test_step_full_surface_contact_updates_cloth_and_rigid_body(self):
        """Resolve a generated face contact that the particle pass misses."""
        model, body = _build_full_surface_dynamic_contact_model(self.device)
        state_in = model.state()

        particle_pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            soft_contact_margin=0.1,
        )
        particle_contacts = particle_pipeline.contacts()
        particle_pipeline.collide(state_in, particle_contacts)
        self.assertEqual(int(particle_contacts.soft_contact_count.numpy()[0]), 0)

        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            soft_contact_margin=0.1,
            enable_rigid_soft_full_surface_contact=True,
        )
        contacts = pipeline.contacts()
        pipeline.collide(state_in, contacts)
        contact_count = int(contacts.soft_contact_count.numpy()[0])
        contact_indices = contacts.soft_contact_indices.numpy()[:contact_count]
        self.assertGreater(contact_count, 0)
        self.assertTrue(np.all(contact_indices[:, 1] >= 0))

        state_out = model.state()
        solver = newton.solvers.SolverKamino(model, config=_make_lox_config(max_iterations=50))
        solver.step(state_in, state_out, None, contacts, 0.01)

        lox = solver._solver_kamino._solver_fd
        self.assertTrue(bool(lox.world_accepted.numpy()[0]))
        self.assertTrue(np.any(lox.deformable_contacts.coefficients.numpy()[:contact_count, 1:] > 0.0))
        self.assertTrue(np.all(lox.deformable_contacts.body.numpy()[:contact_count] == body))
        self.assertGreater(float(np.linalg.norm(state_out.particle_qd.numpy())), 0.0)
        self.assertGreater(float(np.linalg.norm(state_out.body_qd.numpy()[body])), 0.0)

    def test_reset_pure_cloth_state_and_warm_start(self):
        """Reset Newton-owned particle state and clear pure-cloth LOX warm starts."""
        model, _, _ = _build_contact_model(device=self.device)
        model.set_gravity((0.0, 0.0, -9.81))
        state = model.state()
        state_out = model.state()
        solver = newton.solvers.SolverKamino(model, config=_make_lox_config())
        solver.step(state, state_out, None, None, 0.01)
        state.assign(state_out)
        state.particle_f.fill_((1.0, 2.0, 3.0))

        solver.reset(state)

        np.testing.assert_array_equal(state.particle_q.numpy(), model.particle_q.numpy())
        np.testing.assert_array_equal(state.particle_qd.numpy(), model.particle_qd.numpy())
        np.testing.assert_array_equal(state.particle_f.numpy(), 0.0)
        np.testing.assert_array_equal(state.particle_lox_dual_impulse.numpy(), 0.0)
        lox = solver._solver_kamino._solver_fd
        np.testing.assert_array_equal(lox.deformable_splitting.dual_impulse.numpy(), 0.0)

    def test_reset_shared_global_and_local_solve_world(self):
        """Reset source state selectively but invalidate its complete LOX partition."""
        for reset_global in (False, True):
            with self.subTest(reset_global=reset_global):
                model, _, body = _build_global_cloth_local_rigid_model(self.device)
                solver = newton.solvers.SolverKamino(model, config=_make_lox_config())
                state = model.state()

                body_q = state.body_q.numpy()
                body_q[body, 0] += 2.0
                state.body_q.assign(body_q)
                state.body_qd.fill_(1.0)
                particle_q = state.particle_q.numpy()
                particle_q[:, 2] += 2.0
                state.particle_q.assign(particle_q)
                state.particle_qd.fill_(1.0)
                state.particle_f.fill_(1.0)
                state.body_lox_dual_impulse = wp.full(
                    model.body_count,
                    1.0,
                    dtype=wp.spatial_vector,
                    device=self.device,
                )
                state.particle_lox_dual_impulse = wp.full(
                    model.particle_count,
                    1.0,
                    dtype=wp.vec3,
                    device=self.device,
                )

                lox = solver._solver_kamino.solver_fd
                lox.splitting.splitting_dual_impulse.fill_(1.0)
                lox.deformable_splitting.dual_impulse.fill_(1.0)
                world_mask = wp.array([not reset_global, reset_global], dtype=wp.bool, device=self.device)
                success_mask = wp.empty(2, dtype=wp.bool, device=self.device)

                solver.reset(state, world_mask=world_mask, success_mask=success_mask)

                source_world = model.particle_world.numpy()
                selected_particles = source_world == (-1 if reset_global else 0)
                preserved_particles = ~selected_particles
                np.testing.assert_array_equal(
                    state.particle_q.numpy()[selected_particles], model.particle_q.numpy()[selected_particles]
                )
                np.testing.assert_array_equal(
                    state.particle_qd.numpy()[selected_particles], model.particle_qd.numpy()[selected_particles]
                )
                np.testing.assert_array_equal(state.particle_f.numpy()[selected_particles], 0.0)
                np.testing.assert_array_equal(
                    state.particle_q.numpy()[preserved_particles], particle_q[preserved_particles]
                )
                np.testing.assert_array_equal(state.particle_qd.numpy()[preserved_particles], 1.0)
                np.testing.assert_array_equal(state.particle_f.numpy()[preserved_particles], 1.0)

                if reset_global:
                    np.testing.assert_array_equal(state.body_q.numpy(), body_q)
                    np.testing.assert_array_equal(state.body_qd.numpy(), 1.0)
                else:
                    np.testing.assert_array_equal(state.body_q.numpy(), model.body_q.numpy())
                    np.testing.assert_array_equal(state.body_qd.numpy(), model.body_qd.numpy())
                np.testing.assert_array_equal(state.body_lox_dual_impulse.numpy(), 0.0)
                np.testing.assert_array_equal(state.particle_lox_dual_impulse.numpy(), 0.0)
                np.testing.assert_array_equal(lox.splitting.splitting_dual_impulse.numpy(), 0.0)
                np.testing.assert_array_equal(lox.deformable_splitting.dual_impulse.numpy(), 0.0)
                np.testing.assert_array_equal(success_mask.numpy(), [not reset_global, reset_global])

    def test_step_global_cloth_contact_with_local_rigid_body(self):
        """Couple global cloth to a dynamic body in its LOX solve world."""
        model, shape, body = _build_global_cloth_local_rigid_model(self.device)
        state_in = model.state()
        state_out = model.state()
        contacts = _make_particle_contact(model, state_in, shape, gap=-0.01)
        solver = newton.solvers.SolverKamino(model, config=_make_lox_config(max_iterations=50))

        solver.step(state_in, state_out, None, contacts, 0.01)

        contact_system = solver._solver_kamino._solver_fd.deformable_contacts
        self.assertEqual(int(model.particle_world.numpy()[0]), -1)
        self.assertEqual(int(model.body_world.numpy()[body]), 0)
        self.assertEqual(int(contact_system.status.numpy()[0]), 1)
        self.assertGreater(float(np.linalg.norm(state_out.particle_qd.numpy()[0])), 0.0)
        self.assertGreater(float(np.linalg.norm(state_out.body_qd.numpy()[body])), 0.0)

    def test_step_mixed_rigid_and_cloth_without_coupling(self):
        """Advance rigid and cloth state together without dynamic coupling."""
        model, _, _ = _build_contact_model(device=self.device, collider="dynamic")
        state_in = model.state()
        state_out = model.state()
        solver = newton.solvers.SolverKamino(model, config=_make_lox_config())

        solver.step(state_in, state_out, None, None, 0.01)

        self.assertTrue(np.all(np.isfinite(state_out.body_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.body_qd.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))
        self.assertEqual(
            int(solver._solver_kamino._solver_fd.world_status.numpy()[0]),
            LOX_STATUS_CONVERGED,
        )

    def test_capture_public_pure_cloth_step(self):
        """Capture and replay the public pure-cloth LOX step."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        model, _, _ = _build_contact_model(device=self.device)
        solver = newton.solvers.SolverKamino(model, config=_make_lox_config())
        state_in = model.state()
        state_out = model.state()
        solver.step(state_in, state_out, None, None, 0.01)
        state_in, state_out = state_out, state_in

        with wp.ScopedCapture(device=self.device) as capture:
            solver.step(state_in, state_out, None, None, 0.01)
        wp.capture_launch(capture.graph)

        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))

    def test_capture_public_nonlinear_membrane_step(self):
        """Capture and replay nonlinear membrane proximal updates."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        model, _, _ = _build_contact_model(device=self.device)
        solver = newton.solvers.SolverKamino(
            model,
            config=_make_lox_config(
                max_iterations=4,
                deformable_proximal_iterations=2,
                deformable_proximal_relaxation=0.5,
            ),
        )
        state_in = model.state()
        state_out = model.state()
        positions = state_in.particle_q.numpy()
        positions[-1] += np.array((0.1, -0.05, 0.1), dtype=np.float32)
        state_in.particle_q.assign(positions)
        solver.step(state_in, state_out, None, None, 0.01)
        state_in, state_out = state_out, state_in

        with wp.ScopedCapture(device=self.device) as capture:
            solver.step(state_in, state_out, None, None, 0.01)
        wp.capture_launch(capture.graph)

        system = solver._solver_kamino._solver_fd.deformable_system
        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))
        self.assertTrue(np.all(np.isfinite(system.membrane_proximal.multiplier.numpy())))

    def test_capture_public_dynamic_rigid_cloth_contact_step(self):
        """Capture and replay a two-way rigid-cloth contact step."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        model, shapes, _ = _build_contact_model(device=self.device, collider="dynamic")
        solver = newton.solvers.SolverKamino(
            model,
            config=_make_lox_config(),
        )
        state_in = model.state()
        state_out = model.state()
        state_in.particle_qd.fill_((0.0, 0.0, -1.0))
        contacts = _make_particle_contact(model, state_in, shapes[0], gap=0.0)
        solver.step(state_in, state_out, None, contacts, 0.01)
        state_in, state_out = state_out, state_in

        with wp.ScopedCapture(device=self.device) as capture:
            solver.step(state_in, state_out, None, contacts, 0.01)
        wp.capture_launch(capture.graph)

        self.assertTrue(np.all(np.isfinite(state_out.body_qd.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))
        self.assertTrue(bool(solver._solver_kamino._solver_fd.world_accepted.numpy()[0]))

    def test_capture_alternative_mixed_contact_projections(self):
        """Capture every alternative mixed-contact projection step."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        for projection_method in ("gauss_seidel", "apgd"):
            with self.subTest(projection_method=projection_method):
                model, shapes, _ = _build_contact_model(device=self.device, collider="dynamic")
                state_in = model.state()
                state_out = model.state()
                state_in.particle_qd.fill_((0.0, 0.0, -1.0))
                contacts = _make_particle_contact(model, state_in, shapes[0], gap=0.0)
                solver = newton.solvers.SolverKamino(
                    model,
                    config=_make_lox_config(
                        projection_iterations=8,
                        projection_method=projection_method,
                        gauss_seidel_max_colors=4 if projection_method == "gauss_seidel" else 0,
                    ),
                )
                solver.step(state_in, state_out, None, contacts, 0.01)
                state_in, state_out = state_out, state_in

                with wp.ScopedCapture(device=self.device) as capture:
                    solver.step(state_in, state_out, None, contacts, 0.01)
                wp.capture_launch(capture.graph)

                self.assertTrue(np.all(np.isfinite(state_out.body_qd.numpy())))
                self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))

    def test_capture_public_first_dynamic_rigid_cloth_contact_step(self):
        """Capture the first public rigid-cloth contact step without a warmup step."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        if not self.device.is_mempool_enabled:
            self.skipTest("First-step CUDA capture allocation requires the Warp memory pool.")
        model, shapes, _ = _build_contact_model(device=self.device, collider="dynamic")
        solver = newton.solvers.SolverKamino(
            model,
            config=_make_lox_config(),
        )
        state_in = model.state()
        state_out = model.state()
        state_in.particle_qd.fill_((0.0, 0.0, -1.0))
        contacts = _make_particle_contact(model, state_in, shapes[0], gap=0.0)

        with wp.ScopedCapture(device=self.device) as capture:
            solver.step(state_in, state_out, None, contacts, 0.01)
        wp.capture_launch(capture.graph)
        wp.capture_launch(capture.graph)

        self.assertIsNotNone(solver._solver_kamino._solver_fd.deformable_contacts)
        self.assertTrue(np.all(np.isfinite(state_out.body_qd.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))
        self.assertTrue(bool(solver._solver_kamino._solver_fd.world_accepted.numpy()[0]))

    def test_capture_public_full_surface_contact_step(self):
        """Capture collision generation and a two-way cloth-face contact step."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        model, _ = _build_full_surface_dynamic_contact_model(self.device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            soft_contact_margin=0.1,
            enable_rigid_soft_full_surface_contact=True,
        )
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverKamino(model, config=_make_lox_config())
        state_in = model.state()
        state_out = model.state()

        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, 0.01)
        solver.reset(state_in)

        with wp.ScopedCapture(device=self.device) as capture:
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, None, contacts, 0.01)
        wp.capture_launch(capture.graph)

        contact_count = int(contacts.soft_contact_count.numpy()[0])
        contact_indices = contacts.soft_contact_indices.numpy()[:contact_count]
        self.assertGreater(contact_count, 0)
        self.assertTrue(np.any(contact_indices[:, 1] >= 0))
        self.assertTrue(np.all(np.isfinite(state_out.body_qd.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))
        self.assertTrue(bool(solver._solver_kamino._solver_fd.world_accepted.numpy()[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
