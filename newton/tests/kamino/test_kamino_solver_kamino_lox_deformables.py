# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the deformable bodies of the LOX Kamino dynamics backend.

The tests step cloth and tetrahedral particles through :class:`newton.solvers.SolverKamino`
with the LOX backend: rigid-soft contacts with friction and mixed materials, malformed,
prescribed, and cross-world contact rows, the projection schedules and their accelerations,
two-way coupling with rigid bodies and rods, the solver-owned warm starts of failed worlds,
resets, and CUDA graph capture. The self-contact tests live in
``test_kamino_solver_kamino_lox_self_contact``.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.core.types import mat36f
from newton._src.solvers.kamino._src.solvers.lox.deformable.types import DeformableSystemData, DeformableWeightStatus
from newton._src.solvers.kamino._src.solvers.lox.solver import LOXSolver
from newton._src.solvers.kamino._src.solvers.lox.types import ParticleContactStatus
from newton._src.solvers.kamino.config import LOXSolverConfig
from newton.solvers import SolverKamino
from newton.tests.kamino import setup_tests, test_context

###
# Helpers
###


_MIXED_PROJECTION_VARIANTS = ((1, "none"), (4, "none"), (1, "apgd"), (1, "anderson"), (4, "anderson"))
"""Gauss--Seidel color limits and accelerations of the projection schedules; one color selects Jacobi."""


def _make_lox_config(**kwargs) -> SolverKamino.Config:
    """Create a LOX solver configuration with the given LOX options."""
    return SolverKamino.Config(dynamics_solver="lox", lox=LOXSolverConfig(**kwargs))


def _lox(solver: SolverKamino) -> LOXSolver:
    """Return the LOX forward-dynamics solver of a Kamino solver."""
    return solver._solver_kamino.solver_fd


def _world_time_steps(solver: SolverKamino, value: float) -> wp.array[wp.float32]:
    """Return a uniform time step for each solve world of a LOX solver."""
    lox = _lox(solver)
    return wp.full(lox.data.state.num_worlds, value, dtype=wp.float32, device=lox.data.state.world_active.device)


def _build_contact_model(
    *,
    device: wp.DeviceLike,
    world_count: int = 1,
    fix_left: bool = False,
    collider: str = "static",
    shape_margin: float = 0.02,
    body_com: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[newton.Model, list[int], list[int]]:
    """Build one cloth square and one static, kinematic, or dynamic plane per world."""
    builder = newton.ModelBuilder()
    shape_indices: list[int] = []
    body_indices: list[int] = []
    for world in range(world_count):
        builder.begin_world()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.5 + 2.0 * world),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            cell_x=1.0,
            cell_y=1.0,
            mass=1.0,
            fix_left=fix_left,
            tri_ke=100.0,
            tri_ka=80.0,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
            edge_ke=2.0,
            edge_kd=0.0,
        )
        body = -1
        if collider != "static":
            body = builder.add_body(
                xform=wp.transform_identity(),
                com=body_com,
                mass=1.0,
                inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                lock_inertia=body_com != (0.0, 0.0, 0.0),
                is_kinematic=collider == "kinematic",
            )
            body_indices.append(body)
        shape_indices.append(
            builder.add_shape_plane(
                body=body,
                cfg=newton.ModelBuilder.ShapeConfig(margin=shape_margin),
            )
        )
        builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    model.soft_contact_mu = 0.5
    model.soft_contact_restitution = 0.0
    return model, shape_indices, body_indices


def _make_contacts(*, device: wp.DeviceLike, capacity: int, records: list[dict]) -> newton.Contacts:
    """Create Newton soft contacts from manually specified records."""
    contacts = newton.Contacts(0, capacity, device=device)
    indices = np.full((capacity, 3), -1, dtype=np.int32)
    barycentric = np.zeros((capacity, 3), dtype=np.float32)
    shapes = np.full(capacity, -1, dtype=np.int32)
    body_positions = np.zeros((capacity, 3), dtype=np.float32)
    body_velocities = np.zeros((capacity, 3), dtype=np.float32)
    normals = np.zeros((capacity, 3), dtype=np.float32)
    for contact, record in enumerate(records):
        indices[contact] = record["indices"]
        barycentric[contact] = record["barycentric"]
        shapes[contact] = record["shape"]
        body_positions[contact] = record["body_position"]
        body_velocities[contact] = record.get("body_velocity", (0.0, 0.0, 0.0))
        normals[contact] = record.get("normal", (0.0, 0.0, 1.0))

    contacts.soft_contact_count.assign(np.array([len(records)], dtype=np.int32))
    contacts.soft_contact_indices.assign(indices)
    contacts.soft_contact_barycentric.assign(barycentric)
    contacts.soft_contact_shape.assign(shapes)
    contacts.soft_contact_body_pos.assign(body_positions)
    contacts.soft_contact_body_vel.assign(body_velocities)
    contacts.soft_contact_normal.assign(normals)
    return contacts


def _surface_position_for_gap(
    model: newton.Model,
    positions: np.ndarray,
    indices: tuple[int, int, int],
    coefficients: tuple[float, float, float],
    shape: int,
    gap: float,
    normal: np.ndarray,
) -> np.ndarray:
    """Place a collider point at the requested effective gap from a particle feature."""
    feature = np.zeros(3, dtype=np.float64)
    for particle, coefficient in zip(indices, coefficients, strict=True):
        if particle >= 0:
            feature += coefficient * positions[particle]
    radii = model.particle_radius.numpy()
    radius = max(float(radii[particle]) for particle in indices if particle >= 0)
    margin = float(model.shape_margin.numpy()[shape])
    return feature - (gap + radius + margin) * normal


def _make_particle_contact(
    model: newton.Model,
    state: newton.State,
    shape: int,
    *,
    gap: float,
    particle: int = 0,
) -> newton.Contacts:
    """Create one particle-plane soft contact at the requested effective gap."""
    normal = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    body_position = _surface_position_for_gap(
        model,
        state.particle_q.numpy(),
        (particle, -1, -1),
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
                "indices": (particle, -1, -1),
                "barycentric": (1.0, 0.0, 0.0),
                "shape": shape,
                "body_position": body_position,
                "normal": normal,
            }
        ],
    )


def _dense_smooth_matrix(data: DeformableSystemData) -> np.ndarray:
    """Return the smooth matrix ``A`` of the deformable particles as a dense float64 matrix, in packed order."""
    matrix = data.smooth_matrix
    offsets = matrix.offsets.numpy()
    columns = matrix.columns.numpy()
    values = matrix.values.numpy().astype(np.float64)
    dense = np.zeros((3 * matrix.nrow, 3 * matrix.ncol))
    for row in range(matrix.nrow):
        for entry in range(offsets[row], offsets[row + 1]):
            column = columns[entry]
            dense[3 * row : 3 * row + 3, 3 * column : 3 * column + 3] += values[entry]
    return dense


def _build_patch_weight_models() -> dict[str, tuple[newton.ModelBuilder, bool]]:
    """Build single-element meshes, whose patches span the mesh, and grid meshes, whose patches do not.

    Returns:
        The builder of each mesh and whether its patches span the mesh.
    """
    tetrahedron = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for position in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        tetrahedron.add_particle(pos=wp.vec3(*position), vel=wp.vec3(0.0), mass=1.0)
    tetrahedron.add_tetrahedron(0, 1, 2, 3, k_mu=1.0e4, k_lambda=2.0e4)

    cloth_kwargs = {
        "rot": wp.quat_identity(),
        "vel": wp.vec3(0.0),
        "tri_ke": 1.0e4,
        "tri_ka": 1.0e4,
        "tri_kd": 0.0,
        "edge_kd": 0.0,
    }
    cloth_cell = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cloth_cell.add_cloth_grid(
        pos=wp.vec3(0.0), dim_x=1, dim_y=1, cell_x=1.0, cell_y=1.0, mass=1.0, edge_ke=10.0, **cloth_kwargs
    )
    cloth_grid = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    cloth_grid.add_cloth_grid(
        pos=wp.vec3(0.0), dim_x=6, dim_y=6, cell_x=0.1, cell_y=0.1, mass=0.01, edge_ke=1.0, **cloth_kwargs
    )
    tetrahedron_grid = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    tetrahedron_grid.add_soft_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=3,
        dim_y=3,
        dim_z=3,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
        density=1000.0,
        k_mu=1.0e6,
        k_lambda=1.0e6,
        k_damp=0.0,
    )
    return {
        "tetrahedron": (tetrahedron, True),
        "cloth_cell": (cloth_cell, True),
        "cloth_grid": (cloth_grid, False),
        "tetrahedron_grid": (tetrahedron_grid, False),
    }


def _build_full_surface_dynamic_contact_model(device: wp.DeviceLike) -> tuple[newton.Model, int]:
    """Build a dynamic box that crosses a cloth face between its particles."""
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
    """Build a global and a local cloth square, and one local dynamic plane."""
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


###
# Tests
###


class TestSolverKaminoLOXDeformables(unittest.TestCase):
    """Step deformable particles and their contacts through the LOX backend of SolverKamino."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def _require_graph_capture(self) -> None:
        """Skip unless the device captures the conditional splitting loop in a CUDA graph."""
        if not self.default_device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        if not wp.is_conditional_graph_supported():
            self.skipTest("CUDA conditional graph nodes require CUDA 12.4 or newer.")

    def test_step_hanging_stiff_cloth_preserves_pins(self):
        """Keep pinned nodes fixed while a stiff bent cloth responds finitely."""
        model, _, _ = _build_contact_model(device=self.default_device, fix_left=True)
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
        active = (model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0
        velocities = state_in.particle_qd.numpy()
        velocities[~active] = np.array((0.5, -0.25, 0.75), dtype=np.float32)
        state_in.particle_qd.assign(velocities)
        for selective_weights in (False, True):
            with self.subTest(selective_weights=selective_weights):
                state_out = model.state()
                solver = SolverKamino(model, config=_make_lox_config(selective_weights=selective_weights))

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
        model, shapes, _ = _build_contact_model(device=self.default_device)
        state_in = model.state()
        state_in.particle_qd.fill_((1.0, 0.0, -1.0))
        contacts = _make_particle_contact(model, state_in, shapes[0], gap=-0.01)
        state_out = model.state()
        solver = SolverKamino(model, config=_make_lox_config(max_iterations=40))
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
        status = solver.status.numpy()
        np.testing.assert_array_equal(status["converged"], [1])
        np.testing.assert_array_equal(status["failed"], [0])

    def test_step_blends_projected_and_smooth_particle_velocities(self):
        """Integrate the particle velocity blended by the projected fraction of a partially converged solve."""
        model, shapes, _ = _build_contact_model(device=self.default_device)
        state_in = model.state()
        state_in.particle_qd.fill_((1.0, 0.0, -1.0))
        contacts = _make_particle_contact(model, state_in, shapes[0], gap=-0.01)
        dt = 0.01
        velocities = {}
        for fraction in (0.0, 0.5, 1.0):
            with self.subTest(fraction=fraction):
                state_out = model.state()
                config = _make_lox_config(
                    max_iterations=2,
                    fixed_iterations=True,
                    deformable_projected_velocity_fraction=fraction,
                )
                solver = SolverKamino(model, config=config)

                solver.step(state_in, state_out, None, contacts, dt)

                velocities[fraction] = state_out.particle_qd.numpy()
                np.testing.assert_allclose(
                    state_out.particle_q.numpy(),
                    state_in.particle_q.numpy() + dt * velocities[fraction],
                    rtol=2.0e-5,
                    atol=2.0e-6,
                )

        # Two iterations leave the projected and smooth velocities apart, and the fraction weights them linearly
        self.assertGreater(float(np.abs(velocities[1.0] - velocities[0.0]).max()), 1.0e-3)
        np.testing.assert_allclose(velocities[0.5], 0.5 * (velocities[0.0] + velocities[1.0]), rtol=1.0e-5, atol=1.0e-6)

    def test_kinematic_rigid_contact_uses_input_pose_and_velocity(self):
        """Adapt the contact geometry and velocity to the prescribed input state of a kinematic collider.

        The collider body enters the contact row through its Jacobian, which maps the prescribed
        body twist to the row velocity, and its surface velocity through the row bias.
        """
        model, shapes, bodies = _build_contact_model(device=self.default_device, collider="kinematic")
        body = bodies[0]
        state_in = model.state()
        translation = np.asarray((0.2, -0.1, 0.3), dtype=np.float32)
        state_in.body_q.assign([wp.transformf(wp.vec3f(*translation), wp.quat_identity(dtype=wp.float32))])
        body_velocity = np.zeros((model.body_count, 6), dtype=np.float32)
        body_velocity[body, 2] = 0.5
        state_in.body_qd.assign(body_velocity)

        normal = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
        world_body_position = _surface_position_for_gap(
            model,
            state_in.particle_q.numpy(),
            (0, -1, -1),
            (1.0, 0.0, 0.0),
            shapes[0],
            0.0,
            normal,
        )
        contacts = _make_contacts(
            device=self.default_device,
            capacity=1,
            records=[
                {
                    "indices": (0, -1, -1),
                    "barycentric": (1.0, 0.0, 0.0),
                    "shape": shapes[0],
                    "body_position": world_body_position - translation,
                    "normal": normal,
                }
            ],
        )
        state_out = model.state()
        solver = SolverKamino(model, config=_make_lox_config(max_iterations=1))

        solver.step(state_in, state_out, None, contacts, 0.01)

        rows = _lox(solver).deformables.contact_rows
        np.testing.assert_array_equal(rows.status.numpy(), [ParticleContactStatus.ACTIVE])
        np.testing.assert_allclose(rows.gap.numpy(), 0.0, rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(rows.body.numpy(), [body])
        # The body center of mass is at its origin, so the Newton twist is the twist of the row Jacobian
        row_velocity = rows.bias.numpy()[0] + rows.body_jacobian.numpy()[0] @ body_velocity[body]
        np.testing.assert_allclose(row_velocity, [0.0, 0.0, -0.5], rtol=0.0, atol=1.0e-6)

    def test_step_mixes_rigid_deformable_contact_materials(self):
        """Mix friction and restitution with each contacted shape's material."""
        model, shapes, _ = _build_contact_model(device=self.default_device, world_count=2)
        model.soft_contact_mu = 0.6
        model.soft_contact_restitution = 0.7
        model.shape_material_mu.assign(np.array([0.2, 0.8], dtype=np.float32))
        model.shape_material_restitution.assign(np.array([0.3, 0.9], dtype=np.float32))
        state_in = model.state()
        state_in.particle_qd.fill_((0.0, 0.0, -1.0))
        particle_world = model.particle_world.numpy()
        particle_indices = [int(np.flatnonzero(particle_world == world)[0]) for world in range(model.world_count)]
        normal = np.array((0.0, 0.0, 1.0), dtype=np.float32)
        positions = state_in.particle_q.numpy()
        records = []
        for particle, shape in zip(particle_indices, shapes, strict=True):
            records.append(
                {
                    "indices": (particle, -1, -1),
                    "barycentric": (1.0, 0.0, 0.0),
                    "shape": shape,
                    "body_position": _surface_position_for_gap(
                        model,
                        positions,
                        (particle, -1, -1),
                        (1.0, 0.0, 0.0),
                        shape,
                        0.0,
                        normal,
                    ),
                    "normal": normal,
                }
            )
        contacts = _make_contacts(device=self.default_device, capacity=2, records=records)
        state_out = model.state()
        solver = SolverKamino(
            model,
            config=_make_lox_config(
                max_iterations=1,
                projection_iterations=2,
                gauss_seidel_max_colors=1,
                projection_acceleration="anderson",
            ),
        )

        solver.step(state_in, state_out, None, contacts, 0.01)

        lox = _lox(solver)
        rows = lox.deformables.contact_rows
        # Anderson reduces the secant terms of the particle contact rows with atomics
        self.assertFalse(lox._projection.data.anderson.global_reduction)
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0, 0])
        np.testing.assert_allclose(rows.friction.numpy(), [0.4, 0.7], rtol=1.0e-6)
        np.testing.assert_allclose(rows.bias.numpy()[:, 2], [-0.3, -0.7], rtol=1.0e-6)

        config = _make_lox_config(max_iterations=1)
        config.materials.friction_mix_mode = "multiply"
        config.materials.restitution_mix_mode = "max"
        custom_solver = SolverKamino(model, config=config)
        custom_state_out = model.state()

        custom_solver.step(state_in, custom_state_out, None, contacts, 0.01)

        custom_rows = _lox(custom_solver).deformables.contact_rows
        np.testing.assert_allclose(custom_rows.friction.numpy(), [0.12, 0.48], rtol=1.0e-6)
        np.testing.assert_allclose(custom_rows.bias.numpy()[:, 2], [-0.7, -0.9], rtol=1.0e-6)

    def test_step_drops_malformed_contact_without_failing_world(self):
        """Report and ignore one malformed record without suppressing valid contact."""
        model, shapes, _ = _build_contact_model(device=self.default_device)
        state_in = model.state()
        state_in.particle_qd.fill_((0.0, 0.0, -1.0))
        normal = np.array((0.0, 0.0, 1.0), dtype=np.float32)
        body_position = _surface_position_for_gap(
            model,
            state_in.particle_q.numpy(),
            (0, -1, -1),
            (1.0, 0.0, 0.0),
            shapes[0],
            -0.01,
            normal,
        )
        contacts = _make_contacts(
            device=self.default_device,
            capacity=2,
            records=[
                {
                    "indices": (0, -1, -1),
                    "barycentric": (1.0, 0.0, 0.0),
                    "shape": shapes[0],
                    "body_position": body_position,
                    "normal": (0.0, 0.0, 0.0),
                },
                {
                    "indices": (0, -1, -1),
                    "barycentric": (1.0, 0.0, 0.0),
                    "shape": shapes[0],
                    "body_position": body_position,
                    "normal": normal,
                },
            ],
        )
        state_out = model.state()
        solver = SolverKamino(model, config=_make_lox_config(max_iterations=40))

        solver.step(state_in, state_out, None, contacts, 0.01)

        rows = _lox(solver).deformables.contact_rows
        np.testing.assert_array_equal(
            rows.status.numpy(),
            [ParticleContactStatus.MALFORMED, ParticleContactStatus.ACTIVE],
        )
        np.testing.assert_array_equal(rows.reaction.numpy()[0], 0.0)
        np.testing.assert_array_equal(rows.world_count.numpy(), [1])
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])
        self.assertGreater(float(state_out.particle_qd.numpy()[0, 2]), -1.0)

    def test_step_drops_contact_between_prescribed_features(self):
        """Retire a contact without a dynamic endpoint without failing the world."""
        model, shapes, _ = _build_contact_model(device=self.default_device, fix_left=True)
        state_in = model.state()
        contacts = _make_particle_contact(model, state_in, shapes[0], gap=-0.01)
        state_out = model.state()
        solver = SolverKamino(model, config=_make_lox_config(max_iterations=40))

        solver.step(state_in, state_out, None, contacts, 0.01)

        rows = _lox(solver).deformables.contact_rows
        np.testing.assert_array_equal(rows.status.numpy(), [ParticleContactStatus.PRESCRIBED])
        np.testing.assert_array_equal(rows.reaction.numpy(), 0.0)
        np.testing.assert_array_equal(rows.world_count.numpy(), [0])
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])

    def test_step_static_contact_with_alternative_projections(self):
        """Advance pure deformable contact with every projection schedule."""
        for max_colors, acceleration in _MIXED_PROJECTION_VARIANTS:
            with self.subTest(max_colors=max_colors, acceleration=acceleration):
                model, shapes, _ = _build_contact_model(device=self.default_device)
                state_in = model.state()
                state_in.particle_qd.fill_((0.5, 0.0, -1.0))
                contacts = _make_particle_contact(model, state_in, shapes[0], gap=-0.01)
                state_out = model.state()
                solver = SolverKamino(
                    model,
                    config=_make_lox_config(
                        max_iterations=40,
                        projection_iterations=8,
                        projection_acceleration=acceleration,
                        gauss_seidel_max_colors=max_colors,
                    ),
                )

                solver.step(state_in, state_out, None, contacts, 0.01)

                np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])
                self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))

    def _run_single_sweep_contact(
        self, *, collider: str, max_colors: int, acceleration: str, projection_iterations: int
    ) -> tuple[SolverKamino, newton.State]:
        """Step one particle-plane contact with one splitting iteration and return the solver and output state."""
        model, shapes, _ = _build_contact_model(device=self.default_device, collider=collider)
        state_in = model.state()
        state_in.particle_qd.fill_((0.5, 0.25, -1.0))
        contacts = _make_particle_contact(model, state_in, shapes[0], gap=-0.01)
        state_out = model.state()
        solver = SolverKamino(
            model,
            config=_make_lox_config(
                max_iterations=1,
                projection_iterations=projection_iterations,
                gauss_seidel_max_colors=max_colors,
                projection_acceleration=acceleration,
            ),
        )
        solver.step(state_in, state_out, None, contacts, 0.01)
        return solver, state_out

    def test_patch_weights_bracket_particle_delassus(self):
        """Take the inverse particle Delassus as patch weight when the patch spans the mesh, and stay within
        the particle mass and the smooth diagonal block otherwise."""
        for name, (builder, spanning) in _build_patch_weight_models().items():
            with self.subTest(mesh=name):
                model = builder.finalize(device=self.default_device)
                solver = SolverKamino(model, config=_make_lox_config())
                state_in, state_out = model.state(), model.state()
                solver.step(state_in, state_out, None, None, 0.01)
                np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])
                deformables = _lox(solver).deformables
                data = deformables.system_data
                weight = data.full_weight.numpy().astype(np.float64)
                mass = model.particle_mass.numpy()[deformables.topology.packed_to_newton.numpy()]
                np.testing.assert_array_equal(data.weight_status.numpy(), DeformableWeightStatus.VALID)
                matrix = _dense_smooth_matrix(data)
                inverse = np.linalg.inv(matrix)
                blocks = [slice(3 * p, 3 * p + 3) for p in range(len(mass))]
                # The inverse particle Delassus 1 / lambda_max((A^-1)_ii) and the smallest eigenvalue of A_ii
                exact = np.array([1.0 / np.linalg.eigvalsh(inverse[b, b]).max() for b in blocks])
                diagonal = np.array([np.linalg.eigvalsh(matrix[b, b]).min() for b in blocks])
                if spanning:
                    np.testing.assert_allclose(weight, exact, rtol=1.0e-4)
                else:
                    np.testing.assert_array_less(mass * (1.0 - 1.0e-6), weight)
                    np.testing.assert_array_less(weight, diagonal * (1.0 + 1.0e-5))
                    # The elements of the patches stiffen the weights beyond the particle masses
                    self.assertGreater(float(np.min(weight / mass)), 2.0)

    def test_deformable_weight_beta_caps_particle_weights(self):
        """Keep each particle weight between its free bound and deformable_weight_beta times it, and below the
        geometric mean."""
        models = _build_patch_weight_models()
        for name in ("cloth_grid", "tetrahedron_grid"):
            with self.subTest(mesh=name):
                model = models[name][0].finalize(device=self.default_device)

                def weights(beta: float | None, sigma: float, model=model) -> np.ndarray:
                    config = _make_lox_config(deformable_weight_beta=beta, weight_sigma=sigma)
                    solver = SolverKamino(model, config=config)
                    solver.step(model.state(), model.state(), None, None, 0.01)
                    np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])
                    return _lox(solver).deformables.system_data.full_weight.numpy().astype(np.float64)

                mean = weights(None, 0.0)
                free = weights(1.0, 0.0)
                self.assertTrue(np.all(free <= mean * (1.0 + 1.0e-6)))
                # A cap between one and the largest ratio of the mean binds on some particles
                beta = 0.5 * (1.0 + float(np.max(mean / free)))
                self.assertGreater(beta, 1.1)
                capped = weights(beta, 0.0)
                np.testing.assert_allclose(capped, np.minimum(mean, beta * free), rtol=1.0e-5)
                self.assertTrue(np.any(capped < mean * (1.0 - 1.0e-3)))
                # A unit floor keeps the mean under any cap
                np.testing.assert_allclose(weights(1.0, 1.0), mean, rtol=1.0e-6)

    def test_candidate_matrix_takes_deformable_consensus_fraction(self):
        """Add the fraction alpha of the particle weights to the particle candidate matrix A + alpha W."""
        model, _, _ = _build_contact_model(device=self.default_device)
        for fraction in (0.0, 0.5, 1.0):
            with self.subTest(deformable_consensus_fraction=fraction):
                solver = SolverKamino(
                    model, config=_make_lox_config(deformable_consensus_fraction=fraction, selective_weights=False)
                )
                state_in, state_out = model.state(), model.state()
                solver.step(state_in, state_out, None, None, 0.01)
                data = _lox(solver).deformables.system_data
                nnz = data.smooth_matrix.nnz
                smooth = data.smooth_matrix.values.numpy()[:nnz]
                system = data.system_matrix.values.numpy()[:nnz]
                weight = data.weight.numpy()
                self.assertTrue(np.all(weight > 0.0))
                expected = smooth.copy()
                expected[data.diagonal_slots.numpy()] += fraction * weight[:, None, None] * np.eye(3)
                np.testing.assert_allclose(system, expected, rtol=1.0e-6, atol=0.0)

    def test_consensus_fractions_share_contact_fixed_point(self):
        """Converge the frictional particle contacts of every pair of consensus fractions to the same velocities."""
        fractions = ((0.5, 1.0), (1.0, 1.0), (0.5, 0.5), (0.0, 0.0))
        for collider in ("static", "dynamic"):
            velocities = {}
            for rigid_fraction, deformable_fraction in fractions:
                with self.subTest(collider=collider, fractions=(rigid_fraction, deformable_fraction)):
                    model, shapes, bodies = _build_contact_model(device=self.default_device, collider=collider)
                    state_in = model.state()
                    state_in.particle_qd.fill_((0.3, 0.1, -1.0))
                    state_out = model.state()
                    contacts = _make_particle_contact(model, state_in, shapes[0], gap=0.0)
                    solver = SolverKamino(
                        model,
                        config=_make_lox_config(
                            consensus_fraction=rigid_fraction,
                            deformable_consensus_fraction=deformable_fraction,
                            max_iterations=400,
                            use_graph_conditionals=False,
                            velocity_tolerance=1.0e-6,
                            position_tolerance=1.0e-7,
                        ),
                    )
                    solver.step(state_in, state_out, None, contacts, 0.01)
                    status = solver.status.numpy()
                    np.testing.assert_array_equal(status["failed"], [0])
                    np.testing.assert_array_equal(status["converged"], [1])
                    particle_qd = state_out.particle_qd.numpy()
                    if collider == "static":
                        # The contact particle sticks to the plane under the frictional impulse
                        np.testing.assert_allclose(particle_qd[0], 0.0, atol=1.0e-5)
                    body_qd = state_out.body_qd.numpy()[bodies] if bodies else np.zeros((0, 6))
                    velocities[(rigid_fraction, deformable_fraction)] = (particle_qd, body_qd)
            reference = velocities[fractions[0]]
            for pair in fractions[1:]:
                np.testing.assert_allclose(velocities[pair][0], reference[0], atol=1.0e-5)
                np.testing.assert_allclose(velocities[pair][1], reference[1], atol=1.0e-5)

    def test_first_accelerated_step_matches_mixed_contact_jacobi(self):
        """Match one mixed-contact Jacobi sweep before APGD or Anderson has history."""
        results = {}
        for acceleration in ("none", "apgd", "anderson"):
            solver, state_out = self._run_single_sweep_contact(
                collider="dynamic", max_colors=1, acceleration=acceleration, projection_iterations=1
            )
            results[acceleration] = (
                state_out.particle_qd.numpy(),
                state_out.body_qd.numpy(),
                _lox(solver).deformables.contact_rows.reaction.numpy(),
            )

        for acceleration in ("apgd", "anderson"):
            for accelerated_value, jacobi_value in zip(results[acceleration], results["none"], strict=True):
                np.testing.assert_allclose(accelerated_value, jacobi_value, rtol=2.0e-5, atol=2.0e-6)

    def test_first_accelerated_step_matches_static_contact_jacobi(self):
        """Match one pure-deformable Jacobi sweep before APGD or Anderson has history."""
        results = {}
        for acceleration in ("none", "apgd", "anderson"):
            solver, state_out = self._run_single_sweep_contact(
                collider="static", max_colors=1, acceleration=acceleration, projection_iterations=1
            )
            results[acceleration] = (
                state_out.particle_qd.numpy(),
                _lox(solver).deformables.contact_rows.reaction.numpy(),
            )

        for acceleration in ("apgd", "anderson"):
            for accelerated_value, jacobi_value in zip(results[acceleration], results["none"], strict=True):
                np.testing.assert_allclose(accelerated_value, jacobi_value, rtol=2.0e-5, atol=2.0e-6)

    def test_step_mixed_contact_with_alternative_projections(self):
        """Advance two-way rigid-deformable contact with every projection schedule."""
        for max_colors, acceleration in _MIXED_PROJECTION_VARIANTS:
            with self.subTest(max_colors=max_colors, acceleration=acceleration):
                model, shapes, bodies = _build_contact_model(device=self.default_device, collider="dynamic")
                state_in = model.state()
                state_in.particle_qd.fill_((0.0, 0.0, -1.0))
                contacts = _make_particle_contact(model, state_in, shapes[0], gap=0.0)
                state_out = model.state()
                solver = SolverKamino(
                    model,
                    config=_make_lox_config(
                        max_iterations=50,
                        projection_iterations=8,
                        projection_acceleration=acceleration,
                        gauss_seidel_max_colors=max_colors,
                    ),
                )

                solver.step(state_in, state_out, None, contacts, 0.01)

                np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])
                self.assertGreater(float(state_out.particle_qd.numpy()[0, 2]), -1.0)
                self.assertLess(float(state_out.body_qd.numpy()[bodies[0], 2]), 0.0)

    def test_step_dynamic_rigid_contact_updates_both_endpoints(self):
        """Exchange contact momentum in one coupled rigid-cloth solve."""
        model, shapes, bodies = _build_contact_model(
            device=self.default_device,
            collider="dynamic",
            body_com=(0.0, 0.0, 0.25),
        )
        state_in = model.state()
        state_in.particle_qd.fill_((0.0, 0.0, -1.0))
        contacts = _make_particle_contact(model, state_in, shapes[0], gap=0.0)
        state_out = model.state()
        solver = SolverKamino(model, config=_make_lox_config(max_iterations=50))
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
        """Compare in-place LOX stepping and its solver-owned warm starts with a ping-pong reference."""
        model, shapes, _ = _build_contact_model(
            device=self.default_device,
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
        solver_ping_pong = SolverKamino(model, config=_make_lox_config(max_iterations=50))
        solver_in_place = SolverKamino(model, config=_make_lox_config(max_iterations=50))

        graph = None
        if capture:
            solver_in_place.step(state_in_place, state_in_place, None, contacts_in_place, 0.01)
            solver_in_place.reset(state_in_place)
            state_in_place.particle_qd.fill_((0.0, 0.0, -1.0))
            with wp.ScopedCapture(device=self.default_device) as captured:
                solver_in_place.step(state_in_place, state_in_place, None, contacts_in_place, 0.01)
            graph = captured.graph

        for _ in range(3):
            solver_ping_pong.step(state_ping, state_pong, None, contacts_ping, 0.01)
            state_ping, state_pong = state_pong, state_ping
            if graph is None:
                solver_in_place.step(state_in_place, state_in_place, None, contacts_in_place, 0.01)
            else:
                wp.capture_launch(graph)

        reference = _lox(solver_ping_pong).data
        in_place = _lox(solver_in_place).data
        self.assertGreater(float(np.linalg.norm(reference.dual_wrench.numpy())), 0.0)
        self.assertGreater(float(np.linalg.norm(reference.particle_dual_force.numpy())), 0.0)
        for name in ("body_q", "body_qd", "particle_q", "particle_qd"):
            np.testing.assert_allclose(
                getattr(state_in_place, name).numpy(),
                getattr(state_ping, name).numpy(),
                rtol=1.0e-6,
                atol=1.0e-6,
            )
        np.testing.assert_allclose(
            in_place.dual_wrench.numpy(),
            reference.dual_wrench.numpy(),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            in_place.particle_dual_force.numpy(),
            reference.particle_dual_force.numpy(),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    def test_step_in_place_matches_ping_pong_with_nonzero_body_com(self):
        """Match ping-pong LOX stepping in place with persistent warm starts."""
        self._check_step_in_place_matches_ping_pong_with_nonzero_body_com(capture=False)

    def test_rejected_outer_solve_preserves_particle_dual_warmstart(self):
        """Keep the accepted particle warm start when the solve of its world fails."""
        model, shapes, _ = _build_contact_model(device=self.default_device, collider="dynamic")
        state_in = model.state()
        state_in.particle_qd.fill_((0.0, 0.0, -1.0))
        state_out = model.state()
        contacts = _make_particle_contact(model, state_in, shapes[0], gap=0.0)
        solver = SolverKamino(model, config=_make_lox_config(max_iterations=20))
        solver.step(state_in, state_out, None, contacts, 0.01)
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])

        lox = _lox(solver)
        baseline_force = lox.data.particle_dual_force.numpy().copy()
        self.assertGreater(float(np.linalg.norm(baseline_force)), 0.0)
        lox.data.particles.splitting_dual.fill_((3.0, -2.0, 1.0))
        lox.data.state.world_failed.fill_(True)
        lox._store_dual_wrench()

        np.testing.assert_array_equal(lox.data.particle_dual_force.numpy(), baseline_force)

    def test_capture_step_in_place_matches_ping_pong_with_nonzero_body_com(self):
        """Replay in-place LOX capture in origin coordinates with persistent warm starts."""
        self._require_graph_capture()
        self._check_step_in_place_matches_ping_pong_with_nonzero_body_com(capture=True)

    def test_step_dynamic_rigid_contact_against_pinned_particle(self):
        """Resolve a dynamic contact whose cloth endpoint has zero inverse mass."""
        model, shapes, bodies = _build_contact_model(
            device=self.default_device,
            fix_left=True,
            collider="dynamic",
        )
        state_in = model.state()
        body_velocity = np.zeros((1, 6), dtype=np.float32)
        body_velocity[0, 2] = 1.0
        state_in.body_qd.assign(body_velocity)
        contacts = _make_particle_contact(model, state_in, shapes[0], gap=0.0)
        state_out = model.state()
        solver = SolverKamino(model, config=_make_lox_config())

        solver.step(state_in, state_out, None, contacts, 0.01)

        rows = _lox(solver).deformables.contact_rows
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])
        np.testing.assert_array_equal(rows.status.numpy(), [ParticleContactStatus.ACTIVE])
        # The body response forms the unsplit Delassus block of a contact with a pinned particle
        self.assertGreater(float(rows.physical_delassus.numpy()[0, 2, 2]), 0.0)
        np.testing.assert_array_equal(state_out.particle_qd.numpy()[0], 0.0)
        self.assertLess(float(state_out.body_qd.numpy()[bodies[0], 2]), 1.0)

    def test_step_multiworld_cloth_independently(self):
        """Advance multiple cloth worlds with independent external forces."""
        model, _, _ = _build_contact_model(device=self.default_device, world_count=2)
        state_in = model.state()
        forces = np.zeros((model.particle_count, 3), dtype=np.float32)
        particle_world = model.particle_world.numpy()
        forces[particle_world == 0, 0] = 1.0
        forces[particle_world == 1, 0] = -2.0
        state_in.particle_f.assign(forces)
        state_out = model.state()
        solver = SolverKamino(model, config=_make_lox_config())

        solver.step(state_in, state_out, None, None, 0.01)

        velocity = state_out.particle_qd.numpy()
        self.assertGreater(float(np.mean(velocity[particle_world == 0, 0])), 0.0)
        self.assertLess(float(np.mean(velocity[particle_world == 1, 0])), 0.0)
        status = solver.status.numpy()
        np.testing.assert_array_equal(status["converged"], [1, 1])
        np.testing.assert_array_equal(status["failed"], [0, 0])

    def test_step_multiworld_cloth_uses_each_world_time_step(self):
        """Advance deformable worlds with distinct device-side time steps."""
        model, _, _ = _build_contact_model(device=self.default_device, world_count=2)
        state_in = model.state()
        particle_world = model.particle_world.numpy()
        forces = np.zeros((model.particle_count, 3), dtype=np.float32)
        forces[:, 0] = 1.0
        state_in.particle_f.assign(forces)
        state_out = model.state()
        solver = SolverKamino(model, config=_make_lox_config())
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
        model, _, _ = _build_contact_model(device=self.default_device)
        state_in = model.state()
        state_out = model.state()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        solver = SolverKamino(model, config=_make_lox_config())

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
        model = builder.finalize(device=self.default_device)
        model.set_gravity((0.0, 0.0, -9.81))
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = pipeline.contacts()
        state_in = model.state()
        state_out = model.state()
        solver = SolverKamino(model, config=_make_lox_config())

        pipeline.collide(state_in, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        solver.step(state_in, state_out, None, contacts, 0.01)

        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])
        self.assertTrue(np.all(state_out.particle_qd.numpy()[:, 2] < 0.0))

    def test_step_full_surface_contact_updates_cloth_and_rigid_body(self):
        """Resolve a generated face contact that the particle pass misses."""
        model, body = _build_full_surface_dynamic_contact_model(self.default_device)
        state_in = model.state()

        particle_pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            soft_contact_gap=0.1,
        )
        particle_contacts = particle_pipeline.contacts()
        particle_pipeline.collide(state_in, particle_contacts)
        self.assertEqual(int(particle_contacts.soft_contact_count.numpy()[0]), 0)

        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            soft_contact_gap=0.1,
            enable_rigid_soft_full_surface_contact=True,
        )
        contacts = pipeline.contacts()
        pipeline.collide(state_in, contacts)
        contact_count = int(contacts.soft_contact_count.numpy()[0])
        contact_indices = contacts.soft_contact_indices.numpy()[:contact_count]
        self.assertGreater(contact_count, 0)
        self.assertTrue(np.all(contact_indices[:, 1] >= 0))

        state_out = model.state()
        solver = SolverKamino(model, config=_make_lox_config(max_iterations=50))
        solver.step(state_in, state_out, None, contacts, 0.01)

        rows = _lox(solver).deformables.contact_rows
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])
        self.assertTrue(np.any(rows.coefficient.numpy()[:contact_count, 1:] > 0.0))
        self.assertTrue(np.all(rows.body.numpy()[:contact_count] == body))
        self.assertGreater(float(np.linalg.norm(state_out.particle_qd.numpy())), 0.0)
        self.assertGreater(float(np.linalg.norm(state_out.body_qd.numpy()[body])), 0.0)

    def test_reset_pure_cloth_state_and_warm_start(self):
        """Reset Newton-owned particle state and clear the pure-cloth LOX warm start."""
        model, _, _ = _build_contact_model(device=self.default_device)
        model.set_gravity((0.0, 0.0, -9.81))
        state = model.state()
        state_out = model.state()
        solver = SolverKamino(model, config=_make_lox_config())
        solver.step(state, state_out, None, None, 0.01)
        state.assign(state_out)
        state.particle_f.fill_((1.0, 2.0, 3.0))
        lox = _lox(solver)
        lox.data.particle_dual_force.fill_((1.0, 2.0, 3.0))

        solver.reset(state)

        np.testing.assert_array_equal(state.particle_q.numpy(), model.particle_q.numpy())
        np.testing.assert_array_equal(state.particle_qd.numpy(), model.particle_qd.numpy())
        np.testing.assert_array_equal(state.particle_f.numpy(), 0.0)
        np.testing.assert_array_equal(lox.data.particle_dual_force.numpy(), 0.0)

    def test_reset_shared_global_and_local_solve_world(self):
        """Reset source state selectively but invalidate the warm starts of its complete solve world."""
        for reset_global in (False, True):
            with self.subTest(reset_global=reset_global):
                model, _, body = _build_global_cloth_local_rigid_model(self.default_device)
                solver = SolverKamino(model, config=_make_lox_config())
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

                lox = _lox(solver)
                lox.data.dual_wrench.fill_(1.0)
                lox.data.particle_dual_force.fill_(1.0)
                world_mask = wp.array([not reset_global, reset_global], dtype=wp.bool, device=self.default_device)
                success_mask = wp.empty(model.world_count, dtype=wp.bool, device=self.default_device)

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
                # The global particles share the first solve world with the local bodies and particles
                np.testing.assert_array_equal(lox.data.dual_wrench.numpy(), 0.0)
                np.testing.assert_array_equal(lox.data.particle_dual_force.numpy(), 0.0)
                np.testing.assert_array_equal(success_mask.numpy(), [not reset_global])

    def test_step_global_cloth_contact_with_local_rigid_body(self):
        """Couple global cloth to a dynamic body in its LOX solve world."""
        model, shape, body = _build_global_cloth_local_rigid_model(self.default_device)
        state_in = model.state()
        state_out = model.state()
        contacts = _make_particle_contact(model, state_in, shape, gap=-0.01)
        solver = SolverKamino(model, config=_make_lox_config(max_iterations=50))

        solver.step(state_in, state_out, None, contacts, 0.01)

        rows = _lox(solver).deformables.contact_rows
        self.assertEqual(int(model.particle_world.numpy()[0]), -1)
        self.assertEqual(int(model.body_world.numpy()[body]), 0)
        np.testing.assert_array_equal(rows.status.numpy(), [ParticleContactStatus.ACTIVE])
        self.assertGreater(float(np.linalg.norm(state_out.particle_qd.numpy()[0])), 0.0)
        self.assertGreater(float(np.linalg.norm(state_out.body_qd.numpy()[body])), 0.0)

    def test_step_mixed_rigid_and_cloth_without_coupling(self):
        """Advance rigid and cloth state together without dynamic coupling."""
        model, _, _ = _build_contact_model(device=self.default_device, collider="dynamic")
        state_in = model.state()
        state_out = model.state()
        solver = SolverKamino(model, config=_make_lox_config())

        solver.step(state_in, state_out, None, None, 0.01)

        self.assertTrue(np.all(np.isfinite(state_out.body_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.body_qd.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))
        np.testing.assert_array_equal(solver.status.numpy()["converged"], [1])

    def test_cross_world_mixed_contact_fails_only_its_world(self):
        """Fail only the world of a mixed contact whose collider belongs to another world."""
        for max_colors in (1, 4):
            with self.subTest(max_colors=max_colors):
                model, shapes, bodies = _build_contact_model(
                    device=self.default_device, world_count=2, collider="dynamic"
                )
                state_in = model.state()
                state_in.particle_qd.fill_((0.0, 0.0, -1.0))
                particle_world = model.particle_world.numpy()
                particles = [int(np.flatnonzero(particle_world == world)[0]) for world in range(model.world_count)]
                normal = np.array((0.0, 0.0, 1.0), dtype=np.float32)
                positions = state_in.particle_q.numpy()
                records = [
                    {
                        "indices": (particle, -1, -1),
                        "barycentric": (1.0, 0.0, 0.0),
                        "shape": shapes[1],
                        "body_position": _surface_position_for_gap(
                            model, positions, (particle, -1, -1), (1.0, 0.0, 0.0), shapes[1], 0.0, normal
                        ),
                        "normal": normal,
                    }
                    for particle in particles
                ]
                contacts = _make_contacts(device=self.default_device, capacity=2, records=records)
                state_out = model.state()
                solver = SolverKamino(model, config=_make_lox_config(gauss_seidel_max_colors=max_colors))

                solver.step(state_in, state_out, None, contacts, 0.01)

                rows = _lox(solver).deformables.contact_rows
                np.testing.assert_array_equal(
                    rows.status.numpy(),
                    [ParticleContactStatus.CROSS_WORLD, ParticleContactStatus.ACTIVE],
                )
                np.testing.assert_array_equal(solver.status.numpy()["failed"], [1, 0])
                failed_world = particle_world == 0
                np.testing.assert_array_equal(
                    state_out.particle_q.numpy()[failed_world], state_in.particle_q.numpy()[failed_world]
                )
                np.testing.assert_array_equal(
                    state_out.particle_qd.numpy()[failed_world], state_in.particle_qd.numpy()[failed_world]
                )
                self.assertGreater(float(state_out.particle_qd.numpy()[particles[1], 2]), -1.0)
                self.assertLess(float(state_out.body_qd.numpy()[bodies[1], 2]), 0.0)

    def test_nonfinite_mixed_body_correction_fails_world(self):
        """Fail the world of a mixed contact whose body correction in the projection sweeps is non-finite."""
        for max_colors in (1, 4):
            with self.subTest(max_colors=max_colors):
                model, shapes, _ = _build_contact_model(device=self.default_device, collider="dynamic")
                state_in = model.state()
                state_in.particle_qd.fill_((0.0, 0.0, -1.0))
                state_out = model.state()
                contacts = _make_particle_contact(model, state_in, shapes[0], gap=0.0)
                solver = SolverKamino(model, config=_make_lox_config(gauss_seidel_max_colors=max_colors))
                lox = _lox(solver)
                # Binding the contacts sizes the particle contact rows and builds the projection schedule
                lox.bind_particle_step(state_in, state_out, contacts)
                response = lox.problem.data.particle_contact_rows.response
                nonfinite_response = wp.array(
                    np.full((response.shape[0], 3, 6), np.nan, dtype=np.float32),
                    dtype=mat36f,
                    device=self.default_device,
                )
                projection = lox._projection
                original_project = projection.project

                def project_with_nonfinite_body_response(
                    *args,
                    original_project=original_project,
                    projection=projection,
                    nonfinite_response=nonfinite_response,
                ):
                    # The sweeps read the body responses that prepare() stores for the time step
                    projection._weights.particle_contact_response = nonfinite_response
                    original_project(*args)

                projection.project = project_with_nonfinite_body_response

                solver.step(state_in, state_out, None, contacts, 0.01)

                np.testing.assert_array_equal(solver.status.numpy()["failed"], [1])
                np.testing.assert_array_equal(state_out.particle_q.numpy(), state_in.particle_q.numpy())
                np.testing.assert_array_equal(state_out.particle_qd.numpy(), state_in.particle_qd.numpy())
                np.testing.assert_array_equal(state_out.body_qd.numpy(), state_in.body_qd.numpy())
                np.testing.assert_array_equal(lox.data.dual_wrench.numpy(), 0.0)

    def test_material_failure_discards_coupled_rigid_warm_start(self):
        """Reject a material failure without retaining the coupled rigid trial state."""
        model, shapes, _ = _build_contact_model(device=self.default_device, collider="dynamic")
        solver = SolverKamino(model, config=_make_lox_config(max_iterations=5, use_graph_conditionals=False))
        lox = _lox(solver)
        state_in, state_out = model.state(), model.state()
        velocity = state_in.particle_qd.numpy()
        velocity[0] = (0.3, -0.1, -1.0)
        state_in.particle_qd.assign(velocity)
        contacts = _make_particle_contact(model, state_in, shapes[0], gap=-0.005)
        # Seed the solver-owned warm starts of the contacted particle and of the coupled body
        particle_dual_force = np.zeros((model.particle_count, 3), dtype=np.float32)
        particle_dual_force[int(lox.deformables.topology.newton_to_packed.numpy()[0])] = (2.0, -1.0, 3.0)
        lox.data.particle_dual_force.assign(particle_dual_force)
        lox.data.dual_wrench.fill_(2.0)

        deformables = lox.deformables
        original_update = deformables.update_proximal
        calls = 0

        def fail_material(time_step):
            nonlocal calls
            original_update(time_step)
            calls += 1
            if calls == 3:
                deformables.system_data.proximal_failed.fill_(True)

        deformables.update_proximal = fail_material
        solver.step(state_in, state_out, None, contacts, 0.01)

        self.assertGreaterEqual(calls, 3)
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [1])
        np.testing.assert_array_equal(state_out.particle_q.numpy(), state_in.particle_q.numpy())
        np.testing.assert_array_equal(state_out.particle_qd.numpy(), velocity)
        np.testing.assert_array_equal(lox.data.particle_dual_force.numpy(), particle_dual_force)
        np.testing.assert_array_equal(lox.data.dual_wrench.numpy(), 0.0)
        np.testing.assert_array_equal(state_out.body_qd.numpy(), state_in.body_qd.numpy())

        # The world stays failed until a reset, after which the discarded trial leaves the next
        # healthy step unaffected
        deformables.update_proximal = original_update
        healthy_state = model.state()
        solver.step(healthy_state, state_out, None, None, 0.01)
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [1])
        solver.reset(healthy_state)
        solver.step(healthy_state, state_out, None, None, 0.01)
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])
        self.assertTrue(np.isfinite(state_out.body_qd.numpy()).all())

    def test_assembly_failure_survives_proximal_resets(self):
        """Keep an assembly rejection terminal with enabled and disabled proximal updates."""
        for proximal_iterations in (0, 1):
            with self.subTest(proximal_iterations=proximal_iterations):
                model, _, _ = _build_contact_model(device=self.default_device)
                solver = SolverKamino(
                    model,
                    config=_make_lox_config(max_iterations=3, deformable_proximal_iterations=proximal_iterations),
                )
                state_in, state_out = model.state(), model.state()
                positions = state_in.particle_q.numpy()
                positions[:, :2] *= 1.0e10
                state_in.particle_q.assign(positions)
                solver.step(state_in, state_out, None, None, 0.01)
                system_data = _lox(solver).deformables.system_data

                np.testing.assert_array_equal(system_data.assembly_failed.numpy(), [True])
                np.testing.assert_array_equal(solver.status.numpy()["failed"], [1])
                np.testing.assert_array_equal(state_out.particle_q.numpy(), positions)
                np.testing.assert_array_equal(state_out.particle_qd.numpy(), state_in.particle_qd.numpy())

                # The world stays failed until a reset; a valid step then clears the assembly failure
                valid_state = model.state()
                solver.step(valid_state, state_out, None, None, 0.01)
                np.testing.assert_array_equal(solver.status.numpy()["failed"], [1])
                solver.reset(valid_state)
                solver.step(valid_state, state_out, None, None, 0.01)
                np.testing.assert_array_equal(system_data.assembly_failed.numpy(), [False])
                np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])

    def test_public_tetrahedron_preserves_inversion_tolerant_behavior(self):
        """Advance regular, singular, and inverted tetrahedra without imposing a volume barrier."""
        for height in (1.0, 0.0, -0.2):
            with self.subTest(height=height):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                for position in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    builder.add_particle(pos=wp.vec3(*position), vel=wp.vec3(0.0), mass=1.0)
                builder.add_tetrahedron(0, 1, 2, 3, k_mu=10.0, k_lambda=20.0)
                model = builder.finalize(device=self.default_device)
                solver = SolverKamino(model, config=_make_lox_config())
                state_in, state_out = model.state(), model.state()
                positions = state_in.particle_q.numpy()
                positions[3, 2] = height
                state_in.particle_q.assign(positions)
                for _ in range(5):
                    solver.step(state_in, state_out, None, None, 0.001)
                    np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])
                    self.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())
                    self.assertTrue(np.isfinite(state_out.particle_qd.numpy()).all())
                    state_in, state_out = state_out, state_in
                if height < 0.0:
                    final_positions = state_in.particle_q.numpy()
                    deformation = (final_positions[1:] - final_positions[0]).T
                    self.assertLess(float(np.linalg.det(deformation)), 0.0)

    def test_nonfinite_rod_rejects_coupled_trial_without_particle_warmstart_leak(self):
        """Reject a non-finite rod update without leaking the particle trial state."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverKamino.register_custom_attributes(builder)
        builder.begin_world()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        shape = builder.add_shape_sphere(body, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        rod = builder.add_joint_rod(-1, body, bend_stiffness=0.0, bend_damping=10000.0)
        builder.add_articulation([rod])
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.1),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, -0.1),
            dim_x=1,
            dim_y=1,
            cell_x=0.1,
            cell_y=0.1,
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
        builder.end_world()
        model = builder.finalize(device=self.default_device)
        config = SolverKamino.Config(dynamics_solver="lox", sparse_jacobian=True, use_collision_detector=False)
        config.lox.rod_proximal_relaxation = 1.0
        config.lox.use_graph_conditionals = False
        config.lox.max_iterations = 5
        solver = SolverKamino(model, config=config)
        lox = _lox(solver)
        state_in, state_out = model.state(), model.state()
        poses = state_in.body_q.numpy()
        poses[body, 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5))
        state_in.body_q.assign(poses)
        state_in.body_qd.assign([[0.0, 0.0, 0.0, 10.0, 0.0, 0.0]])
        # Only the contacted particle has an active consensus warm start
        particle_dual_force = np.zeros((model.particle_count, 3), dtype=np.float32)
        particle_dual_force[int(lox.deformables.topology.newton_to_packed.numpy()[0])] = (2.0, -1.0, 3.0)
        lox.data.particle_dual_force.assign(particle_dual_force)
        contacts = _make_particle_contact(model, state_in, shape, gap=-0.005)
        initial_particles = state_in.particle_q.numpy()
        initial_velocities = state_in.particle_qd.numpy()
        rods = lox.problem.rods
        original_update = rods.update_proximal
        calls = 0

        def inject_nonfinite_multiplier(*args):
            nonlocal calls
            calls += 1
            if calls == 3:
                self.assertFalse(bool(lox.data.state.world_failed.numpy()[0]))
                multiplier = rods.data.multiplier.numpy()
                multiplier[3] = np.nan
                rods.data.multiplier.assign(multiplier)
            original_update(*args)

        rods.update_proximal = inject_nonfinite_multiplier
        solver.step(state_in, state_out, model.control(), contacts, 0.01)

        status = solver.status.numpy()
        np.testing.assert_array_equal(status["failed"], [1])
        np.testing.assert_array_equal(status["iterations"], [3])
        np.testing.assert_array_equal(state_out.particle_q.numpy(), initial_particles)
        np.testing.assert_array_equal(state_out.particle_qd.numpy(), initial_velocities)
        np.testing.assert_array_equal(lox.data.particle_dual_force.numpy(), particle_dual_force)
        np.testing.assert_allclose(state_out.body_qd.numpy(), state_in.body_qd.numpy(), atol=1.0e-5)
        self.assertTrue(np.isfinite(state_out.body_q.numpy()).all())

    def test_capture_public_pure_cloth_step(self):
        """Capture and replay the public pure-cloth LOX step."""
        self._require_graph_capture()
        model, _, _ = _build_contact_model(device=self.default_device)
        solver = SolverKamino(model, config=_make_lox_config())
        state_in = model.state()
        state_out = model.state()
        solver.step(state_in, state_out, None, None, 0.01)
        state_in, state_out = state_out, state_in

        with wp.ScopedCapture(device=self.default_device) as capture:
            solver.step(state_in, state_out, None, None, 0.01)
        wp.capture_launch(capture.graph)

        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))

    def test_capture_public_nonlinear_membrane_step(self):
        """Capture and replay nonlinear membrane proximal updates."""
        self._require_graph_capture()
        model, _, _ = _build_contact_model(device=self.default_device)
        solver = SolverKamino(
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

        with wp.ScopedCapture(device=self.default_device) as capture:
            solver.step(state_in, state_out, None, None, 0.01)
        wp.capture_launch(capture.graph)

        membrane_proximal = _lox(solver).deformables.system.membrane_proximal
        self.assertIsNotNone(membrane_proximal)
        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))
        self.assertTrue(np.all(np.isfinite(membrane_proximal.data.multiplier.numpy())))

    def test_capture_public_dynamic_rigid_cloth_contact_step(self):
        """Capture and replay a two-way rigid-cloth contact step."""
        self._require_graph_capture()
        model, shapes, _ = _build_contact_model(device=self.default_device, collider="dynamic")
        solver = SolverKamino(model, config=_make_lox_config())
        state_in = model.state()
        state_out = model.state()
        state_in.particle_qd.fill_((0.0, 0.0, -1.0))
        contacts = _make_particle_contact(model, state_in, shapes[0], gap=0.0)
        solver.step(state_in, state_out, None, contacts, 0.01)
        state_in, state_out = state_out, state_in

        with wp.ScopedCapture(device=self.default_device) as capture:
            solver.step(state_in, state_out, None, contacts, 0.01)
        wp.capture_launch(capture.graph)

        self.assertTrue(np.all(np.isfinite(state_out.body_qd.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])

    def test_capture_alternative_mixed_contact_projections(self):
        """Capture every alternative mixed-contact projection schedule."""
        self._require_graph_capture()
        for max_colors, acceleration in ((4, "none"), (1, "apgd"), (1, "anderson"), (4, "anderson")):
            with self.subTest(max_colors=max_colors, acceleration=acceleration):
                model, shapes, _ = _build_contact_model(device=self.default_device, collider="dynamic")
                state_in = model.state()
                state_out = model.state()
                state_in.particle_qd.fill_((0.0, 0.0, -1.0))
                contacts = _make_particle_contact(model, state_in, shapes[0], gap=0.0)
                solver = SolverKamino(
                    model,
                    config=_make_lox_config(
                        projection_iterations=8,
                        projection_acceleration=acceleration,
                        gauss_seidel_max_colors=max_colors,
                    ),
                )
                solver.step(state_in, state_out, None, contacts, 0.01)
                state_in, state_out = state_out, state_in

                with wp.ScopedCapture(device=self.default_device) as capture:
                    solver.step(state_in, state_out, None, contacts, 0.01)
                wp.capture_launch(capture.graph)

                self.assertTrue(np.all(np.isfinite(state_out.body_qd.numpy())))
                self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))

    def test_capture_public_first_dynamic_rigid_cloth_contact_step(self):
        """Capture the first public rigid-cloth contact step without a warmup step."""
        self._require_graph_capture()
        if not self.default_device.is_mempool_enabled:
            self.skipTest("First-step CUDA capture allocation requires the Warp memory pool.")
        model, shapes, _ = _build_contact_model(device=self.default_device, collider="dynamic")
        solver = SolverKamino(model, config=_make_lox_config())
        state_in = model.state()
        state_out = model.state()
        state_in.particle_qd.fill_((0.0, 0.0, -1.0))
        contacts = _make_particle_contact(model, state_in, shapes[0], gap=0.0)

        with wp.ScopedCapture(device=self.default_device) as capture:
            solver.step(state_in, state_out, None, contacts, 0.01)
        wp.capture_launch(capture.graph)
        wp.capture_launch(capture.graph)

        rows = _lox(solver).deformables.contact_rows
        self.assertEqual(rows.rigid_capacity, contacts.soft_contact_max)
        np.testing.assert_array_equal(rows.status.numpy(), [ParticleContactStatus.ACTIVE])
        self.assertTrue(np.all(np.isfinite(state_out.body_qd.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])

    def test_capture_public_full_surface_contact_step(self):
        """Capture collision generation and a two-way cloth-face contact step."""
        self._require_graph_capture()
        model, _ = _build_full_surface_dynamic_contact_model(self.default_device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            soft_contact_gap=0.1,
            enable_rigid_soft_full_surface_contact=True,
        )
        contacts = pipeline.contacts()
        solver = SolverKamino(model, config=_make_lox_config())
        state_in = model.state()
        state_out = model.state()

        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, 0.01)
        solver.reset(state_in)

        with wp.ScopedCapture(device=self.default_device) as capture:
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, None, contacts, 0.01)
        wp.capture_launch(capture.graph)

        contact_count = int(contacts.soft_contact_count.numpy()[0])
        contact_indices = contacts.soft_contact_indices.numpy()[:contact_count]
        self.assertGreater(contact_count, 0)
        self.assertTrue(np.any(contact_indices[:, 1] >= 0))
        self.assertTrue(np.all(np.isfinite(state_out.body_qd.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.particle_qd.numpy())))
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])

    def test_joint_penalty_scale_seed_before_first_step(self):
        """Seed the joint penalty scale of a jointed body next to cloth before the first step."""
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        builder.begin_world()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        joint = builder.add_joint_revolute(-1, body, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([joint])
        builder.add_cloth_grid(
            pos=wp.vec3(1.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            fix_left=True,
            tri_ke=100.0,
            tri_ka=80.0,
            tri_kd=0.0,
            edge_ke=2.0,
        )
        builder.end_world()
        model = builder.finalize(device=self.default_device)
        config = SolverKamino.Config(dynamics_solver="lox", use_collision_detector=False)
        config.lox.use_graph_conditionals = False
        solver = SolverKamino(model, config=config)

        scale = solver.lox_joint_penalty_scale_seed(1.0 / 60.0).numpy()
        self.assertTrue(np.all(np.isfinite(scale)))
        self.assertTrue(np.all(scale > 0.0))

        state_in, state_out = model.state(), model.state()
        solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])
        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
