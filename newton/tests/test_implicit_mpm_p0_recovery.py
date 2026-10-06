# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp
import warp.fem as fem

import newton
from newton._src.solvers.implicit_mpm.implicit_mpm_solver_kernels import compute_density_strain_offset
from newton._src.solvers.implicit_mpm.rasterized_collisions import stabilize_collider_velocity
from newton._src.solvers.implicit_mpm.solve_rheology import _run_solver_loop
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _count_iteration(count: wp.array[int]):
    count[0] += 1


def test_iteration_budget(test, device):
    """Execute the exact iteration budget with and without a CUDA graph."""
    with wp.ScopedDevice(device):
        for use_graph in (False, True) if wp.get_device(device).is_cuda else (False,):
            for budget in (0, 1, 3, 7, 10):
                with test.subTest(use_graph=use_graph, budget=budget):
                    count = wp.zeros(1, dtype=int, device=device)
                    residual = wp.ones((2, 1), dtype=float, device=device)

                    class Counter:
                        name = "counter"
                        solve_granularity = 5
                        strain_environment_offsets = None

                        def __init__(self, count, residual):
                            self.device = wp.get_device(device)
                            self.rheology = self
                            self.count = count
                            self.residual = residual

                        def solve(self):
                            wp.launch(_count_iteration, dim=1, inputs=[self.count], device=device)

                        def eval_residual(self):
                            return self.residual

                    counter = Counter(count, residual)
                    graph = _run_solver_loop(counter, counter, budget, 0.0, 1.0, use_graph, False, fem.TemporaryStore())
                    test.assertEqual(int(count.numpy()[0]), 2 * budget)
                    del graph


def test_signed_volume_offset(test, device):
    """Preserve compression errors and subtract collider volume in the recovery target."""
    with wp.ScopedDevice(device):
        particles = wp.array([1.2, 0.8, 1.0, 0.3], dtype=float, device=device)
        colliders = wp.array([0.0, 0.0, 0.0, 0.2], dtype=float, device=device)
        volumes = wp.ones(4, dtype=float, device=device)
        offset = wp.zeros(4, dtype=float, device=device)
        wp.launch(
            compute_density_strain_offset, dim=4, inputs=[0.05, particles, colliders, volumes, offset], device=device
        )
        np.testing.assert_allclose(offset.numpy(), [-0.01, 0.01, 0.0, 0.025], atol=2.0e-8)


def test_contact_target(test, device):
    """Correct penetration, allow gap closure, and preserve tangential collider motion."""
    with wp.ScopedDevice(device):
        sdf = wp.array([-0.02, 0.05, 0.0, -0.02], dtype=float, device=device)
        normals = wp.array([(0, 0, 1)] * 3 + [(0, 0, 0)], dtype=wp.vec3, device=device)
        velocity = wp.array([(0.2, -0.3, 0.1)] * 4, dtype=wp.vec3, device=device)
        wp.launch(stabilize_collider_velocity, dim=4, inputs=[0.1, 0.25, sdf, normals, velocity], device=device)
        np.testing.assert_allclose(
            velocity.numpy(), [(0.2, -0.3, 0.15), (0.2, -0.3, -0.4), (0.2, -0.3, 0.1), (0.2, -0.3, 0.1)], atol=1.0e-7
        )


def _make_contact_solver(
    device, scheme, height, velocity, fraction, gap, *, grid_type="dense", max_active_cell_count=-1
):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle(pos=(0.5, 0.5, height), vel=velocity, mass=1.0, radius=0.1)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
    model = builder.finalize(device=device)
    config = SolverImplicitMPM.Config(
        voxel_size=1.0,
        grid_type=grid_type,
        max_active_cell_count=max_active_cell_count,
        strain_basis="P0",
        collider_basis="pic8",
        integration_scheme=scheme,
        solver="jacobi",
        max_iterations=1,
        tolerance=0.0,
        warmstart_mode="none",
        air_drag=1.0e-6,
        collider_stabilization_fraction=fraction,
        collider_contact_gap=gap,
    )
    return model, SolverImplicitMPM(model, config=config)


def test_contact_recovery_truncated(test, device):
    """Recover penetration over successive one-iteration P0 steps for both transfer paths."""
    with wp.ScopedDevice(device):
        for scheme in ("pic", "cell"):
            with test.subTest(scheme=scheme):
                model, solver = _make_contact_solver(device, scheme, -0.02, (0.0, 0.0, 0.0), 0.25, 0.1)
                state = model.state()
                for _ in range(8):
                    solver.step(state, state, None, None, 0.1)
                height = state.particle_q.numpy()[0, 2]
                test.assertGreater(height, -0.003)
                test.assertLess(height, 0.03)


def test_predictive_contact_truncated(test, device):
    """Stop approaching PIC particles at the surface using a single P0 iteration."""
    with wp.ScopedDevice(device):
        for scheme in ("pic", "cell"):
            with test.subTest(scheme=scheme):
                model, solver = _make_contact_solver(device, scheme, 0.05, (0.2, 0.0, -1.0), 0.25, 0.1)
                state = model.state()
                solver.step(state, state, None, None, 0.1)
                test.assertAlmostEqual(float(state.particle_q.numpy()[0, 2]), 0.0, delta=2.0e-5)
                test.assertAlmostEqual(float(state.particle_qd.numpy()[0, 0]), 0.2, delta=2.0e-3)


def _make_volume_solver(device, scheme, fraction, *, residual_fraction=0.0):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    coordinates = np.linspace(1.0 / 6.0, 5.0 / 6.0, 3) if scheme == "pic" else np.linspace(0.45, 0.55, 3)
    positions = np.array(np.meshgrid(coordinates, coordinates, coordinates)).reshape(3, -1).T
    for position in positions:
        builder.add_particle(pos=position, vel=(0.0, 0.0, 0.0), mass=1200.0 / 27.0, radius=1.2 ** (1.0 / 3.0) / 6.0)
    model = builder.finalize(device=device)
    config = SolverImplicitMPM.Config(
        voxel_size=1.0,
        grid_type="fixed",
        grid_padding=1,
        strain_basis="P0",
        warmstart_mode="none",
        integration_scheme=scheme,
        solver="gs",
        max_iterations=3,
        tolerance=0.0,
        density_strain_fraction=fraction,
        residual_strain_fraction=residual_fraction,
    )
    return model, SolverImplicitMPM(model, config=config)


def test_volume_recovery_truncated(test, device):
    """Expand overpacked material with a short P0 solve and retain the sparse compliance path."""
    with wp.ScopedDevice(device):
        for scheme in ("pic", "cell"):
            extents = []
            for fraction in (0.0, 0.1):
                with test.subTest(scheme=scheme, fraction=fraction):
                    model, solver = _make_volume_solver(device, scheme, fraction)
                    state = model.state()
                    for _ in range(5):
                        solver.step(state, state, None, None, 0.02)
                    positions = state.particle_q.numpy()
                    test.assertTrue(np.isfinite(positions).all())
                    extents.append(np.prod(np.ptp(positions, axis=0)))
                    test.assertEqual(solver._scratchpad.compliance_matrix.nnz, 0)
                    test.assertFalse(solver.residual_strain_tracking)
            test.assertGreater(extents[1], 1.02 * extents[0])


def test_recovery_config_validation(test, device):
    """Reject invalid recovery fractions and unsupported collider bases."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.1)
        model = builder.finalize(device=device)
        for options in (
            {"density_strain_fraction": float("nan")},
            {"density_strain_fraction": -0.1},
            {"residual_strain_fraction": 1.1},
            {"collider_stabilization_fraction": float("inf")},
            {"collider_contact_gap": -0.01},
            {"collider_stabilization_fraction": 0.1, "collider_basis": "S2"},
        ):
            with test.subTest(options=options), test.assertRaises(ValueError):
                SolverImplicitMPM(model, config=SolverImplicitMPM.Config(**options))


def test_residual_history_recovery_and_reset(test, device):
    """Recover recorded compression consistently with in-place and alternating states, then reset history."""
    with wp.ScopedDevice(device):
        results = []
        for inplace in (False, True):
            model, solver = _make_volume_solver(device, "pic", 0.0, residual_fraction=0.1)
            state = model.state()
            output = state if inplace else model.state()
            state.mpm.particle_residual_deformation_gradient.fill_(wp.mat33(np.eye(3) * 0.9))
            for _ in range(2):
                solver.step(state, output, None, None, 0.02)
                state, output = output, state
            history = state.mpm.particle_residual_deformation_gradient.numpy().copy()
            test.assertGreater(float(np.linalg.det(history).mean()), 0.74)
            results.append(history)
            solver.reset(state)
            np.testing.assert_allclose(
                state.mpm.particle_residual_deformation_gradient.numpy(),
                np.tile(np.eye(3), (model.particle_count, 1, 1)),
            )
        np.testing.assert_allclose(results[0], results[1], atol=2.0e-6)


def test_cell_gravity_world(test, device):
    """Apply each particle's world gravity before sharing cell-center momentum."""
    with wp.ScopedDevice(device):
        template = newton.ModelBuilder(gravity=(0.0, 0.0, -2.0))
        SolverImplicitMPM.register_custom_attributes(template)
        template.add_particle(pos=(0.5, 0.5, 0.5), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.1)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -6.0))
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.add_world(template)
        builder.add_particle(pos=(0.5, 0.5, 0.5), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.1)
        model = builder.finalize(device=device)
        config = SolverImplicitMPM.Config(
            voxel_size=1.0, grid_type="fixed", integration_scheme="cell", air_drag=1.0e-6, max_iterations=0
        )
        solver = SolverImplicitMPM(model, config=config)
        state = model.state()
        solver.step(state, state, None, None, 0.01)
        # Equal masses share the same point stencil and average the two gravities.
        np.testing.assert_allclose(state.particle_qd.numpy()[:, 2], -0.04, atol=2.0e-6)


def test_cell_contact_cuda_graph(test, device):
    """Replay cell transfers and stabilized PIC contacts within a captured full step."""
    with wp.ScopedDevice(device):
        model, solver = _make_contact_solver(
            device,
            "cell",
            -0.02,
            (0.0, 0.0, 0.0),
            0.25,
            0.1,
            grid_type="fixed",
            max_active_cell_count=512,
        )
        state = model.state()
        solver.step(state, state, None, None, 0.1)
        with wp.ScopedCapture() as capture:
            solver.step(state, state, None, None, 0.1)
        for _ in range(4):
            wp.capture_launch(capture.graph)
        positions = state.particle_q.numpy()
        test.assertTrue(np.isfinite(positions).all())
        test.assertGreater(float(positions[0, 2]), -0.003)


class TestImplicitMPMP0Recovery(unittest.TestCase):
    pass


for function in (
    test_iteration_budget,
    test_signed_volume_offset,
    test_contact_target,
    test_contact_recovery_truncated,
    test_predictive_contact_truncated,
    test_volume_recovery_truncated,
    test_recovery_config_validation,
    test_residual_history_recovery_and_reset,
    test_cell_gravity_world,
):
    add_function_test(TestImplicitMPMP0Recovery, function.__name__, function, devices=get_test_devices(mode="basic"))

add_function_test(
    TestImplicitMPMP0Recovery,
    "test_cell_contact_cuda_graph",
    test_cell_contact_cuda_graph,
    devices=get_cuda_test_devices(),
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
