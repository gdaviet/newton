# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp
import warp.fem as fem

import newton
from newton._src.solvers.implicit_mpm.cell_quadrature import CellQuadrature
from newton._src.solvers.implicit_mpm.implicit_mpm_solver_kernels import compute_eigenvalues
from newton._src.solvers.implicit_mpm.rasterized_collisions import (
    _ALL_COLLIDER_WORLDS,
    Collider,
    collision_sdf,
    rasterize_collider_kernel,
)
from newton._src.solvers.implicit_mpm.rheology_solver_kernels import YieldParamVec
from newton._src.solvers.implicit_mpm.solve_rheology import (
    _ITERATIVE_LINEAR_SOLVERS,
    ArraySquaredNorm,
    _compute_environment_l2_tolerance_scales,
    _linear_solver_result_norms,
    _nonlinear_solver_result_norms,
    update_batched_condition,
)
from newton.solvers import SolverImplicitMPM, SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _make_mpm_particle_builder(
    gravity=(0.0, -9.81, 0.0),
    velocity=(0.0, 0.0, 0.0),
    young_modulus=1.0e4,
    initial_plastic_volume_strain=1.0,
    dimensions=(2, 2, 2),
):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=gravity)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle_grid(
        pos=wp.vec3(0.025, 0.025, 0.025),
        rot=wp.quat_identity(),
        vel=wp.vec3(velocity),
        dim_x=dimensions[0],
        dim_y=dimensions[1],
        dim_z=dimensions[2],
        cell_x=0.05,
        cell_y=0.05,
        cell_z=0.05,
        mass=0.01,
        jitter=0.0,
        radius_mean=0.025,
        custom_attributes={
            "mpm:young_modulus": young_modulus,
            "mpm:poisson_ratio": 0.2,
            "mpm:particle_Jp": initial_plastic_volume_strain,
        },
    )
    return builder


def _make_mpm_config(grid_type="dense", integration_scheme="pic", solver="jacobi"):
    config = SolverImplicitMPM.Config()
    config.separate_worlds = True
    config.grid_type = grid_type
    config.voxel_size = 0.1
    config.integration_scheme = integration_scheme
    config.solver = solver
    config.max_iterations = 4
    config.tolerance = 0.0
    config.warmstart_mode = "grid"
    return config


def test_expanded_iterative_solver_names(test, device):
    """Verify expanded iterative solver names resolve to the existing implementations."""
    del device
    test.assertIs(_ITERATIVE_LINEAR_SOLVERS["conjugate-gradient"], _ITERATIVE_LINEAR_SOLVERS["cg"])
    test.assertIs(_ITERATIVE_LINEAR_SOLVERS["conjugate-residual"], _ITERATIVE_LINEAR_SOLVERS["cr"])
    test.assertIs(_ITERATIVE_LINEAR_SOLVERS["generalized-minimal-residual"], _ITERATIVE_LINEAR_SOLVERS["gmres"])


def _make_two_world_particle_model(device, builder=None, local_builder=None):
    if builder is None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    if local_builder is None:
        local_builder = _make_mpm_particle_builder(gravity=(0.0, 0.0, 0.0))
    builder.add_world(local_builder)
    builder.add_world(local_builder)
    return builder.finalize(device=device)


@wp.kernel
def _query_collision_sdf(
    positions: wp.array[wp.vec3],
    environment_indices: wp.array[int],
    collider: Collider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    collider_ids: wp.array[int],
    material_ids: wp.array[int],
    friction: wp.array[float],
    adhesion: wp.array[float],
    projection_threshold: wp.array[float],
):
    i = wp.tid()
    _sdf, _normal, _velocity, collider_id, material_id = collision_sdf(
        positions[i], environment_indices[i], collider, body_q, body_qd, body_q_prev, 0.01
    )
    collider_ids[i] = collider_id
    material_ids[i] = material_id
    friction[i] = collider.material_friction[material_id]
    adhesion[i] = collider.material_adhesion[material_id]
    projection_threshold[i] = collider.material_projection_threshold[material_id]


def _make_box_collider_mesh(device, half_extent=0.5, center=(0.0, 0.0, 0.0)):
    box = newton.Mesh.create_box(
        half_extent,
        half_extent,
        half_extent,
        duplicate_vertices=False,
        compute_normals=False,
        compute_uvs=False,
        compute_inertia=False,
    )
    points = wp.array(box.vertices + np.asarray(center), dtype=wp.vec3, device=device)
    indices = wp.array(box.indices, dtype=int, device=device)
    return wp.Mesh(points, indices, wp.zeros_like(points))


def _step_mpm(model, config, step_count=3, dt=0.01):
    solver = SolverImplicitMPM(model, config=config)
    state_0 = model.state()
    state_1 = model.state()
    for _ in range(step_count):
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0
    return solver, state_0


def _compressive_shear_velocity(positions, amplitude):
    centered = positions - np.mean(positions, axis=0)
    return amplitude * np.column_stack(
        (
            -1.0 * centered[:, 0] + 0.4 * centered[:, 1],
            -0.7 * centered[:, 1] + 0.3 * centered[:, 2],
            -0.5 * centered[:, 2] + 0.6 * centered[:, 0],
        )
    )


def test_array_squared_norm_batches(test, device):
    """Verify array squared norm batches."""
    values = np.arange(1, 524, dtype=np.float32)
    data = wp.array(values, dtype=float, device=device)
    offsets = wp.array((0, 2, 2, 523), dtype=int, device=device)
    norm = ArraySquaredNorm(max_length=523, batch_offsets=offsets, device=device)

    try:
        result = norm.compute_squared_norm(data)
        test.assertEqual(result.shape, (2, 3))
        result_snapshot = result.numpy().copy()
        np.testing.assert_array_equal(result_snapshot[0], np.array((3.0, 0.0, 137023.0)))
        np.testing.assert_array_equal(result_snapshot[1], np.array((2.0, 0.0, 523.0)))

        result_ptr = result.ptr
        sum_values = np.ones(523, dtype=np.float32)
        max_values = np.full(523, 25.0, dtype=np.float32)
        max_values[:2] = (4.0, 7.0)
        two_row_data = wp.array(np.stack((sum_values, max_values)), dtype=float, device=device)
        result = norm.compute_squared_norm(two_row_data)

        test.assertEqual(result.ptr, result_ptr)
        result_snapshot = result.numpy().copy()
        np.testing.assert_array_equal(result_snapshot[0], np.array((2.0, 0.0, 521.0)))
        np.testing.assert_array_equal(result_snapshot[1], np.array((7.0, 0.0, 25.0)))
    finally:
        norm.release()


def test_linear_solver_result_norms(test, device):
    """Verify linear solver result norms."""
    residual, atol = _linear_solver_result_norms(4.0, 2.0, use_graph=False)
    test.assertEqual(residual, 4.0)
    test.assertEqual(atol, 2.0)

    residual_sq = wp.array((9.0, 25.0), dtype=float, device=device)
    atol_sq = wp.array((4.0, 16.0), dtype=float, device=device)
    residual, atol = _linear_solver_result_norms(residual_sq, atol_sq, use_graph=True)
    test.assertEqual(residual, 5.0)
    test.assertEqual(atol, 4.0)


def test_multiworld_residual_tolerance_scales(test, device):
    """Verify multi-world residual tolerance scales."""
    offsets = wp.array((0, 3, 3, 12), dtype=int, device=device)
    scales = wp.empty(3, dtype=float, device=device)
    wp.launch(
        _compute_environment_l2_tolerance_scales,
        dim=3,
        inputs=[offsets],
        outputs=[scales],
        device=device,
    )

    expected_scales = np.sqrt(np.array((4.0, 1.0, 10.0), dtype=np.float32))
    np.testing.assert_allclose(scales.numpy(), expected_scales)

    residual = np.array(((3.6, 0.9, 9.0), (0.25, 0.5, 0.75)), dtype=np.float32)
    l2_norm, linf_norm = _nonlinear_solver_result_norms(residual, expected_scales)
    test.assertAlmostEqual(l2_norm, np.sqrt(0.9), places=6)
    test.assertAlmostEqual(linf_norm, np.sqrt(0.75), places=6)

    residual_device = wp.array(residual, dtype=float, device=device)
    iteration = wp.zeros(1, dtype=int, device=device)
    condition = wp.ones(1, dtype=int, device=device)
    wp.launch(
        update_batched_condition,
        dim=1,
        inputs=[1.0, scales, 5, 100, residual_device, iteration, condition],
        device=device,
    )
    test.assertEqual(condition.numpy()[0], 0)

    residual[0, 1] = 1.1
    residual_device.assign(residual)
    iteration.zero_()
    condition.fill_(1)
    wp.launch(
        update_batched_condition,
        dim=1,
        inputs=[1.0, scales, 5, 100, residual_device, iteration, condition],
        device=device,
    )
    test.assertEqual(condition.numpy()[0], 1)


def test_multiworld_cr_matches_independent(test, device):
    """Verify multi-world CR matches independent."""
    young_moduli = (2.5e3, 4.0e4)
    velocity_amplitudes = (3.0, 11.0)
    particle_dimensions = ((2, 2, 2), (3, 2, 2))
    reference_states = []
    reference_initial_q = []
    reference_initial_qd = []

    config = _make_mpm_config(grid_type="dense", integration_scheme="pic", solver="cr")
    config.max_iterations = 20
    config.tolerance = 1.0e-5
    config.warmstart_mode = "none"

    for young_modulus, velocity_amplitude, dimensions in zip(
        young_moduli, velocity_amplitudes, particle_dimensions, strict=True
    ):
        reference_model = _make_mpm_particle_builder(
            gravity=(0.0, 0.0, 0.0),
            young_modulus=young_modulus,
            dimensions=dimensions,
        ).finalize(device=device)
        initial_q = reference_model.particle_q.numpy()
        initial_qd = _compressive_shear_velocity(initial_q, velocity_amplitude)
        reference_model.particle_qd.assign(initial_qd)
        _, reference_state = _step_mpm(reference_model, config, step_count=2)
        reference_states.append((reference_state.particle_q.numpy(), reference_state.particle_qd.numpy()))
        reference_initial_q.append(initial_q)
        reference_initial_qd.append(initial_qd)

    populated_worlds = (0, 2)
    empty_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    empty_builder.add_body(is_kinematic=True, label="empty_world_marker")
    multiworld_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(multiworld_builder)
    for world_builder in (
        _make_mpm_particle_builder(
            gravity=(0.0, 0.0, 0.0),
            young_modulus=young_moduli[0],
            dimensions=particle_dimensions[0],
        ),
        empty_builder,
        _make_mpm_particle_builder(
            gravity=(0.0, 0.0, 0.0),
            young_modulus=young_moduli[1],
            dimensions=particle_dimensions[1],
        ),
    ):
        multiworld_builder.add_world(world_builder)
    multiworld_model = multiworld_builder.finalize(device=device)
    starts = multiworld_model.particle_world_start.numpy()
    test.assertEqual(starts[1], starts[2])
    multiworld_initial_q = multiworld_model.particle_q.numpy()
    multiworld_initial_qd = np.empty_like(multiworld_initial_q)
    for world, velocity_amplitude in zip(populated_worlds, velocity_amplitudes, strict=True):
        world_slice = slice(starts[world], starts[world + 1])
        multiworld_initial_qd[world_slice] = _compressive_shear_velocity(
            multiworld_initial_q[world_slice], velocity_amplitude
        )
    multiworld_model.particle_qd.assign(multiworld_initial_qd)

    multiworld_solver, multiworld_state = _step_mpm(multiworld_model, config, step_count=2)
    strain_offsets = multiworld_solver._scratchpad.strain_environment_offsets.numpy()
    test.assertEqual(strain_offsets[1], strain_offsets[2])
    multiworld_q = multiworld_state.particle_q.numpy()
    multiworld_qd = multiworld_state.particle_qd.numpy()

    for world, (reference_q, reference_qd), initial_q, initial_qd in zip(
        populated_worlds,
        reference_states,
        reference_initial_q,
        reference_initial_qd,
        strict=True,
    ):
        world_slice = slice(starts[world], starts[world + 1])
        world_q = multiworld_q[world_slice]
        world_qd = multiworld_qd[world_slice]
        np.testing.assert_allclose(world_q, reference_q, rtol=1.0e-5, atol=1.0e-6, equal_nan=False)
        np.testing.assert_allclose(world_qd, reference_qd, rtol=1.0e-5, atol=1.0e-6, equal_nan=False)
        test.assertTrue(np.isfinite(world_q).all())
        test.assertTrue(np.isfinite(world_qd).all())
        test.assertGreater(np.linalg.norm(reference_q - initial_q), 1.0e-4)
        test.assertGreater(np.linalg.norm(reference_qd - initial_qd), 1.0e-4)


def _run_multiworld_reference_case(device, grid_type="dense", integration_scheme="pic", solver="jacobi"):
    world_gravities = ((3.0, -2.0, 0.0), (-5.0, 1.0, 0.0))
    reference_states = []

    for world_gravity in world_gravities:
        reference_model = _make_mpm_particle_builder().finalize(device=device)
        reference_model.set_gravity(world_gravity)
        _, reference_state = _step_mpm(
            reference_model,
            _make_mpm_config(grid_type=grid_type, integration_scheme=integration_scheme, solver=solver),
        )
        reference_states.append((reference_state.particle_q.numpy(), reference_state.particle_qd.numpy()))

    local_builder = _make_mpm_particle_builder()
    multiworld_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(multiworld_builder)
    multiworld_builder.add_world(local_builder)
    multiworld_builder.add_world(local_builder)
    multiworld_model = multiworld_builder.finalize(device=device)
    for world, world_gravity in enumerate(world_gravities):
        multiworld_model.set_gravity(world_gravity, world=world)

    _, multiworld_state = _step_mpm(
        multiworld_model,
        _make_mpm_config(grid_type=grid_type, integration_scheme=integration_scheme, solver=solver),
    )
    starts = multiworld_model.particle_world_start.numpy()
    multiworld_q = multiworld_state.particle_q.numpy()
    multiworld_qd = multiworld_state.particle_qd.numpy()

    mean_velocities = []
    for world, (reference_q, reference_qd) in enumerate(reference_states):
        world_slice = slice(starts[world], starts[world + 1])
        world_q = multiworld_q[world_slice]
        world_qd = multiworld_qd[world_slice]
        np.testing.assert_allclose(world_q, reference_q, rtol=1.0e-5, atol=1.0e-6, equal_nan=False)
        np.testing.assert_allclose(world_qd, reference_qd, rtol=1.0e-5, atol=1.0e-6, equal_nan=False)
        mean_velocities.append(np.mean(world_qd, axis=0))

    mean_velocities = np.asarray(mean_velocities)
    np.testing.assert_array_equal(np.isfinite(mean_velocities), np.ones_like(mean_velocities, dtype=bool))
    np.testing.assert_array_less(np.full(2, 1.0e-3), np.abs(mean_velocities[:, 0]))
    np.testing.assert_array_equal(np.sign(mean_velocities[:, 0]), np.array((1.0, -1.0)))


def test_multiworld_dense_pic_matches_independent(test, device):
    """Verify multi-world dense PIC matches independent."""
    _run_multiworld_reference_case(device, grid_type="dense", integration_scheme="pic")


def test_multiworld_dense_gimp_matches_independent(test, device):
    """Verify multi-world dense GIMP matches independent."""
    _run_multiworld_reference_case(device, grid_type="dense", integration_scheme="gimp")


def test_multiworld_fixed_pic_matches_independent(test, device):
    """Verify multi-world fixed PIC matches independent."""
    _run_multiworld_reference_case(device, grid_type="fixed", integration_scheme="pic")


def _make_multiworld_fixed_outer_graph_case(device, max_active_cell_count=16):
    local_builder = _make_mpm_particle_builder(gravity=(0.0, 0.0, 0.0))
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(local_builder)
    builder.add_world(local_builder)
    model = builder.finalize(device=device)

    world_starts = model.particle_world_start.numpy()
    initial_q = model.particle_q.numpy()
    initial_qd = np.zeros((model.particle_count, 3), dtype=np.float32)
    for world, translation in enumerate((1.0, -1.0)):
        world_slice = slice(world_starts[world], world_starts[world + 1])
        initial_qd[world_slice] = _compressive_shear_velocity(initial_q[world_slice], amplitude=4.0)
        initial_qd[world_slice, 0] += translation
    model.particle_qd.assign(initial_qd)

    config = _make_mpm_config(grid_type="fixed", integration_scheme="pic", solver="jacobi")
    config.grid_padding = 2
    config.max_active_cell_count = max_active_cell_count
    config.max_iterations = 5
    config.tolerance = 0.0
    config.transfer_scheme = "pic"

    solver = SolverImplicitMPM(model, config=config)
    return model, solver, model.state(), model.state(), world_starts


def _mpm_particle_state_arrays(state):
    return {
        "particle_q": state.particle_q,
        "particle_qd": state.particle_qd,
        "particle_qd_grad": state.mpm.particle_qd_grad,
        "particle_elastic_strain": state.mpm.particle_elastic_strain,
        "particle_stress": state.mpm.particle_stress,
    }


def _mpm_strain_field_arrays(solver):
    scratch = solver._scratchpad
    return {
        "stress": scratch.stress_field.dof_values,
        "elastic_strain_delta": scratch.elastic_strain_delta_field.dof_values,
        "plastic_strain_delta": scratch.plastic_strain_delta_field.dof_values,
    }


def test_multiworld_fixed_capped_jacobi_matches_uncapped(test, device):
    """Verify multi-world fixed capped jacobi matches uncapped."""
    _capped_model, capped_solver, capped_state_0, capped_state_1, _capped_world_starts = (
        _make_multiworld_fixed_outer_graph_case(device, max_active_cell_count=16)
    )
    _uncapped_model, uncapped_solver, uncapped_state_0, uncapped_state_1, _uncapped_world_starts = (
        _make_multiworld_fixed_outer_graph_case(device, max_active_cell_count=-1)
    )

    for solver in (capped_solver, uncapped_solver):
        solver._use_cuda_graph = False
        solver.max_iterations = 50

    dt = 0.02
    capped_solver.step(capped_state_0, capped_state_1, control=None, contacts=None, dt=dt)
    uncapped_solver.step(uncapped_state_0, uncapped_state_1, control=None, contacts=None, dt=dt)

    capped_arrays = _mpm_particle_state_arrays(capped_state_1)
    uncapped_arrays = _mpm_particle_state_arrays(uncapped_state_1)
    for name in capped_arrays:
        capped_values = capped_arrays[name].numpy()
        uncapped_values = uncapped_arrays[name].numpy()
        test.assertTrue(np.isfinite(capped_values).all(), f"{name} is non-finite with capped Jacobi")
        np.testing.assert_allclose(
            capped_values,
            uncapped_values,
            rtol=1.0e-5,
            atol=1.0e-6,
            equal_nan=False,
            err_msg=f"{name} differs between capped and uncapped Jacobi",
        )

    test.assertGreater(np.linalg.norm(capped_arrays["particle_qd_grad"].numpy()), 1.0e-5)
    test.assertGreater(np.linalg.norm(capped_arrays["particle_elastic_strain"].numpy()), 1.0e-5)
    test.assertGreater(np.linalg.norm(capped_arrays["particle_stress"].numpy()), 1.0e-5)

    strain_field_arrays = _mpm_strain_field_arrays(capped_solver)
    test.assertEqual(strain_field_arrays["stress"].shape[0], 16)
    for name, array in strain_field_arrays.items():
        test.assertTrue(np.isfinite(array.numpy()).all(), f"Padded {name} contains non-finite values")


def test_multiworld_fixed_outer_graph_matches_eager(test, device):
    """Verify multi-world fixed outer graph matches eager."""
    if (
        not device.is_cuda
        or not device.is_mempool_supported
        or not wp.is_mempool_enabled(device)
        or not wp.is_conditional_graph_supported()
    ):
        test.skipTest("Implicit MPM CUDA capture requires memory pools and conditional graphs.")

    eager_model, eager_solver, eager_state_0, eager_state_1, world_starts = _make_multiworld_fixed_outer_graph_case(
        device
    )
    captured_model, captured_solver, captured_state_0, captured_state_1, captured_world_starts = (
        _make_multiworld_fixed_outer_graph_case(device)
    )
    np.testing.assert_array_equal(captured_world_starts, world_starts)
    initial_q = eager_state_0.particle_q.numpy().copy()
    initial_qd_grad = eager_state_0.mpm.particle_qd_grad.numpy().copy()
    initial_elastic_strain = eager_state_0.mpm.particle_elastic_strain.numpy().copy()
    initial_stress = eager_state_0.mpm.particle_stress.numpy().copy()
    for world in range(eager_model.world_count - 1):
        world_q = initial_q[world_starts[world] : world_starts[world + 1]]
        next_world_q = initial_q[world_starts[world + 1] : world_starts[world + 2]]
        np.testing.assert_array_equal(world_q, next_world_q)

    initial_offsets = captured_solver._scratchpad.strain_environment_offsets.numpy().copy()
    dt = 0.02
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        captured_solver.step(captured_state_0, captured_state_1, control=None, contacts=None, dt=dt)
        captured_solver.step(captured_state_1, captured_state_0, control=None, contacts=None, dt=dt)

    captured_offset_history = []
    for cycle in range(4):
        eager_solver.step(eager_state_0, eager_state_1, control=None, contacts=None, dt=dt)
        eager_solver.step(eager_state_1, eager_state_0, control=None, contacts=None, dt=dt)
        wp.capture_launch(capture.graph)

        captured_offset_history.append(captured_solver._scratchpad.strain_environment_offsets.numpy().copy())
        eager_arrays = _mpm_particle_state_arrays(eager_state_0)
        captured_arrays = _mpm_particle_state_arrays(captured_state_0)
        for name in eager_arrays:
            eager_values = eager_arrays[name].numpy()
            captured_values = captured_arrays[name].numpy()
            test.assertTrue(np.isfinite(eager_values).all(), f"{name} is non-finite after eager cycle {cycle}")
            test.assertTrue(np.isfinite(captured_values).all(), f"{name} is non-finite after captured cycle {cycle}")
            np.testing.assert_allclose(
                captured_values,
                eager_values,
                rtol=1.0e-5,
                atol=1.0e-6,
                equal_nan=False,
                err_msg=f"{name} differs after capture replay cycle {cycle}",
            )
        for name, array in _mpm_strain_field_arrays(captured_solver).items():
            test.assertTrue(
                np.isfinite(array.numpy()).all(), f"Padded {name} is non-finite after captured cycle {cycle}"
            )

    final_q = captured_state_0.particle_q.numpy()
    final_qd = captured_state_0.particle_qd.numpy()
    final_qd_grad = captured_state_0.mpm.particle_qd_grad.numpy()
    final_elastic_strain = captured_state_0.mpm.particle_elastic_strain.numpy()
    final_stress = captured_state_0.mpm.particle_stress.numpy()
    initial_voxels = np.floor(initial_q[:, 0] / captured_solver.voxel_size)
    final_voxels = np.floor(final_q[:, 0] / captured_solver.voxel_size)
    test.assertTrue(np.any(initial_voxels != final_voxels), "No particle crossed a voxel boundary")
    test.assertTrue(
        any(not np.array_equal(initial_offsets, offsets) for offsets in captured_offset_history),
        "Environment offsets did not change as particles crossed cells",
    )
    test.assertGreater(np.linalg.norm(final_qd_grad - initial_qd_grad), 1.0e-5)
    test.assertGreater(np.linalg.norm(final_elastic_strain - initial_elastic_strain), 1.0e-5)
    test.assertGreater(np.linalg.norm(final_stress - initial_stress), 1.0e-5)

    mean_displacements = []
    mean_velocities = []
    for world in range(captured_model.world_count):
        world_slice = slice(world_starts[world], world_starts[world + 1])
        mean_displacements.append(np.mean(final_q[world_slice, 0] - initial_q[world_slice, 0]))
        mean_velocities.append(np.mean(final_qd[world_slice, 0]))

    test.assertGreater(mean_displacements[0], 0.0)
    test.assertLess(mean_displacements[1], 0.0)
    test.assertGreater(mean_velocities[0], 0.0)
    test.assertLess(mean_velocities[1], 0.0)


def test_multiworld_sparse_pic_matches_independent(test, device):
    """Verify multi-world sparse PIC matches independent."""
    _run_multiworld_reference_case(device, grid_type="sparse", integration_scheme="pic")


def test_multiworld_sparse_gimp_matches_independent(test, device):
    """Verify multi-world sparse GIMP matches independent."""
    _run_multiworld_reference_case(device, grid_type="sparse", integration_scheme="gimp")


def test_multiworld_sparse_empty_worlds_padding_matches_independent(test, device):
    """Verify multi-world sparse empty worlds padding matches independent."""
    populated_worlds = (1, 3)
    world_gravities = ((3.0, -2.0, 0.0), (-5.0, 1.0, 0.0))
    reference_states = []

    for world_gravity in world_gravities:
        reference_model = _make_mpm_particle_builder().finalize(device=device)
        reference_model.set_gravity(world_gravity)
        reference_config = _make_mpm_config(grid_type="sparse", integration_scheme="pic")
        reference_config.grid_padding = 1
        _, reference_state = _step_mpm(reference_model, reference_config)
        reference_states.append((reference_state.particle_q.numpy(), reference_state.particle_qd.numpy()))

    local_builder = _make_mpm_particle_builder()
    empty_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    empty_builder.add_body(is_kinematic=True, label="empty_world_marker")
    multiworld_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(multiworld_builder)
    for world_builder in (empty_builder, local_builder, empty_builder, local_builder, empty_builder):
        multiworld_builder.add_world(world_builder)

    multiworld_model = multiworld_builder.finalize(device=device)
    for world, world_gravity in zip(populated_worlds, world_gravities, strict=True):
        multiworld_model.set_gravity(world_gravity, world=world)

    starts = multiworld_model.particle_world_start.numpy()
    particles_per_world = reference_states[0][0].shape[0]
    for world in (0, 2, 4):
        test.assertEqual(starts[world], starts[world + 1])
    for world in populated_worlds:
        test.assertEqual(starts[world + 1] - starts[world], particles_per_world)

    multiworld_config = _make_mpm_config(grid_type="sparse", integration_scheme="pic")
    multiworld_config.grid_padding = 1
    _, multiworld_state = _step_mpm(multiworld_model, multiworld_config)
    multiworld_q = multiworld_state.particle_q.numpy()
    multiworld_qd = multiworld_state.particle_qd.numpy()

    mean_velocities = []
    for world, (reference_q, reference_qd) in zip(populated_worlds, reference_states, strict=True):
        world_slice = slice(starts[world], starts[world + 1])
        world_q = multiworld_q[world_slice]
        world_qd = multiworld_qd[world_slice]
        np.testing.assert_array_equal(np.isfinite(world_q), np.ones_like(world_q, dtype=bool))
        np.testing.assert_array_equal(np.isfinite(world_qd), np.ones_like(world_qd, dtype=bool))
        np.testing.assert_allclose(world_q, reference_q, rtol=1.0e-5, atol=1.0e-6, equal_nan=False)
        np.testing.assert_allclose(world_qd, reference_qd, rtol=1.0e-5, atol=1.0e-6, equal_nan=False)
        mean_velocities.append(np.mean(world_qd, axis=0))

    mean_velocities = np.asarray(mean_velocities)
    np.testing.assert_array_less(np.full(2, 1.0e-3), np.abs(mean_velocities[:, 0]))
    np.testing.assert_array_equal(np.sign(mean_velocities[:, 0]), np.array((1.0, -1.0)))


def test_multiworld_isolation_is_opt_in(test, device):
    """Verify multi-world isolation is opt in."""
    config = SolverImplicitMPM.Config()
    test.assertFalse(config.separate_worlds)


def test_empty_particle_model_rejected(test, device):
    """Verify empty particle model rejected."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, -9.81, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, -9.81, 0.0)))
    builder.add_world(newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, -9.81, 0.0)))
    model = builder.finalize(device=device)

    with test.assertRaisesRegex(ValueError, "at least one particle"):
        SolverImplicitMPM(model, _make_mpm_config())


def test_multiworld_global_particles_rejected(test, device):
    """Verify multi-world global particles rejected."""
    builder = _make_mpm_particle_builder()
    local = _make_mpm_particle_builder()
    builder.add_world(local)
    builder.add_world(local)
    model = builder.finalize(device=device)

    with test.assertRaisesRegex(ValueError, "global MPM particles"):
        SolverImplicitMPM(model, _make_mpm_config())


def test_single_world_global_particles_supported(test, device):
    """Verify single world global particles supported."""
    initial_plastic_volume_strain = 0.975
    model = _make_mpm_particle_builder(initial_plastic_volume_strain=initial_plastic_volume_strain).finalize(
        device=device
    )
    config = _make_mpm_config()
    config.collider_basis = "pic"
    config.strain_basis = "pic"
    config.warmstart_mode = "particles"
    solver, state = _step_mpm(model, config, step_count=1)
    test.assertTrue(np.isfinite(state.particle_q.numpy()).all())
    np.testing.assert_array_equal(model.particle_world.numpy(), -1)
    test.assertTrue(np.all(state.particle_qd.numpy()[:, 1] < 0.0))

    impulse = solver._last_step_data.ws_impulse_field.dof_values
    stress = solver._last_step_data.ws_stress_field.dof_values
    impulse_values = np.ones((model.particle_count, 3), dtype=np.float32)
    stress_values = np.ones((model.particle_count, 6), dtype=np.float32)
    impulse.assign(impulse_values)
    stress.assign(stress_values)
    state.mpm.particle_Jp.fill_(2.0)
    solver.reset(state, world_mask=wp.array((True, False), dtype=wp.bool, device=device))
    np.testing.assert_array_equal(state.mpm.particle_Jp.numpy(), 2.0)
    np.testing.assert_array_equal(impulse.numpy(), impulse_values)
    np.testing.assert_array_equal(stress.numpy(), stress_values)

    solver.reset(state, world_mask=wp.array((False, True), dtype=wp.bool, device=device))
    np.testing.assert_array_equal(
        state.mpm.particle_Jp.numpy(),
        np.full(model.particle_count, initial_plastic_volume_strain, dtype=np.float32),
    )
    np.testing.assert_array_equal(impulse.numpy(), np.zeros_like(impulse_values))
    np.testing.assert_array_equal(stress.numpy(), np.zeros_like(stress_values))


def test_multiworld_default_shared_grid_accepts_global_particles(test, device):
    """Verify multi-world default shared grid accepts global particles."""
    builder = _make_mpm_particle_builder()
    local = _make_mpm_particle_builder()
    builder.add_world(local)
    builder.add_world(local)
    model = builder.finalize(device=device)
    initial_q = model.particle_q.numpy()
    config = _make_mpm_config()
    config.separate_worlds = SolverImplicitMPM.Config().separate_worlds
    test.assertFalse(config.separate_worlds)
    _solver, state = _step_mpm(model, config, step_count=1)
    particle_q = state.particle_q.numpy()
    particle_qd = state.particle_qd.numpy()
    particle_world = model.particle_world.numpy()
    test.assertTrue(np.isfinite(particle_q).all())
    test.assertFalse(np.array_equal(particle_q, initial_q))
    for world in range(-1, model.world_count):
        world_qd = particle_qd[particle_world == world]
        test.assertGreater(world_qd.shape[0], 0)
        test.assertTrue(np.all(world_qd[:, 1] < 0.0))


def test_multiworld_invalid_particle_world_rejected(test, device):
    """Verify multi-world invalid particle world rejected."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, -9.81, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    local = _make_mpm_particle_builder()
    builder.add_world(local)
    builder.add_world(local)
    model = builder.finalize(device=device)
    particle_world = model.particle_world.numpy()

    for invalid_world in (-2, model.world_count):
        invalid_particle_world = particle_world.copy()
        invalid_particle_world[0] = invalid_world
        model.particle_world.assign(invalid_particle_world)
        with test.subTest(invalid_world=invalid_world):
            with test.assertRaisesRegex(ValueError, "invalid MPM particle world IDs"):
                SolverImplicitMPM(model, _make_mpm_config())


def test_multiworld_shared_grid_couples_worlds(test, device):
    """Verify multi-world shared grid couples worlds."""

    def run(separate_worlds):
        local = _make_mpm_particle_builder(gravity=(0.0, 0.0, 0.0))
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.add_world(local)
        builder.add_world(local)
        model = builder.finalize(device=device)
        model.set_gravity((3.0, 0.0, 0.0), world=0)
        model.set_gravity((-3.0, 0.0, 0.0), world=1)

        config = _make_mpm_config()
        config.separate_worlds = separate_worlds
        _, state = _step_mpm(model, config, step_count=1)
        starts = model.particle_world_start.numpy()
        velocities = state.particle_qd.numpy()
        return np.asarray(
            [np.mean(velocities[starts[world] : starts[world + 1]], axis=0) for world in range(model.world_count)]
        )

    isolated_velocity = run(separate_worlds=True)
    shared_velocity = run(separate_worlds=False)

    test.assertGreater(isolated_velocity[0, 0], 1.0e-3)
    test.assertLess(isolated_velocity[1, 0], -1.0e-3)
    np.testing.assert_allclose(shared_velocity, 0.0, rtol=0.0, atol=1.0e-6)


def test_multiworld_collider_world_validation(test, device):
    """Verify multi-world collider world validation."""
    model = _make_two_world_particle_model(device)
    solver = SolverImplicitMPM(model, _make_mpm_config())
    mesh = _make_box_collider_mesh(device)
    collider = solver._mpm_model.collider

    test.assertEqual(collider.collider_world.shape[0], 0)
    test.assertEqual(collider.collider_face_offset.shape[0], 0)
    test.assertEqual(collider.world_collider_ids.shape[0], 0)
    np.testing.assert_array_equal(collider.world_collider_offsets.numpy(), np.zeros(model.world_count + 1))

    solver.setup_collider(collider_meshes=[mesh])
    np.testing.assert_array_equal(collider.collider_world.numpy(), np.array((-1,)))

    with test.assertRaisesRegex(ValueError, "collider world ID"):
        solver.setup_collider(collider_meshes=[mesh], collider_world_ids=[model.world_count])

    with test.assertRaisesRegex(ValueError, "collider_world_ids"):
        solver.setup_collider(collider_meshes=[mesh], collider_world_ids=[])

    meshes = [_make_box_collider_mesh(device, half_extent=scale) for scale in (0.25, 0.5, 0.75)]
    solver.setup_collider(collider_meshes=meshes, collider_world_ids=[1, -1, 0])

    np.testing.assert_array_equal(collider.collider_world.numpy(), np.array((1, -1, 0)))
    np.testing.assert_array_equal(collider.world_collider_ids.numpy(), np.array((1, 2, 1, 0)))
    np.testing.assert_array_equal(collider.world_collider_offsets.numpy(), np.array((0, 2, 4)))
    np.testing.assert_array_equal(collider.collider_body_index.numpy(), np.array((-1, -1, -1)))

    face_counts = [mesh.indices.shape[0] // 3 for mesh in meshes]
    expected_face_offsets = np.cumsum((0, *face_counts[:-1]))
    np.testing.assert_array_equal(collider.collider_face_offset.numpy(), expected_face_offsets)


def test_multiworld_collision_sdf_filters_stable_colliders(test, device):
    """Verify multi-world collision SDF filters stable colliders."""
    model = _make_two_world_particle_model(device)
    solver = SolverImplicitMPM(model, _make_mpm_config())
    meshes = [
        _make_box_collider_mesh(device, half_extent=0.25),
        _make_box_collider_mesh(device, half_extent=0.25, center=(3.0, 0.0, 0.0)),
        _make_box_collider_mesh(device, half_extent=0.25),
    ]
    solver.setup_collider(
        collider_meshes=meshes,
        collider_world_ids=[1, -1, 0],
        collider_friction=[0.1, 0.2, 0.3],
        collider_adhesion=[10.0, 20.0, 30.0],
        collider_projection_threshold=[0.01, 0.02, 0.03],
    )
    collider = solver._mpm_model.collider
    collider.query_max_dist = 1.0

    positions = wp.array(
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        dtype=wp.vec3,
        device=device,
    )
    environment_indices = wp.array((0, 1, 0, 1, _ALL_COLLIDER_WORLDS), dtype=int, device=device)
    collider_ids = wp.empty(5, dtype=int, device=device)
    material_ids = wp.empty(5, dtype=int, device=device)
    friction = wp.empty(5, dtype=float, device=device)
    adhesion = wp.empty(5, dtype=float, device=device)
    projection_threshold = wp.empty(5, dtype=float, device=device)
    state = model.state()

    wp.launch(
        _query_collision_sdf,
        dim=5,
        inputs=[
            positions,
            environment_indices,
            collider,
            state.body_q,
            state.body_qd,
            None,
            collider_ids,
            material_ids,
            friction,
            adhesion,
            projection_threshold,
        ],
        device=device,
    )

    np.testing.assert_array_equal(collider_ids.numpy(), np.array((2, 0, 1, 1, 0)))
    np.testing.assert_array_equal(material_ids.numpy(), np.array((3, 1, 2, 2, 1)))
    np.testing.assert_allclose(friction.numpy(), np.array((0.3, 0.1, 0.2, 0.2, 0.1)))
    np.testing.assert_allclose(adhesion.numpy(), np.array((30.0, 10.0, 20.0, 20.0, 10.0)))
    np.testing.assert_allclose(projection_threshold.numpy(), np.array((0.03, 0.01, 0.02, 0.02, 0.01)))


def test_multiworld_rasterize_collider_node_environments(test, device):
    """Verify multi-world rasterize collider node environments."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    local_builder = _make_mpm_particle_builder(gravity=(0.0, 0.0, 0.0))
    for _ in range(3):
        builder.add_world(local_builder)
    model = builder.finalize(device=device)
    solver = SolverImplicitMPM(model, _make_mpm_config())
    solver.setup_collider(
        collider_meshes=[
            _make_box_collider_mesh(device, half_extent=0.25),
            _make_box_collider_mesh(device, half_extent=0.25, center=(3.0, 0.0, 0.0)),
        ],
        collider_world_ids=[0, -1],
    )
    collider = solver._mpm_model.collider
    collider.query_max_dist = 1.0

    node_positions = wp.array(
        ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
        dtype=wp.vec3,
        device=device,
    )
    node_environment_offsets = wp.array((0, 2, 2, 4), dtype=int, device=device)
    node_volumes = wp.ones(4, dtype=float, device=device)
    collider_sdf = wp.empty(4, dtype=float, device=device)
    collider_velocity = wp.empty(4, dtype=wp.vec3, device=device)
    collider_normals = wp.empty(4, dtype=wp.vec3, device=device)
    collider_friction = wp.empty(4, dtype=float, device=device)
    collider_adhesion = wp.empty(4, dtype=float, device=device)
    collider_ids = wp.empty(4, dtype=int, device=device)
    state = model.state()

    wp.launch(
        rasterize_collider_kernel,
        dim=4,
        inputs=[
            collider,
            state.body_q,
            state.body_qd,
            None,
            0.1,
            0.0,
            0.01,
            node_positions,
            node_environment_offsets,
            node_volumes,
            collider_sdf,
            collider_velocity,
            collider_normals,
            collider_friction,
            collider_adhesion,
            collider_ids,
        ],
        device=device,
    )

    np.testing.assert_array_equal(collider_ids.numpy(), np.array((0, 1, -1, 1)))
    test.assertLess(collider_sdf.numpy()[0], 0.0)
    test.assertLess(collider_sdf.numpy()[1], 0.0)
    test.assertGreater(collider_sdf.numpy()[2], 1.0e6)
    test.assertLess(collider_sdf.numpy()[3], 0.0)


def test_multiworld_project_outside_filters_particle_world(test, device):
    """Verify multi-world project outside filters particle world."""
    model = _make_two_world_particle_model(device)
    solver = SolverImplicitMPM(model, _make_mpm_config())
    collider_mesh = _make_box_collider_mesh(device, half_extent=0.2, center=(0.05, 0.05, 0.05))
    state_in = model.state()
    initial_positions = state_in.particle_q.numpy()
    particle_world = model.particle_world.numpy()

    solver.setup_collider(collider_meshes=[collider_mesh], collider_world_ids=[0])
    local_state_out = model.state()
    solver.project_outside(state_in, local_state_out, dt=0.01, gap=1.0)
    local_positions = local_state_out.particle_q.numpy()

    test.assertFalse(np.array_equal(local_positions[particle_world == 0], initial_positions[particle_world == 0]))
    np.testing.assert_array_equal(local_positions[particle_world == 1], initial_positions[particle_world == 1])

    solver.setup_collider(collider_meshes=[collider_mesh], collider_world_ids=[-1])
    global_state_out = model.state()
    solver.project_outside(state_in, global_state_out, dt=0.01, gap=1.0)
    global_positions = global_state_out.particle_q.numpy()

    test.assertFalse(np.array_equal(global_positions[particle_world == 0], initial_positions[particle_world == 0]))
    test.assertFalse(np.array_equal(global_positions[particle_world == 1], initial_positions[particle_world == 1]))
    np.testing.assert_allclose(
        global_positions[particle_world == 0], global_positions[particle_world == 1], rtol=0.0, atol=1.0e-7
    )


def test_multiworld_render_grains_follow_particle_world(test, device):
    """Verify multi-world render grains follow particle world."""
    empty_world = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    empty_world.add_body(is_kinematic=True, label="empty_world_marker")

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(_make_mpm_particle_builder(gravity=(0.0, 0.0, 0.0), velocity=(0.4, 0.0, 0.0)))
    builder.add_world(empty_world)
    builder.add_world(_make_mpm_particle_builder(gravity=(0.0, 0.0, 0.0), velocity=(-0.4, 0.0, 0.0)))
    model = builder.finalize(device=device)

    temporary_store = fem.TemporaryStore()
    config = _make_mpm_config(grid_type="dense", integration_scheme="pic")
    config.grid_padding = 1
    solver = SolverImplicitMPM(model, config=config, temporary_store=temporary_store)
    state_0 = model.state()
    state_1 = model.state()
    grains = solver.sample_render_grains(state_0, grains_per_particle=2)
    grains_initial = grains.numpy().copy()

    dt = 0.01
    solver.update_render_grains(state_0, state_0, grains, dt=dt)
    np.testing.assert_array_equal(grains.numpy(), grains_initial)

    solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
    # The helper must select the grains' device rather than relying on the
    # caller's current Warp device.
    update_device = "cpu" if device.is_cuda else device
    with wp.ScopedDevice(update_device):
        solver.update_render_grains(state_0, state_1, grains, dt=dt)

    grain_positions = grains.numpy()
    particle_world_start = model.particle_world_start.numpy()
    test.assertEqual(grains.shape, (model.particle_count, 2))
    test.assertTrue(np.isfinite(grain_positions).all())
    test.assertEqual(particle_world_start[1], particle_world_start[2])

    displacement_x = grain_positions[..., 0] - grains_initial[..., 0]
    world_0 = slice(particle_world_start[0], particle_world_start[1])
    world_2 = slice(particle_world_start[2], particle_world_start[3])
    test.assertGreater(np.mean(displacement_x[world_0]), 1.0e-4)
    test.assertLess(np.mean(displacement_x[world_2]), -1.0e-4)

    zero_grains = solver.sample_render_grains(state_1, grains_per_particle=0)
    solver.update_render_grains(state_0, state_1, zero_grains, dt=dt)
    test.assertEqual(zero_grains.shape, (model.particle_count, 0))


def test_multiworld_global_dynamic_collider_rejected(test, device):
    """Verify multi-world global dynamic collider rejected."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    body = builder.add_body(mass=1.0, inertia=inertia, lock_inertia=True)
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    builder.add_shape_box(body, cfg=shape_cfg)
    model = _make_two_world_particle_model(device, builder=builder)

    with test.assertRaisesRegex(ValueError, "global dynamic collider"):
        SolverImplicitMPM(model, _make_mpm_config())


def test_multiworld_global_kinematic_collider_supported(test, device):
    """Verify multi-world global kinematic collider supported."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    body = builder.add_body(mass=1.0, inertia=inertia, lock_inertia=True, is_kinematic=True)
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    builder.add_shape_box(body, cfg=shape_cfg)
    model = _make_two_world_particle_model(device, builder=builder)

    solver = SolverImplicitMPM(model, _make_mpm_config())
    collider = solver._mpm_model.collider
    test.assertGreater(model.body_mass.numpy()[body], 0.0)
    np.testing.assert_array_equal(collider.collider_world.numpy(), np.array((-1,)))
    np.testing.assert_array_equal(solver._mpm_model.collider_body_mass.numpy(), np.zeros(model.body_count))
    test.assertFalse(solver._mpm_model.has_compliant_colliders)

    with test.assertRaisesRegex(ValueError, "global dynamic collider"):
        solver.setup_collider(body_mass=model.body_mass)


def test_multiworld_masked_reset_refreshes_global_kinematic_collider_history(test, device):
    """Verify multi-world masked reset refreshes global kinematic collider history."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    body = builder.add_body(mass=1.0, inertia=inertia, lock_inertia=True, is_kinematic=True)
    builder.add_shape_box(body, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    model = _make_two_world_particle_model(device, builder=builder)
    config = _make_mpm_config()
    config.collider_velocity_mode = "backward"
    config.warmstart_mode = "none"
    solver = SolverImplicitMPM(model, config)
    state = model.state()

    test.assertEqual(int(model.body_world.numpy()[body]), -1)

    expected_previous = solver._last_step_data.body_q_prev.numpy()[body].copy()
    body_q = state.body_q.numpy()
    body_q[body, :3] = (1.0, 2.0, 3.0)
    state.body_q.assign(body_q)
    solver.reset(state, world_mask=wp.array((True, True, False), dtype=wp.bool, device=device))
    np.testing.assert_array_equal(solver._last_step_data.body_q_prev.numpy()[body], expected_previous)

    solver.reset(state, world_mask=wp.array((False, False, True), dtype=wp.bool, device=device))
    np.testing.assert_array_equal(solver._last_step_data.body_q_prev.numpy()[body], body_q[body])

    expected_previous = body_q[body].copy()
    body_q[body, :3] = (4.0, 5.0, 6.0)
    state.body_q.assign(body_q)
    solver.reset(state, world_mask=wp.array((False, False, False), dtype=wp.bool, device=device))
    np.testing.assert_array_equal(solver._last_step_data.body_q_prev.numpy()[body], expected_previous)

    solver.reset(state, world_mask=wp.array((True, True, True), dtype=wp.bool, device=device))
    np.testing.assert_array_equal(solver._last_step_data.body_q_prev.numpy()[body], body_q[body])


def test_multiworld_local_dynamic_collider_and_mass_override(test, device):
    """Verify multi-world local dynamic collider and mass override."""
    local_builder = _make_mpm_particle_builder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    body = local_builder.add_body(mass=1.0, inertia=inertia, lock_inertia=True)
    local_builder.add_shape_box(body, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    model = _make_two_world_particle_model(device, local_builder=local_builder)
    solver = SolverImplicitMPM(model, _make_mpm_config())

    np.testing.assert_array_equal(solver._mpm_model.collider.collider_world.numpy(), np.array((0, 1)))
    test.assertTrue(np.all(solver._mpm_model.collider_body_mass.numpy() > 0.0))
    test.assertTrue(solver._mpm_model.has_compliant_colliders)

    effective_mass = wp.zeros_like(model.body_mass)
    solver.setup_collider(body_mass=effective_mass)
    test.assertIs(solver._mpm_model.collider_body_mass, effective_mass)
    np.testing.assert_array_equal(solver._mpm_model.collider_body_mass.numpy(), np.zeros(model.body_count))
    test.assertFalse(solver._mpm_model.has_compliant_colliders)


def test_multiworld_default_static_colliders_grouped_by_world(test, device):
    """Verify multi-world default static colliders grouped by world."""
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    builder.add_shape_box(-1, cfg=shape_cfg)
    local_builder = _make_mpm_particle_builder(gravity=(0.0, 0.0, 0.0))
    local_builder.add_shape_box(-1, cfg=shape_cfg)
    model = _make_two_world_particle_model(device, builder=builder, local_builder=local_builder)

    collider = SolverImplicitMPM(model, _make_mpm_config())._mpm_model.collider
    np.testing.assert_array_equal(collider.collider_world.numpy(), np.array((-1, 0, 1)))
    np.testing.assert_array_equal(collider.collider_body_index.numpy(), np.array((-1, -1, -1)))
    np.testing.assert_array_equal(collider.world_collider_ids.numpy(), np.array((0, 1, 0, 2)))
    np.testing.assert_array_equal(collider.world_collider_offsets.numpy(), np.array((0, 2, 4)))

    face_offsets = collider.collider_face_offset.numpy()
    face_count = _make_box_collider_mesh(device).indices.shape[0] // 3
    np.testing.assert_array_equal(face_offsets, np.array((0, face_count, 2 * face_count)))
    test.assertEqual(collider.face_material_index.shape[0], 3 * face_count)


def test_multiworld_external_collider_world_count_mismatch(test, device):
    """Verify multi-world external collider world count mismatch."""
    model = _make_two_world_particle_model(device)
    solver = SolverImplicitMPM(model, _make_mpm_config())

    external_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    local_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    local_builder.add_shape_box(-1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    external_builder.add_world(local_builder)
    external_model = external_builder.finalize(device=device)

    with test.assertRaisesRegex(ValueError, "world_count"):
        solver.setup_collider(model=external_model)


def test_shared_solver_globalizes_external_multiworld_colliders(test, device):
    """Verify shared solver globalizes external multi-world colliders."""
    model = _make_mpm_particle_builder(gravity=(0.0, 0.0, 0.0)).finalize(device=device)
    solver = SolverImplicitMPM(model, _make_mpm_config())

    external_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    local_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    local_builder.add_shape_box(-1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    external_builder.add_world(local_builder)
    external_builder.add_world(local_builder)
    external_model = external_builder.finalize(device=device)

    solver.setup_collider(model=external_model)

    collider = solver._mpm_model.collider
    test.assertGreater(collider.collider_world.shape[0], 0)
    np.testing.assert_array_equal(collider.collider_world.numpy(), np.full(collider.collider_world.shape, -1))
    np.testing.assert_array_equal(collider.world_collider_ids.numpy(), np.arange(collider.collider_world.shape[0]))
    test.assertEqual(collider.world_collider_offsets.numpy()[1], collider.collider_world.shape[0])
    face_count = _make_box_collider_mesh(device).indices.shape[0] // 3
    test.assertEqual(collider.face_material_index.shape[0], 2 * face_count)
    test.assertEqual(np.unique(collider.face_material_index.numpy()).shape[0], 2)


def test_cell_quadrature_weights(test, device):
    """Check that cell quadrature transfer weights are continuous partitions of unity.

    Particles on either side of an interior cell face receive the same point
    weights, and particles near the grid boundary renormalize their weights
    over the existing cells.
    """
    voxel_size = 0.25
    res = 4
    grid = fem.Grid3D(bounds_lo=wp.vec3(0.0), bounds_hi=wp.vec3(res * voxel_size), res=wp.vec3i(res))
    domain = fem.Cells(grid)

    rng = np.random.default_rng(42)
    face_count = 16
    face = 2.0 * voxel_size
    offset = 1.0e-5
    base = rng.uniform(0.1, 0.9, size=(face_count, 3))
    below = base.copy()
    below[:, 1] = face - offset
    above = base.copy()
    above[:, 1] = face + offset
    scattered = rng.uniform(0.0, res * voxel_size, size=(64, 3))
    positions = wp.array(np.concatenate([below, above, scattered]), dtype=wp.vec3, device=device)

    for points_per_axis in (1, 2):
        with test.subTest(points_per_axis=points_per_axis), wp.ScopedDevice(device):
            # PicQuadrature launches some of its binning kernels on the current device
            pic = fem.PicQuadrature(domain, positions, use_domain_element_indices=True)
            cell_quadrature = CellQuadrature(points_per_axis, device=device)
            cell_quadrature.compute_weights(pic, positions, voxel_size, temporary_store=None)
            # Copy before releasing, since CPU arrays share memory with their NumPy views
            index = cell_quadrature.point_index.numpy().copy()
            weight = cell_quadrature.point_weight.numpy().copy()
            point_coords = cell_quadrature.point_coords.numpy().copy()
            cell_quadrature.release()

            test.assertTrue(np.all(weight >= 0.0))
            np.testing.assert_allclose(weight.sum(axis=1), 1.0, atol=1.0e-5)
            test.assertTrue(np.all(index[weight > 0.0] >= 0))
            test.assertEqual(point_coords.shape, (res**3, points_per_axis**3, 3))
            np.testing.assert_allclose(np.sort(np.unique(point_coords)), cell_quadrature.abscissae, atol=1.0e-6)

            for p in range(face_count):
                weights_below = {q: w for q, w in zip(index[p], weight[p], strict=True) if q >= 0}
                weights_above = {
                    q: w for q, w in zip(index[face_count + p], weight[face_count + p], strict=True) if q >= 0
                }
                difference = max(
                    abs(weights_below.get(q, 0.0) - weights_above.get(q, 0.0))
                    for q in set(weights_below) | set(weights_above)
                )
                test.assertLess(difference, 1.0e-3)


def _run_elastic_block(device, integration_scheme, strain_basis, step_count=40, collider_basis="S2"):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, -9.81, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle_grid(
        pos=wp.vec3(0.025, 0.025, 0.025),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=6,
        dim_y=6,
        dim_z=6,
        cell_x=0.05,
        cell_y=0.05,
        cell_z=0.05,
        mass=0.125,
        jitter=0.0,
        radius_mean=0.025,
        custom_attributes={
            "mpm:young_modulus": 1.0e5,
            "mpm:poisson_ratio": 0.3,
            "mpm:friction": 10.0,
            "mpm:yield_stress": 1.0e7,
            "mpm:yield_pressure": 1.0e8,
            "mpm:tensile_yield_ratio": 1.0,
        },
    )
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    config = SolverImplicitMPM.Config(
        voxel_size=0.1,
        grid_type="dense",
        strain_basis=strain_basis,
        collider_basis=collider_basis,
        integration_scheme=integration_scheme,
        solver="gs",
        max_iterations=200,
        tolerance=1.0e-6,
    )
    return _step_mpm(model, config, step_count=step_count)


def test_strain_mode_mass_filter(test, device):
    """Keep a positive-mass zero-mean mode while dropping a zero-mass mode."""
    with wp.ScopedDevice(device):
        mass = np.diag((1.0, 1.0 / 12.0, 0.0)).astype(np.float32)
        eigenvalues = wp.empty((1, 3), dtype=float, device=device)
        eigenvectors = wp.empty((1, 3, 3), dtype=float, device=device)
        rotated_volume = wp.empty((1, 3), dtype=float, device=device)
        wp.launch(
            compute_eigenvalues,
            dim=1,
            inputs=[
                wp.array((0, 1), dtype=int, device=device),
                wp.array((0,), dtype=int, device=device),
                wp.array(mass[None], dtype=wp.mat33, device=device),
                wp.array(((1.0, 0.0, 0.0),), dtype=float, device=device),
                # A heterogeneous material can have a negative moment against a zero-mean mode.
                wp.array(((1.0,) * 6, (-1.0 / 12.0,) * 6, (0.0,) * 6), dtype=YieldParamVec, device=device),
                1.0e-4,
                eigenvalues,
                eigenvectors,
                rotated_volume,
            ],
            device=device,
        )
        modes = eigenvectors.numpy()[0]
        np.testing.assert_allclose(modes.T @ modes, np.diag((1.0, 1.0, 0.0)), atol=1.0e-6)
        np.testing.assert_allclose(modes.T @ np.diag(eigenvalues.numpy()[0]) @ modes, mass, atol=1.0e-6)


def test_strain_modes_symmetric_cell(test, device):
    """Retain full-rank strain modes and uniform material parameters on a symmetric cell."""

    class ModeSolver(SolverImplicitMPM):
        def _build_strain_eigenbasis(self, pic, scratch, inv_cell_volume):
            eigenvectors, rotated_volume = super()._build_strain_eigenbasis(pic, scratch, inv_cell_volume)
            self.element = int(pic.cell_indices.numpy().reshape(-1)[0])
            self.mode_count = eigenvectors.shape[1]
            self.modes = eigenvectors.numpy()[self.element].copy()
            self.mode_mass = (
                scratch.strain_node_particle_volume.numpy().reshape(-1, self.mode_count)[self.element].copy()
            )
            return eigenvectors, rotated_volume

        def _solve_rheology(self, pic, scratch, rigidity_operator, last_step_data, inv_cell_volume):
            result = super()._solve_rheology(pic, scratch, rigidity_operator, last_step_data, inv_cell_volume)
            self.mode_yield_parameters = (
                scratch.strain_yield_parameters_field.dof_values.numpy()
                .reshape(-1, self.mode_count, 6)[self.element]
                .copy()
            )
            return result

    abscissae = (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0))
    positions = np.array(np.meshgrid(abscissae, abscissae, abscissae)).reshape(3, -1).T
    expected_yield = np.array((300.0 * np.sqrt(1.5), 150.0 * np.sqrt(1.5), 20.0, 60.0, 0.0, 1000.0))
    with wp.ScopedDevice(device):
        for integration_scheme in ("pic", "gimp", "cell"):
            for strain_basis in ("P1d", "Q1d"):
                with test.subTest(integration_scheme=integration_scheme, strain_basis=strain_basis):
                    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                    SolverImplicitMPM.register_custom_attributes(builder)
                    for position in positions:
                        builder.add_particle(
                            pos=position,
                            vel=(0.0, 0.0, 0.0),
                            mass=0.125,
                            radius=0.25,
                            custom_attributes={
                                "mpm:young_modulus": 1.0e4,
                                "mpm:friction": 0.2,
                                "mpm:yield_pressure": 300.0,
                                "mpm:tensile_yield_ratio": 0.5,
                                "mpm:yield_stress": 20.0,
                                "mpm:viscosity": 10.0,
                            },
                        )
                    model = builder.finalize(device=device)
                    solver = ModeSolver(
                        model,
                        SolverImplicitMPM.Config(
                            voxel_size=1.0,
                            grid_type="fixed",
                            integration_scheme=integration_scheme,
                            strain_basis=strain_basis,
                            max_iterations=0,
                            warmstart_mode="none",
                        ),
                    )
                    state = model.state()
                    solver.step(state, state, None, None, 0.01)
                    test.assertTrue(np.all(solver.mode_mass > 1.0e-6))
                    np.testing.assert_allclose(solver.modes @ solver.modes.T, np.eye(solver.mode_count), atol=2.0e-5)
                    np.testing.assert_allclose(
                        solver.mode_yield_parameters,
                        np.tile(expected_yield, (solver.mode_count, 1)),
                        rtol=2.0e-4,
                        atol=2.0e-4,
                    )


def test_cell_integration_elastic_block(test, device):
    """Step an elastic block resting on the ground with fixed cell quadrature.

    Each supported strain basis keeps the block finite, above the ground, and
    close to its rest shape, consistent with particle quadrature.
    """
    _solver, reference = _run_elastic_block(device, "pic", "P0")
    reference_height = np.mean(reference.particle_q.numpy()[:, 1])

    for strain_basis in ("P0", "P1d", "Q1d"):
        with test.subTest(strain_basis=strain_basis):
            _solver, state = _run_elastic_block(device, "cell", strain_basis)
            positions = state.particle_q.numpy()
            elastic_strain = state.mpm.particle_elastic_strain.numpy()
            test.assertTrue(np.isfinite(positions).all())
            test.assertTrue(np.isfinite(elastic_strain).all())

            extent = np.max(positions, axis=0) - np.min(positions, axis=0)
            np.testing.assert_allclose(extent, 0.25, atol=0.03)
            test.assertGreater(np.min(positions[:, 1]), -0.02)
            test.assertAlmostEqual(np.mean(positions[:, 1]), reference_height, delta=0.02)

            determinant = np.linalg.det(elastic_strain)
            test.assertGreater(np.min(determinant), 0.9)
            test.assertLess(np.max(determinant), 1.1)


def test_cell_integration_particle_contact_rows(test, device):
    """Check that cell-scheme contact rows of particle colliders interpolate through the cell points.

    Each active row combines the trilinear point weights with the nodal shape
    functions at the points, so its coefficients sum to one, and the block
    stays above the ground.
    """
    solver, state = _run_elastic_block(device, "cell", "P0", step_count=10, collider_basis="pic8")
    collider_matrix = solver._scratchpad.collider_matrix
    offsets = collider_matrix.offsets.numpy()
    values = collider_matrix.values.numpy().reshape(-1)
    row_sums = np.array([values[offsets[row] : offsets[row + 1]].sum() for row in range(collider_matrix.nrow)])
    active = np.abs(row_sums) > 0.0
    test.assertGreater(np.count_nonzero(active), 0)
    np.testing.assert_allclose(row_sums[active], 1.0, atol=1.0e-4)
    # Two-hop rows reach the 27 nodes around each particle, beyond its own cell
    test.assertGreater(np.max(np.diff(offsets)), 8)

    positions = state.particle_q.numpy()
    test.assertTrue(np.isfinite(positions).all())
    test.assertGreater(np.min(positions[:, 1]), -0.02)


def _make_cell_viscous_block(device, viscosity, strain_basis="P0", solver="gs", max_iterations=5, young_modulus=1.0e15):
    builder = _make_mpm_particle_builder(gravity=(0.0, 0.0, 0.0), young_modulus=young_modulus, dimensions=(6, 6, 6))
    model = builder.finalize(device=device)
    model.mpm.friction.zero_()
    model.mpm.viscosity.fill_(viscosity)
    config = SolverImplicitMPM.Config(
        voxel_size=0.1,
        grid_type="fixed",
        integration_scheme="cell",
        strain_basis=strain_basis,
        solver=solver,
        max_iterations=max_iterations,
        tolerance=0.0,
        air_drag=1.0e-6,
    )
    return model, SolverImplicitMPM(model, config=config)


def test_cell_integration_viscosity_decay(test, device):
    """Dissipate shear with increasing viscosity, including truncated solves."""
    with wp.ScopedDevice(device):
        for strain_basis, solver, max_iterations in (
            ("P0", "gs", 5),
            ("P0", "jacobi", 5),
            ("P0", "gs", 100),
            ("P1d", "gs", 5),
            ("Q1d", "gs", 5),
        ):
            with test.subTest(strain_basis=strain_basis, solver=solver, max_iterations=max_iterations):
                energies = []
                for viscosity in (0.0, 10.0, 1000.0):
                    model, mpm_solver = _make_cell_viscous_block(
                        device, viscosity, strain_basis, solver, max_iterations
                    )
                    state, output = model.state(), model.state()
                    positions = state.particle_q.numpy()
                    centered = positions - np.mean(positions, axis=0)
                    velocities = np.column_stack((centered[:, 1], centered[:, 0], np.zeros(model.particle_count)))
                    state.particle_qd.assign(wp.array(velocities, dtype=wp.vec3, device=device))
                    initial_energy = np.sum(model.particle_mass.numpy()[:, None] * velocities**2)
                    for _ in range(5):
                        mpm_solver.step(state, output, None, None, 0.01)
                        state, output = output, state
                        velocities = state.particle_qd.numpy()
                        test.assertTrue(np.isfinite(velocities).all())
                        energy = np.sum(model.particle_mass.numpy()[:, None] * velocities**2)
                        test.assertLessEqual(energy, initial_energy * 1.001)
                    energies.append(energy)
                test.assertLess(energies[1], 0.95 * energies[0])
                test.assertLess(energies[2], 0.5 * energies[1])


def test_cell_integration_viscosity_translation(test, device):
    """Preserve uniform translation when viscosity has no strain to dissipate."""
    with wp.ScopedDevice(device):
        model, solver = _make_cell_viscous_block(device, 1000.0)
        state = model.state()
        translation = np.array((0.3, -0.1, 0.2), dtype=np.float32)
        initial_positions = state.particle_q.numpy().copy()
        state.particle_qd.fill_(wp.vec3(translation))
        for _ in range(5):
            solver.step(state, state, None, None, 0.01)
        np.testing.assert_allclose(
            state.particle_qd.numpy(), np.tile(translation, (model.particle_count, 1)), atol=2.0e-5
        )
        np.testing.assert_allclose(state.particle_q.numpy(), initial_positions + 0.05 * translation, atol=2.0e-6)


def test_cell_integration_viscosity_elastic_history(test, device):
    """Keep kinematic and stress-based elastic increments consistent in a converged viscous solve."""
    with wp.ScopedDevice(device):
        strains = []
        for kinematic_update in (False, True):
            model, solver = _make_cell_viscous_block(device, 10.0, max_iterations=300, young_modulus=1.0e4)
            solver._cell_kinematic_update = kinematic_update
            state, output = model.state(), model.state()
            positions = state.particle_q.numpy()
            centered = positions - np.mean(positions, axis=0)
            velocities = np.column_stack((centered[:, 1], centered[:, 0], np.zeros(model.particle_count)))
            state.particle_qd.assign(wp.array(velocities, dtype=wp.vec3, device=device))
            solver.step(state, output, None, None, 0.01)
            strains.append(output.mpm.particle_elastic_strain.numpy().copy())
        test.assertGreater(np.max(np.abs(strains[0] - np.eye(3))), 1.0e-5)
        np.testing.assert_allclose(strains[1], strains[0], atol=2.0e-5, rtol=0.0)


def test_cell_integration_rejects_unsupported(test, device):
    """Reject configurations and materials that cell quadrature does not support."""
    model = _make_mpm_particle_builder().finalize(device=device)
    for options, error in (
        ({"strain_basis": "Q1"}, ValueError),
        ({"strain_basis": "pic8"}, ValueError),
        ({"velocity_basis": "B2"}, NotImplementedError),
        ({"critical_fraction": 0.5}, NotImplementedError),
    ):
        with test.subTest(**options):
            config = SolverImplicitMPM.Config(voxel_size=0.1, grid_type="dense", integration_scheme="cell", **options)
            with test.assertRaises(error):
                SolverImplicitMPM(model, config)

    model.mpm.hardening.fill_(1.0)
    model_config = SolverImplicitMPM.Config(voxel_size=0.1, grid_type="dense", integration_scheme="cell")
    with test.assertRaises(NotImplementedError):
        SolverImplicitMPM(model, model_config)


def test_sand_cube_on_plane(test, device):
    # Emits a cube of particles on the ground

    N = 4
    particles_per_cell = 3
    voxel_size = 0.5
    particle_spacing = voxel_size / particles_per_cell
    friction = 0.6
    dt = 0.04

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)

    # Register MPM custom attributes before adding particles
    SolverImplicitMPM.register_custom_attributes(builder)

    builder.add_particle_grid(
        pos=wp.vec3(0.5 * particle_spacing),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=N * particles_per_cell,
        dim_y=N * particles_per_cell,
        dim_z=N * particles_per_cell,
        cell_x=particle_spacing,
        cell_y=particle_spacing,
        cell_z=particle_spacing,
        mass=1.0,
        jitter=0.0,
        custom_attributes={"mpm:friction": friction},
    )
    builder.add_ground_plane()

    model: newton.Model = builder.finalize(device=device)

    state_0: newton.State = model.state()
    state_1: newton.State = model.state()

    options = SolverImplicitMPM.Config()
    options.grid_type = "dense"  # use dense grid as sparse grid is GPU-only
    options.voxel_size = voxel_size

    solver = SolverImplicitMPM(model, config=options)

    init_pos = state_0.particle_q.numpy()

    # Run a few steps
    for _k in range(25):
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    # Checks the final bounding box corresponds to the expected collapse
    end_pos = state_0.particle_q.numpy()
    bb_min, bb_max = np.min(end_pos, axis=0), np.max(end_pos, axis=0)
    assert bb_min[model.up_axis] > -voxel_size
    assert voxel_size < bb_max[model.up_axis] < N * voxel_size

    assert np.all(bb_min > -N * voxel_size)
    assert np.all(bb_min < np.min(init_pos, axis=0))
    assert np.all(bb_max < 2 * N * voxel_size)

    # Checks that contact impulses are consistent
    impulses, impulse_positions, _collider_ids = solver.collect_collider_impulses(state_0)

    impulses = impulses.numpy()
    impulse_positions = impulse_positions.numpy()

    active_contacts = np.flatnonzero(np.linalg.norm(impulses, axis=1) > 0.01)
    contact_points = impulse_positions[active_contacts]
    contact_impulses = impulses[active_contacts]

    assert np.all(contact_points[:, model.up_axis] == 0.0)
    assert np.all(contact_impulses[:, model.up_axis] < 0.0)


def test_finite_difference_collider_velocity(test, device):
    """Test that finite-difference velocity mode correctly computes collider velocity.

    This test compares the two velocity modes with body_qd=0:
    - instantaneous mode: sees zero velocity (from body_qd), particles don't move with platform
    - finite_difference mode: computes velocity from position change, particles move with platform

    This directly validates that finite-difference mode correctly handles the case where
    body transforms are updated externally but body_qd doesn't reflect the actual motion.
    """
    voxel_size = 0.1
    particles_per_cell = 2
    particle_spacing = voxel_size / particles_per_cell
    dt = 0.02
    n_steps = 15

    # Platform moves in +X direction
    platform_vel_x = 0.5  # m/s

    def run_simulation(velocity_mode):
        """Run simulation with given velocity mode and return particle displacement."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)

        # Register MPM custom attributes before adding particles
        SolverImplicitMPM.register_custom_attributes(builder)

        # Add particles resting on the platform
        builder.add_particle_grid(
            pos=wp.vec3(-0.05, 0.12, -0.05),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=2 * particles_per_cell,
            dim_y=2 * particles_per_cell,
            dim_z=2 * particles_per_cell,
            cell_x=particle_spacing,
            cell_y=particle_spacing,
            cell_z=particle_spacing,
            mass=1.0,
            jitter=0.0,
            custom_attributes={"mpm:friction": 1.0},  # high friction
        )

        # Add a platform that particles rest on
        platform_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        platform_mesh = newton.Mesh.create_box(
            0.5,
            0.1,
            0.5,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)  # kinematic
        shape_cfg.margin = 0.02
        builder.add_shape_mesh(
            body=platform_body,
            mesh=platform_mesh,
            cfg=shape_cfg,
        )

        model = builder.finalize(device=device)

        state_0 = model.state()
        state_1 = model.state()

        options = SolverImplicitMPM.Config()
        options.voxel_size = voxel_size
        options.grid_type = "dense"
        options.collider_velocity_mode = velocity_mode

        solver = SolverImplicitMPM(model, config=options)

        init_mean_x = np.mean(state_0.particle_q.numpy()[:, 0])

        # Move platform with body_qd = 0
        for k in range(n_steps):
            t = (k + 1) * dt
            new_platform_x = platform_vel_x * t

            body_q_np = state_0.body_q.numpy()
            body_q_np[0] = (new_platform_x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            state_0.body_q.assign(body_q_np)

            # KEY: body_qd is ZERO - doesn't reflect actual motion
            body_qd_np = state_0.body_qd.numpy()
            body_qd_np[0] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            state_0.body_qd.assign(body_qd_np)

            solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
            state_0, state_1 = state_1, state_0

        end_mean_x = np.mean(state_0.particle_q.numpy()[:, 0])
        return end_mean_x - init_mean_x

    # 'forward' sees the current collider velocity; 'backward' derives it from
    # the previous-step collider position.
    displacement_instantaneous = run_simulation("forward")
    displacement_finite_diff = run_simulation("backward")

    # With instantaneous mode and body_qd=0, particles should barely move
    # (they see zero collider velocity, so no friction drag)
    test.assertLess(
        abs(displacement_instantaneous),
        0.02,
        f"instantaneous mode with body_qd=0 should show minimal particle movement, "
        f"but got {displacement_instantaneous:.3f}",
    )

    # With finite_difference mode, particles should move significantly
    # (velocity computed from position change)
    test.assertGreater(
        displacement_finite_diff,
        0.05,
        f"finite_difference mode should move particles with platform, "
        f"but displacement was only {displacement_finite_diff:.3f}",
    )

    # finite_difference should show significantly more movement than instantaneous
    test.assertGreater(
        displacement_finite_diff,
        displacement_instantaneous + 0.03,
        f"finite_difference ({displacement_finite_diff:.3f}) should show significantly more "
        f"movement than instantaneous ({displacement_instantaneous:.3f})",
    )


def test_cg_rheology_whole_step_graph_capture(test, device):
    """Capture a whole step with an iterative linear rheology solver.

    Regression for newton-physics/newton#3155: the iterative linear solver synced
    its device-side results to the host inside the capture, raising CUDA error 906.
    Both verbose settings are covered, since the verbose report is what reads those
    device-side results back. The scene has no colliders so ``solver="cg"`` is
    admissible.
    """
    if not device.is_cuda:
        test.skipTest("whole-step graph capture requires a CUDA device")
    if not wp.is_conditional_graph_supported():
        test.skipTest("whole-step graph capture requires conditional CUDA graph support")

    voxel_size = 0.1
    emit_lo = np.array([-0.15, -0.15, 0.1])
    emit_hi = np.array([0.15, 0.15, 0.4])
    dt = 1.0 / 120.0

    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)

    res = np.ceil(3 * (emit_hi - emit_lo) / voxel_size).astype(int)
    cell = (emit_hi - emit_lo) / res
    builder.add_particle_grid(
        pos=wp.vec3(*emit_lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=int(res[0]) + 1,
        dim_y=int(res[1]) + 1,
        dim_z=int(res[2]) + 1,
        cell_x=float(cell[0]),
        cell_y=float(cell[1]),
        cell_z=float(cell[2]),
        mass=float(np.prod(cell) * 1000.0),
        jitter=0.0,
        radius_mean=float(np.max(cell) * 0.5),
    )

    model = builder.finalize(device=device)

    # verbose=True is the path that reads the solver's device-side results back
    # for its report; both settings must capture without forcing a host sync.
    for verbose in (False, True):
        with test.subTest(verbose=verbose):
            options = SolverImplicitMPM.Config()
            options.solver = "cg"
            options.voxel_size = voxel_size
            options.grid_type = "fixed"  # whole-step capture precondition
            options.grid_padding = 8
            options.max_active_cell_count = 1 << 15
            options.max_iterations = 50
            options.tolerance = 1.0e-4

            solver = SolverImplicitMPM(model, options, verbose=verbose)
            state_0, state_1 = model.state(), model.state()

            with wp.ScopedCapture(device=device) as capture:
                solver.step(state_0, state_1, control=None, contacts=None, dt=dt)

            for _ in range(5):
                wp.capture_launch(capture.graph)

            # .numpy() performs the synchronous device-to-host copy that drains the replays.
            test.assertTrue(np.all(np.isfinite(state_1.particle_q.numpy())))


def test_proxy_particle_gravity_is_not_coupling_feedback(test, device):
    gravity = -9.81
    dt = 1.0 / 60.0

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.03)
    builder.add_particle(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.03)
    model = builder.finalize(device=device)
    model.mpm.yield_pressure.fill_(1.0e5)

    config = SolverImplicitMPM.Config()
    config.voxel_size = 0.2
    config.grid_type = "fixed"
    config.grid_padding = 2
    config.warmstart_mode = "none"
    config.transfer_scheme = "pic"
    config.max_iterations = 1

    solver = SolverCoupledProxy(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="xpbd",
                solver=lambda view: SolverXPBD(model=view, iterations=1),
                particles=[0],
            ),
            SolverCoupled.Entry(
                name="mpm",
                solver=lambda view: SolverImplicitMPM(model=view, config=config),
                particles=[1],
                in_place=True,
            ),
        ],
        coupling=SolverCoupledProxy.Config(
            proxies=[SolverCoupledProxy.Proxy(source="xpbd", destination="mpm", particles=[0])]
        ),
    )

    state_0 = model.state()
    state_1 = model.state()
    for step in range(1, 3):
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        expected_velocity = np.array([0.0, 0.0, step * gravity * dt])
        np.testing.assert_allclose(state_1.particle_qd.numpy()[0], expected_velocity, atol=1.0e-4)
        state_0, state_1 = state_1, state_0


devices = get_test_devices()
basic_devices = get_test_devices(mode="basic")
basic_cuda_devices = get_cuda_test_devices(mode="basic")


class TestImplicitMPM(unittest.TestCase):
    pass


add_function_test(
    TestImplicitMPM,
    "test_expanded_iterative_solver_names",
    test_expanded_iterative_solver_names,
    devices=basic_devices,
)


add_function_test(
    TestImplicitMPM,
    "test_array_squared_norm_batches",
    test_array_squared_norm_batches,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_linear_solver_result_norms",
    test_linear_solver_result_norms,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_residual_tolerance_scales",
    test_multiworld_residual_tolerance_scales,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_cr_matches_independent",
    test_multiworld_cr_matches_independent,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_dense_pic_matches_independent",
    test_multiworld_dense_pic_matches_independent,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_dense_gimp_matches_independent",
    test_multiworld_dense_gimp_matches_independent,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_fixed_pic_matches_independent",
    test_multiworld_fixed_pic_matches_independent,
    devices=basic_cuda_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_fixed_outer_graph_matches_eager",
    test_multiworld_fixed_outer_graph_matches_eager,
    devices=basic_cuda_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_fixed_capped_jacobi_matches_uncapped",
    test_multiworld_fixed_capped_jacobi_matches_uncapped,
    devices=basic_cuda_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_sparse_pic_matches_independent",
    test_multiworld_sparse_pic_matches_independent,
    devices=basic_cuda_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_sparse_gimp_matches_independent",
    test_multiworld_sparse_gimp_matches_independent,
    devices=basic_cuda_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_sparse_empty_worlds_padding_matches_independent",
    test_multiworld_sparse_empty_worlds_padding_matches_independent,
    devices=basic_cuda_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_isolation_is_opt_in",
    test_multiworld_isolation_is_opt_in,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_empty_particle_model_rejected",
    test_empty_particle_model_rejected,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_global_particles_rejected",
    test_multiworld_global_particles_rejected,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_single_world_global_particles_supported",
    test_single_world_global_particles_supported,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_default_shared_grid_accepts_global_particles",
    test_multiworld_default_shared_grid_accepts_global_particles,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_invalid_particle_world_rejected",
    test_multiworld_invalid_particle_world_rejected,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_shared_grid_couples_worlds",
    test_multiworld_shared_grid_couples_worlds,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_collider_world_validation",
    test_multiworld_collider_world_validation,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_collision_sdf_filters_stable_colliders",
    test_multiworld_collision_sdf_filters_stable_colliders,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_rasterize_collider_node_environments",
    test_multiworld_rasterize_collider_node_environments,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_project_outside_filters_particle_world",
    test_multiworld_project_outside_filters_particle_world,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_render_grains_follow_particle_world",
    test_multiworld_render_grains_follow_particle_world,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_global_dynamic_collider_rejected",
    test_multiworld_global_dynamic_collider_rejected,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_global_kinematic_collider_supported",
    test_multiworld_global_kinematic_collider_supported,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_masked_reset_refreshes_global_kinematic_collider_history",
    test_multiworld_masked_reset_refreshes_global_kinematic_collider_history,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_local_dynamic_collider_and_mass_override",
    test_multiworld_local_dynamic_collider_and_mass_override,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_default_static_colliders_grouped_by_world",
    test_multiworld_default_static_colliders_grouped_by_world,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_external_collider_world_count_mismatch",
    test_multiworld_external_collider_world_count_mismatch,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_shared_solver_globalizes_external_multiworld_colliders",
    test_shared_solver_globalizes_external_multiworld_colliders,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM, "test_sand_cube_on_plane", test_sand_cube_on_plane, devices=devices, check_output=False
)

add_function_test(
    TestImplicitMPM,
    "test_finite_difference_collider_velocity",
    test_finite_difference_collider_velocity,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestImplicitMPM,
    "test_cg_rheology_whole_step_graph_capture",
    test_cg_rheology_whole_step_graph_capture,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestImplicitMPM,
    "test_proxy_particle_gravity_is_not_coupling_feedback",
    test_proxy_particle_gravity_is_not_coupling_feedback,
    devices=devices,
    check_output=False,
)


add_function_test(
    TestImplicitMPM, "test_cell_integration_elastic_block", test_cell_integration_elastic_block, devices=devices
)
add_function_test(
    TestImplicitMPM, "test_strain_modes_symmetric_cell", test_strain_modes_symmetric_cell, devices=devices
)
add_function_test(TestImplicitMPM, "test_strain_mode_mass_filter", test_strain_mode_mass_filter, devices=devices)
add_function_test(
    TestImplicitMPM,
    "test_cell_integration_particle_contact_rows",
    test_cell_integration_particle_contact_rows,
    devices=devices,
)
add_function_test(
    TestImplicitMPM,
    "test_cell_integration_rejects_unsupported",
    test_cell_integration_rejects_unsupported,
    devices=devices,
)
add_function_test(TestImplicitMPM, "test_cell_quadrature_weights", test_cell_quadrature_weights, devices=devices)
add_function_test(
    TestImplicitMPM, "test_cell_integration_viscosity_decay", test_cell_integration_viscosity_decay, devices=devices
)
add_function_test(
    TestImplicitMPM,
    "test_cell_integration_viscosity_translation",
    test_cell_integration_viscosity_translation,
    devices=devices,
)
add_function_test(
    TestImplicitMPM,
    "test_cell_integration_viscosity_elastic_history",
    test_cell_integration_viscosity_elastic_history,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
