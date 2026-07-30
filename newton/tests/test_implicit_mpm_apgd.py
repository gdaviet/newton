# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

import numpy as np
import warp as wp
import warp.sparse as sp

from newton._src.solvers.implicit_mpm.contact_solver_kernels import project_collider_impulse_orthogonal
from newton._src.solvers.implicit_mpm.implicit_mpm_solver_kernels import mat13
from newton._src.solvers.implicit_mpm.rheology_solver_kernels import (
    YieldParamVec,
    make_apgd_evaluate_stress_kernel,
    mat66,
    project_stress_orthogonal,
    vec6,
)
from newton._src.solvers.implicit_mpm.solve_rheology import (
    CollisionData,
    MomentumData,
    RheologyData,
    _solve_rheology_apgd_prototype,
    solve_rheology,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()

_STRAIN_OPERATOR = np.array(
    [
        [np.sqrt(2.0 / 3.0), 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-np.sqrt(1.0 / 3.0), 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
)


@wp.kernel
def _project_contact_kernel(
    friction: wp.array[float],
    normals: wp.array[wp.vec3],
    adhesion: wp.array[float],
    impulse: wp.array[wp.vec3],
    projected: wp.array[wp.vec3],
):
    i = wp.tid()
    projected[i] = project_collider_impulse_orthogonal(friction[i], normals[i], adhesion[i], impulse[i])


@wp.kernel
def _project_stress_kernel(
    stress: wp.array[vec6],
    yield_params: wp.array[YieldParamVec],
    projected: wp.array[vec6],
):
    i = wp.tid()
    projected[i] = project_stress_orthogonal(stress[i], yield_params[i])


def _project_contacts(friction, normals, adhesion, impulse, device):
    friction_wp = wp.array(friction, dtype=float, device=device)
    normals_wp = wp.array(normals, dtype=wp.vec3, device=device)
    adhesion_wp = wp.array(adhesion, dtype=float, device=device)
    impulse_wp = wp.array(impulse, dtype=wp.vec3, device=device)
    projected_wp = wp.empty(len(friction), dtype=wp.vec3, device=device)
    wp.launch(
        _project_contact_kernel,
        dim=len(friction),
        inputs=[friction_wp, normals_wp, adhesion_wp, impulse_wp, projected_wp],
        device=device,
    )
    return projected_wp.numpy()


def _project_stresses(stress, yield_params, device):
    stress_wp = wp.array(stress, dtype=vec6, device=device)
    yield_params_wp = wp.array(yield_params, dtype=YieldParamVec, device=device)
    projected_wp = wp.empty(len(stress), dtype=vec6, device=device)
    wp.launch(
        _project_stress_kernel,
        dim=len(stress),
        inputs=[stress_wp, yield_params_wp, projected_wp],
        device=device,
    )
    return projected_wp.numpy()


def test_apgd_stress_evaluation_skips_inactive_capacity_rows(test, device):
    """Skip inactive capacity rows in specialized stress evaluation."""
    del test
    compliance_mat = sp.bsr_zeros(2, 2, mat66, device)
    response = wp.empty(2, dtype=vec6, device=device)
    kernel = make_apgd_evaluate_stress_kernel(
        has_compliance_mat=False,
        strain_velocity_node_count=1,
    )

    wp.launch(
        kernel,
        dim=2,
        inputs=[
            compliance_mat.offsets,
            compliance_mat.columns,
            compliance_mat.values,
            wp.array([0, 1, 1], dtype=int, device=device),
            wp.array([0], dtype=int, device=device),
            wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3, device=device),
            wp.zeros(2, dtype=vec6, device=device),
            wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3, device=device),
            wp.zeros(2, dtype=vec6, device=device),
            response,
        ],
        device=device,
    )

    np.testing.assert_allclose(response.numpy()[1], np.zeros(6), atol=0.0)


def _contact_projection_reference(friction, normal, adhesion, impulse):
    if friction < 0.0:
        return np.zeros(3)

    mu = max(0.0, friction)
    shifted = impulse + adhesion * normal
    normal_value = np.dot(shifted, normal)
    tangential = shifted - normal_value * normal
    tangential_norm = np.linalg.norm(tangential)

    if normal_value + mu * tangential_norm <= 0.0:
        projected = np.zeros(3)
    elif tangential_norm <= mu * normal_value:
        projected = shifted
    else:
        projected_normal = (normal_value + mu * tangential_norm) / (1.0 + mu * mu)
        if tangential_norm > 0.0:
            projected_tangential = mu * projected_normal * tangential / tangential_norm
        else:
            projected_tangential = np.zeros(3)
        projected = projected_normal * normal + projected_tangential

    return projected - adhesion * normal


def _yield_params(pmax, tensile_ratio, cohesion, friction_slope):
    return np.array(
        [pmax, tensile_ratio * pmax, cohesion, friction_slope * pmax, 1.0, 0.0],
        dtype=np.float32,
    )


def _stress_polygon_vertices(yield_params):
    pmax = float(yield_params[0])
    pmin = -max(0.0, float(yield_params[1]))
    cohesion = max(0.0, float(yield_params[2]))
    friction_slope = max(0.0, float(yield_params[3]) / pmax) if pmax > 0.0 else 0.0
    p1 = pmin + 0.5 * pmax
    p2 = 0.5 * pmax
    peak = cohesion + friction_slope * p2
    return np.array(
        [
            [pmin, 0.0],
            [pmin, cohesion],
            [p1, peak],
            [p2, peak],
            [pmax, cohesion],
            [pmax, 0.0],
        ]
    )


def _project_segment(point, start, end):
    segment = end - start
    length_squared = np.dot(segment, segment)
    if length_squared == 0.0:
        return start.copy()
    coordinate = np.clip(np.dot(point - start, segment) / length_squared, 0.0, 1.0)
    return start + coordinate * segment


def _stress_projection_reference(stress, yield_params):
    vertices = _stress_polygon_vertices(yield_params)
    normal_value = float(stress[0])
    tangential = np.array(stress, dtype=np.float64)
    tangential[0] = 0.0
    tangential_norm = np.linalg.norm(tangential)

    pmin = vertices[0, 0]
    pmax = vertices[-1, 0]
    friction_slope = max(0.0, float(yield_params[3]) / pmax) if pmax > 0.0 else 0.0
    cohesion = max(0.0, float(yield_params[2]))
    clamped_normal = np.clip(normal_value, pmin, pmax)
    p1 = pmin + 0.5 * pmax
    p2 = 0.5 * pmax
    if clamped_normal < p1:
        yield_stress = cohesion + friction_slope * (clamped_normal - pmin)
    elif clamped_normal > p2:
        yield_stress = cohesion + friction_slope * (pmax - clamped_normal)
    else:
        yield_stress = cohesion + friction_slope * p2

    if yield_params[0] >= 1.0e12 and yield_params[1] >= 1.0e12:
        projected = np.array(stress, dtype=np.float64)
        if tangential_norm > yield_stress:
            projected[1:] *= yield_stress / tangential_norm
        return projected

    if pmin <= normal_value <= pmax and tangential_norm <= yield_stress:
        return np.array(stress, dtype=np.float64)

    point = np.array([normal_value, tangential_norm])
    candidates = [_project_segment(point, vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]
    projected_point = min(candidates, key=lambda candidate: np.dot(candidate - point, candidate - point))

    projected = np.zeros(6)
    projected[0] = projected_point[0]
    if tangential_norm > 0.0:
        projected[1:] = projected_point[1] * tangential[1:] / tangential_norm
    return projected


def _make_bsr(rows_of_blocks, cols_of_blocks, offsets, columns, values, block_type, device):
    matrix = sp.bsr_zeros(rows_of_blocks, cols_of_blocks, block_type, device)
    matrix.offsets = wp.array(offsets, dtype=int, device=device)
    matrix.columns = wp.array(columns, dtype=int, device=device)
    matrix.values = wp.array(values, dtype=block_type, device=device)
    matrix.nnz = len(columns)
    return matrix


def _make_coupled_apgd_data(
    device,
    stress,
    impulse,
    friction=0.5,
    subgrid=False,
    body_inv_mass=0.0,
):
    strain_mat = _make_bsr(
        1,
        1,
        [0, 1],
        [0],
        [[1.0, 0.0, 0.0]],
        mat13,
        device,
    )
    mat31 = wp.types.matrix(shape=(3, 1), dtype=wp.float32)
    transposed_strain_mat = sp.bsr_zeros(1, 1, mat31, device)
    compliance_mat = _make_bsr(
        1,
        1,
        [0, 1],
        [0],
        [0.2 * np.eye(6, dtype=np.float32)],
        mat66,
        device,
    )
    if subgrid:
        collider_mat = _make_bsr(1, 1, [0, 1], [0], [1.0], float, device)
        transposed_collider_mat = sp.bsr_zeros(1, 1, float, device)
    else:
        collider_mat = sp.bsr_zeros(0, 0, float, device)
        transposed_collider_mat = sp.bsr_zeros(0, 0, float, device)

    rigidity_operator = None
    if body_inv_mass > 0.0:
        mat33 = wp.types.matrix(shape=(3, 3), dtype=wp.float32)
        identity = np.eye(3, dtype=np.float32)
        J = _make_bsr(1, 1, [0, 1], [0], [identity], mat33, device)
        IJtm = _make_bsr(1, 1, [0, 1], [0], [body_inv_mass * identity], mat33, device)
        rigidity_operator = (J, IJtm)

    velocity = np.array([[-0.4, 0.2, 0.0]], dtype=np.float32)
    collider_velocity = np.array([[0.1, -0.1, 0.0]], dtype=np.float32)
    strain_rhs = np.array([[0.1, -0.05, 0.0, 0.0, 0.0, -0.1]], dtype=np.float32)
    yield_params = _yield_params(pmax=4.0, tensile_ratio=0.5, cohesion=0.5, friction_slope=0.5)

    momentum = MomentumData(
        inv_volume=wp.array([1.0], dtype=float, device=device),
        velocity=wp.array(velocity, dtype=wp.vec3, device=device),
    )
    rheology = RheologyData(
        strain_mat=strain_mat,
        transposed_strain_mat=transposed_strain_mat,
        compliance_mat=compliance_mat,
        strain_node_volume=wp.array([1.0], dtype=float, device=device),
        yield_params=wp.array([yield_params], dtype=YieldParamVec, device=device),
        unilateral_strain_offset=wp.array([0.0], dtype=float, device=device),
        color_offsets=wp.array([0, 1], dtype=int, device=device),
        color_blocks=wp.array([[0], [1]], dtype=int, ndim=2, device=device),
        elastic_strain_delta=wp.array(strain_rhs, dtype=vec6, device=device),
        plastic_strain_delta=wp.zeros(1, dtype=vec6, device=device),
        stress=wp.array([stress], dtype=vec6, device=device),
        has_dilatancy=True,
    )
    collision = CollisionData(
        collider_mat=collider_mat,
        transposed_collider_mat=transposed_collider_mat,
        collider_friction=wp.array([friction], dtype=float, device=device),
        collider_adhesion=wp.array([0.1], dtype=float, device=device),
        collider_normals=wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3, device=device),
        collider_velocities=wp.array(collider_velocity, dtype=wp.vec3, device=device),
        rigidity_operator=rigidity_operator,
        collider_impulse=wp.array([impulse], dtype=wp.vec3, device=device),
        has_colliders=True,
    )
    return momentum, rheology, collision, velocity[0], collider_velocity[0], strain_rhs[0], yield_params


def _coupled_apgd_reference_step(
    stress,
    impulse,
    base_velocity,
    collider_velocity,
    strain_rhs,
    yield_params,
    step_size,
    body_inv_mass=0.0,
):
    trial_velocity = base_velocity + 0.5 * _STRAIN_OPERATOR.T @ stress + impulse
    trial_collider_velocity = collider_velocity - body_inv_mass * impulse

    stress_response = strain_rhs + _STRAIN_OPERATOR @ trial_velocity + 0.2 * stress
    viscosity_scale = float(yield_params[5])
    delassus = _STRAIN_OPERATOR @ _STRAIN_OPERATOR.T + (0.2 + 1.0e-6) * np.eye(6)
    delassus[0, 1:] = 0.0
    delassus[1:, 0] = 0.0
    viscous_delassus = np.eye(6) + viscosity_scale * delassus
    if yield_params[0] >= 1.0e12 and yield_params[1] >= 1.0e12:
        viscous_delassus[0, 0] = 1.0
    effective_delassus = np.linalg.solve(viscous_delassus, delassus)
    stress_preconditioner = np.trace(effective_delassus)
    yield_stress = stress + viscosity_scale * stress_response
    if yield_params[0] >= 1.0e12 and yield_params[1] >= 1.0e12:
        yield_stress[0] = stress[0]
    pmax = float(yield_params[0])
    pmin = -max(0.0, float(yield_params[1]))
    p1 = pmin + 0.5 * pmax
    p2 = 0.5 * pmax
    friction_slope = max(0.0, float(yield_params[3]) / pmax) if pmax > 0.0 else 0.0
    if yield_stress[0] < p1:
        active_slope = friction_slope
    elif yield_stress[0] > p2:
        active_slope = -friction_slope
    else:
        active_slope = 0.0
    corrected_stress_response = stress_response.copy()
    corrected_stress_response[0] += (1.0 - float(yield_params[4])) * active_slope * np.linalg.norm(stress_response[1:])
    trial_yield_stress = yield_stress - (step_size / stress_preconditioner) * corrected_stress_response
    projected_yield_stress = _stress_projection_reference(trial_yield_stress, yield_params)
    next_stress = stress + np.linalg.solve(viscous_delassus, projected_yield_stress - yield_stress)

    normal = np.array([1.0, 0.0, 0.0])
    relative_velocity = trial_velocity - trial_collider_velocity
    tangential_velocity = relative_velocity - np.dot(relative_velocity, normal) * normal
    corrected_velocity = relative_velocity + 0.5 * np.linalg.norm(tangential_velocity) * normal
    impulse_trial = impulse - (step_size / (1.0 + body_inv_mass)) * corrected_velocity
    next_impulse = _contact_projection_reference(0.5, normal, 0.1, impulse_trial)
    return next_stress, next_impulse


def _coupled_raw_response(
    stress,
    impulse,
    base_velocity,
    collider_velocity,
    strain_rhs,
    body_inv_mass=0.0,
):
    trial_velocity = base_velocity + 0.5 * _STRAIN_OPERATOR.T @ stress + impulse
    trial_collider_velocity = collider_velocity - body_inv_mass * impulse
    return (
        strain_rhs + _STRAIN_OPERATOR @ trial_velocity + 0.2 * stress,
        trial_velocity - trial_collider_velocity,
    )


def test_contact_projection_cases(test, device):
    """Project representative contact impulses onto adhesive Coulomb cones."""
    sqrt_half = np.sqrt(0.5)
    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [sqrt_half, sqrt_half, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    friction = np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.7, -1.0], dtype=np.float32)
    adhesion = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 2.0], dtype=np.float32)
    impulse = np.array(
        [
            [0.1, 0.1, 1.0],
            [0.5, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.2, 0.0, -2.0],
            [1.0, 0.0, -1.0],
            [2.0, -1.0, 3.0],
            [1.5, -0.5, 0.75],
            [4.0, -3.0, 2.0],
        ],
        dtype=np.float32,
    )

    expected = np.array(
        [
            _contact_projection_reference(mu, normal, shift, value)
            for mu, normal, shift, value in zip(friction, normals, adhesion, impulse, strict=True)
        ]
    )
    actual = _project_contacts(friction, normals, adhesion, impulse, device)

    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(actual[4], np.zeros(3), atol=1.0e-7)
    np.testing.assert_allclose(actual[5], np.array([0.0, 0.0, 3.0]), atol=1.0e-7)
    np.testing.assert_allclose(actual[7], np.zeros(3), atol=1.0e-7)


def test_contact_projection_properties(test, device):
    """Preserve feasibility and idempotence of contact projections."""
    rng = np.random.default_rng(8421)
    count = 128
    normals = rng.normal(size=(count, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    friction = rng.uniform(0.0, 1.5, size=count).astype(np.float32)
    friction[::17] = 0.0
    adhesion = rng.uniform(0.0, 1.0, size=count).astype(np.float32)
    impulse = rng.normal(size=(count, 3)).astype(np.float32) * 3.0

    projected = _project_contacts(friction, normals, adhesion, impulse, device)
    projected_twice = _project_contacts(friction, normals, adhesion, projected, device)
    expected = np.array(
        [
            _contact_projection_reference(mu, normal, shift, value)
            for mu, normal, shift, value in zip(friction, normals, adhesion, impulse, strict=True)
        ]
    )

    np.testing.assert_allclose(projected, expected, rtol=8.0e-6, atol=4.0e-6)
    np.testing.assert_allclose(projected_twice, projected, rtol=8.0e-6, atol=4.0e-6)

    shifted = projected + adhesion[:, None] * normals
    normal_values = np.sum(shifted * normals, axis=1)
    tangential_norms = np.linalg.norm(shifted - normal_values[:, None] * normals, axis=1)
    test.assertTrue(np.all(normal_values >= -4.0e-6))
    test.assertTrue(np.all(tangential_norms <= friction * normal_values + 8.0e-6))


def test_stress_projection_cases(test, device):
    """Project stresses onto every face and corner of the yield polygon."""
    standard = _yield_params(pmax=4.0, tensile_ratio=0.5, cohesion=1.0, friction_slope=0.5)
    zero_friction = _yield_params(pmax=4.0, tensile_ratio=0.5, cohesion=1.0, friction_slope=0.0)
    zero_pressure = _yield_params(pmax=0.0, tensile_ratio=0.0, cohesion=1.0, friction_slope=0.0)
    zero_surface = _yield_params(pmax=0.0, tensile_ratio=0.0, cohesion=0.0, friction_slope=0.0)

    reduced_points = [
        (1.0, 0.5),
        (-3.0, 0.5),
        (-1.2, 1.9),
        (1.0, 3.0),
        (3.2, 1.9),
        (5.0, 0.5),
        (-3.0, 0.0),
        (-3.0, 2.0),
        (-0.2, 3.0),
        (2.2, 3.0),
        (5.0, 2.0),
        (5.0, 0.0),
    ]
    stress = np.zeros((len(reduced_points) + 3, 6), dtype=np.float32)
    for index, (normal_value, tangential_norm) in enumerate(reduced_points):
        stress[index, 0] = normal_value
        stress[index, 1] = 0.6 * tangential_norm
        stress[index, 3] = 0.8 * tangential_norm
    stress[-3] = np.array([1.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    stress[-2] = np.array([-2.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    stress[-1] = np.array([3.0, 0.0, 4.0, 0.0, 0.0, 0.0])

    yield_params = np.repeat(standard[None, :], len(stress), axis=0)
    yield_params[-3] = zero_friction
    yield_params[-2] = zero_pressure
    yield_params[-1] = zero_surface

    expected = np.array(
        [_stress_projection_reference(value, params) for value, params in zip(stress, yield_params, strict=True)]
    )
    actual = _project_stresses(stress, yield_params, device)

    np.testing.assert_allclose(actual, expected, rtol=4.0e-6, atol=3.0e-6)

    fluid_stress = np.array([[-0.908248, 0.2, -0.1, 0.0, 0.0, 0.3]], dtype=np.float32)
    fluid_yield_params = np.array([_yield_params(1.0e15, 1.0, 0.0, 0.0)])
    expected_fluid_stress = fluid_stress.copy()
    expected_fluid_stress[:, 1:] = 0.0
    np.testing.assert_allclose(
        _project_stresses(fluid_stress, fluid_yield_params, device),
        expected_fluid_stress,
        rtol=0.0,
        atol=1.0e-6,
    )


def test_stress_projection_properties(test, device):
    """Preserve feasibility, direction, and idempotence of stress projections."""
    rng = np.random.default_rng(971)
    count = 192
    pmax = rng.uniform(0.1, 8.0, size=count)
    tensile_ratio = rng.uniform(0.0, 1.0, size=count)
    cohesion = rng.uniform(0.0, 2.0, size=count)
    friction_slope = rng.uniform(0.0, 1.2, size=count)
    friction_slope[::19] = 0.0
    yield_params = np.array(
        [
            _yield_params(pmax_value, ratio, cohesion_value, slope)
            for pmax_value, ratio, cohesion_value, slope in zip(
                pmax, tensile_ratio, cohesion, friction_slope, strict=True
            )
        ]
    )
    stress = rng.normal(size=(count, 6)).astype(np.float32) * 6.0

    projected = _project_stresses(stress, yield_params, device)
    projected_twice = _project_stresses(projected, yield_params, device)
    expected = np.array(
        [_stress_projection_reference(value, params) for value, params in zip(stress, yield_params, strict=True)]
    )

    np.testing.assert_allclose(projected, expected, rtol=1.0e-5, atol=6.0e-6)
    np.testing.assert_allclose(projected_twice, projected, rtol=1.0e-5, atol=6.0e-6)

    vertices = np.array([_stress_polygon_vertices(params) for params in yield_params])
    normal_values = projected[:, 0]
    tangential_norms = np.linalg.norm(projected[:, 1:], axis=1)
    pmin = vertices[:, 0, 0]
    pmax = vertices[:, -1, 0]
    p1 = vertices[:, 2, 0]
    p2 = vertices[:, 3, 0]
    cohesion = vertices[:, 1, 1]
    peak = vertices[:, 2, 1]
    rising = cohesion + (peak - cohesion) * (normal_values - pmin) / np.maximum(p1 - pmin, 1.0e-12)
    falling = cohesion + (peak - cohesion) * (pmax - normal_values) / np.maximum(pmax - p2, 1.0e-12)
    roof = np.where(normal_values < p1, rising, np.where(normal_values > p2, falling, peak))
    test.assertTrue(np.all(normal_values >= pmin - 6.0e-6))
    test.assertTrue(np.all(normal_values <= pmax + 6.0e-6))
    test.assertTrue(np.all(tangential_norms <= roof + 8.0e-6))

    input_tangential_norms = np.linalg.norm(stress[:, 1:], axis=1)
    nonzero = (input_tangential_norms > 1.0e-8) & (tangential_norms > 1.0e-8)
    normalized_input = stress[nonzero, 1:] / input_tangential_norms[nonzero, None]
    normalized_output = projected[nonzero, 1:] / tangential_norms[nonzero, None]
    np.testing.assert_allclose(normalized_output, normalized_input, rtol=8.0e-6, atol=8.0e-6)


def test_coupled_apgd_nodal_iteration(test, device):
    """Update stress and nodal contact from the same reconstructed trial velocity."""
    del test
    initial_stress = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)
    initial_impulse = np.array([0.2, 0.05, 0.0], dtype=np.float32)
    momentum, rheology, collision, base_velocity, collider_velocity, strain_rhs, yield_params = _make_coupled_apgd_data(
        device, initial_stress, initial_impulse
    )

    step_size = 0.25
    with wp.ScopedDevice(device):
        _solve_rheology_apgd_prototype(
            max_iterations=1,
            momentum=momentum,
            rheology=rheology,
            collision=collision,
            step_size=step_size,
        )

    expected_stress, expected_impulse = _coupled_apgd_reference_step(
        initial_stress,
        initial_impulse,
        base_velocity,
        collider_velocity,
        strain_rhs,
        yield_params,
        step_size,
    )

    actual_stress = rheology.stress.numpy()[0]
    actual_impulse = collision.collider_impulse.numpy()[0]
    actual_velocity = momentum.velocity.numpy()[0]
    expected_velocity = base_velocity + 0.5 * _STRAIN_OPERATOR.T @ expected_stress + expected_impulse

    np.testing.assert_allclose(actual_stress, expected_stress, rtol=2.0e-5, atol=5.0e-6)
    np.testing.assert_allclose(actual_impulse, expected_impulse, rtol=2.0e-5, atol=5.0e-6)
    np.testing.assert_allclose(actual_velocity, expected_velocity, rtol=2.0e-5, atol=5.0e-6)
    np.testing.assert_allclose(collision.collider_velocities.numpy()[0], collider_velocity, atol=1.0e-7)


def test_coupled_apgd_uses_orthogonal_stress_warmstart(test, device):
    """Project the original stress warm start orthogonally before reconstruction."""
    del test
    initial_stress = np.array([-3.0, 1.2, 1.6, 0.0, 0.0, 0.0], dtype=np.float32)
    initial_impulse = np.zeros(3, dtype=np.float32)
    momentum, rheology, collision, base_velocity, _collider_velocity, _strain_rhs, yield_params = (
        _make_coupled_apgd_data(device, initial_stress, initial_impulse)
    )

    with wp.ScopedDevice(device):
        _solve_rheology_apgd_prototype(
            max_iterations=0,
            momentum=momentum,
            rheology=rheology,
            collision=collision,
        )

    expected_stress = _stress_projection_reference(initial_stress, yield_params)
    expected_velocity = base_velocity + 0.5 * _STRAIN_OPERATOR.T @ expected_stress

    np.testing.assert_allclose(rheology.stress.numpy()[0], expected_stress, rtol=2.0e-5, atol=5.0e-6)
    np.testing.assert_allclose(momentum.velocity.numpy()[0], expected_velocity, rtol=2.0e-5, atol=5.0e-6)


def test_coupled_apgd_preserves_viscous_stress_warmstart(test, device):
    """Preserve total viscous stress outside the rate-independent yield set."""
    del test
    initial_stress = np.array([-3.0, 1.2, 1.6, 0.0, 0.0, 0.0], dtype=np.float32)
    initial_impulse = np.zeros(3, dtype=np.float32)
    momentum, rheology, collision, base_velocity, _collider_velocity, _strain_rhs, yield_params = (
        _make_coupled_apgd_data(device, initial_stress, initial_impulse)
    )
    yield_params[5] = 0.4
    rheology.yield_params.assign([yield_params])
    rheology.has_viscosity = True

    with wp.ScopedDevice(device):
        _solve_rheology_apgd_prototype(
            max_iterations=0,
            momentum=momentum,
            rheology=rheology,
            collision=collision,
        )

    expected_velocity = base_velocity + 0.5 * _STRAIN_OPERATOR.T @ initial_stress
    np.testing.assert_allclose(rheology.stress.numpy()[0], initial_stress, rtol=2.0e-5, atol=5.0e-6)
    np.testing.assert_allclose(momentum.velocity.numpy()[0], expected_velocity, rtol=2.0e-5, atol=5.0e-6)


def test_coupled_apgd_reaches_fixed_point(test, device):
    """Converge the coupled nodal prototype to its projected fixed point."""
    del test
    initial_stress = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)
    initial_impulse = np.array([0.2, 0.05, 0.0], dtype=np.float32)
    momentum, rheology, collision, base_velocity, collider_velocity, strain_rhs, yield_params = _make_coupled_apgd_data(
        device, initial_stress, initial_impulse
    )

    step_size = 0.25
    with wp.ScopedDevice(device):
        _solve_rheology_apgd_prototype(
            max_iterations=1200,
            momentum=momentum,
            rheology=rheology,
            collision=collision,
            step_size=step_size,
            accelerated=False,
        )

    stress = rheology.stress.numpy()[0]
    impulse = collision.collider_impulse.numpy()[0]
    next_stress, next_impulse = _coupled_apgd_reference_step(
        stress,
        impulse,
        base_velocity,
        collider_velocity,
        strain_rhs,
        yield_params,
        step_size,
    )

    np.testing.assert_allclose(stress, next_stress, rtol=2.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(impulse, next_impulse, rtol=2.0e-5, atol=1.0e-5)


def test_coupled_apgd_uses_raw_response_for_bb(test, device):
    """Compute the scaled BB step from raw, extrapolated operator differences."""
    initial_stress = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)
    initial_impulse = np.array([0.2, 0.05, 0.0], dtype=np.float32)
    momentum, rheology, collision, base_velocity, collider_velocity, strain_rhs, _yield_params = (
        _make_coupled_apgd_data(device, initial_stress, initial_impulse)
    )

    with wp.ScopedDevice(device):
        result = _solve_rheology_apgd_prototype(
            max_iterations=1,
            momentum=momentum,
            rheology=rheology,
            collision=collision,
            accelerated=True,
        )

    next_stress = rheology.stress.numpy()[0]
    next_impulse = collision.collider_impulse.numpy()[0]
    initial_response = _coupled_raw_response(
        initial_stress,
        initial_impulse,
        base_velocity,
        collider_velocity,
        strain_rhs,
    )
    next_response = _coupled_raw_response(
        next_stress,
        next_impulse,
        base_velocity,
        collider_velocity,
        strain_rhs,
    )
    delta_stress = next_stress - initial_stress
    delta_impulse = next_impulse - initial_impulse
    delta_stress_response = next_response[0] - initial_response[0]
    delta_contact_response = next_response[1] - initial_response[1]

    stress_preconditioner = 4.0 + 6.0 * 0.2 + 6.0e-6
    expected_numerator = np.dot(delta_stress, delta_stress_response) + np.dot(delta_impulse, delta_contact_response)
    expected_denominator = np.dot(delta_stress_response, delta_stress_response) / stress_preconditioner
    expected_denominator += np.dot(delta_contact_response, delta_contact_response)

    np.testing.assert_allclose(result.bb_numerator, expected_numerator, rtol=3.0e-5, atol=2.0e-7)
    np.testing.assert_allclose(result.bb_denominator, expected_denominator, rtol=3.0e-5, atol=2.0e-7)
    test.assertGreater(result.restart_dot, 0.0)

    normal = np.array([1.0, 0.0, 0.0])

    def corrected_contact_response(response):
        tangential = response - np.dot(response, normal) * normal
        return response + 0.5 * np.linalg.norm(tangential) * normal

    corrected_delta = corrected_contact_response(next_response[1]) - corrected_contact_response(initial_response[1])
    corrected_numerator = np.dot(delta_stress, delta_stress_response) + np.dot(delta_impulse, corrected_delta)
    test.assertGreater(abs(float(result.bb_numerator - corrected_numerator)), 1.0e-5)


def test_coupled_apgd_accelerates_with_restart(test, device):
    """Converge both projected residual blocks with coupled inertia and restart."""
    initial_stress = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)
    initial_impulse = np.array([0.2, 0.05, 0.0], dtype=np.float32)
    momentum, rheology, collision, _base_velocity, _collider_velocity, _strain_rhs, _yield_params = (
        _make_coupled_apgd_data(device, initial_stress, initial_impulse)
    )

    tolerance = 1.0e-4
    with wp.ScopedDevice(device):
        result = _solve_rheology_apgd_prototype(
            max_iterations=200,
            momentum=momentum,
            rheology=rheology,
            collision=collision,
            accelerated=True,
            residual_tolerance=tolerance,
        )

    test.assertLess(result.iteration_count, 200)
    test.assertLessEqual(result.residual[0], tolerance)
    test.assertLessEqual(result.residual[1], tolerance)
    test.assertGreater(result.restart_count, 0)
    test.assertGreaterEqual(result.step_size, 1.0e-4)
    test.assertLessEqual(result.step_size, 0.5)


def test_coupled_apgd_handles_non_associated_viscosity(test, device):
    """Apply non-associated bias and viscosity in one coupled APGD step."""
    del test
    initial_stress = np.array([-0.5, 0.3, 0.0, 0.0, 0.0, 0.2], dtype=np.float32)
    initial_impulse = np.array([0.2, 0.05, 0.0], dtype=np.float32)
    momentum, rheology, collision, base_velocity, collider_velocity, strain_rhs, yield_params = _make_coupled_apgd_data(
        device, initial_stress, initial_impulse
    )
    yield_params[4] = 0.25
    yield_params[5] = 0.4
    rheology.yield_params.assign([yield_params])
    rheology.has_dilatancy = True
    rheology.has_viscosity = True

    step_size = 0.25
    with wp.ScopedDevice(device):
        _solve_rheology_apgd_prototype(
            max_iterations=1,
            momentum=momentum,
            rheology=rheology,
            collision=collision,
            step_size=step_size,
        )

    expected_stress, expected_impulse = _coupled_apgd_reference_step(
        initial_stress,
        initial_impulse,
        base_velocity,
        collider_velocity,
        strain_rhs,
        yield_params,
        step_size,
    )
    expected_velocity = base_velocity + 0.5 * _STRAIN_OPERATOR.T @ expected_stress + expected_impulse

    np.testing.assert_allclose(rheology.stress.numpy()[0], expected_stress, rtol=3.0e-5, atol=8.0e-6)
    np.testing.assert_allclose(collision.collider_impulse.numpy()[0], expected_impulse, rtol=2.0e-5, atol=5.0e-6)
    np.testing.assert_allclose(momentum.velocity.numpy()[0], expected_velocity, rtol=3.0e-5, atol=8.0e-6)


def test_coupled_apgd_matches_gs_with_non_associated_viscosity(test, device):
    """Match the converged GS solution with non-associated viscous flow."""
    del test
    initial_stress = np.array([-0.5, 0.3, 0.0, 0.0, 0.0, 0.2], dtype=np.float32)
    initial_impulse = np.array([0.2, 0.05, 0.0], dtype=np.float32)
    apgd = _make_coupled_apgd_data(device, initial_stress, initial_impulse)
    gs = _make_coupled_apgd_data(device, initial_stress, initial_impulse)
    yield_params = apgd[6].copy()
    yield_params[4] = 0.25
    yield_params[5] = 0.4

    for _momentum, rheology, _collision, *_rest in (apgd, gs):
        rheology.yield_params.assign([yield_params])
        rheology.has_dilatancy = True
        rheology.has_viscosity = True

    with wp.ScopedDevice(device):
        _solve_rheology_apgd_prototype(
            max_iterations=400,
            momentum=apgd[0],
            rheology=apgd[1],
            collision=apgd[2],
            residual_tolerance=1.0e-6,
        )
        solve_rheology(
            "gs",
            max_iterations=400,
            tolerance=1.0e-6,
            momentum=gs[0],
            rheology=gs[1],
            collision=gs[2],
            use_graph=False,
            verbose=False,
        )

    np.testing.assert_allclose(apgd[1].stress.numpy(), gs[1].stress.numpy(), rtol=3.0e-5, atol=8.0e-6)
    np.testing.assert_allclose(
        apgd[2].collider_impulse.numpy(),
        gs[2].collider_impulse.numpy(),
        rtol=3.0e-5,
        atol=8.0e-6,
    )
    np.testing.assert_allclose(apgd[0].velocity.numpy(), gs[0].velocity.numpy(), rtol=3.0e-5, atol=8.0e-6)


def test_coupled_apgd_matches_gs_with_large_viscosity(test, device):
    """Match GS while keeping a large-viscosity APGD solve finite."""
    initial_stress = np.array([-0.5, 0.3, 0.0, 0.0, 0.0, 0.2], dtype=np.float32)
    initial_impulse = np.array([0.2, 0.05, 0.0], dtype=np.float32)
    apgd = _make_coupled_apgd_data(device, initial_stress, initial_impulse)
    gs = _make_coupled_apgd_data(device, initial_stress, initial_impulse)
    yield_params = apgd[6].copy()
    yield_params[4] = 0.25
    yield_params[5] = 1.0e4

    for _momentum, rheology, _collision, *_rest in (apgd, gs):
        rheology.yield_params.assign([yield_params])
        rheology.has_dilatancy = True
        rheology.has_viscosity = True

    with wp.ScopedDevice(device):
        result = _solve_rheology_apgd_prototype(
            max_iterations=200,
            momentum=apgd[0],
            rheology=apgd[1],
            collision=apgd[2],
            residual_tolerance=1.0e-6,
        )
        solve_rheology(
            "gs",
            max_iterations=400,
            tolerance=1.0e-6,
            momentum=gs[0],
            rheology=gs[1],
            collision=gs[2],
            use_graph=False,
            verbose=False,
        )

    apgd_stress = apgd[1].stress.numpy()
    apgd_impulse = apgd[2].collider_impulse.numpy()
    apgd_velocity = apgd[0].velocity.numpy()
    test.assertLess(result.iteration_count, 200)
    test.assertTrue(np.isfinite(apgd_stress).all())
    test.assertTrue(np.isfinite(apgd_impulse).all())
    test.assertTrue(np.isfinite(apgd_velocity).all())
    np.testing.assert_allclose(apgd_stress, gs[1].stress.numpy(), rtol=3.0e-5, atol=8.0e-6)
    np.testing.assert_allclose(apgd_impulse, gs[2].collider_impulse.numpy(), rtol=3.0e-5, atol=8.0e-6)
    np.testing.assert_allclose(apgd_velocity, gs[0].velocity.numpy(), rtol=3.0e-5, atol=8.0e-6)


def test_coupled_apgd_matches_gs_for_incompressible_viscous_flow(test, device):
    """Match GS for viscous flow with an unconstrained pressure multiplier."""
    del test
    initial_stress = np.array([-0.5, 0.3, 0.0, 0.0, 0.0, 0.2], dtype=np.float32)
    initial_impulse = np.array([0.2, 0.05, 0.0], dtype=np.float32)
    apgd = _make_coupled_apgd_data(device, initial_stress, initial_impulse)
    gs = _make_coupled_apgd_data(device, initial_stress, initial_impulse)
    yield_params = _yield_params(1.0e15, 1.0, 0.0, 0.0)
    yield_params[5] = 1.0e4

    for _momentum, rheology, _collision, *_rest in (apgd, gs):
        rheology.yield_params.assign([yield_params])
        rheology.has_viscosity = True

    with wp.ScopedDevice(device):
        _solve_rheology_apgd_prototype(
            max_iterations=1000,
            momentum=apgd[0],
            rheology=apgd[1],
            collision=apgd[2],
            residual_tolerance=1.0e-6,
        )
        solve_rheology(
            "gs",
            max_iterations=400,
            tolerance=1.0e-6,
            momentum=gs[0],
            rheology=gs[1],
            collision=gs[2],
            use_graph=False,
            verbose=False,
        )

    np.testing.assert_allclose(apgd[1].stress.numpy(), gs[1].stress.numpy(), rtol=3.0e-5, atol=8.0e-6)
    np.testing.assert_allclose(
        apgd[2].collider_impulse.numpy(),
        gs[2].collider_impulse.numpy(),
        rtol=3.0e-5,
        atol=8.0e-6,
    )
    np.testing.assert_allclose(apgd[0].velocity.numpy(), gs[0].velocity.numpy(), rtol=3.0e-5, atol=8.0e-6)


def test_coupled_apgd_uses_fixed_diagnostic_step(test, device):
    """Keep convergence diagnostics independent of the adaptive update step."""
    initial_stress = np.array([-0.5, 0.3, 0.0, 0.0, 0.0, 0.2], dtype=np.float32)
    initial_impulse = np.array([0.2, 0.05, 0.0], dtype=np.float32)
    slow = _make_coupled_apgd_data(device, initial_stress, initial_impulse)
    fast = _make_coupled_apgd_data(device, initial_stress, initial_impulse)

    with wp.ScopedDevice(device):
        slow_result = _solve_rheology_apgd_prototype(
            max_iterations=1,
            momentum=slow[0],
            rheology=slow[1],
            collision=slow[2],
            step_size=0.1,
            diagnostic_step_size=0.2,
        )
        fast_result = _solve_rheology_apgd_prototype(
            max_iterations=1,
            momentum=fast[0],
            rheology=fast[1],
            collision=fast[2],
            step_size=0.4,
            diagnostic_step_size=0.2,
        )

    np.testing.assert_allclose(slow_result.residual, fast_result.residual, rtol=2.0e-6, atol=2.0e-7)
    test.assertFalse(np.allclose(slow[1].stress.numpy(), fast[1].stress.numpy()))


def test_solve_rheology_dispatches_apgd(test, device):
    """Dispatch the public rheology solver selector to coupled APGD."""
    initial_stress = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)
    initial_impulse = np.array([0.2, 0.05, 0.0], dtype=np.float32)
    momentum, rheology, collision, *_rest = _make_coupled_apgd_data(device, initial_stress, initial_impulse)

    with wp.ScopedDevice(device):
        solve_rheology(
            "apgd",
            max_iterations=1,
            tolerance=0.0,
            momentum=momentum,
            rheology=rheology,
            collision=collision,
            use_graph=False,
        )

    test.assertFalse(np.allclose(rheology.stress.numpy()[0], initial_stress))
    with test.assertRaisesRegex(ValueError, "only rheology solver"):
        solve_rheology(
            ("jacobi", "apgd"),
            max_iterations=1,
            tolerance=0.0,
            momentum=momentum,
            rheology=rheology,
            collision=collision,
            use_graph=False,
        )


def test_solve_rheology_error_mentions_apgd(test, device):
    """List APGD among the accepted solver values."""
    momentum, rheology, collision, *_rest = _make_coupled_apgd_data(
        device,
        stress=np.zeros(6, dtype=np.float32),
        impulse=np.zeros(3, dtype=np.float32),
    )

    with test.assertRaisesRegex(ValueError, r"Accepted values: .*'apgd'"):
        solve_rheology(
            "gsa",
            max_iterations=1,
            tolerance=0.0,
            momentum=momentum,
            rheology=rheology,
            collision=collision,
            use_graph=False,
        )


def test_coupled_apgd_reductions_stay_on_device(test, device):
    """Use only preallocated reductions inside the iteration loop."""
    initial_stress = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)
    initial_impulse = np.array([0.2, 0.05, 0.0], dtype=np.float32)
    momentum, rheology, collision, _base_velocity, _collider_velocity, _strain_rhs, _yield_params = (
        _make_coupled_apgd_data(device, initial_stress, initial_impulse)
    )

    with wp.ScopedDevice(device):
        with mock.patch.object(wp.utils, "array_sum", side_effect=AssertionError("unexpected allocating reduction")):
            result = _solve_rheology_apgd_prototype(
                max_iterations=3,
                momentum=momentum,
                rheology=rheology,
                collision=collision,
            )

    test.assertEqual(result.iteration_count, 3)


def test_coupled_apgd_contact_limits(test, device):
    """Handle disabled and frictionless contact in the coupled iteration."""
    initial_stress = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)
    initial_impulse = np.array([0.2, 0.05, -0.1], dtype=np.float32)

    disabled = _make_coupled_apgd_data(device, initial_stress, initial_impulse, friction=-1.0)
    with wp.ScopedDevice(device):
        _solve_rheology_apgd_prototype(
            max_iterations=1,
            momentum=disabled[0],
            rheology=disabled[1],
            collision=disabled[2],
        )
    np.testing.assert_allclose(disabled[2].collider_impulse.numpy()[0], np.zeros(3), atol=1.0e-7)

    frictionless = _make_coupled_apgd_data(device, initial_stress, initial_impulse, friction=0.0)
    with wp.ScopedDevice(device):
        _solve_rheology_apgd_prototype(
            max_iterations=10,
            momentum=frictionless[0],
            rheology=frictionless[1],
            collision=frictionless[2],
        )
    impulse = frictionless[2].collider_impulse.numpy()[0]
    np.testing.assert_allclose(impulse[1:], np.zeros(2), atol=1.0e-7)
    test.assertGreaterEqual(float(impulse[0] + 0.1), -1.0e-7)


def test_coupled_apgd_subgrid_matches_nodal(test, device):
    """Match nodal contact with an equivalent identity subgrid operator."""
    del test
    initial_stress = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)
    initial_impulse = np.array([0.2, 0.05, 0.0], dtype=np.float32)
    nodal = _make_coupled_apgd_data(device, initial_stress, initial_impulse, body_inv_mass=0.35)
    subgrid = _make_coupled_apgd_data(
        device,
        initial_stress,
        initial_impulse,
        subgrid=True,
        body_inv_mass=0.35,
    )

    with wp.ScopedDevice(device):
        _solve_rheology_apgd_prototype(
            max_iterations=20,
            momentum=nodal[0],
            rheology=nodal[1],
            collision=nodal[2],
        )
        _solve_rheology_apgd_prototype(
            max_iterations=20,
            momentum=subgrid[0],
            rheology=subgrid[1],
            collision=subgrid[2],
        )

    np.testing.assert_allclose(subgrid[1].stress.numpy(), nodal[1].stress.numpy(), rtol=2.0e-5, atol=5.0e-6)
    np.testing.assert_allclose(
        subgrid[2].collider_impulse.numpy(),
        nodal[2].collider_impulse.numpy(),
        rtol=2.0e-5,
        atol=5.0e-6,
    )
    np.testing.assert_allclose(
        subgrid[0].velocity.numpy(),
        nodal[0].velocity.numpy(),
        rtol=2.0e-5,
        atol=5.0e-6,
    )
    np.testing.assert_allclose(
        subgrid[2].collider_velocities.numpy(),
        nodal[2].collider_velocities.numpy(),
        rtol=2.0e-5,
        atol=5.0e-6,
    )


def test_coupled_apgd_rigid_response(test, device):
    """Include rigid collider velocity and inverse mass in one coupled step."""
    del test
    initial_stress = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)
    initial_impulse = np.array([0.2, 0.05, 0.0], dtype=np.float32)
    body_inv_mass = 0.75
    momentum, rheology, collision, base_velocity, collider_velocity, strain_rhs, yield_params = _make_coupled_apgd_data(
        device,
        initial_stress,
        initial_impulse,
        body_inv_mass=body_inv_mass,
    )

    step_size = 0.25
    with wp.ScopedDevice(device):
        _solve_rheology_apgd_prototype(
            max_iterations=1,
            momentum=momentum,
            rheology=rheology,
            collision=collision,
            step_size=step_size,
        )

    expected_stress, expected_impulse = _coupled_apgd_reference_step(
        initial_stress,
        initial_impulse,
        base_velocity,
        collider_velocity,
        strain_rhs,
        yield_params,
        step_size,
        body_inv_mass=body_inv_mass,
    )
    expected_velocity = base_velocity + 0.5 * _STRAIN_OPERATOR.T @ expected_stress + expected_impulse
    expected_collider_velocity = collider_velocity - body_inv_mass * expected_impulse

    np.testing.assert_allclose(rheology.stress.numpy()[0], expected_stress, rtol=2.0e-5, atol=5.0e-6)
    np.testing.assert_allclose(collision.collider_impulse.numpy()[0], expected_impulse, rtol=2.0e-5, atol=5.0e-6)
    np.testing.assert_allclose(momentum.velocity.numpy()[0], expected_velocity, rtol=2.0e-5, atol=5.0e-6)
    np.testing.assert_allclose(
        collision.collider_velocities.numpy()[0],
        expected_collider_velocity,
        rtol=2.0e-5,
        atol=5.0e-6,
    )


class TestImplicitMPMAPGDProjections(unittest.TestCase):
    pass


add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_contact_projection_cases",
    test_contact_projection_cases,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_apgd_stress_evaluation_skips_inactive_capacity_rows",
    test_apgd_stress_evaluation_skips_inactive_capacity_rows,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_contact_projection_properties",
    test_contact_projection_properties,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_stress_projection_cases",
    test_stress_projection_cases,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_stress_projection_properties",
    test_stress_projection_properties,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_coupled_apgd_nodal_iteration",
    test_coupled_apgd_nodal_iteration,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_coupled_apgd_uses_orthogonal_stress_warmstart",
    test_coupled_apgd_uses_orthogonal_stress_warmstart,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_coupled_apgd_preserves_viscous_stress_warmstart",
    test_coupled_apgd_preserves_viscous_stress_warmstart,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_coupled_apgd_reaches_fixed_point",
    test_coupled_apgd_reaches_fixed_point,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_coupled_apgd_uses_raw_response_for_bb",
    test_coupled_apgd_uses_raw_response_for_bb,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_coupled_apgd_accelerates_with_restart",
    test_coupled_apgd_accelerates_with_restart,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_coupled_apgd_handles_non_associated_viscosity",
    test_coupled_apgd_handles_non_associated_viscosity,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_coupled_apgd_matches_gs_with_non_associated_viscosity",
    test_coupled_apgd_matches_gs_with_non_associated_viscosity,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_coupled_apgd_matches_gs_with_large_viscosity",
    test_coupled_apgd_matches_gs_with_large_viscosity,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_coupled_apgd_matches_gs_for_incompressible_viscous_flow",
    test_coupled_apgd_matches_gs_for_incompressible_viscous_flow,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_coupled_apgd_uses_fixed_diagnostic_step",
    test_coupled_apgd_uses_fixed_diagnostic_step,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_solve_rheology_dispatches_apgd",
    test_solve_rheology_dispatches_apgd,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_solve_rheology_error_mentions_apgd",
    test_solve_rheology_error_mentions_apgd,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_coupled_apgd_reductions_stay_on_device",
    test_coupled_apgd_reductions_stay_on_device,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_coupled_apgd_contact_limits",
    test_coupled_apgd_contact_limits,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_coupled_apgd_subgrid_matches_nodal",
    test_coupled_apgd_subgrid_matches_nodal,
    devices=devices,
)
add_function_test(
    TestImplicitMPMAPGDProjections,
    "test_coupled_apgd_rigid_response",
    test_coupled_apgd_rigid_response,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
