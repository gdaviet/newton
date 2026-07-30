# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

APGD_STATE_THETA = 0
APGD_STATE_STEP_SIZE = 1
APGD_STATE_BETA = 2
APGD_STATE_ITERATION_COUNT = 3
APGD_STATE_RESTART_COUNT = 4
APGD_STATE_STRESS_RESIDUAL = 5
APGD_STATE_CONTACT_RESIDUAL = 6
APGD_STATE_RESTART_DOT = 7
APGD_STATE_BB_NUMERATOR = 8
APGD_STATE_BB_DENOMINATOR = 9
APGD_STATE_STRESS_RESIDUAL_INF = 10
APGD_STATE_CONTACT_RESIDUAL_INF = 11
APGD_STATE_SIZE = 12

wp.set_module_options({"enable_backward": False})


@wp.kernel
def apgd_initialize_state(
    step_size: float,
    max_iterations: int,
    state: wp.array2d[float],
    condition: wp.array[int],
):
    """Initialize device-resident APGD acceleration and diagnostic state."""

    environment = wp.tid()
    state[APGD_STATE_THETA, environment] = 1.0
    state[APGD_STATE_STEP_SIZE, environment] = step_size
    state[APGD_STATE_BETA, environment] = 0.0
    state[APGD_STATE_ITERATION_COUNT, environment] = 0.0
    state[APGD_STATE_RESTART_COUNT, environment] = 0.0
    state[APGD_STATE_STRESS_RESIDUAL, environment] = 1.0e30
    state[APGD_STATE_CONTACT_RESIDUAL, environment] = 1.0e30
    state[APGD_STATE_RESTART_DOT, environment] = 0.0
    state[APGD_STATE_BB_NUMERATOR, environment] = 0.0
    state[APGD_STATE_BB_DENOMINATOR, environment] = 0.0
    state[APGD_STATE_STRESS_RESIDUAL_INF, environment] = 1.0e30
    state[APGD_STATE_CONTACT_RESIDUAL_INF, environment] = 1.0e30
    if environment == 0:
        condition[0] = wp.where(max_iterations > 0, 1, 0)


@wp.kernel
def apgd_finalize_restart(
    accelerated: bool,
    stress_metrics: wp.array2d[float],
    contact_metrics: wp.array2d[float],
    stress_l2_scale: wp.array[float],
    contact_l2_scale: wp.array[float],
    state: wp.array2d[float],
    condition: wp.array[int],
):
    """Update residuals, restart state, and inertia from reduced metrics."""

    if condition[0] == 0:
        return

    environment = wp.tid()
    restart_dot = stress_metrics[0, environment] + contact_metrics[0, environment]
    stress_residual = wp.sqrt(wp.max(0.0, stress_metrics[1, environment])) / stress_l2_scale[environment]
    contact_residual = wp.sqrt(wp.max(0.0, contact_metrics[1, environment])) / contact_l2_scale[environment]
    stress_residual_inf = wp.sqrt(wp.max(0.0, stress_metrics[2, environment]))
    contact_residual_inf = wp.sqrt(wp.max(0.0, contact_metrics[2, environment]))

    state[APGD_STATE_ITERATION_COUNT, environment] += 1.0
    state[APGD_STATE_STRESS_RESIDUAL, environment] = stress_residual
    state[APGD_STATE_CONTACT_RESIDUAL, environment] = contact_residual
    state[APGD_STATE_STRESS_RESIDUAL_INF, environment] = stress_residual_inf
    state[APGD_STATE_CONTACT_RESIDUAL_INF, environment] = contact_residual_inf
    state[APGD_STATE_RESTART_DOT, environment] = restart_dot

    if not accelerated:
        state[APGD_STATE_BETA, environment] = 0.0
        return

    theta = state[APGD_STATE_THETA, environment]
    valid_theta = wp.isfinite(theta) and theta > 0.0 and theta <= 1.0
    if not valid_theta or not wp.isfinite(restart_dot) or restart_dot <= 0.0:
        state[APGD_STATE_THETA, environment] = 1.0
        state[APGD_STATE_BETA, environment] = 0.0
        state[APGD_STATE_RESTART_COUNT, environment] += 1.0
        return

    next_theta = 2.0 * theta / (wp.sqrt(theta * theta + 4.0) + theta)
    state[APGD_STATE_BETA, environment] = theta * (1.0 - theta) / (theta * theta + next_theta)
    state[APGD_STATE_THETA, environment] = next_theta


@wp.kernel
def apgd_finalize_bb(
    accelerated: bool,
    min_step_size: float,
    max_step_size: float,
    stress_metrics: wp.array2d[float],
    contact_metrics: wp.array2d[float],
    state: wp.array2d[float],
    condition: wp.array[int],
):
    """Update the guarded spectral step from reduced raw-response metrics."""

    if condition[0] == 0:
        return

    environment = wp.tid()
    numerator = stress_metrics[0, environment] + contact_metrics[0, environment]
    denominator = stress_metrics[1, environment] + contact_metrics[1, environment]
    state[APGD_STATE_BB_NUMERATOR, environment] = numerator
    state[APGD_STATE_BB_DENOMINATOR, environment] = denominator

    if not accelerated:
        return
    if not wp.isfinite(numerator) or not wp.isfinite(denominator):
        return
    if numerator <= 0.0 or denominator <= wp.float32(1.1920928955078125e-7):
        return

    spectral_step = numerator / denominator
    if wp.isfinite(spectral_step):
        state[APGD_STATE_STEP_SIZE, environment] = wp.clamp(spectral_step, min_step_size, max_step_size)


@wp.kernel
def apgd_finalize_iteration(
    max_iterations: int,
    stress_residual_tolerance: float,
    contact_residual_tolerance: float,
    state: wp.array2d[float],
    condition: wp.array[int],
):
    """Update the device loop condition after a complete APGD iteration."""

    if condition[0] == 0:
        return

    converged = stress_residual_tolerance >= 0.0 and contact_residual_tolerance >= 0.0
    iteration_count = int(state[APGD_STATE_ITERATION_COUNT, 0])
    for environment in range(state.shape[1]):
        converged = converged and state[APGD_STATE_STRESS_RESIDUAL, environment] <= stress_residual_tolerance
        converged = converged and state[APGD_STATE_CONTACT_RESIDUAL, environment] <= contact_residual_tolerance
        converged = converged and state[APGD_STATE_STRESS_RESIDUAL_INF, environment] <= stress_residual_tolerance
        converged = converged and state[APGD_STATE_CONTACT_RESIDUAL_INF, environment] <= contact_residual_tolerance
    if converged or iteration_count >= max_iterations:
        condition[0] = 0
