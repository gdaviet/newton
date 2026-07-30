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
APGD_STATE_SIZE = 10

wp.set_module_options({"enable_backward": False})


@wp.kernel
def apgd_initialize_state(
    step_size: float,
    max_iterations: int,
    state: wp.array[float],
    condition: wp.array[int],
):
    """Initialize device-resident APGD acceleration and diagnostic state."""

    state[APGD_STATE_THETA] = 1.0
    state[APGD_STATE_STEP_SIZE] = step_size
    state[APGD_STATE_BETA] = 0.0
    state[APGD_STATE_ITERATION_COUNT] = 0.0
    state[APGD_STATE_RESTART_COUNT] = 0.0
    state[APGD_STATE_STRESS_RESIDUAL] = 1.0e30
    state[APGD_STATE_CONTACT_RESIDUAL] = 1.0e30
    state[APGD_STATE_RESTART_DOT] = 0.0
    state[APGD_STATE_BB_NUMERATOR] = 0.0
    state[APGD_STATE_BB_DENOMINATOR] = 0.0
    condition[0] = wp.where(max_iterations > 0, 1, 0)


@wp.kernel
def apgd_finalize_restart(
    accelerated: bool,
    stress_metrics: wp.array2d[float],
    contact_metrics: wp.array2d[float],
    state: wp.array[float],
    condition: wp.array[int],
):
    """Update residuals, restart state, and inertia from reduced metrics."""

    if condition[0] == 0:
        return

    restart_dot = stress_metrics[0, 0] + contact_metrics[0, 0]
    stress_residual = wp.sqrt(wp.max(0.0, stress_metrics[1, 0]))
    contact_residual = wp.sqrt(wp.max(0.0, contact_metrics[1, 0]))

    state[APGD_STATE_ITERATION_COUNT] += 1.0
    state[APGD_STATE_STRESS_RESIDUAL] = stress_residual
    state[APGD_STATE_CONTACT_RESIDUAL] = contact_residual
    state[APGD_STATE_RESTART_DOT] = restart_dot

    if not accelerated:
        state[APGD_STATE_BETA] = 0.0
        return

    theta = state[APGD_STATE_THETA]
    valid_theta = wp.isfinite(theta) and theta > 0.0 and theta <= 1.0
    if not valid_theta or not wp.isfinite(restart_dot) or restart_dot <= 0.0:
        state[APGD_STATE_THETA] = 1.0
        state[APGD_STATE_BETA] = 0.0
        state[APGD_STATE_RESTART_COUNT] += 1.0
        return

    next_theta = 2.0 * theta / (wp.sqrt(theta * theta + 4.0) + theta)
    state[APGD_STATE_BETA] = theta * (1.0 - theta) / (theta * theta + next_theta)
    state[APGD_STATE_THETA] = next_theta


@wp.kernel
def apgd_finalize_bb(
    accelerated: bool,
    min_step_size: float,
    max_step_size: float,
    stress_metrics: wp.array2d[float],
    contact_metrics: wp.array2d[float],
    state: wp.array[float],
    condition: wp.array[int],
):
    """Update the guarded spectral step from reduced raw-response metrics."""

    if condition[0] == 0:
        return

    numerator = stress_metrics[0, 0] + contact_metrics[0, 0]
    denominator = stress_metrics[1, 0] + contact_metrics[1, 0]
    state[APGD_STATE_BB_NUMERATOR] = numerator
    state[APGD_STATE_BB_DENOMINATOR] = denominator

    if not accelerated:
        return
    if not wp.isfinite(numerator) or not wp.isfinite(denominator):
        return
    if numerator <= 0.0 or denominator <= wp.float32(1.1920928955078125e-7):
        return

    spectral_step = numerator / denominator
    if wp.isfinite(spectral_step):
        state[APGD_STATE_STEP_SIZE] = wp.clamp(spectral_step, min_step_size, max_step_size)


@wp.kernel
def apgd_finalize_iteration(
    max_iterations: int,
    stress_residual_tolerance: float,
    contact_residual_tolerance: float,
    state: wp.array[float],
    condition: wp.array[int],
):
    """Update the device loop condition after a complete APGD iteration."""

    if condition[0] == 0:
        return

    iteration_count = int(state[APGD_STATE_ITERATION_COUNT])
    converged = stress_residual_tolerance >= 0.0 and contact_residual_tolerance >= 0.0
    converged = converged and state[APGD_STATE_STRESS_RESIDUAL] <= stress_residual_tolerance
    converged = converged and state[APGD_STATE_CONTACT_RESIDUAL] <= contact_residual_tolerance
    if converged or iteration_count >= max_iterations:
        condition[0] = 0
