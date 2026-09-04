# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-world timestep validation for LOX."""

from __future__ import annotations

import warp as wp

__all__ = ["validate_world_time_step", "validate_world_time_steps"]


def validate_world_time_step(
    time_step: wp.array[wp.float32],
    world_count: int,
    device: wp.DeviceLike,
) -> wp.array[wp.float32]:
    """Validate a per-world timestep array without reading it on the host."""
    if not isinstance(time_step, wp.array) or time_step.dtype != wp.float32 or time_step.shape != (world_count,):
        raise ValueError(f"time_step must have shape ({world_count},) and dtype float32.")
    if time_step.device != wp.get_device(device):
        raise ValueError(f"time_step must be allocated on {wp.get_device(device)}, found {time_step.device}.")
    return time_step


def validate_world_time_steps(
    time_step: wp.array[wp.float32],
    inverse_time_step: wp.array[wp.float32],
    world_count: int,
    device: wp.DeviceLike,
) -> tuple[wp.array[wp.float32], wp.array[wp.float32]]:
    """Validate matching per-world timestep and inverse-timestep arrays."""
    validate_world_time_step(time_step, world_count, device)
    if (
        not isinstance(inverse_time_step, wp.array)
        or inverse_time_step.dtype != wp.float32
        or inverse_time_step.shape != (world_count,)
    ):
        raise ValueError(f"inverse_time_step must have shape ({world_count},) and dtype float32.")
    if inverse_time_step.device != time_step.device:
        raise ValueError(
            f"inverse_time_step must be allocated on {time_step.device}, found {inverse_time_step.device}."
        )
    return time_step, inverse_time_step
