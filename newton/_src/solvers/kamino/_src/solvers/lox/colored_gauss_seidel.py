# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-batched Gauss--Seidel projection with final Jacobi smoothing."""

from __future__ import annotations

from typing import Any

import warp as wp

from ...core.types import mat36f, mat66f, vec6f
from .deformable_contact import (
    _COEFFICIENT_TOLERANCE,
    DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS,
    DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE,
    DEFORMABLE_CONTACT_STATUS_VALID,
    _make_project_particle_contacts_kernel,
    _merge_rigid_prepared_status,
    _ParticleContactColoredState,
    _ParticleContactProjectionData,
)
from .projection import (
    PROJECTION_STATUS_INVALID,
    PROJECTION_STATUS_VALID,
    _can_fuse_rigid_projection_by_world,
    compute_limit_delassus,
    prepare_contact_coulomb,
    prepare_contact_coulomb_delassus,
    project_contact_coulomb_local,
    project_friction_local,
    project_limit_local,
)
from .sweep import (
    _atomic_add_twist,
    _DeformableRigidContactProjectionData,
    _DeformableRigidProjectionState,
    _initialize_jacobi_projection_status,
    _is_finite_twist,
    _make_colored_projection_index,
    _make_project_contacts_kernel,
    _make_project_scalar_kernel,
    _make_projection_struct,
    _make_rigid_projection_state,
    _project_rigid_contact_colored,
    _project_rigid_scalar_colored,
    _RigidContactProjectionData,
    _sync_threads,
    _warmstart_contacts_jacobi,
    _warmstart_frictions_jacobi,
    _warmstart_limits_jacobi,
    prepare_jacobi_projection_data,
    project_constraints_jacobi,
)

wp.set_module_options({"enable_backward": False})

_COLOR_REPAIR_PASSES = 3
_COLOR_BLOCK_DIM = 128
_WORLD_COLOR_BLOCK_DIM = 64
_COLOR_BLOCKS_PER_SM = 2
# Avoid parallel-scan setup for the small color counts used by normal workloads.
_SERIAL_COLOR_PREFIX_LIMIT = 64
_NO_PROPOSAL = -1
_LOCK_FREE = 0x7FFFFFFF


@wp.struct
class _RigidColoredWorldData:
    world_color_count: wp.array2d[wp.int32]
    world_friction_offset: wp.array[wp.int32]
    world_friction_count: wp.array[wp.int32]
    friction_colors: wp.array[wp.int32]
    friction_body_first: wp.array[wp.int32]
    friction_body_second: wp.array[wp.int32]
    friction_jacobian_first: wp.array[vec6f]
    friction_jacobian_second: wp.array[vec6f]
    friction_bound: wp.array[wp.float32]
    friction_colored_delassus: wp.array[wp.float32]
    friction_jacobi_delassus: wp.array[wp.float32]
    friction_reaction: wp.array[wp.float32]
    world_contact_offset: wp.array[wp.int32]
    world_contact_count: wp.array[wp.int32]
    contact_colors: wp.array[wp.int32]
    contact_jacobi_delassus: wp.array[wp.mat33f]
    world_limit_offset: wp.array[wp.int32]
    world_limit_count: wp.array[wp.int32]
    limit_colors: wp.array[wp.int32]
    limit_body_first: wp.array[wp.int32]
    limit_body_second: wp.array[wp.int32]
    limit_jacobian_first: wp.array[vec6f]
    limit_jacobian_second: wp.array[vec6f]
    limit_bias: wp.array[wp.float32]
    limit_colored_delassus: wp.array[wp.float32]
    limit_jacobi_delassus: wp.array[wp.float32]
    limit_reaction: wp.array[wp.float32]


def _bounded_worker_count(capacity: int, device) -> int:
    if capacity <= 0:
        return 0
    if device.is_cuda:
        return min(capacity, max(_COLOR_BLOCK_DIM, device.sm_count * _COLOR_BLOCKS_PER_SM * _COLOR_BLOCK_DIM))
    return capacity


@wp.func
def _mix_color_key(value: wp.uint32) -> wp.uint32:
    value = (value ^ (value >> wp.uint32(16))) * wp.uint32(0x7FEB352D)
    value = (value ^ (value >> wp.uint32(15))) * wp.uint32(0x846CA68B)
    return value ^ (value >> wp.uint32(16))


@wp.func
def _initial_color(world: int, local: int, first: int, second: int, family: int, color_count: int) -> int:
    key = wp.uint32(world + 1) * wp.uint32(0x9E3779B9)
    key = key ^ (wp.uint32(local + 1) * wp.uint32(0x85EBCA6B))
    key = key ^ (wp.uint32(first + 2) * wp.uint32(0xC2B2AE35))
    key = key ^ (wp.uint32(second + 2) * wp.uint32(0x27D4EB2F))
    key = key ^ wp.uint32(family * 0x165667B1)
    return int(_mix_color_key(key) % wp.uint32(color_count))


@wp.func
def _occupancy(occupancy: wp.array2d[wp.int32], endpoint: int, color: int) -> int:
    value = int(0)
    if endpoint >= 0:
        value = occupancy[endpoint, color]
    return value


@wp.func
def _choose_two_endpoint_color(
    first: int,
    second: int,
    current: int,
    color_count: int,
    occupancy: wp.array2d[wp.int32],
) -> int:
    current_sum = _occupancy(occupancy, first, current)
    endpoint_count = int(0)
    if first >= 0:
        endpoint_count += 1
    if second >= 0 and second != first:
        current_sum += _occupancy(occupancy, second, current)
        endpoint_count += 1
    best = current
    best_sum = current_sum
    best_max = wp.max(_occupancy(occupancy, first, current), _occupancy(occupancy, second, current))
    for color in range(color_count):
        candidate_sum = _occupancy(occupancy, first, color)
        if second >= 0 and second != first:
            candidate_sum += _occupancy(occupancy, second, color)
        candidate_max = wp.max(_occupancy(occupancy, first, color), _occupancy(occupancy, second, color))
        if candidate_sum < best_sum or (candidate_sum == best_sum and candidate_max < best_max):
            best = color
            best_sum = candidate_sum
            best_max = candidate_max
    # Moving one incidence adds one to the destination and removes one from
    # the source. This is exactly the strict-improvement condition for sum(m^2).
    if best != current and best_sum + endpoint_count < current_sum:
        return best
    return _NO_PROPOSAL


@wp.kernel
def _assign_two_endpoint_colors(
    constraint_world: wp.array[wp.int32],
    constraint_local: wp.array[wp.int32],
    world_constraint_count: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    family: int,
    color_count: int,
    colors: wp.array[wp.int32],
):
    constraint = wp.tid()
    world = constraint_world[constraint]
    if constraint_local[constraint] >= world_constraint_count[world]:
        colors[constraint] = _NO_PROPOSAL
        return
    colors[constraint] = _initial_color(
        world,
        constraint_local[constraint],
        endpoint_first[constraint],
        endpoint_second[constraint],
        family,
        color_count,
    )


@wp.kernel
def _count_two_endpoint_occupancy(
    constraint_world: wp.array[wp.int32],
    constraint_local: wp.array[wp.int32],
    world_constraint_count: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    colors: wp.array[wp.int32],
    occupancy: wp.array2d[wp.int32],
    world_color_count: wp.array2d[wp.int32],
):
    constraint = wp.tid()
    world = constraint_world[constraint]
    if constraint_local[constraint] >= world_constraint_count[world]:
        return
    color = colors[constraint]
    wp.atomic_add(world_color_count, world, color, 1)
    first = endpoint_first[constraint]
    second = endpoint_second[constraint]
    if first >= 0:
        wp.atomic_add(occupancy, first, color, 1)
    if second >= 0 and second != first:
        wp.atomic_add(occupancy, second, color, 1)


@wp.kernel
def _propose_and_claim_two_endpoint_repairs(
    constraint_world: wp.array[wp.int32],
    constraint_local: wp.array[wp.int32],
    world_constraint_count: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    color_count: int,
    key_offset: int,
    colors: wp.array[wp.int32],
    occupancy: wp.array2d[wp.int32],
    proposals: wp.array[wp.int32],
    locks: wp.array[wp.int32],
):
    constraint = wp.tid()
    world = constraint_world[constraint]
    if constraint_local[constraint] >= world_constraint_count[world]:
        proposals[constraint] = _NO_PROPOSAL
        return
    proposal = _choose_two_endpoint_color(
        endpoint_first[constraint],
        endpoint_second[constraint],
        colors[constraint],
        color_count,
        occupancy,
    )
    proposals[constraint] = proposal
    if proposal < 0:
        return
    key = key_offset + constraint
    first = endpoint_first[constraint]
    second = endpoint_second[constraint]
    if first >= 0:
        wp.atomic_min(locks, first, key)
    if second >= 0 and second != first:
        wp.atomic_min(locks, second, key)


@wp.kernel
def _commit_two_endpoint_repairs(
    constraint_world: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    key_offset: int,
    colors: wp.array[wp.int32],
    proposals: wp.array[wp.int32],
    locks: wp.array[wp.int32],
    occupancy: wp.array2d[wp.int32],
    world_color_count: wp.array2d[wp.int32],
):
    constraint = wp.tid()
    candidate = proposals[constraint]
    if candidate < 0:
        return
    key = key_offset + constraint
    first = endpoint_first[constraint]
    second = endpoint_second[constraint]
    if (first >= 0 and locks[first] != key) or (second >= 0 and second != first and locks[second] != key):
        return
    current = colors[constraint]
    world = constraint_world[constraint]
    wp.atomic_add(world_color_count, world, current, -1)
    wp.atomic_add(world_color_count, world, candidate, 1)
    if first >= 0:
        wp.atomic_add(occupancy, first, current, -1)
        wp.atomic_add(occupancy, first, candidate, 1)
    if second >= 0 and second != first:
        wp.atomic_add(occupancy, second, current, -1)
        wp.atomic_add(occupancy, second, candidate, 1)
    colors[constraint] = candidate


@wp.func
def _deformable_endpoint_occupancy(
    particle_indices: wp.array2d[wp.int32],
    contact: int,
    body: int,
    color: int,
    particle_occupancy: wp.array2d[wp.int32],
    body_occupancy: wp.array2d[wp.int32],
) -> int:
    score = _occupancy(body_occupancy, body, color)
    for slot in range(4):
        particle = particle_indices[contact, slot]
        unique = particle >= 0
        for previous in range(slot):
            unique = unique and particle_indices[contact, previous] != particle
        if unique:
            score += particle_occupancy[particle, color]
    return score


@wp.func
def _deformable_endpoint_maximum(
    particle_indices: wp.array2d[wp.int32],
    contact: int,
    body: int,
    color: int,
    particle_occupancy: wp.array2d[wp.int32],
    body_occupancy: wp.array2d[wp.int32],
) -> int:
    maximum = _occupancy(body_occupancy, body, color)
    for slot in range(4):
        particle = particle_indices[contact, slot]
        unique = particle >= 0
        for previous in range(slot):
            unique = unique and particle_indices[contact, previous] != particle
        if unique:
            maximum = wp.max(maximum, particle_occupancy[particle, color])
    return maximum


@wp.kernel
def _assign_deformable_colors(
    particle_indices: wp.array2d[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    contact_status: wp.array[wp.int32],
    color_count: int,
    colors: wp.array[wp.int32],
):
    contact = wp.tid()
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
        colors[contact] = _NO_PROPOSAL
        return
    first = particle_indices[contact, 0]
    second = particle_indices[contact, 1]
    colors[contact] = _initial_color(contact_world[contact], contact, first, second, 3, color_count)


@wp.kernel
def _count_deformable_occupancy(
    particle_indices: wp.array2d[wp.int32],
    contact_body: wp.array[wp.int32],
    contact_status: wp.array[wp.int32],
    colors: wp.array[wp.int32],
    particle_occupancy: wp.array2d[wp.int32],
    body_occupancy: wp.array2d[wp.int32],
):
    contact = wp.tid()
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
        return
    color = colors[contact]
    for slot in range(4):
        particle = particle_indices[contact, slot]
        unique = particle >= 0
        for previous in range(slot):
            unique = unique and particle_indices[contact, previous] != particle
        if unique:
            wp.atomic_add(particle_occupancy, particle, color, 1)
    body = contact_body[contact]
    if body >= 0:
        wp.atomic_add(body_occupancy, body, color, 1)


@wp.kernel
def _accumulate_deformable_majorizer_weights(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[wp.float32],
    contact_status: wp.array[wp.int32],
    colors: wp.array[wp.int32],
    particle_weight_sum: wp.array2d[wp.float32],
):
    contact = wp.tid()
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
        return
    color = colors[contact]
    if color < 0:
        return
    for slot in range(4):
        particle = particle_indices[contact, slot]
        coefficient = wp.abs(coefficients[contact, slot])
        if particle >= 0 and coefficient > _COEFFICIENT_TOLERANCE:
            wp.atomic_add(particle_weight_sum, particle, color, coefficient)


@wp.kernel
def _propose_and_claim_deformable_repairs(
    particle_indices: wp.array2d[wp.int32],
    contact_body: wp.array[wp.int32],
    contact_status: wp.array[wp.int32],
    color_count: int,
    key_offset: int,
    colors: wp.array[wp.int32],
    particle_occupancy: wp.array2d[wp.int32],
    body_occupancy: wp.array2d[wp.int32],
    proposals: wp.array[wp.int32],
    particle_locks: wp.array[wp.int32],
    body_locks: wp.array[wp.int32],
):
    contact = wp.tid()
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
        proposals[contact] = _NO_PROPOSAL
        return
    current = colors[contact]
    current_sum = _deformable_endpoint_occupancy(
        particle_indices, contact, contact_body[contact], current, particle_occupancy, body_occupancy
    )
    endpoint_count = int(contact_body[contact] >= 0)
    for slot in range(4):
        particle = particle_indices[contact, slot]
        unique = particle >= 0
        for previous in range(slot):
            unique = unique and particle_indices[contact, previous] != particle
        if unique:
            endpoint_count += 1
    best = current
    best_sum = current_sum
    best_max = _deformable_endpoint_maximum(
        particle_indices, contact, contact_body[contact], current, particle_occupancy, body_occupancy
    )
    for color in range(color_count):
        score = _deformable_endpoint_occupancy(
            particle_indices, contact, contact_body[contact], color, particle_occupancy, body_occupancy
        )
        maximum = _deformable_endpoint_maximum(
            particle_indices, contact, contact_body[contact], color, particle_occupancy, body_occupancy
        )
        if score < best_sum or (score == best_sum and maximum < best_max):
            best = color
            best_sum = score
            best_max = maximum
    proposal = _NO_PROPOSAL
    if best != current and best_sum + endpoint_count < current_sum:
        proposal = best
    proposals[contact] = proposal
    if proposal < 0:
        return
    key = key_offset + contact
    for slot in range(4):
        particle = particle_indices[contact, slot]
        unique = particle >= 0
        for previous in range(slot):
            unique = unique and particle_indices[contact, previous] != particle
        if unique:
            wp.atomic_min(particle_locks, particle, key)
    body = contact_body[contact]
    if body >= 0:
        wp.atomic_min(body_locks, body, key)


@wp.kernel
def _commit_deformable_repairs(
    particle_indices: wp.array2d[wp.int32],
    contact_body: wp.array[wp.int32],
    key_offset: int,
    colors: wp.array[wp.int32],
    proposals: wp.array[wp.int32],
    particle_locks: wp.array[wp.int32],
    body_locks: wp.array[wp.int32],
    particle_occupancy: wp.array2d[wp.int32],
    body_occupancy: wp.array2d[wp.int32],
):
    contact = wp.tid()
    candidate = proposals[contact]
    if candidate < 0:
        return
    key = key_offset + contact
    accepted = True
    for slot in range(4):
        particle = particle_indices[contact, slot]
        unique = particle >= 0
        for previous in range(slot):
            unique = unique and particle_indices[contact, previous] != particle
        if unique:
            accepted = accepted and particle_locks[particle] == key
    body = contact_body[contact]
    if body >= 0:
        accepted = accepted and body_locks[body] == key
    if not accepted:
        return
    current = colors[contact]
    for slot in range(4):
        particle = particle_indices[contact, slot]
        unique = particle >= 0
        for previous in range(slot):
            unique = unique and particle_indices[contact, previous] != particle
        if unique:
            wp.atomic_add(particle_occupancy, particle, current, -1)
            wp.atomic_add(particle_occupancy, particle, candidate, 1)
    if body >= 0:
        wp.atomic_add(body_occupancy, body, current, -1)
        wp.atomic_add(body_occupancy, body, candidate, 1)
    colors[contact] = candidate


@wp.kernel
def _count_colors(colors: wp.array[wp.int32], color_count: int, counts: wp.array[wp.int32]):
    item = wp.tid()
    color = colors[item]
    if color >= 0 and color < color_count:
        wp.atomic_add(counts, color, 1)


@wp.kernel
def _prefix_color_counts(
    color_count: int, counts: wp.array[wp.int32], offsets: wp.array[wp.int32], cursors: wp.array[wp.int32]
):
    offset = int(0)
    for color in range(color_count):
        offsets[color] = offset
        cursors[color] = offset
        offset += counts[color]


@wp.kernel
def _scatter_color_order(
    colors: wp.array[wp.int32],
    color_count: int,
    cursors: wp.array[wp.int32],
    order: wp.array[wp.int32],
):
    item = wp.tid()
    color = colors[item]
    if color >= 0 and color < color_count:
        ordered = wp.atomic_add(cursors, color, 1)
        order[ordered] = item


@wp.kernel
def _prepare_frictions_colored(
    launch_dim: int,
    target_color: int,
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    order: wp.array[wp.int32],
    constraint_world: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    occupancy: wp.array2d[wp.int32],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    lane = wp.tid()
    begin = color_offsets[target_color] + lane
    end = color_offsets[target_color] + color_counts[target_color]
    for ordered in range(begin, end, launch_dim):
        constraint = order[ordered]
        first = endpoint_first[constraint]
        second = endpoint_second[constraint]
        if first < 0 and second < 0:
            delassus[constraint] = 0.0
            continue
        inverse_first = mat66f(0.0)
        inverse_second = mat66f(0.0)
        if first >= 0:
            inverse_first = wp.float32(wp.max(1, occupancy[first, target_color])) * inverse_weight[first]
        if second >= 0:
            inverse_second = wp.float32(wp.max(1, occupancy[second, target_color])) * inverse_weight[second]
        value = compute_limit_delassus(
            jacobian_first[constraint], inverse_first, jacobian_second[constraint], inverse_second
        )
        delassus[constraint] = value
        if not wp.isfinite(value) or value <= 0.0:
            world_status[constraint_world[constraint]] = PROJECTION_STATUS_INVALID


@wp.kernel
def _prepare_contacts_colored(
    launch_dim: int,
    target_color: int,
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    order: wp.array[wp.int32],
    constraint_world: wp.array[wp.int32],
    endpoint_first: wp.array[wp.int32],
    endpoint_second: wp.array[wp.int32],
    jacobian_first: wp.array[mat36f],
    jacobian_second: wp.array[mat36f],
    bias: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    occupancy: wp.array2d[wp.int32],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.mat33f],
    world_status: wp.array[wp.int32],
):
    lane = wp.tid()
    begin = color_offsets[target_color] + lane
    end = color_offsets[target_color] + color_counts[target_color]
    for ordered in range(begin, end, launch_dim):
        constraint = order[ordered]
        first = endpoint_first[constraint]
        second = endpoint_second[constraint]
        if first < 0 and second < 0:
            delassus[constraint] = wp.mat33f(0.0)
            continue
        inverse_first = mat66f(0.0)
        inverse_second = mat66f(0.0)
        if first >= 0:
            inverse_first = wp.float32(wp.max(1, occupancy[first, target_color])) * inverse_weight[first]
        if second >= 0:
            inverse_second = wp.float32(wp.max(1, occupancy[second, target_color])) * inverse_weight[second]
        data = prepare_contact_coulomb(
            jacobian_first[constraint],
            inverse_first,
            jacobian_second[constraint],
            inverse_second,
            bias[constraint],
            friction[constraint],
        )
        delassus[constraint] = data.delassus
        if data.status == PROJECTION_STATUS_INVALID:
            world_status[constraint_world[constraint]] = data.status


@wp.kernel
def _prepare_rigid_colored(
    launch_dim: int,
    target_color: int,
    friction_counts: wp.array[wp.int32],
    friction_offsets: wp.array[wp.int32],
    friction_order: wp.array[wp.int32],
    friction_world: wp.array[wp.int32],
    friction_first: wp.array[wp.int32],
    friction_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    contact_counts: wp.array[wp.int32],
    contact_offsets: wp.array[wp.int32],
    contact_order: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_first: wp.array[wp.int32],
    contact_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    limit_counts: wp.array[wp.int32],
    limit_offsets: wp.array[wp.int32],
    limit_order: wp.array[wp.int32],
    limit_world: wp.array[wp.int32],
    limit_first: wp.array[wp.int32],
    limit_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    occupancy: wp.array2d[wp.int32],
    inverse_weight: wp.array[mat66f],
    friction_delassus: wp.array[wp.float32],
    contact_delassus: wp.array[wp.mat33f],
    limit_delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    lane = wp.tid()
    friction_begin = friction_offsets[target_color] + lane
    friction_end = friction_offsets[target_color] + friction_counts[target_color]
    for ordered in range(friction_begin, friction_end, launch_dim):
        constraint = friction_order[ordered]
        first = friction_first[constraint]
        second = friction_second[constraint]
        if first < 0 and second < 0:
            friction_delassus[constraint] = 0.0
            continue
        inverse_first = mat66f(0.0)
        inverse_second = mat66f(0.0)
        if first >= 0:
            inverse_first = wp.float32(wp.max(1, occupancy[first, target_color])) * inverse_weight[first]
        if second >= 0:
            inverse_second = wp.float32(wp.max(1, occupancy[second, target_color])) * inverse_weight[second]
        value = compute_limit_delassus(
            friction_jacobian_first[constraint],
            inverse_first,
            friction_jacobian_second[constraint],
            inverse_second,
        )
        friction_delassus[constraint] = value
        if not wp.isfinite(value) or value <= 0.0:
            world_status[friction_world[constraint]] = PROJECTION_STATUS_INVALID

    contact_begin = contact_offsets[target_color] + lane
    contact_end = contact_offsets[target_color] + contact_counts[target_color]
    for ordered in range(contact_begin, contact_end, launch_dim):
        constraint = contact_order[ordered]
        first = contact_first[constraint]
        second = contact_second[constraint]
        if first < 0 and second < 0:
            contact_delassus[constraint] = wp.mat33f(0.0)
            continue
        inverse_first = mat66f(0.0)
        inverse_second = mat66f(0.0)
        if first >= 0:
            inverse_first = wp.float32(wp.max(1, occupancy[first, target_color])) * inverse_weight[first]
        if second >= 0:
            inverse_second = wp.float32(wp.max(1, occupancy[second, target_color])) * inverse_weight[second]
        data = prepare_contact_coulomb(
            contact_jacobian_first[constraint],
            inverse_first,
            contact_jacobian_second[constraint],
            inverse_second,
            contact_bias[constraint],
            contact_friction[constraint],
        )
        contact_delassus[constraint] = data.delassus
        if data.status == PROJECTION_STATUS_INVALID:
            world_status[contact_world[constraint]] = data.status

    limit_begin = limit_offsets[target_color] + lane
    limit_end = limit_offsets[target_color] + limit_counts[target_color]
    for ordered in range(limit_begin, limit_end, launch_dim):
        constraint = limit_order[ordered]
        first = limit_first[constraint]
        second = limit_second[constraint]
        if first < 0 and second < 0:
            limit_delassus[constraint] = 0.0
            continue
        inverse_first = mat66f(0.0)
        inverse_second = mat66f(0.0)
        if first >= 0:
            inverse_first = wp.float32(wp.max(1, occupancy[first, target_color])) * inverse_weight[first]
        if second >= 0:
            inverse_second = wp.float32(wp.max(1, occupancy[second, target_color])) * inverse_weight[second]
        value = compute_limit_delassus(
            limit_jacobian_first[constraint], inverse_first, limit_jacobian_second[constraint], inverse_second
        )
        limit_delassus[constraint] = value
        if not wp.isfinite(value) or value <= 0.0:
            world_status[limit_world[constraint]] = PROJECTION_STATUS_INVALID


@wp.kernel
def _prepare_deformable_colored(
    launch_dim: int,
    target_color: int,
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    order: wp.array[wp.int32],
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    body_jacobian: wp.array[mat36f],
    rigid_bias: wp.array[wp.vec3f],
    friction: wp.array[wp.float32],
    particle_weight_sum: wp.array2d[wp.float32],
    body_occupancy: wp.array2d[wp.int32],
    particle_inverse_weight: wp.array[wp.float32],
    body_inverse_weight: wp.array[mat66f],
    include_rigid: bool,
    contact_status: wp.array[wp.int32],
    scalar_delassus: wp.array[wp.float32],
    delassus: wp.array[wp.mat33f],
    prepared_status: wp.array[wp.int32],
    contact_world_status: wp.array[wp.int32],
):
    lane = wp.tid()
    begin = color_offsets[target_color] + lane
    end = color_offsets[target_color] + color_counts[target_color]
    for ordered in range(begin, end, launch_dim):
        contact = order[ordered]
        particle_value = wp.float32(0.0)
        for particle_slot in range(4):
            particle = particle_indices[contact, particle_slot]
            if particle >= 0:
                coefficient = wp.abs(coefficients[contact, particle_slot])
                if coefficient > _COEFFICIENT_TOLERANCE:
                    particle_value += (
                        coefficient * particle_weight_sum[particle, target_color] * particle_inverse_weight[particle]
                    )
        value = particle_value * wp.identity(3, dtype=wp.float32)
        body = contact_body[contact]
        if include_rigid and body >= 0:
            multiplicity = wp.max(1, body_occupancy[body, target_color])
            jacobian = body_jacobian[contact]
            value += jacobian @ (wp.float32(multiplicity) * body_inverse_weight[body]) @ wp.transpose(jacobian)
        data = prepare_contact_coulomb_delassus(value, rigid_bias[contact], friction[contact])
        scalar_delassus[contact] = particle_value
        delassus[contact] = data.delassus
        if data.status == PROJECTION_STATUS_INVALID or (not include_rigid and particle_value <= 0.0):
            contact_status[contact] = DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS
            world = contact_world[contact]
            prepared_status[world] = PROJECTION_STATUS_INVALID
            wp.atomic_max(contact_world_status, world, DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS)


@wp.kernel
def _project_rigid_colored(
    launch_dim: int,
    target_color: int,
    friction_counts: wp.array[wp.int32],
    friction_offsets: wp.array[wp.int32],
    friction_order: wp.array[wp.int32],
    friction_world: wp.array[wp.int32],
    friction_first: wp.array[wp.int32],
    friction_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_bound: wp.array[wp.float32],
    friction_delassus: wp.array[wp.float32],
    contact_counts: wp.array[wp.int32],
    contact_offsets: wp.array[wp.int32],
    contact_order: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_data: Any,
    limit_counts: wp.array[wp.int32],
    limit_offsets: wp.array[wp.int32],
    limit_order: wp.array[wp.int32],
    limit_world: wp.array[wp.int32],
    limit_first: wp.array[wp.int32],
    limit_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    limit_delassus: wp.array[wp.float32],
    state: Any,
    friction_reaction: wp.array[wp.float32],
    limit_reaction: wp.array[wp.float32],
):
    lane = wp.tid()
    friction_begin = friction_offsets[target_color] + lane
    friction_end = friction_offsets[target_color] + friction_counts[target_color]
    for ordered in range(friction_begin, friction_end, launch_dim):
        constraint = friction_order[ordered]
        world = friction_world[constraint]
        if not state.world_active[world] or state.world_status[world] != PROJECTION_STATUS_VALID:
            continue
        _project_rigid_scalar_colored(
            constraint,
            world,
            target_color,
            friction_first,
            friction_second,
            friction_jacobian_first,
            friction_jacobian_second,
            friction_bound,
            friction_bound,
            friction_delassus,
            friction_reaction,
            state,
            False,
        )

    contact_begin = contact_offsets[target_color] + lane
    contact_end = contact_offsets[target_color] + contact_counts[target_color]
    for ordered in range(contact_begin, contact_end, launch_dim):
        constraint = contact_order[ordered]
        world = contact_world[constraint]
        if state.world_active[world] and state.world_status[world] == PROJECTION_STATUS_VALID:
            _project_rigid_contact_colored(constraint, world, target_color, contact_data, state)

    limit_begin = limit_offsets[target_color] + lane
    limit_end = limit_offsets[target_color] + limit_counts[target_color]
    for ordered in range(limit_begin, limit_end, launch_dim):
        constraint = limit_order[ordered]
        world = limit_world[constraint]
        if not state.world_active[world] or state.world_status[world] != PROJECTION_STATUS_VALID:
            continue
        _project_rigid_scalar_colored(
            constraint,
            world,
            target_color,
            limit_first,
            limit_second,
            limit_jacobian_first,
            limit_jacobian_second,
            limit_bias,
            limit_bias,
            limit_delassus,
            limit_reaction,
            state,
            True,
        )


@wp.kernel
def _project_rigid_colored_by_world(
    iterations: wp.int32,
    color_count: wp.int32,
    world_body_offset: wp.array[wp.int32],
    world_body_count: wp.array[wp.int32],
    data: Any,
    contact_data: Any,
    state: Any,
    prepared_status: wp.array[wp.int32],
):
    """Project one rigid world per block, including warm start and smoothing."""
    world, lane = wp.tid()
    thread_count = wp.block_dim()

    local = lane
    while local < world_body_count[world]:
        state.twist_delta[world_body_offset[world] + local] = vec6f(0.0)
        local += thread_count
    if state.world_active[world]:
        state.world_status[world] = prepared_status[world]
    _sync_threads()
    if not state.world_active[world] or state.world_status[world] != PROJECTION_STATUS_VALID:
        return

    # Apply the existing reactions before the first colored sweep.
    local = lane
    while local < data.world_friction_count[world]:
        constraint = data.world_friction_offset[world] + local
        first = data.friction_body_first[constraint]
        second = data.friction_body_second[constraint]
        friction_reaction = data.friction_reaction[constraint]
        correction_first = vec6f(0.0)
        correction_second = vec6f(0.0)
        if first >= 0:
            correction_first = data.friction_jacobian_first[constraint] * friction_reaction
        if second >= 0:
            correction_second = data.friction_jacobian_second[constraint] * friction_reaction
        if _is_finite_twist(correction_first) and _is_finite_twist(correction_second):
            _atomic_add_twist(state.twist_delta, first, correction_first)
            _atomic_add_twist(state.twist_delta, second, correction_second)
        else:
            state.world_status[world] = PROJECTION_STATUS_INVALID
        local += thread_count

    local = lane
    while local < data.world_contact_count[world]:
        constraint = data.world_contact_offset[world] + local
        first = contact_data.body_first[constraint]
        second = contact_data.body_second[constraint]
        contact_reaction = contact_data.reaction[constraint]
        correction_first = vec6f(0.0)
        correction_second = vec6f(0.0)
        if first >= 0:
            correction_first = wp.transpose(contact_data.jacobian_first[constraint]) @ contact_reaction
        if second >= 0:
            correction_second = wp.transpose(contact_data.jacobian_second[constraint]) @ contact_reaction
        if _is_finite_twist(correction_first) and _is_finite_twist(correction_second):
            _atomic_add_twist(state.twist_delta, first, correction_first)
            _atomic_add_twist(state.twist_delta, second, correction_second)
        else:
            state.world_status[world] = PROJECTION_STATUS_INVALID
        local += thread_count

    local = lane
    while local < data.world_limit_count[world]:
        constraint = data.world_limit_offset[world] + local
        first = data.limit_body_first[constraint]
        second = data.limit_body_second[constraint]
        limit_reaction = data.limit_reaction[constraint]
        correction_first = vec6f(0.0)
        correction_second = vec6f(0.0)
        if first >= 0:
            correction_first = data.limit_jacobian_first[constraint] * limit_reaction
        if second >= 0:
            correction_second = data.limit_jacobian_second[constraint] * limit_reaction
        if _is_finite_twist(correction_first) and _is_finite_twist(correction_second):
            _atomic_add_twist(state.twist_delta, first, correction_first)
            _atomic_add_twist(state.twist_delta, second, correction_second)
        else:
            state.world_status[world] = PROJECTION_STATUS_INVALID
        local += thread_count

    _sync_threads()
    local = lane
    while local < world_body_count[world]:
        body = world_body_offset[world] + local
        if state.world_status[world] == PROJECTION_STATUS_VALID:
            correction = state.inverse_weight[body] @ state.twist_delta[body]
            if _is_finite_twist(correction):
                state.projected_twist[body] += correction
            else:
                state.world_status[world] = PROJECTION_STATUS_INVALID
        state.twist_delta[body] = vec6f(0.0)
        local += thread_count
    _sync_threads()

    # Colors and iterations are block-local for independent rigid worlds.
    for _iteration in range(iterations):
        for color in range(color_count):
            if data.world_color_count[world, color] != 0:
                if state.world_status[world] == PROJECTION_STATUS_VALID:
                    local = lane
                    while local < data.world_friction_count[world]:
                        constraint = data.world_friction_offset[world] + local
                        if data.friction_colors[constraint] == color:
                            _project_rigid_scalar_colored(
                                constraint,
                                world,
                                color,
                                data.friction_body_first,
                                data.friction_body_second,
                                data.friction_jacobian_first,
                                data.friction_jacobian_second,
                                data.friction_bound,
                                data.friction_bound,
                                data.friction_colored_delassus,
                                data.friction_reaction,
                                state,
                                False,
                            )
                        local += thread_count

                    local = lane
                    while local < data.world_contact_count[world]:
                        constraint = data.world_contact_offset[world] + local
                        if data.contact_colors[constraint] == color:
                            _project_rigid_contact_colored(constraint, world, color, contact_data, state)
                        local += thread_count

                    local = lane
                    while local < data.world_limit_count[world]:
                        constraint = data.world_limit_offset[world] + local
                        if data.limit_colors[constraint] == color:
                            _project_rigid_scalar_colored(
                                constraint,
                                world,
                                color,
                                data.limit_body_first,
                                data.limit_body_second,
                                data.limit_jacobian_first,
                                data.limit_jacobian_second,
                                data.limit_bias,
                                data.limit_bias,
                                data.limit_colored_delassus,
                                data.limit_reaction,
                                state,
                                True,
                            )
                        local += thread_count

                _sync_threads()
                local = lane
                while local < world_body_count[world]:
                    body = world_body_offset[world] + local
                    if state.world_status[world] == PROJECTION_STATUS_VALID:
                        state.projected_twist[body] += state.twist_delta[body]
                    state.twist_delta[body] = vec6f(0.0)
                    local += thread_count
                _sync_threads()

    # Preserve the existing final mass-split Jacobi smoothing sweep. Accumulate
    # raw wrenches and defer W^-1 until each body is visited once.
    if state.world_status[world] == PROJECTION_STATUS_VALID:
        local = lane
        while local < data.world_friction_count[world]:
            constraint = data.world_friction_offset[world] + local
            first = data.friction_body_first[constraint]
            second = data.friction_body_second[constraint]
            friction_velocity = wp.float32(0.0)
            if first >= 0:
                friction_velocity += wp.dot(data.friction_jacobian_first[constraint], state.projected_twist[first])
            if second >= 0:
                friction_velocity += wp.dot(data.friction_jacobian_second[constraint], state.projected_twist[second])
            friction_projection = project_friction_local(
                friction_velocity,
                data.friction_reaction[constraint],
                data.friction_jacobi_delassus[constraint],
                data.friction_bound[constraint],
            )
            correction_first = vec6f(0.0)
            correction_second = vec6f(0.0)
            if first >= 0:
                correction_first = data.friction_jacobian_first[constraint] * friction_projection.reaction_delta
            if second >= 0:
                correction_second = data.friction_jacobian_second[constraint] * friction_projection.reaction_delta
            if (
                friction_projection.status == PROJECTION_STATUS_VALID
                and _is_finite_twist(correction_first)
                and _is_finite_twist(correction_second)
            ):
                data.friction_reaction[constraint] = friction_projection.reaction
                _atomic_add_twist(state.twist_delta, first, correction_first)
                _atomic_add_twist(state.twist_delta, second, correction_second)
            else:
                state.world_status[world] = PROJECTION_STATUS_INVALID
            local += thread_count

        local = lane
        while local < data.world_contact_count[world]:
            constraint = data.world_contact_offset[world] + local
            first = contact_data.body_first[constraint]
            second = contact_data.body_second[constraint]
            if first < 0 and second < 0:
                contact_data.reaction[constraint] = wp.vec3f(0.0)
            else:
                contact_velocity = contact_data.bias[constraint]
                if first >= 0:
                    contact_velocity += contact_data.jacobian_first[constraint] @ state.projected_twist[first]
                if second >= 0:
                    contact_velocity += contact_data.jacobian_second[constraint] @ state.projected_twist[second]
                contact_projection = project_contact_coulomb_local(
                    contact_velocity,
                    contact_data.reaction[constraint],
                    data.contact_jacobi_delassus[constraint],
                    contact_data.friction[constraint],
                )
                correction_first = vec6f(0.0)
                correction_second = vec6f(0.0)
                if first >= 0:
                    correction_first = (
                        wp.transpose(contact_data.jacobian_first[constraint]) @ contact_projection.reaction_delta
                    )
                if second >= 0:
                    correction_second = (
                        wp.transpose(contact_data.jacobian_second[constraint]) @ contact_projection.reaction_delta
                    )
                if (
                    contact_projection.status == PROJECTION_STATUS_VALID
                    and _is_finite_twist(correction_first)
                    and _is_finite_twist(correction_second)
                ):
                    contact_data.reaction[constraint] = contact_projection.reaction
                    _atomic_add_twist(state.twist_delta, first, correction_first)
                    _atomic_add_twist(state.twist_delta, second, correction_second)
                else:
                    state.world_status[world] = PROJECTION_STATUS_INVALID
            local += thread_count

        local = lane
        while local < data.world_limit_count[world]:
            constraint = data.world_limit_offset[world] + local
            first = data.limit_body_first[constraint]
            second = data.limit_body_second[constraint]
            if first < 0 and second < 0:
                data.limit_reaction[constraint] = 0.0
            else:
                limit_velocity = data.limit_bias[constraint]
                if first >= 0:
                    limit_velocity += wp.dot(data.limit_jacobian_first[constraint], state.projected_twist[first])
                if second >= 0:
                    limit_velocity += wp.dot(data.limit_jacobian_second[constraint], state.projected_twist[second])
                limit_projection = project_limit_local(
                    limit_velocity,
                    data.limit_reaction[constraint],
                    data.limit_jacobi_delassus[constraint],
                )
                correction_first = vec6f(0.0)
                correction_second = vec6f(0.0)
                if first >= 0:
                    correction_first = data.limit_jacobian_first[constraint] * limit_projection.reaction_delta
                if second >= 0:
                    correction_second = data.limit_jacobian_second[constraint] * limit_projection.reaction_delta
                if (
                    limit_projection.status == PROJECTION_STATUS_VALID
                    and _is_finite_twist(correction_first)
                    and _is_finite_twist(correction_second)
                ):
                    data.limit_reaction[constraint] = limit_projection.reaction
                    _atomic_add_twist(state.twist_delta, first, correction_first)
                    _atomic_add_twist(state.twist_delta, second, correction_second)
                else:
                    state.world_status[world] = PROJECTION_STATUS_INVALID
            local += thread_count

    _sync_threads()
    local = lane
    while local < world_body_count[world]:
        body = world_body_offset[world] + local
        if state.world_status[world] == PROJECTION_STATUS_VALID:
            correction = state.inverse_weight[body] @ state.twist_delta[body]
            if _is_finite_twist(correction):
                state.projected_twist[body] += correction
            else:
                state.world_status[world] = PROJECTION_STATUS_INVALID
        state.twist_delta[body] = vec6f(0.0)
        local += thread_count


@wp.kernel
def _apply_body_delta(
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    delta: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
):
    body = wp.tid()
    world = body_world[body]
    if world_active[world] and world_status[world] == PROJECTION_STATUS_VALID:
        projected_twist[body] += delta[body]
    delta[body] = vec6f(0.0)


@wp.kernel
def _apply_particle_delta_colored(
    particle_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    delta: wp.array[wp.vec3f],
    projected_velocity: wp.array[wp.vec3f],
):
    particle = wp.tid()
    world = particle_world[particle]
    if world_active[world] and world_status[world] == PROJECTION_STATUS_VALID:
        projected_velocity[particle] += delta[particle]
    delta[particle] = wp.vec3f(0.0)


@wp.kernel
def _apply_body_particle_delta_colored(
    launch_dim: int,
    body_count: int,
    particle_count: int,
    body_world: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    body_delta: wp.array[vec6f],
    particle_delta: wp.array[wp.vec3f],
    projected_twist: wp.array[vec6f],
    projected_velocity: wp.array[wp.vec3f],
):
    lane = wp.tid()
    for body in range(lane, body_count, launch_dim):
        world = body_world[body]
        if world_active[world] and world_status[world] == PROJECTION_STATUS_VALID:
            projected_twist[body] += body_delta[body]
        body_delta[body] = vec6f(0.0)
    for particle in range(lane, particle_count, launch_dim):
        world = particle_world[particle]
        if world_active[world] and world_status[world] == PROJECTION_STATUS_VALID:
            projected_velocity[particle] += particle_delta[particle]
        particle_delta[particle] = wp.vec3f(0.0)


class _ColorFamily:
    def __init__(self, capacity: int, color_count: int, device):
        self.capacity = capacity
        self.worker_count = _bounded_worker_count(capacity, device)
        self.colors = wp.full(capacity, _NO_PROPOSAL, dtype=wp.int32, device=device)
        self.proposals = wp.full(capacity, _NO_PROPOSAL, dtype=wp.int32, device=device)
        self.counts = wp.zeros(color_count, dtype=wp.int32, device=device)
        self.offsets = wp.zeros(color_count, dtype=wp.int32, device=device)
        self.cursors = wp.zeros(color_count, dtype=wp.int32, device=device)
        self.order = wp.full(capacity, -1, dtype=wp.int32, device=device)

    def _prefix_counts(self, color_count: int, device) -> None:
        if color_count <= _SERIAL_COLOR_PREFIX_LIMIT:
            wp.launch(
                _prefix_color_counts,
                dim=1,
                inputs=[color_count, self.counts],
                outputs=[self.offsets, self.cursors],
                device=device,
            )
        else:
            wp.utils.array_scan(self.counts, self.offsets, inclusive=False)
            wp.copy(self.cursors, self.offsets)

    def compact(self, color_count: int, device) -> None:
        self.counts.zero_()
        if self.capacity == 0:
            return
        wp.launch(
            _count_colors, dim=self.capacity, inputs=[self.colors, color_count], outputs=[self.counts], device=device
        )
        self._prefix_counts(color_count, device)
        wp.launch(
            _scatter_color_order,
            dim=self.capacity,
            inputs=[self.colors, color_count],
            outputs=[self.cursors, self.order],
            device=device,
        )


class ColoredGaussSeidelProjection:
    """Own fixed-capacity coloring and projection scratch for one LOX solver.

    The effective color count is the requested maximum bounded by total
    allocated unilateral capacity, with one internal color retained for empty
    or single-constraint systems. The endpoint occupancy tables use int32
    atomics, and deformable coefficient weights use an additional float32
    particle-by-color table. They are only allocated for requested counts of
    at least two; the one-color endpoint uses the existing Jacobi
    implementation directly.
    """

    def __init__(self, rigid_adapter, deformable_contacts, color_count: int):
        if color_count < 2:
            raise ValueError("Colored Gauss-Seidel requires at least two colors.")
        self.rigid_adapter = rigid_adapter
        self.deformable_contacts = deformable_contacts
        rigid_capacity = 0
        if rigid_adapter is not None:
            rigid_capacity = (
                rigid_adapter.friction_capacity + rigid_adapter.contact_capacity + rigid_adapter.limit_capacity
            )
        deformable_capacity = deformable_contacts.contact_capacity if deformable_contacts is not None else 0
        self.color_count = max(1, min(color_count, rigid_capacity + deformable_capacity))
        self.device = rigid_adapter.device if rigid_adapter is not None else deformable_contacts.device
        body_count = rigid_adapter.body_constraint_count.shape[0] if rigid_adapter is not None else 0
        particle_count = deformable_contacts.cloth_system.particle_count if deformable_contacts is not None else 0
        self.body_occupancy = wp.zeros((body_count, self.color_count), dtype=wp.int32, device=self.device)
        self.particle_occupancy = wp.zeros((particle_count, self.color_count), dtype=wp.int32, device=self.device)
        self.particle_majorizer_weight_sum = wp.zeros(
            (particle_count, self.color_count), dtype=wp.float32, device=self.device
        )
        self.rigid_world_color_count = wp.zeros(
            (rigid_adapter.world_contact_count.shape[0] if rigid_adapter is not None else 0, self.color_count),
            dtype=wp.int32,
            device=self.device,
        )
        self.body_locks = wp.full(body_count, _LOCK_FREE, dtype=wp.int32, device=self.device)
        self.particle_locks = wp.full(particle_count, _LOCK_FREE, dtype=wp.int32, device=self.device)
        self.friction = _ColorFamily(
            rigid_adapter.friction_capacity if rigid_adapter is not None else 0, self.color_count, self.device
        )
        self.contact = _ColorFamily(
            rigid_adapter.contact_capacity if rigid_adapter is not None else 0, self.color_count, self.device
        )
        self.limit = _ColorFamily(
            rigid_adapter.limit_capacity if rigid_adapter is not None else 0, self.color_count, self.device
        )
        self.deformable = _ColorFamily(
            deformable_contacts.contact_capacity if deformable_contacts is not None else 0,
            self.color_count,
            self.device,
        )
        self.friction_delassus = wp.zeros(self.friction.capacity, dtype=wp.float32, device=self.device)
        self.contact_delassus = wp.zeros(self.contact.capacity, dtype=wp.mat33f, device=self.device)
        self.limit_delassus = wp.zeros(self.limit.capacity, dtype=wp.float32, device=self.device)
        self._families = (self.friction, self.contact, self.limit)
        self.rigid_worker_count = _bounded_worker_count(rigid_capacity, self.device)
        deformable_per_color_capacity = (self.deformable.capacity + self.color_count - 1) // self.color_count
        self.deformable_projection_worker_count = _bounded_worker_count(deformable_per_color_capacity, self.device)
        self._fuse_rigid_families = sum(family.capacity > 0 for family in self._families) > 1
        self.apply_worker_count = _bounded_worker_count(max(body_count, particle_count), self.device)
        # Split domain kernels preserve occupancy on older architectures; launch fusion wins on Blackwell and newer.
        self._combine_apply_domains = self.device.is_cuda and self.device.arch >= 100

    def matches(self, deformable_contacts) -> bool:
        return self.deformable_contacts is deformable_contacts

    def _launch_rigid_families(self, kernel, extra_inputs: tuple = ()) -> None:
        rigid_adapter = self.rigid_adapter
        if rigid_adapter is None:
            return
        entries = (
            (
                self.friction,
                rigid_adapter.friction_world,
                rigid_adapter.friction_local,
                rigid_adapter.world_friction_count,
                rigid_adapter.friction_body_first,
                rigid_adapter.friction_body_second,
                0,
            ),
            (
                self.contact,
                rigid_adapter.contact_world,
                rigid_adapter.contact_local,
                rigid_adapter.world_contact_count,
                rigid_adapter.contact_body_first,
                rigid_adapter.contact_body_second,
                1,
            ),
            (
                self.limit,
                rigid_adapter.limit_world,
                rigid_adapter.limit_local,
                rigid_adapter.world_limit_count,
                rigid_adapter.limit_body_first,
                rigid_adapter.limit_body_second,
                2,
            ),
        )
        key_offset = 0
        for family, worlds, local, counts, first, second, salt in entries:
            if family.capacity > 0:
                if kernel is _assign_two_endpoint_colors:
                    inputs = [worlds, local, counts, first, second, salt, self.color_count]
                    outputs = [family.colors]
                elif kernel is _count_two_endpoint_occupancy:
                    inputs = [worlds, local, counts, first, second, family.colors]
                    outputs = [self.body_occupancy, self.rigid_world_color_count]
                elif kernel is _propose_and_claim_two_endpoint_repairs:
                    inputs = [
                        worlds,
                        local,
                        counts,
                        first,
                        second,
                        self.color_count,
                        key_offset,
                        family.colors,
                        self.body_occupancy,
                    ]
                    outputs = [family.proposals, self.body_locks]
                else:
                    inputs = [worlds, first, second, key_offset, family.colors, family.proposals, self.body_locks]
                    outputs = [self.body_occupancy, self.rigid_world_color_count]
                wp.launch(kernel, dim=family.capacity, inputs=inputs, outputs=outputs, device=self.device)
            key_offset += family.capacity

    def build_colors(self) -> None:
        self.body_occupancy.zero_()
        self.particle_occupancy.zero_()
        self.particle_majorizer_weight_sum.zero_()
        self.rigid_world_color_count.zero_()
        self._launch_rigid_families(_assign_two_endpoint_colors)
        self._launch_rigid_families(_count_two_endpoint_occupancy)
        deformable = self.deformable_contacts
        deformable_key_offset = sum(family.capacity for family in self._families)
        if deformable is not None:
            wp.launch(
                _assign_deformable_colors,
                dim=deformable.contact_capacity,
                inputs=[
                    deformable.particle_indices,
                    deformable.contact_world,
                    deformable.body,
                    deformable.status,
                    self.color_count,
                ],
                outputs=[self.deformable.colors],
                device=self.device,
            )
            wp.launch(
                _count_deformable_occupancy,
                dim=deformable.contact_capacity,
                inputs=[
                    deformable.particle_indices,
                    deformable.body,
                    deformable.status,
                    self.deformable.colors,
                ],
                outputs=[self.particle_occupancy, self.body_occupancy],
                device=self.device,
            )
        for _repair in range(_COLOR_REPAIR_PASSES):
            self.body_locks.fill_(_LOCK_FREE)
            self.particle_locks.fill_(_LOCK_FREE)
            self._launch_rigid_families(_propose_and_claim_two_endpoint_repairs)
            if deformable is not None:
                wp.launch(
                    _propose_and_claim_deformable_repairs,
                    dim=deformable.contact_capacity,
                    inputs=[
                        deformable.particle_indices,
                        deformable.body,
                        deformable.status,
                        self.color_count,
                        deformable_key_offset,
                        self.deformable.colors,
                        self.particle_occupancy,
                        self.body_occupancy,
                    ],
                    outputs=[self.deformable.proposals, self.particle_locks, self.body_locks],
                    device=self.device,
                )
            self._launch_rigid_families(_commit_two_endpoint_repairs)
            if deformable is not None:
                wp.launch(
                    _commit_deformable_repairs,
                    dim=deformable.contact_capacity,
                    inputs=[
                        deformable.particle_indices,
                        deformable.body,
                        deformable_key_offset,
                        self.deformable.colors,
                        self.deformable.proposals,
                        self.particle_locks,
                        self.body_locks,
                    ],
                    outputs=[self.particle_occupancy, self.body_occupancy],
                    device=self.device,
                )
        for family in (*self._families, self.deformable):
            family.compact(self.color_count, self.device)
        if deformable is not None:
            wp.launch(
                _accumulate_deformable_majorizer_weights,
                dim=deformable.contact_capacity,
                inputs=[
                    deformable.particle_indices,
                    deformable.coefficients,
                    deformable.status,
                    self.deformable.colors,
                ],
                outputs=[self.particle_majorizer_weight_sum],
                device=self.device,
            )

    def prepare(self, inverse_weight: wp.array[mat66f] | None, prepared_status: wp.array[wp.int32]) -> None:
        self.build_colors()
        rigid_adapter = self.rigid_adapter
        empty_body_weight = None
        if rigid_adapter is not None:
            prepare_jacobi_projection_data(
                rigid_adapter.friction_world,
                rigid_adapter.friction_local,
                rigid_adapter.world_friction_count,
                rigid_adapter.friction_body_first,
                rigid_adapter.friction_body_second,
                rigid_adapter.friction_jacobian_first,
                rigid_adapter.friction_jacobian_second,
                rigid_adapter.contact_world,
                rigid_adapter.contact_local,
                rigid_adapter.world_contact_count,
                rigid_adapter.contact_body_first,
                rigid_adapter.contact_body_second,
                rigid_adapter.contact_jacobian_first,
                rigid_adapter.contact_jacobian_second,
                rigid_adapter.contact_bias,
                rigid_adapter.contact_friction,
                rigid_adapter.limit_world,
                rigid_adapter.limit_local,
                rigid_adapter.world_limit_count,
                rigid_adapter.limit_body_first,
                rigid_adapter.limit_body_second,
                rigid_adapter.limit_jacobian_first,
                rigid_adapter.limit_jacobian_second,
                rigid_adapter.body_constraint_count,
                rigid_adapter.static_body_constraint_count,
                inverse_weight,
                rigid_adapter.friction_projection_delassus,
                rigid_adapter.contact_projection_delassus,
                rigid_adapter.limit_projection_delassus,
                prepared_status,
            )
            if self.deformable_contacts is not None:
                self.deformable_contacts.prepare_rigid_projection(
                    rigid_adapter.body_constraint_count,
                    rigid_adapter.static_body_constraint_count,
                    inverse_weight,
                    prepared_status,
                )
            empty_body_weight = inverse_weight
            if self.rigid_worker_count > 0:
                for color in range(self.color_count):
                    if self._fuse_rigid_families:
                        wp.launch(
                            _prepare_rigid_colored,
                            dim=self.rigid_worker_count,
                            inputs=[
                                self.rigid_worker_count,
                                color,
                                self.friction.counts,
                                self.friction.offsets,
                                self.friction.order,
                                rigid_adapter.friction_world,
                                rigid_adapter.friction_body_first,
                                rigid_adapter.friction_body_second,
                                rigid_adapter.friction_jacobian_first,
                                rigid_adapter.friction_jacobian_second,
                                self.contact.counts,
                                self.contact.offsets,
                                self.contact.order,
                                rigid_adapter.contact_world,
                                rigid_adapter.contact_body_first,
                                rigid_adapter.contact_body_second,
                                rigid_adapter.contact_jacobian_first,
                                rigid_adapter.contact_jacobian_second,
                                rigid_adapter.contact_bias,
                                rigid_adapter.contact_friction,
                                self.limit.counts,
                                self.limit.offsets,
                                self.limit.order,
                                rigid_adapter.limit_world,
                                rigid_adapter.limit_body_first,
                                rigid_adapter.limit_body_second,
                                rigid_adapter.limit_jacobian_first,
                                rigid_adapter.limit_jacobian_second,
                                self.body_occupancy,
                                inverse_weight,
                            ],
                            outputs=[
                                self.friction_delassus,
                                self.contact_delassus,
                                self.limit_delassus,
                                prepared_status,
                            ],
                            device=self.device,
                            block_dim=_COLOR_BLOCK_DIM,
                        )
                    else:
                        if self.friction.capacity > 0:
                            wp.launch(
                                _prepare_frictions_colored,
                                dim=self.friction.worker_count,
                                inputs=[
                                    self.friction.worker_count,
                                    color,
                                    self.friction.counts,
                                    self.friction.offsets,
                                    self.friction.order,
                                    rigid_adapter.friction_world,
                                    rigid_adapter.friction_body_first,
                                    rigid_adapter.friction_body_second,
                                    rigid_adapter.friction_jacobian_first,
                                    rigid_adapter.friction_jacobian_second,
                                    self.body_occupancy,
                                    inverse_weight,
                                ],
                                outputs=[self.friction_delassus, prepared_status],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                        elif self.contact.capacity > 0:
                            wp.launch(
                                _prepare_contacts_colored,
                                dim=self.contact.worker_count,
                                inputs=[
                                    self.contact.worker_count,
                                    color,
                                    self.contact.counts,
                                    self.contact.offsets,
                                    self.contact.order,
                                    rigid_adapter.contact_world,
                                    rigid_adapter.contact_body_first,
                                    rigid_adapter.contact_body_second,
                                    rigid_adapter.contact_jacobian_first,
                                    rigid_adapter.contact_jacobian_second,
                                    rigid_adapter.contact_bias,
                                    rigid_adapter.contact_friction,
                                    self.body_occupancy,
                                    inverse_weight,
                                ],
                                outputs=[self.contact_delassus, prepared_status],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                        else:
                            wp.launch(
                                _prepare_frictions_colored,
                                dim=self.limit.worker_count,
                                inputs=[
                                    self.limit.worker_count,
                                    color,
                                    self.limit.counts,
                                    self.limit.offsets,
                                    self.limit.order,
                                    rigid_adapter.limit_world,
                                    rigid_adapter.limit_body_first,
                                    rigid_adapter.limit_body_second,
                                    rigid_adapter.limit_jacobian_first,
                                    rigid_adapter.limit_jacobian_second,
                                    self.body_occupancy,
                                    inverse_weight,
                                ],
                                outputs=[self.limit_delassus, prepared_status],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
        else:
            prepared_status.fill_(PROJECTION_STATUS_VALID)
        deformable = self.deformable_contacts
        if deformable is not None:
            include_rigid = rigid_adapter is not None
            body_weight = empty_body_weight if include_rigid else deformable._empty_body_inverse_weight
            for color in range(self.color_count):
                wp.launch(
                    _prepare_deformable_colored,
                    dim=self.deformable.worker_count,
                    inputs=[
                        self.deformable.worker_count,
                        color,
                        self.deformable.counts,
                        self.deformable.offsets,
                        self.deformable.order,
                        deformable.particle_indices,
                        deformable.coefficients,
                        deformable.contact_world,
                        deformable.body,
                        deformable.body_jacobian,
                        deformable.bias,
                        deformable.friction,
                        self.particle_majorizer_weight_sum,
                        self.body_occupancy,
                        deformable.cloth_system.inverse_weight,
                        body_weight,
                        include_rigid,
                    ],
                    outputs=[
                        deformable.status,
                        deformable.gauss_seidel_scalar_delassus,
                        deformable.gauss_seidel_delassus,
                        prepared_status,
                        deformable.world_status,
                    ],
                    device=self.device,
                    block_dim=_COLOR_BLOCK_DIM,
                )
            wp.launch(
                _merge_rigid_prepared_status,
                dim=prepared_status.shape[0],
                inputs=[deformable.world_status, deformable.global_status],
                outputs=[prepared_status],
                device=self.device,
            )

    def project(
        self,
        iterations: int,
        world_active: wp.array[wp.bool],
        body_world: wp.array[wp.int32] | None,
        inverse_weight: wp.array[mat66f] | None,
        projected_twist: wp.array[vec6f] | None,
        twist_delta: wp.array[vec6f] | None,
        projected_velocity: wp.array[wp.vec3f] | None,
        prepared_status: wp.array[wp.int32],
        projection_status: wp.array[wp.int32],
    ) -> None:
        rigid_adapter = self.rigid_adapter
        deformable = self.deformable_contacts
        world_count = world_active.shape[0]
        world_body_offset = None
        world_body_count = None
        world_friction_offset = None
        world_contact_offset = None
        world_limit_offset = None
        if rigid_adapter is not None:
            model_info = getattr(getattr(rigid_adapter, "model", None), "info", None)
            if model_info is not None:
                world_body_offset = model_info.bodies_offset
                world_body_count = model_info.num_bodies
            world_friction_offset = getattr(rigid_adapter, "world_friction_offset", None)
            world_contact_offset = getattr(rigid_adapter, "world_contact_offset", None)
            world_limit_offset = getattr(rigid_adapter, "world_limit_offset", None)
        use_world_projection = _can_fuse_rigid_projection_by_world(
            self.device,
            world_count,
            has_deformable_contacts=deformable is not None,
            required_world_arrays=(
                world_body_offset,
                world_body_count,
                world_friction_offset,
                world_contact_offset,
                world_limit_offset,
            ),
            parallel_constraint_capacity=(
                (
                    rigid_adapter.friction_capacity
                    + rigid_adapter.contact_capacity
                    + rigid_adapter.limit_capacity
                    + self.color_count
                    - 1
                )
                // self.color_count
                if rigid_adapter is not None
                else None
            ),
            world_block_dim=_WORLD_COLOR_BLOCK_DIM,
            minimum_blocks_per_sm=1,
        )
        if use_world_projection:
            state = _make_rigid_projection_state(
                world_active,
                projected_twist,
                twist_delta,
                projection_status,
                self.body_occupancy,
                inverse_weight,
            )
            contact_data = _make_projection_struct(
                _RigidContactProjectionData,
                body_first=rigid_adapter.contact_body_first,
                body_second=rigid_adapter.contact_body_second,
                jacobian_first=rigid_adapter.contact_jacobian_first,
                jacobian_second=rigid_adapter.contact_jacobian_second,
                delassus=self.contact_delassus,
                bias=rigid_adapter.contact_bias,
                friction=rigid_adapter.contact_friction,
                reaction=rigid_adapter.contact_reaction,
            )
            data = _make_projection_struct(
                _RigidColoredWorldData,
                world_color_count=self.rigid_world_color_count,
                world_friction_offset=world_friction_offset,
                world_friction_count=rigid_adapter.world_friction_count,
                friction_colors=self.friction.colors,
                friction_body_first=rigid_adapter.friction_body_first,
                friction_body_second=rigid_adapter.friction_body_second,
                friction_jacobian_first=rigid_adapter.friction_jacobian_first,
                friction_jacobian_second=rigid_adapter.friction_jacobian_second,
                friction_bound=rigid_adapter.friction_impulse_bound,
                friction_colored_delassus=self.friction_delassus,
                friction_jacobi_delassus=rigid_adapter.friction_projection_delassus,
                friction_reaction=rigid_adapter.friction_reaction,
                world_contact_offset=world_contact_offset,
                world_contact_count=rigid_adapter.world_contact_count,
                contact_colors=self.contact.colors,
                contact_jacobi_delassus=rigid_adapter.contact_projection_delassus,
                world_limit_offset=world_limit_offset,
                world_limit_count=rigid_adapter.world_limit_count,
                limit_colors=self.limit.colors,
                limit_body_first=rigid_adapter.limit_body_first,
                limit_body_second=rigid_adapter.limit_body_second,
                limit_jacobian_first=rigid_adapter.limit_jacobian_first,
                limit_jacobian_second=rigid_adapter.limit_jacobian_second,
                limit_bias=rigid_adapter.limit_bias,
                limit_colored_delassus=self.limit_delassus,
                limit_jacobi_delassus=rigid_adapter.limit_projection_delassus,
                limit_reaction=rigid_adapter.limit_reaction,
            )
            wp.launch(
                _project_rigid_colored_by_world,
                dim=(world_count, _WORLD_COLOR_BLOCK_DIM),
                block_dim=_WORLD_COLOR_BLOCK_DIM,
                inputs=[
                    iterations,
                    self.color_count,
                    world_body_offset,
                    world_body_count,
                    data,
                    contact_data,
                    state,
                    prepared_status,
                ],
                device=self.device,
            )
            return
        if rigid_adapter is not None:
            wp.launch(
                _initialize_jacobi_projection_status,
                dim=world_active.shape[0],
                inputs=[world_active, prepared_status],
                outputs=[projection_status],
                device=self.device,
            )
            twist_delta.zero_()
            if deformable is not None:
                deformable.particle_delta.zero_()
            if self.friction.capacity > 0:
                wp.launch(
                    _warmstart_frictions_jacobi,
                    dim=self.friction.capacity,
                    inputs=[
                        rigid_adapter.friction_world,
                        rigid_adapter.friction_local,
                        world_active,
                        prepared_status,
                        rigid_adapter.world_friction_count,
                        rigid_adapter.friction_body_first,
                        rigid_adapter.friction_body_second,
                        rigid_adapter.friction_jacobian_first,
                        rigid_adapter.friction_jacobian_second,
                        inverse_weight,
                        True,
                        rigid_adapter.friction_reaction,
                    ],
                    outputs=[twist_delta],
                    device=self.device,
                )
            if self.contact.capacity > 0:
                wp.launch(
                    _warmstart_contacts_jacobi,
                    dim=self.contact.capacity,
                    inputs=[
                        rigid_adapter.contact_world,
                        rigid_adapter.contact_local,
                        world_active,
                        prepared_status,
                        rigid_adapter.world_contact_count,
                        rigid_adapter.contact_body_first,
                        rigid_adapter.contact_body_second,
                        rigid_adapter.contact_jacobian_first,
                        rigid_adapter.contact_jacobian_second,
                        inverse_weight,
                        True,
                        rigid_adapter.contact_reaction,
                    ],
                    outputs=[twist_delta],
                    device=self.device,
                )
            if self.limit.capacity > 0:
                wp.launch(
                    _warmstart_limits_jacobi,
                    dim=self.limit.capacity,
                    inputs=[
                        rigid_adapter.limit_world,
                        rigid_adapter.limit_local,
                        world_active,
                        prepared_status,
                        rigid_adapter.world_limit_count,
                        rigid_adapter.limit_body_first,
                        rigid_adapter.limit_body_second,
                        rigid_adapter.limit_jacobian_first,
                        rigid_adapter.limit_jacobian_second,
                        inverse_weight,
                        True,
                        rigid_adapter.limit_reaction,
                    ],
                    outputs=[twist_delta],
                    device=self.device,
                )
            if deformable is not None:
                deformable.accumulate_rigid_reaction_warm_start(
                    world_active,
                    prepared_status,
                    deformable.cloth_system.inverse_weight,
                    inverse_weight,
                    True,
                    projected_velocity,
                    projected_twist,
                    twist_delta,
                    projection_status,
                )
            if deformable is not None and self._combine_apply_domains:
                wp.launch(
                    _apply_body_particle_delta_colored,
                    dim=self.apply_worker_count,
                    inputs=[
                        self.apply_worker_count,
                        projected_twist.shape[0],
                        projected_velocity.shape[0],
                        body_world,
                        deformable.cloth_system.topology.packed_solve_world,
                        world_active,
                        projection_status,
                    ],
                    outputs=[twist_delta, deformable.particle_delta, projected_twist, projected_velocity],
                    device=self.device,
                    block_dim=_COLOR_BLOCK_DIM,
                )
            else:
                wp.launch(
                    _apply_body_delta,
                    dim=projected_twist.shape[0],
                    inputs=[body_world, world_active, projection_status],
                    outputs=[twist_delta, projected_twist],
                    device=self.device,
                )
                if deformable is not None:
                    wp.launch(
                        _apply_particle_delta_colored,
                        dim=projected_velocity.shape[0],
                        inputs=[deformable.cloth_system.topology.packed_solve_world, world_active, projection_status],
                        outputs=[deformable.particle_delta, projected_velocity],
                        device=self.device,
                    )
        else:
            wp.copy(projection_status, prepared_status)
            deformable.apply_reaction_warm_start(projected_velocity)
            deformable.particle_delta.zero_()

        rigid_projection_state = None
        rigid_contact_data = None
        friction_index = None
        contact_index = None
        limit_index = None
        deformable_index = None
        deformable_data = None
        deformable_state = None
        if rigid_adapter is not None:
            rigid_projection_state = _make_rigid_projection_state(
                world_active,
                projected_twist,
                twist_delta,
                projection_status,
                self.body_occupancy,
                inverse_weight,
            )
            rigid_contact_data = _make_projection_struct(
                _RigidContactProjectionData,
                body_first=rigid_adapter.contact_body_first,
                body_second=rigid_adapter.contact_body_second,
                jacobian_first=rigid_adapter.contact_jacobian_first,
                jacobian_second=rigid_adapter.contact_jacobian_second,
                delassus=self.contact_delassus,
                bias=rigid_adapter.contact_bias,
                friction=rigid_adapter.contact_friction,
                reaction=rigid_adapter.contact_reaction,
            )
            friction_index = _make_colored_projection_index(
                rigid_adapter.friction_world,
                self.friction.counts,
                self.friction.offsets,
                self.friction.order,
            )
            contact_index = _make_colored_projection_index(
                rigid_adapter.contact_world,
                self.contact.counts,
                self.contact.offsets,
                self.contact.order,
            )
            limit_index = _make_colored_projection_index(
                rigid_adapter.limit_world,
                self.limit.counts,
                self.limit.offsets,
                self.limit.order,
            )
        if deformable is not None:
            deformable_index = _make_colored_projection_index(
                deformable.contact_world,
                self.deformable.counts,
                self.deformable.offsets,
                self.deformable.order,
            )
            if rigid_adapter is not None:
                deformable_data = _make_projection_struct(
                    _DeformableRigidContactProjectionData,
                    particle_indices=deformable.particle_indices,
                    coefficients=deformable.coefficients,
                    body=deformable.body,
                    frame=deformable.frame,
                    body_jacobian=deformable.body_jacobian,
                    delassus=deformable.gauss_seidel_delassus,
                    bias=deformable.bias,
                    friction=deformable.friction,
                    reaction=deformable.reaction,
                    status=deformable.status,
                    contact_world_status=deformable.world_status,
                )
                deformable_state = _make_projection_struct(
                    _DeformableRigidProjectionState,
                    world_active=world_active,
                    projected_twist=projected_twist,
                    twist_delta=twist_delta,
                    world_status=projection_status,
                    occupancy=self.body_occupancy,
                    inverse_weight=inverse_weight,
                    particle_occupancy=self.particle_occupancy,
                    particle_inverse_weight=deformable.cloth_system.inverse_weight,
                    projected_velocity=projected_velocity,
                    particle_delta=deformable.particle_delta,
                )
            else:
                deformable_data = _make_projection_struct(
                    _ParticleContactProjectionData,
                    particle_indices=deformable.particle_indices,
                    coefficients=deformable.coefficients,
                    contact_body=deformable.body,
                    frame=deformable.frame,
                    bias=deformable.bias,
                    friction=deformable.friction,
                    delassus=deformable.gauss_seidel_scalar_delassus,
                    status=deformable.status,
                    reaction=deformable.reaction,
                )
                deformable_state = _make_projection_struct(
                    _ParticleContactColoredState,
                    world_active=world_active,
                    world_status=projection_status,
                    contact_world_status=deformable.world_status,
                    occupancy=self.particle_occupancy,
                    inverse_weight=deformable.cloth_system.inverse_weight,
                    projected_velocity=projected_velocity,
                    particle_delta=deformable.particle_delta,
                )
        for _iteration in range(iterations):
            for color in range(self.color_count):
                if rigid_adapter is not None:
                    if self.rigid_worker_count > 0:
                        if self._fuse_rigid_families:
                            wp.launch(
                                _project_rigid_colored,
                                dim=self.rigid_worker_count,
                                inputs=[
                                    self.rigid_worker_count,
                                    color,
                                    self.friction.counts,
                                    self.friction.offsets,
                                    self.friction.order,
                                    rigid_adapter.friction_world,
                                    rigid_adapter.friction_body_first,
                                    rigid_adapter.friction_body_second,
                                    rigid_adapter.friction_jacobian_first,
                                    rigid_adapter.friction_jacobian_second,
                                    rigid_adapter.friction_impulse_bound,
                                    self.friction_delassus,
                                    self.contact.counts,
                                    self.contact.offsets,
                                    self.contact.order,
                                    rigid_adapter.contact_world,
                                    rigid_contact_data,
                                    self.limit.counts,
                                    self.limit.offsets,
                                    self.limit.order,
                                    rigid_adapter.limit_world,
                                    rigid_adapter.limit_body_first,
                                    rigid_adapter.limit_body_second,
                                    rigid_adapter.limit_jacobian_first,
                                    rigid_adapter.limit_jacobian_second,
                                    rigid_adapter.limit_bias,
                                    self.limit_delassus,
                                    rigid_projection_state,
                                ],
                                outputs=[
                                    rigid_adapter.friction_reaction,
                                    rigid_adapter.limit_reaction,
                                ],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                        elif self.friction.capacity > 0:
                            wp.launch(
                                _make_project_scalar_kernel(True, False),
                                dim=self.friction.worker_count,
                                inputs=[
                                    self.friction.worker_count,
                                    color,
                                    friction_index,
                                    rigid_adapter.friction_body_first,
                                    rigid_adapter.friction_body_second,
                                    rigid_adapter.friction_jacobian_first,
                                    rigid_adapter.friction_jacobian_second,
                                    rigid_adapter.friction_impulse_bound,
                                    rigid_adapter.friction_impulse_bound,
                                    self.friction_delassus,
                                    rigid_adapter.friction_reaction,
                                    rigid_projection_state,
                                ],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                        elif self.contact.capacity > 0:
                            wp.launch(
                                _make_project_contacts_kernel(True),
                                dim=self.contact.worker_count,
                                inputs=[
                                    self.contact.worker_count,
                                    color,
                                    contact_index,
                                    rigid_contact_data,
                                    rigid_projection_state,
                                ],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                        else:
                            wp.launch(
                                _make_project_scalar_kernel(True, True),
                                dim=self.limit.worker_count,
                                inputs=[
                                    self.limit.worker_count,
                                    color,
                                    limit_index,
                                    rigid_adapter.limit_body_first,
                                    rigid_adapter.limit_body_second,
                                    rigid_adapter.limit_jacobian_first,
                                    rigid_adapter.limit_jacobian_second,
                                    rigid_adapter.limit_bias,
                                    rigid_adapter.limit_bias,
                                    self.limit_delassus,
                                    rigid_adapter.limit_reaction,
                                    rigid_projection_state,
                                ],
                                device=self.device,
                                block_dim=_COLOR_BLOCK_DIM,
                            )
                if deformable is not None:
                    include_rigid = rigid_adapter is not None
                    if include_rigid:
                        wp.launch(
                            _make_project_contacts_kernel(
                                True,
                                True,
                                DEFORMABLE_CONTACT_STATUS_VALID,
                                DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE,
                            ),
                            dim=self.deformable_projection_worker_count,
                            inputs=[
                                self.deformable_projection_worker_count,
                                color,
                                deformable_index,
                                deformable_data,
                                deformable_state,
                            ],
                            device=self.device,
                            block_dim=_COLOR_BLOCK_DIM,
                        )
                    else:
                        wp.launch(
                            _make_project_particle_contacts_kernel(True),
                            dim=self.deformable_projection_worker_count,
                            inputs=[
                                self.deformable_projection_worker_count,
                                color,
                                deformable_index,
                                deformable_data,
                                deformable_state,
                            ],
                            device=self.device,
                            block_dim=_COLOR_BLOCK_DIM,
                        )
                if rigid_adapter is not None and deformable is not None and self._combine_apply_domains:
                    wp.launch(
                        _apply_body_particle_delta_colored,
                        dim=self.apply_worker_count,
                        inputs=[
                            self.apply_worker_count,
                            projected_twist.shape[0],
                            projected_velocity.shape[0],
                            body_world,
                            deformable.cloth_system.topology.packed_solve_world,
                            world_active,
                            projection_status,
                        ],
                        outputs=[twist_delta, deformable.particle_delta, projected_twist, projected_velocity],
                        device=self.device,
                        block_dim=_COLOR_BLOCK_DIM,
                    )
                else:
                    if rigid_adapter is not None:
                        wp.launch(
                            _apply_body_delta,
                            dim=projected_twist.shape[0],
                            inputs=[body_world, world_active, projection_status],
                            outputs=[twist_delta, projected_twist],
                            device=self.device,
                        )
                    if deformable is not None:
                        wp.launch(
                            _apply_particle_delta_colored,
                            dim=projected_velocity.shape[0],
                            inputs=[
                                deformable.cloth_system.topology.packed_solve_world,
                                world_active,
                                projection_status,
                            ],
                            outputs=[deformable.particle_delta, projected_velocity],
                            device=self.device,
                        )

        if rigid_adapter is not None:
            project_constraints_jacobi(
                1,
                world_active,
                body_world,
                rigid_adapter.friction_world,
                rigid_adapter.friction_local,
                rigid_adapter.world_friction_count,
                rigid_adapter.friction_body_first,
                rigid_adapter.friction_body_second,
                rigid_adapter.friction_jacobian_first,
                rigid_adapter.friction_jacobian_second,
                rigid_adapter.friction_impulse_bound,
                rigid_adapter.friction_projection_delassus,
                rigid_adapter.contact_world,
                rigid_adapter.contact_local,
                rigid_adapter.world_contact_count,
                rigid_adapter.contact_body_first,
                rigid_adapter.contact_body_second,
                rigid_adapter.contact_jacobian_first,
                rigid_adapter.contact_jacobian_second,
                rigid_adapter.contact_bias,
                rigid_adapter.contact_friction,
                rigid_adapter.contact_projection_delassus,
                rigid_adapter.limit_world,
                rigid_adapter.limit_local,
                rigid_adapter.world_limit_count,
                rigid_adapter.limit_body_first,
                rigid_adapter.limit_body_second,
                rigid_adapter.limit_jacobian_first,
                rigid_adapter.limit_jacobian_second,
                rigid_adapter.limit_bias,
                rigid_adapter.limit_projection_delassus,
                inverse_weight,
                projected_twist,
                twist_delta,
                rigid_adapter.contact_reaction,
                rigid_adapter.limit_reaction,
                rigid_adapter.friction_reaction,
                prepared_status,
                projection_status,
                deformable_contacts=deformable,
                deformable_projected_velocity=projected_velocity,
                warm_start=False,
            )
        else:
            deformable.project_jacobi_smoothing_sweep(world_active, projected_velocity, projection_status)
