# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body-space unilateral projection sweeps."""

from __future__ import annotations

from functools import cache
from typing import Any

import warp as wp

from ...core.types import mat36f, mat66f, vec6f
from .contact import compute_contact_scaled_alart_curnier_residual, project_contact_coulomb_cone
from .projection import (
    PROJECTION_STATUS_INVALID,
    PROJECTION_STATUS_VALID,
    ContactProjectionData,
    _can_fuse_rigid_projection_by_world,
    _contact_projection_inputs_are_finite,
    compute_contact_delassus,
    compute_limit_delassus,
    prepare_contact_coulomb,
    prepare_contact_coulomb_delassus,
    project_contact_coulomb_local,
    project_friction_local,
    project_limit_local,
)

__all__ = [
    "compute_projection_residuals",
    "prepare_jacobi_projection_data",
    "prepare_physical_projection_data",
    "project_constraints_jacobi",
]

wp.set_module_options({"enable_backward": False})

_JACOBI_CONTACT_BLOCK_DIM = 128
_JACOBI_CONTACT_PROJECTION_BLOCKS_PER_SM = 5
_JACOBI_WORLD_BLOCK_DIM = 128


@wp.struct
class _ProjectionIndexData:
    constraint_world: wp.array[wp.int32]
    constraint_local: wp.array[wp.int32]
    world_constraint_count: wp.array[wp.int32]
    color_counts: wp.array[wp.int32]
    color_offsets: wp.array[wp.int32]
    order: wp.array[wp.int32]


@wp.struct
class _RigidProjectionState:
    world_active: wp.array[wp.bool]
    occupancy: wp.array2d[wp.int32]
    inverse_weight: wp.array[mat66f]
    projected_twist: wp.array[vec6f]
    twist_delta: wp.array[vec6f]
    world_status: wp.array[wp.int32]


@wp.struct
class _RigidContactProjectionData:
    body_first: wp.array[wp.int32]
    body_second: wp.array[wp.int32]
    jacobian_first: wp.array[mat36f]
    jacobian_second: wp.array[mat36f]
    delassus: wp.array[wp.mat33f]
    bias: wp.array[wp.vec3f]
    friction: wp.array[wp.float32]
    reaction: wp.array[wp.vec3f]


def _make_direct_projection_index(
    constraint_world: wp.array[wp.int32],
    constraint_local: wp.array[wp.int32] | None = None,
    world_constraint_count: wp.array[wp.int32] | None = None,
) -> _ProjectionIndexData:
    index = _ProjectionIndexData()
    index.constraint_world = constraint_world
    if constraint_local is not None:
        index.constraint_local = constraint_local
    if world_constraint_count is not None:
        index.world_constraint_count = world_constraint_count
    return index


def _make_colored_projection_index(
    constraint_world: wp.array[wp.int32],
    color_counts: wp.array[wp.int32],
    color_offsets: wp.array[wp.int32],
    order: wp.array[wp.int32],
) -> _ProjectionIndexData:
    index = _ProjectionIndexData()
    index.constraint_world = constraint_world
    index.color_counts = color_counts
    index.color_offsets = color_offsets
    index.order = order
    return index


def _make_rigid_projection_state(
    world_active: wp.array[wp.bool],
    projected_twist: wp.array[vec6f],
    twist_delta: wp.array[vec6f],
    world_status: wp.array[wp.int32],
    occupancy: wp.array2d[wp.int32] | None = None,
    inverse_weight: wp.array[mat66f] | None = None,
) -> _RigidProjectionState:
    state = _RigidProjectionState()
    state.world_active = world_active
    state.projected_twist = projected_twist
    state.twist_delta = twist_delta
    state.world_status = world_status
    if occupancy is not None:
        state.occupancy = occupancy
    if inverse_weight is not None:
        state.inverse_weight = inverse_weight
    return state


def _make_projection_struct(struct_type: type, **fields: Any) -> Any:
    value = struct_type()
    for name, field in fields.items():
        setattr(value, name, field)
    return value


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


@wp.func
def _prepare_contact_projection(
    contact: wp.int32,
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
) -> ContactProjectionData:
    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    first = contact_body_first[contact]
    second = contact_body_second[contact]
    if first < 0 and second < 0:
        data = ContactProjectionData()
        data.delassus = wp.mat33f(0.0)
        data.status = PROJECTION_STATUS_VALID
        return data
    if first >= 0:
        inverse_weight_first = inverse_weight[first]
    if second >= 0:
        inverse_weight_second = inverse_weight[second]
    return prepare_contact_coulomb(
        contact_jacobian_first[contact],
        inverse_weight_first,
        contact_jacobian_second[contact],
        inverse_weight_second,
        contact_bias[contact],
        contact_friction[contact],
    )


@wp.kernel
def _prepare_contact_physical_projection_data(
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    physical_delassus: wp.array[wp.mat33f],
    prepared_delassus: wp.array[wp.mat33f],
    world_status: wp.array[wp.int32],
):
    contact = wp.tid()
    world = contact_world[contact]
    if contact_local[contact] >= world_contact_count[world]:
        return

    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    first = contact_body_first[contact]
    second = contact_body_second[contact]
    if first < 0 and second < 0:
        physical_delassus[contact] = wp.mat33f(0.0)
        prepared_delassus[contact] = wp.mat33f(0.0)
        return
    if first >= 0:
        inverse_weight_first = inverse_weight[first]
    if second >= 0:
        inverse_weight_second = inverse_weight[second]
    jacobian_first = contact_jacobian_first[contact]
    jacobian_second = contact_jacobian_second[contact]
    physical = compute_contact_delassus(
        jacobian_first,
        inverse_weight_first,
        jacobian_second,
        inverse_weight_second,
    )
    data = prepare_contact_coulomb_delassus(
        physical,
        contact_bias[contact],
        contact_friction[contact],
    )
    if not _contact_projection_inputs_are_finite(
        jacobian_first,
        inverse_weight_first,
        jacobian_second,
        inverse_weight_second,
    ):
        data.status = PROJECTION_STATUS_INVALID
    physical_delassus[contact] = physical
    prepared_delassus[contact] = data.delassus
    if data.status == PROJECTION_STATUS_INVALID:
        world_status[world] = data.status


@wp.kernel
def _prepare_scalar_projection_data(
    constraint_world: wp.array[wp.int32],
    constraint_local: wp.array[wp.int32],
    world_constraint_count: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    constraint = wp.tid()
    world = constraint_world[constraint]
    if constraint_local[constraint] >= world_constraint_count[world]:
        return

    first = body_first[constraint]
    second = body_second[constraint]
    if first < 0 and second < 0:
        delassus[constraint] = 0.0
        return
    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    if first >= 0:
        inverse_weight_first = inverse_weight[first]
    if second >= 0:
        inverse_weight_second = inverse_weight[second]
    value = compute_limit_delassus(
        jacobian_first[constraint],
        inverse_weight_first,
        jacobian_second[constraint],
        inverse_weight_second,
    )
    delassus[constraint] = value
    if not wp.isfinite(value) or value <= 0.0:
        world_status[world] = PROJECTION_STATUS_INVALID


@wp.func
def _atomic_add_twist(values: wp.array[vec6f], body: wp.int32, increment: vec6f):
    if body >= 0:
        wp.atomic_add(values, body, increment)


@wp.func
def _accumulate_twist_by_occupancy(
    values: wp.array[vec6f],
    occupancy: wp.array2d[wp.int32],
    body: wp.int32,
    color: wp.int32,
    increment: vec6f,
):
    if occupancy[body, color] == 1:
        values[body] += increment
    else:
        wp.atomic_add(values, body, increment)


@wp.func
def _is_finite_twist(value: vec6f) -> wp.bool:
    result = wp.bool(True)
    for index in range(6):
        result = result and wp.isfinite(value[index])
    return result


@wp.func
def _is_finite_vec3(value: wp.vec3f) -> wp.bool:
    return wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])


@wp.func
def _is_zero_vec3(value: wp.vec3f) -> wp.bool:
    return value[0] == 0.0 and value[1] == 0.0 and value[2] == 0.0


@wp.func
def _compute_contact_velocity(
    contact: wp.int32,
    first: wp.int32,
    second: wp.int32,
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    projected_twist: wp.array[vec6f],
) -> wp.vec3f:
    velocity = contact_bias[contact]
    if first >= 0:
        velocity += contact_jacobian_first[contact] @ projected_twist[first]
    if second >= 0:
        velocity += contact_jacobian_second[contact] @ projected_twist[second]
    return velocity


@wp.func
def _project_rigid_contact_colored(
    contact: wp.int32,
    world: wp.int32,
    target_color: wp.int32,
    data: Any,
    state: Any,
):
    first = data.body_first[contact]
    second = data.body_second[contact]
    if first < 0 and second < 0:
        data.reaction[contact] = wp.vec3f(0.0)
        return
    velocity = _compute_contact_velocity(
        contact,
        first,
        second,
        data.jacobian_first,
        data.jacobian_second,
        data.bias,
        state.projected_twist,
    )
    projection = project_contact_coulomb_local(
        velocity,
        data.reaction[contact],
        data.delassus[contact],
        data.friction[contact],
    )
    if projection.status == PROJECTION_STATUS_INVALID:
        state.world_status[world] = PROJECTION_STATUS_INVALID
        return
    correction_first = vec6f(0.0)
    correction_second = vec6f(0.0)
    if first >= 0:
        correction_first = state.inverse_weight[first] @ (
            wp.transpose(data.jacobian_first[contact]) @ projection.reaction_delta
        )
    if second >= 0:
        correction_second = state.inverse_weight[second] @ (
            wp.transpose(data.jacobian_second[contact]) @ projection.reaction_delta
        )
    if not _is_finite_twist(correction_first) or not _is_finite_twist(correction_second):
        state.world_status[world] = PROJECTION_STATUS_INVALID
        return
    if first >= 0:
        if first == second:
            _accumulate_twist_by_occupancy(
                state.twist_delta,
                state.occupancy,
                first,
                target_color,
                correction_first + correction_second,
            )
        else:
            _accumulate_twist_by_occupancy(
                state.twist_delta,
                state.occupancy,
                first,
                target_color,
                correction_first,
            )
    if second >= 0 and second != first:
        _accumulate_twist_by_occupancy(
            state.twist_delta,
            state.occupancy,
            second,
            target_color,
            correction_second,
        )
    data.reaction[contact] = projection.reaction


@wp.func
def _project_rigid_scalar_colored(
    constraint: wp.int32,
    world: wp.int32,
    target_color: wp.int32,
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    bias: wp.array[wp.float32],
    bound: wp.array[wp.float32],
    delassus: wp.array[wp.float32],
    reaction: wp.array[wp.float32],
    state: Any,
    limit: wp.bool,
):
    first = body_first[constraint]
    second = body_second[constraint]
    if first < 0 and second < 0:
        reaction[constraint] = 0.0
        return
    velocity = wp.float32(0.0)
    if limit:
        velocity = bias[constraint]
    if first >= 0:
        velocity += wp.dot(jacobian_first[constraint], state.projected_twist[first])
    if second >= 0:
        velocity += wp.dot(jacobian_second[constraint], state.projected_twist[second])
    if limit:
        projection = project_limit_local(velocity, reaction[constraint], delassus[constraint])
    else:
        projection = project_friction_local(
            velocity,
            reaction[constraint],
            delassus[constraint],
            bound[constraint],
        )
    if projection.status == PROJECTION_STATUS_INVALID:
        state.world_status[world] = PROJECTION_STATUS_INVALID
        return
    correction_first = vec6f(0.0)
    correction_second = vec6f(0.0)
    if first >= 0:
        correction_first = (state.inverse_weight[first] @ jacobian_first[constraint]) * projection.reaction_delta
    if second >= 0:
        correction_second = (state.inverse_weight[second] @ jacobian_second[constraint]) * projection.reaction_delta
    if not _is_finite_twist(correction_first) or not _is_finite_twist(correction_second):
        state.world_status[world] = PROJECTION_STATUS_INVALID
        return
    if first >= 0:
        if first == second:
            _accumulate_twist_by_occupancy(
                state.twist_delta,
                state.occupancy,
                first,
                target_color,
                correction_first + correction_second,
            )
        else:
            _accumulate_twist_by_occupancy(
                state.twist_delta,
                state.occupancy,
                first,
                target_color,
                correction_first,
            )
    if second >= 0 and second != first:
        _accumulate_twist_by_occupancy(
            state.twist_delta,
            state.occupancy,
            second,
            target_color,
            correction_second,
        )
    reaction[constraint] = projection.reaction


@wp.kernel
def _prepare_contacts_jacobi(
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    body_constraint_count: wp.array[wp.int32],
    static_body_constraint_count: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.mat33f],
    world_status: wp.array[wp.int32],
):
    contact = wp.tid()
    world = contact_world[contact]
    if contact_local[contact] >= world_contact_count[world]:
        return

    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    first = contact_body_first[contact]
    second = contact_body_second[contact]
    if first < 0 and second < 0:
        delassus[contact] = wp.mat33f(0.0)
        return
    if first >= 0:
        multiplicity = wp.max(1, body_constraint_count[first] - static_body_constraint_count[first])
        inverse_weight_first = wp.float32(multiplicity) * inverse_weight[first]
    if second >= 0:
        multiplicity = wp.max(1, body_constraint_count[second] - static_body_constraint_count[second])
        inverse_weight_second = wp.float32(multiplicity) * inverse_weight[second]
    data = prepare_contact_coulomb(
        contact_jacobian_first[contact],
        inverse_weight_first,
        contact_jacobian_second[contact],
        inverse_weight_second,
        contact_bias[contact],
        contact_friction[contact],
    )
    delassus[contact] = data.delassus
    if data.status == PROJECTION_STATUS_INVALID:
        world_status[world] = data.status


@wp.kernel
def _prepare_limits_jacobi(
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    body_constraint_count: wp.array[wp.int32],
    static_body_constraint_count: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    limit = wp.tid()
    world = limit_world[limit]
    if limit_local[limit] >= world_limit_count[world]:
        return

    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    first = limit_body_first[limit]
    second = limit_body_second[limit]
    if first < 0 and second < 0:
        delassus[limit] = 0.0
        return
    if first >= 0:
        multiplicity = wp.max(1, body_constraint_count[first] - static_body_constraint_count[first])
        inverse_weight_first = wp.float32(multiplicity) * inverse_weight[first]
    if second >= 0:
        multiplicity = wp.max(1, body_constraint_count[second] - static_body_constraint_count[second])
        inverse_weight_second = wp.float32(multiplicity) * inverse_weight[second]
    value = compute_limit_delassus(
        limit_jacobian_first[limit],
        inverse_weight_first,
        limit_jacobian_second[limit],
        inverse_weight_second,
    )
    delassus[limit] = value
    if not wp.isfinite(value) or value <= 0.0:
        world_status[world] = PROJECTION_STATUS_INVALID


@wp.kernel
def _prepare_frictions_jacobi(
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    body_constraint_count: wp.array[wp.int32],
    static_body_constraint_count: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    friction = wp.tid()
    world = friction_world[friction]
    if friction_local[friction] >= world_friction_count[world]:
        return
    inverse_weight_first = mat66f(0.0)
    inverse_weight_second = mat66f(0.0)
    first = friction_body_first[friction]
    second = friction_body_second[friction]
    if first >= 0:
        multiplicity = wp.max(1, body_constraint_count[first] - static_body_constraint_count[first])
        inverse_weight_first = wp.float32(multiplicity) * inverse_weight[first]
    if second >= 0:
        multiplicity = wp.max(1, body_constraint_count[second] - static_body_constraint_count[second])
        inverse_weight_second = wp.float32(multiplicity) * inverse_weight[second]
    value = compute_limit_delassus(
        friction_jacobian_first[friction],
        inverse_weight_first,
        friction_jacobian_second[friction],
        inverse_weight_second,
    )
    delassus[friction] = value
    if not wp.isfinite(value) or value <= 0.0:
        world_status[world] = PROJECTION_STATUS_INVALID


@wp.kernel
def _initialize_jacobi_projection_status(
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
):
    world = wp.tid()
    if world_active[world]:
        world_status[world] = prepared_status[world]


@wp.kernel
def _warmstart_contacts_jacobi(
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    inverse_weight: wp.array[mat66f],
    apply_inverse_weight: wp.bool,
    reaction: wp.array[wp.vec3f],
    twist_delta: wp.array[vec6f],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_local[contact] >= world_contact_count[world]
        or not world_active[world]
        or prepared_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    first = contact_body_first[contact]
    second = contact_body_second[contact]
    impulse = reaction[contact]
    if not _is_zero_vec3(impulse):
        if first >= 0:
            wrench = wp.transpose(contact_jacobian_first[contact]) @ impulse
            if apply_inverse_weight:
                wrench = inverse_weight[first] @ wrench
            _atomic_add_twist(twist_delta, first, wrench)
        if second >= 0:
            wrench = wp.transpose(contact_jacobian_second[contact]) @ impulse
            if apply_inverse_weight:
                wrench = inverse_weight[second] @ wrench
            _atomic_add_twist(twist_delta, second, wrench)


@wp.kernel
def _warmstart_limits_jacobi(
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    inverse_weight: wp.array[mat66f],
    apply_inverse_weight: wp.bool,
    reaction: wp.array[wp.float32],
    twist_delta: wp.array[vec6f],
):
    limit = wp.tid()
    world = limit_world[limit]
    if (
        limit_local[limit] >= world_limit_count[world]
        or not world_active[world]
        or prepared_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    first = limit_body_first[limit]
    second = limit_body_second[limit]
    impulse = reaction[limit]
    if first >= 0:
        wrench = limit_jacobian_first[limit] * impulse
        if apply_inverse_weight:
            wrench = inverse_weight[first] @ wrench
        _atomic_add_twist(twist_delta, first, wrench)
    if second >= 0:
        wrench = limit_jacobian_second[limit] * impulse
        if apply_inverse_weight:
            wrench = inverse_weight[second] @ wrench
        _atomic_add_twist(twist_delta, second, wrench)


@wp.kernel
def _warmstart_frictions_jacobi(
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    inverse_weight: wp.array[mat66f],
    apply_inverse_weight: wp.bool,
    reaction: wp.array[wp.float32],
    twist_delta: wp.array[vec6f],
):
    friction = wp.tid()
    world = friction_world[friction]
    if (
        friction_local[friction] >= world_friction_count[world]
        or not world_active[world]
        or prepared_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    first = friction_body_first[friction]
    second = friction_body_second[friction]
    impulse = reaction[friction]
    if first >= 0:
        wrench = friction_jacobian_first[friction] * impulse
        if apply_inverse_weight:
            wrench = inverse_weight[first] @ wrench
        _atomic_add_twist(twist_delta, first, wrench)
    if second >= 0:
        wrench = friction_jacobian_second[friction] * impulse
        if apply_inverse_weight:
            wrench = inverse_weight[second] @ wrench
        _atomic_add_twist(twist_delta, second, wrench)


@cache
def _make_project_contacts_kernel(colored: bool):
    """Specialize local-frame Coulomb projection by indexing."""

    @wp.kernel(module="unique")
    def project_contacts_kernel(
        launch_dim: wp.int32,
        target_color: wp.int32,
        index: _ProjectionIndexData,
        contact_data: Any,
        state: Any,
    ):
        lane = wp.tid()
        begin = lane
        end = lane + 1
        stride = 1
        if wp.static(colored):
            begin = index.color_offsets[target_color] + lane
            end = index.color_offsets[target_color] + index.color_counts[target_color]
            stride = launch_dim
        for ordered in range(begin, end, stride):
            contact = ordered
            if wp.static(colored):
                contact = index.order[ordered]
            world = index.constraint_world[contact]
            if wp.static(not colored):
                if index.constraint_local[contact] >= index.world_constraint_count[world]:
                    continue
            if not state.world_active[world] or state.world_status[world] != PROJECTION_STATUS_VALID:
                continue
            if wp.static(colored):
                _project_rigid_contact_colored(contact, world, target_color, contact_data, state)
                continue
            first = contact_data.body_first[contact]
            second = contact_data.body_second[contact]
            if first < 0 and second < 0:
                contact_data.reaction[contact] = wp.vec3f(0.0)
                continue
            velocity = _compute_contact_velocity(
                contact,
                first,
                second,
                contact_data.jacobian_first,
                contact_data.jacobian_second,
                contact_data.bias,
                state.projected_twist,
            )
            reaction_old = contact_data.reaction[contact]
            if _is_zero_vec3(reaction_old) and velocity[2] >= 0.0:
                continue
            projection = project_contact_coulomb_local(
                velocity,
                reaction_old,
                contact_data.delassus[contact],
                contact_data.friction[contact],
            )
            if projection.status == PROJECTION_STATUS_INVALID:
                state.world_status[world] = PROJECTION_STATUS_INVALID
                continue
            if _is_zero_vec3(projection.reaction_delta):
                contact_data.reaction[contact] = projection.reaction
                continue
            correction_first = vec6f(0.0)
            correction_second = vec6f(0.0)
            if first >= 0:
                correction_first = wp.transpose(contact_data.jacobian_first[contact]) @ projection.reaction_delta
            if second >= 0:
                correction_second = wp.transpose(contact_data.jacobian_second[contact]) @ projection.reaction_delta
            if not _is_finite_twist(correction_first) or not _is_finite_twist(correction_second):
                state.world_status[world] = PROJECTION_STATUS_INVALID
                continue
            _atomic_add_twist(state.twist_delta, first, correction_first)
            _atomic_add_twist(state.twist_delta, second, correction_second)
            contact_data.reaction[contact] = projection.reaction

    return project_contacts_kernel


@cache
def _make_project_scalar_kernel(colored: bool, limit: bool):
    """Specialize scalar unilateral projection by indexing and projection law."""

    @wp.kernel(module="unique")
    def project_scalar_kernel(
        launch_dim: wp.int32,
        target_color: wp.int32,
        index: _ProjectionIndexData,
        body_first: wp.array[wp.int32],
        body_second: wp.array[wp.int32],
        jacobian_first: wp.array[vec6f],
        jacobian_second: wp.array[vec6f],
        bias: wp.array[wp.float32],
        impulse_bound: wp.array[wp.float32],
        delassus: wp.array[wp.float32],
        reaction: wp.array[wp.float32],
        state: _RigidProjectionState,
    ):
        lane = wp.tid()
        begin = lane
        end = lane + 1
        stride = 1
        if wp.static(colored):
            begin = index.color_offsets[target_color] + lane
            end = index.color_offsets[target_color] + index.color_counts[target_color]
            stride = launch_dim

        for ordered in range(begin, end, stride):
            constraint = ordered
            if wp.static(colored):
                constraint = index.order[ordered]
            world = index.constraint_world[constraint]
            if wp.static(not colored):
                if index.constraint_local[constraint] >= index.world_constraint_count[world]:
                    continue
            if not state.world_active[world] or state.world_status[world] != PROJECTION_STATUS_VALID:
                continue
            if wp.static(colored):
                _project_rigid_scalar_colored(
                    constraint,
                    world,
                    target_color,
                    body_first,
                    body_second,
                    jacobian_first,
                    jacobian_second,
                    bias,
                    impulse_bound,
                    delassus,
                    reaction,
                    state,
                    wp.static(limit),
                )
                continue

            first = body_first[constraint]
            second = body_second[constraint]
            if wp.static(limit):
                if first < 0 and second < 0:
                    reaction[constraint] = 0.0
                    continue
            current_velocity = wp.float32(0.0)
            if wp.static(limit):
                current_velocity = bias[constraint]
            if first >= 0:
                current_velocity += wp.dot(jacobian_first[constraint], state.projected_twist[first])
            if second >= 0:
                current_velocity += wp.dot(jacobian_second[constraint], state.projected_twist[second])
            if wp.static(limit):
                projection = project_limit_local(current_velocity, reaction[constraint], delassus[constraint])
            else:
                projection = project_friction_local(
                    current_velocity,
                    reaction[constraint],
                    delassus[constraint],
                    impulse_bound[constraint],
                )
            if projection.status == PROJECTION_STATUS_INVALID:
                state.world_status[world] = PROJECTION_STATUS_INVALID
                continue

            correction_first = vec6f(0.0)
            correction_second = vec6f(0.0)
            if first >= 0:
                correction_first = jacobian_first[constraint] * projection.reaction_delta
            if second >= 0:
                correction_second = jacobian_second[constraint] * projection.reaction_delta
            if not _is_finite_twist(correction_first) or not _is_finite_twist(correction_second):
                state.world_status[world] = PROJECTION_STATUS_INVALID
                continue

            reaction[constraint] = projection.reaction
            _atomic_add_twist(state.twist_delta, first, correction_first)
            _atomic_add_twist(state.twist_delta, second, correction_second)

    return project_scalar_kernel


@wp.kernel
def _project_rigid_constraints_jacobi_by_world(
    projection_iterations: wp.int32,
    warm_start: wp.bool,
    world_active: wp.array[wp.bool],
    body_offset: wp.array[wp.int32],
    body_count: wp.array[wp.int32],
    world_friction_offset: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_impulse_bound: wp.array[wp.float32],
    friction_delassus: wp.array[wp.float32],
    world_contact_offset: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_delassus: wp.array[wp.mat33f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    world_limit_offset: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    limit_delassus: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    friction_reaction: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    twist_delta: wp.array[vec6f],
    world_status: wp.array[wp.int32],
):
    world, lane = wp.tid()
    if not world_active[world] or world_status[world] != PROJECTION_STATUS_VALID:
        return

    thread_count = wp.block_dim()
    local = lane
    while local < body_count[world]:
        twist_delta[body_offset[world] + local] = vec6f(0.0)
        local += thread_count
    _sync_threads()

    if warm_start:
        local = lane
        while local < world_friction_count[world]:
            friction = world_friction_offset[world] + local
            friction_first = friction_body_first[friction]
            friction_second = friction_body_second[friction]
            friction_impulse = friction_reaction[friction]
            if friction_first >= 0:
                _atomic_add_twist(
                    twist_delta,
                    friction_first,
                    friction_jacobian_first[friction] * friction_impulse,
                )
            if friction_second >= 0:
                _atomic_add_twist(
                    twist_delta,
                    friction_second,
                    friction_jacobian_second[friction] * friction_impulse,
                )
            local += thread_count

        local = lane
        while local < world_contact_count[world]:
            contact = world_contact_offset[world] + local
            contact_first = contact_body_first[contact]
            contact_second = contact_body_second[contact]
            contact_impulse = contact_reaction[contact]
            if contact_first >= 0:
                _atomic_add_twist(
                    twist_delta,
                    contact_first,
                    wp.transpose(contact_jacobian_first[contact]) @ contact_impulse,
                )
            if contact_second >= 0:
                _atomic_add_twist(
                    twist_delta,
                    contact_second,
                    wp.transpose(contact_jacobian_second[contact]) @ contact_impulse,
                )
            local += thread_count

        local = lane
        while local < world_limit_count[world]:
            limit = world_limit_offset[world] + local
            limit_first = limit_body_first[limit]
            limit_second = limit_body_second[limit]
            limit_impulse = limit_reaction[limit]
            if limit_first >= 0:
                _atomic_add_twist(
                    twist_delta,
                    limit_first,
                    limit_jacobian_first[limit] * limit_impulse,
                )
            if limit_second >= 0:
                _atomic_add_twist(
                    twist_delta,
                    limit_second,
                    limit_jacobian_second[limit] * limit_impulse,
                )
            local += thread_count

        _sync_threads()
        local = lane
        while local < body_count[world]:
            body = body_offset[world] + local
            warmstart_correction = inverse_weight[body] @ twist_delta[body]
            if _is_finite_twist(warmstart_correction):
                projected_twist[body] += warmstart_correction
            else:
                world_status[world] = PROJECTION_STATUS_INVALID
            twist_delta[body] = vec6f(0.0)
            local += thread_count
        _sync_threads()
        if world_status[world] != PROJECTION_STATUS_VALID:
            return

    for _sweep in range(projection_iterations):
        local = lane
        while local < world_friction_count[world]:
            friction = world_friction_offset[world] + local
            first = friction_body_first[friction]
            second = friction_body_second[friction]
            current_velocity = wp.float32(0.0)
            if first >= 0:
                current_velocity += wp.dot(friction_jacobian_first[friction], projected_twist[first])
            if second >= 0:
                current_velocity += wp.dot(friction_jacobian_second[friction], projected_twist[second])
            reaction_old = friction_reaction[friction]
            friction_projection = project_friction_local(
                current_velocity,
                reaction_old,
                friction_delassus[friction],
                friction_impulse_bound[friction],
            )
            if friction_projection.status == PROJECTION_STATUS_VALID:
                correction_first = vec6f(0.0)
                correction_second = vec6f(0.0)
                if first >= 0:
                    correction_first = friction_jacobian_first[friction] * friction_projection.reaction_delta
                if second >= 0:
                    correction_second = friction_jacobian_second[friction] * friction_projection.reaction_delta
                if _is_finite_twist(correction_first) and _is_finite_twist(correction_second):
                    friction_reaction[friction] = friction_projection.reaction
                    _atomic_add_twist(twist_delta, first, correction_first)
                    _atomic_add_twist(twist_delta, second, correction_second)
                else:
                    world_status[world] = PROJECTION_STATUS_INVALID
            else:
                world_status[world] = PROJECTION_STATUS_INVALID
            local += thread_count

        local = lane
        while local < world_contact_count[world]:
            contact = world_contact_offset[world] + local
            first = contact_body_first[contact]
            second = contact_body_second[contact]
            if first < 0 and second < 0:
                contact_reaction[contact] = wp.vec3f(0.0)
            else:
                twist_first = vec6f(0.0)
                twist_second = vec6f(0.0)
                if first >= 0:
                    twist_first = projected_twist[first]
                if second >= 0:
                    twist_second = projected_twist[second]
                contact_reaction_old = contact_reaction[contact]
                contact_velocity_value = (
                    contact_jacobian_first[contact] @ twist_first
                    + contact_jacobian_second[contact] @ twist_second
                    + contact_bias[contact]
                )
                contact_projection = project_contact_coulomb_local(
                    contact_velocity_value,
                    contact_reaction_old,
                    contact_delassus[contact],
                    contact_friction[contact],
                )
                if contact_projection.status == PROJECTION_STATUS_VALID:
                    contact_correction_first = vec6f(0.0)
                    contact_correction_second = vec6f(0.0)
                    if first >= 0:
                        contact_correction_first = (
                            wp.transpose(contact_jacobian_first[contact]) @ contact_projection.reaction_delta
                        )
                    if second >= 0:
                        contact_correction_second = (
                            wp.transpose(contact_jacobian_second[contact]) @ contact_projection.reaction_delta
                        )
                    if _is_finite_twist(contact_correction_first) and _is_finite_twist(contact_correction_second):
                        contact_reaction[contact] = contact_projection.reaction
                        _atomic_add_twist(twist_delta, first, contact_correction_first)
                        _atomic_add_twist(twist_delta, second, contact_correction_second)
                    else:
                        world_status[world] = PROJECTION_STATUS_INVALID
                else:
                    world_status[world] = PROJECTION_STATUS_INVALID
            local += thread_count

        local = lane
        while local < world_limit_count[world]:
            limit = world_limit_offset[world] + local
            first = limit_body_first[limit]
            second = limit_body_second[limit]
            if first < 0 and second < 0:
                limit_reaction[limit] = 0.0
            else:
                current_velocity = limit_bias[limit]
                if first >= 0:
                    current_velocity += wp.dot(limit_jacobian_first[limit], projected_twist[first])
                if second >= 0:
                    current_velocity += wp.dot(limit_jacobian_second[limit], projected_twist[second])
                reaction_old = limit_reaction[limit]
                limit_projection = project_limit_local(current_velocity, reaction_old, limit_delassus[limit])
                if limit_projection.status == PROJECTION_STATUS_VALID:
                    correction_first = vec6f(0.0)
                    correction_second = vec6f(0.0)
                    if first >= 0:
                        correction_first = limit_jacobian_first[limit] * limit_projection.reaction_delta
                    if second >= 0:
                        correction_second = limit_jacobian_second[limit] * limit_projection.reaction_delta
                    if _is_finite_twist(correction_first) and _is_finite_twist(correction_second):
                        limit_reaction[limit] = limit_projection.reaction
                        _atomic_add_twist(twist_delta, first, correction_first)
                        _atomic_add_twist(twist_delta, second, correction_second)
                    else:
                        world_status[world] = PROJECTION_STATUS_INVALID
                else:
                    world_status[world] = PROJECTION_STATUS_INVALID
            local += thread_count

        _sync_threads()
        local = lane
        while local < body_count[world]:
            body = body_offset[world] + local
            if world_status[world] == PROJECTION_STATUS_VALID:
                correction = inverse_weight[body] @ twist_delta[body]
                if _is_finite_twist(correction):
                    projected_twist[body] += correction
                else:
                    world_status[world] = PROJECTION_STATUS_INVALID
            twist_delta[body] = vec6f(0.0)
            local += thread_count
        _sync_threads()
        if world_status[world] != PROJECTION_STATUS_VALID:
            return


@wp.kernel
def _apply_jacobi_twist_delta(
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    twist_delta: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
):
    body = wp.tid()
    world = body_world[body]
    if world_active[world] and world_status[world] == PROJECTION_STATUS_VALID:
        correction = inverse_weight[body] @ twist_delta[body]
        if _is_finite_twist(correction):
            projected_twist[body] += correction
        else:
            world_status[world] = PROJECTION_STATUS_INVALID
    twist_delta[body] = vec6f(0.0)


@wp.kernel
def _apply_jacobi_warmstart(
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    twist_delta: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
):
    body = wp.tid()
    world = body_world[body]
    if world_active[world] and world_status[world] == PROJECTION_STATUS_VALID:
        correction = inverse_weight[body] @ twist_delta[body]
        if _is_finite_twist(correction):
            projected_twist[body] += correction
        else:
            world_status[world] = PROJECTION_STATUS_INVALID
    twist_delta[body] = vec6f(0.0)


@wp.kernel
def _initialize_projection_residuals(
    world_contact_residual_max: wp.array[wp.float32],
    world_limit_residual_max: wp.array[wp.float32],
    world_friction_residual_max: wp.array[wp.float32],
):
    world = wp.tid()
    world_contact_residual_max[world] = 0.0
    world_limit_residual_max[world] = 0.0
    world_friction_residual_max[world] = 0.0


@wp.kernel
def _compute_friction_projection_residuals(
    world_mask: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_impulse_bound: wp.array[wp.float32],
    friction_reaction: wp.array[wp.float32],
    friction_delassus: wp.array[wp.float32],
    projected_twist: wp.array[vec6f],
    friction_velocity: wp.array[wp.float32],
    friction_residual: wp.array[wp.float32],
    world_friction_residual_max: wp.array[wp.float32],
):
    friction = wp.tid()
    world = friction_world[friction]
    if (
        friction_local[friction] >= world_friction_count[world]
        or not world_mask[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return

    first = friction_body_first[friction]
    second = friction_body_second[friction]
    value = wp.float32(0.0)
    if first >= 0:
        value += wp.dot(friction_jacobian_first[friction], projected_twist[first])
    if second >= 0:
        value += wp.dot(friction_jacobian_second[friction], projected_twist[second])
    friction_velocity[friction] = value
    delassus = friction_delassus[friction]
    scale = wp.sqrt(delassus)
    scaled_reaction = scale * friction_reaction[friction]
    scaled_velocity = value / scale
    scaled_bound = scale * friction_impulse_bound[friction]
    residual = wp.abs(scaled_reaction - wp.clamp(scaled_reaction - scaled_velocity, -scaled_bound, scaled_bound))
    friction_residual[friction] = residual
    wp.atomic_max(world_friction_residual_max, world, residual)


@wp.kernel
def _compute_contact_projection_residuals(
    world_mask: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_delassus: wp.array[wp.mat33f],
    projected_twist: wp.array[vec6f],
    contact_velocity: wp.array[wp.vec3f],
    contact_residual: wp.array[wp.float32],
    world_contact_residual_max: wp.array[wp.float32],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_local[contact] >= world_contact_count[world]
        or not world_mask[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return

    first = contact_body_first[contact]
    second = contact_body_second[contact]
    if first < 0 and second < 0:
        contact_velocity[contact] = wp.vec3f(0.0)
        contact_residual[contact] = 0.0
        return
    contact_velocity_final = wp.vec3f(0.0)
    if first >= 0:
        contact_velocity_final += contact_jacobian_first[contact] @ projected_twist[first]
    if second >= 0:
        contact_velocity_final += contact_jacobian_second[contact] @ projected_twist[second]
    contact_velocity[contact] = contact_velocity_final
    contact_residual_vector = compute_contact_scaled_alart_curnier_residual(
        contact_delassus[contact],
        contact_reaction[contact],
        contact_velocity_final + contact_bias[contact],
        contact_friction[contact],
    )
    residual_max = wp.max(wp.abs(contact_residual_vector[0]), wp.abs(contact_residual_vector[1]))
    residual_max = wp.max(residual_max, wp.abs(contact_residual_vector[2]))
    contact_residual[contact] = residual_max
    wp.atomic_max(world_contact_residual_max, world, residual_max)


@wp.kernel
def _compute_limit_projection_residuals(
    world_mask: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    limit_reaction: wp.array[wp.float32],
    limit_delassus: wp.array[wp.float32],
    projected_twist: wp.array[vec6f],
    limit_velocity: wp.array[wp.float32],
    limit_residual: wp.array[wp.float32],
    world_limit_residual_max: wp.array[wp.float32],
):
    limit = wp.tid()
    world = limit_world[limit]
    if (
        limit_local[limit] >= world_limit_count[world]
        or not world_mask[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return

    first = limit_body_first[limit]
    second = limit_body_second[limit]
    if first < 0 and second < 0:
        limit_velocity[limit] = 0.0
        limit_residual[limit] = 0.0
        return
    limit_velocity_final = wp.float32(0.0)
    if first >= 0:
        limit_velocity_final += wp.dot(limit_jacobian_first[limit], projected_twist[first])
    if second >= 0:
        limit_velocity_final += wp.dot(limit_jacobian_second[limit], projected_twist[second])
    limit_velocity[limit] = limit_velocity_final
    scale = wp.sqrt(limit_delassus[limit])
    scaled_reaction = scale * limit_reaction[limit]
    scaled_velocity = (limit_velocity_final + limit_bias[limit]) / scale
    limit_residual_value = wp.abs(scaled_reaction - wp.max(scaled_reaction - scaled_velocity, 0.0))
    limit_residual[limit] = limit_residual_value
    wp.atomic_max(world_limit_residual_max, world, limit_residual_value)


def compute_projection_residuals(
    world_mask: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_impulse_bound: wp.array[wp.float32],
    friction_reaction: wp.array[wp.float32],
    friction_delassus: wp.array[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_delassus: wp.array[wp.mat33f],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    limit_reaction: wp.array[wp.float32],
    limit_delassus: wp.array[wp.float32],
    projected_twist: wp.array[vec6f],
    friction_velocity: wp.array[wp.float32],
    contact_velocity: wp.array[wp.vec3f],
    limit_velocity: wp.array[wp.float32],
    contact_residual: wp.array[wp.float32],
    limit_residual: wp.array[wp.float32],
    friction_residual: wp.array[wp.float32],
    world_contact_residual_max: wp.array[wp.float32],
    world_limit_residual_max: wp.array[wp.float32],
    world_friction_residual_max: wp.array[wp.float32],
) -> None:
    """Recompute final unilateral velocities and per-world natural-map residual maxima."""
    world_count = world_mask.shape[0]
    if (
        projection_status.shape[0] != world_count
        or world_friction_count.shape[0] != world_count
        or world_contact_count.shape[0] != world_count
        or world_limit_count.shape[0] != world_count
        or world_contact_residual_max.shape[0] != world_count
        or world_limit_residual_max.shape[0] != world_count
        or world_friction_residual_max.shape[0] != world_count
    ):
        raise ValueError("Projection diagnostic world arrays must have identical lengths.")
    if (
        friction_world.shape[0] != friction_local.shape[0]
        or contact_world.shape[0] != contact_local.shape[0]
        or limit_world.shape[0] != limit_local.shape[0]
    ):
        raise ValueError("Projection diagnostic world and local arrays must have identical lengths.")
    wp.launch(
        _initialize_projection_residuals,
        dim=world_count,
        inputs=[],
        outputs=[
            world_contact_residual_max,
            world_limit_residual_max,
            world_friction_residual_max,
        ],
        device=projected_twist.device,
    )
    if friction_world.shape[0] > 0:
        wp.launch(
            _compute_friction_projection_residuals,
            dim=friction_world.shape[0],
            inputs=[
                world_mask,
                projection_status,
                friction_world,
                friction_local,
                world_friction_count,
                friction_body_first,
                friction_body_second,
                friction_jacobian_first,
                friction_jacobian_second,
                friction_impulse_bound,
                friction_reaction,
                friction_delassus,
                projected_twist,
            ],
            outputs=[friction_velocity, friction_residual, world_friction_residual_max],
            device=projected_twist.device,
        )
    if contact_world.shape[0] > 0:
        wp.launch(
            _compute_contact_projection_residuals,
            dim=contact_world.shape[0],
            inputs=[
                world_mask,
                projection_status,
                contact_world,
                contact_local,
                world_contact_count,
                contact_body_first,
                contact_body_second,
                contact_jacobian_first,
                contact_jacobian_second,
                contact_bias,
                contact_friction,
                contact_reaction,
                contact_delassus,
                projected_twist,
            ],
            outputs=[contact_velocity, contact_residual, world_contact_residual_max],
            device=projected_twist.device,
        )
    if limit_world.shape[0] > 0:
        wp.launch(
            _compute_limit_projection_residuals,
            dim=limit_world.shape[0],
            inputs=[
                world_mask,
                projection_status,
                limit_world,
                limit_local,
                world_limit_count,
                limit_body_first,
                limit_body_second,
                limit_jacobian_first,
                limit_jacobian_second,
                limit_bias,
                limit_reaction,
                limit_delassus,
                projected_twist,
            ],
            outputs=[limit_velocity, limit_residual, world_limit_residual_max],
            device=projected_twist.device,
        )


def prepare_physical_projection_data(
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    inverse_weight: wp.array[mat66f],
    friction_delassus: wp.array[wp.float32],
    contact_physical_delassus: wp.array[wp.mat33f],
    contact_prepared_delassus: wp.array[wp.mat33f],
    limit_delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
) -> None:
    """Cache physical local Delassus blocks once for fixed body-space data."""
    world_count = world_status.shape[0]
    if (
        world_friction_count.shape[0] != world_count
        or world_contact_count.shape[0] != world_count
        or world_limit_count.shape[0] != world_count
    ):
        raise ValueError("Friction, contact, limit, and status world arrays must have identical lengths.")
    world_status.fill_(PROJECTION_STATUS_VALID)
    for world, local, count, first, second, jacobian_first, jacobian_second, output in (
        (
            friction_world,
            friction_local,
            world_friction_count,
            friction_body_first,
            friction_body_second,
            friction_jacobian_first,
            friction_jacobian_second,
            friction_delassus,
        ),
        (
            limit_world,
            limit_local,
            world_limit_count,
            limit_body_first,
            limit_body_second,
            limit_jacobian_first,
            limit_jacobian_second,
            limit_delassus,
        ),
    ):
        if world.shape[0] > 0:
            wp.launch(
                _prepare_scalar_projection_data,
                dim=world.shape[0],
                inputs=[world, local, count, first, second, jacobian_first, jacobian_second, inverse_weight],
                outputs=[output, world_status],
                device=inverse_weight.device,
            )
    if contact_world.shape[0] > 0:
        wp.launch(
            _prepare_contact_physical_projection_data,
            dim=contact_world.shape[0],
            inputs=[
                contact_world,
                contact_local,
                world_contact_count,
                contact_body_first,
                contact_body_second,
                contact_jacobian_first,
                contact_jacobian_second,
                contact_bias,
                contact_friction,
                inverse_weight,
            ],
            outputs=[contact_physical_delassus, contact_prepared_delassus, world_status],
            device=inverse_weight.device,
        )


def prepare_jacobi_projection_data(
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    body_constraint_count: wp.array[wp.int32],
    static_body_constraint_count: wp.array[wp.int32],
    inverse_weight: wp.array[mat66f],
    friction_delassus: wp.array[wp.float32],
    contact_delassus: wp.array[wp.mat33f],
    limit_delassus: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
) -> None:
    """Prepare mass-split body-space blocks for true Jacobi projection."""
    world_count = world_status.shape[0]
    if (
        world_friction_count.shape[0] != world_count
        or world_contact_count.shape[0] != world_count
        or world_limit_count.shape[0] != world_count
    ):
        raise ValueError("Friction, contact, limit, and status world arrays must have identical lengths.")
    if body_constraint_count.shape[0] != static_body_constraint_count.shape[0]:
        raise ValueError("Body incidence arrays must have identical lengths.")
    world_status.fill_(PROJECTION_STATUS_VALID)
    if friction_world.shape[0] > 0:
        wp.launch(
            _prepare_frictions_jacobi,
            dim=friction_world.shape[0],
            inputs=[
                friction_world,
                friction_local,
                world_friction_count,
                friction_body_first,
                friction_body_second,
                friction_jacobian_first,
                friction_jacobian_second,
                body_constraint_count,
                static_body_constraint_count,
                inverse_weight,
            ],
            outputs=[friction_delassus, world_status],
            device=inverse_weight.device,
        )
    if contact_world.shape[0] > 0:
        wp.launch(
            _prepare_contacts_jacobi,
            dim=contact_world.shape[0],
            inputs=[
                contact_world,
                contact_local,
                world_contact_count,
                contact_body_first,
                contact_body_second,
                contact_jacobian_first,
                contact_jacobian_second,
                contact_bias,
                contact_friction,
                body_constraint_count,
                static_body_constraint_count,
                inverse_weight,
            ],
            outputs=[contact_delassus, world_status],
            device=inverse_weight.device,
        )
    if limit_world.shape[0] > 0:
        wp.launch(
            _prepare_limits_jacobi,
            dim=limit_world.shape[0],
            inputs=[
                limit_world,
                limit_local,
                world_limit_count,
                limit_body_first,
                limit_body_second,
                limit_jacobian_first,
                limit_jacobian_second,
                body_constraint_count,
                static_body_constraint_count,
                inverse_weight,
            ],
            outputs=[limit_delassus, world_status],
            device=inverse_weight.device,
        )


@wp.kernel
def _initialize_acceleration_worlds(
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    theta: wp.array[wp.float32],
    beta: wp.array[wp.float32],
    restart_dot: wp.array[wp.float32],
    world_status: wp.array[wp.int32],
):
    world = wp.tid()
    if world_active[world]:
        theta[world] = 1.0
        beta[world] = 0.0
        restart_dot[world] = 0.0
        world_status[world] = prepared_status[world]


@wp.kernel
def _initialize_accelerated_reactions(
    friction_capacity: int,
    contact_capacity: int,
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_bound: wp.array[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_friction: wp.array[wp.float32],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    friction_reaction: wp.array[wp.float32],
    friction_trial: wp.array[wp.float32],
    friction_previous: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    limit_trial: wp.array[wp.float32],
    limit_previous: wp.array[wp.float32],
):
    constraint = wp.tid()
    if constraint < friction_capacity:
        world = friction_world[constraint]
        if friction_local[constraint] >= world_friction_count[world] or not world_active[world]:
            return
        friction_value = wp.clamp(
            friction_reaction[constraint], -friction_bound[constraint], friction_bound[constraint]
        )
        friction_reaction[constraint] = friction_value
        friction_trial[constraint] = friction_value
        friction_previous[constraint] = friction_value
        return

    constraint -= friction_capacity
    if constraint < contact_capacity:
        world = contact_world[constraint]
        if contact_local[constraint] >= world_contact_count[world] or not world_active[world]:
            return
        contact_value = wp.vec3f(0.0)
        if contact_body_first[constraint] >= 0 or contact_body_second[constraint] >= 0:
            contact_value = project_contact_coulomb_cone(contact_reaction[constraint], contact_friction[constraint])
        contact_reaction[constraint] = contact_value
        contact_trial[constraint] = contact_value
        contact_previous[constraint] = contact_value
        return

    constraint -= contact_capacity
    world = limit_world[constraint]
    if limit_local[constraint] >= world_limit_count[world] or not world_active[world]:
        return
    limit_value = wp.max(0.0, limit_reaction[constraint])
    limit_reaction[constraint] = limit_value
    limit_trial[constraint] = limit_value
    limit_previous[constraint] = limit_value


@wp.kernel
def _accumulate_rigid_restart(
    friction_capacity: int,
    contact_capacity: int,
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    friction_reaction: wp.array[wp.float32],
    friction_trial: wp.array[wp.float32],
    friction_previous: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    limit_trial: wp.array[wp.float32],
    limit_previous: wp.array[wp.float32],
    restart_dot: wp.array[wp.float32],
):
    constraint = wp.tid()
    if constraint < friction_capacity:
        world = friction_world[constraint]
        if (
            friction_local[constraint] < world_friction_count[world]
            and world_active[world]
            and world_status[world] == PROJECTION_STATUS_VALID
        ):
            friction_current = friction_reaction[constraint]
            wp.atomic_add(
                restart_dot,
                world,
                (friction_current - friction_trial[constraint]) * (friction_current - friction_previous[constraint]),
            )
        return
    constraint -= friction_capacity
    if constraint < contact_capacity:
        world = contact_world[constraint]
        if (
            contact_local[constraint] < world_contact_count[world]
            and world_active[world]
            and world_status[world] == PROJECTION_STATUS_VALID
        ):
            contact_current = contact_reaction[constraint]
            wp.atomic_add(
                restart_dot,
                world,
                wp.dot(
                    contact_current - contact_trial[constraint],
                    contact_current - contact_previous[constraint],
                ),
            )
        return
    constraint -= contact_capacity
    world = limit_world[constraint]
    if (
        limit_local[constraint] < world_limit_count[world]
        and world_active[world]
        and world_status[world] == PROJECTION_STATUS_VALID
    ):
        limit_current = limit_reaction[constraint]
        wp.atomic_add(
            restart_dot,
            world,
            (limit_current - limit_trial[constraint]) * (limit_current - limit_previous[constraint]),
        )


@wp.kernel
def _finalize_acceleration(
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    restart_dot: wp.array[wp.float32],
    theta: wp.array[wp.float32],
    beta: wp.array[wp.float32],
):
    world = wp.tid()
    if not world_active[world]:
        return
    value = restart_dot[world]
    restart_dot[world] = 0.0
    current = theta[world]
    if (
        world_status[world] != PROJECTION_STATUS_VALID
        or not wp.isfinite(value)
        or value <= 0.0
        or not wp.isfinite(current)
        or current <= 0.0
    ):
        theta[world] = 1.0
        beta[world] = 0.0
        return
    next_theta = 2.0 * current / (wp.sqrt(current * current + 4.0) + current)
    beta[world] = current * (1.0 - current) / (current * current + next_theta)
    theta[world] = next_theta


@wp.kernel
def _extrapolate_rigid_reactions(
    friction_capacity: int,
    contact_capacity: int,
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    world_active: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    beta: wp.array[wp.float32],
    friction_reaction: wp.array[wp.float32],
    friction_trial: wp.array[wp.float32],
    friction_previous: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    limit_trial: wp.array[wp.float32],
    limit_previous: wp.array[wp.float32],
    twist_delta: wp.array[vec6f],
):
    constraint = wp.tid()
    if constraint < friction_capacity:
        world = friction_world[constraint]
        if (
            friction_local[constraint] >= world_friction_count[world]
            or not world_active[world]
            or world_status[world] != PROJECTION_STATUS_VALID
        ):
            return
        friction_current = friction_reaction[constraint]
        friction_extrapolated = friction_current + beta[world] * (friction_current - friction_previous[constraint])
        friction_delta = friction_extrapolated - friction_current
        friction_previous[constraint] = friction_current
        friction_trial[constraint] = friction_extrapolated
        friction_reaction[constraint] = friction_extrapolated
        first = friction_body_first[constraint]
        second = friction_body_second[constraint]
        _atomic_add_twist(twist_delta, first, friction_jacobian_first[constraint] * friction_delta)
        _atomic_add_twist(twist_delta, second, friction_jacobian_second[constraint] * friction_delta)
        return

    constraint -= friction_capacity
    if constraint < contact_capacity:
        world = contact_world[constraint]
        if (
            contact_local[constraint] >= world_contact_count[world]
            or not world_active[world]
            or world_status[world] != PROJECTION_STATUS_VALID
        ):
            return
        contact_current = contact_reaction[constraint]
        contact_extrapolated = contact_current + beta[world] * (contact_current - contact_previous[constraint])
        contact_delta = contact_extrapolated - contact_current
        contact_previous[constraint] = contact_current
        contact_trial[constraint] = contact_extrapolated
        contact_reaction[constraint] = contact_extrapolated
        first = contact_body_first[constraint]
        second = contact_body_second[constraint]
        if first >= 0:
            _atomic_add_twist(twist_delta, first, wp.transpose(contact_jacobian_first[constraint]) @ contact_delta)
        if second >= 0:
            _atomic_add_twist(twist_delta, second, wp.transpose(contact_jacobian_second[constraint]) @ contact_delta)
        return

    constraint -= contact_capacity
    world = limit_world[constraint]
    if (
        limit_local[constraint] >= world_limit_count[world]
        or not world_active[world]
        or world_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    limit_current = limit_reaction[constraint]
    limit_extrapolated = limit_current + beta[world] * (limit_current - limit_previous[constraint])
    limit_delta = limit_extrapolated - limit_current
    limit_previous[constraint] = limit_current
    limit_trial[constraint] = limit_extrapolated
    limit_reaction[constraint] = limit_extrapolated
    first = limit_body_first[constraint]
    second = limit_body_second[constraint]
    _atomic_add_twist(twist_delta, first, limit_jacobian_first[constraint] * limit_delta)
    _atomic_add_twist(twist_delta, second, limit_jacobian_second[constraint] * limit_delta)


def _apply_jacobi_delta(
    body_world,
    world_active,
    world_status,
    inverse_weight,
    twist_delta,
    projected_twist,
) -> None:
    wp.launch(
        _apply_jacobi_twist_delta,
        dim=projected_twist.shape[0],
        inputs=[body_world, world_active, world_status, inverse_weight, twist_delta],
        outputs=[projected_twist],
        device=projected_twist.device,
    )


def _apply_jacobi_warmstart_delta(
    body_world,
    world_active,
    world_status,
    inverse_weight,
    twist_delta,
    projected_twist,
) -> None:
    wp.launch(
        _apply_jacobi_warmstart,
        dim=projected_twist.shape[0],
        inputs=[body_world, world_active, world_status, inverse_weight, twist_delta],
        outputs=[projected_twist],
        device=projected_twist.device,
    )


def project_constraints_jacobi(
    projection_iterations: int,
    world_active: wp.array[wp.bool],
    body_world: wp.array[wp.int32],
    friction_world: wp.array[wp.int32],
    friction_local: wp.array[wp.int32],
    world_friction_count: wp.array[wp.int32],
    friction_body_first: wp.array[wp.int32],
    friction_body_second: wp.array[wp.int32],
    friction_jacobian_first: wp.array[vec6f],
    friction_jacobian_second: wp.array[vec6f],
    friction_impulse_bound: wp.array[wp.float32],
    friction_delassus: wp.array[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_first: wp.array[wp.int32],
    contact_body_second: wp.array[wp.int32],
    contact_jacobian_first: wp.array[mat36f],
    contact_jacobian_second: wp.array[mat36f],
    contact_bias: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    contact_delassus: wp.array[wp.mat33f],
    limit_world: wp.array[wp.int32],
    limit_local: wp.array[wp.int32],
    world_limit_count: wp.array[wp.int32],
    limit_body_first: wp.array[wp.int32],
    limit_body_second: wp.array[wp.int32],
    limit_jacobian_first: wp.array[vec6f],
    limit_jacobian_second: wp.array[vec6f],
    limit_bias: wp.array[wp.float32],
    limit_delassus: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    projected_twist: wp.array[vec6f],
    twist_delta: wp.array[vec6f],
    contact_reaction: wp.array[wp.vec3f],
    limit_reaction: wp.array[wp.float32],
    friction_reaction: wp.array[wp.float32],
    prepared_status: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    warm_start: bool = True,
    world_body_offset: wp.array[wp.int32] | None = None,
    world_body_count: wp.array[wp.int32] | None = None,
    world_friction_offset: wp.array[wp.int32] | None = None,
    world_contact_offset: wp.array[wp.int32] | None = None,
    world_limit_offset: wp.array[wp.int32] | None = None,
    accelerated: bool = False,
    theta: wp.array[wp.float32] | None = None,
    beta: wp.array[wp.float32] | None = None,
    restart_dot: wp.array[wp.float32] | None = None,
    friction_trial: wp.array[wp.float32] | None = None,
    friction_previous: wp.array[wp.float32] | None = None,
    contact_trial: wp.array[wp.vec3f] | None = None,
    contact_previous: wp.array[wp.vec3f] | None = None,
    limit_trial: wp.array[wp.float32] | None = None,
    limit_previous: wp.array[wp.float32] | None = None,
) -> None:
    """Run mass-split Jacobi sweeps over all body-space unilaterals.

    Set ``warm_start`` to false when the projected state already contains the
    current reactions, such as for a final smoothing sweep after Gauss--Seidel.
    """
    if (
        not isinstance(projection_iterations, int)
        or isinstance(projection_iterations, bool)
        or projection_iterations < 1
    ):
        raise ValueError("projection_iterations must be an integer greater than or equal to one.")
    world_count = world_active.shape[0]
    if prepared_status.shape[0] != world_count or world_status.shape[0] != world_count:
        raise ValueError("Active, prepared-status, and status world arrays must have identical lengths.")
    if body_world.shape[0] != projected_twist.shape[0] or twist_delta.shape[0] != projected_twist.shape[0]:
        raise ValueError("Body world, projected twist, and Jacobi delta arrays must have identical lengths.")
    if not isinstance(warm_start, bool):
        raise ValueError("warm_start must be a boolean.")
    if not isinstance(accelerated, bool):
        raise ValueError("accelerated must be a boolean.")
    acceleration_arrays = (
        theta,
        beta,
        restart_dot,
        friction_trial,
        friction_previous,
        contact_trial,
        contact_previous,
        limit_trial,
        limit_previous,
    )
    if accelerated and (not warm_start or any(value is None for value in acceleration_arrays)):
        raise ValueError("Accelerated Jacobi requires warm start and all acceleration arrays.")

    contact_projection_max_blocks = 0
    if projected_twist.device.is_cuda:
        contact_projection_max_blocks = projected_twist.device.sm_count * _JACOBI_CONTACT_PROJECTION_BLOCKS_PER_SM

    use_world_projection = not accelerated and _can_fuse_rigid_projection_by_world(
        projected_twist.device,
        world_count,
        required_world_arrays=(
            world_body_offset,
            world_body_count,
            world_friction_offset,
            world_contact_offset,
            world_limit_offset,
        ),
        parallel_constraint_capacity=friction_world.shape[0] + contact_world.shape[0] + limit_world.shape[0],
        world_block_dim=_JACOBI_WORLD_BLOCK_DIM,
    )

    rigid_capacity = friction_world.shape[0] + contact_world.shape[0] + limit_world.shape[0]
    if accelerated:
        wp.launch(
            _initialize_acceleration_worlds,
            dim=world_count,
            inputs=[world_active, prepared_status],
            outputs=[theta, beta, restart_dot, world_status],
            device=projected_twist.device,
        )
        if rigid_capacity > 0:
            wp.launch(
                _initialize_accelerated_reactions,
                dim=rigid_capacity,
                inputs=[
                    friction_world.shape[0],
                    contact_world.shape[0],
                    friction_world,
                    friction_local,
                    world_friction_count,
                    friction_impulse_bound,
                    contact_world,
                    contact_local,
                    world_contact_count,
                    contact_body_first,
                    contact_body_second,
                    contact_friction,
                    limit_world,
                    limit_local,
                    world_limit_count,
                    world_active,
                ],
                outputs=[
                    friction_reaction,
                    friction_trial,
                    friction_previous,
                    contact_reaction,
                    contact_trial,
                    contact_previous,
                    limit_reaction,
                    limit_trial,
                    limit_previous,
                ],
                device=projected_twist.device,
            )
    elif warm_start:
        wp.launch(
            _initialize_jacobi_projection_status,
            dim=world_count,
            inputs=[world_active, prepared_status],
            outputs=[world_status],
            device=projected_twist.device,
        )
    if not use_world_projection:
        twist_delta.zero_()
    if warm_start and not use_world_projection and contact_world.shape[0] > 0:
        wp.launch(
            _warmstart_contacts_jacobi,
            dim=contact_world.shape[0],
            inputs=[
                contact_world,
                contact_local,
                world_active,
                prepared_status,
                world_contact_count,
                contact_body_first,
                contact_body_second,
                contact_jacobian_first,
                contact_jacobian_second,
                inverse_weight,
                False,
                contact_reaction,
            ],
            outputs=[twist_delta],
            device=projected_twist.device,
        )
    if warm_start and not use_world_projection and friction_world.shape[0] > 0:
        wp.launch(
            _warmstart_frictions_jacobi,
            dim=friction_world.shape[0],
            inputs=[
                friction_world,
                friction_local,
                world_active,
                prepared_status,
                world_friction_count,
                friction_body_first,
                friction_body_second,
                friction_jacobian_first,
                friction_jacobian_second,
                inverse_weight,
                False,
                friction_reaction,
            ],
            outputs=[twist_delta],
            device=projected_twist.device,
        )
    if warm_start and not use_world_projection and limit_world.shape[0] > 0:
        wp.launch(
            _warmstart_limits_jacobi,
            dim=limit_world.shape[0],
            inputs=[
                limit_world,
                limit_local,
                world_active,
                prepared_status,
                world_limit_count,
                limit_body_first,
                limit_body_second,
                limit_jacobian_first,
                limit_jacobian_second,
                inverse_weight,
                False,
                limit_reaction,
            ],
            outputs=[twist_delta],
            device=projected_twist.device,
        )
    if warm_start and not use_world_projection:
        _apply_jacobi_warmstart_delta(
            body_world,
            world_active,
            world_status,
            inverse_weight,
            twist_delta,
            projected_twist,
        )

    if use_world_projection:
        wp.launch(
            _project_rigid_constraints_jacobi_by_world,
            dim=(world_count, _JACOBI_WORLD_BLOCK_DIM),
            block_dim=_JACOBI_WORLD_BLOCK_DIM,
            inputs=[
                projection_iterations,
                warm_start,
                world_active,
                world_body_offset,
                world_body_count,
                world_friction_offset,
                world_friction_count,
                friction_body_first,
                friction_body_second,
                friction_jacobian_first,
                friction_jacobian_second,
                friction_impulse_bound,
                friction_delassus,
                world_contact_offset,
                world_contact_count,
                contact_body_first,
                contact_body_second,
                contact_jacobian_first,
                contact_jacobian_second,
                contact_delassus,
                contact_bias,
                contact_friction,
                world_limit_offset,
                world_limit_count,
                limit_body_first,
                limit_body_second,
                limit_jacobian_first,
                limit_jacobian_second,
                limit_bias,
                limit_delassus,
                inverse_weight,
                projected_twist,
            ],
            outputs=[
                friction_reaction,
                contact_reaction,
                limit_reaction,
                twist_delta,
                world_status,
            ],
            device=projected_twist.device,
        )
        return

    projection_state = _make_rigid_projection_state(
        world_active,
        projected_twist,
        twist_delta,
        world_status,
    )
    contact_data = _make_projection_struct(
        _RigidContactProjectionData,
        body_first=contact_body_first,
        body_second=contact_body_second,
        jacobian_first=contact_jacobian_first,
        jacobian_second=contact_jacobian_second,
        delassus=contact_delassus,
        bias=contact_bias,
        friction=contact_friction,
        reaction=contact_reaction,
    )
    friction_index = _make_direct_projection_index(friction_world, friction_local, world_friction_count)
    contact_index = _make_direct_projection_index(contact_world, contact_local, world_contact_count)
    limit_index = _make_direct_projection_index(limit_world, limit_local, world_limit_count)
    for _sweep in range(projection_iterations):
        if friction_world.shape[0] > 0:
            wp.launch(
                _make_project_scalar_kernel(False, False),
                dim=friction_world.shape[0],
                inputs=[
                    friction_world.shape[0],
                    0,
                    friction_index,
                    friction_body_first,
                    friction_body_second,
                    friction_jacobian_first,
                    friction_jacobian_second,
                    friction_impulse_bound,
                    friction_impulse_bound,
                    friction_delassus,
                    friction_reaction,
                    projection_state,
                ],
                device=projected_twist.device,
            )
        if contact_world.shape[0] > 0:
            contact_inputs = [
                contact_world.shape[0],
                0,
                contact_index,
                contact_data,
                projection_state,
            ]
            wp.launch(
                _make_project_contacts_kernel(False),
                dim=contact_world.shape[0],
                inputs=contact_inputs,
                device=projected_twist.device,
                max_blocks=contact_projection_max_blocks,
                block_dim=_JACOBI_CONTACT_BLOCK_DIM,
            )
        if limit_world.shape[0] > 0:
            wp.launch(
                _make_project_scalar_kernel(False, True),
                dim=limit_world.shape[0],
                inputs=[
                    limit_world.shape[0],
                    0,
                    limit_index,
                    limit_body_first,
                    limit_body_second,
                    limit_jacobian_first,
                    limit_jacobian_second,
                    limit_bias,
                    limit_bias,
                    limit_delassus,
                    limit_reaction,
                    projection_state,
                ],
                device=projected_twist.device,
            )
        _apply_jacobi_delta(
            body_world,
            world_active,
            world_status,
            inverse_weight,
            twist_delta,
            projected_twist,
        )
        if accelerated and _sweep + 1 < projection_iterations:
            if rigid_capacity > 0:
                wp.launch(
                    _accumulate_rigid_restart,
                    dim=rigid_capacity,
                    inputs=[
                        friction_world.shape[0],
                        contact_world.shape[0],
                        friction_world,
                        friction_local,
                        world_friction_count,
                        contact_world,
                        contact_local,
                        world_contact_count,
                        limit_world,
                        limit_local,
                        world_limit_count,
                        world_active,
                        world_status,
                        friction_reaction,
                        friction_trial,
                        friction_previous,
                        contact_reaction,
                        contact_trial,
                        contact_previous,
                        limit_reaction,
                        limit_trial,
                        limit_previous,
                    ],
                    outputs=[restart_dot],
                    device=projected_twist.device,
                )
            wp.launch(
                _finalize_acceleration,
                dim=world_count,
                inputs=[world_active, world_status],
                outputs=[restart_dot, theta, beta],
                device=projected_twist.device,
            )
            if rigid_capacity > 0:
                wp.launch(
                    _extrapolate_rigid_reactions,
                    dim=rigid_capacity,
                    inputs=[
                        friction_world.shape[0],
                        contact_world.shape[0],
                        friction_world,
                        friction_local,
                        world_friction_count,
                        friction_body_first,
                        friction_body_second,
                        friction_jacobian_first,
                        friction_jacobian_second,
                        contact_world,
                        contact_local,
                        world_contact_count,
                        contact_body_first,
                        contact_body_second,
                        contact_jacobian_first,
                        contact_jacobian_second,
                        limit_world,
                        limit_local,
                        world_limit_count,
                        limit_body_first,
                        limit_body_second,
                        limit_jacobian_first,
                        limit_jacobian_second,
                        world_active,
                        world_status,
                        beta,
                        friction_reaction,
                        friction_trial,
                        friction_previous,
                        contact_reaction,
                        contact_trial,
                        contact_previous,
                        limit_reaction,
                        limit_trial,
                        limit_previous,
                    ],
                    outputs=[twist_delta],
                    device=projected_twist.device,
                )
            _apply_jacobi_delta(
                body_world,
                world_active,
                world_status,
                inverse_weight,
                twist_delta,
                projected_twist,
            )
