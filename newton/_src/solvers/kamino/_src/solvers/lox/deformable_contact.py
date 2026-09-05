# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body-space contact projection for LOX deformable particles."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ......utils.mesh import MeshAdjacencyData
from ...core.types import mat36f, mat66f, vec6f
from ...geometry.contacts import make_contact_frame_znorm
from .bias import compute_contact_velocity_target
from .contact import (
    compute_contact_scaled_alart_curnier_residual,
    project_contact_coulomb_cone,
    solve_contact_coulomb_isotropic,
)
from .projection import (
    PROJECTION_STATUS_INVALID,
    PROJECTION_STATUS_VALID,
    prepare_contact_coulomb_delassus,
)
from .soft_contact_filter import (
    compute_soft_edge_normal_cones,
    soft_surface_contact_normal_cone_contains,
)
from .sweep import (
    _DeformableRigidContactProjectionData,
    _DeformableRigidProjectionState,
    _finalize_acceleration,
    _initialize_acceleration_worlds,
    _make_direct_projection_index,
    _make_project_contacts_kernel,
    _make_projection_struct,
)
from .time import validate_world_time_step

if TYPE_CHECKING:
    from ......sim import Contacts, Model, State
    from .deformable_self_contact import DeformableSelfContactDetector
    from .deformable_system import DeformableFEMSystem

__all__ = [
    "DEFORMABLE_CONTACT_STATUS_CROSS_WORLD",
    "DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS",
    "DEFORMABLE_CONTACT_STATUS_MALFORMED",
    "DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE",
    "DEFORMABLE_CONTACT_STATUS_UNUSED",
    "DEFORMABLE_CONTACT_STATUS_VALID",
    "DeformableContactSystem",
    "compute_deformable_contact_residual",
    "project_deformable_contact_coulomb",
]

DEFORMABLE_CONTACT_STATUS_UNUSED = 0
"""The contact-capacity slot does not contain an active source record."""

DEFORMABLE_CONTACT_STATUS_VALID = 1
"""The source record and its scalar Delassus coefficient are valid."""

DEFORMABLE_CONTACT_STATUS_MALFORMED = 2
"""The source record has invalid indices, coefficients, geometry, or shape data."""

DEFORMABLE_CONTACT_STATUS_CROSS_WORLD = 3
"""The contact feature or collider spans incompatible Newton worlds."""

DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS = 5
"""The contact has no finite positive scalar Delassus coefficient."""

DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE = 6
"""The projection produced a non-finite reaction or particle correction."""

_BODY_FLAG_KINEMATIC = 1 << 1
_PARTICLE_FLAG_ACTIVE = 1
_COEFFICIENT_TOLERANCE = 1.0e-5
_NORMAL_EPSILON = 1.0e-12
# The Coulomb solve's register footprint benefits from smaller projection blocks.
_RIGID_CONTACT_PROJECTION_BLOCK_DIM = 128


@wp.struct
class _ParticleContactProjectionData:
    particle_indices: wp.array2d[wp.int32]
    coefficients: wp.array2d[wp.float32]
    contact_body: wp.array[wp.int32]
    frame: wp.array[wp.mat33f]
    bias: wp.array[wp.vec3f]
    friction: wp.array[wp.float32]
    delassus: wp.array[wp.float32]
    status: wp.array[wp.int32]
    reaction: wp.array[wp.vec3f]


@wp.struct
class _ParticleContactDirectState:
    world_active: wp.array[wp.int32]
    world_status: wp.array[wp.int32]
    contact_world_status: wp.array[wp.int32]
    global_status: wp.array[wp.int32]
    inverse_weight: wp.array[wp.float32]
    projected_velocity: wp.array[wp.vec3f]
    particle_delta: wp.array[wp.vec3f]


@wp.struct
class _ParticleContactColoredState:
    world_active: wp.array[wp.bool]
    world_status: wp.array[wp.int32]
    contact_world_status: wp.array[wp.int32]
    occupancy: wp.array2d[wp.int32]
    inverse_weight: wp.array[wp.float32]
    projected_velocity: wp.array[wp.vec3f]
    particle_delta: wp.array[wp.vec3f]


wp.set_module_options({"enable_backward": False})


@wp.func
def _is_finite_vec3(value: wp.vec3) -> bool:
    return wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])


@wp.func
def _mark_numerical_failure(
    contact: int,
    world: int,
    contact_status: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
):
    contact_status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
    wp.atomic_max(world_status, world, DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE)


@wp.func
def _fail_projection(
    contact: int,
    world: int,
    contact_status: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    projection_status: wp.array[wp.int32],
):
    _mark_numerical_failure(contact, world, contact_status, world_status)
    projection_status[world] = PROJECTION_STATUS_INVALID


@wp.func
def _reject_contact(
    contact: int,
    world: int,
    status: int,
    contact_status: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    global_status: wp.array[wp.int32],
):
    contact_status[contact] = status
    if world >= 0 and world < world_status.shape[0]:
        wp.atomic_max(world_status, world, status)
    else:
        wp.atomic_max(global_status, 0, status)


@wp.struct
class _SoftContactFeature:
    status: wp.int32
    world: wp.int32
    particle_indices: wp.vec3i
    coefficients: wp.vec3
    position: wp.vec3
    velocity: wp.vec3
    prescribed_velocity: wp.vec3
    radius: wp.float32


@wp.func
def _prepare_soft_contact_feature(
    source_particles: wp.vec3i,
    source_coefficients: wp.vec3,
    particle_position: wp.array[wp.vec3],
    particle_velocity: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    newton_to_packed: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    inverse_weight: wp.array[float],
) -> _SoftContactFeature:
    """Validate and map one source feature to packed deformable particles."""
    result = _SoftContactFeature()
    result.status = DEFORMABLE_CONTACT_STATUS_VALID
    result.world = -1
    result.particle_indices = wp.vec3i(-1)
    result.coefficients = wp.vec3(0.0)
    result.position = wp.vec3(0.0)
    result.velocity = wp.vec3(0.0)
    result.prescribed_velocity = wp.vec3(0.0)
    result.radius = 0.0

    slot_count = int(0)
    coefficient_sum = float(0.0)
    found_padding = False
    malformed = False
    for slot in range(3):
        source_particle = source_particles[slot]
        coefficient = source_coefficients[slot]
        if source_particle < 0:
            found_padding = True
            malformed = malformed or source_particle != -1 or wp.abs(coefficient) > _COEFFICIENT_TOLERANCE
        else:
            malformed = (
                malformed
                or found_padding
                or source_particle >= newton_to_packed.shape[0]
                or not wp.isfinite(coefficient)
                or coefficient < -_COEFFICIENT_TOLERANCE
            )
            if not malformed:
                for previous_slot in range(3):
                    if previous_slot < slot:
                        malformed = malformed or source_particles[previous_slot] == source_particle
            if not malformed:
                packed_particle = newton_to_packed[source_particle]
                particle_world = packed_world[packed_particle]
                if result.world < 0:
                    result.world = particle_world
                elif particle_world != result.world:
                    result.status = DEFORMABLE_CONTACT_STATUS_CROSS_WORLD
                    return result
                result.particle_indices[slot] = packed_particle
                result.coefficients[slot] = coefficient
                result.position += coefficient * particle_position[source_particle]
                if (particle_flags[source_particle] & _PARTICLE_FLAG_ACTIVE) != 0:
                    result.velocity += coefficient * particle_velocity[source_particle]
                    if inverse_weight[packed_particle] <= 0.0:
                        result.prescribed_velocity += coefficient * particle_velocity[source_particle]
                result.radius = wp.max(result.radius, particle_radius[source_particle])
                coefficient_sum += coefficient
                slot_count += 1

    if (
        malformed
        or slot_count == 0
        or result.world < 0
        or wp.abs(coefficient_sum - 1.0) > _COEFFICIENT_TOLERANCE
        or not _is_finite_vec3(result.position)
        or not _is_finite_vec3(result.velocity)
        or not wp.isfinite(result.radius)
        or result.radius < 0.0
    ):
        result.status = DEFORMABLE_CONTACT_STATUS_MALFORMED
    return result


@wp.struct
class _SoftContactCollider:
    status: wp.int32
    dynamic_body: wp.int32
    position: wp.vec3
    velocity: wp.vec3
    prescribed_velocity: wp.vec3
    center_of_mass: wp.vec3


@wp.func
def _prepare_soft_contact_collider(
    contact: wp.int32,
    shape: wp.int32,
    world: wp.int32,
    source_body_position: wp.array[wp.vec3],
    source_body_velocity: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    body_pose: wp.array[wp.transform],
    body_velocity: wp.array[wp.spatial_vector],
    body_center_of_mass: wp.array[wp.vec3],
    body_flags: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
) -> _SoftContactCollider:
    """Prepare collider geometry and velocity in world coordinates."""
    result = _SoftContactCollider()
    result.status = DEFORMABLE_CONTACT_STATUS_VALID
    result.dynamic_body = -1
    result.position = source_body_position[contact]
    result.velocity = source_body_velocity[contact]
    result.prescribed_velocity = result.velocity
    result.center_of_mass = wp.vec3(0.0)

    body = shape_body[shape]
    if body < 0:
        return result
    if (
        body >= body_pose.shape[0]
        or body >= body_velocity.shape[0]
        or body >= body_center_of_mass.shape[0]
        or body >= body_flags.shape[0]
        or body >= body_world.shape[0]
    ):
        result.status = DEFORMABLE_CONTACT_STATUS_MALFORMED
        return result
    collider_world = body_world[body]
    if collider_world >= 0 and collider_world != world:
        result.status = DEFORMABLE_CONTACT_STATUS_CROSS_WORLD
        return result

    pose = body_pose[body]
    result.position = wp.transform_point(pose, result.position)
    result.center_of_mass = wp.transform_point(pose, body_center_of_mass[body])
    result.prescribed_velocity = wp.transform_vector(pose, source_body_velocity[contact])
    spatial_velocity = body_velocity[body]
    linear_velocity = wp.spatial_top(spatial_velocity)
    angular_velocity = wp.spatial_bottom(spatial_velocity)
    result.velocity = (
        linear_velocity
        + wp.cross(angular_velocity, result.position - result.center_of_mass)
        + result.prescribed_velocity
    )
    if (body_flags[body] & _BODY_FLAG_KINEMATIC) == 0:
        result.dynamic_body = body
    return result


@wp.struct
class _SoftContactGeometry:
    gap: wp.float32
    bias: wp.vec3
    frame: wp.mat33f
    body_jacobian: mat36f


@wp.func
def _prepare_soft_contact_geometry(
    feature: _SoftContactFeature,
    collider: _SoftContactCollider,
    contact_normal: wp.vec3,
    shape_margin: wp.float32,
    time_step: wp.float32,
    stabilization_fraction: wp.float32,
    dead_zone: wp.float32,
    impact_velocity_threshold: wp.float32,
    recoverable_response: wp.bool,
    restitution: wp.float32,
) -> _SoftContactGeometry:
    """Build the contact frame, velocity targets, and rigid Jacobian."""
    result = _SoftContactGeometry()
    result.gap = wp.dot(contact_normal, feature.position - collider.position) - feature.radius - shape_margin
    previous_normal_velocity = wp.dot(contact_normal, feature.velocity - collider.velocity)
    velocity_target = compute_contact_velocity_target(
        result.gap,
        previous_normal_velocity,
        restitution,
        time_step,
        stabilization_fraction,
        dead_zone,
        impact_velocity_threshold,
        recoverable_response,
    )
    result.frame = make_contact_frame_znorm(contact_normal)
    frame_transpose = wp.transpose(result.frame)
    result.bias = frame_transpose @ (feature.prescribed_velocity - collider.velocity - velocity_target * contact_normal)
    result.body_jacobian = mat36f(0.0)
    if collider.dynamic_body >= 0:
        result.bias = frame_transpose @ (
            feature.prescribed_velocity - collider.prescribed_velocity - velocity_target * contact_normal
        )
        angular_jacobian = frame_transpose @ wp.skew(collider.position - collider.center_of_mass)
        for row in range(3):
            for col in range(3):
                result.body_jacobian[row, col] = -frame_transpose[row, col]
                result.body_jacobian[row, 3 + col] = angular_jacobian[row, col]
    return result


@wp.kernel
def _adapt_soft_contacts(
    source_count: wp.array[wp.int32],
    source_capacity: int,
    source_indices: wp.array[wp.vec3i],
    source_barycentric: wp.array[wp.vec3],
    source_shape: wp.array[wp.int32],
    source_body_position: wp.array[wp.vec3],
    source_body_velocity: wp.array[wp.vec3],
    source_normal: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    cone_axis: wp.array[wp.vec3],
    cone_cosine: wp.array[float],
    filter_surface_contacts: bool,
    normal_cone_filtering_min_distance: float,
    particle_position: wp.array[wp.vec3],
    particle_velocity: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    newton_to_packed: wp.array[wp.int32],
    packed_world: wp.array[wp.int32],
    inverse_weight: wp.array[float],
    shape_body: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    shape_margin: wp.array[float],
    body_pose: wp.array[wp.transform],
    body_velocity: wp.array[wp.spatial_vector],
    body_center_of_mass: wp.array[wp.vec3],
    body_flags: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    time_step: wp.array[wp.float32],
    stabilization_fraction: float,
    dead_zone: float,
    impact_velocity_threshold: float,
    recoverable_response: bool,
    friction: float,
    restitution: float,
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_shape: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    gap: wp.array[float],
    bias: wp.array[wp.vec3],
    contact_friction: wp.array[float],
    contact_status: wp.array[wp.int32],
    particle_multiplicity: wp.array[wp.int32],
    particle_majorizer_weight_sum: wp.array[float],
    world_contact_count: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    global_status: wp.array[wp.int32],
):
    contact = wp.tid()
    if contact == 0:
        count = source_count[0]
        if count < 0:
            wp.atomic_max(global_status, 0, DEFORMABLE_CONTACT_STATUS_MALFORMED)
        elif count > source_capacity:
            wp.atomic_max(global_status, 0, DEFORMABLE_CONTACT_STATUS_MALFORMED)
    for slot in range(4):
        particle_indices[contact, slot] = -1
        coefficients[contact, slot] = 0.0
    contact_world[contact] = -1
    contact_shape[contact] = -1
    contact_body[contact] = -1
    frame[contact] = wp.mat33f(0.0)
    body_jacobian[contact] = mat36f(0.0)
    gap[contact] = 0.0
    bias[contact] = wp.vec3(0.0)
    contact_friction[contact] = 0.0
    contact_status[contact] = DEFORMABLE_CONTACT_STATUS_UNUSED

    count = wp.min(source_count[0], source_indices.shape[0])
    if contact >= count:
        return

    source_particles = source_indices[contact]
    source_coefficients = source_barycentric[contact]
    shape = source_shape[contact]
    if (
        shape < 0
        or shape >= shape_body.shape[0]
        or shape >= shape_world.shape[0]
        or shape >= shape_margin.shape[0]
        or not _is_finite_vec3(source_body_position[contact])
        or not _is_finite_vec3(source_body_velocity[contact])
        or not _is_finite_vec3(source_normal[contact])
    ):
        _reject_contact(
            contact,
            -1,
            DEFORMABLE_CONTACT_STATUS_MALFORMED,
            contact_status,
            world_status,
            global_status,
        )
        return

    feature = _prepare_soft_contact_feature(
        source_particles,
        source_coefficients,
        particle_position,
        particle_velocity,
        particle_radius,
        particle_flags,
        newton_to_packed,
        packed_world,
        inverse_weight,
    )
    for slot in range(3):
        particle_indices[contact, slot] = feature.particle_indices[slot]
        coefficients[contact, slot] = feature.coefficients[slot]
    if feature.status != DEFORMABLE_CONTACT_STATUS_VALID:
        _reject_contact(
            contact,
            feature.world,
            feature.status,
            contact_status,
            world_status,
            global_status,
        )
        return

    normal_length = wp.length(source_normal[contact])
    if not wp.isfinite(normal_length) or normal_length <= _NORMAL_EPSILON:
        _reject_contact(
            contact,
            feature.world,
            DEFORMABLE_CONTACT_STATUS_MALFORMED,
            contact_status,
            world_status,
            global_status,
        )
        return
    contact_normal = source_normal[contact] / normal_length
    collider_world = shape_world[shape]
    if collider_world >= 0 and collider_world != feature.world:
        _reject_contact(
            contact,
            feature.world,
            DEFORMABLE_CONTACT_STATUS_CROSS_WORLD,
            contact_status,
            world_status,
            global_status,
        )
        return

    collider = _prepare_soft_contact_collider(
        contact,
        shape,
        feature.world,
        source_body_position,
        source_body_velocity,
        shape_body,
        body_pose,
        body_velocity,
        body_center_of_mass,
        body_flags,
        body_world,
    )
    if collider.status != DEFORMABLE_CONTACT_STATUS_VALID:
        _reject_contact(
            contact,
            feature.world,
            collider.status,
            contact_status,
            world_status,
            global_status,
        )
        return
    if (
        not _is_finite_vec3(collider.position)
        or not _is_finite_vec3(collider.velocity)
        or not wp.isfinite(shape_margin[shape])
        or shape_margin[shape] < 0.0
        or not wp.isfinite(friction)
        or friction < 0.0
        or not wp.isfinite(restitution)
        or restitution < 0.0
    ):
        _reject_contact(
            contact,
            feature.world,
            DEFORMABLE_CONTACT_STATUS_MALFORMED,
            contact_status,
            world_status,
            global_status,
        )
        return

    surface_separation = wp.length(feature.position - collider.position)
    # At nearly coincident closest points, small positional errors can rotate
    # the derived normal arbitrarily, so retain the contact conservatively.
    if (
        filter_surface_contacts
        and surface_separation > normal_cone_filtering_min_distance
        and not soft_surface_contact_normal_cone_contains(
            source_particles,
            source_coefficients,
            -contact_normal,
            particle_position,
            edge_indices,
            adjacency,
            cone_axis,
            cone_cosine,
        )
    ):
        return

    geometry = _prepare_soft_contact_geometry(
        feature,
        collider,
        contact_normal,
        shape_margin[shape],
        time_step[feature.world],
        stabilization_fraction,
        dead_zone,
        impact_velocity_threshold,
        recoverable_response,
        restitution,
    )
    if not wp.isfinite(geometry.gap) or not _is_finite_vec3(geometry.bias):
        _reject_contact(
            contact,
            feature.world,
            DEFORMABLE_CONTACT_STATUS_MALFORMED,
            contact_status,
            world_status,
            global_status,
        )
        return

    contact_world[contact] = feature.world
    contact_shape[contact] = shape
    contact_body[contact] = collider.dynamic_body
    frame[contact] = geometry.frame
    body_jacobian[contact] = geometry.body_jacobian
    gap[contact] = geometry.gap
    bias[contact] = geometry.bias
    contact_friction[contact] = friction
    contact_status[contact] = DEFORMABLE_CONTACT_STATUS_VALID
    wp.atomic_max(world_status, feature.world, DEFORMABLE_CONTACT_STATUS_VALID)
    wp.atomic_add(world_contact_count, feature.world, 1)
    for slot in range(3):
        packed_particle = particle_indices[contact, slot]
        if (
            packed_particle >= 0
            and coefficients[contact, slot] > _COEFFICIENT_TOLERANCE
            and inverse_weight[packed_particle] > 0.0
        ):
            wp.atomic_add(particle_multiplicity, packed_particle, 1)
            wp.atomic_add(
                particle_majorizer_weight_sum,
                packed_particle,
                wp.abs(coefficients[contact, slot]),
            )


@wp.kernel
def _finalize_particle_majorizer_scale(
    inverse_weight: wp.array[float],
    majorizer_weight_sum: wp.array[float],
    majorizer_scale: wp.array[float],
):
    particle = wp.tid()
    value = majorizer_weight_sum[particle] * inverse_weight[particle]
    if not wp.isfinite(value) or value < 0.0:
        value = 0.0
    majorizer_scale[particle] = value


@wp.kernel
def _finalize_contact_delassus(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    particle_majorizer_scale: wp.array[float],
    contact_status: wp.array[wp.int32],
    delassus: wp.array[float],
    world_contact_count: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    global_status: wp.array[wp.int32],
):
    contact = wp.tid()
    delassus[contact] = 0.0
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
        return

    value = float(0.0)
    for slot in range(4):
        particle = particle_indices[contact, slot]
        coefficient = coefficients[contact, slot]
        if particle >= 0 and wp.abs(coefficient) > _COEFFICIENT_TOLERANCE:
            # Weighted Cauchy--Schwarz with the per-particle partition
            # omega_cp = |a_cp| / sum_k |a_kp| gives the separable term
            # a_cp^2 M_p^-1 / omega_cp = |a_cp| sum_k |a_kp| M_p^-1.
            value += wp.abs(coefficient) * particle_majorizer_scale[particle]

    if not wp.isfinite(value) or (value <= 0.0 and contact_body[contact] < 0):
        world = contact_world[contact]
        if world >= 0:
            wp.atomic_add(world_contact_count, world, -1)
        _reject_contact(
            contact,
            world,
            DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS,
            contact_status,
            world_status,
            global_status,
        )
        return
    delassus[contact] = value


@wp.func
def project_deformable_contact_coulomb(
    free_velocity: wp.vec3,
    normal: wp.vec3,
    delassus: float,
    friction: float,
) -> wp.vec3:
    """Solve one isotropic Coulomb contact in world space."""
    return solve_contact_coulomb_isotropic(delassus, free_velocity, normal, friction)


@wp.func
def project_deformable_contact_coulomb_local(
    current_velocity: wp.vec3,
    reaction_old: wp.vec3,
    delassus: float,
    friction: float,
) -> wp.vec3:
    """Apply the isotropic local Coulomb map used by parallel projections."""
    return project_deformable_contact_coulomb(
        current_velocity - delassus * reaction_old,
        wp.vec3(0.0, 0.0, 1.0),
        delassus,
        friction,
    )


@wp.func
def _project_world_coulomb_cone(value: wp.vec3, normal: wp.vec3, friction: float) -> wp.vec3:
    normal_value = wp.dot(normal, value)
    tangent_value = value - normal_value * normal
    tangent_norm = wp.length(tangent_value)
    if friction * tangent_norm <= -normal_value:
        return wp.vec3(0.0)
    if tangent_norm <= friction * normal_value:
        return value

    projected_normal = (friction * tangent_norm + normal_value) / (friction * friction + 1.0)
    if tangent_norm > 0.0:
        return projected_normal * normal + friction * projected_normal * tangent_value / tangent_norm
    return projected_normal * normal


@wp.func
def compute_deformable_contact_residual(
    delassus: float,
    reaction: wp.vec3,
    velocity: wp.vec3,
    normal: wp.vec3,
    friction: float,
) -> wp.vec3:
    """Compute the rotationally invariant scaled contact natural-map residual."""
    scale = wp.sqrt(delassus)
    scaled_reaction = scale * reaction
    scaled_velocity = velocity / scale
    normal_velocity = wp.dot(normal, scaled_velocity)
    tangent_velocity = scaled_velocity - normal_velocity * normal
    modified_velocity = scaled_velocity + friction * wp.length(tangent_velocity) * normal
    projected = _project_world_coulomb_cone(scaled_reaction - modified_velocity, normal, friction)
    return scaled_reaction - projected


@wp.kernel
def _warm_start_contacts(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    world_status: wp.array[wp.int32],
    global_status: wp.array[wp.int32],
    inverse_weight: wp.array[float],
    reaction: wp.array[wp.vec3],
    particle_delta: wp.array[wp.vec3],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        global_status[0] != DEFORMABLE_CONTACT_STATUS_UNUSED
        or contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
        or contact_body[contact] >= 0
        or world < 0
        or world_active[world] == 0
        or world_status[world] != DEFORMABLE_CONTACT_STATUS_VALID
    ):
        return

    contact_reaction = frame[contact] @ reaction[contact]
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            correction = inverse_weight[particle] * coefficients[contact, slot] * contact_reaction
            wp.atomic_add(particle_delta, particle, correction)


@cache
def _make_project_particle_contacts_kernel(colored: bool):
    """Specialize the particle-only contact map by update strategy."""

    @wp.kernel(module="unique")
    def project_particle_contacts_kernel(
        launch_dim: wp.int32,
        target_color: wp.int32,
        index: Any,
        data: _ParticleContactProjectionData,
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
            contact = index.order[ordered] if wp.static(colored) else ordered
            world = index.constraint_world[contact]
            if wp.static(not colored):
                if (
                    state.global_status[0] != DEFORMABLE_CONTACT_STATUS_UNUSED
                    or data.contact_body[contact] >= 0
                    or world < 0
                ):
                    continue
            if (
                data.status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
                or not state.world_active[world]
                or state.world_status[world] != DEFORMABLE_CONTACT_STATUS_VALID
            ):
                continue

            contact_frame = data.frame[contact]
            velocity = data.bias[contact]
            for slot in range(4):
                particle = data.particle_indices[contact, slot]
                if particle >= 0:
                    velocity += data.coefficients[contact, slot] * (
                        wp.transpose(contact_frame) @ state.projected_velocity[particle]
                    )
            old = data.reaction[contact]
            value = data.delassus[contact]
            new = project_deformable_contact_coulomb_local(
                velocity,
                old,
                value,
                data.friction[contact],
            )
            delta = contact_frame @ (new - old)
            if not _is_finite_vec3(new) or not _is_finite_vec3(delta):
                data.status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
                wp.atomic_max(
                    state.contact_world_status,
                    world,
                    DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE,
                )
                if wp.static(colored):
                    state.world_status[world] = PROJECTION_STATUS_INVALID
                continue

            correction_is_finite = wp.bool(True)
            for slot in range(4):
                particle = data.particle_indices[contact, slot]
                if particle >= 0:
                    coefficient = data.coefficients[contact, slot]
                    if wp.static(colored):
                        for later in range(slot + 1, 4):
                            if data.particle_indices[contact, later] == particle:
                                coefficient += data.coefficients[contact, later]
                    unique = wp.bool(True)
                    if wp.static(colored):
                        for previous in range(slot):
                            unique = unique and data.particle_indices[contact, previous] != particle
                    if unique:
                        correction = state.inverse_weight[particle] * coefficient * delta
                        if not _is_finite_vec3(correction):
                            correction_is_finite = False
                        elif wp.static(colored):
                            if state.occupancy[particle, target_color] == 1:
                                state.particle_delta[particle] += correction
                            else:
                                wp.atomic_add(state.particle_delta, particle, correction)
                        else:
                            wp.atomic_add(state.particle_delta, particle, correction)
            if not correction_is_finite:
                data.status[contact] = DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE
                wp.atomic_max(
                    state.contact_world_status,
                    world,
                    DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE,
                )
                if wp.static(colored):
                    state.world_status[world] = PROJECTION_STATUS_INVALID
                continue
            data.reaction[contact] = new

    return project_particle_contacts_kernel


@wp.kernel
def _apply_particle_delta(
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    particle_delta: wp.array[wp.vec3],
    projected_velocity: wp.array[wp.vec3],
):
    particle = wp.tid()
    if world_active[packed_world[particle]] != 0:
        projected_velocity[particle] += particle_delta[particle]
    particle_delta[particle] = wp.vec3(0.0)


@wp.kernel
def _compute_contact_residuals(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    bias: wp.array[wp.vec3],
    friction: wp.array[float],
    scalar_delassus: wp.array[float],
    rigid_delassus: wp.array[wp.mat33f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
    projected_velocity: wp.array[wp.vec3],
    projected_twist: wp.array[vec6f],
    reaction: wp.array[wp.vec3],
    rigid_coordinates: bool,
    world_contact_residual: wp.array[float],
):
    contact = wp.tid()
    world = contact_world[contact]
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID or world < 0 or world_active[world] == 0:
        return

    contact_frame = frame[contact]
    velocity = bias[contact]
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            particle_velocity = projected_velocity[particle]
            particle_velocity = wp.transpose(contact_frame) @ particle_velocity
            velocity += coefficients[contact, slot] * particle_velocity
    body = contact_body[contact]
    if rigid_coordinates and body >= 0:
        velocity += body_jacobian[contact] @ projected_twist[body]

    if rigid_coordinates:
        residual_vector = compute_contact_scaled_alart_curnier_residual(
            rigid_delassus[contact],
            reaction[contact],
            velocity,
            friction[contact],
        )
    else:
        residual_vector = compute_deformable_contact_residual(
            scalar_delassus[contact],
            reaction[contact],
            velocity,
            wp.vec3(0.0, 0.0, 1.0),
            friction[contact],
        )
    residual = wp.max(wp.abs(residual_vector[0]), wp.abs(residual_vector[1]))
    residual = wp.max(residual, wp.abs(residual_vector[2]))
    wp.atomic_max(world_contact_residual, world, residual)


@wp.func
def _is_finite_vec6(value: vec6f) -> bool:
    finite = True
    for index in range(6):
        finite = finite and wp.isfinite(value[index])
    return finite


@wp.kernel
def _accumulate_rigid_incidence(
    contact_body: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_status: wp.array[wp.int32],
    body_constraint_count: wp.array[wp.int32],
    body_has_unilateral: wp.array[wp.int32],
    world_has_unilateral: wp.array[wp.bool],
):
    contact = wp.tid()
    body = contact_body[contact]
    if contact_status[contact] == DEFORMABLE_CONTACT_STATUS_VALID and body >= 0:
        wp.atomic_add(body_constraint_count, body, 1)
        wp.atomic_max(body_has_unilateral, body, 1)
        world_has_unilateral[contact_world[contact]] = True


@wp.kernel
def _prepare_rigid_contacts(
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    body_jacobian: wp.array[mat36f],
    rigid_bias: wp.array[wp.vec3],
    friction: wp.array[float],
    particle_delassus: wp.array[float],
    body_constraint_count: wp.array[wp.int32],
    static_body_constraint_count: wp.array[wp.int32],
    body_inverse_weight: wp.array[mat66f],
    contact_status: wp.array[wp.int32],
    delassus: wp.array[wp.mat33f],
    world_status: wp.array[wp.int32],
):
    contact = wp.tid()
    delassus[contact] = wp.mat33f(0.0)
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID:
        return

    value = particle_delassus[contact] * wp.identity(3, dtype=wp.float32)
    body = contact_body[contact]
    if body >= 0:
        multiplicity = wp.max(1, body_constraint_count[body] - static_body_constraint_count[body])
        split_inverse_weight = wp.float32(multiplicity) * body_inverse_weight[body]
        jacobian = body_jacobian[contact]
        value += jacobian @ split_inverse_weight @ wp.transpose(jacobian)

    data = prepare_contact_coulomb_delassus(
        value,
        rigid_bias[contact],
        friction[contact],
    )
    delassus[contact] = data.delassus
    if data.status == PROJECTION_STATUS_INVALID:
        contact_status[contact] = DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS
        wp.atomic_max(
            world_status,
            contact_world[contact],
            DEFORMABLE_CONTACT_STATUS_INVALID_DELASSUS,
        )


@wp.kernel
def _merge_rigid_prepared_status(
    contact_world_status: wp.array[wp.int32],
    contact_global_status: wp.array[wp.int32],
    rigid_prepared_status: wp.array[wp.int32],
):
    world = wp.tid()
    if (
        contact_global_status[0] > DEFORMABLE_CONTACT_STATUS_VALID
        or contact_world_status[world] > DEFORMABLE_CONTACT_STATUS_VALID
    ):
        rigid_prepared_status[world] = 0


@wp.kernel
def _warm_start_rigid_contacts(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    prepared_status: wp.array[wp.int32],
    particle_inverse_weight: wp.array[float],
    body_inverse_weight: wp.array[mat66f],
    apply_inverse_weight: wp.bool,
    reaction: wp.array[wp.vec3],
    particle_delta: wp.array[wp.vec3],
    body_delta: wp.array[vec6f],
    projection_status: wp.array[wp.int32],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
        or world < 0
        or not world_active[world]
        or prepared_status[world] != PROJECTION_STATUS_VALID
    ):
        return

    impulse = reaction[contact]
    world_impulse = frame[contact] @ impulse
    if not _is_finite_vec3(world_impulse):
        projection_status[world] = 0
        return
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            particle_correction = particle_inverse_weight[particle] * coefficients[contact, slot] * world_impulse
            if not _is_finite_vec3(particle_correction):
                projection_status[world] = 0
                return
            wp.atomic_add(particle_delta, particle, particle_correction)

    body = contact_body[contact]
    if body >= 0:
        body_wrench = wp.transpose(body_jacobian[contact]) @ impulse
        if apply_inverse_weight:
            body_wrench = body_inverse_weight[body] @ body_wrench
        if not _is_finite_vec6(body_wrench):
            projection_status[world] = 0
            return
        wp.atomic_add(body_delta, body, body_wrench)


@wp.kernel
def _initialize_accelerated_contacts(
    contact_world: wp.array[wp.int32],
    contact_status: wp.array[wp.int32],
    friction: wp.array[float],
    world_active: wp.array[wp.bool],
    reaction: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
):
    contact = wp.tid()
    world = contact_world[contact]
    if contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID or world < 0 or not world_active[world]:
        return
    value = project_contact_coulomb_cone(reaction[contact], friction[contact])
    reaction[contact] = value
    trial[contact] = value
    previous[contact] = value


@wp.kernel
def _accumulate_contact_restart(
    contact_world: wp.array[wp.int32],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    reaction: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    restart_dot: wp.array[float],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
        or world < 0
        or not world_active[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    current = reaction[contact]
    wp.atomic_add(restart_dot, world, wp.dot(current - trial[contact], current - previous[contact]))


@wp.kernel
def _extrapolate_contact_reactions(
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    contact_world: wp.array[wp.int32],
    contact_body: wp.array[wp.int32],
    frame: wp.array[wp.mat33f],
    body_jacobian: wp.array[mat36f],
    contact_status: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    beta: wp.array[float],
    particle_inverse_weight: wp.array[float],
    reaction: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    particle_delta: wp.array[wp.vec3],
    body_delta: wp.array[vec6f],
):
    contact = wp.tid()
    world = contact_world[contact]
    if (
        contact_status[contact] != DEFORMABLE_CONTACT_STATUS_VALID
        or world < 0
        or not world_active[world]
        or projection_status[world] != PROJECTION_STATUS_VALID
    ):
        return
    current = reaction[contact]
    extrapolated = current + beta[world] * (current - previous[contact])
    impulse_delta = extrapolated - current
    previous[contact] = current
    trial[contact] = extrapolated
    reaction[contact] = extrapolated
    world_delta = frame[contact] @ impulse_delta
    for slot in range(4):
        particle = particle_indices[contact, slot]
        if particle >= 0:
            correction = particle_inverse_weight[particle] * coefficients[contact, slot] * world_delta
            wp.atomic_add(particle_delta, particle, correction)
    body = contact_body[contact]
    if body >= 0:
        wp.atomic_add(body_delta, body, wp.transpose(body_jacobian[contact]) @ impulse_delta)


@wp.kernel
def _apply_rigid_particle_delta(
    packed_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    projection_status: wp.array[wp.int32],
    particle_delta: wp.array[wp.vec3],
    projected_velocity: wp.array[wp.vec3],
):
    particle = wp.tid()
    world = packed_world[particle]
    if world_active[world] and projection_status[world] == PROJECTION_STATUS_VALID:
        projected_velocity[particle] += particle_delta[particle]
    particle_delta[particle] = wp.vec3(0.0)


class DeformableContactSystem:
    """Own packed soft contacts and their projection buffers."""

    def __init__(
        self,
        model: Model,
        cloth_system: DeformableFEMSystem,
        contact_capacity: int,
        self_contact_capacity: int = 0,
        stabilization_fraction: float = 0.01,
        dead_zone: float = 1.0e-6,
        impact_velocity_threshold: float = 1.0e-3,
        recoverable_response: bool = False,
        enable_rigid_normal_cone_filtering: bool = False,
        normal_cone_filtering_min_distance: float = 1.0e-4,
        projection_method: str = "jacobi",
    ):
        """Allocate one fixed-capacity deformable contact path.

        Args:
            model: Newton model shared with ``cloth_system``.
            cloth_system: Packed frozen cloth system supplying scalar weights.
            contact_capacity: Maximum number of rigid-soft contact records.
            self_contact_capacity: Maximum number of cloth self-contact candidate records.
            stabilization_fraction: Penetration recovery fraction.
            dead_zone: Symmetric contact distance dead zone [m].
            impact_velocity_threshold: Minimum approaching impact speed [m/s].
            recoverable_response: Whether to permit restitution-recoverable overlap.
            enable_rigid_normal_cone_filtering: Whether to prune rigid-soft contacts using soft normal cones.
            normal_cone_filtering_min_distance: Separation below which normal-cone filtering is bypassed [m].
            projection_method: Projection method whose private scratch storage to allocate.
        """
        if cloth_system.model is not model:
            raise ValueError("LOX deformable contacts require the same model as the cloth system.")
        if not isinstance(contact_capacity, int) or isinstance(contact_capacity, bool) or contact_capacity < 0:
            raise ValueError("LOX rigid-soft contact capacity must be a non-negative integer.")
        if (
            not isinstance(self_contact_capacity, int)
            or isinstance(self_contact_capacity, bool)
            or self_contact_capacity < 0
        ):
            raise ValueError("LOX deformable self-contact capacity must be a non-negative integer.")
        total_contact_capacity = contact_capacity + self_contact_capacity
        if total_contact_capacity < 1:
            raise ValueError("LOX deformable contact capacity must be a positive integer.")
        if not np.isfinite(stabilization_fraction) or not 0.0 <= stabilization_fraction <= 1.0:
            raise ValueError("LOX deformable contact stabilization fraction must be in [0, 1].")
        if not np.isfinite(dead_zone) or dead_zone < 0.0:
            raise ValueError("LOX deformable contact dead zone must be finite and non-negative.")
        if not np.isfinite(impact_velocity_threshold) or impact_velocity_threshold < 0.0:
            raise ValueError("LOX deformable impact velocity threshold must be finite and non-negative.")
        if not isinstance(recoverable_response, bool):
            raise ValueError("LOX deformable recoverable_response must be a boolean.")
        if not isinstance(enable_rigid_normal_cone_filtering, bool):
            raise ValueError("LOX rigid-deformable normal-cone filtering flag must be a boolean.")
        if not np.isfinite(normal_cone_filtering_min_distance) or normal_cone_filtering_min_distance < 0.0:
            raise ValueError("LOX normal-cone filtering minimum distance must be finite and non-negative.")
        if projection_method not in ("jacobi", "gauss_seidel", "apgd"):
            raise ValueError("projection_method must be 'jacobi', 'gauss_seidel', or 'apgd'.")
        if contact_capacity > 0 and model.shape_count < 1:
            raise ValueError("LOX deformable contacts require at least one collider shape.")

        self.model = model
        self.cloth_system = cloth_system
        self.device = model.device
        self.rigid_contact_capacity = contact_capacity
        self.self_contact_capacity = self_contact_capacity
        self.contact_capacity = total_contact_capacity
        allocate_acceleration_storage = projection_method == "apgd"
        allocate_gauss_seidel_storage = projection_method == "gauss_seidel"
        self.stabilization_fraction = float(stabilization_fraction)
        self.dead_zone = float(dead_zone)
        self.impact_velocity_threshold = float(impact_velocity_threshold)
        self.recoverable_response = recoverable_response
        self.enable_rigid_normal_cone_filtering = enable_rigid_normal_cone_filtering
        self.normal_cone_filtering_min_distance = float(normal_cone_filtering_min_distance)
        self._empty_body_pose = wp.empty(0, dtype=wp.transform, device=self.device)
        self._empty_body_velocity = wp.empty(0, dtype=wp.spatial_vector, device=self.device)
        self._empty_body_vector = wp.empty(0, dtype=wp.vec3, device=self.device)
        self._empty_body_index = wp.empty(0, dtype=wp.int32, device=self.device)
        self._empty_body_inverse_weight = wp.empty(0, dtype=mat66f, device=self.device)
        self._empty_body_twist = wp.empty(0, dtype=vec6f, device=self.device)
        self._empty_edge_indices = wp.empty((0, 4), dtype=wp.int32, device=self.device)
        self.edge_cone_axis = wp.zeros(model.edge_count, dtype=wp.vec3, device=self.device)
        self.edge_cone_cosine = wp.full(model.edge_count, -1.0, dtype=wp.float32, device=self.device)

        self.particle_indices = wp.full(
            (self.contact_capacity, 4),
            -1,
            dtype=wp.int32,
            device=self.device,
        )
        self.coefficients = wp.zeros((self.contact_capacity, 4), dtype=wp.float32, device=self.device)
        self.contact_world = wp.full(self.contact_capacity, -1, dtype=wp.int32, device=self.device)
        self.contact_shape = wp.full(self.contact_capacity, -1, dtype=wp.int32, device=self.device)
        self.body = wp.full(self.contact_capacity, -1, dtype=wp.int32, device=self.device)
        self.frame = wp.zeros(self.contact_capacity, dtype=wp.mat33f, device=self.device)
        self.body_jacobian = wp.zeros(self.contact_capacity, dtype=mat36f, device=self.device)
        self.gap = wp.zeros(self.contact_capacity, dtype=wp.float32, device=self.device)
        self.bias = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=self.device)
        self.friction = wp.zeros(self.contact_capacity, dtype=wp.float32, device=self.device)
        self.status = wp.zeros(self.contact_capacity, dtype=wp.int32, device=self.device)
        self.delassus = wp.zeros(self.contact_capacity, dtype=wp.float32, device=self.device)
        self.reaction = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=self.device)
        self.rigid_delassus = wp.zeros(self.contact_capacity, dtype=wp.mat33f, device=self.device)
        self.gauss_seidel_scalar_delassus = (
            wp.zeros(self.contact_capacity, dtype=wp.float32, device=self.device)
            if allocate_gauss_seidel_storage
            else None
        )
        self.gauss_seidel_delassus = (
            wp.zeros(self.contact_capacity, dtype=wp.mat33f, device=self.device)
            if allocate_gauss_seidel_storage
            else None
        )
        self.acceleration_trial = (
            wp.zeros(self.contact_capacity, dtype=wp.vec3, device=self.device)
            if allocate_acceleration_storage
            else None
        )
        self.acceleration_previous = (
            wp.zeros(self.contact_capacity, dtype=wp.vec3, device=self.device)
            if allocate_acceleration_storage
            else None
        )
        self.particle_multiplicity = wp.zeros(
            cloth_system.particle_count,
            dtype=wp.int32,
            device=self.device,
        )
        self.particle_majorizer_weight_sum = wp.zeros(
            cloth_system.particle_count,
            dtype=wp.float32,
            device=self.device,
        )
        self.particle_majorizer_scale = wp.zeros(
            cloth_system.particle_count,
            dtype=wp.float32,
            device=self.device,
        )
        self.particle_delta = wp.zeros(
            cloth_system.particle_count,
            dtype=wp.vec3,
            device=self.device,
        )
        world_count = model.world_count
        self.world_contact_count = wp.zeros(world_count, dtype=wp.int32, device=self.device)
        self.projection_status = wp.zeros(world_count, dtype=wp.int32, device=self.device)
        self.world_status = wp.zeros(world_count, dtype=wp.int32, device=self.device)
        self.global_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.world_contact_residual = wp.zeros(world_count, dtype=wp.float32, device=self.device)

    def _validate_source_arrays(
        self,
        contacts: Contacts | None,
        state: State,
        body_pose: wp.array[wp.transform] | None = None,
    ) -> None:
        if self.rigid_contact_capacity > 0 and contacts is None:
            raise ValueError("LOX rigid-soft contacts require a Newton contacts container.")
        if contacts is not None and contacts.soft_contact_max < self.rigid_contact_capacity:
            raise ValueError(
                f"LOX deformable contacts require source capacity at least {self.rigid_contact_capacity}, "
                f"found {contacts.soft_contact_max}."
            )
        arrays = {
            "particle_q": state.particle_q,
            "particle_qd": state.particle_qd,
        }
        if contacts is not None:
            arrays.update(
                {
                    "soft_contact_count": contacts.soft_contact_count,
                    "soft_contact_indices": contacts.soft_contact_indices,
                    "soft_contact_barycentric": contacts.soft_contact_barycentric,
                    "soft_contact_shape": contacts.soft_contact_shape,
                    "soft_contact_body_pos": contacts.soft_contact_body_pos,
                    "soft_contact_body_vel": contacts.soft_contact_body_vel,
                    "soft_contact_normal": contacts.soft_contact_normal,
                }
            )
        if self.model.body_count > 0:
            arrays["body_q"] = body_pose if body_pose is not None else state.body_q
            arrays["body_qd"] = state.body_qd
        for name, value in arrays.items():
            if value is None:
                raise ValueError(f"LOX deformable contacts require {name}.")
            if value.device != self.device:
                raise ValueError(f"LOX deformable contacts expected {name} on {self.device}, found {value.device}.")

    def prepare(
        self,
        contacts: Contacts | None,
        state: State,
        time_step: wp.array[wp.float32],
        self_contact_detector: DeformableSelfContactDetector | None = None,
        body_pose: wp.array[wp.transform] | None = None,
    ) -> None:
        """Adapt and validate frozen Newton soft-contact records for one step.

        Numeric record failures are reported through :attr:`status` and rejected
        by the projection without a host synchronization.

        Args:
            contacts: Newton contacts containing soft particle/edge/face records.
            state: Beginning-of-step Newton state.
            time_step: Per-world simulation time steps [s].
            self_contact_detector: Optional frozen cloth self-contact detector.
            body_pose: Optional body-origin poses captured before adapting the state for another backend.
        """
        validate_world_time_step(time_step, self.model.world_count, self.device)
        self._validate_source_arrays(contacts, state, body_pose)
        body_pose = body_pose if body_pose is not None else state.body_q
        friction = float(self.model.soft_contact_mu)
        restitution = float(self.model.soft_contact_restitution)
        if not np.isfinite(friction) or friction < 0.0:
            raise ValueError("LOX deformable contact friction must be finite and non-negative.")
        if not np.isfinite(restitution) or restitution < 0.0:
            raise ValueError("LOX deformable contact restitution must be finite and non-negative.")

        self.particle_multiplicity.zero_()
        self.particle_majorizer_weight_sum.zero_()
        self.world_contact_count.zero_()
        self.world_status.zero_()
        self.global_status.zero_()
        self.reaction.zero_()
        # Full-surface records carry enough topology to reduce boundary features before cone tests.
        filter_surface_contacts = bool(
            self.enable_rigid_normal_cone_filtering
            and contacts is not None
            and contacts._enable_rigid_soft_full_surface_contact
            and self.model.tri_count > 0
            and self.model.edge_count > 0
        )
        if filter_surface_contacts and self.model.edge_count > 0:
            wp.launch(
                compute_soft_edge_normal_cones,
                dim=self.model.edge_count,
                inputs=[state.particle_q, self.model.edge_indices],
                outputs=[self.edge_cone_axis, self.edge_cone_cosine],
                device=self.device,
            )
        if contacts is not None and self.rigid_contact_capacity > 0:
            wp.launch(
                _adapt_soft_contacts,
                dim=self.rigid_contact_capacity,
                inputs=[
                    contacts.soft_contact_count,
                    self.rigid_contact_capacity,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_barycentric,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    self.model.edge_indices if self.model.edge_indices is not None else self._empty_edge_indices,
                    self.model.soft_mesh_adjacency_device,
                    self.edge_cone_axis,
                    self.edge_cone_cosine,
                    filter_surface_contacts,
                    self.normal_cone_filtering_min_distance,
                    state.particle_q,
                    state.particle_qd,
                    self.model.particle_radius,
                    self.model.particle_flags,
                    self.cloth_system.topology.newton_to_packed,
                    self.cloth_system.topology.packed_solve_world,
                    self.cloth_system.full_inverse_weight,
                    self.model.shape_body,
                    self.model.shape_world,
                    self.model.shape_margin,
                    body_pose if body_pose is not None else self._empty_body_pose,
                    state.body_qd if state.body_qd is not None else self._empty_body_velocity,
                    self.model.body_com if self.model.body_com is not None else self._empty_body_vector,
                    self.model.body_flags if self.model.body_flags is not None else self._empty_body_index,
                    self.model.body_world if self.model.body_world is not None else self._empty_body_index,
                    time_step,
                    self.stabilization_fraction,
                    self.dead_zone,
                    self.impact_velocity_threshold,
                    self.recoverable_response,
                    friction,
                    restitution,
                ],
                outputs=[
                    self.particle_indices,
                    self.coefficients,
                    self.contact_world,
                    self.contact_shape,
                    self.body,
                    self.frame,
                    self.body_jacobian,
                    self.gap,
                    self.bias,
                    self.friction,
                    self.status,
                    self.particle_multiplicity,
                    self.particle_majorizer_weight_sum,
                    self.world_contact_count,
                    self.world_status,
                    self.global_status,
                ],
                device=self.device,
            )
        if self_contact_detector is not None:
            self_contact_detector.adapt(
                self,
                state.particle_q,
                state.particle_qd,
                time_step,
                friction,
                restitution,
            )
        self._update_projection_majorizer(self.cloth_system.full_inverse_weight)

    def _update_projection_majorizer(self, inverse_weight: wp.array[float]) -> None:
        """Build the particle-side majorizer shared by parallel projections."""
        wp.launch(
            _finalize_particle_majorizer_scale,
            dim=self.cloth_system.particle_count,
            inputs=[inverse_weight, self.particle_majorizer_weight_sum],
            outputs=[self.particle_majorizer_scale],
            device=self.device,
        )
        wp.launch(
            _finalize_contact_delassus,
            dim=self.contact_capacity,
            inputs=[
                self.particle_indices,
                self.coefficients,
                self.contact_world,
                self.body,
                self.particle_majorizer_scale,
            ],
            outputs=[
                self.status,
                self.delassus,
                self.world_contact_count,
                self.world_status,
                self.global_status,
            ],
            device=self.device,
        )

    def update_weight_metric(self) -> None:
        """Refresh the shared parallel-projection majorizer after cloth reassembly."""
        self._update_projection_majorizer(self.cloth_system.inverse_weight)

    def reset(self) -> None:
        """Clear adapted-contact status and within-step reaction warm starts."""
        self.particle_multiplicity.zero_()
        self.particle_majorizer_weight_sum.zero_()
        self.world_contact_count.zero_()
        self.world_status.zero_()
        self.global_status.zero_()
        self.reaction.zero_()
        self.particle_delta.zero_()
        self.projection_status.zero_()

    def _validate_particle_velocity(self, value: wp.array[wp.vec3], name: str) -> None:
        if value.shape != (self.cloth_system.particle_count,) or value.dtype != wp.vec3:
            raise ValueError(f"LOX deformable {name} must be a vec3 array with one entry per packed particle.")
        if value.device != self.device:
            raise ValueError(f"LOX deformable contacts expected {name} on {self.device}, found {value.device}.")

    def accumulate_rigid_incidence(
        self,
        body_constraint_count: wp.array[wp.int32],
        body_has_unilateral: wp.array[wp.int32],
        world_has_unilateral: wp.array[wp.bool],
    ) -> None:
        """Merge dynamic soft contacts into rigid unilateral incidence."""
        body_count = int(self.model.body_count)
        if body_constraint_count.shape != (body_count,) or body_has_unilateral.shape != (body_count,):
            raise ValueError("LOX mixed contact body incidence must match the Newton body count.")
        if world_has_unilateral.shape != (int(self.model.world_count),):
            raise ValueError("LOX mixed contact world incidence must match the Newton world count.")
        wp.launch(
            _accumulate_rigid_incidence,
            dim=self.contact_capacity,
            inputs=[self.body, self.contact_world, self.status],
            outputs=[
                body_constraint_count,
                body_has_unilateral,
                world_has_unilateral,
            ],
            device=self.device,
        )

    def prepare_rigid_projection(
        self,
        body_constraint_count: wp.array[wp.int32],
        static_body_constraint_count: wp.array[wp.int32],
        body_inverse_weight: wp.array[mat66f],
        prepared_status: wp.array[wp.int32],
    ) -> None:
        """Prepare full anisotropic blocks for the shared rigid Jacobi sweep."""
        body_count = int(self.model.body_count)
        if (
            body_constraint_count.shape != (body_count,)
            or static_body_constraint_count.shape != (body_count,)
            or body_inverse_weight.shape != (body_count,)
        ):
            raise ValueError("LOX mixed contact rigid arrays must match the Newton body count.")
        if prepared_status.shape != (int(self.model.world_count),):
            raise ValueError("LOX mixed contact prepared status must match the Newton world count.")
        wp.launch(
            _prepare_rigid_contacts,
            dim=self.contact_capacity,
            inputs=[
                self.contact_world,
                self.body,
                self.body_jacobian,
                self.bias,
                self.friction,
                self.delassus,
                body_constraint_count,
                static_body_constraint_count,
                body_inverse_weight,
            ],
            outputs=[
                self.status,
                self.rigid_delassus,
                self.world_status,
            ],
            device=self.device,
        )
        wp.launch(
            _merge_rigid_prepared_status,
            dim=int(self.model.world_count),
            inputs=[self.world_status, self.global_status],
            outputs=[prepared_status],
            device=self.device,
        )

    def prepare_particle_projection_status(self) -> wp.array[wp.int32]:
        """Reset and validate the status used by particle-only projection."""
        status = self.projection_status
        status.fill_(PROJECTION_STATUS_VALID)
        wp.launch(
            _merge_rigid_prepared_status,
            dim=int(self.model.world_count),
            inputs=[self.world_status, self.global_status],
            outputs=[status],
            device=self.device,
        )
        return status

    def initialize_acceleration(self, world_active: wp.array[wp.bool]) -> None:
        """Project warm starts and initialize accelerated reaction fields."""
        wp.launch(
            _initialize_accelerated_contacts,
            dim=self.contact_capacity,
            inputs=[
                self.contact_world,
                self.status,
                self.friction,
                world_active,
            ],
            outputs=[self.reaction, self.acceleration_trial, self.acceleration_previous],
            device=self.device,
        )

    def accumulate_acceleration_restart(
        self,
        world_active: wp.array[wp.bool],
        projection_status: wp.array[wp.int32],
        restart_dot: wp.array[float],
    ) -> None:
        """Accumulate the safeguarded inertial restart criterion."""
        wp.launch(
            _accumulate_contact_restart,
            dim=self.contact_capacity,
            inputs=[
                self.contact_world,
                self.status,
                world_active,
                projection_status,
                self.reaction,
                self.acceleration_trial,
                self.acceleration_previous,
            ],
            outputs=[restart_dot],
            device=self.device,
        )

    def extrapolate_accelerated_reactions(
        self,
        world_active: wp.array[wp.bool],
        projection_status: wp.array[wp.int32],
        beta: wp.array[float],
        body_delta: wp.array[vec6f] | None = None,
    ) -> None:
        """Extrapolate reactions and scatter the resulting state correction."""
        delta = body_delta if body_delta is not None else self._empty_body_twist
        wp.launch(
            _extrapolate_contact_reactions,
            dim=self.contact_capacity,
            inputs=[
                self.particle_indices,
                self.coefficients,
                self.contact_world,
                self.body,
                self.frame,
                self.body_jacobian,
                self.status,
                world_active,
                projection_status,
                beta,
                self.cloth_system.inverse_weight,
                self.reaction,
                self.acceleration_trial,
                self.acceleration_previous,
            ],
            outputs=[self.particle_delta, delta],
            device=self.device,
        )

    def begin_rigid_jacobi_accumulation(self) -> None:
        """Clear the mixed-contact particle Jacobi accumulator."""
        self.particle_delta.zero_()

    def accumulate_rigid_reaction_warm_start(
        self,
        world_active: wp.array[wp.bool],
        prepared_status: wp.array[wp.int32],
        particle_inverse_weight: wp.array[float],
        body_inverse_weight: wp.array[mat66f],
        apply_inverse_weight: bool,
        particle_velocity: wp.array[wp.vec3],
        body_twist: wp.array[vec6f],
        body_delta: wp.array[vec6f],
        projection_status: wp.array[wp.int32],
    ) -> None:
        """Accumulate generalized soft-contact warm starts into both endpoints."""
        del particle_velocity, body_twist
        wp.launch(
            _warm_start_rigid_contacts,
            dim=self.contact_capacity,
            inputs=[
                self.particle_indices,
                self.coefficients,
                self.contact_world,
                self.body,
                self.frame,
                self.body_jacobian,
                self.status,
                world_active,
                prepared_status,
                particle_inverse_weight,
                body_inverse_weight,
                apply_inverse_weight,
                self.reaction,
            ],
            outputs=[
                self.particle_delta,
                body_delta,
                projection_status,
            ],
            device=self.device,
        )

    def project_rigid_jacobi(
        self,
        world_active: wp.array[wp.bool],
        particle_inverse_weight: wp.array[float],
        particle_velocity: wp.array[wp.vec3],
        body_twist: wp.array[vec6f],
        body_delta: wp.array[vec6f],
        projection_status: wp.array[wp.int32],
    ) -> None:
        """Accumulate one anisotropic generalized-contact Jacobi sweep."""
        index = _make_direct_projection_index(self.contact_world)
        contact_data = _make_projection_struct(
            _DeformableRigidContactProjectionData,
            particle_indices=self.particle_indices,
            coefficients=self.coefficients,
            body=self.body,
            frame=self.frame,
            body_jacobian=self.body_jacobian,
            delassus=self.rigid_delassus,
            bias=self.bias,
            friction=self.friction,
            reaction=self.reaction,
            status=self.status,
            contact_world_status=self.world_status,
        )
        state = _make_projection_struct(
            _DeformableRigidProjectionState,
            world_active=world_active,
            projected_twist=body_twist,
            twist_delta=body_delta,
            world_status=projection_status,
            particle_inverse_weight=particle_inverse_weight,
            projected_velocity=particle_velocity,
            particle_delta=self.particle_delta,
        )
        wp.launch(
            _make_project_contacts_kernel(
                False,
                True,
                DEFORMABLE_CONTACT_STATUS_VALID,
                DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE,
            ),
            dim=self.contact_capacity,
            inputs=[self.contact_capacity, 0, index, contact_data, state],
            device=self.device,
            block_dim=_RIGID_CONTACT_PROJECTION_BLOCK_DIM,
        )

    def apply_reaction_warm_start(self, projected_velocity: wp.array[wp.vec3]) -> None:
        """Apply stored within-step reactions to a fresh projected velocity."""
        self._validate_particle_velocity(projected_velocity, "projected velocity")
        self.particle_delta.zero_()
        wp.launch(
            _warm_start_contacts,
            dim=self.contact_capacity,
            inputs=[
                self.particle_indices,
                self.coefficients,
                self.contact_world,
                self.body,
                self.frame,
                self.status,
                self.cloth_system.world_active,
                self.world_status,
                self.global_status,
                self.cloth_system.inverse_weight,
                self.reaction,
            ],
            outputs=[self.particle_delta],
            device=self.device,
        )
        wp.launch(
            _apply_particle_delta,
            dim=self.cloth_system.particle_count,
            inputs=[
                self.cloth_system.topology.packed_solve_world,
                self.cloth_system.world_active,
                self.particle_delta,
            ],
            outputs=[projected_velocity],
            device=self.device,
        )

    def _project_particle_jacobi(self, projected_velocity: wp.array[wp.vec3]) -> None:
        self.particle_delta.zero_()
        data = _make_projection_struct(
            _ParticleContactProjectionData,
            particle_indices=self.particle_indices,
            coefficients=self.coefficients,
            contact_body=self.body,
            frame=self.frame,
            bias=self.bias,
            friction=self.friction,
            delassus=self.delassus,
            status=self.status,
            reaction=self.reaction,
        )
        state = _make_projection_struct(
            _ParticleContactDirectState,
            world_active=self.cloth_system.world_active,
            world_status=self.world_status,
            contact_world_status=self.world_status,
            global_status=self.global_status,
            inverse_weight=self.cloth_system.inverse_weight,
            projected_velocity=projected_velocity,
            particle_delta=self.particle_delta,
        )
        wp.launch(
            _make_project_particle_contacts_kernel(False),
            dim=self.contact_capacity,
            inputs=[self.contact_capacity, 0, _make_direct_projection_index(self.contact_world), data, state],
            device=self.device,
        )

    def project(
        self,
        projected_velocity: wp.array[wp.vec3],
        iterations: int = 1,
        *,
        warm_start: bool = False,
        world_active: wp.array[wp.bool] | None = None,
        theta: wp.array[float] | None = None,
        beta: wp.array[float] | None = None,
        restart_dot: wp.array[float] | None = None,
    ) -> None:
        """Run fixed-count mass-split Jacobi contact sweeps in place.

        Args:
            projected_velocity: Packed nodal velocity updated in place [m/s].
            iterations: Number of Jacobi sweeps.
            warm_start: Whether to apply the stored reactions before the first sweep.
            world_active: Optional per-world mask required by acceleration.
            theta: Optional per-world inertial schedule state.
            beta: Optional per-world extrapolation coefficient.
            restart_dot: Optional per-world restart reduction buffer.
        """
        self._validate_particle_velocity(projected_velocity, "projected velocity")
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
            raise ValueError("LOX deformable contact iterations must be a positive integer.")
        accelerated = theta is not None or beta is not None or restart_dot is not None
        if accelerated and (
            theta is None or beta is None or restart_dot is None or world_active is None or not warm_start
        ):
            raise ValueError("Accelerated deformable projection requires warm start and all acceleration arrays.")
        projection_status = self.prepare_particle_projection_status()
        if accelerated:
            self.initialize_acceleration(world_active)
            wp.launch(
                _initialize_acceleration_worlds,
                dim=int(self.model.world_count),
                inputs=[world_active, projection_status],
                outputs=[theta, beta, restart_dot, projection_status],
                device=self.device,
            )
        if warm_start:
            self.apply_reaction_warm_start(projected_velocity)
        for iteration in range(iterations):
            self._project_particle_jacobi(projected_velocity)
            wp.launch(
                _merge_rigid_prepared_status,
                dim=int(self.model.world_count),
                inputs=[self.world_status, self.global_status],
                outputs=[projection_status],
                device=self.device,
            )
            wp.launch(
                _apply_particle_delta,
                dim=self.cloth_system.particle_count,
                inputs=[
                    self.cloth_system.topology.packed_solve_world,
                    self.cloth_system.world_active,
                    self.particle_delta,
                ],
                outputs=[projected_velocity],
                device=self.device,
            )
            if accelerated and iteration + 1 < iterations:
                self.accumulate_acceleration_restart(
                    world_active,
                    projection_status,
                    restart_dot,
                )
                wp.launch(
                    _finalize_acceleration,
                    dim=int(self.model.world_count),
                    inputs=[world_active, projection_status],
                    outputs=[restart_dot, theta, beta],
                    device=self.device,
                )
                self.extrapolate_accelerated_reactions(
                    world_active,
                    projection_status,
                    beta,
                )
                wp.launch(
                    _apply_particle_delta,
                    dim=self.cloth_system.particle_count,
                    inputs=[
                        self.cloth_system.topology.packed_solve_world,
                        self.cloth_system.world_active,
                        self.particle_delta,
                    ],
                    outputs=[projected_velocity],
                    device=self.device,
                )

    def project_jacobi_smoothing_sweep(
        self,
        world_active: wp.array[wp.bool],
        projected_velocity: wp.array[wp.vec3],
        projection_status: wp.array[wp.int32],
    ) -> None:
        """Accumulate and apply one Jacobi sweep without a reaction warm start."""
        self._validate_particle_velocity(projected_velocity, "projected velocity")
        if world_active.shape != (int(self.model.world_count),) or projection_status.shape != world_active.shape:
            raise ValueError("LOX deformable smoothing status arrays must match the model world count.")
        self._project_particle_jacobi(projected_velocity)
        wp.launch(
            _merge_rigid_prepared_status,
            dim=projection_status.shape[0],
            inputs=[self.world_status, self.global_status],
            outputs=[projection_status],
            device=self.device,
        )
        wp.launch(
            _apply_rigid_particle_delta,
            dim=self.cloth_system.particle_count,
            inputs=[
                self.cloth_system.topology.packed_solve_world,
                world_active,
                projection_status,
                self.particle_delta,
            ],
            outputs=[projected_velocity],
            device=self.device,
        )

    def compute_contact_residuals(
        self,
        projected_velocity: wp.array[wp.vec3],
        projected_twist: wp.array[vec6f] | None = None,
    ) -> None:
        """Compute per-contact and per-world scaled natural-map residuals."""
        self._validate_particle_velocity(projected_velocity, "projected velocity")
        self.world_contact_residual.zero_()
        rigid_coordinates = projected_twist is not None
        if rigid_coordinates:
            if projected_twist.shape != (int(self.model.body_count),) or projected_twist.dtype != vec6f:
                raise ValueError("LOX mixed contact projected twist must contain one vec6 per body.")
        twist = projected_twist if rigid_coordinates else self._empty_body_twist
        wp.launch(
            _compute_contact_residuals,
            dim=self.contact_capacity,
            inputs=[
                self.particle_indices,
                self.coefficients,
                self.contact_world,
                self.body,
                self.frame,
                self.body_jacobian,
                self.bias,
                self.friction,
                self.delassus,
                self.rigid_delassus,
                self.status,
                self.cloth_system.world_active,
                projected_velocity,
                twist,
                self.reaction,
                rigid_coordinates,
            ],
            outputs=[self.world_contact_residual],
            device=self.device,
        )
