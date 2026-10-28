# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernels of the particle contacts of the LOX deformable bodies.

The loading kernel converts the soft contacts that Newton's collision pipeline
detects between particle features and collider shapes into particle contact rows:
the packed particles and interpolation coefficients of the feature, the contact
frame, the Jacobian of the collider body, the velocity bias, and the mixed
friction coefficient. The finishing kernel counts the rows that touch each
particle and body, which split their inverse weights between the rows of a
Jacobi sweep and select the bodies and particles that receive splitting weights.
"""

import warp as wp

from .......geometry.soft_contacts_mesh import _MESH_FEATURE_SIGN_UNKNOWN
from .......utils.mesh import MeshAdjacencyData
from ....core.materials import MaterialMixMode
from ....core.types import mat36f, vec6f
from ....geometry.contacts import make_contact_frame_znorm
from ..contact import compute_contact_velocity_target
from ..projection_kernels import PARTICLE_COEFFICIENT_TOLERANCE
from ..types import ParticleContactStatus
from .contact_filter import (
    CONTACT_CONE_INSIDE,
    CONTACT_CONE_OUTSIDE,
    CONTACT_CONE_PENETRATING,
    classify_surface_contact,
)

###
# Module interface
###

__all__ = [
    "_finish_particle_contacts",
    "_load_particle_contacts",
    "mix_contact_material",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


_PARTICLE_FLAG_ACTIVE = wp.constant(1)
"""Newton particle flag of the simulated particles."""


_COEFFICIENT_SUM_TOLERANCE = 1.0e-5
"""Tolerance on the unit sum of the interpolation coefficients of a contact feature."""


_NORMAL_EPSILON = 1.0e-12
"""Length below which a contact normal is degenerate."""


###
# Types
###


@wp.struct
class _SoftContactFeature:
    """Packed particles, interpolated state, and world of the particle feature of one soft contact."""

    status: wp.int32
    world: wp.int32
    particle: wp.vec3i
    coefficient: wp.vec3f
    position: wp.vec3f
    velocity: wp.vec3f
    prescribed_velocity: wp.vec3f
    radius: wp.float32


@wp.struct
class _SoftContactCollider:
    """Contact point, velocity, and body of the collider shape of one soft contact."""

    status: wp.int32
    body: wp.int32
    position: wp.vec3f
    velocity: wp.vec3f
    surface_velocity: wp.vec3f
    center_of_mass: wp.vec3f


###
# Functions
###


@wp.func
def mix_contact_material(value_a: wp.float32, value_b: wp.float32, mix_mode: wp.int32) -> wp.float32:
    """Mix two material coefficients with a :class:`MaterialMixMode`."""
    if mix_mode == MaterialMixMode.MULTIPLY:
        return value_a * value_b
    if mix_mode == MaterialMixMode.MAX:
        return wp.max(value_a, value_b)
    if mix_mode == MaterialMixMode.MIN:
        return wp.min(value_a, value_b)
    return 0.5 * (value_a + value_b)


@wp.func
def _prepare_soft_contact_feature(
    source_particles: wp.vec3i,
    source_coefficients: wp.vec3f,
    particle_q: wp.array[wp.vec3f],
    particle_qd: wp.array[wp.vec3f],
    particle_radius: wp.array[wp.float32],
    particle_flags: wp.array[wp.int32],
    newton_to_packed: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    particle_inverse_weight: wp.array[wp.float32],
) -> _SoftContactFeature:
    """Validate the particle feature of one soft contact and map it to packed particles.

    The velocity of the prescribed particles, which have zero inverse weight, enters the
    feature velocity through :attr:`_SoftContactFeature.prescribed_velocity`.
    """
    result = _SoftContactFeature()
    result.status = ParticleContactStatus.ACTIVE
    result.world = -1
    result.particle = wp.vec3i(-1)
    result.coefficient = wp.vec3f(0.0)
    result.position = wp.vec3f(0.0)
    result.velocity = wp.vec3f(0.0)
    result.prescribed_velocity = wp.vec3f(0.0)
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
            malformed = malformed or source_particle != -1 or wp.abs(coefficient) > PARTICLE_COEFFICIENT_TOLERANCE
        else:
            malformed = (
                malformed
                or found_padding
                or source_particle >= newton_to_packed.shape[0]
                or not wp.isfinite(coefficient)
                or coefficient < -PARTICLE_COEFFICIENT_TOLERANCE
            )
            for previous_slot in range(slot):
                malformed = malformed or source_particles[previous_slot] == source_particle
            if not malformed:
                pid = newton_to_packed[source_particle]
                wid = particle_world[pid]
                if result.world < 0:
                    result.world = wid
                elif wid != result.world:
                    result.status = ParticleContactStatus.CROSS_WORLD
                    return result
                result.particle[slot] = pid
                result.coefficient[slot] = coefficient
                result.position += coefficient * particle_q[source_particle]
                if (particle_flags[source_particle] & _PARTICLE_FLAG_ACTIVE) != 0:
                    result.velocity += coefficient * particle_qd[source_particle]
                    if particle_inverse_weight[pid] <= 0.0:
                        result.prescribed_velocity += coefficient * particle_qd[source_particle]
                result.radius = wp.max(result.radius, particle_radius[source_particle])
                coefficient_sum += coefficient
                slot_count += 1

    if (
        malformed
        or slot_count == 0
        or result.world < 0
        or wp.abs(coefficient_sum - 1.0) > _COEFFICIENT_SUM_TOLERANCE
        or not wp.isfinite(result.position)
        or not wp.isfinite(result.velocity)
        or not wp.isfinite(result.radius)
        or result.radius < 0.0
    ):
        result.status = ParticleContactStatus.MALFORMED
    return result


@wp.func
def _prepare_soft_contact_collider(
    cid: wp.int32,
    shape: wp.int32,
    wid: wp.int32,
    contacts_body_pos: wp.array[wp.vec3f],
    contacts_body_vel: wp.array[wp.vec3f],
    shape_body: wp.array[wp.int32],
    model_bodies_wid: wp.array[wp.int32],
    body_com: wp.array[wp.vec3f],
    data_bodies_q_i: wp.array[wp.transformf],
    data_bodies_u_i: wp.array[vec6f],
) -> _SoftContactCollider:
    """Return the world contact point and velocity of the collider shape of one soft contact.

    The contact point and surface velocity of a body shape are given in the body origin
    frame; the body pose and twist are Kamino's, about the center of mass.
    """
    result = _SoftContactCollider()
    result.status = ParticleContactStatus.ACTIVE
    result.body = -1
    result.position = contacts_body_pos[cid]
    result.velocity = contacts_body_vel[cid]
    result.surface_velocity = result.velocity
    result.center_of_mass = wp.vec3f(0.0)

    bid = shape_body[shape]
    if bid < 0:
        return result
    if bid >= data_bodies_q_i.shape[0]:
        result.status = ParticleContactStatus.MALFORMED
        return result
    if model_bodies_wid[bid] != wid:
        result.status = ParticleContactStatus.CROSS_WORLD
        return result

    pose = data_bodies_q_i[bid]
    rotation = wp.transform_get_rotation(pose)
    result.center_of_mass = wp.transform_get_translation(pose)
    result.position = result.center_of_mass + wp.quat_rotate(rotation, contacts_body_pos[cid] - body_com[bid])
    result.surface_velocity = wp.quat_rotate(rotation, contacts_body_vel[cid])
    twist = data_bodies_u_i[bid]
    linear_velocity = wp.vec3f(twist[0], twist[1], twist[2])
    angular_velocity = wp.vec3f(twist[3], twist[4], twist[5])
    result.velocity = (
        linear_velocity + wp.cross(angular_velocity, result.position - result.center_of_mass) + result.surface_velocity
    )
    result.body = bid
    return result


@wp.func
def _classify_collider_contact(
    shape: wp.int32,
    body: wp.int32,
    body_point: wp.vec3f,
    normal: wp.vec3f,
    data_bodies_q_i: wp.array[wp.transformf],
    shape_transform: wp.array[wp.transformf],
    shape_scale: wp.array[wp.vec3f],
    shape_source_ptr: wp.array[wp.uint64],
    shape_tri_offset: wp.array[wp.int32],
    collider_position: wp.array[wp.vec3f],
    collider_tri_indices: wp.array2d[wp.int32],
    collider_edge_indices: wp.array2d[wp.int32],
    collider_adjacency: MeshAdjacencyData,
    collider_face_side: wp.array[wp.float32],
):
    """Classify the contact normal against the collider mesh feature of a rigid-soft contact point.

    The collider point is located on its mesh in the scaled shape frame. Shapes without a collider
    mesh admit every direction.

    Returns:
        The cone verdict, and for a concave or penetrated feature its outward face normal in world frame.
    """
    tri_offset = shape_tri_offset[shape]
    if tri_offset < 0:
        return CONTACT_CONE_INSIDE, wp.vec3f(0.0)
    X_bs = shape_transform[shape]
    rotation = wp.transform_get_rotation(X_bs)
    if body >= 0:
        rotation = wp.transform_get_rotation(data_bodies_q_i[body]) * rotation
    local_point = wp.transform_point(wp.transform_inverse(X_bs), body_point)
    query = wp.mesh_query_point_no_sign(shape_source_ptr[shape], wp.cw_div(local_point, shape_scale[shape]), 1.0e6)
    if not query.result:
        return CONTACT_CONE_INSIDE, wp.vec3f(0.0)
    face = tri_offset + query.face
    verdict, face_normal = classify_surface_contact(
        wp.vec3i(collider_tri_indices[face, 0], collider_tri_indices[face, 1], collider_tri_indices[face, 2]),
        wp.vec3(query.u, query.v, 1.0 - query.u - query.v),
        wp.quat_rotate_inv(rotation, normal),
        collider_position,
        collider_tri_indices,
        collider_edge_indices,
        collider_adjacency,
        collider_face_side,
    )
    return verdict, wp.quat_rotate(rotation, face_normal)


###
# Kernels
###


@wp.kernel
def _load_particle_contacts(
    # Inputs:
    contacts_soft_contact_count: wp.array[wp.int32],
    contacts_soft_contact_indices: wp.array[wp.vec3i],
    contacts_soft_contact_barycentric: wp.array[wp.vec3f],
    contacts_soft_contact_shape: wp.array[wp.int32],
    contacts_soft_contact_body_pos: wp.array[wp.vec3f],
    contacts_soft_contact_body_vel: wp.array[wp.vec3f],
    contacts_soft_contact_normal: wp.array[wp.vec3f],
    contacts_soft_contact_mesh_feature: wp.array[wp.vec3i],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    face_side: wp.array[wp.float32],
    collider_position: wp.array[wp.vec3f],
    collider_tri_indices: wp.array2d[wp.int32],
    collider_edge_indices: wp.array2d[wp.int32],
    collider_adjacency: MeshAdjacencyData,
    collider_face_side: wp.array[wp.float32],
    shape_tri_offset: wp.array[wp.int32],
    shape_source_ptr: wp.array[wp.uint64],
    shape_transform: wp.array[wp.transformf],
    shape_scale: wp.array[wp.vec3f],
    filter_surface_contacts: wp.bool,
    normal_cone_filtering_min_distance: wp.float32,
    particle_q: wp.array[wp.vec3f],
    particle_qd: wp.array[wp.vec3f],
    particle_radius: wp.array[wp.float32],
    particle_flags: wp.array[wp.int32],
    newton_to_packed: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    particle_inverse_weight: wp.array[wp.float32],
    shape_body: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    shape_margin: wp.array[wp.float32],
    shape_material_mu: wp.array[wp.float32],
    shape_material_restitution: wp.array[wp.float32],
    model_bodies_wid: wp.array[wp.int32],
    body_com: wp.array[wp.vec3f],
    data_bodies_q_i: wp.array[wp.transformf],
    data_bodies_u_i: wp.array[vec6f],
    model_time_dt: wp.array[wp.float32],
    stabilization_fraction: wp.float32,
    dead_zone: wp.float32,
    soft_friction: wp.float32,
    soft_restitution: wp.float32,
    friction_mix_mode: wp.int32,
    restitution_mix_mode: wp.int32,
    # Outputs:
    row_world: wp.array[wp.int32],
    row_status: wp.array[wp.int32],
    row_particle: wp.array2d[wp.int32],
    row_coefficient: wp.array2d[wp.float32],
    row_body: wp.array[wp.int32],
    row_shape: wp.array[wp.int32],
    row_frame: wp.array[wp.mat33f],
    row_body_jacobian: wp.array[mat36f],
    row_gap: wp.array[wp.float32],
    row_bias: wp.array[wp.vec3f],
    row_friction: wp.array[wp.float32],
    row_reaction: wp.array[wp.vec3f],
):
    """Load each soft contact detected by Newton into its particle contact row.

    Malformed records leave their row unused; a contact between worlds fails its world
    when the projection blocks are prepared. With ``filter_surface_contacts``, a contact whose
    direction leaves the normal cone of the particle feature or of the collider mesh feature is
    filtered out, and a contact on a concave feature takes the face normal of that feature.
    """
    cid = wp.tid()
    row_world[cid] = -1
    row_status[cid] = ParticleContactStatus.UNUSED
    # The consumers read the rows with an active status only, so the slots past the records keep
    # their other fields
    if cid >= wp.min(contacts_soft_contact_count[0], contacts_soft_contact_indices.shape[0]):
        return
    for slot in range(4):
        row_particle[cid, slot] = -1
        row_coefficient[cid, slot] = 0.0
    row_shape[cid] = -1
    row_body[cid] = -1
    row_frame[cid] = wp.mat33f(0.0)
    row_body_jacobian[cid] = mat36f(0.0)
    row_gap[cid] = 0.0
    row_bias[cid] = wp.vec3f(0.0)
    row_friction[cid] = 0.0
    row_reaction[cid] = wp.vec3f(0.0)

    # Full-surface mesh records with an unresolved inside/outside sign or without an active
    # particle serve penetration prevention only, so their rows stay unused
    if cid < contacts_soft_contact_mesh_feature.shape[0]:
        mesh_feature = contacts_soft_contact_mesh_feature[cid][0]
        if mesh_feature >= 0:
            if (mesh_feature & _MESH_FEATURE_SIGN_UNKNOWN) != 0:
                return
            active = int(0)
            for slot in range(3):
                source_particle = contacts_soft_contact_indices[cid][slot]
                if source_particle >= 0 and source_particle < particle_flags.shape[0]:
                    active |= particle_flags[source_particle] & _PARTICLE_FLAG_ACTIVE
            if active == 0:
                return

    shape = contacts_soft_contact_shape[cid]
    if (
        shape < 0
        or shape >= shape_body.shape[0]
        or not wp.isfinite(contacts_soft_contact_body_pos[cid])
        or not wp.isfinite(contacts_soft_contact_body_vel[cid])
        or not wp.isfinite(contacts_soft_contact_normal[cid])
    ):
        row_status[cid] = ParticleContactStatus.MALFORMED
        return

    source_particles = contacts_soft_contact_indices[cid]
    source_coefficients = contacts_soft_contact_barycentric[cid]
    feature = _prepare_soft_contact_feature(
        source_particles,
        source_coefficients,
        particle_q,
        particle_qd,
        particle_radius,
        particle_flags,
        newton_to_packed,
        particle_world,
        particle_inverse_weight,
    )
    for slot in range(3):
        row_particle[cid, slot] = feature.particle[slot]
        row_coefficient[cid, slot] = feature.coefficient[slot]
    row_world[cid] = feature.world
    if feature.status != ParticleContactStatus.ACTIVE:
        row_status[cid] = feature.status
        return

    normal_length = wp.length(contacts_soft_contact_normal[cid])
    if not wp.isfinite(normal_length) or normal_length <= _NORMAL_EPSILON:
        row_status[cid] = ParticleContactStatus.MALFORMED
        return
    normal = contacts_soft_contact_normal[cid] / normal_length
    collider_world = shape_world[shape]
    if collider_world >= 0 and collider_world != feature.world:
        row_status[cid] = ParticleContactStatus.CROSS_WORLD
        return

    collider = _prepare_soft_contact_collider(
        cid,
        shape,
        feature.world,
        contacts_soft_contact_body_pos,
        contacts_soft_contact_body_vel,
        shape_body,
        model_bodies_wid,
        body_com,
        data_bodies_q_i,
        data_bodies_u_i,
    )
    margin = shape_margin[shape]
    friction = mix_contact_material(soft_friction, shape_material_mu[shape], friction_mix_mode)
    restitution = mix_contact_material(soft_restitution, shape_material_restitution[shape], restitution_mix_mode)
    if collider.status != ParticleContactStatus.ACTIVE:
        row_status[cid] = collider.status
        return
    if (
        not wp.isfinite(collider.position)
        or not wp.isfinite(collider.velocity)
        or not wp.isfinite(margin)
        or margin < 0.0
        or not wp.isfinite(friction)
        or friction < 0.0
        or not wp.isfinite(restitution)
        or restitution < 0.0
    ):
        row_status[cid] = ParticleContactStatus.MALFORMED
        return

    # At nearly coincident closest points, small positional errors can rotate the derived
    # normal arbitrarily, and a feature inside the collider recovers along its contact
    # normal, so the filter keeps those contacts.
    separation = feature.position - collider.position
    if (
        filter_surface_contacts
        and wp.length(separation) > normal_cone_filtering_min_distance
        and wp.dot(normal, separation) > 0.0
    ):
        particle_verdict, particle_face_normal = classify_surface_contact(
            source_particles, source_coefficients, -normal, particle_q, tri_indices, edge_indices, adjacency, face_side
        )
        if particle_verdict == CONTACT_CONE_OUTSIDE:
            row_status[cid] = ParticleContactStatus.UNUSED
            return
        collider_verdict, collider_face_normal = _classify_collider_contact(
            shape,
            collider.body,
            contacts_soft_contact_body_pos[cid],
            normal,
            data_bodies_q_i,
            shape_transform,
            shape_scale,
            shape_source_ptr,
            shape_tri_offset,
            collider_position,
            collider_tri_indices,
            collider_edge_indices,
            collider_adjacency,
            collider_face_side,
        )
        # A collider point behind the particle surface recovers along its outward normal
        if collider_verdict == CONTACT_CONE_OUTSIDE and particle_verdict != CONTACT_CONE_PENETRATING:
            row_status[cid] = ParticleContactStatus.UNUSED
            return
        # A concave or penetrated feature presses along its face normal
        if particle_verdict != CONTACT_CONE_INSIDE:
            normal = -particle_face_normal
        elif collider_verdict != CONTACT_CONE_INSIDE:
            normal = collider_face_normal

    dt = model_time_dt[feature.world]
    gap = wp.dot(normal, feature.position - collider.position) - feature.radius - margin
    previous_normal_velocity = wp.dot(normal, feature.velocity - collider.velocity)
    velocity_target = compute_contact_velocity_target(
        gap, previous_normal_velocity, restitution, dt, stabilization_fraction, dead_zone
    )
    frame = make_contact_frame_znorm(normal)
    frame_transpose = wp.transpose(frame)
    # The collider body enters through its Jacobian, and its surface velocity through the bias
    body_jacobian = mat36f(0.0)
    collider_velocity = collider.velocity
    if collider.body >= 0:
        collider_velocity = collider.surface_velocity
        angular_jacobian = frame_transpose @ wp.skew(collider.position - collider.center_of_mass)
        for row in range(3):
            for col in range(3):
                body_jacobian[row, col] = -frame_transpose[row, col]
                body_jacobian[row, 3 + col] = angular_jacobian[row, col]
    bias = frame_transpose @ (feature.prescribed_velocity - collider_velocity - velocity_target * normal)
    if not wp.isfinite(gap) or not wp.isfinite(bias):
        row_status[cid] = ParticleContactStatus.MALFORMED
        return

    row_shape[cid] = shape
    row_body[cid] = collider.body
    row_frame[cid] = frame
    row_body_jacobian[cid] = body_jacobian
    row_gap[cid] = gap
    row_bias[cid] = bias
    row_friction[cid] = friction
    row_status[cid] = ParticleContactStatus.ACTIVE


@wp.kernel
def _finish_particle_contacts(
    # Inputs:
    row_world: wp.array[wp.int32],
    row_particle: wp.array2d[wp.int32],
    row_coefficient: wp.array2d[wp.float32],
    row_body: wp.array[wp.int32],
    body_vector_index: wp.array[wp.int32],
    particle_inverse_weight: wp.array[wp.float32],
    # Outputs:
    row_status: wp.array[wp.int32],
    world_row_count: wp.array[wp.int32],
    body_incidence: wp.array2d[wp.int32],
    particle_incidence: wp.array2d[wp.int32],
    particle_weight_sum: wp.array2d[wp.float32],
    particle_unilateral_count: wp.array[wp.int32],
):
    """Retire the rows between prescribed endpoints and count the active rows touching each body and particle.

    The particle incidence counts every particle of a row, which receives its corrections. The sums
    of the absolute coefficients of each particle split its inverse weight between the rows of a
    Jacobi sweep, and the unilateral count of the dynamic particles with a significant coefficient
    enables their consensus weight.
    """
    row = wp.tid()
    if row_status[row] != ParticleContactStatus.ACTIVE:
        return
    bid = row_body[row]
    coupled = bid >= 0 and body_vector_index[bid] >= 0
    for slot in range(4):
        pid = row_particle[row, slot]
        if pid >= 0 and wp.abs(row_coefficient[row, slot]) > PARTICLE_COEFFICIENT_TOLERANCE:
            coupled = coupled or particle_inverse_weight[pid] > 0.0
    if not coupled:
        row_status[row] = ParticleContactStatus.PRESCRIBED
        return
    wp.atomic_add(world_row_count, row_world[row], 1)
    if bid >= 0:
        wp.atomic_add(body_incidence, bid, 0, 1)
    for slot in range(4):
        pid = row_particle[row, slot]
        unique = pid >= 0
        for previous in range(slot):
            unique = unique and row_particle[row, previous] != pid
        if unique:
            wp.atomic_add(particle_incidence, pid, 0, 1)
        weight = wp.abs(row_coefficient[row, slot])
        if pid >= 0 and weight > PARTICLE_COEFFICIENT_TOLERANCE:
            wp.atomic_add(particle_weight_sum, pid, 0, weight)
            if particle_inverse_weight[pid] > 0.0:
                wp.atomic_add(particle_unilateral_count, pid, 1)
