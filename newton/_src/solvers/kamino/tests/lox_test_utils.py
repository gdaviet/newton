# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared builders for functional LOX tests."""

import numpy as np
import warp as wp

import newton


def build_contact_model(
    *,
    device: wp.DeviceLike,
    world_count: int = 1,
    fix_left: bool = False,
    collider: str = "static",
    shape_margin: float = 0.02,
    body_com: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[newton.Model, list[int], list[int]]:
    """Build cloth grids and one static, kinematic, or dynamic plane per world."""
    builder = newton.ModelBuilder()
    shape_indices: list[int] = []
    body_indices: list[int] = []
    for world in range(world_count):
        builder.begin_world()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.5 + 2.0 * world),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            cell_x=1.0,
            cell_y=1.0,
            mass=1.0,
            fix_left=fix_left,
            tri_ke=100.0,
            tri_ka=80.0,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
            edge_ke=2.0,
            edge_kd=0.0,
        )
        body = -1
        if collider != "static":
            body = builder.add_body(
                xform=wp.transform_identity(),
                com=body_com,
                mass=1.0,
                inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                lock_inertia=body_com != (0.0, 0.0, 0.0),
                is_kinematic=collider == "kinematic",
            )
            body_indices.append(body)
        shape_indices.append(
            builder.add_shape_plane(
                body=body,
                cfg=newton.ModelBuilder.ShapeConfig(margin=shape_margin),
            )
        )
        builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    model.soft_contact_mu = 0.5
    model.soft_contact_restitution = 0.0
    return model, shape_indices, body_indices


def make_contacts(
    *,
    device: wp.DeviceLike,
    capacity: int,
    records: list[dict],
) -> newton.Contacts:
    """Create manually specified Newton soft-contact records."""
    contacts = newton.Contacts(0, capacity, device=device)
    indices = np.full((capacity, 3), -1, dtype=np.int32)
    barycentric = np.zeros((capacity, 3), dtype=np.float32)
    shapes = np.full(capacity, -1, dtype=np.int32)
    body_positions = np.zeros((capacity, 3), dtype=np.float32)
    body_velocities = np.zeros((capacity, 3), dtype=np.float32)
    normals = np.zeros((capacity, 3), dtype=np.float32)
    for contact, record in enumerate(records):
        indices[contact] = record["indices"]
        barycentric[contact] = record["barycentric"]
        shapes[contact] = record["shape"]
        body_positions[contact] = record["body_position"]
        body_velocities[contact] = record.get("body_velocity", (0.0, 0.0, 0.0))
        normals[contact] = record.get("normal", (0.0, 0.0, 1.0))

    contacts.soft_contact_count.assign(np.array([len(records)], dtype=np.int32))
    contacts.soft_contact_indices.assign(indices)
    contacts.soft_contact_barycentric.assign(barycentric)
    contacts.soft_contact_shape.assign(shapes)
    contacts.soft_contact_body_pos.assign(body_positions)
    contacts.soft_contact_body_vel.assign(body_velocities)
    contacts.soft_contact_normal.assign(normals)
    return contacts


def surface_position_for_gap(
    model: newton.Model,
    positions: np.ndarray,
    indices: tuple[int, int, int],
    coefficients: tuple[float, float, float],
    shape: int,
    gap: float,
    normal: np.ndarray,
) -> np.ndarray:
    """Place a raw collider point at a requested effective gap."""
    feature = np.zeros(3, dtype=np.float64)
    for particle, coefficient in zip(indices, coefficients, strict=True):
        if particle >= 0:
            feature += coefficient * positions[particle]
    radii = model.particle_radius.numpy()
    radius = max(float(radii[particle]) for particle in indices if particle >= 0)
    margin = float(model.shape_margin.numpy()[shape])
    return feature - (gap + radius + margin) * normal
