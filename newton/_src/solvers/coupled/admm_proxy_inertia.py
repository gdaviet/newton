# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inertial realization of the local ADMM joint quadratic.

In the force convention used by the coupler, the added spatial inertia is
``P = dt * rho * J.T W**2 J``. A point row contributes a point mass at its
anchor; an angular row contributes a tensor at the physical center of mass.
The arrays below store mass, first moment, and inertia about that center in
body coordinates. Numerical bodies carry this inertia while the physical
mass and COM stay unchanged.
"""

import warp as wp


@wp.func
def _parallel_axis(r: wp.vec3):
    return wp.dot(r, r) * wp.identity(n=3, dtype=float) - wp.outer(r, r)


@wp.kernel(enable_backward=False)
def initialize_inertia(
    gamma_mass: wp.array[float],
    gamma_inertia: wp.array[float],
    mass: wp.array[float],
    moment: wp.array[wp.vec3],
    inertia: wp.array[wp.mat33],
):
    body = wp.tid()
    mass[body] = gamma_mass[body]
    moment[body] = wp.vec3(0.0)
    inertia[body] = gamma_inertia[body] * wp.identity(n=3, dtype=float)


@wp.kernel(enable_backward=False)
def accumulate_point_inertia(
    bodies: wp.array[int],
    anchors: wp.array[wp.vec3],
    com: wp.array[wp.vec3],
    weights: wp.array[float],
    scale: float,
    mass: wp.array[float],
    moment: wp.array[wp.vec3],
    inertia: wp.array[wp.mat33],
):
    row = wp.tid()
    body = bodies[row]
    m = scale * weights[row] * weights[row]
    r = anchors[row] - com[body]
    wp.atomic_add(mass, body, m)
    wp.atomic_add(moment, body, m * r)
    wp.atomic_add(inertia, body, m * _parallel_axis(r))


@wp.kernel(enable_backward=False)
def accumulate_angular_inertia(
    bodies: wp.array[int],
    body_q: wp.array[wp.transform],
    frame_bodies: wp.array[int],
    frame_body_q: wp.array[wp.transform],
    frames: wp.array[wp.transform],
    axes: wp.vec3,
    weights: wp.array[float],
    scale: float,
    inertia: wp.array[wp.mat33],
):
    row = wp.tid()
    body = bodies[row]
    frame = frame_body_q[frame_bodies[row]] * frames[row]
    rotation = wp.quat_inverse(wp.transform_get_rotation(body_q[body])) * wp.transform_get_rotation(frame)
    R = wp.quat_to_matrix(rotation)
    tensor = R * wp.diag(axes) * wp.transpose(R)
    wp.atomic_add(inertia, body, (scale * weights[row] * weights[row]) * tensor)
