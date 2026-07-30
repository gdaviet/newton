# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
import warp.sparse as sp

from .apgd_solver_kernels import APGD_STATE_BETA, APGD_STATE_STEP_SIZE

wp.set_module_options({"enable_backward": False})


@wp.kernel
def compute_collider_inv_mass(
    J_mat_offsets: wp.array[int],
    J_mat_columns: wp.array[int],
    J_mat_values: wp.array[wp.mat33],
    IJtm_mat_offsets: wp.array[int],
    IJtm_mat_columns: wp.array[int],
    IJtm_mat_values: wp.array[wp.mat33],
    collider_inv_mass: wp.array[float],
):
    i = wp.tid()

    block_beg = J_mat_offsets[i]
    block_end = J_mat_offsets[i + 1]

    w_mat = wp.mat33(0.0)

    for b in range(block_beg, block_end):
        col = J_mat_columns[b]
        transposed_block = sp.bsr_block_index(col, i, IJtm_mat_offsets, IJtm_mat_columns)
        if transposed_block == -1:
            continue

        # Mass-splitting: divide by number of nodes overlapping with this body
        multiplicity = float(IJtm_mat_offsets[col + 1] - IJtm_mat_offsets[col])

        w_mat += (J_mat_values[b] @ IJtm_mat_values[transposed_block]) * multiplicity

    _eigvecs, eigvals = wp.eig3(w_mat)
    collider_inv_mass[i] = wp.max(0.0, wp.max(eigvals))


@wp.func
def project_on_friction_cone(
    mu: float,
    nor: wp.vec3,
    r: wp.vec3,
):
    """Projects a stress vector ``r`` onto the Coulomb friction cone (non-orthogonally)."""

    r_n = wp.dot(r, nor)
    r_t = r - r_n * nor

    r_n = wp.max(0.0, r_n)
    mu_rn = mu * r_n

    r_t_n2 = wp.length_sq(r_t)
    if r_t_n2 > mu_rn * mu_rn:
        r_t *= mu_rn / wp.sqrt(r_t_n2)

    return r_n * nor + r_t


@wp.func
def project_on_friction_cone_orthogonal(
    mu: float,
    nor: wp.vec3,
    r: wp.vec3,
):
    """Projects a vector ``r`` orthogonally onto the Coulomb friction cone."""

    mu = wp.max(0.0, mu)
    r_n = wp.dot(r, nor)
    r_t = r - r_n * nor
    r_t_n = wp.length(r_t)

    if r_n + mu * r_t_n <= 0.0:
        return wp.vec3(0.0)

    if r_t_n <= mu * r_n:
        return r

    projected_n = (r_n + mu * r_t_n) / (1.0 + mu * mu)
    projected_t = wp.vec3(0.0)
    if r_t_n > 0.0:
        projected_t = (mu * projected_n / r_t_n) * r_t

    return projected_n * nor + projected_t


@wp.func
def project_collider_impulse_orthogonal(
    friction: float,
    nor: wp.vec3,
    adhesion: float,
    impulse: wp.vec3,
):
    """Projects a collider impulse orthogonally onto its adhesive friction cone."""

    if friction < 0.0:
        return wp.vec3(0.0)

    shifted_impulse = impulse + adhesion * nor
    return project_on_friction_cone_orthogonal(friction, nor, shifted_impulse) - adhesion * nor


@wp.func
def apply_contact_desaxce_correction(
    friction: float,
    nor: wp.vec3,
    velocity: wp.vec3,
):
    """Apply the de Saxcé correction to a relative contact velocity."""

    normal_velocity = wp.dot(velocity, nor)
    tangential_velocity = velocity - normal_velocity * nor
    return velocity + wp.max(0.0, friction) * wp.length(tangential_velocity) * nor


@wp.kernel
def apgd_project_collider_impulse(
    collider_friction: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    collider_adhesion: wp.array[float],
    collider_impulse: wp.array[wp.vec3],
):
    """Project APGD collider-impulse warm starts onto their admissible sets."""

    i = wp.tid()
    collider_impulse[i] = project_collider_impulse_orthogonal(
        collider_friction[i],
        collider_normals[i],
        collider_adhesion[i],
        collider_impulse[i],
    )


@wp.kernel
def apgd_apply_nodal_impulse(
    inv_mass: wp.array[float],
    collider_impulse: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
):
    """Apply a complete nodal APGD impulse field to a baseline velocity."""

    i = wp.tid()
    velocity[i] += inv_mass[i] * collider_impulse[i]


@wp.kernel
def apgd_compute_nodal_delassus_diagonal(
    inv_mass: wp.array[float],
    collider_inv_mass: wp.array[float],
    collider_delassus_diagonal: wp.array[float],
):
    """Compute scalar nodal contact preconditioners."""

    i = wp.tid()
    collider_delassus_diagonal[i] = inv_mass[i] + collider_inv_mass[i]


@wp.func
def apgd_project_collider_step(
    step_size: float,
    effective_inv_mass: float,
    friction: float,
    adhesion: float,
    normal: wp.vec3,
    relative_velocity: wp.vec3,
    collider_impulse: wp.vec3,
):
    """Project one preconditioned, de Saxcé-corrected collider impulse step."""

    if friction < 0.0:
        return wp.vec3(0.0)
    if effective_inv_mass <= 0.0:
        return collider_impulse

    corrected_velocity = apply_contact_desaxce_correction(friction, normal, relative_velocity)
    trial_impulse = collider_impulse - (step_size / effective_inv_mass) * corrected_velocity
    return project_collider_impulse_orthogonal(
        friction,
        normal,
        adhesion,
        trial_impulse,
    )


@wp.kernel
def apgd_evaluate_nodal_contact(
    velocity: wp.array[wp.vec3],
    collider_velocity: wp.array[wp.vec3],
    response: wp.array[wp.vec3],
):
    """Evaluate the raw nodal relative-velocity response."""

    i = wp.tid()
    response[i] = velocity[i] - collider_velocity[i]


@wp.kernel
def apgd_evaluate_subgrid_contact(
    velocity: wp.array[wp.vec3],
    collider_mat_offsets: wp.array[int],
    collider_mat_columns: wp.array[int],
    collider_mat_values: wp.array[float],
    collider_velocity: wp.array[wp.vec3],
    response: wp.array[wp.vec3],
):
    """Evaluate the raw subgrid relative-velocity response."""

    i = wp.tid()
    relative_velocity = -collider_velocity[i]
    for b in range(collider_mat_offsets[i], collider_mat_offsets[i + 1]):
        relative_velocity += collider_mat_values[b] * velocity[collider_mat_columns[b]]
    response[i] = relative_velocity


@wp.kernel
def apgd_project_contact_response(
    state: wp.array2d[float],
    collider_environment: wp.array[int],
    condition: wp.array[int],
    collider_delassus_diagonal: wp.array[float],
    collider_friction: wp.array[float],
    collider_adhesion: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    response: wp.array[wp.vec3],
    extrapolated_collider_impulse: wp.array[wp.vec3],
    collider_impulse: wp.array[wp.vec3],
    next_collider_impulse: wp.array[wp.vec3],
):
    """Project one contact step from a previously evaluated raw response."""

    i = wp.tid()
    if condition[0] == 0:
        next_collider_impulse[i] = collider_impulse[i]
        return

    environment = collider_environment[i]
    next_collider_impulse[i] = apgd_project_collider_step(
        state[APGD_STATE_STEP_SIZE, environment],
        collider_delassus_diagonal[i],
        collider_friction[i],
        collider_adhesion[i],
        collider_normals[i],
        response[i],
        extrapolated_collider_impulse[i],
    )


@wp.kernel
def apgd_extrapolate_collider_impulse(
    state: wp.array2d[float],
    collider_environment: wp.array[int],
    condition: wp.array[int],
    collider_impulse: wp.array[wp.vec3],
    next_collider_impulse: wp.array[wp.vec3],
    extrapolated_collider_impulse: wp.array[wp.vec3],
):
    """Extrapolate a complete collider-impulse field from feasible iterates."""

    i = wp.tid()
    if condition[0] == 0:
        extrapolated_collider_impulse[i] = collider_impulse[i]
        return

    next_impulse = next_collider_impulse[i]
    beta = state[APGD_STATE_BETA, collider_environment[i]]
    extrapolated_collider_impulse[i] = next_impulse + beta * (next_impulse - collider_impulse[i])


@wp.kernel
def apgd_compute_contact_restart_metrics(
    state: wp.array2d[float],
    condition: wp.array[int],
    diagnostic_step_size: float,
    collider_delassus_diagonal: wp.array[float],
    collider_friction: wp.array[float],
    collider_adhesion: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    response: wp.array[wp.vec3],
    extrapolated_collider_impulse: wp.array[wp.vec3],
    collider_impulse: wp.array[wp.vec3],
    next_collider_impulse: wp.array[wp.vec3],
    metrics: wp.array2d[float],
):
    """Compute contact restart and projected fixed-point residual terms."""

    i = wp.tid()
    if condition[0] == 0 or collider_friction[i] < 0.0:
        metrics[0, i] = 0.0
        metrics[1, i] = 0.0
        metrics[2, i] = 0.0
        return

    normal = collider_normals[i]
    direction = -apply_contact_desaxce_correction(collider_friction[i], normal, response[i])
    feasible_delta = next_collider_impulse[i] - collider_impulse[i]

    restart = wp.dot(feasible_delta, direction)
    residual = float(0.0)
    preconditioner = collider_delassus_diagonal[i]
    if preconditioner > 0.0 and diagnostic_step_size > 0.0:
        diagnostic_impulse = apgd_project_collider_step(
            diagnostic_step_size,
            preconditioner,
            collider_friction[i],
            collider_adhesion[i],
            normal,
            response[i],
            extrapolated_collider_impulse[i],
        )
        fixed_point_delta = diagnostic_impulse - extrapolated_collider_impulse[i]
        residual = preconditioner * wp.length_sq(fixed_point_delta) / (diagnostic_step_size * diagnostic_step_size)
    metrics[0, i] = restart
    metrics[1, i] = residual
    metrics[2, i] = residual


@wp.kernel
def apgd_compute_contact_bb_metrics(
    condition: wp.array[int],
    collider_delassus_diagonal: wp.array[float],
    collider_friction: wp.array[float],
    previous_collider_impulse: wp.array[wp.vec3],
    collider_impulse: wp.array[wp.vec3],
    previous_response: wp.array[wp.vec3],
    response: wp.array[wp.vec3],
    metrics: wp.array2d[float],
):
    """Compute raw-response contact contributions to the scaled BB step."""

    i = wp.tid()
    if condition[0] == 0 or collider_friction[i] < 0.0:
        metrics[0, i] = 0.0
        metrics[1, i] = 0.0
        metrics[2, i] = 0.0
        return

    delta_impulse = collider_impulse[i] - previous_collider_impulse[i]
    delta_response = response[i] - previous_response[i]
    numerator = wp.dot(delta_impulse, delta_response)
    denominator = float(0.0)
    preconditioner = collider_delassus_diagonal[i]
    if preconditioner > 0.0:
        denominator = wp.length_sq(delta_response) / preconditioner
    metrics[0, i] = numerator
    metrics[1, i] = denominator
    metrics[2, i] = 0.0


@wp.kernel
def apgd_step_nodal_contact(
    step_size: float,
    collider_delassus_diagonal: wp.array[float],
    collider_friction: wp.array[float],
    collider_adhesion: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    collider_velocity: wp.array[wp.vec3],
    collider_impulse: wp.array[wp.vec3],
    next_collider_impulse: wp.array[wp.vec3],
):
    """Take one preconditioned, de Saxcé-corrected nodal APGD contact step."""

    i = wp.tid()
    normal = collider_normals[i]
    relative_velocity = velocity[i] - collider_velocity[i]
    next_collider_impulse[i] = apgd_project_collider_step(
        step_size,
        collider_delassus_diagonal[i],
        collider_friction[i],
        collider_adhesion[i],
        normal,
        relative_velocity,
        collider_impulse[i],
    )


@wp.kernel
def apgd_step_subgrid_contact(
    step_size: float,
    velocity: wp.array[wp.vec3],
    collider_mat_offsets: wp.array[int],
    collider_mat_columns: wp.array[int],
    collider_mat_values: wp.array[float],
    collider_delassus_diagonal: wp.array[float],
    collider_friction: wp.array[float],
    collider_adhesion: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    collider_velocity: wp.array[wp.vec3],
    collider_impulse: wp.array[wp.vec3],
    next_collider_impulse: wp.array[wp.vec3],
):
    """Take one preconditioned, de Saxcé-corrected subgrid APGD contact step."""

    i = wp.tid()
    relative_velocity = -collider_velocity[i]
    for b in range(collider_mat_offsets[i], collider_mat_offsets[i + 1]):
        relative_velocity += collider_mat_values[b] * velocity[collider_mat_columns[b]]

    next_collider_impulse[i] = apgd_project_collider_step(
        step_size,
        collider_delassus_diagonal[i],
        collider_friction[i],
        collider_adhesion[i],
        collider_normals[i],
        relative_velocity,
        collider_impulse[i],
    )


@wp.func
def solve_coulomb_isotropic(
    mu: float,
    nor: wp.vec3,
    u: wp.vec3,
):
    """Solves for the relative velocity in the Coulomb friction model,
    assuming an isotropic velocity-impulse relationship, u = r + b
    """

    u_n = wp.dot(u, nor)
    if u_n < 0.0:
        u -= u_n * nor
        tau = wp.length_sq(u)
        alpha = mu * u_n
        if tau <= alpha * alpha:
            u = wp.vec3(0.0)
        else:
            u *= 1.0 + mu * u_n / wp.sqrt(tau)

    return u


@wp.func
def filter_collider_impulse_warmstart(
    friction: float,
    nor: wp.vec3,
    adhesion: float,
    impulse: wp.vec3,
):
    """Filters the collider impulse to be within the friction cone"""

    if friction < 0.0:
        return wp.vec3(0.0)

    return project_on_friction_cone(friction, nor, impulse + adhesion * nor) - adhesion * nor


@wp.kernel
def apply_nodal_impulse_warmstart(
    collider_impulse: wp.array[wp.vec3],
    collider_friction: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    collider_adhesion: wp.array[float],
    inv_mass: wp.array[float],
    velocities: wp.array[wp.vec3],
    delta_impulse: wp.array[wp.vec3],
):
    """
    Applies pre-computed impulses to particles and colliders.
    """
    i = wp.tid()

    impulse = filter_collider_impulse_warmstart(
        collider_friction[i], collider_normals[i], collider_adhesion[i], collider_impulse[i]
    )

    collider_impulse[i] = impulse
    delta_impulse[i] = impulse
    velocities[i] += inv_mass[i] * impulse


@wp.kernel
def solve_nodal_friction(
    inv_mass: wp.array[float],
    collider_friction: wp.array[float],
    collider_adhesion: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    collider_inv_mass: wp.array[float],
    velocities: wp.array[wp.vec3],
    collider_velocities: wp.array[wp.vec3],
    impulse: wp.array[wp.vec3],
    delta_impulse: wp.array[wp.vec3],
):
    """
    Solves for frictional impulses at nodes interacting with colliders.

    For each node (i) potentially in contact:
    1. Skips if friction coefficient is negative (no friction).
    2. Calculates the relative velocity `u0` between the particle and collider,
       accounting for the existing impulse and adhesion.
    3. Computes the effective inverse mass `w` for the interaction.
    4. Calls `solve_coulomb_isotropic` to determine the change in relative
       velocity `u` due to friction.
    5. Calculates the change in impulse `delta_impulse` required to achieve this
       change in relative velocity.
    6. Updates the total impulse, particle velocity, and collider velocity.
    """
    i = wp.tid()

    friction_coeff = collider_friction[i]
    if friction_coeff < 0.0:
        return

    n = collider_normals[i]
    u0 = velocities[i] - collider_velocities[i]

    w = inv_mass[i] + collider_inv_mass[i]

    u = solve_coulomb_isotropic(friction_coeff, n, u0 - (impulse[i] + collider_adhesion[i] * n) * w)

    delta_u = u - u0
    delta_lambda = delta_u / w

    delta_impulse[i] = delta_lambda
    impulse[i] += delta_lambda
    velocities[i] += inv_mass[i] * delta_lambda


@wp.kernel
def apply_subgrid_impulse(
    tr_collider_mat_offsets: wp.array[int],
    tr_collider_mat_columns: wp.array[int],
    tr_collider_mat_values: wp.array[float],
    inv_mass: wp.array[float],
    impulses: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
):
    """
    Applies pre-computed impulses to particles and colliders.
    """

    u_i = wp.tid()
    block_beg = tr_collider_mat_offsets[u_i]
    block_end = tr_collider_mat_offsets[u_i + 1]

    delta_f = wp.vec3(0.0)
    for b in range(block_beg, block_end):
        delta_f += tr_collider_mat_values[b] * impulses[tr_collider_mat_columns[b]]

    velocities[u_i] += inv_mass[u_i] * delta_f


@wp.kernel
def apply_subgrid_impulse_warmstart(
    collider_friction: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    collider_adhesion: wp.array[float],
    collider_impulse: wp.array[wp.vec3],
    delta_impulse: wp.array[wp.vec3],
):
    i = wp.tid()

    impulse = filter_collider_impulse_warmstart(
        collider_friction[i], collider_normals[i], collider_adhesion[i], collider_impulse[i]
    )

    collider_impulse[i] = impulse
    delta_impulse[i] = impulse


@wp.kernel
def compute_collider_delassus_diagonal(
    collider_mat_offsets: wp.array[int],
    collider_mat_columns: wp.array[int],
    collider_mat_values: wp.array[float],
    collider_inv_mass: wp.array[float],
    transposed_collider_mat_offsets: wp.array[int],
    inv_volume: wp.array[float],
    delassus_diagonal: wp.array[float],
):
    i = wp.tid()

    block_beg = collider_mat_offsets[i]
    block_end = collider_mat_offsets[i + 1]

    inv_mass = collider_inv_mass[i]
    w = inv_mass

    for b in range(block_beg, block_end):
        u_i = collider_mat_columns[b]
        weight = collider_mat_values[b]

        multiplicity = transposed_collider_mat_offsets[u_i + 1] - transposed_collider_mat_offsets[u_i]

        w += weight * weight * inv_volume[u_i] * float(multiplicity)

    delassus_diagonal[i] = w


@wp.kernel
def solve_subgrid_friction(
    velocity: wp.array[wp.vec3],
    collider_mat_offsets: wp.array[int],
    collider_mat_columns: wp.array[int],
    collider_mat_values: wp.array[float],
    collider_friction: wp.array[float],
    collider_adhesion: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    collider_delassus_diagonal: wp.array[float],
    collider_velocities: wp.array[wp.vec3],
    impulse: wp.array[wp.vec3],
    delta_impulse: wp.array[wp.vec3],
):
    i = wp.tid()

    w = collider_delassus_diagonal[i]
    friction_coeff = collider_friction[i]
    if w <= 0.0 or friction_coeff < 0.0:
        return

    beg = collider_mat_offsets[i]
    end = collider_mat_offsets[i + 1]

    u0 = -collider_velocities[i]
    for b in range(beg, end):
        u_i = collider_mat_columns[b]
        u0 += collider_mat_values[b] * velocity[u_i]

    n = collider_normals[i]

    u = solve_coulomb_isotropic(friction_coeff, n, u0 - (impulse[i] + collider_adhesion[i] * n) * w)

    delta_u = u - u0
    delta_lambda = delta_u / w

    impulse[i] += delta_lambda
    delta_impulse[i] = delta_lambda
