# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Material parameters and particle history shared by all implicit MPM quadratures."""

import warp as wp
import warp.fem as fem

from .implicit_mpm_model import MaterialParameters
from .rheology_solver_kernels import YieldParamVec, project_stress

wp.set_module_options({"enable_backward": False})

USE_HENCKY_STRAIN_MEASURE = wp.constant(True)
"""Use Hencky instead of co-rotated elastic model (replaces (S - I) with log S in Hooke's law)"""

MIN_PRINCIPAL_STRAIN = wp.constant(1.0e-6 if USE_HENCKY_STRAIN_MEASURE else 1.0e-2)
"""Minimum elastic strain for the elastic model (singular value of the elastic deformation gradient)"""

MAX_PRINCIPAL_STRAIN = wp.constant(1.0e6 if USE_HENCKY_STRAIN_MEASURE else 1.0e2)
"""Maximum elastic strain for the elastic model (singular value of the elastic deformation gradient)"""

MIN_HARDENING_JP = wp.constant(0.1)
"""Minimum plastic compression ratio for the hardening law (determinant of the plastic deformation gradient)"""

MIN_JP_DELTA = wp.constant(0.01)
"""Minimum delta for the plastic deformation gradient"""

MAX_JP_DELTA = wp.constant(10.0)
"""Maximum delta for the plastic deformation gradient"""

INFINITY = wp.constant(1.0e12)
"""Value above which quantities are considered infinite"""

EPSILON = wp.constant(1.0 / INFINITY)
"""Value below which quantities are considered zero"""


@wp.func
def hardening_law(Jp: float, hardening: float):
    if hardening == 0.0:
        return 1.0

    eps = wp.log(wp.clamp(Jp, MIN_HARDENING_JP, 1.0))
    h = wp.sinh(-hardening * eps)

    return h


@wp.func
def get_elastic_parameters(
    i: int,
    material_parameters: MaterialParameters,
):
    # Hardening only affects yield parameters, not elastic stiffness.
    # This separates the elastic response from the plastic history.
    E = material_parameters.young_modulus[i]
    nu = material_parameters.poisson_ratio[i]
    d = material_parameters.damping[i]

    return wp.vec3(E, nu, d)


@wp.func
def extract_elastic_parameters(
    params_vec: wp.vec3,
):
    compliance = 1.0 / params_vec[0]
    poisson = params_vec[1]
    damping = params_vec[2]
    return compliance, poisson, damping


@wp.func
def get_yield_parameters(i: int, material_parameters: MaterialParameters, particle_Jp: float, dt: float):
    h = hardening_law(particle_Jp, material_parameters.hardening[i])

    mu = material_parameters.friction[i]

    return YieldParamVec.from_values(
        mu,
        material_parameters.yield_pressure[i] * h,
        material_parameters.tensile_yield_ratio[i],
        material_parameters.yield_stress[i] * h,
        material_parameters.dilatancy[i],
        material_parameters.viscosity[i] / dt,
    )


@wp.func
def project_particle_strain(
    F: wp.mat33,
    F_prev: wp.mat33,
    compliance: float,
):
    if compliance <= EPSILON:
        return wp.identity(n=3, dtype=float)

    _U, xi, _V = wp.svd3(F)

    if wp.min(xi) < MIN_PRINCIPAL_STRAIN or wp.max(xi) > MAX_PRINCIPAL_STRAIN:
        return F_prev  # non-recoverable, discard update

    return F


@wp.func
def stress_strain_relationship(sig: wp.mat33, compliance: float, poisson: float):
    return (sig * (1.0 + poisson) - poisson * (wp.trace(sig) * wp.identity(n=3, dtype=float))) * compliance


@wp.func
def hencky_strain(F: wp.mat33):
    """Compute the spatial logarithmic strain from a deformation gradient."""
    U, xi, _V = wp.svd3(F)
    return U @ wp.diag(wp.vec3(wp.log(xi[0]), wp.log(xi[1]), wp.log(xi[2]))) @ wp.transpose(U)


@wp.func
def elastic_strain_measure(F: wp.mat33):
    """Compute the strain measure used by the elastic right-hand side."""
    if wp.static(USE_HENCKY_STRAIN_MEASURE):
        return hencky_strain(F)
    U, xi, _V = wp.svd3(F)
    return wp.identity(n=3, dtype=float) - U @ wp.diag(1.0 / xi) @ wp.transpose(U)


@wp.func
def update_particle_history(
    p: int,
    dt: float,
    kinematic_update: int,
    material_parameters: MaterialParameters,
    F_prev: wp.mat33,
    Jp_prev: float,
    elastic_delta: wp.mat33,
    plastic_delta: wp.mat33,
    stress: wp.mat33,
    velocity_gradient: wp.mat33,
):
    """Update plastic volume, projected stress, and elastic deformation at one particle.

    Sampling and accumulation belong to the quadrature. Cell integration can
    use the full kinematic increment; particle integration uses the solved
    elastic increment and the spin of the sampled velocity gradient.
    """
    p_rate = wp.trace(plastic_delta)
    delta_Jp = wp.exp(
        p_rate * wp.where(p_rate < 0.0, material_parameters.hardening_rate[p], material_parameters.softening_rate[p])
    )
    Jp_new = Jp_prev * wp.clamp(delta_Jp, MIN_JP_DELTA, MAX_JP_DELTA)

    compliance, _poisson, _damping = extract_elastic_parameters(get_elastic_parameters(p, material_parameters))
    yield_parameters = get_yield_parameters(p, material_parameters, Jp_new, dt)
    stress_dof = fem.SymmetricTensorMapper.value_to_dof_3d(stress)
    stress_new = fem.SymmetricTensorMapper.dof_to_value_3d(project_stress(stress_dof, yield_parameters))

    if kinematic_update != 0:
        F_new = F_prev + (dt * velocity_gradient - plastic_delta) @ F_prev
    else:
        skew = 0.5 * dt * (velocity_gradient - wp.transpose(velocity_gradient))
        F_new = F_prev + (elastic_delta + skew) @ F_prev

    return project_particle_strain(F_new, F_prev, compliance), Jp_new, stress_new
