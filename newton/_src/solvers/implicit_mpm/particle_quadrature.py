# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PIC and GIMP quadrature, material integration, and particle updates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
import warp.fem as fem

import newton

from .implicit_mpm_model import ImplicitMPMModel, MaterialParameters
from .implicit_mpm_solver_kernels import integrate_active_fraction
from .integration import ElasticityInputs
from .material_kernels import (
    EPSILON,
    USE_HENCKY_STRAIN_MEASURE,
    elastic_strain_measure,
    extract_elastic_parameters,
    get_elastic_parameters,
    get_yield_parameters,
    project_particle_strain,
    stress_strain_relationship,
    update_particle_history,
)

if TYPE_CHECKING:
    from .solver_implicit_mpm import ImplicitMPMScratchpad

wp.set_module_options({"enable_backward": False})


@fem.integrand
def integrate_elastic_parameters(
    s: fem.Sample,
    u: fem.Field,
    inv_cell_volume: float,
    material_parameters: MaterialParameters,
    particle_flags: wp.array[wp.int32],
):
    if ~particle_flags[s.qp_index] & newton.ParticleFlags.ACTIVE:
        return 0.0

    i = s.qp_index
    params_vec = get_elastic_parameters(i, material_parameters)
    return wp.dot(u(s), params_vec) * inv_cell_volume


@fem.integrand
def integrate_yield_parameters(
    s: fem.Sample,
    u: fem.Field,
    inv_cell_volume: float,
    material_parameters: MaterialParameters,
    particle_Jp: wp.array[float],
    dt: float,
    particle_flags: wp.array[wp.int32],
):
    if ~particle_flags[s.qp_index] & newton.ParticleFlags.ACTIVE:
        return 0.0

    i = s.qp_index
    params_vec = get_yield_parameters(i, material_parameters, particle_Jp[i], dt)
    return wp.dot(u(s), params_vec) * inv_cell_volume


@wp.kernel
def average_elastic_parameters(
    elastic_parameters_int: wp.array[wp.vec3],
    particle_volume: wp.array[float],
    elastic_parameters_avg: wp.array[wp.vec3],
):
    i = wp.tid()
    pvol = particle_volume[i]
    elastic_parameters_avg[i] = elastic_parameters_int[i] / wp.max(pvol, EPSILON)


@fem.integrand
def _advect_particles(
    s: fem.Sample,
    domain: fem.Domain,
    grid_vel: fem.Field,
    dt: float,
    max_vel: float,
    particle_flags: wp.array[wp.int32],
    particle_volume: wp.array[float],
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    vel_grad: wp.array[wp.mat33],
):
    if ~particle_flags[s.qp_index] & newton.ParticleFlags.ACTIVE:
        return

    p_vel = grid_vel(s)
    vel_n_sq = wp.length_sq(p_vel)

    p_vel_cfl = wp.where(vel_n_sq > max_vel * max_vel, p_vel * max_vel / wp.sqrt(vel_n_sq), p_vel)

    p_vel_grad = fem.grad(grid_vel, s)

    delta_pos = dt * p_vel_cfl

    gimp_weight = s.qp_weight * fem.measure(domain, s) / particle_volume[s.qp_index]
    wp.atomic_add(pos, s.qp_index, gimp_weight * delta_pos)
    wp.atomic_add(vel, s.qp_index, gimp_weight * p_vel_cfl)
    wp.atomic_add(vel_grad, s.qp_index, gimp_weight * p_vel_grad)


@fem.integrand
def update_particle_strains(
    s: fem.Sample,
    domain: fem.Domain,
    grid_vel: fem.Field,
    plastic_strain_delta: fem.Field,
    elastic_strain_delta: fem.Field,
    stress: fem.Field,
    dt: float,
    particle_flags: wp.array[wp.int32],
    particle_density: wp.array[float],
    particle_volume: wp.array[float],
    material_parameters: MaterialParameters,
    elastic_strain_prev: wp.array[wp.mat33],
    particle_Jp_prev: wp.array[float],
    elastic_strain: wp.array[wp.mat33],
    particle_Jp: wp.array[float],
    particle_stress: wp.array[wp.mat33],
    residual_strain_tracking: bool,
    residual_deformation_gradient_prev: wp.array[wp.mat33],
    residual_deformation_gradient: wp.array[wp.mat33],
):
    if ~particle_flags[s.qp_index] & newton.ParticleFlags.ACTIVE:
        elastic_strain[s.qp_index] = elastic_strain_prev[s.qp_index]
        particle_Jp[s.qp_index] = particle_Jp_prev[s.qp_index]
        residual_deformation_gradient[s.qp_index] = residual_deformation_gradient_prev[s.qp_index]
        return
    if particle_density[s.qp_index] == 0.0:
        elastic_strain[s.qp_index] = elastic_strain_prev[s.qp_index]
        particle_Jp[s.qp_index] = particle_Jp_prev[s.qp_index]
        residual_deformation_gradient[s.qp_index] = residual_deformation_gradient_prev[s.qp_index]
        return

    p_strain_delta = plastic_strain_delta(s)
    vel_grad = fem.grad(grid_vel, s)
    e_strain_delta = elastic_strain_delta(s)
    elastic_strain_new, particle_Jp_new, particle_stress_new = update_particle_history(
        s.qp_index,
        dt,
        0,
        material_parameters,
        elastic_strain_prev[s.qp_index],
        particle_Jp_prev[s.qp_index],
        e_strain_delta,
        p_strain_delta,
        stress(s),
        vel_grad,
    )

    gimp_weight = s.qp_weight * fem.measure(domain, s) / particle_volume[s.qp_index]
    wp.atomic_add(particle_Jp, s.qp_index, gimp_weight * particle_Jp_new)
    wp.atomic_add(particle_stress, s.qp_index, gimp_weight * particle_stress_new)
    wp.atomic_add(elastic_strain, s.qp_index, gimp_weight * elastic_strain_new)

    if residual_strain_tracking:
        # Subtract every increment already represented by the weak solve.
        skew = 0.5 * dt * (vel_grad - wp.transpose(vel_grad))
        residual_delta = dt * vel_grad - skew - p_strain_delta - e_strain_delta
        residual_prev = residual_deformation_gradient_prev[s.qp_index]
        residual_new = residual_prev + residual_delta @ residual_prev
        residual_new = project_particle_strain(residual_new, residual_prev, 1.0)
        wp.atomic_add(residual_deformation_gradient, s.qp_index, gimp_weight * residual_new)


@fem.integrand
def strain_rhs(
    s: fem.Sample,
    tau: fem.Field,
    elastic_parameters: fem.Field,
    elastic_strains: wp.array[wp.mat33],
    inv_cell_volume: float,
    dt: float,
    particle_flags: wp.array[wp.int32],
):
    if ~particle_flags[s.qp_index] & newton.ParticleFlags.ACTIVE:
        return 0.0

    _compliance, _poisson, damping = extract_elastic_parameters(elastic_parameters(s))
    alpha = 1.0 / (1.0 + damping / dt)

    strain = alpha * wp.ddot(tau(s), elastic_strain_measure(elastic_strains[s.qp_index]))
    return strain * inv_cell_volume


@fem.integrand
def compliance_form(
    s: fem.Sample,
    domain: fem.Domain,
    tau: fem.Field,
    sig: fem.Field,
    elastic_parameters: fem.Field,
    elastic_strains: wp.array[wp.mat33],
    inv_cell_volume: float,
    dt: float,
    particle_flags: wp.array[wp.int32],
):
    if ~particle_flags[s.qp_index] & newton.ParticleFlags.ACTIVE:
        return 0.0

    F = elastic_strains[s.qp_index]

    compliance, poisson, damping = extract_elastic_parameters(elastic_parameters(s))
    gamma = compliance / (1.0 + damping / dt)

    U, xi, V = wp.svd3(F)
    Rt = V @ wp.transpose(U)

    if wp.static(USE_HENCKY_STRAIN_MEASURE):
        R = wp.transpose(Rt)
        return wp.ddot(Rt @ tau(s) @ R, stress_strain_relationship(Rt @ sig(s) @ R, gamma, poisson)) * inv_cell_volume
    else:
        FinvT = U @ wp.diag(1.0 / xi) @ wp.transpose(V)
        return (
            wp.ddot(Rt @ tau(s) @ FinvT, stress_strain_relationship(Rt @ sig(s) @ FinvT, gamma, poisson))
            * inv_cell_volume
        )


def particle_grid_locations_gimp(
    domain: fem.GeometryDomain,
    positions: wp.array,
    radii: wp.array,
    particle_environment: wp.array | None,
    *,
    separate_worlds: bool,
    temporary_store: fem.TemporaryStore | None,
) -> tuple[wp.array, wp.array, wp.array]:
    """Distribute particle domains across their overlapping grid cells."""

    cell_lookup = domain.element_partition_lookup
    cell_closest_point = domain.element_closest_point

    @wp.func
    def add_cell(
        particle_cell_indices: wp.array[fem.ElementIndex],
        particle_cell_coords: wp.array[fem.Coords],
        particle_cell_fractions: wp.array[float],
        cell_index: int,
        cell_coords: fem.Coords,
        cell_weight: float,
    ):
        for i in range(8):
            if particle_cell_indices[i] == fem.NULL_NODE_INDEX:
                particle_cell_indices[i] = cell_index
                particle_cell_coords[i] = cell_coords
                particle_cell_fractions[i] = cell_weight
                return

            if particle_cell_indices[i] == cell_index:
                particle_cell_fractions[i] += cell_weight
                return

    @fem.cache.dynamic_kernel(suffix=f"{domain.name}_{'isolated' if separate_worlds else 'shared'}")
    def particle_locations_gimp(
        cell_arg_value: domain.ElementArg,
        domain_index_arg_value: domain.ElementIndexArg,
        positions: wp.array[wp.vec3],
        radii: wp.array[float],
        particle_environment: wp.array[int],
        cell_index: wp.array2d[fem.ElementIndex],
        cell_coords: wp.array2d[fem.Coords],
        cell_fractions: wp.array2d[float],
    ):
        p = wp.tid()
        domain_arg = domain.DomainArg(cell_arg_value, domain_index_arg_value)

        center = positions[p]
        radius = radii[p]

        tot_weight = float(0.0)

        # Find cell containing each corner of the particle,
        # merging repeated cell indices
        for vtx in range(8):
            i = (vtx & 4) >> 2
            j = (vtx & 2) >> 1
            k = vtx & 1

            pos = center - wp.vec3(radius) + 2.0 * radius * wp.vec3(float(i), float(j), float(k))
            if wp.static(separate_worlds):
                sample = cell_lookup(domain_arg, pos, int(particle_environment[p]))
            else:
                sample = cell_lookup(domain_arg, pos)

            if sample.element_index == fem.NULL_ELEMENT_INDEX:
                continue

            elem_index = domain.element_partition_index(domain_index_arg_value, sample.element_index)
            cell_weight = wp.min(wp.min(sample.element_coords), 1.0 - wp.max(sample.element_coords))

            if cell_weight > 0.0:
                tot_weight += cell_weight
                cell_center_coords, _ = cell_closest_point(cell_arg_value, sample.element_index, center)
                add_cell(
                    cell_index[p],
                    cell_coords[p],
                    cell_fractions[p],
                    elem_index,
                    cell_center_coords,
                    cell_weight,
                )

        # Normalize the weights over the cells
        for vtx in range(8):
            if cell_index[p, vtx] != fem.NULL_NODE_INDEX:
                cell_fractions[p, vtx] /= tot_weight

    device = positions.device

    cell_indices = fem.borrow_temporary(temporary_store, shape=(positions.shape[0], 8), dtype=fem.ElementIndex)
    cell_coords = fem.borrow_temporary(temporary_store, shape=(positions.shape[0], 8), dtype=fem.Coords)
    cell_fractions = fem.borrow_temporary(temporary_store, shape=(positions.shape[0], 8), dtype=float)

    cell_indices.fill_(fem.NULL_NODE_INDEX)

    wp.launch(
        particle_locations_gimp,
        dim=positions.shape[0],
        inputs=[
            domain.element_arg_value(device=device),
            domain.element_index_arg_value(device=device),
            positions,
            radii,
            particle_environment,
            cell_indices,
            cell_coords,
            cell_fractions,
        ],
        device=device,
    )

    return cell_indices, cell_coords, cell_fractions


def make_quadrature(
    domain: fem.GeometryDomain,
    positions: wp.array[wp.vec3],
    mpm_model: ImplicitMPMModel,
    particle_environment: wp.array[int] | None,
    *,
    gimp: bool,
    separate_worlds: bool,
    temporary_store: fem.TemporaryStore | None,
    use_domain_element_indices: bool,
) -> fem.PicQuadrature:
    """Bin particle centers or GIMP domain samples into the selected grid partition."""
    if gimp:
        locations = particle_grid_locations_gimp(
            domain,
            positions,
            mpm_model.particle_radius,
            particle_environment,
            separate_worlds=separate_worlds,
            temporary_store=temporary_store,
        )
        return fem.PicQuadrature(
            domain=domain,
            positions=locations,
            measures=mpm_model.particle_volume,
            temporary_store=temporary_store,
            use_domain_element_indices=use_domain_element_indices,
        )
    return fem.PicQuadrature(
        domain=domain,
        positions=positions,
        env_indices=particle_environment,
        measures=mpm_model.particle_volume,
        temporary_store=temporary_store,
        use_domain_element_indices=use_domain_element_indices,
    )


def elasticity_inputs(
    pic: fem.PicQuadrature,
    elastic_strains: wp.array[wp.mat33],
    mpm_model: ImplicitMPMModel,
    dt: float,
    scratch: ImplicitMPMScratchpad,
    inv_cell_volume: float,
    temporary_store: fem.TemporaryStore | None,
) -> ElasticityInputs:
    """Average particle elastic parameters onto velocity nodes and bind the elastic forms."""
    values = {
        "material_parameters": mpm_model.material_parameters,
        "particle_flags": mpm_model.material_particle_flags,
        "inv_cell_volume": inv_cell_volume,
    }
    node_particle_volume = fem.integrate(
        integrate_active_fraction,
        quadrature=pic,
        fields={"phi": scratch.fraction_test},
        values={"particle_flags": mpm_model.material_particle_flags, "inv_cell_volume": inv_cell_volume},
        output_dtype=float,
        temporary_store=temporary_store,
    )
    elastic_parameters_int = fem.integrate(
        integrate_elastic_parameters,
        quadrature=pic,
        fields={"u": scratch.velocity_test},
        values=values,
        output_dtype=wp.vec3,
        temporary_store=temporary_store,
    )
    wp.launch(
        average_elastic_parameters,
        dim=scratch.elastic_parameters_field.space_partition.node_count(),
        inputs=[elastic_parameters_int, node_particle_volume, scratch.elastic_parameters_field.dof_values],
    )
    return ElasticityInputs(
        quadrature=pic,
        strain_rhs=strain_rhs,
        compliance=compliance_form,
        fields={"elastic_parameters": scratch.elastic_parameters_field},
        values={
            "elastic_strains": elastic_strains,
            "particle_flags": mpm_model.material_particle_flags,
            "inv_cell_volume": inv_cell_volume,
            "dt": dt,
        },
    )


def yield_parameters_inputs(
    particle_Jp: wp.array[float],
    mpm_model: ImplicitMPMModel,
    dt: float,
    inv_cell_volume: float,
) -> tuple[fem.Integrand, dict]:
    """Bind particle material parameters to the shared yield-parameter assembly."""
    return integrate_yield_parameters, {
        "particle_Jp": particle_Jp,
        "material_parameters": mpm_model.material_parameters,
        "particle_flags": mpm_model.material_particle_flags,
        "inv_cell_volume": inv_cell_volume,
        "dt": dt,
    }


def update_particles(
    dt: float,
    grid_vel: fem.DiscreteField,
    elastic_strain_delta: fem.DiscreteField,
    plastic_strain_delta: fem.DiscreteField,
    stress: fem.DiscreteField,
    particle_flags: wp.array[wp.int32],
    particle_density: wp.array[float],
    material_parameters: MaterialParameters,
    elastic_strain_prev: wp.array[wp.mat33],
    particle_Jp_prev: wp.array[float],
    elastic_strain: wp.array[wp.mat33],
    particle_Jp: wp.array[float],
    particle_stress: wp.array[wp.mat33],
    *,
    pic: fem.PicQuadrature,
    particle_volume: wp.array[float],
    residual_strain_tracking: bool,
    residual_deformation_gradient_prev: wp.array[wp.mat33],
    residual_deformation_gradient: wp.array[wp.mat33],
    temporary_store: fem.TemporaryStore | None,
):
    """Sample particle history and accumulate GIMP contributions safely for in-place states."""
    if elastic_strain_prev.ptr == elastic_strain.ptr:
        elastic_strain_prev = wp.clone(elastic_strain_prev)
    if particle_Jp_prev.ptr == particle_Jp.ptr:
        particle_Jp_prev = wp.clone(particle_Jp_prev)
    if residual_strain_tracking:
        if residual_deformation_gradient_prev.ptr == residual_deformation_gradient.ptr:
            residual_deformation_gradient_prev = wp.clone(residual_deformation_gradient_prev)
        residual_deformation_gradient.zero_()

    particle_Jp.zero_()
    particle_stress.zero_()
    elastic_strain.zero_()
    fem.interpolate(
        update_particle_strains,
        at=pic,
        fields={
            "grid_vel": grid_vel,
            "elastic_strain_delta": elastic_strain_delta,
            "plastic_strain_delta": plastic_strain_delta,
            "stress": stress,
        },
        values={
            "dt": dt,
            "particle_flags": particle_flags,
            "particle_density": particle_density,
            "particle_volume": particle_volume,
            "material_parameters": material_parameters,
            "elastic_strain_prev": elastic_strain_prev,
            "particle_Jp_prev": particle_Jp_prev,
            "elastic_strain": elastic_strain,
            "particle_Jp": particle_Jp,
            "particle_stress": particle_stress,
            "residual_strain_tracking": residual_strain_tracking,
            "residual_deformation_gradient_prev": residual_deformation_gradient_prev,
            "residual_deformation_gradient": residual_deformation_gradient,
        },
        temporary_store=temporary_store,
    )


def advect_particles(
    dt: float,
    max_vel: float,
    grid_vel: fem.DiscreteField,
    particle_flags: wp.array[wp.int32],
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    vel_grad: wp.array[wp.mat33],
    *,
    pic: fem.PicQuadrature,
    particle_volume: wp.array[float],
    temporary_store: fem.TemporaryStore | None,
):
    """Advect with particle samples, averaging the contributions from GIMP domains."""
    fem.interpolate(
        _advect_particles,
        at=pic,
        fields={"grid_vel": grid_vel},
        values={
            "particle_flags": particle_flags,
            "particle_volume": particle_volume,
            "pos": pos,
            "vel": vel,
            "vel_grad": vel_grad,
            "dt": dt,
            "max_vel": max_vel,
        },
        temporary_store=temporary_store,
    )
