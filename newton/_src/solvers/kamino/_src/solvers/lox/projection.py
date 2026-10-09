# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Jacobi and colored Gauss--Seidel schedules of the LOX projection sweeps.

Both schedules share the same row updates (see :mod:`.projection_kernels`).
Jacobi projects every row in each sweep; colored Gauss--Seidel projects one
color at a time, applying the body corrections between colors, and finishes
with one Jacobi smoothing sweep. The ordered color sweeps favor the rows of the
last colors; the order-independent smoothing sweep restores the symmetry of the
result, e.g. between identical contacts of a symmetric body.

Each sweep is one row-parallel launch. When the worlds fill the device or their
rows are few, one thread block runs every sweep of a world, in a single launch.
With deformable bodies, the sweeps also project the particle contact rows, which
share the colors and the body corrections of the rigid rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ...core.types import mat66f, vec6f
from .anderson import AndersonProjection
from .apgd import APGDAcceleration
from .coloring import RowColoring
from .projection_kernels import (
    _apply_particle_delta,
    _apply_twist_delta,
    _prepare_box_blocks,
    _prepare_contact_blocks,
    _prepare_particle_contact_responses,
    _prepare_particle_contacts,
    _prepare_physical_particle_contacts,
    _prepare_physical_spatial_contacts,
    _prepare_spatial_contacts,
    _warmstart_box_rows,
    _warmstart_contacts,
    _warmstart_particle_contacts,
    _warmstart_spatial_contacts,
    make_sweep_kernel,
)
from .system_kernels import ProjectionWeights
from .types import ProjectionAcceleration, ProjectionData
from .world_projection_kernels import NoAccelerationState, can_project_by_world, make_project_by_world_kernel

if TYPE_CHECKING:
    from .types import LOXProblemData

###
# Module interface
###

__all__ = [
    "ProjectionSchedule",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


_SWEEP_BLOCK_DIM = 128


_JACOBI_BLOCKS_PER_SM = 5


_COLORED_BLOCKS_PER_SM = 2


_JACOBI_WORLD_BLOCK_DIM = 128


_COLORED_WORLD_BLOCK_DIM = 64


###
# Interfaces
###


class ProjectionSchedule:
    """Prepare and run the projection sweeps of one LOX problem.

    Args:
        problem_data: Box rows, contacts, and body incidence of the LOX problem.
        color_count: Maximum number of Gauss--Seidel colors; one selects Jacobi.
        acceleration: Acceleration of the projection sweeps; APGD requires the Jacobi schedule.
    """

    def __init__(
        self,
        problem_data: LOXProblemData,
        color_count: int,
        acceleration: ProjectionAcceleration = ProjectionAcceleration.NONE,
    ):
        self.problem_data = problem_data
        self.device = problem_data.device
        self._data = ProjectionData(problem_data, color_count, acceleration)
        self._acceleration = acceleration
        data = self._data
        self.coloring = RowColoring(problem_data, data.coloring) if data.coloring is not None else None
        self.color_count = data.coloring.color_count if data.coloring is not None else 1
        colored = self.coloring is not None
        self.apgd = APGDAcceleration(problem_data, data.apgd) if data.apgd is not None else None
        self.anderson = AndersonProjection(problem_data, data.anderson) if data.anderson is not None else None

        particle_rows = problem_data.particle_contact_rows
        self._particles = particle_rows is not None
        row_capacity = problem_data.box_rows.capacity + problem_data.contact_rows.capacity
        if self._particles:
            row_capacity += particle_rows.capacity
        blocks_per_sm = _COLORED_BLOCKS_PER_SM if colored else _JACOBI_BLOCKS_PER_SM
        self._worker_count = row_capacity
        if self.device.is_cuda:
            max_workers = max(_SWEEP_BLOCK_DIM, self.device.sm_count * blocks_per_sm * _SWEEP_BLOCK_DIM)
            self._worker_count = min(row_capacity, max_workers)
        if self._particles:
            # The per-world kernels sweep the rigid rows of each world only
            self._world_block_dim = _COLORED_WORLD_BLOCK_DIM if colored else _JACOBI_WORLD_BLOCK_DIM
            self._by_world = False
        elif colored:
            self._world_block_dim = _COLORED_WORLD_BLOCK_DIM
            self._by_world = can_project_by_world(
                self.device,
                problem_data.num_worlds,
                problem_data.max_world_body_count,
                (row_capacity + self.color_count - 1) // self.color_count,
                _COLORED_WORLD_BLOCK_DIM,
                min_blocks_per_sm=1,
            )
        else:
            self._world_block_dim = _JACOBI_WORLD_BLOCK_DIM
            self._by_world = can_project_by_world(
                self.device,
                problem_data.num_worlds,
                problem_data.max_world_body_count,
                row_capacity,
                _JACOBI_WORLD_BLOCK_DIM,
            )
        if data.coloring is not None and not self._by_world:
            # Only the per-world sweeps read the rows of each color of each world
            data.coloring.world_color_count = None
            data.coloring.world_color_cursor = None
            data.coloring.world_color_box_count = None
            data.coloring.world_order = None

    ###
    # Properties
    ###

    @property
    def data(self) -> ProjectionData:
        """Returns the local blocks, the sweep buffer, and the coloring or acceleration state."""
        return self._data

    ###
    # Public API
    ###

    def prepare(
        self,
        inverse_weight: wp.array[mat66f],
        world_failed: wp.array[wp.bool],
        particle_inverse_weight: wp.array[wp.float32] | None = None,
    ) -> None:
        """Color the rows and prepare the local blocks for the current body and particle weights.

        Args:
            inverse_weight: Inverse body weight ``W^-1`` of each body.
            world_failed: Failure flag of each world, set for worlds with an invalid local block.
            particle_inverse_weight: Inverse weight ``w^-1`` of each packed particle, with deformable bodies.
        """
        p = self.problem_data
        self._weights = self._prepare_weights(inverse_weight)
        self._particle_inverse_weight = particle_inverse_weight
        coloring = self._data.coloring
        if self.coloring is not None:
            self.coloring.build()
            # The colored sweeps split each body over the rows of a color, the smoothing sweep over all its rows
            self._prepare_blocks(
                inverse_weight,
                coloring.occupancy,
                coloring.box.color,
                coloring.contact.color,
                p.body_incidence,
                world_failed,
            )
            if self._particles:
                self._prepare_particle_blocks(
                    inverse_weight,
                    coloring.occupancy,
                    coloring.particle.color,
                    coloring.particle_weight_sum,
                    particle_inverse_weight,
                    self._data.particle_delassus,
                    world_failed,
                )
                self._prepare_particle_blocks(
                    inverse_weight,
                    p.body_incidence,
                    None,
                    p.particle_weight_sum,
                    particle_inverse_weight,
                    self._data.particle_smoothing_delassus,
                    world_failed,
                )
        else:
            self._prepare_blocks(inverse_weight, p.body_incidence, None, None, None, world_failed)
            if self._particles:
                self._prepare_particle_blocks(
                    inverse_weight,
                    p.body_incidence,
                    None,
                    p.particle_weight_sum,
                    particle_inverse_weight,
                    self._data.particle_delassus,
                    world_failed,
                )
        if self._particles:
            self._prepare_physical_particle_blocks(particle_inverse_weight)
        if self.anderson is not None:
            self.anderson.reset()

    def project(
        self,
        iterations: int,
        recycle: bool,
        world_active: wp.array[wp.bool],
        world_failed: wp.array[wp.bool],
        body_world: wp.array[wp.int32],
        projected_twist: wp.array[vec6f],
        projected_velocity: wp.array[wp.vec3f] | None = None,
    ) -> None:
        """Warm start with the current reactions and run the projection sweeps with the weights of :meth:`prepare`.

        Worlds that already failed are skipped, and any failed local update marks
        its world in ``world_failed``.

        Args:
            iterations: Number of sweeps; a colored sweep visits every color once.
            recycle: Whether Anderson retains its direction from the previous splitting iteration.
            world_active: Worlds to project.
            world_failed: Failure flag of each world.
            body_world: World of each body.
            projected_twist: Projected body twists, updated in place.
            projected_velocity: Projected particle velocities, updated in place, with deformable bodies.
        """
        self._begin_outer_iteration(world_active, recycle=recycle)
        if self._by_world:
            self._project_by_world(iterations, world_active, self._weights, projected_twist, world_failed)
            return

        weights = (self._weights, self._particle_inverse_weight)
        velocities = (projected_twist, projected_velocity)
        if self.apgd is not None:
            self.apgd.begin(world_active)
        self._warmstart(world_active, weights, world_failed)
        self._apply(body_world, world_active, world_failed, velocities)

        if self.anderson is not None:
            self.anderson.begin_map(world_active, world_failed)
        for iteration in range(iterations):
            for color in range(self.color_count):
                self._sweep(color, world_active, weights, velocities, world_failed)
                self._apply(body_world, world_active, world_failed, velocities)
            if self.anderson is not None:
                self.anderson.finish_map(
                    world_active,
                    world_failed,
                    self._weights,
                    self._data.twist_delta,
                    self._particle_inverse_weight,
                    self._data.particle_delta,
                )
                self._apply(body_world, world_active, world_failed, velocities)
            if self.apgd is not None and iteration + 1 < iterations:
                self.apgd.extrapolate(
                    world_active,
                    world_failed,
                    self._weights,
                    self._data.twist_delta,
                    self._particle_inverse_weight,
                    self._data.particle_delta,
                )
                self._apply(body_world, world_active, world_failed, velocities)
        if self.coloring is not None:
            # An order-independent Jacobi sweep restores the symmetry lost to the color ordering.
            self._sweep(0, world_active, weights, velocities, world_failed, smoothing=True)
            self._apply(body_world, world_active, world_failed, velocities)

    ###
    # Internals
    ###

    def _begin_outer_iteration(self, world_active: wp.array[wp.bool], recycle: bool) -> None:
        """Restart the Anderson samples of a new splitting iteration."""
        if self.anderson is not None:
            self.anderson.begin_outer_iteration(world_active, recycle=recycle)

    def _prepare_weights(self, inverse_weight: wp.array[mat66f]) -> ProjectionWeights:
        """Return the weights of the projection kernels and prepare the twist responses of the particle contact rows.

        :meth:`_prepare_blocks` fills the twist responses of the rigid rows.
        """
        p = self.problem_data
        weights = ProjectionWeights()
        weights.inverse_weight = inverse_weight
        weights.box_response = p.box_rows.response
        weights.contact_response = p.contact_rows.response
        if self._particles:
            rows = p.particle_contact_rows
            if rows.capacity > 0:
                wp.launch(
                    _prepare_particle_contact_responses,
                    dim=rows.capacity,
                    inputs=[rows.status, rows.body, rows.body_jacobian, inverse_weight],
                    outputs=[rows.response],
                    device=self.device,
                )
            weights.particle_contact_response = rows.response
        return weights

    def _prepare_blocks(self, inverse_weight, occupancy, box_color, contact_color, smoothing_occupancy, world_failed):
        """Prepare the twist responses, the split and smoothing local blocks, and the unsplit residual blocks.

        The smoothing blocks are prepared with ``smoothing_occupancy``, or skipped when it is ``None``.
        """
        p = self.problem_data
        smoothing = smoothing_occupancy is not None
        if p.box_rows.capacity > 0:
            rows = p.box_rows
            wp.launch(
                _prepare_box_blocks,
                dim=rows.capacity,
                inputs=[
                    rows.world,
                    rows.local,
                    rows.world_count,
                    box_color,
                    rows.body_a,
                    rows.body_b,
                    rows.jacobian_a,
                    rows.jacobian_b,
                    inverse_weight,
                    occupancy,
                    smoothing_occupancy,
                ],
                outputs=[
                    rows.response,
                    self._data.box_delassus,
                    self._data.box_smoothing_delassus if smoothing else None,
                    rows.physical_delassus,
                    world_failed,
                ],
                device=self.device,
            )
        if p.contact_rows.capacity == 0:
            return
        rows = p.contact_rows
        if self._data.spatial_blocks is None:
            wp.launch(
                _prepare_contact_blocks,
                dim=rows.capacity,
                inputs=[
                    rows.world,
                    rows.local,
                    rows.world_count,
                    contact_color,
                    rows.body_a,
                    rows.body_b,
                    rows.jacobian_a,
                    rows.jacobian_b,
                    rows.bias,
                    rows.friction,
                    rows.normal_compliance,
                    inverse_weight,
                    occupancy,
                    smoothing_occupancy,
                ],
                outputs=[
                    rows.response,
                    self._data.contact_delassus,
                    self._data.contact_smoothing_delassus if smoothing else None,
                    rows.physical_delassus,
                    world_failed,
                ],
                device=self.device,
            )
            return
        # Spatial contacts apply the inverse weights directly
        self._prepare_spatial_blocks(inverse_weight, occupancy, contact_color, self._data.spatial_blocks, world_failed)
        if smoothing:
            self._prepare_spatial_blocks(
                inverse_weight, smoothing_occupancy, None, self._data.spatial_smoothing_blocks, world_failed
            )
        wp.launch(
            _prepare_physical_spatial_contacts,
            dim=rows.capacity,
            inputs=[
                rows.world,
                rows.local,
                rows.world_count,
                rows.body_a,
                rows.body_b,
                rows.jacobian_a,
                rows.jacobian_b,
                rows.frame,
                rows.normal_compliance,
                inverse_weight,
            ],
            outputs=[rows.physical_delassus],
            device=self.device,
        )

    def _prepare_spatial_blocks(self, inverse_weight, occupancy, contact_color, spatial_blocks, world_failed):
        rows = self.problem_data.contact_rows
        wp.launch(
            _prepare_spatial_contacts,
            dim=rows.capacity,
            inputs=[
                rows.world,
                rows.local,
                rows.world_count,
                contact_color,
                rows.body_a,
                rows.body_b,
                rows.jacobian_a,
                rows.jacobian_b,
                rows.frame,
                rows.friction,
                rows.angular_friction,
                rows.normal_compliance,
                occupancy,
                inverse_weight,
            ],
            outputs=[
                spatial_blocks.delassus,
                spatial_blocks.eigenvectors,
                spatial_blocks.eigenvalues,
                world_failed,
            ],
            device=self.device,
        )

    def _prepare_physical_particle_blocks(self, particle_inverse_weight: wp.array[wp.float32]) -> None:
        """Prepare the unsplit Delassus blocks of the particle contact rows, which scale their convergence residuals."""
        rows = self.problem_data.particle_contact_rows
        if rows.capacity == 0:
            return
        wp.launch(
            _prepare_physical_particle_contacts,
            dim=rows.capacity,
            inputs=[
                rows.status,
                rows.particle,
                rows.coefficient,
                rows.body,
                rows.body_jacobian,
                particle_inverse_weight,
                rows.response,
            ],
            outputs=[rows.physical_delassus],
            device=self.device,
        )

    def _prepare_particle_blocks(
        self,
        inverse_weight,
        occupancy,
        particle_color,
        particle_weight_sum,
        particle_inverse_weight,
        particle_delassus,
        world_failed,
    ):
        rows = self.problem_data.particle_contact_rows
        if rows.capacity == 0:
            return
        wp.launch(
            _prepare_particle_contacts,
            dim=rows.capacity,
            inputs=[
                rows.world,
                rows.status,
                particle_color,
                rows.particle,
                rows.coefficient,
                rows.body,
                rows.body_jacobian,
                rows.bias,
                rows.friction,
                particle_weight_sum,
                particle_inverse_weight,
                occupancy,
                rows.response,
            ],
            outputs=[particle_delassus, world_failed],
            device=self.device,
        )

    def _warmstart(self, world_active, weights, world_failed) -> None:
        p = self.problem_data
        weights, particle_inverse_weight = weights
        if self._particles and p.particle_contact_rows.capacity > 0:
            rows = p.particle_contact_rows
            wp.launch(
                _warmstart_particle_contacts,
                dim=rows.capacity,
                inputs=[
                    rows.world,
                    rows.status,
                    rows.particle,
                    rows.coefficient,
                    rows.body,
                    rows.frame,
                    rows.body_jacobian,
                    rows.reaction,
                    world_active,
                    world_failed,
                    particle_inverse_weight,
                    weights,
                ],
                outputs=[self._data.particle_delta, self._data.twist_delta],
                device=self.device,
            )
        if p.box_rows.capacity > 0:
            wp.launch(
                _warmstart_box_rows,
                dim=p.box_rows.capacity,
                inputs=[
                    p.box_rows.world,
                    p.box_rows.local,
                    p.box_rows.world_count,
                    p.box_rows.body_a,
                    p.box_rows.body_b,
                    p.box_rows.jacobian_a,
                    p.box_rows.jacobian_b,
                    p.box_rows.reaction,
                    world_active,
                    world_failed,
                    weights,
                ],
                outputs=[self._data.twist_delta],
                device=self.device,
            )
        if p.contact_rows.capacity == 0:
            return
        rows = p.contact_rows
        if self._data.spatial_blocks is not None:
            wp.launch(
                _warmstart_spatial_contacts,
                dim=rows.capacity,
                inputs=[
                    rows.world,
                    rows.local,
                    rows.world_count,
                    rows.body_a,
                    rows.body_b,
                    rows.jacobian_a,
                    rows.jacobian_b,
                    rows.frame,
                    rows.friction,
                    rows.angular_friction,
                    world_active,
                    world_failed,
                    weights,
                ],
                outputs=[rows.reaction, rows.angular_reaction, self._data.twist_delta],
                device=self.device,
            )
        else:
            wp.launch(
                _warmstart_contacts,
                dim=p.contact_rows.capacity,
                inputs=[
                    p.contact_rows.world,
                    p.contact_rows.local,
                    p.contact_rows.world_count,
                    p.contact_rows.body_a,
                    p.contact_rows.body_b,
                    p.contact_rows.jacobian_a,
                    p.contact_rows.jacobian_b,
                    p.contact_rows.reaction,
                    world_active,
                    world_failed,
                    weights,
                ],
                outputs=[self._data.twist_delta],
                device=self.device,
            )

    def _sweep(self, color, world_active, weights, velocities, world_failed, smoothing=False) -> None:
        p = self.problem_data
        if self._worker_count == 0:
            return
        weights, particle_inverse_weight = weights
        projected_twist, projected_velocity = velocities
        colored = self.coloring is not None and not smoothing
        box = self._data.coloring.box if colored else None
        contact = self._data.coloring.contact if colored else None
        spatial = self._data.spatial_smoothing_blocks if smoothing else self._data.spatial_blocks
        particles = self._particles
        rows = p.particle_contact_rows
        particle = self._data.coloring.particle if colored and particles else None
        if particles:
            particle_delassus = self._data.particle_smoothing_delassus if smoothing else self._data.particle_delassus
            particle_occupancy = self._data.coloring.particle_occupancy if colored else p.particle_incidence
        wp.launch(
            make_sweep_kernel(spatial is not None, particles),
            dim=self._worker_count,
            inputs=[
                self._worker_count,
                color,
                box.order if colored else None,
                box.color_offset if colored else None,
                box.color_count if colored else None,
                p.box_rows.world,
                p.box_rows.local,
                p.box_rows.world_count,
                p.box_rows.body_a,
                p.box_rows.body_b,
                p.box_rows.jacobian_a,
                p.box_rows.jacobian_b,
                p.box_rows.bias,
                p.box_rows.lower,
                p.box_rows.upper,
                self._data.box_smoothing_delassus if smoothing else self._data.box_delassus,
                contact.order if colored else None,
                contact.color_offset if colored else None,
                contact.color_count if colored else None,
                p.contact_rows.world,
                p.contact_rows.local,
                p.contact_rows.world_count,
                p.contact_rows.body_a,
                p.contact_rows.body_b,
                p.contact_rows.jacobian_a,
                p.contact_rows.jacobian_b,
                p.contact_rows.bias,
                p.contact_rows.friction,
                self._data.contact_smoothing_delassus if smoothing else self._data.contact_delassus,
                p.contact_rows.normal_compliance,
                p.contact_rows.restitution,
                p.contact_rows.frame,
                p.contact_rows.angular_friction,
                spatial.delassus if spatial is not None else None,
                spatial.eigenvectors if spatial is not None else None,
                spatial.eigenvalues if spatial is not None else None,
                particle.order if particle is not None else None,
                particle.color_offset if particle is not None else None,
                particle.color_count if particle is not None else None,
                rows.world if particles else None,
                rows.status if particles else None,
                rows.particle if particles else None,
                rows.coefficient if particles else None,
                rows.body if particles else None,
                rows.frame if particles else None,
                rows.body_jacobian if particles else None,
                rows.bias if particles else None,
                rows.friction if particles else None,
                particle_delassus if particles else None,
                particle_inverse_weight,
                particle_occupancy if particles else None,
                projected_velocity,
                world_active,
                weights,
                self._data.coloring.occupancy if colored else p.body_incidence,
                projected_twist,
            ],
            outputs=[
                p.box_rows.reaction,
                p.contact_rows.reaction,
                p.contact_rows.angular_reaction,
                rows.reaction if particles else None,
                self._data.twist_delta,
                self._data.particle_delta,
                world_failed,
            ],
            device=self.device,
            block_dim=_SWEEP_BLOCK_DIM,
        )

    def _apply(self, body_world, world_active, world_failed, velocities) -> None:
        projected_twist, projected_velocity = velocities
        wp.launch(
            _apply_twist_delta,
            dim=projected_twist.shape[0],
            inputs=[body_world, world_active, world_failed],
            outputs=[self._data.twist_delta, projected_twist],
            device=self.device,
        )
        if self._particles and projected_velocity.shape[0] > 0:
            wp.launch(
                _apply_particle_delta,
                dim=projected_velocity.shape[0],
                inputs=[self.problem_data.particle_world, world_active, world_failed],
                outputs=[self._data.particle_delta, projected_velocity],
                device=self.device,
            )

    def _project_by_world(self, iterations, world_active, weights, projected_twist, world_failed) -> None:
        p = self.problem_data
        coloring = self._data.coloring
        spatial = self._data.spatial_blocks
        smoothing = self._data.spatial_smoothing_blocks
        acceleration_state = NoAccelerationState()
        if self.anderson is not None:
            acceleration_state = self.anderson.world_state
        elif self.apgd is not None:
            acceleration_state = self.apgd.row_state
        wp.launch(
            make_project_by_world_kernel(
                spatial is not None, self.coloring is not None, self._acceleration, p.max_world_body_count
            ),
            dim=(p.num_worlds, self._world_block_dim),
            block_dim=self._world_block_dim,
            inputs=[
                iterations,
                self.color_count,
                p.world_body_offset,
                p.world_body_count,
                p.box_rows.capacity,
                p.box_rows.world_offset,
                p.box_rows.world_count,
                p.box_rows.body_a,
                p.box_rows.body_b,
                p.box_rows.jacobian_a,
                p.box_rows.jacobian_b,
                p.box_rows.bias,
                p.box_rows.lower,
                p.box_rows.upper,
                self._data.box_delassus,
                self._data.box_smoothing_delassus,
                p.contact_rows.world_offset,
                p.contact_rows.world_count,
                p.contact_rows.body_a,
                p.contact_rows.body_b,
                p.contact_rows.jacobian_a,
                p.contact_rows.jacobian_b,
                p.contact_rows.bias,
                p.contact_rows.friction,
                self._data.contact_delassus,
                self._data.contact_smoothing_delassus,
                p.contact_rows.normal_compliance,
                p.contact_rows.restitution,
                p.contact_rows.frame,
                p.contact_rows.angular_friction,
                spatial.delassus if spatial is not None else None,
                spatial.eigenvectors if spatial is not None else None,
                spatial.eigenvalues if spatial is not None else None,
                smoothing.delassus if smoothing is not None else None,
                smoothing.eigenvectors if smoothing is not None else None,
                smoothing.eigenvalues if smoothing is not None else None,
                acceleration_state,
                coloring.world_color_count if coloring is not None else None,
                coloring.world_color_box_count if coloring is not None else None,
                coloring.world_order if coloring is not None else None,
                coloring.occupancy if coloring is not None else None,
                p.body_incidence,
                world_active,
                weights,
            ],
            outputs=[
                p.box_rows.reaction,
                p.contact_rows.reaction,
                p.contact_rows.angular_reaction,
                projected_twist,
                self._data.twist_delta,
                world_failed,
            ],
            device=self.device,
        )
