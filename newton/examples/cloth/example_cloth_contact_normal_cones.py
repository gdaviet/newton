# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Contact Normal Cones
#
# Two overlapping cloth patches rest on an inclined plane. LOX advances the
# scene with normal-cone filtering enabled, while a diagnostic LOX instance
# evaluates the same frozen candidates without filtering. Every detected
# contact is yellow; its normal is green when kept and magenta when filtered.
# Right-drag the orange upper cloth to pull it with an example-local spring.
#
# Command: python -m newton.examples cloth_contact_normal_cones
#
###########################################################################

import math

import warp as wp

import newton
import newton.examples

_CONTACT_STATUS_VALID = wp.constant(1)
_COEFFICIENT_TOLERANCE = wp.constant(1.0e-5)


@wp.func
def _pick_lock_acquire(lock: wp.array[wp.int32]):
    while wp.atomic_cas(lock, 0, 0, 1) == 1:
        pass


@wp.func
def _pick_lock_release(lock: wp.array[wp.int32]):
    wp.atomic_exch(lock, 0, 0)


@wp.kernel
def _raycast_upper_cloth(
    triangle_start: int,
    particle_q: wp.array[wp.vec3],
    particle_inv_mass: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    lock: wp.array[wp.int32],
    minimum_distance: wp.array[float],
    picked_triangle: wp.array[wp.int32],
    picked_barycentric: wp.array[wp.vec3],
):
    local_triangle = wp.tid()
    triangle = triangle_start + local_triangle
    particle_0 = tri_indices[triangle, 0]
    particle_1 = tri_indices[triangle, 1]
    particle_2 = tri_indices[triangle, 2]
    if (
        particle_inv_mass[particle_0] <= 0.0
        and particle_inv_mass[particle_1] <= 0.0
        and particle_inv_mass[particle_2] <= 0.0
    ):
        return

    point_0 = particle_q[particle_0]
    edge_0 = particle_q[particle_1] - point_0
    edge_1 = particle_q[particle_2] - point_0
    cross_direction_edge = wp.cross(ray_direction, edge_1)
    determinant = wp.dot(edge_0, cross_direction_edge)
    if wp.abs(determinant) <= 1.0e-8:
        return
    inverse_determinant = 1.0 / determinant
    origin_offset = ray_origin - point_0
    barycentric_1 = wp.dot(origin_offset, cross_direction_edge) * inverse_determinant
    if barycentric_1 < 0.0 or barycentric_1 > 1.0:
        return
    cross_origin_edge = wp.cross(origin_offset, edge_0)
    barycentric_2 = wp.dot(ray_direction, cross_origin_edge) * inverse_determinant
    if barycentric_2 < 0.0 or barycentric_1 + barycentric_2 > 1.0:
        return
    distance = wp.dot(edge_1, cross_origin_edge) * inverse_determinant
    if distance < 0.0 or distance >= minimum_distance[0]:
        return

    _pick_lock_acquire(lock)
    old_minimum = wp.atomic_min(minimum_distance, 0, distance)
    if distance <= old_minimum:
        picked_triangle[0] = triangle
        picked_barycentric[0] = wp.vec3(1.0 - barycentric_1 - barycentric_2, barycentric_1, barycentric_2)
    _pick_lock_release(lock)


@wp.kernel
def _update_cloth_pick_target(
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    pick_target: wp.array[wp.vec3],
):
    target_distance = wp.length(pick_target[0] - ray_origin)
    pick_target[0] = ray_origin + target_distance * ray_direction


@wp.kernel
def _apply_cloth_pick_force(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    particle_inv_mass: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    picked_triangle: wp.array[wp.int32],
    picked_barycentric: wp.array[wp.vec3],
    pick_target: wp.array[wp.vec3],
    pick_stiffness: float,
    pick_damping: float,
    pick_max_acceleration: float,
    picked_point: wp.array[wp.vec3],
):
    triangle = picked_triangle[0]
    if triangle < 0:
        return

    barycentric = picked_barycentric[0]
    point = wp.vec3(0.0)
    velocity = wp.vec3(0.0)
    inverse_effective_mass = float(0.0)
    for slot in range(3):
        particle = tri_indices[triangle, slot]
        weight = barycentric[slot]
        point += weight * particle_q[particle]
        velocity += weight * particle_qd[particle]
        inverse_effective_mass += weight * weight * particle_inv_mass[particle]
    picked_point[0] = point
    if inverse_effective_mass <= 0.0:
        return

    force = pick_stiffness * (pick_target[0] - point) - pick_damping * velocity
    maximum_force = pick_max_acceleration * 9.81 / inverse_effective_mass
    force_magnitude = wp.length(force)
    if force_magnitude > maximum_force:
        force *= maximum_force / force_magnitude

    for slot in range(3):
        particle = tri_indices[triangle, slot]
        if particle_inv_mass[particle] > 0.0:
            wp.atomic_add(particle_f, particle, barycentric[slot] * force)


@wp.kernel
def _classify_contact_points(
    contact_offset: int,
    unfiltered_status: wp.array[wp.int32],
    filtered_status: wp.array[wp.int32],
    particle_indices: wp.array2d[wp.int32],
    coefficients: wp.array2d[float],
    packed_to_newton: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    frame: wp.array[wp.mat33f],
    arrow_length: float,
    detected_points: wp.array[wp.vec3],
    detected_normal_ends: wp.array[wp.vec3],
    detected_normal_colors: wp.array[wp.vec3],
    frame_counts: wp.array[wp.int32],
    total_counts: wp.array[wp.int32],
):
    candidate = wp.tid()
    contact = contact_offset + candidate
    if unfiltered_status[contact] != _CONTACT_STATUS_VALID:
        return

    positive_point = wp.vec3(0.0)
    negative_point = wp.vec3(0.0)
    positive_weight = float(0.0)
    negative_weight = float(0.0)
    for slot in range(4):
        packed_particle = particle_indices[contact, slot]
        coefficient = coefficients[contact, slot]
        if packed_particle < 0 or wp.abs(coefficient) <= _COEFFICIENT_TOLERANCE:
            continue
        point = particle_q[packed_to_newton[packed_particle]]
        if coefficient > 0.0:
            positive_point += coefficient * point
            positive_weight += coefficient
        else:
            negative_point -= coefficient * point
            negative_weight -= coefficient

    if positive_weight <= _COEFFICIENT_TOLERANCE or negative_weight <= _COEFFICIENT_TOLERANCE:
        return

    point = 0.5 * (positive_point / positive_weight + negative_point / negative_weight)
    normal_end = point + arrow_length * (frame[contact] @ wp.vec3(0.0, 0.0, 1.0))
    output = wp.atomic_add(frame_counts, 0, 1)
    detected_points[output] = point
    detected_normal_ends[output] = normal_end
    wp.atomic_add(total_counts, 0, 1)

    if filtered_status[contact] == _CONTACT_STATUS_VALID:
        detected_normal_colors[output] = wp.vec3(0.15, 1.0, 0.20)
        return

    detected_normal_colors[output] = wp.vec3(1.0, 0.05, 0.72)
    wp.atomic_add(frame_counts, 1, 1)
    wp.atomic_add(total_counts, 1, 1)


class _UpperClothPicking:
    """Example-local right-drag picking for the upper cloth triangles."""

    def __init__(self, model: newton.Model, triangle_start: int, triangle_end: int):
        self.model = model
        self.triangle_start = triangle_start
        self.triangle_count = triangle_end - triangle_start
        self.pick_stiffness = 200.0
        self.pick_damping = 10.0
        self.pick_max_acceleration = 20.0
        self.picking_active = False
        self.pick_dist = 0.0
        self.visible_worlds_mask = None
        self.world_offsets = None

        device = model.device
        self.pick_body = wp.array([-1], dtype=wp.int32, pinned=device.is_cuda, device=device)
        self.picked_triangle = wp.array([-1], dtype=wp.int32, pinned=device.is_cuda, device=device)
        self.picked_barycentric = wp.zeros(1, dtype=wp.vec3, device=device)
        self.pick_target = wp.zeros(1, dtype=wp.vec3, device=device)
        self.picked_point = wp.zeros(1, dtype=wp.vec3, device=device)
        self.pick_line_color = wp.full(1, wp.vec3(0.0, 1.0, 1.0), device=device)
        self.minimum_distance = wp.full(1, 1.0e10, dtype=wp.float32, device=device)
        self.lock = wp.zeros(1, dtype=wp.int32, device=device)

    def is_picking(self) -> bool:
        """Return whether the upper cloth is being dragged."""
        return self.picking_active

    def release(self) -> None:
        """Release the current upper-cloth pick."""
        self.pick_body.fill_(-1)
        self.picked_triangle.fill_(-1)
        self.picking_active = False

    def pick(self, state: newton.State, ray_start: wp.vec3, ray_dir: wp.vec3) -> None:
        """Pick the nearest upper-cloth triangle intersected by the mouse ray."""
        self.release()
        self.minimum_distance.fill_(1.0e10)
        self.picked_barycentric.zero_()
        self.lock.zero_()
        wp.launch(
            _raycast_upper_cloth,
            dim=self.triangle_count,
            inputs=[
                self.triangle_start,
                state.particle_q,
                self.model.particle_inv_mass,
                self.model.tri_indices,
                ray_start,
                ray_dir,
                self.lock,
            ],
            outputs=[self.minimum_distance, self.picked_triangle, self.picked_barycentric],
            device=self.model.device,
        )
        distance = float(self.minimum_distance.numpy()[0])
        triangle = int(self.picked_triangle.numpy()[0])
        if triangle < 0 or distance >= 1.0e10:
            return
        self.pick_dist = distance
        hit_point = ray_start + distance * ray_dir
        self.pick_target.assign([hit_point])
        self.picked_point.assign([hit_point])
        self.picking_active = True

    def update(self, ray_start: wp.vec3, ray_dir: wp.vec3) -> None:
        """Move the cloth-picking target along the mouse ray."""
        if not self.picking_active:
            return
        wp.launch(
            _update_cloth_pick_target,
            dim=1,
            inputs=[ray_start, ray_dir],
            outputs=[self.pick_target],
            device=self.model.device,
        )

    def _apply_picking_force(self, state: newton.State) -> None:
        """Apply the captured picking spring to the upper cloth."""
        wp.launch(
            _apply_cloth_pick_force,
            dim=1,
            inputs=[
                state.particle_q,
                state.particle_qd,
                state.particle_f,
                self.model.particle_inv_mass,
                self.model.tri_indices,
                self.picked_triangle,
                self.picked_barycentric,
                self.pick_target,
                self.pick_stiffness,
                self.pick_damping,
                self.pick_max_acceleration,
            ],
            outputs=[self.picked_point],
            device=self.model.device,
        )


def _make_lox_config(*, enable_normal_cone_filtering: bool) -> newton.solvers.SolverKamino.Config:
    """Create the LOX configuration shared by simulation and diagnostics."""
    config = newton.solvers.SolverKamino.Config(dynamics_solver="lox")
    config.lox.max_iterations = 12
    config.lox.projection_iterations = 3
    config.lox.deformable_cr_iterations = 6
    config.lox.deformable_enable_self_contact = True
    config.lox.deformable_enable_normal_cone_filtering = enable_normal_cone_filtering
    config.lox.deformable_enable_rigid_contact_normal_cone_filtering = False
    config.lox.deformable_normal_cone_filtering_min_distance = 1.0e-4
    config.lox.deformable_enable_penetration_free_contact = False
    config.lox.deformable_self_contact_margin = 0.035
    config.lox.deformable_self_contact_gap = 0.04
    config.lox.deformable_self_contact_topological_filter_threshold = 0
    config.lox.deformable_self_contact_rest_exclusion_radius = 0.0
    return config


def _add_bumpy_cloth_grid(
    builder: newton.ModelBuilder,
    *,
    pos: wp.vec3,
    rot: wp.quat,
    dim_x: int,
    dim_y: int,
    cell_x: float,
    cell_y: float,
    mass: float,
    amplitude: float,
    fix_left: bool,
    label: str,
) -> tuple[int, int]:
    """Add a sharp eggbox cloth rest mesh with optional left-edge pins."""
    vertices = []
    indices = []
    for y in range(dim_y + 1):
        for x in range(dim_x + 1):
            height = amplitude * math.cos(math.pi * x) * math.cos(math.pi * y)
            vertices.append(wp.vec3(x * cell_x, y * cell_y, height))
            if x > 0 and y > 0:
                vertex_0 = (y - 1) * (dim_x + 1) + x - 1
                vertex_1 = vertex_0 + 1
                vertex_3 = y * (dim_x + 1) + x - 1
                vertex_2 = vertex_3 + 1
                indices.extend((vertex_0, vertex_1, vertex_3, vertex_1, vertex_2, vertex_3))

    particle_start = builder.particle_count
    triangle_start = builder.tri_count
    planar_area = dim_x * cell_x * dim_y * cell_y
    density = mass * len(vertices) / planar_area
    builder.add_cloth_mesh(
        pos=pos,
        rot=rot,
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=vertices,
        indices=indices,
        density=density,
        tri_ke=5.0e2,
        tri_ka=5.0e2,
        tri_kd=5.0,
        edge_ke=0.5,
        edge_kd=0.05,
        particle_radius=0.01,
        label=label,
    )
    for y in range(dim_y + 1):
        for x in range(dim_x + 1):
            particle = particle_start + y * (dim_x + 1) + x
            builder.particle_mass[particle] = mass
            if x == 0 and fix_left:
                builder.particle_mass[particle] = 0.0
                builder.particle_flags[particle] &= ~newton.ParticleFlags.ACTIVE
    return triangle_start, builder.tri_count


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        slope_angle = 25.0 * wp.pi / 180.0
        slope_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), slope_angle)
        slope_transform = wp.transform(wp.vec3(0.0), slope_rotation)
        incident_rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, -1.0, 0.0)), 6.0 * wp.pi / 180.0)
        slope_cfg = newton.ModelBuilder.ShapeConfig(mu=0.45)
        builder.add_shape_plane(
            xform=slope_transform,
            width=2.4,
            length=1.6,
            cfg=slope_cfg,
            color=(0.18, 0.20, 0.24),
            label="inclined plane",
        )

        lower_tri_start, lower_tri_end = _add_bumpy_cloth_grid(
            builder,
            pos=wp.transform_point(slope_transform, wp.vec3(-0.45, -0.32, 0.055)),
            rot=slope_rotation,
            dim_x=6,
            dim_y=4,
            cell_x=0.15,
            cell_y=0.16,
            mass=2.0e-3,
            amplitude=0.04,
            fix_left=True,
            label="lower cloth",
        )
        self.cloth_tri_ranges = [(lower_tri_start, lower_tri_end, (0.20, 0.48, 0.95))]

        upper_tri_start, upper_tri_end = _add_bumpy_cloth_grid(
            builder,
            pos=wp.transform_point(slope_transform, wp.vec3(-0.30, -0.16, 0.080)),
            rot=slope_rotation * incident_rotation,
            dim_x=4,
            dim_y=3,
            cell_x=0.18,
            cell_y=0.16,
            mass=2.0e-3,
            amplitude=0.04,
            fix_left=False,
            label="upper cloth",
        )
        self.cloth_tri_ranges.append((upper_tri_start, upper_tri_end, (0.95, 0.48, 0.12)))

        builder.color(include_bending=True)
        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e2
        # LOX currently uses one deformable-contact coefficient for the model.
        # The lower sheet stays anchored, while this near-critical value lets
        # the free upper sheet drift slowly down the 25-degree slope.
        self.model.soft_contact_mu = 0.43

        self.solver = newton.solvers.SolverKamino(
            self.model,
            config=_make_lox_config(enable_normal_cone_filtering=True),
        )
        self.unfiltered_solver = newton.solvers.SolverKamino(
            self.model,
            config=_make_lox_config(enable_normal_cone_filtering=False),
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_margin=0.015)
        self.contacts = self.collision_pipeline.contacts()

        # These implementation-owned arrays are intentionally used only by this
        # diagnostic example so the displayed classification exactly matches LOX.
        filtered_lox = self.solver._solver_kamino.solver_fd
        unfiltered_lox = self.unfiltered_solver._solver_kamino.solver_fd
        self.filtered_contact_system = filtered_lox.deformable_contacts
        self.unfiltered_contact_system = unfiltered_lox.deformable_contacts
        self.self_contact_capacity = filtered_lox.deformable_self_contact_detector.capacity
        self.contact_frame_counts = wp.zeros(2, dtype=wp.int32, device=self.model.device)
        self.contact_total_counts = wp.zeros(2, dtype=wp.int32, device=self.model.device)
        self.detected_points = wp.empty(self.self_contact_capacity, dtype=wp.vec3, device=self.model.device)
        self.detected_normal_ends = wp.empty_like(self.detected_points)
        self.detected_normal_colors = wp.empty_like(self.detected_points)
        self.detected_colors = wp.full(
            self.self_contact_capacity,
            wp.vec3(1.0, 0.75, 0.08),
            device=self.model.device,
        )

        self.viewer.set_model(self.model)
        self.upper_cloth_picking = None
        viewer_picking = getattr(self.viewer, "picking", None)
        if viewer_picking is not None:
            upper_tri_start, upper_tri_end, _ = self.cloth_tri_ranges[1]
            self.upper_cloth_picking = _UpperClothPicking(self.model, upper_tri_start, upper_tri_end)
            self.upper_cloth_picking.visible_worlds_mask = viewer_picking.visible_worlds_mask
            self.upper_cloth_picking.world_offsets = viewer_picking.world_offsets
            self.viewer.picking = self.upper_cloth_picking
        self.viewer.show_triangles = False
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.renderer.arrow_scale = 2.0
            self.viewer.set_camera(pos=wp.vec3(1.55, -1.65, 1.15), pitch=-18.0, yaw=43.0)
            if hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.0, 0.0, 0.15))

        self.capture()

    def _prepare_unfiltered_contacts(self):
        """Evaluate the current candidates without normal-cone filtering."""
        self.unfiltered_solver._solver_kamino.prepare_lox_deformables(
            state=self.state_0,
            contacts=self.contacts,
            dt=self.sim_dt,
        )

    def _update_contact_visualization(self):
        """Classify matching unfiltered and filtered LOX candidate slots."""
        filtered_lox = self.solver._solver_kamino.solver_fd
        unfiltered_lox = self.unfiltered_solver._solver_kamino.solver_fd
        self.filtered_contact_system = filtered_lox.deformable_contacts
        self.unfiltered_contact_system = unfiltered_lox.deformable_contacts
        self.contact_frame_counts.zero_()
        wp.launch(
            _classify_contact_points,
            dim=self.self_contact_capacity,
            inputs=[
                self.unfiltered_contact_system.rigid_contact_capacity,
                self.unfiltered_contact_system.status,
                self.filtered_contact_system.status,
                self.unfiltered_contact_system.particle_indices,
                self.unfiltered_contact_system.coefficients,
                self.unfiltered_contact_system.cloth_system.topology.packed_to_newton,
                self.state_0.particle_q,
                self.unfiltered_contact_system.frame,
                0.055,
            ],
            outputs=[
                self.detected_points,
                self.detected_normal_ends,
                self.detected_normal_colors,
                self.contact_frame_counts,
                self.contact_total_counts,
            ],
            device=self.model.device,
        )

    def simulate(self):
        """Advance one frame and record the final substep's cone decisions."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self._prepare_unfiltered_contacts()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self._update_contact_visualization()

    def _warmup_lox_capture(self):
        """Allocate lazy LOX contact storage without advancing the example."""
        state_0_reference = self.state_0
        state_1_reference = self.state_1
        state_0_snapshot = self.model.state()
        state_1_snapshot = self.model.state()
        state_0_snapshot.assign(self.state_0)
        state_1_snapshot.assign(self.state_1)

        self.simulate()

        self.state_0 = state_0_reference
        self.state_1 = state_1_reference
        self.state_0.assign(state_0_snapshot)
        self.state_1.assign(state_1_snapshot)
        self.solver.reset(self.state_0, flags=newton.StateFlags.NONE)
        self.unfiltered_solver.reset(self.state_0, flags=newton.StateFlags.NONE)
        self.contact_frame_counts.zero_()
        self.contact_total_counts.zero_()

        filtered_lox = self.solver._solver_kamino.solver_fd
        unfiltered_lox = self.unfiltered_solver._solver_kamino.solver_fd
        self.filtered_contact_system = filtered_lox.deformable_contacts
        self.unfiltered_contact_system = unfiltered_lox.deformable_contacts

    def capture(self):
        """Capture the simulation after initializing lazy LOX storage."""
        self._warmup_lox_capture()
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph
        self.contact_frame_counts.zero_()
        self.contact_total_counts.zero_()

    def step(self):
        """Advance one rendered frame."""
        wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        """Render cloth and color-coded normal-cone contact decisions."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        for cloth, (tri_start, tri_end, color) in enumerate(self.cloth_tri_ranges):
            self.viewer.log_mesh(
                f"/cloth_{cloth}",
                self.state_0.particle_q,
                self.model.tri_indices[tri_start:tri_end].flatten(),
                color=color,
                backface_culling=False,
            )

        detected_count, filtered_count = self.contact_frame_counts.numpy()
        self._log_detected_contacts(int(detected_count))
        self._log_cloth_pick()
        self.viewer.log_scalar("/normal cone/yellow detected", int(detected_count))
        self.viewer.log_scalar("/normal cone/magenta filtered out", int(filtered_count))
        self.viewer.end_frame()

    def _log_detected_contacts(self, count):
        """Log every contact with a normal colored by its cone decision."""
        if count == 0:
            self.viewer.log_points("/normal_cone/yellow_all_detected", None)
            self.viewer.log_arrows("/normal_cone/yellow_all_detected_normals", None, None, None)
            return
        self.viewer.log_points(
            "/normal_cone/yellow_all_detected",
            self.detected_points[:count],
            radii=0.003,
            colors=self.detected_colors[:count],
        )
        self.viewer.log_arrows(
            "/normal_cone/yellow_all_detected_normals",
            self.detected_points[:count],
            self.detected_normal_ends[:count],
            colors=self.detected_normal_colors[:count],
        )

    def _log_cloth_pick(self) -> None:
        """Log the spring between the mouse target and picked cloth point."""
        picking = self.upper_cloth_picking
        if picking is None or not picking.is_picking():
            self.viewer.log_lines("/upper_cloth_pick", None, None, None)
            return
        self.viewer.log_lines(
            "/upper_cloth_pick",
            picking.picked_point,
            picking.pick_target,
            picking.pick_line_color,
            width=0.006,
        )

    def test_final(self):
        """Keep both cloth patches bounded and exercise both cone outcomes."""
        lower = wp.vec3(-2.0, -2.0, -1.0)
        upper = wp.vec3(2.0, 2.0, 2.0)
        newton.examples.test_particle_state(
            self.state_0,
            "cloth particles remain in a bounded volume",
            lambda q, qd: newton.math.vec_inside_limits(q, lower, upper),
        )
        detected_total, filtered_total = self.contact_total_counts.numpy()
        assert detected_total > 0, "LOX did not detect any cloth contacts."
        assert filtered_total > 0, "Normal-cone filtering did not remove any cloth contacts."


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
