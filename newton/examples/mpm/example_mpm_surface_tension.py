# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Surface tension on a fluid cube using MPM.

A cube of fluid particles is initialized with surface tension enabled.
Without a ground plane (default), the CSF force pulls corners inward in zero
gravity, causing the cube to evolve toward a spherical shape.

With ``--ground-plane``, the cube rests on a solid surface under gravity and
a contact angle can be prescribed via ``--contact-angle`` (degrees). The
contact angle controls wetting: 90 is neutral, smaller values are hydrophilic,
larger values are hydrophobic.
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


class Example:
    def __init__(self, viewer, options):
        self.fps = options.fps
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = options.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        builder = newton.ModelBuilder()

        SolverImplicitMPM.register_custom_attributes(builder)

        # Cube of fluid particles — spacing linked to voxel size
        particles_per_cell_axis = options.particles_per_cell_axis
        spacing = options.voxel_size / particles_per_cell_axis
        half = options.cube_size / 2.0
        n = max(1, int(round(options.cube_size / spacing)))

        cell_volume = spacing**3
        mass = cell_volume * options.density
        radius = spacing * 0.5

        # Position the cube depending on the scene mode
        if options.corner:
            # Resting in a ground + wall corner
            origin = wp.vec3(radius, -half, radius)
        elif options.ground_plane:
            # Resting on a ground plane
            origin = wp.vec3(-half, -half, radius)
        else:
            # Floating in zero gravity
            origin = wp.vec3(-half, -half, -half)

        builder.add_particle_grid(
            pos=origin,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=n,
            dim_y=n,
            dim_z=n,
            cell_x=spacing,
            cell_y=spacing,
            cell_z=spacing,
            mass=mass,
            jitter=0.0,
            radius_mean=radius,
        )

        builder.default_shape_cfg.mu = options.collider_friction

        if options.ground_plane or options.corner:
            builder.add_shape_plane()  # z=0 ground, normal +Z
        if options.corner:
            builder.add_shape_plane(plane=[1.0, 0.0, 0.0, 0.0])  # x=0 wall, normal +X

        self.model = builder.finalize()

        if options.ground_plane or options.corner:
            self.model.set_gravity(options.gravity)
        else:
            self.model.set_gravity((0.0, 0.0, 0.0))

        # Fluid material: no shear resistance, allows tension, viscous damping
        self.model.mpm.tensile_yield_ratio.fill_(1.0)
        self.model.mpm.friction.fill_(0.0)
        self.model.mpm.viscosity.fill_(options.viscosity)
        self.model.mpm.surface_tension.fill_(options.surface_tension)

        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size = options.voxel_size
        mpm_options.tolerance = options.tolerance
        mpm_options.max_iterations = options.max_iterations
        mpm_options.grid_type = "sparse"
        mpm_options.collider_basis = "pic"
        mpm_options.contact_angle_mode = options.contact_angle_mode

        self.solver = SolverImplicitMPM(self.model, mpm_options)

        if options.ground_plane or options.corner:
            contact_angle_rad = math.radians(options.contact_angle)
            self.solver.setup_collider(collider_contact_angle=[contact_angle_rad])

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.ground_plane = options.ground_plane or options.corner
        self.show_surface = options.show_surface

        self.color_mode = options.color_mode
        self.particle_colors = wp.full(
            shape=self.model.particle_count, value=wp.vec3(0.2, 0.4, 0.8), device=self.model.device
        )

        self.viewer.show_particles = options.show_particles
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "camera"):
            if options.corner:
                self.viewer.set_camera(pos=wp.vec3(0.15, -0.15, half), pitch=0.0, yaw=120.0)
            else:
                self.viewer.set_camera(pos=wp.vec3(0.15, -0.15, 0.1), pitch=0.0, yaw=150.0)

        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        positions = self.state_0.particle_q.numpy()
        if self.ground_plane:
            # With ground plane: just check particles haven't fallen through
            if np.any(positions[:, 2] < -0.01):
                raise ValueError("Particles fell through the ground plane")
        else:
            center = positions.mean(axis=0)
            dists = np.linalg.norm(positions - center, axis=1)
            cv = np.std(dists) / np.mean(dists)
            if cv > 0.3:
                raise ValueError(f"Particles not becoming spherical (CV={cv:.3f})")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        # Particle coloring (only when viewer.show_particles is on)
        if self.color_mode != "none" and self.viewer.show_particles:
            fields = self.solver.gather_surface_tension_fields()
            if fields:
                if self.color_mode == "curvature":
                    s = fields["curvature"].numpy()
                elif self.color_mode == "indicator":
                    s = fields["indicator"].numpy()
                else:
                    s = np.zeros(self.model.particle_count)
                self._apply_colormap(s)

            self.viewer.log_points(
                name="/model/particles",
                points=self.state_0.particle_q,
                radii=self.model.particle_radius,
                colors=self.particle_colors,
            )

        # Reconstructed surface mesh (from the solver's internal SDF)
        surface = getattr(self.solver, "_st_surface", None)
        if surface is not None and surface.verts is not None and surface.verts.shape[0] > 0:
            normals = surface.normals
            if normals is None:
                from newton._src.utils.mesh import compute_vertex_normals

                normals = compute_vertex_normals(surface.verts, surface.indices)
            self.viewer.log_mesh(
                "/model/particle_surface", surface.verts, surface.indices, normals,
                dynamic=True, hidden=not self.show_surface,
            )

        self.viewer.end_frame()

    def _apply_colormap(self, values):
        """Blue->green->red colormap from 10th to 90th percentile."""
        s_min, s_max = np.percentile(values, [10, 90])
        s_range = s_max - s_min if s_max > s_min else 1.0
        s_norm = np.clip((values - s_min) / s_range, 0.0, 1.0)

        colors_np = np.zeros((len(s_norm), 3), dtype=np.float32)

        low = s_norm < 0.5
        t1 = s_norm[low] / 0.5
        colors_np[low, 0] = 0.0
        colors_np[low, 1] = t1
        colors_np[low, 2] = 1.0 - t1

        high = ~low
        t2 = (s_norm[high] - 0.5) / 0.5
        colors_np[high, 0] = t2
        colors_np[high, 1] = 1.0 - t2
        colors_np[high, 2] = 0.0

        self.particle_colors.assign(colors_np)

    def render_ui(self, imgui):
        _, self.show_surface = imgui.checkbox("Show Surface", self.show_surface)

        for mode in ["none", "indicator", "curvature"]:
            clicked, _ = imgui.selectable(mode.capitalize(), self.color_mode == mode)
            if clicked:
                self.color_mode = mode

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        # Scene
        parser.add_argument("--cube-size", type=float, default=0.05, help="Side length of the initial cube [m]")
        parser.add_argument("--particles-per-cell-axis", type=int, default=3, help="Particles per voxel per axis")
        parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -9.81])
        parser.add_argument("--ground-plane", action="store_true", default=False, help="Add a ground plane collider")
        parser.add_argument("--corner", action="store_true", default=False, help="Add ground + wall corner colliders")
        parser.add_argument("--contact-angle", type=float, default=90.0, help="Contact angle [degrees] (with --ground-plane or --corner)")
        parser.add_argument("--collider-friction", type=float, default=0.5, help="Collider friction coefficient")
        parser.add_argument("--contact-angle-mode", type=str, default="union",
                            choices=["force", "sdf", "union", "virtual"],
                            help="Contact angle enforcement mode")
        parser.add_argument("--fps", type=float, default=120.0)
        parser.add_argument("--substeps", type=int, default=4)

        # Material
        parser.add_argument("--density", type=float, default=1000.0)
        parser.add_argument("--surface-tension", "-st", type=float, default=1.0, help="Surface tension coefficient")
        parser.add_argument("--viscosity", type=float, default=1.0, help="Viscosity [Pa*s]")

        # Visualization
        parser.add_argument(
            "--color-mode", type=str, default="curvature", choices=["none", "indicator", "curvature"],
            help="Particle color mode",
        )
        parser.add_argument("--show-surface", action="store_true", default=True, help="Show reconstructed surface mesh")
        parser.add_argument("--no-surface", dest="show_surface", action="store_false", help="Hide surface mesh")
        parser.add_argument("--show-particles", action="store_true", default=False, help="Show particles")

        # Solver
        parser.add_argument("--max-iterations", "-it", type=int, default=100)
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-4)
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.004)

        return parser


if __name__ == "__main__":
    parser = Example.create_parser()

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
