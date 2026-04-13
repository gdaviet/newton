# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Surface tension on a fluid cube using MPM.

A cube of fluid particles is initialized in zero gravity with surface tension
enabled. The CSF force pulls corners inward, causing the cube to evolve
toward a spherical shape.
"""

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

        builder.add_particle_grid(
            pos=wp.vec3(-half, -half, -half),
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

        self.model = builder.finalize()
        self.model.set_gravity(options.gravity)

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

        self.solver = SolverImplicitMPM(self.model, mpm_options)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.color_mode = options.color_mode
        self.particle_colors = wp.full(
            shape=self.model.particle_count, value=wp.vec3(0.2, 0.4, 0.8), device=self.model.device
        )

        self.viewer.show_particles = True
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "camera"):
            self.viewer.set_camera(pos=wp.vec3(0.15, -0.15, 0.0), pitch=0.0, yaw=150.0)

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
        center = positions.mean(axis=0)
        dists = np.linalg.norm(positions - center, axis=1)
        cv = np.std(dists) / np.mean(dists)
        if cv > 0.3:
            raise ValueError(f"Particles not becoming spherical (CV={cv:.3f})")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        if self.color_mode != "none":
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

        self.viewer.end_frame()

    def _apply_colormap(self, values):
        """Blue→green→red colormap from 10th to 90th percentile."""
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
        changed = False
        for mode in ["none", "indicator", "curvature"]:
            clicked, _ = imgui.selectable(mode.capitalize(), self.color_mode == mode)
            if clicked:
                self.color_mode = mode
                changed = True

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        # Scene
        parser.add_argument("--cube-size", type=float, default=0.05, help="Side length of the initial cube [m]")
        parser.add_argument("--particles-per-cell-axis", type=int, default=2, help="Particles per voxel per axis")
        parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, 0])
        parser.add_argument("--fps", type=float, default=120.0)
        parser.add_argument("--substeps", type=int, default=4)

        # Material
        parser.add_argument("--density", type=float, default=1000.0)
        parser.add_argument("--surface-tension", "-st", type=float, default=50.0, help="Surface tension [N/m]")
        parser.add_argument("--viscosity", type=float, default=10.0, help="Viscosity [Pa*s]")

        # Visualization
        parser.add_argument(
            "--color-mode", type=str, default="curvature", choices=["none", "indicator", "curvature"],
            help="Particle color mode",
        )

        # Solver
        parser.add_argument("--max-iterations", "-it", type=int, default=100)
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-4)
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.008)

        return parser


if __name__ == "__main__":
    parser = Example.create_parser()

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
