# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MPM surface extraction example.

Demonstrates extracting a smooth triangle mesh from MPM particle
simulations using :class:`ParticleSurface` and Warp's marching cubes.

Supports isotropic and anisotropic (deformation-aware) splatting,
Gaussian field blur, and Taubin mesh smoothing.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


class Example:
    def __init__(self, viewer, args):
        self.fps = 60.0
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)

        Example.emit_particles(builder, args)
        builder.add_ground_plane()
        self.model = builder.finalize()

        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size = args.voxel_size
        self.solver = SolverImplicitMPM(self.model, mpm_options)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Surface extraction context (Yu & Turk 2010 anisotropic kernels)
        self.surface_ctx = self.solver.create_particle_surface(
            voxel_size=args.surface_voxel_size,
            kernel_radius=args.kernel_radius,
            threshold=args.threshold,
            mesh_smooth_iterations=args.mesh_smooth_iterations,
            anisotropic=args.anisotropic,
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = not args.hide_particles

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -0.05,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        verts, indices, normals = self.solver.extract_particle_surface(
            self.state_0, self.surface_ctx,
        )

        if verts is not None and verts.shape[0] > 0:
            self.viewer.log_mesh(
                "/model/particle_surface",
                verts,
                indices,
                normals,
                backface_culling=False,
            )

        self.viewer.end_frame()

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        voxel_size = args.voxel_size
        particles_per_cell = 3
        particle_lo = np.array([-0.5, -0.5, 0.0])
        particle_hi = np.array([0.5, 0.5, 2.0])
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        cell_size = (particle_hi - particle_lo) / particle_res
        cell_volume = np.prod(cell_size)
        radius = np.max(cell_size) * 0.5
        mass = np.prod(cell_volume) * 2500.0

        builder.add_particle_grid(
            pos=wp.vec3(particle_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=particle_res[0] + 1,
            dim_y=particle_res[1] + 1,
            dim_z=particle_res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.1)
        parser.add_argument("--surface-voxel-size", type=float, default=None,
                            help="Voxel size for the surface grid (default: same as --voxel-size)")
        parser.add_argument("--kernel-radius", type=float, default=None,
                            help="Splatting kernel radius (default: 3 * surface_voxel_size)")
        parser.add_argument("--threshold", type=float, default=0.2,
                            help="Isosurface level (field ~1.0 inside, default 0.5)")
        parser.add_argument("--mesh-smooth-iterations", type=int, default=3,
                            help="Taubin mesh smoothing passes (default 3)")
        parser.add_argument("--anisotropic", action="store_true",
                            help="Enable per-particle WPCA anisotropic kernels")
        parser.add_argument("--hide-particles", action="store_true",
                            help="Hide particle spheres, show only extracted surface")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    if args.surface_voxel_size is None:
        args.surface_voxel_size = args.voxel_size

    example = Example(viewer, args)
    newton.examples.run(example, args)
