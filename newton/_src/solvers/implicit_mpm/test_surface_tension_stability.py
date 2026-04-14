# Test blur intensity effect on SDF smoothness and rounding

import numpy as np
import warp as wp
import newton
from newton.solvers import SolverImplicitMPM
from newton._src.solvers.implicit_mpm.particle_surface import ParticleSurface

wp.init()


def make_cube(n, spacing):
    half = (n - 1) * spacing / 2.0
    return np.array([
        [-half + i * spacing, -half + j * spacing, -half + k * spacing]
        for i in range(n) for j in range(n) for k in range(n)
    ])


def corner_face_ratio(pos):
    c = pos.mean(axis=0)
    d = np.linalg.norm(pos - c, axis=1)
    med = np.median(d)
    return d[d > med].mean() / d[d <= med].mean()


cube_pts = make_cube(10, 0.005)
fps = 120.0
substeps = 4
dt = 1.0 / (fps * substeps)
sigma = 30.0
solver_dx = 0.008

configs = [
    ("blur r=3 it=2 (baseline)", 3, 2),
    ("blur r=3 it=4", 3, 4),
    ("blur r=3 it=6", 3, 6),
    ("blur r=5 it=4", 5, 4),
    ("blur r=5 it=6", 5, 6),
]

for name, blur_r, blur_it in configs:
    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)
    for p in cube_pts:
        builder.add_particle(pos=tuple(p), vel=(0, 0, 0), mass=0.005**3 * 1000, radius=0.0025)
    model = builder.finalize(device="cuda:0")
    model.set_gravity([0.0, 0.0, 0.0])
    model.mpm.tensile_yield_ratio.fill_(1.0)
    model.mpm.friction.fill_(0.0)
    model.mpm.viscosity.fill_(10.0)
    model.mpm.surface_tension.fill_(sigma)
    solver = SolverImplicitMPM(model, SolverImplicitMPM.Config(
        voxel_size=solver_dx, grid_type="sparse", max_iterations=100, tolerance=1e-4))

    solver._st_surface = ParticleSurface(
        voxel_size=0.5 * solver_dx,
        kernel_radius=3.0 * solver_dx,
        field_smooth_radius=blur_r,
        field_smooth_iterations=blur_it,
    )

    # Check SDF bumpiness: measure curvature std on sphere
    solver._st_surface.extract(model.particle_q, radii=solver._mpm_model.particle_radius, compute_normals=False)
    sdf = solver._st_surface.field.numpy()
    near_surface = np.abs(sdf) < 2 * 0.5 * solver_dx
    sdf_std = sdf[near_surface].std() if near_surface.any() else 0

    # Run sim
    s0, s1 = model.state(), model.state()
    ctrl, cont = model.control(), model.contacts()
    for frame in range(30):
        for _ in range(substeps):
            solver.step(s0, s1, ctrl, cont, dt)
            s0, s1 = s1, s0

    ratio = corner_face_ratio(s0.particle_q.numpy())
    maxv = np.linalg.norm(s0.particle_qd.numpy(), axis=1).max()
    print(f"{name:30s}: ratio={ratio:.4f}  vel={maxv:.4e}  sdf_std_surface={sdf_std:.4f}")
