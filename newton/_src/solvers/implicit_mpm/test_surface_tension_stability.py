# Quick B2 indicator test: sphere stability + cube rounding

import numpy as np
import warp as wp
import newton
from newton.solvers import SolverImplicitMPM

wp.init()


def make_model(positions, spacing, density=1000.0, sigma=50.0, viscosity=10.0, voxel_size=0.008):
    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)
    mass = spacing**3 * density
    radius = spacing * 0.5
    for p in positions:
        builder.add_particle(pos=tuple(p), vel=(0, 0, 0), mass=mass, radius=radius)
    model = builder.finalize(device="cuda:0")
    model.set_gravity([0.0, 0.0, 0.0])
    model.mpm.tensile_yield_ratio.fill_(1.0)
    model.mpm.friction.fill_(0.0)
    model.mpm.viscosity.fill_(viscosity)
    model.mpm.surface_tension.fill_(sigma)
    config = SolverImplicitMPM.Config(
        voxel_size=voxel_size, grid_type="sparse",
        max_iterations=100, tolerance=1e-4,
    )
    return SolverImplicitMPM(model, config), model


def make_sphere(R, spacing):
    n = int(2 * R / spacing) + 1
    half = (n - 1) * spacing / 2.0
    return np.array([
        [-half + i * spacing, -half + j * spacing, -half + k * spacing]
        for i in range(n) for j in range(n) for k in range(n)
        if np.linalg.norm([-half + i * spacing, -half + j * spacing, -half + k * spacing]) <= R
    ])


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


fps = 120.0
substeps = 4
dt = 1.0 / (fps * substeps)

# Sphere
print("=== Sphere (sigma=50/3, 60 frames) ===")
sphere_pts = make_sphere(0.025, 0.004)
solver, model = make_model(sphere_pts, 0.004, sigma=50.0 / 3.0)
s0, s1 = model.state(), model.state()
ctrl, cont = model.control(), model.contacts()
rms_init = None
for frame in range(60):
    for _ in range(substeps):
        solver.step(s0, s1, ctrl, cont, dt)
        s0, s1 = s1, s0
    pos = s0.particle_q.numpy()
    rms = np.sqrt(np.mean(np.sum((pos - pos.mean(0)) ** 2, 1)))
    if rms_init is None:
        rms_init = rms
    if (frame + 1) % 20 == 0:
        drift = (rms - rms_init) / rms_init * 100
        vel = s0.particle_qd.numpy()
        maxv = np.linalg.norm(vel, axis=1).max()
        print(f"  frame {frame+1}: drift={drift:+.2f}%  max_vel={maxv:.4e}")

# Cube
print("\n=== Cube (sigma=50, 60 frames) ===")
cube_pts = make_cube(10, 0.005)
solver, model = make_model(cube_pts, 0.005, sigma=50.0)
s0, s1 = model.state(), model.state()
ctrl, cont = model.control(), model.contacts()
ratio_init = corner_face_ratio(s0.particle_q.numpy())
for frame in range(60):
    for _ in range(substeps):
        solver.step(s0, s1, ctrl, cont, dt)
        s0, s1 = s1, s0
    if (frame + 1) % 20 == 0:
        pos = s0.particle_q.numpy()
        ratio = corner_face_ratio(pos)
        vel = s0.particle_qd.numpy()
        maxv = np.linalg.norm(vel, axis=1).max()
        print(f"  frame {frame+1}: ratio={ratio:.4f}  max_vel={maxv:.4e}")

ratio_final = corner_face_ratio(s0.particle_q.numpy())
print(f"\nCube ratio: {ratio_init:.4f} → {ratio_final:.4f}")
