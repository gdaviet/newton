# Test stability at different dt values to find the working regime

import numpy as np
import warp as wp
import newton
from newton.solvers import SolverImplicitMPM

wp.init()


def make_solver(sigma=1.0, E=500.0, voxel_size=0.008):
    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)

    n = 10
    spacing = 0.005
    half = (n - 1) * spacing / 2.0
    cell_vol = spacing**3
    mass = cell_vol * 1000.0
    radius = spacing * 0.5

    builder.add_particle_grid(
        pos=wp.vec3(-half, -half, -half),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=n, dim_y=n, dim_z=n,
        cell_x=spacing, cell_y=spacing, cell_z=spacing,
        mass=mass, jitter=0.0, radius_mean=radius,
    )

    model = builder.finalize(device="cuda:0")
    model.set_gravity([0.0, 0.0, 0.0])
    model.mpm.young_modulus.fill_(E)
    model.mpm.poisson_ratio.fill_(0.3)
    model.mpm.tensile_yield_ratio.fill_(1.0)
    model.mpm.friction.fill_(0.0)
    model.mpm.surface_tension.fill_(sigma)

    config = SolverImplicitMPM.Config(
        voxel_size=voxel_size,
        grid_type="sparse",
        max_iterations=100,
        tolerance=1e-4,
    )
    return SolverImplicitMPM(model, config), model


def test_stability(dt, n_steps=100, sigma=1.0, E=500.0):
    solver, model = make_solver(sigma=sigma, E=E)
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    cont = model.contacts()

    for i in range(n_steps):
        solver.step(s0, s1, ctrl, cont, dt)
        s0, s1 = s1, s0
        vel = s0.particle_qd.numpy()
        max_v = np.linalg.norm(vel, axis=1).max()
        if max_v > 100:
            print(f"  dt={dt:.5f}: UNSTABLE at step {i+1} (max_vel={max_v:.1f})")
            return False

    pos = s0.particle_q.numpy()
    rms = np.sqrt(np.mean(np.sum((pos - pos.mean(axis=0))**2, axis=1)))
    print(f"  dt={dt:.5f}: stable, max_vel={max_v:.4e}, rms_r={rms:.6f}")
    return True


print("=== Stability vs dt (sigma=1.0, E=500) ===")
for dt in [0.0002, 0.0005, 0.001, 0.002, 0.004]:
    test_stability(dt)

print("\n=== With visual example params: fps=120, varying substeps ===")
for substeps in [4, 8, 16, 32]:
    dt = 1.0 / (120.0 * substeps)
    print(f"  substeps={substeps}:")
    test_stability(dt)

print("\n=== Lower sigma (0.1), fps=120, substeps=4 ===")
dt = 1.0 / (120.0 * 4)
test_stability(dt, sigma=0.1)
