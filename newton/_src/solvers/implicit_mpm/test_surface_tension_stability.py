# Quick diagnostic with density field

import numpy as np
import warp as wp
import newton
from newton.solvers import SolverImplicitMPM

wp.init()

builder = newton.ModelBuilder()
SolverImplicitMPM.register_custom_attributes(builder)

n, spacing = 10, 0.005
half = (n - 1) * spacing / 2.0
for i in range(n):
    for j in range(n):
        for k in range(n):
            builder.add_particle(
                pos=(-half + i * spacing, -half + j * spacing, -half + k * spacing),
                vel=(0, 0, 0), mass=spacing**3 * 1000.0, radius=spacing * 0.5)

model = builder.finalize(device="cuda:0")
model.set_gravity([0.0, 0.0, 0.0])
model.mpm.tensile_yield_ratio.fill_(1.0)
model.mpm.friction.fill_(0.0)
model.mpm.viscosity.fill_(10.0)
model.mpm.surface_tension.fill_(30.0)

solver = SolverImplicitMPM(model, SolverImplicitMPM.Config(
    voxel_size=0.008, grid_type="sparse", max_iterations=100, tolerance=1e-4))

s0, s1 = model.state(), model.state()
ctrl, cont = model.control(), model.contacts()

solver.step(s0, s1, ctrl, cont, dt=1.0 / 480.0)

vel = s1.particle_qd.numpy()
print(f"max velocity: {np.linalg.norm(vel, axis=1).max():.6e}")

fields = solver.gather_surface_tension_fields()
if fields:
    c = fields["indicator"].numpy()
    k = fields["curvature"].numpy()
    print(f"indicator: min={c.min():.4f} max={c.max():.4f} mean={c.mean():.4f}")
    print(f"  at surface (0.1<c<0.9): {np.sum((c > 0.1) & (c < 0.9))}/{len(c)}")
    print(f"curvature: min={k.min():.4f} max={k.max():.4f} mean abs={np.abs(k).mean():.4f}")


def corner_face_ratio(pos):
    c = pos.mean(axis=0)
    d = np.linalg.norm(pos - c, axis=1)
    med = np.median(d)
    return d[d > med].mean() / d[d <= med].mean()


# Run 60 frames
for frame in range(60):
    for _ in range(4):
        solver.step(s0, s1, ctrl, cont, dt=1.0 / 480.0)
        s0, s1 = s1, s0
    if (frame + 1) % 20 == 0:
        ratio = corner_face_ratio(s0.particle_q.numpy())
        maxv = np.linalg.norm(s0.particle_qd.numpy(), axis=1).max()
        print(f"frame {frame+1}: ratio={ratio:.4f}  max_vel={maxv:.4e}")
