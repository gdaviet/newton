# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Twist
#
# This simulation demonstrates twisting an FEM cloth model using the VBD
# solver, showcasing its ability to handle complex self-contacts while
# ensuring it remains intersection-free.
#
# Command: python -m newton.examples cloth_twist
#
###########################################################################

import math
import os

import numpy as np
import warp as wp
import warp.examples
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton import ParticleFlags


@wp.kernel
def initialize_rotation(
    # input
    vertex_indices_to_rot: wp.array[wp.int32],
    pos: wp.array[wp.vec3],
    rot_centers: wp.array[wp.vec3],
    rot_axes: wp.array[wp.vec3],
    t: wp.array[float],
    # output
    roots: wp.array[wp.vec3],
    roots_to_ps: wp.array[wp.vec3],
):
    tid = wp.tid()
    v_index = vertex_indices_to_rot[wp.tid()]

    p = pos[v_index]
    rot_center = rot_centers[tid]
    rot_axis = rot_axes[tid]
    op = p - rot_center

    root = wp.dot(op, rot_axis) * rot_axis

    root_to_p = p - root

    roots[tid] = root
    roots_to_ps[tid] = root_to_p

    if tid == 0:
        t[0] = 0.0


@wp.func
def compute_rotation_target(
    rot_axis: wp.vec3,
    root: wp.vec3,
    root_to_p: wp.vec3,
    theta: float,
) -> wp.vec3:
    """Compute a point's target on its prescribed rotation orbit."""
    ux = rot_axis[0]
    uy = rot_axis[1]
    uz = rot_axis[2]

    rotation = wp.mat33(
        wp.cos(theta) + ux * ux * (1.0 - wp.cos(theta)),
        ux * uy * (1.0 - wp.cos(theta)) - uz * wp.sin(theta),
        ux * uz * (1.0 - wp.cos(theta)) + uy * wp.sin(theta),
        uy * ux * (1.0 - wp.cos(theta)) + uz * wp.sin(theta),
        wp.cos(theta) + uy * uy * (1.0 - wp.cos(theta)),
        uy * uz * (1.0 - wp.cos(theta)) - ux * wp.sin(theta),
        uz * ux * (1.0 - wp.cos(theta)) - uy * wp.sin(theta),
        uz * uy * (1.0 - wp.cos(theta)) + ux * wp.sin(theta),
        wp.cos(theta) + uz * uz * (1.0 - wp.cos(theta)),
    )
    return root + rotation * root_to_p


@wp.kernel
def apply_rotation(
    vertex_indices_to_rot: wp.array[wp.int32],
    rot_axes: wp.array[wp.vec3],
    roots: wp.array[wp.vec3],
    roots_to_ps: wp.array[wp.vec3],
    t: wp.array[float],
    angular_velocity: float,
    dt: float,
    end_time: float,
    pos_0: wp.array[wp.vec3],
    pos_1: wp.array[wp.vec3],
):
    """Move prescribed VBD points directly to their rotation targets."""
    cur_t = t[0]
    if cur_t >= end_time:
        return

    tid = wp.tid()
    v_index = vertex_indices_to_rot[tid]
    target_t = wp.min(cur_t + dt, end_time)
    target = compute_rotation_target(rot_axes[tid], roots[tid], roots_to_ps[tid], target_t * angular_velocity)
    pos_0[v_index] = target
    pos_1[v_index] = target


@wp.kernel
def prescribe_rotation_velocity(
    vertex_indices_to_rot: wp.array[wp.int32],
    rot_axes: wp.array[wp.vec3],
    roots: wp.array[wp.vec3],
    roots_to_ps: wp.array[wp.vec3],
    t: wp.array[float],
    angular_velocity: float,
    dt: float,
    end_time: float,
    pos: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
):
    """Set velocities that integrate prescribed LOX points to their targets."""
    tid = wp.tid()
    v_index = vertex_indices_to_rot[tid]
    cur_t = t[0]
    if cur_t >= end_time:
        velocity[v_index] = wp.vec3(0.0)
        return

    target_t = wp.min(cur_t + dt, end_time)
    target = compute_rotation_target(rot_axes[tid], roots[tid], roots_to_ps[tid], target_t * angular_velocity)
    velocity[v_index] = (target - pos[v_index]) / dt


@wp.kernel
def advance_rotation_time(t: wp.array[float], dt: float, end_time: float):
    """Advance the prescribed rotation clock while the motion is active."""
    if t[0] < end_time:
        cur_t = t[0]
        t[0] = wp.min(cur_t + dt, end_time)


class Example:
    def __init__(self, viewer, args):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        # group related attributes by prefix
        self.sim_time = 0.0
        self.sim_substeps = 10  # must be an even number when using CUDA Graph
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = 4
        # the BVH used by SolverVBD will be rebuilt every self.bvh_rebuild_frames
        # When the simulated object deforms significantly, simply refitting the BVH can lead to deterioration of the BVH's
        # quality, in this case we need to completely rebuild the tree to achieve better query efficiency.
        self.bvh_rebuild_frames = 10

        self.rot_angular_velocity = math.pi / 3
        self.rot_end_time = 10

        # save a reference to the viewer
        self.viewer = viewer
        self.solver_type = str(getattr(args, "solver", "vbd")).lower()

        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/cloth/cloth")

        cloth_mesh = newton.usd.get_mesh(usd_prim)
        mesh_points = cloth_mesh.vertices
        mesh_indices = cloth_mesh.indices

        vertices = [wp.vec3(v) for v in mesh_points]
        self.faces = mesh_indices.reshape(-1, 3)

        scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        if self.solver_type == "lox":
            newton.solvers.SolverKamino.register_custom_attributes(scene)
        scene.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2),
            scale=0.01,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.2,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=2.0e-4,
            edge_ke=1e-3,
            edge_kd=1e-2,
        )
        scene.color()
        self.model = scene.finalize()
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e-1
        self.model.soft_contact_mu = 0.2

        cloth_size = 50
        left_side = [cloth_size - 1 + i * cloth_size for i in range(cloth_size)]
        right_side = [i * cloth_size for i in range(cloth_size)]
        rot_point_indices = left_side + right_side

        if len(rot_point_indices):
            flags = self.model.particle_flags.numpy()
            if self.solver_type == "lox":
                masses = self.model.particle_mass.numpy()
                inverse_masses = self.model.particle_inv_mass.numpy()
                for driven_vertex_id in rot_point_indices:
                    flags[driven_vertex_id] = flags[driven_vertex_id] | ParticleFlags.ACTIVE
                    masses[driven_vertex_id] = 0.0
                    inverse_masses[driven_vertex_id] = 0.0
                self.model.particle_mass = wp.array(masses, dtype=wp.float32, device=self.model.device)
                self.model.particle_inv_mass = wp.array(inverse_masses, dtype=wp.float32, device=self.model.device)
            else:
                for fixed_vertex_id in rot_point_indices:
                    flags[fixed_vertex_id] = flags[fixed_vertex_id] & ~ParticleFlags.ACTIVE

            self.model.particle_flags = wp.array(flags, dtype=wp.int32, device=self.model.device)

        if self.solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=self.iterations,
                particle_enable_self_contact=True,
                particle_self_contact_radius=0.002,
                particle_self_contact_margin=0.0035,
            )
        else:
            config = newton.solvers.SolverKamino.Config.from_model(self.model, dynamics_solver="lox")
            config.use_collision_detector = False
            config.lox.projection_iterations = 10
            config.lox.projection_method = "gauss_seidel"
            config.lox.deformable_enable_self_contact = True
            config.lox.deformable_self_contact_margin = 0.002
            config.lox.deformable_self_contact_gap = 0.0015
            config.lox.deformable_self_contact_vertex_buffer_size = 64
            config.lox.deformable_self_contact_edge_buffer_size = 64
            config.lox.deformable_enable_penetration_free_contact = True
            self.solver = newton.solvers.SolverKamino(self.model, config=config)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        rot_axes = [[0, 1, 0]] * len(right_side) + [[0, -1, 0]] * len(left_side)

        self.rot_point_indices = wp.array(rot_point_indices, dtype=int)
        self.t = wp.zeros((1,), dtype=float)
        self.rot_centers = wp.zeros(len(rot_point_indices), dtype=wp.vec3)
        self.rot_axes = wp.array(rot_axes, dtype=wp.vec3)

        self.roots = wp.zeros_like(self.rot_centers)
        self.roots_to_ps = wp.zeros_like(self.rot_centers)

        wp.launch(
            kernel=initialize_rotation,
            dim=self.rot_point_indices.shape[0],
            inputs=[
                self.rot_point_indices,
                self.state_0.particle_q,
                self.rot_centers,
                self.rot_axes,
                self.t,
            ],
            outputs=[
                self.roots,
                self.roots_to_ps,
            ],
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(2.25, 0.0, 0.0), 0.0, -180.0)

        # put graph capture into it's own function
        self.capture()

    def capture(self):
        self.graph = None
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        self.collision_pipeline.collide(self.state_0, self.contacts)
        if self.solver_type == "vbd":
            self.solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            rotation_inputs = [
                self.rot_point_indices,
                self.rot_axes,
                self.roots,
                self.roots_to_ps,
                self.t,
                self.rot_angular_velocity,
                self.sim_dt,
                self.rot_end_time,
            ]
            if self.solver_type == "lox":
                wp.launch(
                    kernel=prescribe_rotation_velocity,
                    dim=self.rot_point_indices.shape[0],
                    inputs=[*rotation_inputs, self.state_0.particle_q],
                    outputs=[self.state_0.particle_qd],
                )
            else:
                wp.launch(
                    kernel=apply_rotation,
                    dim=self.rot_point_indices.shape[0],
                    inputs=rotation_inputs,
                    outputs=[self.state_0.particle_q, self.state_1.particle_q],
                )
            wp.launch(kernel=advance_rotation_time, dim=1, inputs=[self.t, self.sim_dt, self.rot_end_time])

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        if self.viewer is None:
            return

        # Begin frame with time
        self.viewer.begin_frame(self.sim_time)

        # Render model-driven content (ground plane)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        p_lower = wp.vec3(-0.6, -0.9, -0.6)
        p_upper = wp.vec3(0.6, 0.9, 0.6)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 1.5,
        )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    parser.add_argument("--solver", choices=("vbd", "lox"), default="vbd")
    parser.set_defaults(num_frames=300)

    viewer, args = newton.examples.init(parser)

    # Create example and run
    newton.examples.run(Example(viewer, args), args)
