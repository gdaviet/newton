# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example MPM Humanoid 2-Way Coupling
#
# Shows Unitree G1 locomotion with a pretrained policy coupled with implicit MPM sand.
#
# Example usage (via unified runner):
#   uv run newton/examples/sim2sim/mpm_humanoid_twoway_coupling.py --config=newton/examples/assets/g1_29dof_rev_1_0/g1_29dof.yaml
###########################################################################

import sys
from collections.abc import Sequence
import yaml

import numpy as np
import torch
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.examples.robot.example_robot_anymal_c_walk import compute_obs, lab_to_mujoco, mujoco_to_lab
from newton.solvers import SolverImplicitMPM

"""
math utils
"""

@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion.

    Args:
        q: The quaternion in (x, y, z, w). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 3]  # w component is at index 3 for XYZW format
    q_vec = q[..., :3]  # xyz components are at indices 0, 1, 2
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    # for two-dimensional tensors, bmm is faster than einsum
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c


def find_physx_mjwarp_mapping(mjwarp_joint_names, physx_joint_names):
    """
    Finds the mapping between PhysX and MJWarp joint names.
    Returns a tuple of two lists: (mjc_to_physx, physx_to_mjc).
    """
    mjc_to_physx = []
    physx_to_mjc = []
    for j in mjwarp_joint_names:
        if j in physx_joint_names:
            mjc_to_physx.append(physx_joint_names.index(j))

    for j in physx_joint_names:
        if j in mjwarp_joint_names:
            physx_to_mjc.append(mjwarp_joint_names.index(j))

    return mjc_to_physx, physx_to_mjc


"""
circular buffer from IsaacLab
"""


class CircularBuffer:
    """Circular buffer for storing a history of batched tensor data.

    This class implements a circular buffer for storing a history of batched tensor data. The buffer is
    initialized with a maximum length and a batch size. The data is stored in a circular fashion, and the
    data can be retrieved in a LIFO (Last-In-First-Out) fashion. The buffer is designed to be used in
    multi-environment settings, where each environment has its own data.

    The shape of the appended data is expected to be (batch_size, ...), where the first dimension is the
    batch dimension. Correspondingly, the shape of the ring buffer is (max_len, batch_size, ...).
    """

    def __init__(self, max_len: int, batch_size: int, device: str):
        """Initialize the circular buffer.

        Args:
            max_len: The maximum length of the circular buffer. The minimum allowed value is 1.
            batch_size: The batch dimension of the data.
            device: The device used for processing.

        Raises:
            ValueError: If the buffer size is less than one.
        """
        if max_len < 1:
            raise ValueError(f"The buffer size should be greater than zero. However, it is set to {max_len}!")
        # set the parameters
        self._batch_size = batch_size
        self._device = device
        self._ALL_INDICES = torch.arange(batch_size, device=device)

        # max length tensor for comparisons
        self._max_len = torch.full((batch_size,), max_len, dtype=torch.int, device=device)
        # number of data pushes passed since the last call to :meth:`reset`
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
        # the pointer to the current head of the circular buffer (-1 means not initialized)
        self._pointer: int = -1
        # the actual buffer for data storage
        # note: this is initialized on the first call to :meth:`append`
        self._buffer: torch.Tensor = None  # type: ignore

    """
    Properties.
    """

    @property
    def batch_size(self) -> int:
        """The batch size of the ring buffer."""
        return self._batch_size

    @property
    def device(self) -> str:
        """The device used for processing."""
        return self._device

    @property
    def max_length(self) -> int:
        """The maximum length of the ring buffer."""
        return int(self._max_len[0].item())

    @property
    def current_length(self) -> torch.Tensor:
        """The current length of the buffer. Shape is (batch_size,).

        Since the buffer is circular, the current length is the minimum of the number of pushes
        and the maximum length.
        """
        return torch.minimum(self._num_pushes, self._max_len)

    @property
    def buffer(self) -> torch.Tensor:
        """Complete circular buffer with most recent entry at the end and oldest entry at the beginning.
        Returns:
            Complete circular buffer with most recent entry at the end and oldest entry at the beginning of dimension 1. The shape is [batch_size, max_length, data.shape[1:]].
        """
        buf = self._buffer.clone()
        buf = torch.roll(buf, shifts=self.max_length - self._pointer - 1, dims=0)
        return torch.transpose(buf, dim0=0, dim1=1)

    """
    Operations.
    """

    def reset(self, batch_ids: Sequence[int] | None = None):
        """Reset the circular buffer at the specified batch indices.

        Args:
            batch_ids: Elements to reset in the batch dimension. Default is None, which resets all the batch indices.
        """
        # resolve all indices
        if batch_ids is None:
            batch_ids = slice(None)
        # reset the number of pushes for the specified batch indices
        self._num_pushes[batch_ids] = 0
        if self._buffer is not None:
            # set buffer at batch_id reset indices to 0.0 so that the buffer() getter returns the cleared circular buffer after reset.
            self._buffer[:, batch_ids, :] = 0.0

    def append(self, data: torch.Tensor):
        """Append the data to the circular buffer.

        Args:
            data: The data to append to the circular buffer. The first dimension should be the batch dimension.
                Shape is (batch_size, ...).

        Raises:
            ValueError: If the input data has a different batch size than the buffer.
        """
        # check the batch size
        if data.shape[0] != self.batch_size:
            raise ValueError(f"The input data has '{data.shape[0]}' batch size while expecting '{self.batch_size}'")

        # move the data to the device
        data = data.to(self._device)
        # at the first call, initialize the buffer size
        if self._buffer is None:
            self._pointer = -1
            self._buffer = torch.empty((self.max_length, *data.shape), dtype=data.dtype, device=self._device)
        # move the head to the next slot
        self._pointer = (self._pointer + 1) % self.max_length
        # add the new data to the last layer
        self._buffer[self._pointer] = data
        # Check for batches with zero pushes and initialize all values in batch to first append
        is_first_push = self._num_pushes == 0
        if torch.any(is_first_push):
            self._buffer[:, is_first_push] = data[is_first_push]
        # increment number of number of pushes for all batches
        self._num_pushes += 1

    def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
        """Retrieve the data from the circular buffer in last-in-first-out (LIFO) fashion.

        If the requested index is larger than the number of pushes since the last call to :meth:`reset`,
        the oldest stored data is returned.

        Args:
            key: The index to retrieve from the circular buffer. The index should be less than the number of pushes
                since the last call to :meth:`reset`. Shape is (batch_size,).

        Returns:
            The data from the circular buffer. Shape is (batch_size, ...).

        Raises:
            ValueError: If the input key has a different batch size than the buffer.
            RuntimeError: If the buffer is empty.
        """
        # check the batch size
        if len(key) != self.batch_size:
            raise ValueError(f"The argument 'key' has length {key.shape[0]}, while expecting {self.batch_size}")
        # check if the buffer is empty
        if torch.any(self._num_pushes == 0) or self._buffer is None:
            raise RuntimeError("Attempting to retrieve data on an empty circular buffer. Please append data first.")

        # admissible lag
        valid_keys = torch.minimum(key, self._num_pushes - 1)
        # the index in the circular buffer (pointer points to the last+1 index)
        index_in_buffer = torch.remainder(self._pointer - valid_keys, self.max_length)
        # return output
        return self._buffer[index_in_buffer, self._ALL_INDICES]


"""
kernels
"""

@wp.kernel
def compute_body_forces(
    dt: float,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    collider_impulse_pos: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Compute forces applied by sand to rigid bodies.

    Sum the impulses applied on each mpm grid node and convert to
    forces and torques at the body's center of mass.
    """

    i = wp.tid()

    cid = collider_ids[i]
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        if body_index == -1:
            return

        f_world = collider_impulses[i] / dt

        X_wb = body_q[body_index]
        X_com = body_com[body_index]
        r = collider_impulse_pos[i] - wp.transform_point(X_wb, X_com)
        body_wrench = wp.spatial_vector(f_world, wp.cross(r, f_world))
        # mask = wp.sqrt(wp.dot(f_world, f_world)) < 1.0e3
        # body_wrench = body_wrench * float(mask)
        wp.atomic_add(body_f, body_index, body_wrench)


@wp.kernel
def subtract_body_force(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_mass: wp.array(dtype=float),
    body_q_res: wp.array(dtype=wp.transform),
    body_qd_res: wp.array(dtype=wp.spatial_vector),
):
    """Update the rigid bodies velocity to remove the forces applied by sand at the last step.

    This is necessary to compute the total impulses that are required to enforce the complementarity-based
    frictional contact boundary conditions.
    """

    body_id = wp.tid()

    # Remove previously applied force
    f = body_f[body_id]
    inv_mass = wp.where(body_mass[body_id] > 0.0, 1.0 / body_mass[body_id], 0.0)
    delta_v = dt * inv_mass * wp.spatial_top(f)
    r = wp.transform_get_rotation(body_q[body_id])

    delta_w = dt * wp.quat_rotate(r, body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))

    body_q_res[body_id] = body_q[body_id]
    body_qd_res[body_id] = body_qd[body_id] - wp.spatial_vector(delta_v, delta_w)

"""
Env class
"""

class NewtonEnv:
    def __init__(
        self,
        viewer,
        config,
        mjc_to_physx: list[int],
        physx_to_mjc: list[int],
    ):
        self.config = config
        # setup simulation parameters first
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        # sim time step track
        self.sim_time = 0.0
        self.sim_step = 0

        self.viewer = viewer
        self.history_length = self.config.get("history_length", 10)

        """
        setup rigid body builder
        """
        builder = newton.ModelBuilder()
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75
        # add robot
        self.add_robot(builder)
        # add ground plane
        builder.add_ground_plane()

        """
        setup sand builder
        """
        sand_builder = newton.ModelBuilder()
        # add sand
        self.add_sand(sand_builder)

        """
        finalize models
        """
        self.model = builder.finalize()
        self.sand_model = sand_builder.finalize()

        # basic particle material params
        self.sand_model.particle_mu = 0.48
        self.sand_model.particle_ke = 1.0e15

        # device setup
        self.device = self.model.device
        self.torch_device = "cuda" if self.device.is_cuda else "cpu"

        """
        setup MPM solver
        """
        tolerance=1.0e-6
        grid_type = 'fixed'  # 'fixed' or 'sparse'
        voxel_size = 0.05

        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = voxel_size
        mpm_options.tolerance = tolerance
        mpm_options.transfer_scheme = "pic"
        # mpm_options.collider_basis = "pic27"
        # mpm_options.collider_velocity_mode = "finite_difference" # not working? 
        mpm_options.grid_type = grid_type
        mpm_options.grid_padding = 50
        mpm_options.max_active_cell_count = 2**15

        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.critical_fraction = 0.0
        # mpm_options.hardening = 0.0
        # mpm_options.air_drag = 1.0

        mpm_model = SolverImplicitMPM.Model(self.sand_model, mpm_options)
        # read colliders from the RB model rather than the sand model

        coupling_relaxation = 10.0
        mpm_model.setup_collider(
            model=self.model, 
            # body_mass=wp.zeros_like(self.model.body_mass) # kinematic setup
            body_mass=self.model.body_mass * coupling_relaxation,
            body_inv_inertia=self.model.body_inv_inertia / coupling_relaxation,
        )
        self.mpm_solver = SolverImplicitMPM(mpm_model, mpm_options)

        """
        setup rigid body solver
        """
        self.rb_solver = newton.solvers.SolverMuJoCo(self.model, ls_parallel=True, njmax=500)
        # self.rb_solver = newton.solvers.SolverXPBD(self.model)

        """
        prepare simulation states
        """

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.sand_state_0 = self.sand_model.state()
        self.mpm_solver.enrich_state(self.sand_state_0)

        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        """
        setup viewer
        """
        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")
        self.viewer.show_particles = True
        self.show_impulses = False

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Additional buffers for tracking two-way coupling forces
        max_nodes = 2**15
        self.collider_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_ids = wp.full(max_nodes, value=-1, dtype=int, device=self.model.device)
        self.collect_collider_impulses()

        # map from collider index to body index
        self.collider_body_id = mpm_model.collider.collider_body_index

        # per-body forces and torques applied by sand to rigid bodies
        self.body_sand_forces = wp.zeros_like(self.state_0.body_f)

        self.particle_render_colors = wp.full(
            self.sand_model.particle_count, value=wp.vec3(0.27, 0.24, 0.21), dtype=wp.vec3, device=self.sand_model.device
        )
        self.capture()
        
        """
        policy setups
        """
        self.physx_to_mjc_indices = torch.tensor(physx_to_mjc, device=self.torch_device, dtype=torch.long)
        self.mjc_to_physx_indices = torch.tensor(mjc_to_physx, device=self.torch_device, dtype=torch.long)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self.command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self._reset_key_prev = False
        # create observation buffer
        self.create_obs_buffer()
        self._auto_forward = False
        # Load policy and setup tensors
        self.load_policy_and_setup_tensors()


    """
    initializations
    """

    def add_robot(self, builder: newton.ModelBuilder):
        asset_path = self.config["asset_path"]
        if asset_path.endswith(".usd"):
            builder.add_usd(
                source=self.config["asset_path"], 
                xform=wp.transform(wp.vec3(*self.config["mjw_init_pos"])),
                collapse_fixed_joints=False,
                enable_self_collisions=False,
                joint_ordering="dfs",
                hide_collision_shapes=True,
            )
        elif asset_path.endswith(".urdf"):
            builder.add_urdf(
                source=self.config["asset_path"],
                xform=wp.transform(wp.vec3(*self.config["mjw_init_pos"])),
                floating=True,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                ignore_inertial_definitions=False,
            )
        elif asset_path.endswith(".xml"):
            builder.add_mjcf(
                source=self.config["asset_path"],
                xform=wp.transform(wp.vec3(*self.config["mjw_init_pos"])),
                floating=True,
                enable_self_collisions=False,
                # parse_visuals_as_colliders=True,
                ignore_inertial_definitions=False,
            )
        # builder.approximate_meshes("bounding_box")
        
        # -- set initial pose
        builder.joint_q[:3] = self.config["mjw_init_pos"]
        builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
        builder.joint_q[7:] = self.config["mjw_joint_pos"]
        # -- set joint gains
        for i in range(len(self.config["mjw_joint_stiffness"])):
            builder.joint_target_ke[i + 6] = self.config["mjw_joint_stiffness"][i]
            builder.joint_target_kd[i + 6] = self.config["mjw_joint_damping"][i]
            builder.joint_armature[i + 6] = self.config["mjw_joint_armature"][i]

    
    def add_sand(self, sand_builder: newton.ModelBuilder):
        particles_per_cell = 3.0
        voxel_size = 0.05
        density = 2500.0 # bulk density kg/m3
        particle_lo = np.array([-1.0, -1.0, 0.0])  # emission lower bound
        particle_hi = np.array([1.0, 1.0, 0.5])  # emission upper bound
        # particle_lo = np.array([-0.3, -0.3, 0.0])  # emission lower bound
        # particle_hi = np.array([0.3, 0.3, 0.5])  # emission upper bound
        # particle_lo = np.array([-0.5, -0.5, 0.0])  # emission lower bound
        # particle_hi = np.array([0.5, 0.5, 0.35])  # emission upper bound
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        cell_size = (particle_hi - particle_lo) / particle_res
        cell_volume = np.prod(cell_size)
        radius = float(np.max(cell_size) * 0.5)
        mass = float(np.prod(cell_volume) * density)

        sand_builder.add_particle_grid(
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

    def create_obs_buffer(self):
        self.base_ang_vel = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self.projected_gravity = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self.velocity_command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self.joint_pos = torch.zeros((1, self.config["num_dofs"]), device=self.torch_device, dtype=torch.float32)
        self.joint_vel = torch.zeros((1, self.config["num_dofs"]), device=self.torch_device, dtype=torch.float32)
        self.last_actions = torch.zeros((1, self.config["num_dofs"]), device=self.torch_device, dtype=torch.float32)
        self.group_obs_term_hisotry_buffer = {
            "base_ang_vel": CircularBuffer(self.history_length, 1, self.torch_device),
            "projected_gravity": CircularBuffer(self.history_length, 1, self.torch_device),
            "velocity_command": CircularBuffer(self.history_length, 1, self.torch_device),
            "joint_pos": CircularBuffer(self.history_length, 1, self.torch_device),
            "joint_vel": CircularBuffer(self.history_length, 1, self.torch_device),
            "last_actions": CircularBuffer(self.history_length, 1, self.torch_device),
        }
        obs_dim = self.history_length * (
            self.base_ang_vel.shape[1] + self.projected_gravity.shape[1] + self.velocity_command.shape[1] + \
                self.joint_pos.shape[1] + self.joint_vel.shape[1] + self.last_actions.shape[1])
        self.obs = torch.zeros(1, obs_dim, device=self.torch_device, dtype=torch.float32)

    """
    graph
    """
    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None


    """
    physics step
    """

    def step(self):
        # Build command from viewer keyboard
        if hasattr(self.viewer, "is_key_down"):
            fwd = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)

            if fwd or lat or rot:
                # disable forward motion
                self._auto_forward = False

            self.command[0, 0] = float(fwd)
            self.command[0, 1] = float(lat)
            self.command[0, 2] = float(rot)

        if self._auto_forward:
            self.command[0, 0] = 1

        """
        apply control from RL policy
        """
        self.apply_control()

        """
        step rigid body physics
        """
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        body_force = self.body_sand_forces.to("cpu").numpy()
        if np.linalg.norm(body_force, axis=1).max() > 1.0:
            print("Body force exceeds threshold:", body_force)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

        self.viewer.log_points(
            "/sand",
            points=self.sand_state_0.particle_q,
            radii=self.sand_model.particle_radius,
            colors=self.particle_render_colors,
            hidden=not self.viewer.show_particles,
        )

        if self.show_impulses:
            impulses, pos, _cid = self.mpm_solver.collect_collider_impulses(self.sand_state_0)
            self.viewer.log_lines(
                "/impulses",
                starts=pos,
                ends=pos + impulses,
                colors=wp.full(pos.shape[0], value=wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3),
            )
        else:
            self.viewer.log_lines("/impulses", None, None, None)

        self.viewer.end_frame()

    """
    physics
    """

    def simulate(self):
        # simulate robot
        self.simulate_robot()
        # # simulate sand
        # self.simulate_sand()

    def simulate_robot(self):
        # robot substeps
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            wp.launch(
                compute_body_forces,
                dim=self.collider_impulse_ids.shape[0],
                inputs=[
                    self.sim_dt,
                    self.collider_impulse_ids,
                    self.collider_impulses,
                    self.collider_impulse_pos,
                    self.collider_body_id,
                    self.state_0.body_q,
                    self.model.body_com,
                    self.state_0.body_f,
                ],
            )
            # saved applied force to subtract later on
            self.body_sand_forces.assign(self.state_0.body_f)

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0)
            self.rb_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

            self.simulate_sand()

    def simulate_sand(self):
        # Subtract previously applied impulses from body velocities

        if self.sand_state_0.body_q is not None:
            wp.launch(
                subtract_body_force,
                dim=self.sand_state_0.body_q.shape,
                inputs=[
                    self.sim_dt,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.body_sand_forces,
                    self.mpm_solver.mpm_model.collider_body_inv_inertia,
                    self.mpm_solver.mpm_model.collider_body_mass,
                    self.sand_state_0.body_q,
                    self.sand_state_0.body_qd,
                ],
            )

        self.mpm_solver.step(self.sand_state_0, self.sand_state_0, contacts=None, control=None, dt=self.sim_dt)

        # Save impulses to apply back to rigid bodies
        self.collect_collider_impulses()

    def collect_collider_impulses(self):
        collider_impulses, collider_impulse_pos, collider_impulse_ids = self.mpm_solver.collect_collider_impulses(
            self.sand_state_0
        )
        self.collider_impulse_ids.fill_(-1)
        n_colliders = min(collider_impulses.shape[0], self.collider_impulses.shape[0])
        self.collider_impulses[:n_colliders].assign(collider_impulses[:n_colliders])
        self.collider_impulse_pos[:n_colliders].assign(collider_impulse_pos[:n_colliders])
        self.collider_impulse_ids[:n_colliders].assign(collider_impulse_ids[:n_colliders])

    """
    mdp
    """
    
    def compute_obs(self):
        # Extract state information with proper handling
        joint_q = self.state_0.joint_q if self.state_0.joint_q is not None else []
        joint_qd = self.state_0.joint_qd if self.state_0.joint_qd is not None else []

        root_quat_w = torch.tensor(joint_q[3:7], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        # root_lin_vel_w = torch.tensor(joint_qd[:3], device=self.torch_device, dtype=torch.float32).unsqueeze(0) # maybe used 
        root_ang_vel_w = torch.tensor(joint_qd[3:6], device=self.torch_device, dtype=torch.float32).unsqueeze(0)

        joint_pos_current = torch.tensor(joint_q[7:], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        joint_vel_current = torch.tensor(joint_qd[6:], device=self.torch_device, dtype=torch.float32).unsqueeze(0)

        self.base_ang_vel = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
        self.projected_gravity = quat_rotate_inverse(root_quat_w, self.gravity_vec)
        self.joint_pos = torch.index_select(joint_pos_current - self.joint_pos_initial, 1, self.physx_to_mjc_indices)
        self.joint_vel = torch.index_select(joint_vel_current, 1, self.physx_to_mjc_indices)
        # self.joint_pos = joint_pos_current - self.joint_pos_initial
        # self.joint_vel = joint_vel_current
        self.velocity_command = self.command
        self.last_actions = self.act
        
        # add to history buffer
        self.group_obs_term_hisotry_buffer["base_ang_vel"].append(self.base_ang_vel)
        self.group_obs_term_hisotry_buffer["projected_gravity"].append(self.projected_gravity)
        self.group_obs_term_hisotry_buffer["velocity_command"].append(self.velocity_command)
        self.group_obs_term_hisotry_buffer["joint_pos"].append(self.joint_pos)
        self.group_obs_term_hisotry_buffer["joint_vel"].append(self.joint_vel)
        self.group_obs_term_hisotry_buffer["last_actions"].append(self.last_actions)
        
        self.obs = torch.cat([
            self.group_obs_term_hisotry_buffer["base_ang_vel"].buffer.reshape(1, -1), # (1, history_length * 3)
            self.group_obs_term_hisotry_buffer["projected_gravity"].buffer.reshape(1, -1), # (1, history_length * 3)
            self.group_obs_term_hisotry_buffer["velocity_command"].buffer.reshape(1, -1), # (1, history_length * 3)
            self.group_obs_term_hisotry_buffer["joint_pos"].buffer.reshape(1, -1), # (1, history_length * num_dofs)
            self.group_obs_term_hisotry_buffer["joint_vel"].buffer.reshape(1, -1), # (1, history_length * num_dofs)
            self.group_obs_term_hisotry_buffer["last_actions"].buffer.reshape(1, -1), # (1, history_length * num_dofs)
        ], dim=1)

    def apply_control(self):
        self.compute_obs()
        with torch.no_grad():
            self.act = self.policy(self.obs)
            self.rearranged_act = torch.index_select(self.act, 1, self.mjc_to_physx_indices) 
            a = self.joint_pos_initial + self.config["action_scale"] * self.rearranged_act
            # add extra base dof zeros
            a_with_zeros = torch.cat([torch.zeros(6, device=self.torch_device, dtype=torch.float32), a.squeeze(0)])
            a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
            wp.copy(self.control.joint_target_pos, a_wp)


    """
    utilities
    """
    def render_ui(self, imgui):
        _changed, self.show_impulses = imgui.checkbox("Show Impulses", self.show_impulses)
        
    def load_policy_and_setup_tensors(self):
        """Load policy and setup initial tensors for robot control.
        """
        print("[INFO] Loading policy from:", self.config["policy_path"])
        self.policy = torch.jit.load(self.config["policy_path"], map_location=self.torch_device)

        # Handle potential None state
        joint_q = self.state_0.joint_q if self.state_0.joint_q is not None else []
        self.joint_pos_initial = torch.tensor(joint_q[7:], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self.act = torch.zeros(1, self.config["num_dofs"], device=self.torch_device, dtype=torch.float32)
        self.rearranged_act = torch.zeros(1, self.config["num_dofs"], device=self.torch_device, dtype=torch.float32)


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds
    # example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument("--config", type=str, 
                        default="newton/examples/assets/g1_29dof_rev_1_0/g1_29dof.yaml", 
                        help="Path to robot config YAML file.")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Load robot configuration from YAML file in the downloaded assets
    # TODO: resolve hardcoding later
    yaml_file_path = args.config
    try:
        with open(yaml_file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[ERROR] Robot config file not found: {yaml_file_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing YAML file: {e}")
        exit(1)

    print(f"[INFO] Loaded config with {config['num_dofs']} DOFs")

    mjc_to_physx = list(range(config["num_dofs"]))
    physx_to_mjc = list(range(config["num_dofs"]))

    # if args.physx:
    if "physx_joint_names" in config.keys():
        # when importing policy trained in IsaacLab
        mjc_to_physx, physx_to_mjc = find_physx_mjwarp_mapping(config["mjw_joint_names"], config["physx_joint_names"])

    env = NewtonEnv(viewer, config, mjc_to_physx, physx_to_mjc)

    total_sim_time = 100.0  # seconds
    while env.sim_time < total_sim_time:
        if not env.viewer.is_paused():
            with wp.ScopedTimer("step", active=False):
                env.step()

        with wp.ScopedTimer("render", active=False):
            env.render()
    print("[INFO] Simulation completed")
    env.viewer.close()