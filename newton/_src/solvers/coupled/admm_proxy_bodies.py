# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Recentered numerical rigid bodies for positional ADMM local solves.

An auxiliary body realizes each owned body's accumulated spatial inertia.
Its inertial target follows the last outer iterate, so the intended proximal
reaction vanishes at a fixed point. Physical body inertia and timestep history
stay unchanged. The augmented model is private; public entry indices stay physical.

Known limitation (correction deferred): canceling the added inertial bias does
not cancel changes to the native integrator's velocity derivatives. With MuJoCo
3.15.0 CPU implicitfast, a free rigid subtree includes the numerical descendants
in its implicit gyroscopic derivative. Let P be numerical inertia, B_p the added
bias velocity derivative, A the physical integration matrix, h the timestep,
v_n the physical input velocity, and v_k the current ADMM target. After the
existing bias compensation, the local equation has the form::

    (A + P + h B_p) (v_next - v_n) = h f_physical + P (v_k - v_n)

At convergence P cancels, but h B_p (v_star - v_n) remains. More ADMM iterations
cannot remove this error; it also occurs with gamma zero. The installed
mujoco-warp 3.15.0 instead excludes free roots with massive fixed descendants
from its standalone-body gyroscopic treatment, changing the physical derivative.
Both behaviors can change the physical fixed point.

In the off-center spinning-body replay case at h=0.02 s and gamma=0, CPU
implicitfast linear/angular velocity norm errors were 0.19%/0.08% at rho=20
and 9.6%/4.0% at rho=1000. Results at 256 and 1024 iterations were identical
for rho=1000. GPU angular error was about 0.47% at both penalties; Euler replay
errors were approximately roundoff. These are single-step measurements, not
bounds for other models or trajectory errors.

Deferred design: a solver-owned coupling hook should evaluate the discrete
inertial correction using physical and numerical inertia, input state, target
velocity, and timestep. For the CPU equation above, adding B_p (v_k - v_n) to
the applied force leaves only B_p (v_next - v_k), which vanishes at convergence.
The hook must also account for changes to physical integration caused by the
augmented topology, as on the GPU. Keep backend logic inside the solver and
retain one native solve per local update; no inner solve is proposed. Validate
physical replay, parameter invariance, and convergence before enabling a fix.
"""

from dataclasses import replace

import numpy as np
import warp as wp

from ...sim import Model, ModelBuilder, ModelFlags, StateFlags
from ...sim.articulation import eval_fk
from .admm_inertial import ReducedInertia
from .model_view import ModelView
from .solver_coupled import _entry_control


@wp.kernel(enable_backward=False)
def _update_inertia(
    bodies: wp.array[int],
    offset: int,
    joint_offset: int,
    point_proxies: wp.array[int],
    physical_com: wp.array[wp.vec3],
    mass: wp.array[float],
    moment: wp.array[wp.vec3],
    inertia: wp.array[wp.mat33],
    proxy_mass: wp.array[float],
    proxy_inv_mass: wp.array[float],
    proxy_com: wp.array[wp.vec3],
    proxy_inertia: wp.array[wp.mat33],
    proxy_inv_inertia: wp.array[wp.mat33],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
):
    row = wp.tid()
    body = bodies[row]
    proxy = offset + row
    m = wp.max(mass[body], 1.0e-8)
    r = moment[body] / m
    I = inertia[body] - m * (wp.dot(r, r) * wp.identity(n=3, dtype=float) - wp.outer(r, r))
    # A point mass has singular rotational inertia. A small positive tensor
    # makes native rigid integration well defined; it too is iterate-centered.
    I += (1.0e-6 * wp.max(m, wp.trace(inertia[body]))) * wp.identity(n=3, dtype=float)
    if point_proxies[row] != 0:
        I = wp.identity(n=3, dtype=float)
        anchor = wp.transform(physical_com[body] + r, wp.quat_identity())
        joint_X_p[joint_offset + row] = anchor
        joint_X_c[joint_offset + row] = anchor
    proxy_mass[proxy] = m
    proxy_inv_mass[proxy] = 1.0 / m
    proxy_com[proxy] = physical_com[body] + r
    proxy_inertia[proxy] = I
    proxy_inv_inertia[proxy] = wp.inverse(I)


@wp.kernel(enable_backward=False)
def _recenter(
    bodies: wp.array[int],
    offset: int,
    target: wp.array[wp.transform],
    gravity: wp.array[wp.vec3],
    mass: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
):
    row = wp.tid()
    body = bodies[row]
    proxy = offset + row
    body_q[proxy] = target[body]
    body_qd[proxy] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
    body_f[proxy] = wp.spatial_vector(-mass[proxy] * gravity[body], wp.vec3(0.0))


@wp.kernel(enable_backward=False)
def _prepare_reduced_coordinates(
    bodies: wp.array[int],
    offset: int,
    velocity_k: wp.array[wp.spatial_vector],
    bias: wp.array[wp.spatial_vector],
    gravity: wp.array[wp.vec3],
    mass: wp.array[float],
    com: wp.array[wp.vec3],
    inertia: wp.array[wp.mat33],
    dt: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
):
    row = wp.tid()
    body = bodies[row]
    proxy = offset + row
    q = body_q[body]
    R = wp.quat_to_matrix(wp.transform_get_rotation(q))
    r = R * (com[proxy] - com[body])
    I = R * inertia[proxy] * wp.transpose(R)
    vn = body_qd[body]
    vk = velocity_k[proxy]
    wn = wp.spatial_bottom(vn)
    wk = wp.spatial_bottom(vk)
    linear_n = wp.spatial_top(vn) + wp.cross(wn, r)
    linear_k = wp.spatial_top(vk)
    body_q[proxy] = q
    body_qd[proxy] = wp.spatial_vector(linear_n, wn)
    body_f[proxy] = (
        wp.spatial_vector(
            mass[proxy] * ((linear_k - linear_n) / dt - gravity[body]),
            I * ((wk - wn) / dt),
        )
        + bias[proxy]
    )


def _copy_state_prefix(source, destination):
    for name, value in vars(source).items():
        other = getattr(destination, name, None)
        if isinstance(value, wp.array) and isinstance(other, wp.array):
            wp.copy(other, value, count=min(value.shape[0], other.shape[0]))


class ProxyBodies:
    """Private augmented entry with one numerical body per participating body."""

    def __init__(self, entry, factory, bodies, *, reduced_coordinates=False, point_bodies=()):
        physical = entry.view
        device = physical.device
        self.offset = physical.body_count
        self.bodies = wp.array(sorted(bodies), dtype=int, device=device)
        self.point_proxies = wp.array([int(body in point_bodies) for body in sorted(bodies)], dtype=int, device=device)
        self.joint_offset = physical.joint_count
        self.reduced_coordinates = reduced_coordinates
        builder = ModelBuilder()
        for _ in range(self.offset):
            builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        proxies = {body: builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for body in sorted(bodies)}
        fixed_joints = []
        for body in range(self.offset):
            joints = [builder.add_joint_free(body)]
            if body in proxies:
                joint = (
                    builder.add_joint_ball(body, proxies[body], armature=0.0, damping=0.0)
                    if body in point_bodies
                    else builder.add_joint_fixed(body, proxies[body])
                )
                joints.append(joint)
                fixed_joints.append(joint)
            builder.add_articulation(joints)
        template = builder.finalize(device=device)
        view = ModelView(physical, physical.name)
        frequency = Model.AttributeFrequency
        counts = {
            frequency.BODY: self.offset + len(bodies),
            frequency.JOINT: physical.joint_count + len(bodies),
            frequency.JOINT_CONSTRAINT: physical.joint_constraint_count + template.joint_constraint_count,
            frequency.JOINT_COORD: physical.joint_coord_count + 4 * len(point_bodies),
            frequency.JOINT_DOF: physical.joint_dof_count + 3 * len(point_bodies),
        }
        selected = {frequency.JOINT: fixed_joints}
        for freq, starts in (
            (frequency.JOINT_COORD, template.joint_q_start),
            (frequency.JOINT_DOF, template.joint_qd_start),
        ):
            offsets = starts.numpy()
            selected[freq] = [i for j in fixed_joints for i in range(offsets[j], offsets[j + 1])]
        overrides = {}
        # Append private rigid rows without renumbering physical entities.
        # Ball proxies also append unused rotational coordinates and controls.
        for name, spec in physical._iter_attribute_specs():
            if spec.frequency not in counts or spec.assignment is not None or ":" in name:
                continue
            value = getattr(physical, name, None)
            extra = getattr(template, name, None)
            if value is None or extra is None:
                continue
            if spec.compaction_policy == "world_start":
                data = value.numpy().copy()
                world = int(physical.body_world.numpy()[0])
                if world == -1:
                    data[-1] = counts[spec.frequency]
                else:
                    data[world + 1 :] = counts[spec.frequency]
                overrides[name] = wp.array(data, dtype=value.dtype, device=device)
            elif spec.compaction_policy == "start":
                widths = np.diff(extra.numpy())[fixed_joints]
                data = np.concatenate((value.numpy(), np.cumsum(widths) + int(value.numpy()[-1])))
                overrides[name] = wp.array(data, dtype=value.dtype, device=device)
            elif spec.compaction_policy == "generic":
                start = self.offset * spec.row_width if spec.frequency == frequency.BODY else 0
                if isinstance(value, wp.array):
                    data = extra.numpy()[start:].copy()
                    if spec.frequency in selected:
                        data = extra.numpy()[selected[spec.frequency]].copy()
                    if name in ("body_world", "joint_world"):
                        data[:] = int(physical.body_world.numpy()[0])
                    if spec.references == frequency.JOINT:
                        data[:] = -1
                    if name == "joint_articulation":
                        data[:] = 0 if reduced_coordinates else -1
                    data = np.concatenate((value.numpy(), data))
                    overrides[name] = wp.array(data, dtype=value.dtype, device=device)
                elif isinstance(value, list):
                    tail = [extra[i] for i in selected[spec.frequency]] if spec.frequency in selected else extra[start:]
                    overrides[name] = value + tail
        for name, value in overrides.items():
            setattr(view, name, value)
        view.body_count = counts[frequency.BODY]
        view.joint_count = counts[frequency.JOINT]
        view.joint_constraint_count = counts[frequency.JOINT_CONSTRAINT]
        view.joint_coord_count = counts[frequency.JOINT_COORD]
        view.joint_dof_count = counts[frequency.JOINT_DOF]
        view.body_shapes = dict(physical.body_shapes)
        for proxy in range(self.offset, view.body_count):
            view.body_shapes[proxy] = []
        view.body_color_groups = [
            *physical.body_color_groups,
            wp.array(range(self.offset, view.body_count), dtype=int, device=device),
        ]
        if reduced_coordinates:
            starts = physical.articulation_start.numpy().copy()
            starts[-1] = view.joint_count
            view.articulation_start = wp.array(starts, dtype=int, device=device)
            ends = physical.articulation_end.numpy().copy()
            ends[0] = view.joint_count
            view.articulation_end = wp.array(ends, dtype=int, device=device)
            view.max_joints_per_articulation = view.joint_count
            view.max_dofs_per_articulation = view.joint_dof_count
            incoming = {int(body): joint for joint, body in enumerate(physical.joint_child.numpy())}
            ancestors = view.joint_ancestor.numpy()
            for row, body in enumerate(sorted(bodies)):
                ancestors[physical.joint_count + row] = incoming[body]
            view.joint_ancestor = wp.array(ancestors, dtype=int, device=device)
            for name in (
                "_fk_articulation_level_start",
                "_fk_level_joint_start",
                "_fk_level_joints",
                "_fk_level_parent_pos",
            ):
                setattr(view, name, None)
            view._fk_level_capacity = 0
        self.entry = replace(
            entry,
            view=view,
            solver=factory(view),
            state_0=view.state(),
            state_1=view.state(),
            state_tmp=view.state() if entry.substeps > 1 and not entry.in_place else None,
            control=_entry_control(view),
        )
        self._prefix_arrays = [
            name
            for name, value in overrides.items()
            if isinstance(value, wp.array) and physical._attribute_spec(name).compaction_policy == "generic"
        ]
        self.target = wp.clone(physical.body_q)
        self.reduced_inertia = None
        if reduced_coordinates:
            mass = wp.zeros(view.body_count, dtype=float, device=device)
            inertia = wp.zeros(view.body_count, dtype=wp.mat33, device=device)
            self.reduced_inertia = ReducedInertia(self.entry, mass, inertia, isotropic=False)

    def refresh(self, entry, buf):
        """Refresh numerical inertia once per physical timestep."""
        view = self.entry.view
        self.sync_model(entry)
        wp.launch(
            _update_inertia,
            dim=self.bodies.shape[0],
            inputs=[
                self.bodies,
                self.offset,
                self.joint_offset,
                self.point_proxies,
                buf.body_com,
                buf.proxy_mass,
                buf.proxy_moment,
                buf.proxy_inertia,
                view.body_mass,
                view.body_inv_mass,
                view.body_com,
                view.body_inertia,
                view.body_inv_inertia,
                view.joint_X_p,
                view.joint_X_c,
            ],
            device=view.device,
        )
        self.entry.solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES | ModelFlags.JOINT_PROPERTIES)
        wp.copy(self.target, buf.body_q_n)
        if self.reduced_inertia is not None:
            numerical = self.reduced_inertia.view
            count = view.body_count - self.offset
            wp.copy(numerical.body_mass, view.body_mass, src_offset=self.offset, dest_offset=self.offset, count=count)
            wp.copy(
                numerical.body_inertia, view.body_inertia, src_offset=self.offset, dest_offset=self.offset, count=count
            )
            _copy_state_prefix(entry.state_0, self.entry.state_0)
            eval_fk(view, self.entry.state_0.joint_q, self.entry.state_0.joint_qd, self.entry.state_0)
            self.reduced_inertia.refresh(self.entry.state_0)

    def sync_model(self, entry):
        """Copy changed physical rows while preserving private topology."""
        for name in self._prefix_arrays:
            physical = getattr(entry.view, name)
            wp.copy(getattr(self.entry.view, name), physical, count=physical.shape[0])

    def reset(self, entry, world_mask):
        """Reset native caches without changing the caller's physical state."""
        _copy_state_prefix(entry.state_0, self.entry.state_0)
        self.entry.solver.reset(self.entry.state_0, world_mask=world_mask, flags=0)

    def prepare(self, entry, buf, *, dt):
        """Teleport the numerical bodies without changing physical input state."""
        local = self.entry
        _copy_state_prefix(entry.state_0, local.state_0)
        if self.reduced_coordinates:
            velocity_k = self.reduced_inertia.target_velocity(local.state_0.joint_q, buf.joint_qd_k)
            wp.launch(
                _prepare_reduced_coordinates,
                dim=self.bodies.shape[0],
                inputs=[
                    self.bodies,
                    self.offset,
                    velocity_k,
                    self.reduced_inertia.bias,
                    entry.body_gravity_acceleration,
                    local.view.body_mass,
                    local.view.body_com,
                    local.view.body_inertia,
                    float(dt),
                    local.state_0.body_q,
                    local.state_0.body_qd,
                    local.state_0.body_f,
                ],
                device=local.view.device,
            )
        else:
            wp.launch(
                _recenter,
                dim=self.bodies.shape[0],
                inputs=[
                    self.bodies,
                    self.offset,
                    self.target,
                    entry.body_gravity_acceleration,
                    local.view.body_mass,
                    local.state_0.body_q,
                    local.state_0.body_qd,
                    local.state_0.body_f,
                ],
                device=local.view.device,
            )
        local.solver.coupling_notify_input_state_update(
            local.state_0,
            StateFlags.BODY | StateFlags.JOINT | StateFlags.PARTICLE,
            # Recentered numerical poses are instantaneous inertial targets,
            # not displacements to convert into a velocity over the timestep.
            dt=0.0,
        )

    def finish(self, entry):
        output = self.entry.state_0 if self.entry.in_place else self.entry.state_1
        _copy_state_prefix(output, entry.state_1)
        wp.copy(self.target, entry.state_1.body_q)
