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

from ...sim import Model, ModelFlags, StateFlags
from ...sim.articulation import eval_fk
from .admm_inertial import ReducedInertia
from .admm_proxy_model import build_proxy_model
from .solver_coupled import _entry_control, _nested_attribute_value


@wp.kernel(enable_backward=False)
def _update_inertia(
    bodies: wp.array[int],
    proxies: wp.array[int],
    joints: wp.array[int],
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
    proxy = proxies[row]
    m = wp.max(mass[body], 1.0e-8)
    r = moment[body] / m
    I = inertia[body] - m * (wp.dot(r, r) * wp.identity(n=3, dtype=float) - wp.outer(r, r))
    # A point mass has singular rotational inertia. A small positive tensor
    # makes native rigid integration well defined; it too is iterate-centered.
    I += (1.0e-6 * wp.max(m, wp.trace(inertia[body]))) * wp.identity(n=3, dtype=float)
    if point_proxies[row] != 0:
        I = wp.identity(n=3, dtype=float)
        anchor = wp.transform(physical_com[body] + r, wp.quat_identity())
        joint_X_p[joints[row]] = anchor
        joint_X_c[joints[row]] = anchor
    proxy_mass[proxy] = m
    proxy_inv_mass[proxy] = 1.0 / m
    proxy_com[proxy] = physical_com[body] + r
    proxy_inertia[proxy] = I
    proxy_inv_inertia[proxy] = wp.inverse(I)


@wp.kernel(enable_backward=False)
def _prepare_reduced_coordinates(
    bodies: wp.array[int],
    augmented_bodies: wp.array[int],
    proxies: wp.array[int],
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
    proxy = proxies[row]
    augmented_body = augmented_bodies[body]
    q = body_q[augmented_body]
    R = wp.quat_to_matrix(wp.transform_get_rotation(q))
    r = R * (com[proxy] - com[augmented_body])
    I = R * inertia[proxy] * wp.transpose(R)
    vn = body_qd[augmented_body]
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


class ProxyBodies:
    """Private augmented entry with one numerical body per participating body."""

    def __init__(self, entry, factory, bodies, *, reduced_coordinates=False, point_bodies=()):
        physical = entry.view
        device = physical.device
        self.bodies = wp.array(sorted(bodies), dtype=int, device=device)
        self.point_proxies = wp.array([int(body in point_bodies) for body in sorted(bodies)], dtype=int, device=device)
        self.reduced_coordinates = reduced_coordinates
        view, maps, numerical_bodies, numerical_joints, self._model_copies = build_proxy_model(
            physical, sorted(bodies), point_bodies
        )
        self._maps = {freq: wp.array(mapping, dtype=int, device=device) for freq, mapping in maps.items()}
        self.proxies = wp.array([numerical_bodies[body] for body in sorted(bodies)], dtype=int, device=device)
        self.joints = wp.array([numerical_joints[body] for body in sorted(bodies)], dtype=int, device=device)
        frequency = Model.AttributeFrequency

        def compose(local_to_global, freq):
            source = local_to_global.numpy()
            result = np.full(Model._attribute_frequency_count(view, freq), -1, dtype=np.int32)
            result[maps[freq]] = source
            return wp.array(result, dtype=int, device=device)

        def inverse(global_to_local, freq):
            result = global_to_local.numpy().copy()
            valid = result >= 0
            result[valid] = maps[freq][result[valid]]
            return wp.array(result, dtype=int, device=device)

        attributes = dict(entry.attribute_local_to_global)
        for freq in maps:
            if freq in attributes:
                attributes[freq] = compose(attributes[freq], freq)
        self.entry = replace(
            entry,
            view=view,
            solver=factory(view),
            body_local_to_global=compose(entry.body_local_to_global, frequency.BODY),
            body_global_to_local=inverse(entry.body_global_to_local, frequency.BODY),
            body_dynamics_disabled_local_indices=wp.array(
                maps[frequency.BODY][entry.body_dynamics_disabled_local_indices.numpy()], dtype=int, device=device
            ),
            proxy_body_local_indices=wp.array(
                maps[frequency.BODY][entry.proxy_body_local_indices.numpy()], dtype=int, device=device
            ),
            joint_dynamics_disabled_local_indices=wp.array(
                maps[frequency.JOINT][entry.joint_dynamics_disabled_local_indices.numpy()], dtype=int, device=device
            ),
            joint_coord_local_to_global=compose(entry.joint_coord_local_to_global, frequency.JOINT_COORD),
            joint_coord_global_to_local=inverse(entry.joint_coord_global_to_local, frequency.JOINT_COORD),
            joint_dof_local_to_global=compose(entry.joint_dof_local_to_global, frequency.JOINT_DOF),
            joint_dof_global_to_local=inverse(entry.joint_dof_global_to_local, frequency.JOINT_DOF),
            attribute_local_to_global=attributes,
            state_0=view.state(),
            state_1=view.state(),
            state_tmp=view.state() if entry.substeps > 1 and not entry.in_place else None,
            control=_entry_control(view),
        )
        self._state_maps = {}
        for name, spec in view._iter_attribute_specs():
            if spec.frequency in maps:
                rows = (maps[spec.frequency][:, None] * spec.row_width + np.arange(spec.row_width)).ravel()
                self._state_maps[name] = wp.array(rows, dtype=int, device=device)
        self.parents = wp.array(maps[frequency.BODY][sorted(bodies)], dtype=int, device=device)
        self.target = wp.empty(len(bodies), dtype=wp.transform, device=device)
        self.target_velocity = wp.empty(len(bodies), dtype=wp.spatial_vector, device=device)
        self.gravity = wp.empty(len(bodies), dtype=wp.vec3, device=device)
        wp.copy(self.target, physical.body_q[self.bodies])
        wp.copy(self.gravity, entry.body_gravity_acceleration[self.bodies])
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
                self.proxies,
                self.joints,
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
        wp.copy(self.target, buf.body_q_n[self.bodies])
        wp.copy(self.gravity, entry.body_gravity_acceleration[self.bodies])
        if self.reduced_inertia is not None:
            numerical = self.reduced_inertia.view
            wp.copy(numerical.body_mass[self.proxies], view.body_mass[self.proxies])
            wp.copy(numerical.body_inertia[self.proxies], view.body_inertia[self.proxies])
            self._copy_state(entry.state_0, self.entry.state_0)
            eval_fk(view, self.entry.state_0.joint_q, self.entry.state_0.joint_qd, self.entry.state_0)
            self.reduced_inertia.refresh(self.entry.state_0)

    def sync_model(self, entry):
        """Copy changed physical rows while preserving private topology."""
        for name, mapping in self._model_copies:
            physical = _nested_attribute_value(entry.view, name)
            wp.copy(_nested_attribute_value(self.entry.view, name)[mapping], physical)

    def _copy_state(self, source, destination, *, gather=False):
        names = dict.fromkeys(vars(source))
        names.update(dict.fromkeys(name for name, _ in self.entry.view._iter_attribute_specs() if ":" in name))
        for name in names:
            value = _nested_attribute_value(source, name)
            other = _nested_attribute_value(destination, name)
            if not isinstance(value, wp.array) or not isinstance(other, wp.array):
                continue
            mapping = self._state_maps.get(name)
            if mapping is None:
                wp.copy(other, value)
            elif gather:
                wp.copy(other, value[mapping])
            else:
                wp.copy(other[mapping], value)

    def reset(self, entry, world_mask):
        """Reset native caches without changing the caller's physical state."""
        self._copy_state(entry.state_0, self.entry.state_0)
        self.entry.solver.reset(self.entry.state_0, world_mask=world_mask, flags=0)

    def prepare(self, entry, buf, *, dt):
        """Teleport the numerical bodies without changing physical input state."""
        local = self.entry
        self._copy_state(entry.state_0, local.state_0)
        if self.reduced_coordinates:
            wp.copy(
                self.reduced_inertia.target.joint_qd[self._maps[Model.AttributeFrequency.JOINT_DOF]], buf.joint_qd_k
            )
            velocity_k = self.reduced_inertia.target_velocity(
                local.state_0.joint_q, self.reduced_inertia.target.joint_qd
            )
            wp.launch(
                _prepare_reduced_coordinates,
                dim=self.bodies.shape[0],
                inputs=[
                    self.bodies,
                    self._maps[Model.AttributeFrequency.BODY],
                    self.proxies,
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
            wp.copy(self.target_velocity, buf.body_qd_k[self.bodies])
            local.solver.coupling_set_inertial_target(
                local.state_0,
                self.proxies,
                self.parents,
                self.target,
                self.target_velocity,
                self.gravity,
                dt=dt,
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
        self._copy_state(output, entry.state_1, gather=True)
        wp.copy(self.target, entry.state_1.body_q[self.bodies])
