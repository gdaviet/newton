# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a private numerical-body model with explicit physical row mappings."""

import numpy as np
import warp as wp

from ...sim import JointType, Model, ModelBuilder
from .model_view import ModelView, _AttributeNamespaceView
from .solver_coupled import _nested_attribute_value


def _set_attribute(view, name, value):
    if ":" not in name:
        setattr(view, name, value)
        return
    namespace, attr = name.split(":", 1)
    if namespace not in view.overrides:
        setattr(view, namespace, _AttributeNamespaceView(getattr(view.parent, namespace)))
    setattr(getattr(view, namespace), attr, value)


def build_proxy_model(physical, bodies, point_bodies):
    """Preserve worlds and articulation blocks when inserting numerical bodies.

    Return the augmented view, physical-to-augmented row maps, numerical body
    and joint ids, and arrays whose physical rows must track model updates.
    """
    frequency = Model.AttributeFrequency
    builder = ModelBuilder()
    body_world = physical.body_world.numpy()
    joint_world = physical.joint_world.numpy()
    parent = physical.joint_parent.numpy()
    child = physical.joint_child.numpy()
    types = physical.joint_type.numpy()
    dims = physical.joint_dof_dim.numpy()
    axes = physical.joint_axis.numpy()
    qd_start = physical.joint_qd_start.numpy()
    joint_art = physical.joint_articulation.numpy()
    art_world = physical.articulation_world.numpy()
    maps = {
        frequency.BODY: np.full(physical.body_count, -1, dtype=np.int32),
        frequency.JOINT: np.full(physical.joint_count, -1, dtype=np.int32),
        frequency.ARTICULATION: np.full(physical.articulation_count, -1, dtype=np.int32),
    }
    numerical_bodies = {}
    numerical_joints = {}

    def add_joint(joint):
        start = int(qd_start[joint])
        linear, angular = map(int, dims[joint])
        configs = [ModelBuilder.JointDofConfig(axis=axis) for axis in axes[start : start + linear + angular]]
        index = builder.add_joint(
            JointType(int(types[joint])),
            int(maps[frequency.BODY][parent[joint]]) if parent[joint] >= 0 else -1,
            int(maps[frequency.BODY][child[joint]]),
            linear_axes=configs[:linear],
            angular_axes=configs[linear:],
            label=physical.joint_label[joint],
        )
        maps[frequency.JOINT][joint] = index
        return index

    def add_proxy(body):
        method = builder.add_joint_ball if body in point_bodies else builder.add_joint_fixed
        kwargs = {"armature": 0.0, "damping": 0.0} if body in point_bodies else {}
        joint = method(int(maps[frequency.BODY][body]), numerical_bodies[body], **kwargs)
        numerical_joints[body] = joint
        return joint

    for world in range(-1, physical.world_count):
        if world >= 0:
            builder.begin_world()
        for body in np.flatnonzero(body_world == world):
            maps[frequency.BODY][body] = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        for body in bodies:
            if body_world[body] == world:
                numerical_bodies[body] = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        for articulation in np.flatnonzero(art_world == world):
            physical_joints = np.flatnonzero(joint_art == articulation)
            joints = [add_joint(int(joint)) for joint in physical_joints]
            for body in child[physical_joints]:
                if body in numerical_bodies and body not in numerical_joints:
                    joints.append(add_proxy(int(body)))
            maps[frequency.ARTICULATION][articulation] = builder.articulation_count
            builder.add_articulation(joints)
        for joint in np.flatnonzero((joint_world == world) & (joint_art < 0)):
            add_joint(int(joint))
        for body in bodies:
            if body_world[body] == world and body not in numerical_joints:
                add_proxy(body)
        if world >= 0:
            builder.end_world()
    template = builder.finalize(device=physical.device)
    for freq, name in ((frequency.JOINT_COORD, "joint_q_start"), (frequency.JOINT_DOF, "joint_qd_start")):
        old = getattr(physical, name).numpy()
        new = getattr(template, name).numpy()
        mapping = np.empty(int(old[-1]), dtype=np.int32)
        for joint, augmented in enumerate(maps[frequency.JOINT]):
            mapping[old[joint] : old[joint + 1]] = np.arange(new[augmented], new[augmented + 1])
        maps[freq] = mapping
    old_widths = [JointType(int(kind)).constraint_count(int(sum(dim))) for kind, dim in zip(types, dims, strict=True)]
    new_widths = [
        JointType(int(kind)).constraint_count(int(sum(dim)))
        for kind, dim in zip(template.joint_type.numpy(), template.joint_dof_dim.numpy(), strict=True)
    ]
    old_starts, new_starts = np.cumsum([0, *old_widths]), np.cumsum([0, *new_widths])
    maps[frequency.JOINT_CONSTRAINT] = np.empty(int(old_starts[-1]), dtype=np.int32)
    for joint, augmented in enumerate(maps[frequency.JOINT]):
        maps[frequency.JOINT_CONSTRAINT][old_starts[joint] : old_starts[joint + 1]] = np.arange(
            new_starts[augmented], new_starts[augmented + 1]
        )

    view = ModelView(physical, physical.name)
    topology = {"joint_parent", "joint_child", "joint_articulation", "joint_ancestor", "articulation_end"}
    copies = []
    # Install every expanded array before exposing the larger counts through
    # ModelView. Custom attributes use their registered defaults for new rows.
    for name, spec in physical._iter_attribute_specs():
        if spec.deprecated:
            continue
        value = _nested_attribute_value(physical, name)
        if value is None:
            continue
        if name in topology or (spec.frequency in maps and spec.compaction_policy in ("start", "world_start")):
            _set_attribute(view, name, getattr(template, name))
            continue
        if spec.compaction_policy != "generic" or (spec.frequency not in maps and spec.references not in maps):
            continue
        if not isinstance(value, (wp.array, list)):
            continue
        source = value.numpy() if isinstance(value, wp.array) else np.asarray(value)
        mapping = maps.get(spec.frequency, np.arange(len(source) // spec.row_width, dtype=np.int32))
        rows = (mapping[:, None] * spec.row_width + np.arange(spec.row_width)).ravel()
        count = template._attribute_frequency_count(spec.frequency) if spec.frequency in maps else len(mapping)
        extra = _nested_attribute_value(template, name)
        if spec.frequency not in maps:
            data = source.copy()
        elif isinstance(extra, wp.array):
            data = extra.numpy().copy()
        elif isinstance(extra, list):
            data = np.asarray(extra, dtype=object).copy()
        else:
            data = np.empty((count * spec.row_width, *source.shape[1:]), dtype=source.dtype)
            data[...] = spec.default if spec.default is not None else 0
        if spec.references in maps:
            source = source.copy()
            valid = source >= 0
            source[valid] = maps[spec.references][source[valid]]
        data[rows] = source
        result = (
            wp.array(data, dtype=value.dtype, device=physical.device) if isinstance(value, wp.array) else data.tolist()
        )
        _set_attribute(view, name, result)
        if isinstance(result, wp.array) and spec.references is None and spec.assignment is None:
            copies.append((name, wp.array(rows, dtype=int, device=physical.device)))
    for freq in maps:
        count_name = Model._ATTRIBUTE_FREQUENCY_COUNT_ATTRS[freq]
        setattr(view, count_name, getattr(template, count_name))
    view.max_joints_per_articulation = template.max_joints_per_articulation
    view.max_dofs_per_articulation = template.max_dofs_per_articulation
    view.body_shapes = {
        int(maps[frequency.BODY][body]) if body >= 0 else body: shapes for body, shapes in physical.body_shapes.items()
    }
    view.body_shapes.update({proxy: [] for proxy in numerical_bodies.values()})
    view.body_color_groups = [
        wp.array(maps[frequency.BODY][group.numpy()], dtype=int, device=physical.device)
        for group in physical.body_color_groups
    ]
    view.body_color_groups.append(wp.array(list(numerical_bodies.values()), dtype=int, device=physical.device))
    for name in ("_fk_articulation_level_start", "_fk_level_joint_start", "_fk_level_joints", "_fk_level_parent_pos"):
        setattr(view, name, None)
    view._fk_level_capacity = 0
    return view, maps, numerical_bodies, numerical_joints, copies
