# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LOX colored Gauss--Seidel projection."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import mat36f, mat66f, vec6f
from newton._src.solvers.kamino._src.solvers.lox.colored_gauss_seidel import (
    ColoredGaussSeidelProjection,
    _ColorFamily,
)
from newton._src.solvers.kamino._src.solvers.lox.deformable_contact import (
    DEFORMABLE_CONTACT_STATUS_CROSS_WORLD,
    DEFORMABLE_CONTACT_STATUS_MALFORMED,
    DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE,
    DEFORMABLE_CONTACT_STATUS_VALID,
)
from newton._src.solvers.kamino._src.solvers.lox.sweep import (
    _DeformableRigidContactProjectionData,
    _DeformableRigidProjectionState,
    _make_colored_projection_index,
    _make_project_contacts_kernel,
    _make_projection_struct,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


def _empty(device, dtype):
    return wp.empty(0, dtype=dtype, device=device)


def _make_adapter(device, endpoints):
    count = len(endpoints)
    empty_int = _empty(device, wp.int32)
    empty_vec6 = _empty(device, vec6f)
    contact_jacobian = np.zeros((count, 3, 6), dtype=np.float32)
    contact_jacobian[:, 0, 1] = 1.0
    contact_jacobian[:, 1, 2] = 1.0
    contact_jacobian[:, 2, 0] = 1.0
    contact_jacobian_first = wp.array(contact_jacobian, dtype=mat36f, device=device)
    contact_jacobian_second = wp.zeros(count, dtype=mat36f, device=device)
    body_count = max((max(pair) for pair in endpoints), default=-1) + 1
    return SimpleNamespace(
        device=device,
        body_constraint_count=wp.zeros(body_count, dtype=wp.int32, device=device),
        static_body_constraint_count=wp.zeros(body_count, dtype=wp.int32, device=device),
        friction_capacity=0,
        friction_world=empty_int,
        friction_local=empty_int,
        world_friction_count=wp.zeros(1, dtype=wp.int32, device=device),
        friction_body_first=empty_int,
        friction_body_second=empty_int,
        friction_jacobian_first=empty_vec6,
        friction_jacobian_second=empty_vec6,
        friction_impulse_bound=_empty(device, wp.float32),
        friction_projection_delassus=_empty(device, wp.float32),
        friction_reaction=_empty(device, wp.float32),
        contact_capacity=count,
        contact_world=wp.zeros(count, dtype=wp.int32, device=device),
        contact_local=wp.array(np.arange(count, dtype=np.int32), dtype=wp.int32, device=device),
        world_contact_count=wp.array([count], dtype=wp.int32, device=device),
        contact_body_first=wp.array([pair[0] for pair in endpoints], dtype=wp.int32, device=device),
        contact_body_second=wp.array([pair[1] for pair in endpoints], dtype=wp.int32, device=device),
        contact_jacobian_first=contact_jacobian_first,
        contact_jacobian_second=contact_jacobian_second,
        contact_bias=wp.zeros(count, dtype=wp.vec3f, device=device),
        contact_friction=wp.zeros(count, dtype=wp.float32, device=device),
        contact_projection_delassus=wp.zeros(count, dtype=wp.mat33f, device=device),
        contact_reaction=wp.zeros(count, dtype=wp.vec3f, device=device),
        limit_capacity=0,
        limit_world=empty_int,
        limit_local=empty_int,
        world_limit_count=wp.zeros(1, dtype=wp.int32, device=device),
        limit_body_first=empty_int,
        limit_body_second=empty_int,
        limit_jacobian_first=empty_vec6,
        limit_jacobian_second=empty_vec6,
        limit_bias=_empty(device, wp.float32),
        limit_projection_delassus=_empty(device, wp.float32),
        limit_reaction=_empty(device, wp.float32),
    )


def _make_all_rigid_family_adapter(device):
    endpoints = [(0, -1), (0, 1), (0, -1), (2, -1), (3, 4), (5, -1), (6, 0), (7, -1)]
    adapter = _make_adapter(device, endpoints)
    count = len(endpoints)
    scalar_jacobian_first = np.zeros((count, 6), dtype=np.float32)
    scalar_jacobian_first[:, 0] = 1.0
    scalar_jacobian_second = np.zeros((count, 6), dtype=np.float32)
    scalar_jacobian_second[:, 0] = -0.25
    contact_jacobian_second = -0.25 * adapter.contact_jacobian_first.numpy()

    adapter.contact_jacobian_second.assign(contact_jacobian_second)
    adapter.contact_bias.assign([[-0.15, 0.1, -1.0 - 0.05 * constraint] for constraint in range(count)])
    adapter.contact_friction.fill_(0.4)
    for family, biases in (("friction", None), ("limit", -0.6 - 0.03 * np.arange(count))):
        setattr(adapter, f"{family}_capacity", count)
        setattr(adapter, f"{family}_world", wp.zeros(count, dtype=wp.int32, device=device))
        setattr(
            adapter,
            f"{family}_local",
            wp.array(np.arange(count, dtype=np.int32), dtype=wp.int32, device=device),
        )
        setattr(adapter, f"world_{family}_count", wp.array([count], dtype=wp.int32, device=device))
        setattr(
            adapter,
            f"{family}_body_first",
            wp.array([pair[0] for pair in endpoints], dtype=wp.int32, device=device),
        )
        setattr(
            adapter,
            f"{family}_body_second",
            wp.array([pair[1] for pair in endpoints], dtype=wp.int32, device=device),
        )
        setattr(
            adapter,
            f"{family}_jacobian_first",
            wp.array(scalar_jacobian_first, dtype=vec6f, device=device),
        )
        setattr(
            adapter,
            f"{family}_jacobian_second",
            wp.array(scalar_jacobian_second, dtype=vec6f, device=device),
        )
        setattr(
            adapter,
            f"{family}_projection_delassus",
            wp.zeros(count, dtype=wp.float32, device=device),
        )
        setattr(adapter, f"{family}_reaction", wp.zeros(count, dtype=wp.float32, device=device))
        if biases is not None:
            setattr(adapter, f"{family}_bias", wp.array(biases, dtype=wp.float32, device=device))
    adapter.friction_impulse_bound = wp.full(count, 10.0, dtype=wp.float32, device=device)
    return adapter


def _multiplicity_objective(occupancy):
    return int(np.sum(np.asarray(occupancy, dtype=np.int64) ** 2))


def _make_deformable(device, *, body, status, world_status, global_status):
    particle_indices = wp.array([[0, -1, -1, -1]], dtype=wp.int32, device=device)
    deformable = SimpleNamespace(
        device=device,
        contact_capacity=1,
        cloth_system=SimpleNamespace(
            particle_count=1,
            inverse_weight=wp.ones(1, dtype=wp.float32, device=device),
        ),
        particle_indices=particle_indices,
        coefficients=wp.array([[1.0, 0.0, 0.0, 0.0]], dtype=wp.float32, device=device),
        contact_world=wp.zeros(1, dtype=wp.int32, device=device),
        body=wp.array([body], dtype=wp.int32, device=device),
        body_jacobian=wp.zeros(1, dtype=mat36f, device=device),
        frame=wp.array([np.eye(3, dtype=np.float32)], dtype=wp.mat33f, device=device),
        bias=wp.zeros(1, dtype=wp.vec3f, device=device),
        friction=wp.zeros(1, dtype=wp.float32, device=device),
        status=wp.full(1, status, dtype=wp.int32, device=device),
        gauss_seidel_scalar_delassus=wp.zeros(1, dtype=wp.float32, device=device),
        gauss_seidel_delassus=wp.zeros(1, dtype=wp.mat33f, device=device),
        rigid_delassus=wp.zeros(1, dtype=wp.mat33f, device=device),
        reaction=wp.zeros(1, dtype=wp.vec3f, device=device),
        particle_delta=wp.zeros(1, dtype=wp.vec3f, device=device),
        world_status=wp.array([world_status], dtype=wp.int32, device=device),
        global_status=wp.array([global_status], dtype=wp.int32, device=device),
        invalid_count=wp.full(
            1,
            int(status != DEFORMABLE_CONTACT_STATUS_VALID),
            dtype=wp.int32,
            device=device,
        ),
        _empty_body_inverse_weight=wp.empty(0, dtype=mat66f, device=device),
    )

    def prepare_rigid_projection(*_args):
        return None

    deformable.prepare_rigid_projection = prepare_rigid_projection
    return deformable


class TestLOXColoredGaussSeidel(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_colors_propagate_updates_sequentially(self):
        """Expose each completed color's velocity update to the following color."""
        adapter = _make_adapter(self.device, [(0, -1), (0, -1)])
        adapter.contact_bias.assign([[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]])
        projection = ColoredGaussSeidelProjection(adapter, None, 2)
        inverse_weight = wp.array([np.eye(6, dtype=np.float32)], dtype=mat66f, device=self.device)
        prepared_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        projection.prepare(inverse_weight, prepared_status)
        self.assertEqual(len(np.unique(projection.contact.colors.numpy())), 2)

        projected_twist = wp.zeros(1, dtype=vec6f, device=self.device)
        projection_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        projection.project(
            1,
            wp.ones(1, dtype=wp.bool, device=self.device),
            wp.zeros(1, dtype=wp.int32, device=self.device),
            inverse_weight,
            projected_twist,
            wp.zeros(1, dtype=vec6f, device=self.device),
            None,
            prepared_status,
            projection_status,
        )
        np.testing.assert_allclose(projected_twist.numpy()[0, 0], 2.0, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(adapter.contact_reaction.numpy()[:, 2], [0.0, 2.0], rtol=0.0, atol=2.0e-6)

    def test_projection_clears_scratch_across_consecutive_calls(self):
        """Clear stale body deltas after valid and rejected projection calls."""
        adapter = _make_adapter(self.device, [(0, -1)])
        adapter.contact_bias.assign([[0.0, 0.0, -1.0]])
        projection = ColoredGaussSeidelProjection(adapter, None, 2)
        inverse_weight = wp.array([np.eye(6, dtype=np.float32)], dtype=mat66f, device=self.device)
        prepared_status = wp.ones(1, dtype=wp.int32, device=self.device)
        projection.prepare(inverse_weight, prepared_status)
        twist_delta = wp.full(1, vec6f(3.0), dtype=vec6f, device=self.device)
        projected_twist = wp.zeros(1, dtype=vec6f, device=self.device)
        projection_status = wp.ones(1, dtype=wp.int32, device=self.device)
        arguments = (
            wp.ones(1, dtype=wp.bool, device=self.device),
            wp.zeros(1, dtype=wp.int32, device=self.device),
            inverse_weight,
            projected_twist,
            twist_delta,
            None,
            prepared_status,
            projection_status,
        )

        projection.project(1, *arguments)
        np.testing.assert_array_equal(twist_delta.numpy(), np.zeros((1, 6), dtype=np.float32))

        twist_delta.fill_(vec6f(5.0))
        prepared_status.zero_()
        projection.project(1, *arguments)
        np.testing.assert_array_equal(projection_status.numpy(), [0])
        np.testing.assert_array_equal(twist_delta.numpy(), np.zeros((1, 6), dtype=np.float32))

    def test_invalid_world_discards_only_its_color_delta(self):
        """Keep a malformed world's partial update out of every color barrier."""
        adapter = _make_adapter(self.device, [(0, -1), (1, -1)])
        adapter.contact_world.assign([0, 1])
        adapter.contact_local.assign([0, 0])
        adapter.world_friction_count = wp.zeros(2, dtype=wp.int32, device=self.device)
        adapter.world_contact_count = wp.ones(2, dtype=wp.int32, device=self.device)
        adapter.world_limit_count = wp.zeros(2, dtype=wp.int32, device=self.device)
        jacobians = adapter.contact_jacobian_first.numpy()
        jacobians[1] = 0.0
        adapter.contact_jacobian_first.assign(jacobians)
        adapter.contact_bias.assign([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
        projection = ColoredGaussSeidelProjection(adapter, None, 2)
        inverse_weight = wp.array([np.eye(6, dtype=np.float32)] * 2, dtype=mat66f, device=self.device)
        prepared_status = wp.zeros(2, dtype=wp.int32, device=self.device)
        projection.prepare(inverse_weight, prepared_status)
        np.testing.assert_array_equal(prepared_status.numpy(), [1, 0])

        projected_twist = wp.zeros(2, dtype=vec6f, device=self.device)
        projection_status = wp.zeros(2, dtype=wp.int32, device=self.device)
        twist_delta = wp.zeros(2, dtype=vec6f, device=self.device)
        projection.project(
            1,
            wp.ones(2, dtype=wp.bool, device=self.device),
            wp.array([0, 1], dtype=wp.int32, device=self.device),
            inverse_weight,
            projected_twist,
            twist_delta,
            None,
            prepared_status,
            projection_status,
        )
        np.testing.assert_allclose(projected_twist.numpy()[:, 0], [1.0, 0.0], rtol=0.0, atol=2.0e-6)
        np.testing.assert_array_equal(projection_status.numpy(), [1, 0])
        np.testing.assert_array_equal(twist_delta.numpy(), np.zeros((2, 6), dtype=np.float32))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required for block synchronization")
    def test_world_colored_matches_global_colored(self):
        """Match the single-launch world traversal to the global colored path."""
        device = wp.get_device("cuda:0")
        world_count = device.sm_count * 4
        world_ids = np.arange(world_count, dtype=np.int32)
        zeros = np.zeros(world_count, dtype=np.int32)
        ones = np.ones(world_count, dtype=np.int32)
        empty_int = _empty(device, wp.int32)
        empty_vec6 = _empty(device, vec6f)
        world_offset = wp.array(world_ids, dtype=wp.int32, device=device)
        zero_world_offset = wp.zeros(world_count, dtype=wp.int32, device=device)
        zero_world_count = wp.zeros(world_count, dtype=wp.int32, device=device)
        contact_jacobian = np.zeros((world_count, 3, 6), dtype=np.float32)
        contact_jacobian[:, 2, 0] = 1.0
        adapter = SimpleNamespace(
            device=device,
            model=SimpleNamespace(
                info=SimpleNamespace(
                    bodies_offset=world_offset,
                    num_bodies=wp.array(ones, dtype=wp.int32, device=device),
                )
            ),
            body_constraint_count=wp.ones(world_count, dtype=wp.int32, device=device),
            static_body_constraint_count=wp.zeros(world_count, dtype=wp.int32, device=device),
            friction_capacity=0,
            friction_world=empty_int,
            friction_local=empty_int,
            world_friction_offset=zero_world_offset,
            world_friction_count=zero_world_count,
            friction_body_first=empty_int,
            friction_body_second=empty_int,
            friction_jacobian_first=empty_vec6,
            friction_jacobian_second=empty_vec6,
            friction_impulse_bound=_empty(device, wp.float32),
            friction_projection_delassus=_empty(device, wp.float32),
            friction_reaction=_empty(device, wp.float32),
            contact_capacity=world_count,
            contact_world=wp.array(world_ids, dtype=wp.int32, device=device),
            contact_local=wp.array(zeros, dtype=wp.int32, device=device),
            world_contact_offset=world_offset,
            world_contact_count=wp.array(ones, dtype=wp.int32, device=device),
            contact_body_first=wp.array(world_ids, dtype=wp.int32, device=device),
            contact_body_second=wp.full(world_count, -1, dtype=wp.int32, device=device),
            contact_jacobian_first=wp.array(contact_jacobian, dtype=mat36f, device=device),
            contact_jacobian_second=wp.zeros(world_count, dtype=mat36f, device=device),
            contact_bias=wp.full(world_count, wp.vec3f(0.0, 0.0, -1.0), dtype=wp.vec3f, device=device),
            contact_friction=wp.full(world_count, 0.4, dtype=wp.float32, device=device),
            contact_projection_delassus=wp.zeros(world_count, dtype=wp.mat33f, device=device),
            contact_reaction=wp.full(
                world_count,
                wp.vec3f(0.0, 0.0, 0.2),
                dtype=wp.vec3f,
                device=device,
            ),
            limit_capacity=0,
            limit_world=empty_int,
            limit_local=empty_int,
            world_limit_offset=zero_world_offset,
            world_limit_count=zero_world_count,
            limit_body_first=empty_int,
            limit_body_second=empty_int,
            limit_jacobian_first=empty_vec6,
            limit_jacobian_second=empty_vec6,
            limit_bias=_empty(device, wp.float32),
            limit_projection_delassus=_empty(device, wp.float32),
            limit_reaction=_empty(device, wp.float32),
        )
        projection = ColoredGaussSeidelProjection(adapter, None, 2)
        inverse_weight = wp.array(
            np.repeat(np.eye(6, dtype=np.float32)[None, :, :], world_count, axis=0),
            dtype=mat66f,
            device=device,
        )
        prepared_status = wp.zeros(world_count, dtype=wp.int32, device=device)
        projection.prepare(inverse_weight, prepared_status)

        def run(use_world_projection):
            adapter.model.info.bodies_offset = world_offset if use_world_projection else None
            projected_twist = wp.zeros(world_count, dtype=vec6f, device=device)
            twist_delta = wp.zeros(world_count, dtype=vec6f, device=device)
            projection_status = wp.zeros(world_count, dtype=wp.int32, device=device)
            adapter.contact_reaction.fill_(wp.vec3f(0.0, 0.0, 0.2))
            projection.project(
                3,
                wp.ones(world_count, dtype=wp.bool, device=device),
                wp.array(world_ids, dtype=wp.int32, device=device),
                inverse_weight,
                projected_twist,
                twist_delta,
                None,
                prepared_status,
                projection_status,
            )
            return projected_twist.numpy(), adapter.contact_reaction.numpy(), projection_status.numpy()

        global_result = run(False)
        world_result = run(True)
        for global_value, world_value in zip(global_result[:-1], world_result[:-1], strict=True):
            np.testing.assert_allclose(world_value, global_value, rtol=0.0, atol=2.0e-6)
        np.testing.assert_array_equal(world_result[-1], global_result[-1])

    def test_pure_deformable_preserves_global_preparation_failure(self):
        """Reject a pure deformable world when contact adaptation failed globally."""
        deformable = _make_deformable(
            self.device,
            body=-1,
            status=DEFORMABLE_CONTACT_STATUS_MALFORMED,
            world_status=0,
            global_status=DEFORMABLE_CONTACT_STATUS_MALFORMED,
        )
        projection = ColoredGaussSeidelProjection(None, deformable, 2)
        prepared_status = wp.ones(1, dtype=wp.int32, device=self.device)

        projection.prepare(None, prepared_status)

        np.testing.assert_array_equal(prepared_status.numpy(), [0])

    def test_mixed_deformable_preserves_world_preparation_failure(self):
        """Reject only the mixed-contact world whose adapted record crossed worlds."""
        adapter = _make_adapter(self.device, [(0, -1)])
        adapter.world_contact_count.zero_()
        deformable = _make_deformable(
            self.device,
            body=0,
            status=DEFORMABLE_CONTACT_STATUS_MALFORMED,
            world_status=DEFORMABLE_CONTACT_STATUS_CROSS_WORLD,
            global_status=0,
        )
        projection = ColoredGaussSeidelProjection(adapter, deformable, 2)
        prepared_status = wp.ones(1, dtype=wp.int32, device=self.device)

        projection.prepare(
            wp.array([np.eye(6, dtype=np.float32)], dtype=mat66f, device=self.device),
            prepared_status,
        )

        np.testing.assert_array_equal(prepared_status.numpy(), [0])

    def test_nonfinite_mixed_body_correction_marks_contact_failure(self):
        """Record deformable diagnostics when a mixed body correction becomes nonfinite."""
        adapter = _make_adapter(self.device, [(0, -1)])
        adapter.world_contact_count.zero_()
        deformable = _make_deformable(
            self.device,
            body=0,
            status=DEFORMABLE_CONTACT_STATUS_VALID,
            world_status=DEFORMABLE_CONTACT_STATUS_VALID,
            global_status=0,
        )
        projection = ColoredGaussSeidelProjection(adapter, deformable, 2)
        valid_inverse_weight = wp.array([np.eye(6, dtype=np.float32)], dtype=mat66f, device=self.device)
        prepared_status = wp.ones(1, dtype=wp.int32, device=self.device)
        projection.prepare(valid_inverse_weight, prepared_status)
        invalid_inverse_weight = wp.array(
            [np.full((6, 6), np.nan, dtype=np.float32)],
            dtype=mat66f,
            device=self.device,
        )
        projection_status = wp.ones(1, dtype=wp.int32, device=self.device)
        body_delta = wp.zeros(1, dtype=vec6f, device=self.device)
        color = int(projection.deformable.colors.numpy()[0])
        index = _make_colored_projection_index(
            deformable.contact_world,
            projection.deformable.counts,
            projection.deformable.offsets,
            projection.deformable.order,
        )
        data = _make_projection_struct(
            _DeformableRigidContactProjectionData,
            particle_indices=deformable.particle_indices,
            coefficients=deformable.coefficients,
            body=deformable.body,
            frame=deformable.frame,
            body_jacobian=deformable.body_jacobian,
            delassus=deformable.gauss_seidel_delassus,
            bias=deformable.bias,
            friction=deformable.friction,
            reaction=deformable.reaction,
            status=deformable.status,
            contact_world_status=deformable.world_status,
        )
        state = _make_projection_struct(
            _DeformableRigidProjectionState,
            world_active=wp.ones(1, dtype=wp.bool, device=self.device),
            projected_twist=wp.zeros(1, dtype=vec6f, device=self.device),
            twist_delta=body_delta,
            world_status=projection_status,
            occupancy=projection.body_occupancy,
            inverse_weight=invalid_inverse_weight,
            particle_occupancy=projection.particle_occupancy,
            particle_inverse_weight=deformable.cloth_system.inverse_weight,
            projected_velocity=wp.zeros(1, dtype=wp.vec3f, device=self.device),
            particle_delta=deformable.particle_delta,
        )
        wp.launch(
            _make_project_contacts_kernel(
                True,
                True,
                DEFORMABLE_CONTACT_STATUS_VALID,
                DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE,
            ),
            dim=1,
            inputs=[1, color, index, data, state],
            device=self.device,
        )

        np.testing.assert_array_equal(deformable.status.numpy(), [DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE])
        np.testing.assert_array_equal(
            deformable.world_status.numpy(),
            [DEFORMABLE_CONTACT_STATUS_NUMERICAL_FAILURE],
        )
        np.testing.assert_array_equal(projection_status.numpy(), [0])

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required for graph capture")
    def test_cuda_graph_capture_replays_large_color_compaction(self):
        """Capture the parallel color-prefix scan and cursor copy path."""
        device = wp.get_device("cuda:0")
        color_count = 96
        capacity = 2 * color_count + 11
        colors = np.arange(capacity, dtype=np.int32) % color_count
        colors[::17] = -1
        family = _ColorFamily(capacity, color_count, device)
        family.colors.assign(colors)
        family.compact(color_count, device)

        with wp.ScopedCapture(device=device) as capture:
            family.compact(color_count, device)
        wp.capture_launch(capture.graph)

        valid = colors >= 0
        expected_counts = np.bincount(colors[valid], minlength=color_count).astype(np.int32)
        expected_offsets = np.zeros(color_count, dtype=np.int32)
        expected_offsets[1:] = np.cumsum(expected_counts[:-1], dtype=np.int32)
        counts = family.counts.numpy()
        offsets = family.offsets.numpy()
        order = family.order.numpy()[: int(np.sum(expected_counts))]
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_array_equal(offsets, expected_offsets)
        np.testing.assert_array_equal(np.sort(order), np.flatnonzero(valid))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required for graph capture")
    def test_cuda_graph_capture_replays_coloring_and_projection(self):
        """Capture and replay device coloring, metric preparation, and projection."""
        device = wp.get_device("cuda:0")
        adapter = _make_adapter(device, [(0, -1), (0, -1)])
        adapter.contact_bias.assign([[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]])
        projection = ColoredGaussSeidelProjection(adapter, None, 2)
        inverse_weight = wp.array([np.eye(6, dtype=np.float32)], dtype=mat66f, device=device)
        prepared_status = wp.zeros(1, dtype=wp.int32, device=device)
        projection_status = wp.zeros(1, dtype=wp.int32, device=device)
        projected_twist = wp.zeros(1, dtype=vec6f, device=device)
        twist_delta = wp.zeros(1, dtype=vec6f, device=device)
        world_active = wp.ones(1, dtype=wp.bool, device=device)
        body_world = wp.zeros(1, dtype=wp.int32, device=device)

        def run():
            projection.prepare(inverse_weight, prepared_status)
            projection.project(
                1,
                world_active,
                body_world,
                inverse_weight,
                projected_twist,
                twist_delta,
                None,
                prepared_status,
                projection_status,
            )

        run()
        projected_twist.zero_()
        adapter.contact_reaction.zero_()
        with wp.ScopedCapture(device=device) as capture:
            run()
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(projected_twist.numpy()[0, 0], 2.0, rtol=0.0, atol=2.0e-6)


if __name__ == "__main__":
    unittest.main()
