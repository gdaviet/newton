# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cross-platform quality gates for the Kamino LOX solver."""

import unittest

from newton._src.solvers.kamino.tests.test_lox_coupled_rejection import TestLOXCoupledRejection
from newton._src.solvers.kamino.tests.test_lox_joint_covariance import TestLOXJointCovariance
from newton._src.solvers.kamino.tests.test_lox_rod_feedback import TestLOXRodFeedback
from newton._src.solvers.kamino.tests.test_solver_kamino_lox import TestSolverKaminoLOX

_LOX_QUALITY_TESTS = (
    "test_binary_rod_balances_wrenches_and_respects_enabled",
    "test_box_on_plane_projects_detected_contact",
    "test_rod_accepts_implicit_single_world",
    "test_rod_bend_and_twist_restore_rotation",
    "test_relaxed_joint_proximal_preserves_nonlinear_fixed_point",
    "test_rod_world_parent_stretch_and_damping",
    "test_cartpole_projects_detected_joint_limit",
    "test_cartpole_sustained_joint_force_remains_bounded",
    "test_free_fall_advances_projected_velocity_and_pose",
    "test_joint_damping_is_implicit_in_the_smooth_row",
    "test_joint_proximal_uses_frozen_frame_for_full_position_blocks",
    "test_joint_proximal_handles_two_body_frames_and_orderings",
    "test_joint_proximal_relaxations_share_position_fixed_point",
    "test_joint_proximal_recovery_matches_remaining_frozen_budget",
    "test_joint_proximal_accepts_finite_root_translation_transient",
    "test_joint_proximal_rejects_last_iteration_nonfinite_feedback",
    "test_joint_proximal_recovery_is_per_world",
    "test_joint_proximal_recovery_restores_apgd_contact_state",
    "test_product_space_structural_split_hinged_contact",
)


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    """Load a focused cross-platform subset into the main test suite."""
    del loader, tests, pattern
    suite = unittest.TestSuite(TestSolverKaminoLOX(name) for name in _LOX_QUALITY_TESTS)
    suite.addTest(TestLOXJointCovariance("test_joint_feedback_rotates_poses_velocities_and_wrenches"))
    suite.addTest(TestLOXCoupledRejection("test_unsafe_rod_rejects_coupled_trial_without_particle_warmstart_leak"))
    suite.addTests(
        TestLOXRodFeedback(name)
        for name in (
            "test_twist_damping_crosses_both_principal_branches",
            "test_mixed_twist_matches_independent_implicit_reference",
            "test_unsafe_bend_geometry_recovers_to_frozen_solution",
        )
    )
    return suite


if __name__ == "__main__":
    unittest.main(verbosity=2)
