# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Functional quality gates for LOX deformables and mixed contact."""

import unittest

from newton._src.solvers.kamino.tests.test_solvers_lox_deformable_integration import (
    TestLOXDeformableIntegration,
)

_FUNCTIONAL_TESTS = (
    "test_step_hanging_stiff_cloth_preserves_pins",
    "test_step_static_contact_applies_isotropic_friction",
    "test_step_static_contact_with_alternative_projections",
    "test_first_apgd_step_matches_static_contact_jacobi",
    "test_first_apgd_step_matches_mixed_contact_jacobi",
    "test_one_color_gauss_seidel_matches_mixed_contact_jacobi",
    "test_one_color_gauss_seidel_matches_static_contact_jacobi",
    "test_step_mixed_contact_with_alternative_projections",
    "test_step_dynamic_rigid_contact_updates_both_endpoints",
    "test_step_in_place_matches_ping_pong_with_nonzero_body_com",
    "test_step_dynamic_rigid_contact_against_pinned_particle",
    "test_step_multiworld_cloth_independently",
    "test_step_pure_cloth_with_collision_pipeline_contacts",
    "test_step_ignores_fully_prescribed_rigid_contacts",
    "test_step_pure_cloth_self_contact",
    "test_deformable_projection_uses_weighted_mass_split_majorizer",
    "test_step_self_contact_with_alternative_projections",
    "test_keep_boundary_corner_self_contact",
    "test_bypass_normal_cone_filter_for_close_self_contact",
    "test_penetration_free_contact_uses_cone_pruned_candidate",
    "test_penetration_free_contact_truncates_frozen_candidates",
    "test_penetration_free_contact_truncates_edge_crossing",
    "test_penetration_free_contact_derives_tetrahedral_surface_edges",
    "test_step_applies_penetration_free_isotropic_bound",
    "test_step_combined_rigid_and_cloth_self_contact",
    "test_step_full_surface_contact_updates_cloth_and_rigid_body",
    "test_reset_pure_cloth_state_and_warm_start",
    "test_step_global_cloth_contact_with_local_rigid_body",
    "test_step_mixed_rigid_and_cloth_without_coupling",
)


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    """Load end-to-end deformable and mixed-contact regressions."""
    del loader, tests, pattern
    return unittest.TestSuite(TestLOXDeformableIntegration(name) for name in _FUNCTIONAL_TESTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
