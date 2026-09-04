# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph-capture quality gates for the LOX solver."""

import unittest

from newton._src.solvers.kamino.tests.test_solver_kamino_lox import TestSolverKaminoLOX
from newton._src.solvers.kamino.tests.test_solvers_lox_deformable_integration import (
    TestLOXDeformableIntegration,
)


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    """Load end-to-end CUDA graph replay regressions."""
    del loader, tests, pattern
    return unittest.TestSuite(
        (
            TestSolverKaminoLOX("test_cuda_graph_capture_uses_conditional_loop"),
            TestLOXDeformableIntegration("test_capture_public_pure_cloth_step"),
            TestLOXDeformableIntegration("test_capture_step_in_place_matches_ping_pong_with_nonzero_body_com"),
            TestLOXDeformableIntegration("test_capture_public_dynamic_rigid_cloth_contact_step"),
            TestLOXDeformableIntegration("test_capture_public_first_dynamic_rigid_cloth_contact_step"),
        )
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
