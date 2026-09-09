# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-body primal splitting for Kamino.

``solver`` orchestrates the solve; ``problem`` owns its topology and rows.
``adapter`` converts Kamino inputs and outputs, while ``system`` assembles
and solves the primal body operator. ``projection`` supplies shared local
operations to the ``jacobi`` and ``colored_gauss_seidel`` schedules.
``types`` owns splitting state and ``kernels`` updates it and its residuals.
"""

from .bias import compute_contact_velocity_target, compute_limit_velocity_target
from .contact import (
    compute_contact_scaled_alart_curnier_residual,
    project_contact_coulomb_cone,
    solve_contact_coulomb_newton,
)
from .jacobi import project_constraints_jacobi
from .problem import LOXProblem
from .projection import (
    PROJECTION_STATUS_INVALID,
    PROJECTION_STATUS_VALID,
    compute_contact_delassus,
    compute_limit_delassus,
    compute_projection_residuals,
    prepare_jacobi_projection_data,
)
from .solver import (
    LOX_STATUS_ACTIVE,
    LOX_STATUS_CONVERGED,
    LOX_STATUS_FAILED,
    LOX_STATUS_ITERATION_LIMIT,
    LOXSolver,
    LOXStatus,
)
from .system import (
    BatchedPrimalBodySystem,
    PrimalRowContribution,
    compute_augmented_joint_row,
    compute_body_explicit_wrench,
    compute_body_inertial_system,
    compute_dynamic_joint_row,
    make_spatial_mass_matrix,
)
from .types import SplittingState
from .weight import (
    BODY_WEIGHT_STATUS_INVALID,
    BODY_WEIGHT_STATUS_VALID,
    BodyWeightResult,
    compute_body_weight_mass_proportional,
)

__all__ = [
    "BODY_WEIGHT_STATUS_INVALID",
    "BODY_WEIGHT_STATUS_VALID",
    "LOX_STATUS_ACTIVE",
    "LOX_STATUS_CONVERGED",
    "LOX_STATUS_FAILED",
    "LOX_STATUS_ITERATION_LIMIT",
    "PROJECTION_STATUS_INVALID",
    "PROJECTION_STATUS_VALID",
    "BatchedPrimalBodySystem",
    "BodyWeightResult",
    "LOXProblem",
    "LOXSolver",
    "LOXStatus",
    "PrimalRowContribution",
    "SplittingState",
    "compute_augmented_joint_row",
    "compute_body_explicit_wrench",
    "compute_body_inertial_system",
    "compute_body_weight_mass_proportional",
    "compute_contact_delassus",
    "compute_contact_scaled_alart_curnier_residual",
    "compute_contact_velocity_target",
    "compute_dynamic_joint_row",
    "compute_limit_delassus",
    "compute_limit_velocity_target",
    "compute_projection_residuals",
    "make_spatial_mass_matrix",
    "prepare_jacobi_projection_data",
    "project_constraints_jacobi",
    "project_contact_coulomb_cone",
    "solve_contact_coulomb_newton",
]
