# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical primitives for the LOX rigid-contact backend."""

from .adapter import LOXKaminoAdapter
from .bias import compute_contact_velocity_target, compute_limit_velocity_target
from .contact import (
    compute_contact_scaled_alart_curnier_residual,
    project_contact_coulomb_cone,
    solve_contact_coulomb_newton,
)
from .deformable_system import validate_deformable_model
from .iteration import SplittingState
from .problem import (
    PrimalRowContribution,
    compute_augmented_joint_multiplier,
    compute_augmented_joint_row,
    compute_body_explicit_wrench,
    compute_body_inertial_system,
    compute_dynamic_joint_row,
    compute_velocity_distance,
    make_spatial_mass_matrix,
)
from .projection import (
    PROJECTION_STATUS_INVALID,
    PROJECTION_STATUS_REGULARIZED,
    PROJECTION_STATUS_VALID,
    apply_contact_desaxce_correction,
    compute_contact_delassus,
    compute_limit_delassus,
)
from .rod import validate_rod_model
from .solver import (
    LOX_STATUS_ACTIVE,
    LOX_STATUS_CONVERGED,
    LOX_STATUS_FAILED,
    LOX_STATUS_ITERATION_LIMIT,
    LOXSolver,
)
from .sweep import (
    compute_projection_residuals,
    prepare_jacobi_projection_data,
    project_constraints_jacobi,
)
from .system import BatchedPrimalBodySystem
from .weight import (
    BODY_WEIGHT_BETA_DEFAULT,
    BODY_WEIGHT_SIGMA_DEFAULT,
    BODY_WEIGHT_STATUS_INVALID,
    BODY_WEIGHT_STATUS_REGULARIZED,
    BODY_WEIGHT_STATUS_VALID,
    BodyWeightResult,
    compute_body_weight_mass_proportional,
)

__all__ = [
    "BODY_WEIGHT_BETA_DEFAULT",
    "BODY_WEIGHT_SIGMA_DEFAULT",
    "BODY_WEIGHT_STATUS_INVALID",
    "BODY_WEIGHT_STATUS_REGULARIZED",
    "BODY_WEIGHT_STATUS_VALID",
    "LOX_STATUS_ACTIVE",
    "LOX_STATUS_CONVERGED",
    "LOX_STATUS_FAILED",
    "LOX_STATUS_ITERATION_LIMIT",
    "PROJECTION_STATUS_INVALID",
    "PROJECTION_STATUS_REGULARIZED",
    "PROJECTION_STATUS_VALID",
    "BatchedPrimalBodySystem",
    "BodyWeightResult",
    "LOXKaminoAdapter",
    "LOXSolver",
    "PrimalRowContribution",
    "SplittingState",
    "apply_contact_desaxce_correction",
    "compute_augmented_joint_multiplier",
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
    "compute_velocity_distance",
    "make_spatial_mass_matrix",
    "prepare_jacobi_projection_data",
    "project_constraints_jacobi",
    "project_contact_coulomb_cone",
    "solve_contact_coulomb_newton",
    "validate_deformable_model",
    "validate_rod_model",
]
