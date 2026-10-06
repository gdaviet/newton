# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FEM assembly shared by particle and fixed-cell quadratures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import warp.fem as fem

if TYPE_CHECKING:
    from .solver_implicit_mpm import ImplicitMPMScratchpad


@dataclass
class ElasticityInputs:
    """Bind quadrature-specific material data to the shared elastic assembly."""

    quadrature: fem.Quadrature
    strain_rhs: fem.Integrand
    compliance: fem.Integrand
    fields: dict[str, fem.Field]
    values: dict[str, Any]
    compliance_values: dict[str, Any] | None = None


def assemble_elasticity(
    inputs: ElasticityInputs,
    scratch: ImplicitMPMScratchpad,
    temporary_store: fem.TemporaryStore | None,
):
    """Assemble the elastic right-hand side and compliance using the selected quadrature."""
    fields = {"tau": scratch.sym_strain_test, **inputs.fields}
    fem.integrate(
        inputs.strain_rhs,
        quadrature=inputs.quadrature,
        fields=fields,
        values=inputs.values,
        output=scratch.elastic_strain_delta_field.dof_values,
        temporary_store=temporary_store,
    )
    fem.integrate(
        inputs.compliance,
        quadrature=inputs.quadrature,
        fields={**fields, "sig": scratch.sym_strain_trial},
        values=inputs.values if inputs.compliance_values is None else inputs.compliance_values,
        output=scratch.compliance_matrix,
        temporary_store=temporary_store,
    )
