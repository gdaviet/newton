# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Membrane geometry shared by the LOX deformable assembly and membrane proximal updates."""

import warp as wp

###
# Module interface
###

__all__ = ["membrane_area_ratio_gradient"]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


_MEMBRANE_AREA_EPSILON = 1.0e-8
"""Regularization of the membrane area ratio, which keeps the area gradient of collapsed membranes finite."""


###
# Functions
###


@wp.func
def membrane_area_ratio_gradient(deformation_0: wp.vec3, deformation_1: wp.vec3):
    """Evaluate one coherently regularized membrane area and its gradient."""
    normal = wp.cross(deformation_0, deformation_1)
    area_ratio = wp.sqrt(wp.dot(normal, normal) + _MEMBRANE_AREA_EPSILON * _MEMBRANE_AREA_EPSILON)
    inverse_area_ratio = 1.0 / area_ratio
    gradient_0 = inverse_area_ratio * wp.cross(deformation_1, normal)
    gradient_1 = inverse_area_ratio * wp.cross(normal, deformation_0)
    return area_ratio, gradient_0, gradient_1
