# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deformable particle systems of the LOX backend.

``problem`` owns the smooth system and the contact rows of the particles of a LOX
problem, which the LOX solver drives through the phases of each time step;
``solver_kernels`` holds the particle kernels of those phases. ``contact`` loads
the soft contacts that Newton detects between particle features and collider
shapes into the particle contact rows of the projection, with the kernels of
``contact_kernels`` and the normal-cone filters of ``contact_filter``.

``system`` packs the particles of the Newton cloth and tetrahedral meshes by solve
world and structural component, and assembles their frozen smooth system once per
time step: particle masses, stable membrane triangles, dihedral bending edges, and
stable Neo-Hookean tetrahedra, together with a scalar consensus weight per particle.
It solves the candidate of each splitting iteration with a fixed number of
preconditioned conjugate-residual iterations, optionally recycling recent
corrections, with dense ``LLT`` solves of small components (``direct``), or with
cuDSS (``cudss``). ``proximal`` and ``tetrahedron_proximal`` correct the right-hand
side of the candidate solves with element-local nonlinear proximal updates of the
membranes and tetrahedra. ``energy``, ``tetrahedron_energy``, and ``membrane`` hold
the constitutive functions. ``preconditioner`` (block incomplete ``LDLT``),
``jacobi`` (block Jacobi), and ``two_level`` (block Jacobi with aggregate coarse
corrections) precondition the conjugate-residual solves. ``types`` holds the data
containers, ``system_kernels`` holds the kernels of ``system``, and
``consensus_kernels`` the kernels of the patch consensus weights.
"""
