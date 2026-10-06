.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Implicit MPM
============

:class:`~newton.solvers.SolverImplicitMPM` supports experimental recovery
controls for simulations that stop the nonlinear solve after a small number
of iterations. The controls correct accumulated volume and contact errors
over subsequent steps. They retain the selected strain basis and iteration
budget.

P0 transfers and recovery
------------------------

For a P0 strain basis, ``integration_scheme="cell"`` evaluates strain at
fixed cell centers. Particle mass and momentum travel through the centers
to velocity nodes, and solved velocities return through the same centers.
PIC collider rows use that same two-hop interpolation. This keeps velocity
modes invisible to the P0 strain samples from reaching particles through
a different transfer path.

The following runtime configuration combines the three controls:

.. code-block:: python

   from newton.solvers import SolverImplicitMPM

   config = SolverImplicitMPM.Config(
       voxel_size=0.02,
       velocity_basis="Q1",
       strain_basis="P0",
       integration_scheme="cell",
       collider_basis="pic8",
       solver="gs",
       max_iterations=5,
       density_strain_fraction=0.05,
       collider_stabilization_fraction=0.2,
       collider_contact_gap=0.01,
   )
   solver = SolverImplicitMPM(model, config=config)

The values illustrate a starting configuration; tune the fractions and
contact distance for the scene and timestep. These experimental options
are runtime-only and are not authored through USD schemas.

``density_strain_fraction`` injects a fraction of the signed filling error
into the existing divergence right-hand side. Particle reference volume is
compared against grid volume minus collider volume. Overfilled cells request
expansion. A value of zero disables correction, and supported fractions lie
in ``[0, 1]``. This measurement uses the strain quadrature already needed by
the solver and requires no deformation history. It also works with ordinary
PIC and GIMP integration.

``collider_stabilization_fraction`` requests an outward relative normal
velocity of ``fraction * penetration / dt``. ``collider_contact_gap`` activates
separated PIC contacts within the specified distance in meters. Their normal
target permits closing the current gap during the step, rather than forcing
zero approach velocity. Both controls support ``"pic"`` and ``"picN"`` collider
bases and default to zero. They do not change collider tangential velocity.
Choose a contact distance large enough to cover expected relative motion
within one step; this is a finite predictive range.

``max_iterations`` bounds the nonlinear solve, including budgets below a
solver's usual batch size. CUDA graphs retain batches of five when the
budget is divisible by five; other budgets use individual iterations.

.. experimental::

   Cell integration and recovery controls may change without prior notice.
   Cell integration currently requires Q1 velocity, one shared FEM
   environment, positive particle density, zero ``critical_fraction``, and
   materials without hardening, viscosity, or dilatancy. Standard PIC/GIMP
   integration retains its existing material support. Recovery improves
   error handling after truncated solves; it does not guarantee convergence
   for every scene or iteration budget.

Residual volume history
-----------------------

With PIC or GIMP integration, ``residual_strain_fraction`` can correct the
log determinant of accumulated residual deformation instead of measuring
cell filling. Residual history subtracts the solver's elastic and plastic
increments from the sampled symmetric velocity-gradient increment.
Positive fractions enable recording automatically; ``residual_strain_tracking``
can also record history with feedback disabled. Resetting a state restores
its residual deformation gradient to identity, including masked world resets.
Cell integration uses the density controller and does not support this
residual-history mode.

Validation
----------

The recovery regression module checks volume recovery with three iterations,
penetration recovery and predictive gap closure with one iteration, exact
iteration budgets on CPU and CUDA graphs, residual-history reset, and full
step graph replay on both supported test GPUs.

``ImplicitMPMP0Recovery`` in the ASV simulation benchmarks compares captured
steps with 2,744 elastic particles and ten Jacobi iterations on a fixed grid.
On an RTX PRO 6000 Blackwell with Warp 1.17.0, the median of five measurements
of ten steps was:

.. list-table::
   :header-rows: 1

   * - Transfer path
     - Density fraction
     - Time per step [ms]
   * - PIC
     - 0
     - 1.508
   * - PIC
     - 0.05
     - 1.574
   * - Cell centers
     - 0
     - 1.515
   * - Cell centers
     - 0.05
     - 1.554

This measures transfer and volume-feedback costs in a small scene without
colliders. Contact-heavy workloads and larger scenes need separate timing.
