.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Kamino
======

:class:`~newton.solvers.SolverKamino` simulates constrained rigid multi-body
systems in maximal coordinates. It is designed for mechanical assemblies with
kinematic loops, under- or overactuation, joint limits, hard frictional
contacts, and restitutive impacts.

Unlike the other maximal-coordinate solvers, Kamino focuses on constrained
rigid mechanical assemblies. Its experimental LOX backend additionally supports
rod joints and cloth and tetrahedral deformable bodies.
Kamino is currently in BETA 1, and Newton users are discouraged from depending
on it. Evaluate it only when kinematic loops and hard contact constraints are
primary requirements and an experimental solver is acceptable.

.. experimental::

   :class:`~newton.solvers.SolverKamino` is experimental. Its public API,
   behavior, feature support, performance, and implementation may change
   without prior notice.

See the :class:`~newton.solvers.SolverKamino` API reference for construction
and configuration details. Runnable workflows are available in the
`Kamino examples <https://github.com/newton-physics/newton/tree/main/newton/examples/kamino>`_.

Body acceleration
-----------------

Kamino populates :attr:`~newton.State.body_qdd` when the extended state
attribute is requested. Values are center-of-mass spatial accelerations in the
world frame, with linear acceleration followed by angular acceleration. For a
step of duration ``dt``, Kamino reports the discrete step average
``(body_qd_out - body_qd_in) / dt``. Consequently, a contact impact reports its
velocity impulse divided by ``dt`` rather than a continuous midpoint
acceleration. Resetting a state clears the requested acceleration in each reset
world.

Choosing a dynamics solver
--------------------------

Kamino provides three forward-dynamics backends:

* ``"padmm"`` (default): proximal ADMM, dense Jacobians/dynamics, and the Euler
  integrator. It is the slower, more robust option because it solves equality
  and inequality constraints together.
* ``"dvi"`` (opt-in): projected dual iterations, sparse Jacobians, dense dynamics
  with the RCM-reordered blocked LLT solver, and the Euler integrator. It is
  generally faster, but approximates the coupled problem by alternating between
  a direct solve for equality constraints and projected iterations for
  inequality constraints. As a rule of thumb, DVI solves inequality constraints
  less accurately than PADMM, particularly as the number of active inequalities
  grows. Dual preconditioning is not supported.
* ``"lox"`` (opt-in, experimental): a primal splitting method. Like DVI, it
  alternates between solving for equality constraints (and smooth dynamics in
  general) and solving for inequality constraints; however, in LOX the split is
  performed at the velocity level, and ADMM is used to reach a consensus.

Select the backend when constructing the configuration so dependent defaults
initialize consistently:

.. code-block:: python

   config = newton.solvers.SolverKamino.Config(dynamics_solver="dvi")
   solver = newton.solvers.SolverKamino(model, config=config)

DVI is best suited to performance-sensitive rigid mechanisms with relatively
few active contacts; PADMM remains the safer and more broadly validated choice.
Set ``sparse_jacobian=False`` for fully dense DVI, or set
``sparse_dynamics=True`` to use sparse dynamics with the Conjugate Residual
solver.

For large bilateral systems, opt into RCM-reordered factorization explicitly:

.. code-block:: python

   config.dvi.bilateral_solver_type = "LLTBRCM"
   config.dvi.bilateral_solver_kwargs = {
       "block_size": 32,
       "reuse_permutation": True,
       "parallel_factorization": True,
   }

The cached permutation remains mathematically valid when matrix values or
sparsity change and is recomputed automatically if the active dimension
changes. Keep the default ``"LLTB"`` solver for small systems.

DVI unilateral subsolvers
^^^^^^^^^^^^^^^^^^^^^^^^^

DVI offers two subsolvers for bounded joint rows, limits, and contacts:

* **PGS** (default) uses projected Gauss-Seidel updates with less work per
  iteration, making it a simple choice for modest contact problems.
* **APGD** offers a faster convergence rate for each frozen-correction
  quadratic problem, at the cost of objective-gradient evaluations and an
  inner backtracking loop.
  CUDA execution requires conditional-graph support.

Both support dense and sparse operators and bilateral Schur elimination.
See the :class:`~newton.solvers.SolverKamino` configuration reference for
subsolver selection and iteration controls.

LOX prescribed bodies and shared settings
-----------------------------------------

LOX advances bodies flagged :attr:`~newton.BodyFlags.KINEMATIC`, and massless
bodies whose only joints are free joints to the world, with their velocity in
``state_in.body_qd``. To animate such a body, write the velocity that reaches its
target pose within the step. PADMM and DVI keep the pose of these bodies and only
read their velocity.

Bodies welded to a kinematic body by fixed joints of its articulation move
rigidly with it, like the bodies of a MuJoCo tree without joints: LOX sets their
pose and velocity from the kinematic body at the beginning and at the end of each
step. Massless bodies on other joints stay dynamic and move with the bodies they
are attached to. With ``config.lox.eliminate_fixed_world_islands`` enabled, LOX
also keeps the bodies welded to the world at their pose.

LOX simulates bodies flagged :attr:`~newton.BodyFlags.PROXY` as dynamic bodies
with their mass and inertia.

LOX keeps speculative contacts, whose shapes are still separated, and lets each
close its gap within the step. It ignores
``config.dynamics.cull_speculative_contacts``.

LOX bounds joint drift with ``config.lox.joint_max_correction``,
``config.lox.position_tolerance`` and ``config.lox.rotation_tolerance``, so it
ignores ``config.constraints.alpha``. ``config.collect_solver_info`` collects the
PADMM and DVI iteration logs; LOX reports its iterations and residuals through
:attr:`~newton.solvers.SolverKamino.status`.

LOX solves its own primal problem, so ``sparse_dynamics``,
``config.dynamics.linear_solver_type``, ``config.dynamics.linear_solver_kwargs``
and ``config.dynamics.preconditioning`` only affect the dual problem of the
optional solution metrics. LOX always warm-starts from its containers;
``config.lox.contact_warmstart_method`` selects how it matches contacts.

LOX unilateral projection
-------------------------

LOX projects contacts, joint limits, and joint friction with mass-split
sweeps. ``config.lox.gauss_seidel_max_colors`` selects the sweep: one gives a
fully parallel Jacobi sweep, and larger values process up to that many
approximate constraint colors sequentially. ``config.lox.projection_acceleration``
optionally accelerates the resulting map:

* ``"none"`` (default) runs plain sweeps.
* ``"apgd"`` applies restarted Nesterov extrapolation and requires
  ``gauss_seidel_max_colors=1``.
* ``"anderson"`` applies one safeguarded Anderson direction
  after every sweep, for either the Jacobi or the colored map. Set
  ``projection_anderson_recycle=False`` to discard the direction between
  splitting iterations.

LOX contact extensions
----------------------

Enable these features through ``config.lox`` after selecting
``dynamics_solver="lox"``. They are disabled by default.

* ``contact_compliance=True`` turns contacts with a positive per-contact
  stiffness ``k`` [N/m] (:attr:`newton.Contacts.rigid_contact_stiffness`) into
  implicit spring-dampers, with the damping ``d`` [N s/m] of
  :attr:`newton.Contacts.rigid_contact_damping`. They add
  ``1 / (dt (k dt + d))`` to the normal contact operator and recover the fraction
  ``k dt / (k dt + d)`` of existing penetration per step. Their bounce comes from
  the spring-damper, so they ignore restitution. Contacts without a positive
  stiffness, and contacts detected by Kamino's own collision pipelines, remain
  hard. :class:`newton.CollisionPipeline` provides per-contact stiffness only for
  hydroelastic contacts.
* ``contact_implicit_restitution=True`` decides the restitution of speculative
  impacts inside each local contact update, from its current reaction-free normal
  velocity, which better preserves energy in speculative impacts. Restitution
  applies either way: by default, the bounce target is fixed at the beginning of
  the step.
* ``contact_spatial_friction=True`` accounts for the shape ``mu_torsional`` and
  ``mu_rolling`` coefficients, both in metres. Sliding, torsion, and rolling
  share one elliptic friction budget. Contacts with zero angular coefficients
  retain the classical 3D contact solve. Kamino's primitive collision pipeline
  provides no angular coefficients.

LOX rod joints
--------------

LOX supports ``JointType.ROD`` joints between rigid bodies, as built by
:meth:`newton.ModelBuilder.add_joint_rod`, which the other Kamino backends
reject. Each rod adds six smooth material rows to the body system: stretch and
shear in the joint frame of the parent body, then bend and twist, with the
stiffness and damping of its six joint DOFs. As in ``SolverVBD``, the rest bend
and twist come from the angular rest rotation in ``Model.joint_target_q``;
call ``notify_model_changed`` with
``ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES`` after changing it. The rows are
linearized at the beginning of each time step. Set
``config.lox.rod_proximal_relaxation`` in
``(0, 1]`` to also correct the bend and twist rows towards their exact
candidate-pose strain during the splitting iterations. Rod joints require zero
armature, joint damping, and joint friction, and unbounded joint limits.

LOX deformables
---------------

LOX simulates the particles of Newton cloth and tetrahedral meshes, such as
those built by :meth:`newton.ModelBuilder.add_cloth_grid` and
:meth:`newton.ModelBuilder.add_soft_grid`, in the same splitting iterations as
the rigid bodies. Triangles add a stable membrane energy, bending edges a
dihedral bending energy, and tetrahedra a stable Neo-Hookean energy, with the
stiffness and damping coefficients of the model. Particles with zero mass or
without the ``ACTIVE`` flag are prescribed. The particle velocities join the
consensus and change residuals of the terminal status.

Each time step assembles one frozen smooth particle system. Its candidate
solves run ``config.lox.deformable_cr_iterations`` preconditioned
conjugate-residual iterations, with the preconditioner of
``config.lox.deformable_preconditioner``, and solve the components of at most
``config.lox.deformable_direct_max_particles`` particles with a dense Cholesky
factorization. ``config.lox.deformable_linear_solver="cudss"`` solves the
whole system with a cuDSS factorization, which requires the
``nvmath-python`` package and ``config.lox.fixed_iterations=True``;
``"cudss_then_cr"`` solves the first candidate of each step with cuDSS and the
next ones with conjugate residuals. ``config.lox.deformable_proximal_iterations``
sets the local iterations of the nonlinear membrane and tetrahedron
corrections. The projection weight of each particle comes from the Schur
complements of the particle in a patch of at most eight particles of its
elements, and ``config.lox.deformable_consensus_fraction`` sets the fraction of
the weights in the particle solve, the counterpart of
``config.lox.consensus_fraction``.

The soft contacts of the ``contacts`` argument of
:meth:`~newton.solvers.SolverKamino.step`, between particle features and the
shapes of the collision pipeline, share the unilateral projection with the
rigid contacts and transfer their reactions to dynamic bodies. Set
``config.lox.deformable_enable_self_contact=True`` to also project the
vertex-triangle and edge-edge self-contacts of the deformable surfaces, and
``config.lox.deformable_enable_penetration_free_contact=True`` to truncate the
particle steps against the self-contact candidates so that the surfaces keep
their sides.

LOX deformables support particles, triangles, bending edges, and tetrahedra;
models with springs or muscles are rejected. Global particles require a single
world, and models with particles use the Euler integrator.

Inspecting terminal status
--------------------------

After each step, :attr:`~newton.solvers.SolverKamino.status` provides one
device-resident terminal status record per world. PADMM and DVI both provide
``converged``, ``iterations``, ``r_p``, ``r_d``, and ``r_c`` fields. Their
residual definitions are backend-specific:

* **PADMM:** Let ``x`` and ``y`` be the current preconditioned impulse iterates,
  ``x_prev`` and ``y_prev`` their previous values, ``P`` the diagonal dual
  preconditioner, and ``eta`` and ``rho`` the proximal and penalty parameters.
  ``r_p = ||P (x - y)||_inf`` is the primal consensus residual
  [N·s or N·m·s].
  ``r_d = ||P^-1 (eta (x - x_prev) + rho (y - y_prev))||_inf`` is the ADMM dual
  residual [m/s or rad/s]. ``r_c`` is the maximum absolute impulse-velocity
  inner product over inequality blocks [J]. The ``P`` factors
  convert the first two residuals from solver scaling to physical constraint
  units.
* **DVI:** With physical impulse ``lambda`` and augmented constraint velocity
  ``v``, ``r_p`` is the maximum infinity-norm projection distance of unilateral
  impulses from the nonnegative limit cone or Coulomb contact cone
  [N·s or N·m·s]. ``r_d`` is the maximum of the corresponding velocity distance
  from the dual cone and the bilateral velocity violation [m/s or rad/s].
  ``r_c = max |lambda_k dot v_k|`` is the maximum inequality complementarity
  violation [J].
* **LOX:** ``converged`` and ``iterations`` report LOX's native splitting
  termination state. With body twists ``v`` of the smooth solve and ``p`` of
  the unilateral projection, and the splitting body weight ``W``,
  ``r_p = ||W (v - p)||_inf`` is the consensus residual [N·s or N·m·s] and
  ``r_d = ||v - v_prev||_inf`` is the last change of the smooth solution
  [m/s or rad/s]. ``r_c`` is the squared maximum Delassus-scaled natural-map
  residual of the joint-friction, limit, and contact rows under their contact
  law [J], which measures feasibility and complementarity together. These
  residuals do not require solution metrics; they are NaN before a solve and
  for failed worlds. LOX additionally reports ``failed``; a world that is
  neither converged nor failed reached the iteration limit and uses its last
  projected iterate. A failed world stays failed until it is reset, for example
  with ``solver.reset(..., world_mask=...)``. Until then, LOX skips its solve,
  and its bodies keep their input velocity, without gravity, joint or contact
  forces, so check ``failed`` to detect and reset these worlds.

These are absolute maxima: no backend divides them by a reference norm,
constraint count, or tolerance. Additional fields are not portable between
backends.

.. code-block:: python

   status = solver.status
   assert status.device == model.device
   assert status.shape == (model.world_count,)

   # Host inspection is explicit and synchronizes the device-to-host copy.
   status_host = status.numpy()
   unconverged_worlds = (~status_host["converged"].astype(bool)).nonzero()[0]

Terminal status is always maintained. ``collect_solver_info=True`` enables
additional solver diagnostics and adds runtime and memory overhead; it is not
required to access ``status``.

Actuation and forward kinematics
--------------------------------

Kamino dynamics routes actuation independently for each joint DoF. A DoF can
use explicit effort, unbounded implicit PD, or effort-limited implicit PD;
passive armature, damping, and Coulomb friction are likewise configured per
DoF. Implicit-PD target modes require a non-zero applicable gain: velocity
mode requires derivative gain, while position-based modes require proportional
or derivative gain. Coulomb friction supports all non-free joint types, while
joint dynamics and implicit PD currently support revolute, prismatic, and
gimbal joint types only.

The forward-kinematics solver still partitions each joint as entirely passive
or entirely actuated. Different non-passive target modes are allowed within a
joint, but mixing passive and actuated DoFs within one joint is not yet
supported. The ``fk_actuation_flag`` model attribute provides an explicit
joint-level override for this FK partition.
