# Implicit MPM APGD Solver Plan

Status: implemented opt-in solver; validation and tuning remain ongoing

Progress:

- Stage 1 projection prototype completed on 2026-07-29.
- Stage 2 fixed-step nodal prototype completed on 2026-07-29.
- Stage 3 subgrid contact and rigid collider response completed on 2026-07-29.
- The internal Stage 4 prototype now separates feasible and extrapolated
  states, implements the Algorithm B.2 inertia/restart, computes the scaled
  BB2 step from raw coupled responses, and checks stress and contact residuals
  independently.
- Twenty-four test functions generate 72 passing cases across CPU, `cuda:0`,
  and `cuda:1`.
- On the synthetic coupled regression, acceleration reaches a `1e-4`
  two-family residual tolerance in 55 iterations, compared with 884
  fixed-step iterations.
- APGD is exposed through `SolverImplicitMPM.Config(solver="apgd")` and the
  tracked MPM examples accept `--solver apgd`. Restart, residual, inertia, BB2,
  spectral step, iteration count, and termination state remain on-device
  throughout the iteration loop. Reductions always use explicit device
  outputs; the host reads only the final compact diagnostic state.
- The device-conditional CUDA graph loop is capture-safe and validated on an
  RTX PRO 6000 Blackwell and an RTX 3080 Ti. The first implementation using
  `wp.utils.array_sum` exposed a hidden allocation during conditional capture;
  it was replaced by a preallocated tiled metric reduction.
- Separated MPM worlds now keep independent acceleration, restart, and BB2
  state. Stress and contact residuals are reduced per environment, including
  empty worlds and fixed-grid capacity padding, and termination requires every
  environment to satisfy both scaled L2 and block-infinity tolerances.
- Warmed graph benchmarks show negligible acceleration overhead per iteration
  and approximately 12x lower time to a matched residual on the associated
  snow-ball snapshot. A larger B2/P1d subgrid snapshot shows an 18x reduction
  in time to its selected residual threshold.
- A solver-independent fixed-step diagnostic now compares the prototype with
  the production Gauss-Seidel and Jacobi paths. Production solvers win on the
  nodal snapshot and at a loose subgrid tolerance; APGD wins by 2.32x and
  1.96x over Jacobi on the two test GPUs at the tighter subgrid tolerance.
- Stress-side de Saxcé/bipotential bias and viscosity are implemented in the
  same projected fixed-point step. The viscosity-preconditioned local
  resolvent keeps the large-viscosity mapping finite. Converged coupled
  regressions agree with the existing Gauss-Seidel solver for moderate and
  large non-associated viscosity.
- Unbounded fluid pressure uses an exact cylindrical projection instead of
  the finite \(10^{15}\) pressure polygon. This preserves the float32 pressure
  multiplier, excludes it from the viscous flow metric, and restores
  incompressibility in the complete funnel example.
- One-sided unbounded pressure limits also use their exact rays instead of
  artificial far-cap vertices. This covers the default infinite
  Drucker-Prager cone and the less common infinite-tension configuration
  without overflow or sentinel-dependent projection errors.
- Sparse associated, sparse viscous, and fixed-grid non-associated granular
  examples pass CUDA smoke tests. The fixed-grid test also covers collider
  contact and outer graph capture.
- On the unmodified non-associated B2/P1d granular snapshot, batched
  Gauss-Seidel reaches a joint `1.0` residual threshold 7.23x faster than
  APGD. APGD is 2.18x faster than Jacobi and 2.41x faster than ordinary
  Gauss-Seidel, but its contact residual is over-resolved by about 15x at the
  threshold.

## Objective

Add a matrix-free accelerated projected-gradient variant for the implicit MPM
solve. Each iteration advances one coupled unknown

\[
z = (\sigma, \lambda),
\]

where each stress block \(\sigma_i\) has six components and each collider
impulse block \(\lambda_j\) has three components. Every iteration must project
both block families:

- stresses onto the material admissible set;
- collider impulses onto the adhesive Coulomb cone.

The local admissible-set operation is an orthogonal projection onto the
piecewise-linear yield surface. Non-associated material flow is represented by
a stress-side de Saxcé/bipotential correction, and viscosity by shifting to
the rate-dependent yield stress before projection. Collider contact uses the
analogous de Saxcé velocity correction.

This is an additional solver path. Existing Gauss-Seidel, Jacobi, batched, and
linear solver behavior must remain unchanged.

## Initial scope

The first working version should support:

- six-component stress constraints and three-component collider constraints in
  one APGD iteration;
- nodal and subgrid collider bases;
- collider adhesion and friction;
- optional rigid-body feedback through the existing rigidity operator;
- stress and collider-impulse warm starts;
- a scalar preconditioner per stress block and per contact block;
- convergence tests covering both stress and contact fixed-point residuals.

The current version requires the piecewise-linear yield surface
(`_USE_CAM_CLAY == False`). It supports viscosity and both associated and
non-associated flow. APGD is intentionally a standalone joint solver and
cannot be chained with the alternating solver names. Existing solvers remain
the default.

## Coupled operator

Use the current matrix conventions:

- \(M^{-1}\): `MomentumData.inv_volume`;
- \(B\): `RheologyData.strain_mat`, mapping grid velocity to strain rate;
- \(C\): `RheologyData.compliance_mat`;
- \(A\): `CollisionData.collider_mat`, mapping grid velocity to collider
  samples. For nodal contact, \(A=I\);
- \(R=J M_b^{-1}J^T\): rigid collider response represented by
  `CollisionData.rigidity_operator`.

Keep immutable baseline grid and collider velocities, \(v_f\) and \(v_{c,f}\),
for the duration of the solve. For an extrapolated APGD iterate
\(z^k=(\sigma^k,\lambda^k)\), reconstruct

\[
v^k = v_f + M^{-1}(B^T\sigma^k + A^T\lambda^k),
\]

\[
v_c^k = v_{c,f} - R\lambda^k.
\]

The raw coupled response is

\[
u^k =
\begin{bmatrix}
u_\sigma^k \\
u_\lambda^k
\end{bmatrix}
=
\begin{bmatrix}
e + Bv^k + C\sigma^k \\
Av^k-v_c^k
\end{bmatrix}
=Wz^k+q,
\]

with

\[
W =
\begin{bmatrix}
C+BM^{-1}B^T & BM^{-1}A^T \\
AM^{-1}B^T & AM^{-1}A^T+R
\end{bmatrix}.
\]

Here \(e\) contains the existing elastic/unilateral strain offset entering the
local stress relation. Its exact construction should reuse the current
preprocessing conventions rather than duplicate constitutive setup.

The implementation must stay matrix-free. One operator evaluation should:

1. restore or overwrite the trial velocities from the immutable baselines;
2. apply \(B^T\sigma^k\) and \(A^T\lambda^k\) to the grid;
3. apply rigid-body response to obtain \(v_c^k\);
4. evaluate \(u_\sigma^k=e+Bv^k+C\sigma^k\);
5. evaluate \(u_\lambda^k=Av^k-v_c^k\).

Do not incrementally accumulate physical grid or collider velocities across
APGD iterations. Extrapolated iterates are generally infeasible, so the final
physical velocities must be reconstructed once from the selected feasible
projected iterate.

## Local projections

### Associated stress projection

Add a new exact orthogonal projector; do not change or reuse the existing
`project_stress()`, which is deliberately non-orthogonal.

In the local stress basis, write

\[
\sigma=(r_N,r_T), \qquad t=\lVert r_T\rVert.
\]

For the current piecewise-linear yield law, the admissible section in
\((r_N,t)\) is the convex polygon

\[
p_{\min}\le r_N\le p_{\max}, \qquad
0\le t\le y(r_N),
\]

where `shear_yield_stress()` supplies the three roof segments. Project
\((r_N,t)\) onto that polygon by testing the interior and the closest point on
each edge. Restore the deviatoric direction afterward:

\[
r_T^{\,\mathrm{proj}} =
\begin{cases}
(t_\mathrm{proj}/t)r_T,&t>0,\\
0,&t=0.
\end{cases}
\]

This construction covers the tensile/compressive caps, the rising and falling
frictional branches, the plateau, cohesion, and their corners. Degenerate
zero-friction and zero-pressure cases require explicit tests. If either
pressure cap is represented by the infinite sentinel, replace the incident
far edges by their analytic rays. In particular, the default granular law is
the semi-infinite Drucker-Prager cone

\[
r_N\ge p_{\min}, \qquad
0\le t\le s+\mu(r_N-p_{\min}),
\]

whose boundary is the finite tensile cap followed by a rising ray.

### Collider impulse projection

Add an orthogonal projection helper for

\[
K_\mu=\{r:r_N\ge0,\ \lVert r_T\rVert\le\mu r_N\}.
\]

For \(x=(x_N,x_T)\), \(t=\lVert x_T\rVert\):

- return zero when \(x_N+\mu t\le0\);
- return \(x\) when \(t\le\mu x_N\);
- otherwise project onto the cone boundary with

\[
r_N=\frac{x_N+\mu t}{1+\mu^2},\qquad
r_T=\frac{\mu r_N}{t}x_T.
\]

Adhesion is a translation:

\[
\Pi_{K_{\mu,a}}(\lambda)
=\Pi_{K_\mu}(\lambda+a n)-a n.
\]

Negative friction disables the contact node and projects its impulse to zero.
Keep the existing non-orthogonal `project_on_friction_cone()` unchanged for
the existing solvers.

## De Saxcé-corrected APGD iteration

Maintain two coupled states:

- \(\widehat z^k\): the feasible projected state;
- \(z^k\): the potentially infeasible extrapolated state at which the raw
  operator is evaluated.

Evaluate \(u^k=Wz^k+q\). For a material dilatancy ratio \(\theta\), evaluate
the active yield-surface slope \(y'\) at the viscous yield stress and correct
the material response by

\[
\widetilde u_\sigma^k
=u_\sigma^k+
(1-\theta)y'(\sigma_{\mathrm y,N}^k)
\lVert(u_\sigma^k)_T\rVert e_N.
\]

The correction vanishes for associated flow (\(\theta=1\)). With viscosity
\(\eta\) and strain-node volume \(V\), first shift the total stress to

\[
\sigma_\mathrm y^k
=\sigma^k+\frac{\eta}{V}u_\sigma^k.
\]

Let \(D_\sigma\) be the local stress Delassus block and
\(T_\eta=I+(\eta/V)D_\sigma\). Project the viscous yield stress,

\[
\sigma_{\mathrm y,p}^{k+1}
=\Pi_{K_\sigma}\left(
\sigma_\mathrm y^k-\xi_kH_\sigma^{-1}\widetilde u_\sigma^k
\right),
\]

then recover a consistent total stress through the local resolvent,

\[
\widehat\sigma^{k+1}
=\sigma^k+T_\eta^{-1}
\left(\sigma_{\mathrm y,p}^{k+1}-\sigma_\mathrm y^k\right).
\]

Thus the projection remains the same simple orthogonal projection; the
non-associated bias changes its direction and viscosity changes the stress
variable in which it is applied. The resolvent prevents the explicit
\(-(\eta/V)u_\sigma\) map-back from amplifying updates when viscosity or the
inverse strain-node volume is large.

When both pressure bounds are the infinite material sentinel, the admissible
set is cylindrical: pressure is an unconstrained incompressibility multiplier.
In that case, set \((T_\eta)_{NN}=1\), omit the analytically cancelling viscous
pressure shift, preserve the normal stress exactly in the projector, and
project only the deviatoric component. This avoids subtracting
\(O(10^{15})\) polygon vertices to recover ordinary fluid pressures in
float32.

For every enabled collider node, split the relative velocity into normal and
tangential parts and apply the de Saxcé correction

\[
s_\lambda(u_\lambda^k)
=\mu\lVert (u_\lambda^k)_T\rVert n,
\]

\[
d_\lambda^k
=-\left(u_\lambda^k+s_\lambda(u_\lambda^k)\right).
\]

The corrected relative velocity must come from the same extrapolated stress
and contact-impulse fields. It must not be lagged from a previous contact
sweep.

With block-scalar preconditioner \(H\), update both families before
extrapolating either. The contact update is

\[
\widehat\lambda^{k+1}
=\Pi_{K_{\lambda,a}}\left(
\lambda^k+\xi_kH_\lambda^{-1}d_\lambda^k
\right).
\]

Then extrapolate the complete feasible state:

\[
z^{k+1}
=\widehat z^{k+1}
+\beta_k(\widehat z^{k+1}-\widehat z^k).
\]

Restart acceleration using the projected-state/search-direction test from
Algorithm B.2. Recompute the raw response \(u^{k+1}=Wz^{k+1}+q\) after
extrapolation.

The spectral step uses raw operator differences, not de Saxcé-corrected
directions. In the block-scaled metric, use

\[
\xi_{k+1}
=\frac{\langle\Delta z,\Delta u\rangle}
{\langle\Delta u,H^{-1}\Delta u\rangle},
\]

with finite-value guards and conservative lower/upper clamps.

Once the de Saxcé correction is present, this is a corrected fixed-point
iteration rather than minimization of the original quadratic. Do not use
quadratic-objective decrease as a correctness or line-search criterion.

Reference: G. Daviet, *Modeling and Simulating Complex Materials Subject to
Frictional Contact*, Algorithm B.2, page 229, and the de Saxcé change of
variables in the friction chapter:
<https://gdaviet.fr/files/these-gdaviet-archivage.pdf>.

## Preconditioning

Use positive scalar factors within each local block so the projections remain
ordinary Euclidean projections:

- \(H_{\sigma,i}=h_{\sigma,i}I_6\), initialized from a safe upper bound of the
  existing local split-mass Delassus/compliance block;
- \(H_{\lambda,j}=h_{\lambda,j}I_3\), initialized from the existing collider
  Delassus diagonal including rigid-body response.

Clamp both factors away from zero. Keep the global spectral multiplier
\(\xi_k\) separate from these local factors. More elaborate anisotropic
preconditioners would require metric projections and are out of scope for the
first implementation.

## Residual and returned iterate

Evaluate convergence for both constraint families. A preconditioned
fixed-point residual can be formed with a fixed diagnostic step \(\rho\):

\[
F_\sigma(\sigma)
=\sigma-\left[
\Pi_{K_\sigma}
\left(\sigma_\mathrm y-\rho H_\sigma^{-1}\widetilde u_\sigma\right)
-\frac{\eta}{V}u_\sigma
\right],
\]

\[
F_\lambda(\lambda)
=\lambda-\Pi_{K_{\lambda,a}}
\left[
\lambda-\rho H_\lambda^{-1}
\left(u_\lambda+s_\lambda(u_\lambda)\right)
\right].
\]

Track the weighted L2 and block-infinity norms for each family; termination
requires both to meet their independently scaled tolerances in every
environment. Acceleration, restart, and BB2 state are environment-local so
unrelated worlds cannot alter one another's APGD trajectory.

Return the best or latest feasible \(\widehat z\), never the extrapolated
\(z\). Reconstruct grid velocity, collider velocity, stress, and collider
impulse from that feasible state before existing rheology postprocessing.

## Code structure

- `rheology_solver_kernels.py`
  - exact stress projection and non-associated de Saxcé correction;
  - viscoplastic stress APGD update, extrapolation, and residual kernels.
- `contact_solver_kernels.py`
  - exact orthogonal adhesive-cone projection helper;
  - de Saxcé velocity correction;
  - contact APGD update, extrapolation, and residual kernels.
- `solve_rheology.py`
  - a coupled `_APGDSolver` owning both stress and impulse temporaries;
  - a matrix-free raw operator evaluation shared by nodal and subgrid contact;
  - BB reductions, restart state, joint stopping criteria, and final feasible
    state reconstruction.
- `solver_implicit_mpm.py`
  - public `"apgd"` solver selection and user-facing configuration docs.

Avoid forcing `_APGDSolver` through `_RheologySolver` plus `_ContactSolver` if
that preserves their current alternating, in-place update semantics. A single
owner for the coupled APGD state is clearer and makes it harder to evaluate
stress and contact at different extrapolated iterates.

## Implementation stages

### Stage 1: projection prototype

1. Implement the exact orthogonal friction-cone projector alongside the
   existing projector.
2. Implement the exact piecewise-linear stress projector alongside
   `project_stress()`.
3. Add focused Warp/unittest coverage on CPU and available CUDA devices.
4. Do not wire a new public solver option yet.

### Stage 2: unaccelerated coupled projected iteration

1. Introduce `_APGDSolver` and the immutable velocity baselines.
2. Implement the complete matrix-free raw operator for stress plus nodal
   contact.
3. Apply both projections in every iteration with \(\beta=0\) and a fixed,
   conservative step.
4. Add de Saxcé correction to the contact direction.
5. Validate contact-free, frictionless, and frictional nodal cases.

This stage isolates operator/sign/projection errors before adding acceleration.

### Stage 3: subgrid and rigid collider response

1. Reuse `collider_mat` and its transpose for \(A\) and \(A^T\).
2. Include \(R\lambda\) through the existing rigidity matrices.
3. Validate nodal versus equivalent subgrid configurations and dynamic rigid
   colliders.

### Stage 4: acceleration and robust convergence

1. [Complete] Add projected/extrapolated state separation.
2. [Complete] Add Algorithm B.2 restart behavior.
3. [Complete] Add the guarded block-scaled BB step using raw responses.
4. [Complete] Add separate per-environment stress/contact L2 and block-infinity
   residual reductions and always return the latest feasible iterate.
5. [Complete for the synthetic fixture] Compare fixed-step projected-gradient
   and accelerated convergence on the same coupled system. Repeat this on
   end-to-end MPM scenes during Stage 5.

### Stage 5: integration and capture

1. [Complete] Add the internal and public `"apgd"` solver selector.
2. [Complete] Document behavior in `SolverImplicitMPM.Config.solver`.
3. [Complete] Add a changelog entry because solver selection is user-facing.
4. [Complete] Keep all reductions and acceleration control on-device and
   provide a capture-safe `wp.capture_while` loop.
5. [Complete] Expose APGD in the tracked snow-ball, granular, and viscous MPM
   examples.
6. [Complete] Validate sparse and fixed-grid outer graph capture on CUDA.
7. Run broader implicit MPM tests and pre-commit before committing.

## Prototype benchmark results

Measurements use warmed conditional CUDA graphs and CUDA events. Dataset
loading, APGD construction and preprocessing, kernel compilation, and graph
capture are excluded. Each reported nodal value is the median of five fresh
solves; the larger subgrid values are medians of three.

The following APGD-versus-fixed-PGD tables use the prototype's internal
residual, which is normalized by its current adaptive step. That is suitable
for comparing these two variants of the same iteration, but not for comparing
against another solver. The production-solver comparison below instead
evaluates every final state with one fresh fixed diagnostic projection using
\(\rho=0.25\).

The associated `rheology_snow_ball_0000.npz` snapshot contains 92,910 stress
blocks, 114,993 nodal contact blocks, and 20,296 enabled contacts.

| Device | Variant | Iterations | Time [ms] | Time/iteration [ms] | Stress RMS | Contact RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | Fixed | 100 | 9.741 | 0.0974 | 1.227e-2 | 7.815e-4 |
| RTX PRO 6000 | Accelerated | 100 | 9.796 | 0.0980 | 6.389e-4 | 3.350e-5 |
| RTX 3080 Ti | Fixed | 100 | 32.522 | 0.3252 | 1.227e-2 | 7.815e-4 |
| RTX 3080 Ti | Accelerated | 100 | 32.568 | 0.3257 | 6.389e-4 | 3.350e-5 |

For a matched absolute two-family residual threshold of `1.0`:

| Device | Variant | Iterations | Time [ms] | Stress RMS | Contact RMS |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | Fixed | 464 | 42.984 | 3.275e-3 | 2.051e-4 |
| RTX PRO 6000 | Accelerated | 38 | 3.601 | 3.279e-3 | 1.908e-4 |
| RTX 3080 Ti | Fixed | 464 | 147.612 | 3.275e-3 | 2.051e-4 |
| RTX 3080 Ti | Accelerated | 38 | 12.020 | 3.279e-3 | 1.908e-4 |

This is an 11.9x speedup on the RTX PRO 6000 and 12.3x on the RTX 3080 Ti.
No restart was triggered for this snapshot.

The `rheology_granular_b2p1d_0000.npz` topology contains 277,016 stress blocks,
295,571 subgrid contact blocks, and 5,928 enabled contacts. Its stored
constitutive parameters are non-associated; the benchmark deliberately
replays its operator and contact topology through the associated APGD
projection, so these numbers measure the current prototype rather than the
original physical law.

At 100 iterations, fixed and accelerated costs are 137.926 ms and 136.696 ms,
respectively. For an absolute residual threshold of `25.0`, fixed-step takes
1,399 iterations and 2,006.464 ms, while acceleration takes 81 iterations and
110.588 ms, an 18.1x time-to-threshold improvement.

### Comparison with production solvers

This comparison uses the same warm-started inputs and exact fixed iteration
counts. After each timed solve, a fresh copy of the original problem evaluates
one common associated-law projected-gradient mapping with fixed
\(\rho=0.25\). The reported stress and contact values are its separate
absolute L2 norms, and a solver reaches a threshold only when both values are
below it.

Only the captured iteration graph is timed. Loading, solver preprocessing,
compilation, graph capture, final-state copies, and the common diagnostic
projection are excluded. Each table entry is the median of five fresh solves.
The production paths use unrolled fixed-iteration graphs without their native
periodic convergence check; this slightly favors them. APGD retains the
reductions required by its restart and BB update.

On the associated nodal snow-ball snapshot, the first crossings of the common
`0.05` threshold are:

| Device | Solver | Iterations | Time [ms] | Stress residual | Contact residual |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | Gauss-Seidel | 65 | 6.922 | 0.031017 | 0.049102 |
| RTX PRO 6000 | Gauss-Seidel, reordered | 65 | 6.541 | 0.031017 | 0.049102 |
| RTX PRO 6000 | Gauss-Seidel, batched | 65 | 7.456 | 0.031017 | 0.049102 |
| RTX PRO 6000 | Jacobi | 270 | 10.423 | 0.049181 | 0.011043 |
| RTX PRO 6000 | APGD | 220 | 20.794 | 0.049691 | 0.001146 |
| RTX 3080 Ti | Gauss-Seidel | 65 | 18.394 | 0.031017 | 0.049102 |
| RTX 3080 Ti | Gauss-Seidel, reordered | 65 | 18.157 | 0.031017 | 0.049102 |
| RTX 3080 Ti | Gauss-Seidel, batched | 65 | 17.365 | 0.031017 | 0.049102 |
| RTX 3080 Ti | Jacobi | 270 | 36.763 | 0.049181 | 0.011043 |
| RTX 3080 Ti | APGD | 220 | 69.900 | 0.049691 | 0.001146 |

Fixed PGD does not reach `0.05` within 2,000 iterations: its final stress and
contact residuals are 0.188568 and 0.004410. Its median times are 186.151 ms on
the RTX PRO 6000 and 639.494 ms on the RTX 3080 Ti. APGD is therefore a large
improvement over the fixed projected iteration, but the production solvers
remain faster on this nodal snapshot.

For the subgrid replay, every solver receives an in-memory copy with material
dilatancy set to full association. This preserves the saved operator,
warm-start, and contact topology, but is not a replay of the snapshot's
original non-associated material trajectory.

At the loose common threshold `25.0`, Jacobi wins the early-transient
comparison:

| Device | Solver | Iterations | Time [ms] | Stress residual | Contact residual |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | Jacobi | 55 | 64.632 | 24.9182 | 6.01518 |
| RTX PRO 6000 | APGD | 68 | 92.993 | 24.9354 | 1.12913 |
| RTX 3080 Ti | Jacobi | 55 | 305.035 | 24.9182 | 6.01518 |
| RTX 3080 Ti | APGD | 68 | 495.624 | 24.9354 | 1.12913 |

None of the Gauss-Seidel layouts reaches `25.0` within the snapshot's
250-iteration budget. Their final maximum family residuals on the RTX PRO
6000 are 84.9945 for ordinary Gauss-Seidel, 85.2824 for reordered
Gauss-Seidel, and 35.7518 for batched Gauss-Seidel.

At the tighter common threshold `20.0`, acceleration amortizes its higher
iteration cost and APGD overtakes Jacobi:

| Device | Solver | Iterations | Time [ms] | Stress residual | Contact residual |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | Jacobi | 610 | 741.987 | 19.9888 | 3.90277 |
| RTX PRO 6000 | APGD | 239 | 320.317 | 19.9966 | 0.676995 |
| RTX 3080 Ti | Jacobi | 610 | 3,377.042 | 19.9888 | 3.90277 |
| RTX 3080 Ti | APGD | 239 | 1,724.860 | 19.9966 | 0.676995 |

This is a 2.32x APGD time-to-threshold improvement on the RTX PRO 6000 and
1.96x on the RTX 3080 Ti. APGD also finishes with a 5.8x smaller contact
residual at nearly the same threshold-setting stress residual.

### Non-associated piecewise-linear benchmark

The physical non-associated replay uses the unmodified
`rheology_granular_b2p1d_0000.npz` constitutive data: 277,016 stress blocks,
295,571 subgrid contact blocks, 5,928 enabled contacts, zero dilatancy
(\(\theta=0\)) at every material node, and zero viscosity. This exercises the
piecewise-linear yield surface with the stress-side de Saxcé correction
enabled; it does not convert the material to association.

Every solver starts from the same saved warm start. After each timed solve, a
fresh problem evaluates the same corrected viscoplastic/contact fixed-point
mapping with diagnostic step \(\rho=0.25\). The table reports the first integer
iteration for which both absolute L2 family residuals are below `1.0`.
Compilation, graph capture, data loading, and the common diagnostic are
excluded. Times are medians of three fresh solves on the idle RTX 3080 Ti:

| Solver | Iterations | Time [ms] | Stress residual | Contact residual |
| --- | ---: | ---: | ---: | ---: |
| Gauss-Seidel, batched | 154 | 350.296 | 0.442955 | 0.997064 |
| Gauss-Seidel, reordered | 250 | 1,187.186 | 0.105882 | 0.998780 |
| APGD | 350 | 2,532.140 | 0.981915 | 0.066228 |
| Jacobi | 907 | 5,509.157 | 0.999565 | 0.190606 |
| Gauss-Seidel | 251 | 6,099.327 | 0.105288 | 0.997486 |

Batched Gauss-Seidel is 7.23x faster than APGD at this threshold. APGD is
2.18x faster than Jacobi and 2.41x faster than ordinary Gauss-Seidel, but
2.13x slower than reordered Gauss-Seidel. APGD reaches the threshold with a
contact residual about 15x smaller than the threshold-setting contact
residual of the Gauss-Seidel variants, suggesting that its joint tolerance or
block scaling is currently imbalanced toward contact. CUDA 0 timing was
deferred because an unrelated workload occupied that device throughout this
run.

### Stage 6: non-associated material correction and viscosity

1. [Complete] Correct the normal strain response with
   \((1-\theta)y'\lVert u_{\sigma,T}\rVert\), evaluated on the active branch
   of the shifted yield stress.
2. [Complete] Apply the orthogonal projection in the viscous yield-stress
   variable \(\sigma_\mathrm y=\sigma+(\eta/V)u_\sigma\), then map back to
   total stress through the viscosity-scaled local Delassus resolvent.
3. [Complete] Use the same corrected viscoplastic mapping for the fixed-step
   convergence diagnostic and acceleration restart metric.
4. [Complete] Retain raw, uncorrected operator differences for the BB2
   spectral step, measured in the viscous yield-stress variable.
5. [Complete] Compare a converged non-associated viscous coupled solve with
   the existing Gauss-Seidel path on CPU and CUDA.
6. [Complete] Smoke-test the public granular and viscous examples with
   collider contact on CUDA.
7. [Complete] Run the full viscous-funnel example and compare a
   large-viscosity coupled regression with Gauss-Seidel on CPU and CUDA.
8. [Complete] Preserve the unbounded pressure multiplier with an exact
   cylindrical projection, retain viscous total-stress warm starts, and
   enforce a relative-divergence check in the viscous example.
9. [Complete] Project one-sided unbounded yield sets using exact rays,
   including the default infinite Drucker-Prager cone, rather than finite
   sentinel caps.

The raw coupled operator, exact admissible-set projection, acceleration, and
residual framework remain shared with associated materials. A future
Cam-Clay extension will require a corresponding exact local projector before
it can use this APGD path.

## Validation plan

Use `unittest` throughout and give every test a concise imperative docstring.

Projection tests:

- contact cone interior, boundary, sliding face, polar/apex, zero friction,
  disabled contact, adhesion translation, idempotence, and rotational
  invariance;
- stress interior, tensile/compressive caps, all three roof segments, every
  corner, cohesion, zero friction, zero deviatoric input, idempotence, and
  deviatoric rotational invariance;
- unbounded fluid pressure remains unchanged while deviatoric stress projects
  onto the cylindrical yield set;
- infinite Drucker-Prager and infinite-tension surfaces agree with analytic
  ray projections and remain idempotent for very large sentinels;
- compare projected points against a small NumPy boundary enumeration or dense
  sampled reference and verify feasibility and projection optimality.

Operator tests:

- verify linearity of the matrix-free \(W\) action after subtracting \(q\);
- verify the stress/contact cross-term adjoint relation;
- compare nodal and identity-matrix subgrid contact;
- verify rigid response signs with a one-body impulse.

Solver tests:

- associated material without contact, compared with the converged existing
  solver;
- contact-only frictionless and frictional problems;
- simultaneous yielding and collider sliding;
- adhesion, subgrid contact, and rigid collider feedback;
- feasibility of every returned stress and impulse block;
- decrease of the corrected fixed-point residual, without asserting monotonic
  quadratic energy;
- warm-start and cold-start agreement;
- non-associated and viscous one-step agreement with an independent reference;
- converged moderate- and large-viscosity non-associated agreement with the
  existing Gauss-Seidel solver;
- incompressible viscous agreement with Gauss-Seidel and preservation of
  viscous total-stress warm starts;
- inactive fixed-grid capacity rows do not enter the specialized stencil
  evaluation.

Before committing, demonstrate that the new regression tests fail without
their implementation, then run the focused tests with `uv run --extra dev -m
newton.tests -k ...` and finish with `uvx pre-commit run -a`.

## Current checkpoint

The public opt-in path covers both exact projectors, simultaneous
six-component stress and three-component contact updates, nodal and subgrid
contact, rigid collider feedback, contact- and stress-side de Saxcé
corrections, viscosity, coupled Algorithm B.2 acceleration, guarded
raw-response BB2 updates, device-resident stopping reductions, and feasible
final reconstruction.

The focused suite has 72 passing CPU/CUDA cases. Scene-level tests cover
associated snow, non-associated granular flow with fixed-grid inactive
capacity, the complete viscous funnel, collider contact, and CUDA graph
capture. Dedicated regressions also cover per-environment APGD reductions,
independent-world replay, empty environments, fixed-grid capacity padding,
all finite and unbounded piecewise-linear yield topologies, large-viscosity
and incompressible-fluid agreement with Gauss-Seidel, viscous warm starts,
and multi-world outer graph capture. The viscous funnel now checks relative
particle-velocity divergence rather than finite state alone. Remaining work
is broader regression testing and further scene-level performance/robustness
tuning, including additional accuracy thresholds and cross-device
non-associated timing.
