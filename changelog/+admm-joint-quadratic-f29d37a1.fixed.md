Implement rho-weighted ADMM joint penalties with recentered numerical bodies
without changing physical centers of mass. Solve physical and numerical bodies
together once per local update; remove the inner solver and joint-curvature
mode options. Retain the optional proxy enable and destination-entry selection
controls, with numerical bodies on both endpoints by default. Unselected entries
use the linearized local update. Keep gamma independently optional. Recenter through the coupling
input-state notification without resetting physical joint or contact history,
including VBD entries with owned joints, shapes, and particles. Check physical
fixed-point replay and parameter invariance. Make MuJoCo inertia-only refresh
usable inside CUDA graphs, and keep VBD joint-frame refresh on the device to
avoid captured writes into freed host memory.
Select numerical-body preparation through a reduced-coordinate capability,
without backend type checks or private backend access in the coupler. Keep
backend validation in the solver, and refresh Featherstone's spatial inertia
and COM caches when inertial properties change. Document the implemented
update order, state persistence, and coordinate-dependent preparation.

Correct gamma compensation on entries without proxies: retain physical rigid
velocities and apply matching target-momentum forces. Evaluate reduced-coordinate
targets at the initial configuration and cancel the full added inertial bias,
including for numerical proxies. Cover spinning anisotropic bodies and moving
articulations with gamma fixed-point regressions.

Give receiving body proxies synthesized FREE roots, including proxies whose
source is constrained by a world joint. Synchronize body-only input notifications
into MuJoCo and Featherstone joint state while preserving explicitly supplied
joint components and native solver history. Honor MuJoCo notifications when
automatic state uploads are disabled or infrequent.
Allocate default controls from the compact view's joint layout and retain
continuous joint angles when synchronizing body poses.
