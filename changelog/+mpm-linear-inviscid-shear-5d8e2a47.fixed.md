Fix `SolverImplicitMPM` linear warm starts such as `("cg", "gs")` resisting shear in inviscid fluids, which then moved rigidly until the nonlinear solve converged.
