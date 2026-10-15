Add cloth and tetrahedral deformable bodies to the experimental LOX backend of
`SolverKamino`, coupled with the rigid bodies through their soft contacts, with
optional self-contact, penetration-free self-contact steps, and cuDSS candidate
solves through `config.lox.deformable_*`. The candidate solves default to an
adaptive FSAI preconditioner, with an optional coarse-corrected warm start.
