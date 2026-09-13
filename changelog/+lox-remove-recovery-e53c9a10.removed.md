Remove LOX nonlinear rollback snapshots and automatic frozen-model retries to reduce solver storage and work.
Nonfinite feedback now rejects the affected world directly; use the solver failure status to detect rejected steps.
Remove preemptive rod bend and angular-increment rejection thresholds; large finite updates no longer mark a world as failed.
