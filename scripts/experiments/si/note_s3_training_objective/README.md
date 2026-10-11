# Supplementary Note S3: Training Objective Parameters

This note specifies the loss coefficients and key-date weights, then examines
temporal regularization and the trough-timing auxiliary prediction.

## Objective coefficients and temporal weights: Tables S1-S2

The coefficients are defined in [config.py](../../../../src/mecasnet/config.py).
[focal_weights and total_loss](../../../../src/mecasnet/losses.py) define the
severity weights and composite objective. The objective combines trajectory
and peak-loss regression, cascade-extent and trough-timing supervision, and
temporal regularization. Target-derived quantities contribute to supervision;
they do not become predictive inputs.

## Temporal regularization and event context: Tables S3-S4

[temporal_regularization_diagnostics.py](temporal_regularization_diagnostics.py)
examines the recovery-rate term, event-context dependence and late-stage
declines. [temporal_regularization_ablation.py](temporal_regularization_ablation.py)
contains matched retraining procedures for temporal-regularization comparisons.
These procedures require the companion research modules described in the
[reproduction guide](../../REPRODUCTION.md).

Late-stage non-monotonicity is evaluated over Days 70-100, 100-150 and 150-199.
A reference trajectory is non-monotone when production decreases by more than
0.001 in at least one interval. Counts distinguish the complete network,
prediction nodes and cascade nodes. The late-monotonicity comparison uses three
matched partitions and changes only the corresponding loss weight.

Under the stated bounded predictions, key-date spacing and recovery-rate bounds,
the recovery-rate loss is inactive. The late-stage term is interpreted as
empirical smoothing: it can reduce spurious declines while reducing sensitivity
to genuine secondary declines.

## Trough timing and ordinal evaluation: Tables S5-S6

[evaluate_trough_timing.py](evaluate_trough_timing.py) evaluates the earliest
production minimum among the ten key dates on initially unshocked cascade nodes
across ten repeated partitions. Report exact and balanced accuracy, timing MAE,
and accuracy within one neighboring key-date class.

The ranked probability score and its time-weighted counterpart assess the
ordered probability distribution. Argmax and posterior-median decoding are
compared on the same cross-entropy-trained probabilities. Posterior-median
results are an additional evaluation; the main model retains argmax decoding.
