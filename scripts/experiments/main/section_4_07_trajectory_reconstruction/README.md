# 4.7 Damage-Trajectory Reconstruction in the Henriet Chemical Domain

Figures 4 and 5 examine the temporal evolution and spatial distribution of
cascade damage. Representative trajectories cover medium, heavy and
catastrophic disruptions under sector, hub and random shock modes.

For each event, both prediction and reference are evaluated over the same
cascade-node set and the ten prescribed key dates. Mean damage is the arithmetic
mean of one minus the production ratio. The shaded reference bands in Figure 4
represent cross-node heterogeneity in simulated damage, rather than predictive
coverage. Straight segments between key dates serve only to display the sampled
trajectory.

Figure 5 compares the cascade footprint at Days 20, 30, 100 and 199. Its mean
damage uses the event-specific cascade-node population. The full-network mean
shown for the repair examples in Figure 6 uses a different population.

[recovery_parameters.py](recovery_parameters.py) provides a complementary
analysis of learned structured-trajectory parameters. Those parameters describe
one prediction branch; final predictions also incorporate the recurrent and
direct branches. They are not uniquely identified simulator constants.

The available evaluation code produces key-date predictions. Recreating the
published figures additionally requires the corresponding event selection,
network visualization and reference data. Trough-timing accuracy is described
in [Supplementary Note S3](../../si/note_s3_training_objective/README.md).
