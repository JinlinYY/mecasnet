# Supplementary Note S2: Cross-Simulator Evaluation on the Inoue-Domain Benchmark

The Inoue-domain NEEQ benchmark examines cascade prediction under a different
network, sector composition and simulator. Chemical and NEEQ models are trained
independently; the experiment concerns the applicability of the model structure
after domain-specific training.

The repeated model comparison uses ten event partitions of 4,000 training,
500 validation and 500 test events. All comparators share the same event
assignments within each repeat. The corresponding training and comparison
procedures are described in [Section 4.3](../../main/section_4_03_overall_performance/README.md).

Figure S2 compares representative predicted and reference damage trajectories
on the same event-specific cascade-node populations. Reference variation across
nodes describes heterogeneity of the simulated response. It does not measure
predictive uncertainty.

The NEEQ data, event characteristics and simulation assumptions are described
in [Supplementary Note S1](../note_s1_neeq_dataset/README.md).
