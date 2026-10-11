# Supplementary Note S7: Robustness and Predictive-Uncertainty Evaluation

This note complements Sections 4.4 and 4.11. Repeated model comparisons use
seed-specific partitions; the frozen-model diagnostics retain the reference
event sets stated for each analysis.

| SI subsection | Scientific question | Analysis location | Tables |
|---|---|---|---|
| S7.1 Random Event-Level Split | New disruption events on a shared static network | [Repeated event partitions](../../main/section_4_03_overall_performance/repeated_split_experiment.py) | Study definition |
| S7.2 Cross-Split Event Similarity Audit | Exact and near-duplicate shock configurations | [Event similarity characterization](../../main/section_4_04_split_and_topology/event_split.py) | S12 |
| S7.3 Sensitivity to Similar-Event Exclusion | Accuracy after excluding similar test events | [Frozen-model exclusion analysis](../../main/section_4_04_split_and_topology/event_similarity.py) | S13 |
| S7.4 Post-Training Robustness to Subgraph Topology Shift | Modified networks with newly simulated targets | [Topology perturbation](../../main/section_4_04_split_and_topology/README.md) | S14-S15 |
| S7.5 Sensitivity to Input-Share Structure and Supplier Relations | Uniform shares, simulator-label changes and missing suppliers | [Input and network sensitivity](../../main/section_4_11_uncertainty/README.md) | S16-S17 |
| S7.6 Recovery-Time Variability and Event-Level Predictive Coverage | Unobserved future dynamics and independent-event coverage | [Recovery and coverage analysis](../../main/section_4_11_uncertainty/README.md) | S18-S19 |

Uniform-share inference retains the original simulator targets. Simulator-side
share changes require a recomputed equilibrium and new trajectories. The
edge-omission control preserves at least one supplier per buyer and redistributes
removed share mass; it differs from the unreallocated-edge analysis in main
Table 11.

The recovery-time study uses 100 fixed Day-0 states and 30 recovery-time draws
per state. Resampling retains all draws associated with a state. The estimated
conditional variance concerns the sampled recovery-time distribution and is
not an estimate of every source of prediction error.

Conformal calibration uses 500 new calibration events and 500 separate test
events, with complete events as the calibration units. Peak and trajectory
intervals are calibrated separately. Simultaneous coverage within each event
retains dependence among nodes and dates; it does not assume independent
node-level residuals or provide joint coverage of both target types.
