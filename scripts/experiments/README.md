# Experimental studies

The experiments examine cascade prediction in the Henriet chemical supply-chain
benchmark and the Inoue-domain NEEQ benchmark. They follow Sections 4.1-4.11 of
the main manuscript and Supplementary Notes S1-S9.

- [Main manuscript](main/README.md): prediction accuracy, robustness, propagation
  structure, ablation, trajectory reconstruction, repair screening, computational
  cost, threshold sensitivity and uncertainty.
- [Supplementary Information](si/README.md): dataset construction, model and
  comparator definitions, temporal diagnostics, robustness and coverage,
  computational comparisons and repair-set evaluation.
- [Reproduction of the experiments](REPRODUCTION.md): data requirements,
  repeated-partition design and availability of the research procedures.

The principal model comparison uses ten seed-specific event partitions, each
containing 4,000 training, 500 validation and 500 test events. All models share
the same partition within a repeat. The architecture ablation uses three matched
partitions. Frozen-model diagnostics use the reference event sets specified in
their respective sections; their sample counts are not pooled with the repeated
model comparison.

The primary cascade population consists of initially unshocked prediction nodes
whose target peak production loss exceeds 0.05. Peak targets are obtained from
the full daily simulation; predicted trajectories are evaluated at Days 0, 5,
10, 20, 30, 50, 70, 100, 150 and 199.

The standard event split evaluates new disruption scenarios on a known supply
network. The Chemical and NEEQ models are trained separately. Topology
perturbation, input sensitivity and predictive coverage address distinct
scientific questions and are reported separately.
