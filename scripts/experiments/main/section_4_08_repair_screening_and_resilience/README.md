# 4.8 Repair-Oriented Screening and Resilience Analysis

This section evaluates the selection of firms for repair under a fixed-count
intervention budget. Repair is represented as an instantaneous reduction in
Day-0 damage. Selected repair sets are evaluated with the full cascade simulator.

Three protocols answer different questions and retain their own scenario sets
and intervention strengths.

| Protocol | Design | Scientific purpose |
|---|---|---|
| Simulator-driven reference | Five seeds, four damage levels, 20 candidates, budget 10; residual-damage factor 0.1 | Compare FINDER with fixed heuristics and illustrate Figure 6 |
| MeCaSNet screening | 48 scenarios from eight seeds, three shock-set strategies and two damage settings; budgets 2, 6 and 10; residual-damage factor 0.5 | Compare frozen-model greedy and single-swap screening with fixed heuristics across 144 paired conditions |
| Exhaustive small-set reference | 15 conditions; five candidates, budget three; residual-damage factor 0.1 | Compare greedy selection with all ten feasible repair sets |

| Analysis code | Scientific role |
|---|---|
| [repair_experiment.py](repair_experiment.py), [run_surrogate_repair_experiment.py](run_surrogate_repair_experiment.py) | Surrogate and simulator-based candidate comparisons and verified outcomes |
| [run_repair_diversity_experiment.py](run_repair_diversity_experiment.py) | Diverse shock scenarios, fixed heuristics, greedy selection and single-swap refinement |
| [run_repair_oracle_experiment.py](run_repair_oracle_experiment.py) | Exhaustive finite-set optimality comparisons |

FINDER selects repairs using simulator feedback, independently of MeCaSNet.
In the surrogate-screening study, MeCaSNet remains frozen and the full 200-day
simulator verifies the selected sets. Confidence intervals for the diverse
scenario study resample seed clusters, preserving dependence among budgets
applied to the same scenario.

The formulation assumes equal repair cost and instantaneous intervention. A
greedy selection order identifies a repair set; it is not a time-dependent
repair schedule. The exhaustive comparison supplies a finite-problem reference,
not a general optimality guarantee.

Complete protocols and Tables S24-S26 are described in
[Supplementary Note S9](../../si/note_s9_repair_screening/README.md).
