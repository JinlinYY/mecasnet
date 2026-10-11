# Supplementary Note S9: Repair-Screening Evaluation

This note complements [Section 4.8](../../main/section_4_08_repair_screening_and_resilience/README.md).
The decision variable is a subset of candidate firms under an equal-cost,
fixed-count repair budget. All final repair outcomes are evaluated with the
full cascade simulator.

| SI subsection | Protocol and comparison | Analysis location | Tables |
|---|---|---|---|
| S9.1 Simulator-Driven Reference and Fixed Heuristics | Five seeds and four uniform damage levels; 20 candidates, budget 10; residual-damage factor 0.1 | [Repair reference and verified comparisons](../../main/section_4_08_repair_screening_and_resilience/repair_experiment.py) | S24 |
| S9.2 MeCaSNet-Based Repair Screening | 48 scenarios and budgets 2, 6 and 10; residual-damage factor 0.5; fixed heuristics, greedy and single-swap selection | [Diverse repair conditions](../../main/section_4_08_repair_screening_and_resilience/run_repair_diversity_experiment.py) | S25 |
| S9.3 Small-Set Exhaustive Optimality Reference | Five candidates, budget three, 15 conditions and all ten feasible sets; residual-damage factor 0.1 | [Exhaustive comparison](../../main/section_4_08_repair_screening_and_resilience/run_repair_oracle_experiment.py) | S26 |
| S9.4 Scope of the Repair Formulation | Instantaneous Day-0 intervention under equal unit cost | [Scientific assumptions](../../main/section_4_08_repair_screening_and_resilience/README.md) | Interpretation |

The FINDER reference uses simulator feedback and does not use MeCaSNet for
action selection. The separate 144-condition screening study keeps MeCaSNet
frozen, evaluates heterogeneous scenarios and verifies the selected repair sets
with the full 200-day simulator. Its uncertainty estimates resample seed
clusters because several budgets share the same underlying scenario.

The exhaustive experiment distinguishes the set minimizing surrogate loss from
the set minimizing simulator loss. It measures the finite-set gap of greedy
selection without assuming that the two objectives have identical optima.

The three protocols differ in scenario composition, repair strength and budget.
Their results are not pooled. Repair duration, heterogeneous cost, crew
availability and time-dependent resource constraints are outside the stated
formulation. Greedy ordering describes subset selection rather than an
operational recovery schedule.
