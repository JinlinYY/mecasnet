# 4.9 Computational Efficiency and Practical Value

The computational study considers prediction accuracy together with marginal
prediction cost, training-data generation, model training and complete repair
tasks. Tables 7-9 summarize online throughput, cost amortization and repair-task
cost; Supplementary Note S8 provides the detailed comparisons.

| Comparison | Analysis code | Evaluation design |
|---|---|---|
| Low-cost mechanistic references | [evaluate_coarse_ario_mechanistic_baseline.py](evaluate_coarse_ario_mechanistic_baseline.py), [evaluate_inventory_leontief_mechanistic_baseline.py](evaluate_inventory_leontief_mechanistic_baseline.py) | Same 500 held-out events and frozen MeCaSNet reference; fixed 60-day recovery for both mechanistic families; ARIO steps of 2, 3, 5 and 7 days |
| Batched simulator throughput | [benchmark_henriet_batched_breakeven.py](benchmark_henriet_batched_breakeven.py) | Separate 100-event worker-count selection; 500 evaluation events; median of five batches; matched serial and parallel simulations |
| Data-generation cost | [measure_henriet_generation_cost.py](measure_henriet_generation_cost.py) | Measured event-generation cost used to estimate the cost of 5,000 training events |
| Complete prediction and repair comparisons | [benchmark_henriet_matched_workflow.py](benchmark_henriet_matched_workflow.py) | Prediction costs and candidate-screening costs including final simulator verification |
| Cost amortization | [calculate_henriet_breakeven.py](calculate_henriet_breakeven.py) | Upfront cost divided by the measured per-query or per-task saving |
| Additional domain-specific timing | [runtime_benchmark.py](runtime_benchmark.py) | Startup, warm-up, prediction and simulation components for Chemical and Inoue settings |

Report hardware, event count, output scope and the operations included in every
time measurement. Preloaded-input inference and evaluation including data
loading, feature calculation and device transfer are distinct quantities. The
full simulator produces daily full-network trajectories; the surrogate predicts
key-date trajectories on the prediction subgraph.

The inventory-buffered Leontief approximation and coarse-step ARIO references
belong to this accuracy-cost comparison rather than the repeated neural-model
comparison in Table 2. Their lower costs or lower accuracies must be interpreted
together. The cost-amortization estimate is conditional on the measured study
settings and requires a positive simulator-minus-surrogate cost difference.

See [Supplementary Note S8](../../si/note_s8_computational_benchmark/README.md)
for Tables S20-S23 and the definition of the upfront cost.
