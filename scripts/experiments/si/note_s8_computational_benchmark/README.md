# Supplementary Note S8: Computational Benchmark and Break-Even Analysis

This note supplies the detailed accuracy-cost comparisons underlying
[Section 4.9](../../main/section_4_09_efficiency/README.md).

| SI subsection | Comparison | Available analysis | Tables |
|---|---|---|---|
| S8.1 Model Computational Profile | Parameter count, storage, training and training-data generation | [Model training](../../../../src/mecasnet/train.py); [event-generation cost](../../main/section_4_09_efficiency/measure_henriet_generation_cost.py) | S20 |
| S8.2 Low-Cost Mechanistic Accuracy-Cost References | Inventory-buffered Leontief and coarse-step ARIO against the frozen model | [Leontief](../../main/section_4_09_efficiency/evaluate_inventory_leontief_mechanistic_baseline.py); [coarse-step ARIO](../../main/section_4_09_efficiency/evaluate_coarse_ario_mechanistic_baseline.py) | S21 |
| S8.3 Batched Simulator Benchmark | Serial and event-parallel simulation, online prediction and total evaluation time | [Batched comparison](../../main/section_4_09_efficiency/benchmark_henriet_batched_breakeven.py) | S22 |
| S8.4 Upfront Cost and Break-Even Analysis | Training-data generation plus training, amortized over queries or repair tasks | [Cost calculation](../../main/section_4_09_efficiency/calculate_henriet_breakeven.py) | Main Table 8 |
| S8.5 Complete Repair-Task Timing | Candidate screening and final simulator verification | [Matched prediction and repair comparisons](../../main/section_4_09_efficiency/benchmark_henriet_matched_workflow.py) | S23 |

The mechanistic comparison uses the same 500 held-out Chemical events, a common
cascade population and a fixed recovery time of 60 days. Coarse ARIO steps are
2, 3, 5 and 7 days. Accuracy and latency are reported together; computational
advantage over the daily simulator does not imply an advantage over every
simplified approximation.

The batched benchmark selects the CPU worker count on 100 separate tuning
events and evaluates five complete 500-event batches. Serial and parallel
simulation outcomes are compared before interpreting throughput. Startup is
excluded. Preloaded-input inference and timing including data loading, feature
calculation and device transfer are reported separately.

Upfront cost is the sum of training-data generation and one complete model
training run. The break-even count is that cost divided by the simulator-minus-
surrogate cost per query or complete repair task. Report the measured hardware,
output population and timing scope with every estimate.

Complete repair tasks include final simulator verification of the selected
repair sequence. Their cost comparison is considered alongside verified final
loss, rather than in isolation from decision quality.
