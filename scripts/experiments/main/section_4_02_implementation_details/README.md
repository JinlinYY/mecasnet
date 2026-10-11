# 4.2 Implementation Details

The two benchmarks each contain 5,000 independently simulated disruption
events. Every repeated partition assigns 4,000 events to training, 500 to
validation and 500 to testing. A repeat changes both the event partition and
the model initialization. All methods compared within a repeat use identical
event assignments and training conditions.

| Study setting | Specification |
|---|---|
| Overall model comparison | Ten repeated event partitions |
| Architecture ablation | Three matched event partitions |
| Event batch size | One |
| Optimization | AdamW; initial learning rate 0.0003; weight decay 0.00001 |
| Learning-rate schedule | Linear warm-up followed by cosine decay |
| Gradient accumulation | Four events |
| Gradient clipping | Maximum norm 1.0 |
| Model selection | Validation cascade peak-loss R-squared |

The Chemical and NEEQ models are trained separately. Within each benchmark,
event subsets share the same pre-disruption graph. The resulting evaluation
concerns unseen disruption scenarios on a known network.

Training is defined in [train.py](../../../../src/mecasnet/train.py), with
coefficients in [config.py](../../../../src/mecasnet/config.py). The
[repeated-partition procedures](../section_4_03_overall_performance/repeated_split_experiment.py)
provide the event assignments used for matched comparisons. Increasing only the
number of initializations in a single training call retains a common partition;
independent event partitions must be specified for the repeated-split study.

Loss coefficients and temporal weights are described in
[Supplementary Note S3](../../si/note_s3_training_objective/README.md).
