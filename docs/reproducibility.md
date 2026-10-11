# Reproducing the experimental studies

The current manuscript uses ten seed-specific event partitions for the overall
model comparison and three matched partitions for architecture ablations.
The [experimental methods guide](../scripts/experiments/REPRODUCTION.md)
describes data requirements, available research procedures and the distinct
populations used in frozen-model studies.

## Repeated event partitions

Each repeat assigns 4,000 events to training, 500 to validation and 500 to
testing. All models share the same assignments within that repeat. Model
initialization and the event partition vary together across repeats. Chemical
and NEEQ models are trained separately.

For an authorized 5,000-event dataset, the supplied partition procedure can
create the ten assignments:

```python
from argparse import Namespace
from scripts.experiments.main.section_4_03_overall_performance.repeated_split_experiment import write_manifests

write_manifests(Namespace(
    data_root="/path/to/authorized/data",
    output_dir="runs/event_partitions",
    seeds="0,1,2,3,4,5,6,7,8,9",
    train_fraction=0.8,
    validation_fraction=0.1,
))
```

Train every comparator on each saved partition. The following example uses the
public manuscript model settings; the original numerical comparison additionally
requires its corresponding research data and model definitions.

```bash
for repeat in 0 1 2 3 4 5 6 7 8 9; do
  for model in MLP GCN GAT STGNN DirGNN MeCaSNet; do
    mecasnet-train \
      --profile paper --variant "$model" \
      --data-root /path/to/authorized/data \
      --split-manifest "runs/event_partitions/repeat_0${repeat}.json" \
      --n-train 4000 --n-val 500 --n-test 500 \
      --epochs 60 --warmup-epochs 3 --lr-schedule cosine \
      --seeds 1 --seed-start "$repeat" --save-ckpt \
      --out-dir "runs/chemical/repeat_${repeat}/${model}"
  done
done
```

A training call with multiple `--seeds` retains one event partition and varies
initialization. Specifying one saved partition per repeat is necessary for the
repeated-partition design. The original architecture-ablation procedures are
described in [Section 4.6](../scripts/experiments/main/section_4_06_ablation/README.md).

## Model selection and evaluation

Select each model using validation cascade peak-loss R-squared. Evaluate the
selected model on its corresponding test set. Report the mean and sample
standard deviation across repeats, retaining the pairing of event assignments
for model comparisons.

Record the scientific model definition, event assignments, simulation settings,
random seeds, selected epoch, evaluation population and metric definitions.
The `paper` and `compat` settings describe different trajectory models; analyses
must use the definition associated with their trained model.

## Complementary analyses

Reference-checkpoint diagnostics retain their stated event sets. Their results
are not merged with the ten-repeat comparison. New reduced-network simulations
are required for topology perturbation; full simulation is retained for final
repair-set verification.

Input reconstruction sensitivity, ensemble variation and independent event-level
coverage are distinct analyses. The independent coverage experiment uses 500
new calibration events and 500 separate test events, with one maximum-residual
score per complete event. Paired event analyses retain within-event dependence;
repair comparisons retain dependence among budgets sharing a disruption scenario.

[Section 4.9](../scripts/experiments/main/section_4_09_efficiency/README.md)
and [Supplementary Note S8](../scripts/experiments/si/note_s8_computational_benchmark/README.md)
define the computational comparison. Report hardware, output scope, data-loading
cost and final simulator verification alongside prediction accuracy and runtime.

The available data and simulator conditions are described in the
[data availability statement](../DATA_AND_SOFTWARE_AVAILABILITY.md). Exact
numerical reproduction requires the corresponding research data, models and
companion simulation software.
