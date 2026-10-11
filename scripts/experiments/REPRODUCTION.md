# Reproduction of the experiments

The experimental descriptions follow the current manuscript and Supplementary
Information. The [main-section guide](main/README.md) and
[Supplementary Information guide](si/README.md) identify the scientific purpose
of each analysis and its relationship to the reported tables and figures.

## Common study design

The principal Chemical and NEEQ comparisons use ten seed-specific partitions.
Each partition contains 4,000 training, 500 validation and 500 test events; all
models share the same assignments within a repeat. A repeat changes the event
partition and initialization together. Architecture ablations use three matched
partitions. Reference-checkpoint analyses retain their specified event sets.

The primary cascade population is initially unshocked prediction nodes with
full-daily target peak loss greater than 0.05. Predictions use ten key dates.
Model selection uses validation data; test outcomes are reported after selection.
The Chemical and NEEQ models are trained separately.

The model-training command accepts a saved event partition with
`--split-manifest`. Its `--seeds` option changes initialization while retaining
that partition. A ten-initialization call alone therefore does not perform the
ten-partition experiment. Use one training run for each saved partition and the
same assignments for every comparator within that repeat.

## Data and simulation requirements

Reproducing numerical results requires the corresponding static network,
simulation-generated events, trained models and simulation settings. These
materials are described in the [data definitions](../../docs/data_contract.md)
and [data availability statement](../../DATA_AND_SOFTWARE_AVAILABILITY.md).
Topology perturbation and simulator-side input changes require newly computed
network equilibria and event trajectories. Repair results require independent
mechanistic evaluation of the selected repair sets.

The available Python files include complete analyses and selected research
procedures. Files containing selected procedures may not include the complete
experiment command or all companion modules. Their presence identifies the
available calculation, rather than establishing that the full published table
can be regenerated from this repository alone.

The repeated NEEQ and architecture-ablation procedures, temporal diagnostics,
independent-event coverage, and several simulator comparisons use the original
research environment. Depending on the analysis, companion modules include
`pi_dl`, `y8_reviewer_experiments`, `comment12_common`,
`independent_conformal_common`, `evaluate_y8_propagation_depth`, the Henriet
event-generation modules and `fix` simulator/FINDER modules. These companion
modules are not all included in this repository. Several procedures also retain
relative data-location assumptions from that environment, which must be
resolved against the investigator's authorized data and software locations.

## Model definitions and comparison populations

The public `paper` settings and the `compat` analysis settings define different
trajectory models. Evaluate each trained model with its own architecture and
information set. Some supplied analyses explicitly use `compat`; their results
should not be attributed to a newly trained `paper` model without matching its
definition and evaluation data.

Reference full-network populations, five-hop prediction populations and
cascade-node populations are different. Event-firm observations are not unique
firms. State the population and sample count for every reported metric.

## Statistical interpretation

Use matched repeats for neural-model comparisons, complete-event resampling for
paired event analyses, fixed-Day-0-state resampling for recovery-time variation,
and seed-cluster resampling for repair comparisons with shared scenarios.

Distinguish reconstructed-input sensitivity, ensemble variation and calibrated
predictive coverage. The independent coverage experiment calibrates one score
per complete event on 500 newly generated events and evaluates 500 separate
events. Pooled node-level residual analysis answers a different coverage question.

Record the model definition, event assignments, simulation conditions, random
seeds, selected epoch, evaluation population and uncertainty procedure with
scientific results. Computational comparisons additionally identify hardware,
output scope, data-loading cost and final simulator verification.
