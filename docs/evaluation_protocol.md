# Evaluation protocol

This document defines the quantities, subsets, aggregation rules, and analysis
scripts used by the public code. Metric names should not be interpreted without
their subset suffix.

## 1. Prediction target

For node `v` at key date `k`, the target is the production ratio:

```text
u[k,v] = actual production[k,v] / business-as-usual production[v]
```

The model predicts ten values at Days:

```text
[0, 5, 10, 20, 30, 50, 70, 100, 150, 199]
```

Peak node loss is:

```text
p[v] = 1 - min_k u[k,v]
```

The stored `peak_loss_node` may be computed from the full simulator trajectory,
whereas the model peak is computed from predicted key dates. A dataset producer
must document this distinction if between-key-date minima are possible.

## 2. Evaluation population

Each event is reduced to the downstream `reach_hops=5` subgraph constructed by
`CascadeEventDataset`. The metrics in `mecasnet.runner` pool node-event rows
from these event-local subgraphs. Nodes outside the extracted subgraph are not
included in the implemented metrics.

The same static firm can appear in multiple events. Therefore `n_total` counts
node-event observations, not unique firms.

## 3. Subset definitions

| Name | Boolean definition | Purpose |
|---|---|---|
| All evaluated | every node-event row in the reach subgraphs | Overall peak performance |
| Direct shock | `shock_mask > 0.5` | Accuracy on observed damaged firms |
| Primary cascade | `(shock_mask <= 0.5) and (true peak loss > 0.05)` | Propagated loss on non-directly shocked firms |
| Reach auxiliary positive | `true peak loss > 0.001` | Training label for the reachability head |
| Trough auxiliary population | `true peak loss > 0.05` with additional masking in the loss | Training the trough-date head |

The `0.001` threshold is not the headline cascade threshold. It supports a
sensitive auxiliary classification task. The manuscript reporting subset uses
`0.05` unless an explicitly labeled sensitivity analysis says otherwise.

All primary subsets depend on ground-truth loss and are for evaluation only.
They are not available to the predictor at deployment time.

## 4. Metric definitions

For predictions `y_hat` and targets `y`:

```text
R² = 1 - sum((y_hat - y)^2) / (sum((y - mean(y))^2) + 1e-8)
MAE = mean(abs(y_hat - y))
```

The implementation returns `NaN` for R-squared when the subset is empty or the
target standard deviation is below `0.01`. This avoids presenting unstable
R-squared values for nearly constant strata.

| Output key | Population | Quantity |
|---|---|---|
| `r2_pk` | all evaluated rows | Peak-loss R-squared |
| `r2_pk_shk` | directly shocked | Peak-loss R-squared |
| `r2_pk_csc` | primary cascade | Peak-loss R-squared |
| `mae_pk_csc` | primary cascade | Peak-loss MAE |
| `r2_kf_csc[k]` | primary cascade at date `k` | Production-ratio R-squared |
| `r2_kf_csc_mean` | primary cascade | Mean of finite per-date R-squared values |
| `n_csc` | primary cascade | Node-event sample count |
| `n_shk` | directly shocked | Node-event sample count |
| `n_total` | all evaluated rows | Node-event sample count |

`r2_kf_csc_mean` excludes dates whose R-squared is `NaN`; always report the
individual date values alongside the mean so poor or undefined dates are not
hidden.

## 5. Multi-seed aggregation

`agg_seeds` computes the arithmetic mean and sample standard deviation
(`ddof=1`) over finite seed-level metrics. The seed is the unit of replication
for matched model comparisons. Event nodes must not be treated as independent
training replicates for a model-level significance test.

For a fair comparator table:

- use identical event IDs for every model and seed;
- use identical seed labels across models;
- select each model only on validation data;
- compare paired seed-level differences;
- report parameter counts and any capacity-matched variants separately.

## 6. Core evaluation during training

The training command writes per-seed and aggregate validation/test metrics to
`<variant>_summary.json`. This is the supported evaluation path for newly
trained `paper`-profile checkpoints.

```bash
mecasnet-train \
  --profile paper --variant MeCaSNet \
  --data-root /path/to/data \
  --epochs 60 --seeds 5 --save-ckpt \
  --out-dir runs/paper
```

The best epoch is selected by validation `r2_pk_csc`; test values are evaluated
only from the selected state.

## 7. Compatibility-profile utilities

Several post-training scripts use the fixed `compat` architecture. A `paper`
checkpoint is not guaranteed to load into these scripts unless the builder is
updated to match its profile.

Compatibility-profile entry points include:

- `mecasnet-thresholds` (`mecasnet.evaluation`);
- `scripts/experiments/main/section_4_03_overall_performance/threshold_multiseed.py`;
- `scripts/experiments/main/section_4_11_uncertainty/input_uncertainty.py`;
- `scripts/experiments/main/section_4_11_uncertainty/predictive_uncertainty.py`;
- `scripts/experiments/main/section_4_07_trajectory_reconstruction/recovery_parameters.py`;
- `scripts/experiments/main/section_4_05_propagation_depth/propagation_depth.py`;
- `scripts/experiments/main/section_4_04_split_and_topology/event_similarity.py`;
- `scripts/experiments/main/section_4_04_split_and_topology/evaluate_subgraph.py`.

Check the checkpoint `config.profile` and the builder used by a script before
running it. A strict state-dictionary mismatch is evidence of a profile error,
not a reason to disable strict loading.

## 8. Threshold sensitivity and runtime

For a compatibility-profile checkpoint:

```bash
mecasnet-thresholds \
  --data-root /path/to/data \
  --checkpoint /path/to/compat-seed0.pt \
  --output-dir runs/thresholds \
  --thresholds 0.01 0.02 0.03 0.05 0.075 0.10 0.15 0.20 \
  --benchmark-events 100 \
  --device cuda
```

This reports post-hoc cascade-threshold sensitivity and separates cached model
inference timing from end-to-end event evaluation. Runtime reports should state:

- hardware and torch/CUDA versions;
- batch size and event count;
- startup and warm-up policy;
- whether data loading and host-to-device transfer are included;
- checkpoint parameter count and precision.

Do not compare a warmed, cached surrogate forward pass with a simulator run that
includes initialization and file I/O without reporting both scopes.

## 9. Matched-seed model comparison

The multiseed utility expects checkpoint files named from the model and seed
under one root directory:

```bash
python scripts/experiments/main/section_4_03_overall_performance/threshold_multiseed.py \
  --root runs/matched-checkpoints \
  --data-root /path/to/data \
  --output-dir runs/table2-threshold-005 \
  --models MLP,GCN,GAT,STGNN,DirGNN,MeCaSNet \
  --threshold 0.05 --n-test 500 --device cuda
```

It reports mean/standard deviation and paired tests against MeCaSNet. SciPy is
required for p-values; without SciPy, descriptive paired differences are still
available but the p-value is `null`.

## 10. Predictive uncertainty

```bash
python scripts/experiments/main/section_4_11_uncertainty/predictive_uncertainty.py \
  --data-root /path/to/data \
  --checkpoints /path/to/seed0.pt /path/to/seed1.pt /path/to/seed2.pt \
                /path/to/seed3.pt /path/to/seed4.pt \
  --output-dir runs/predictive-uncertainty \
  --n-val 500 --n-test 500 --alpha 0.10 --device cuda
```

Validation events calibrate split-conformal radii; test events are used once to
measure coverage and width. The report distinguishes:

- a seed-0 split-conformal interval;
- an ensemble-mean split-conformal interval;
- an ensemble-scaled split-conformal interval;
- raw central ensemble spread.

Raw spread from five models is descriptive and is not automatically a
calibrated 90% predictive interval. Report marginal coverage, mean width, and
the exact calibration population.

## 11. Input-reconstruction uncertainty

```bash
python scripts/experiments/main/section_4_11_uncertainty/input_uncertainty.py \
  --data-root /path/to/data \
  --checkpoint /path/to/compat-seed0.pt \
  --output-dir runs/input-uncertainty \
  --n-test 500 --replicates 10 --bootstrap 2000 --device cuda
```

The script perturbs deployment inputs while keeping labels and model weights
fixed. Built-in scenarios include:

- log-normal noise in input shares, re-normalized within each customer;
- 5%, 10%, or 20% missing observed edges;
- additive Day-0 damage noise;
- log-normal business-as-usual capacity noise with dependent features updated;
- a joint plausible perturbation.

Intervals from this workflow quantify sensitivity to reconstructed inputs. They
are not general predictive intervals and do not include simulator-label or
model-training uncertainty.

## 12. Propagation-depth analysis

```bash
python scripts/experiments/main/section_4_05_propagation_depth/propagation_depth.py \
  --data-root /path/to/data \
  --checkpoint /path/to/compat-seed0.pt \
  --output runs/propagation-depth.json \
  --threshold 0.05 --n-test 500 --device cuda
```

The report stratifies cascade nodes by undirected hop encoding and by directed
distance from/to the directly shocked set. The matched multiseed workflow in
`network_propagation.py` further distinguishes feed-forward downstream
positions from feedback-connected positions.

Distance-stratified results describe where errors occur. They do not identify a
causal transmission mechanism by themselves.

## 13. Event-similarity audit

First generate the split/similarity audit:

```bash
python scripts/experiments/main/section_4_04_split_and_topology/event_split.py \
  --data-root /path/to/data \
  --output-dir runs/event-split-audit \
  --seed 0 --train-frac 0.8 --val-frac 0.1
```

Then evaluate a frozen checkpoint after excluding flagged test events:

```bash
python scripts/experiments/main/section_4_04_split_and_topology/event_similarity.py \
  --data-root /path/to/data \
  --checkpoint /path/to/compat-seed0.pt \
  --audit-csv runs/event-split-audit/event_pair_nearest_neighbors.csv \
  --output-dir runs/event-similarity-sensitivity \
  --threshold 0.05 --device cuda
```

This is a post-hoc sensitivity analysis. Excluding similar events after looking
at results is not equivalent to a preregistered alternative split.

## 14. Topology-shift analyses

The topology tools fall into two categories:

1. generic audit/search logic for node-removal blocks and retention constraints;
2. network-specific workflows containing fixed feature counts, simulator
   interfaces, or compatibility-profile checkpoint assumptions.

`scripts/experiments/main/section_4_04_split_and_topology/evaluate_subgraph.py`,
for example, expects the compatibility-profile feature contract and a
re-equilibrated induced-subgraph data directory. It
should not be pointed at an arbitrary new network without reviewing and adapting
those checks.

Reduced-graph evaluation changes a known graph. It does not establish fully
node-disjoint inductive transfer unless training and testing have disjoint node
identities by construction.

## 15. Reporting checklist

Every reported metric should identify:

- profile and checkpoint commit;
- event IDs and static-network hash;
- subset definition and threshold;
- whether counts are nodes, node-events, events, or seeds;
- aggregation unit and standard-deviation convention;
- validation selection rule;
- all undefined/`NaN` strata;
- hardware and timing scope for runtime claims;
- whether the analysis is primary, preregistered, or post-hoc.
