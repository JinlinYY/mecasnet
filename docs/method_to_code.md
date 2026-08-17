# Manuscript and SI mapping

This document maps the scientific architecture to the public code. It also
distinguishes the final manuscript profile from archived experimental variants.

| Manuscript component | Public implementation |
|---|---|
| Directed weighted supply graph and static firm features | `src/mecasnet/data.py::StaticNetwork` |
| Day-0 shock mask, damage, and event descriptor | `src/mecasnet/data.py::CascadeEventDataset` |
| Shock-conditioned firm encoder | `src/mecasnet/model_v3.py::NodeEncoderV3` |
| Persistent relation-state updates | `src/mecasnet/model_v3_decoder_ac.py::EdgeStateGATBlock` |
| Forward supply and reverse demand messages | `EdgeStateGATBlock` and `KSGATv3` spatial updates |
| Three-component analytical trajectory | `_reconstruct_trajectory_trimodal` in `model_v3.py` |
| Recurrent key-date rollout | `KSGATStep` and the rollout branch in `KSGATv3.forward` |
| Direct key-date readout and three-stream fusion | `kf_direct_head` and `blend_logits` in `KSGATv3` |
| Cascade-extent and trough-timing heads | `reach_head` and `trough_day_head` in `KSGATv3` |
| Composite objective | `src/mecasnet/losses.py::total_loss` |
| Unified cascade metrics | `src/mecasnet/runner.py::evaluate` |
| Manuscript profile | `src/mecasnet/factory.py::apply_profile` and `build_mecasnet` |

## Forward-pass execution trace

The following trace is useful when reviewing or modifying the implementation:

| Order | Operation | Main input | Main output | Source |
|---:|---|---|---|---|
| 1 | Load/validate static graph | `static_meta.pkl` | `StaticNetwork` | `data.py::StaticNetwork.__init__` |
| 2 | Load one event | `event_XXXXXX.npz` | global event arrays | `data.py::CascadeEventDataset.__getitem__` |
| 3 | Extract downstream reach | shock set, static graph | `reach_idx` | `data.py::StaticNetwork.shock_reach` |
| 4 | Induce local graph/features | global arrays, `reach_idx` | model batch | `CascadeEventDataset.__getitem__` |
| 5 | Encode node condition | `x_v`, shock, `delta0`, event scalar | `h` | `model_v3.py::NodeEncoderV3` |
| 6 | Set Day-0 state | shock, `delta0` | `u0` | `KSGATv3._predict_u0` |
| 7 | Propagate persistent relations | `h`, `u0`, local edges | updated `h`, cached edge states | `model_v3_decoder_ac.py::EdgeStateGATBlock` |
| 8 | Predict structured parameters | updated `h` | 13 trimodal parameters | `KSGATv3.param_head` |
| 9 | Reconstruct lower envelope | parameters, key dates | `u_struct` | `_reconstruct_trajectory_trimodal` |
| 10 | Recurrently roll between dates | `h`, `u0`, date gaps, graph | `u_rollout` | `model_v3.py::KSGATStep` |
| 11 | Predict direct dates | updated `h` | `u_direct` | `KSGATv3.kf_direct_head` |
| 12 | Fuse three streams | three trajectories | `u_fused` | `KSGATv3.forward` |
| 13 | Apply late correction | `h_final`, `u_fused` | `u_keyframes` | `model_v3_residual.py::KSGATv3Residual.forward` |
| 14 | Compute heads/diagnostics | final trajectory/latent | peak, reach, trough, parameters | `KSGATv3.forward` |
| 15 | Compute objective | outputs and targets | loss dictionary | `losses.py::total_loss` |
| 16 | Compute pooled metrics | outputs and event labels | metric dictionary | `runner.py::evaluate` |

## Main manuscript

### Dataset construction

The public loader implements the post-simulation data contract. The private
network reconstruction and cascade simulators are not included. Static features
are computed from the pre-disruption graph and business-as-usual production:
sector encoding, normalized log in/out degree, normalized log production,
reverse PageRank, and largest-SCC membership.

The event loader does not reconstruct firms, infer edges, equilibrate flows, or
run ARIO/Inoue-Todo dynamics. Those steps precede the public schema. The public
code begins with a fixed graph and simulator- or observation-derived event
targets.

### Predictive information set

The `paper` profile uses only the graph, static firm features, direct-shock mask,
Day-0 damage vector, mean Day-0 shock damage, and prescribed key dates. Recovery
time, shock mode, severity tier, later simulator states, and target-derived
quantities do not enter the forward predictor.

The fixed-width event vector can represent legacy metadata, but the paper
profile zeros every slot except the mean observed Day-0 damage at index 2. This
masking happens before tensor construction. See
[data_contract.md](data_contract.md#information-boundary) for the exact vector.

### Architecture

The public `MeCaSNet` builder composes the edge-state backbone with the
three-component decoder and the rollout/direct fusion paths. Day 0 is fixed to
`1 - shock * damage`. Auxiliary labels affect optimization only and are not fed
back into trajectory generation.

The paper builder additionally fixes:

- hidden width 64 from `Config`;
- four persistent edge-state propagation blocks;
- a trimodal decoder with early/mid/late parameter ranges;
- super-Gaussian decline exponent `c=6`;
- node-specific recovery exponent `q_v in [0.5,2.0]`;
- lower-envelope temperature `tau=0.02`;
- initial three-stream fusion logits `[2,0,0]`;
- no legacy Day-0 demand pullback, free residual, per-node seam temperature,
  MoE router, or graph-coupled recovery override.

The late zero-initialized correction inherited from `KSGATv3Residual` is part of
the public `MeCaSNet` class lineage. It is gated off at Days 0 and 5 and cannot
change the deterministic Day-0 boundary.

### Experiments

`mecasnet.train` implements the fixed event split, matched optimization budget,
validation-based checkpoint selection, multi-seed aggregation, and cascade
metrics. Baselines use the same event inputs and supervision; the graph-free MLP
intentionally omits topology and edge shares.

The default split is event-disjoint but graph-transductive. The optional strict
test pool must share the exact same static graph; it provides a locked event
holdout, not unseen-network transfer. Induced-subgraph scripts implement a
different topology-perturbation question.

## Supplementary Information

### S1–S2: network and event generation

The public architecture can be trained independently on any network exported to
the documented schema. NEEQ reconstruction, protected firm records, and the
private simulator-generation pipelines are outside this repository. The public
boundary is explicit in `OPEN_SOURCE_SCOPE.md`.

### S3: objective and optimization

`Config` stores the paper loss coefficients, key-date weights, optimizer
hyperparameters, gradient accumulation, and clipping. `losses.py` implements
trajectory/peak fitting, focal node weighting, recovery and late monotonicity,
reachability, and trough timing. `runner.py::train_with_val` implements AdamW,
warm-up/cosine scheduling, epoch-level validation, and best-state selection.

### S4: comparators

`train.py` constructs all comparators under one data/split/selection loop.
`model_v2.py` contains MLP, GCN, and GAT baselines;
`model_baselines_strong.py` contains DirGNN, STGNN, and the analytical physics
comparator. Deep/capacity variants are labeled separately in the CLI.

### S5: structured dynamics

`_collapse_shape` implements the buffered super-Gaussian decline front.
`_recovery_shape` implements the bounded recovery family and the paper's learned
`q_v`. `_reconstruct_trajectory` builds a single decline–trough–recovery curve;
`_reconstruct_trajectory_trimodal` composes three curves by soft lower envelope.

### S6: training/inference separation

`CascadeEventDataset` constructs targets and auxiliary labels, but
`NodeEncoderV3` receives only `x_v`, shock mask, `delta0`, and the masked event
descriptor. The forward signature does not consume true trajectories, peak
labels, reach labels, or trough labels. `total_loss` is the first place where
predictions and targets meet.

### S7: robustness and audits

- input reconstruction: `scripts/evaluation/input_uncertainty.py`;
- ensemble/conformal uncertainty: `predictive_uncertainty.py`;
- propagation distance and feedback position: `propagation_depth.py` and
  `network_propagation.py`;
- split similarity: `scripts/audits/event_split.py` and `event_similarity.py`;
- topology blocks and induced subgraphs: `scripts/audits/` and
  `scripts/topology/`.

Many of these scripts reproduce archived Y8-H analyses and therefore construct
`legacy-y8`. This checkpoint lineage is stated explicitly in
[evaluation_protocol.md](evaluation_protocol.md#7-archived-checkpoint-utilities).

## Training-only versus inference-time quantities

| Quantity | Encoder input | Used in loss/evaluation | Available at deployment |
|---|---:|---:|---:|
| Static graph and pre-event attributes | yes | yes | required |
| Direct-shock mask | yes | yes | required |
| Node Day-0 damage | yes | yes | required |
| Mean observed direct damage | yes | yes | derived at Day 0 |
| Key dates | controls output dates | aligns targets | required |
| True production trajectory | no | yes | no |
| True peak loss | no | yes | no |
| Reach/cascade label | no | yes | no |
| True trough date | no | yes | no |
| Simulator recovery parameter | no | audit/legacy modes only | no |
| Shock sampling mode or severity tier | no | optional stratification only | not required |

## Outputs that require careful interpretation

- `u_full` is an alias of ten key-date predictions, not a 200-day daily path.
- `params` contains compatibility/regularization fields and should not be
  treated as a complete identified simulator parameter vector.
- learned curve parameters describe the structured prediction branch; the final
  trajectory also includes learned fusion and a late correction.
- fusion weights are global per key date, not per-node causal attributions.
- reach and trough heads are auxiliary predictive tasks, not observed physical
  mechanisms.

## Historical names

Internal class names such as `KSGATv3`, `KSGATv3EdgeState`, and `Y8` are retained
for checkpoint compatibility. New users should construct the model through
`mecasnet.build_mecasnet` and select an explicit named profile.

When citing or discussing the public implementation, use `MeCaSNet` for the
paper-profile factory output and reserve internal names for exact checkpoint
lineage or ablation descriptions.
