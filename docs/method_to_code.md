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

## Main manuscript

### Dataset construction

The public loader implements the post-simulation data contract. The private
network reconstruction and cascade simulators are not included. Static features
are computed from the pre-disruption graph and business-as-usual production:
sector encoding, normalized log in/out degree, normalized log production,
reverse PageRank, and largest-SCC membership.

### Predictive information set

The `paper` profile uses only the graph, static firm features, direct-shock mask,
Day-0 damage vector, mean Day-0 shock damage, and prescribed key dates. Recovery
time, shock mode, severity tier, later simulator states, and target-derived
quantities do not enter the forward predictor.

### Architecture

The public `MeCaSNet` builder composes the edge-state backbone with the
three-component decoder and the rollout/direct fusion paths. Day 0 is fixed to
`1 - shock * damage`. Auxiliary labels affect optimization only and are not fed
back into trajectory generation.

### Experiments

`mecasnet.train` implements the fixed event split, matched optimization budget,
validation-based checkpoint selection, multi-seed aggregation, and cascade
metrics. Baselines use the same event inputs and supervision; the graph-free MLP
intentionally omits topology and edge shares.

## Supplementary Information

- S1–S2: NEEQ data generation is private; the public architecture can be trained
  separately on a compatible NEEQ-format dataset.
- S3: coefficients and key-date weights are defaults in `Config` and
  `losses.py`.
- S4: baseline capacity and training comparisons are built in `train.py` and
  `model_baselines_strong.py`.
- S5: lower-envelope, buffered decline, and bounded recovery are implemented in
  `model_v3.py`.
- S6: training/inference separation is enforced by `data.py`, `factory.py`, and
  `losses.py`.
- S7: event similarity and topology shift are implemented in `scripts/audits/`
  and `scripts/topology/`.

## Historical names

Internal class names such as `KSGATv3`, `KSGATv3EdgeState`, and `Y8` are retained
for checkpoint compatibility. New users should construct the model through
`mecasnet.build_mecasnet` and select an explicit named profile.

