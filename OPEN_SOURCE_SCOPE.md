# Open-source release scope

## Included

- MeCaSNet and benchmark model implementations.
- Event-level training, validation checkpoint selection, and unified metrics.
- Threshold, uncertainty, propagation-depth, event-similarity, and topology
  audit workflows.
- The expected static-network and event-file schema.
- Research-software packaging, citation metadata, and CPU smoke tests.

## Not included

- Firm identities, transaction records, geographic attributes, or the original
  directed supply network.
- Generated event files or any derived artifact that can reveal protected
  enterprise information.
- ARIO/Henriet or Inoue-Todo simulator implementations and their private input
  pipelines.
- Trained checkpoints, result tables, figures, server paths, credentials, or
  remote execution scripts.
- A public data-download or data-access promise.

## Consequences for reproducibility

The release supports inspection and reuse of the method, training protocol, and
evaluation logic. Exact numerical reproduction of the manuscript requires an
authorized dataset and the corresponding simulator-generated events. Workflows
in manuscript Sections 4.8 and 4.9, together with simulator-dependent topology
generation in Section 4.4, remain reference implementations rather than
standalone public reproductions. Their locations and dependencies are indexed
under `scripts/experiments/main/`.

## Data governance

Do not commit raw or derived files that reveal firm identity, supply relations,
production, location, or recoverable proxies for those quantities. Before
publishing any new dataset or checkpoint, complete the relevant permission,
de-identification, licensing, and ethics review.
