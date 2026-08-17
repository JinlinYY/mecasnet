# Data and software availability

## Available components

- MeCaSNet and benchmark model implementations.
- Event-level training, validation checkpoint selection, and unified metrics.
- Threshold, uncertainty, propagation-depth, event-similarity, and topology
  audit workflows.
- The expected static-network and event-file schema.
- Research-software packaging, citation metadata, and CPU smoke tests.

## External or restricted assets

- Firm identities, transaction records, geographic attributes, or the original
  directed supply network.
- Generated event files or any derived artifact that can reveal protected
  enterprise information.
- ARIO/Henriet or Inoue-Todo simulator implementations and their private input
  pipelines.
- Trained checkpoints, result tables, or figures derived from restricted data.
- A data-download service or unrestricted data-access entitlement.

## Reproducibility boundary

The repository supports inspection and reuse of the method, training protocol,
and evaluation logic. Exact numerical reproduction of the article requires an
authorized dataset and the corresponding simulator-generated events. Workflows
in manuscript Sections 4.8 and 4.9, together with simulator-dependent topology
generation in Section 4.4, require the external interfaces identified in their
section documentation under `scripts/experiments/main/`.

## Data governance

Raw or derived files that reveal firm identity, supply relations, production,
location, or recoverable proxies are restricted. Distribution of any additional
dataset or checkpoint is subject to permission, de-identification, licensing,
privacy, and ethics requirements.
