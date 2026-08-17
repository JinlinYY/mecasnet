# Documentation guide

The documentation is organized by task. New users do not need to read every
file before running the code.

## Recommended reading paths

### I want to understand the method

1. [Architecture and tensor flow](architecture.md)
2. [Manuscript and SI mapping](method_to_code.md)
3. [Evaluation protocol](evaluation_protocol.md)

### I want to train the model on my own network

1. [Data contract](data_contract.md)
2. [Training guide](training.md)
3. [Reproducibility guide](reproducibility.md)
4. [Troubleshooting](troubleshooting.md)

### I want to reproduce an analysis

1. [Evaluation protocol](evaluation_protocol.md)
2. [Reproducibility guide](reproducibility.md)
3. The relevant article-aligned entry point under `scripts/experiments/main/`
   or `scripts/experiments/si/`

### I am reviewing the scientific claims

1. [Manuscript and SI mapping](method_to_code.md)
2. [Architecture and tensor flow](architecture.md)
3. [Data contract](data_contract.md), especially the information boundary
4. [Evaluation protocol](evaluation_protocol.md), especially subset definitions

## Document index

| Document | Purpose |
|---|---|
| [architecture.md](architecture.md) | End-to-end model flow, equations, shapes, prediction streams, and output dictionary |
| [data_contract.md](data_contract.md) | On-disk schemas, edge orientation, derived features, event subgraphs, validation, and a minimal example |
| [training.md](training.md) | Installation, paper profile, CLI options, split manifests, baselines, checkpoints, and output files |
| [evaluation_protocol.md](evaluation_protocol.md) | Metric definitions, cascade subset, threshold roles, uncertainty analyses, and topology audits |
| [reproducibility.md](reproducibility.md) | Exact run protocol, provenance checklist, result interpretation, and claim boundaries |
| [method_to_code.md](method_to_code.md) | Section-level mapping from the manuscript and SI to implementation symbols |
| [troubleshooting.md](troubleshooting.md) | Common environment, data, checkpoint, CUDA, and metric problems |

The public-release boundary and data-governance rules are documented separately
in [`OPEN_SOURCE_SCOPE.md`](../OPEN_SOURCE_SCOPE.md).

## Shared notation

| Symbol | Meaning | Typical shape |
|---|---|---|
| `V` | Number of nodes in the static network | scalar |
| `E` | Number of directed supply edges | scalar |
| `Nr` | Nodes in one event's downstream reach subgraph | scalar, varies by event |
| `Er` | Edges induced by the reach subgraph | scalar, varies by event |
| `K` | Number of prediction dates; `K=10` in the paper profile | scalar |
| `Fv` | Static node-feature width | scalar |
| `d` | Hidden width; `d=64` by default | scalar |
| `u[k,v]` | Production divided by business-as-usual production | `(K, Nr)` |
| `p[v]` | Peak production loss, `1 - min_k u[k,v]` | `(Nr,)` |
| `a[e]` | Input share of supplier `src[e]` in customer `dst[e]` | `(Er,)` |

All node indices in event batches are local to the reach subgraph. Use
`reach_idx` to map them back to the static network.
