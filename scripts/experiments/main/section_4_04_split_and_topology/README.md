# Section 4.4: event-split and topology-shift robustness

This directory also implements Supplementary Note S7. It contains three related
but distinct stages.

## Event-similarity audit

```bash
python scripts/experiments/main/section_4_04_split_and_topology/event_split.py \
  --data-root /path/to/authorized/data \
  --output-dir runs/main/section_4_4/event-split

python scripts/experiments/main/section_4_04_split_and_topology/event_similarity.py \
  --data-root /path/to/authorized/data \
  --checkpoint /path/to/legacy-y8-seed0.pt \
  --audit-csv runs/main/section_4_4/event-split/event_pair_nearest_neighbors.csv \
  --output-dir runs/main/section_4_4/similarity-sensitivity
```

`event_split.py` uses metadata and labels only to audit similarity; it does not
change model weights. `event_similarity.py` performs a post-hoc frozen-checkpoint
sensitivity analysis.

## Static topology design

| Script | Role |
|---|---|
| `create_topology_blocks.py` | Search connected blocks from exported static metadata |
| `audit_topology_blocks.py` | Fail-fast audit of generated block-constrained event pools |
| `search_induced_subgraphs.py` | Search matched connected removal blocks |
| `sweep_subgraph_deletion.py` | Map the feasible removal-size envelope |
| `build_severe_manifests.py` | Rebuild selected severe removal witnesses |
| `audit_subgraph_design.py` | Audit retained topology and comparability constraints |

The latter four scripts retain manuscript-specific Henriet network-builder
interfaces. They document the selection procedure but require the authorized
backend to execute.

## Frozen evaluation on regenerated subgraphs

After the simulator has regenerated events on a re-equilibrated reduced graph:

```bash
python scripts/experiments/main/section_4_04_split_and_topology/evaluate_subgraph.py \
  --data-root /path/to/regenerated/subgraph-data \
  --checkpoint /path/to/full-graph-checkpoint.pt \
  --model mecasnet \
  --output runs/main/section_4_4/subgraph-b1-mecasnet.json
```

Do not crop full-graph labels to create this input. The reduced-network
simulator must be rerun so targets are consistent with the modified topology.
