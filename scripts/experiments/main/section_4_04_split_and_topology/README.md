# 4.4 Event-Split Robustness and Topology-Shift Evaluation

Two complementary analyses examine whether cascade prediction is sensitive to
similar disruption configurations or changes in the operating supply network.
The event-similarity analysis retains its reference partition and frozen model.
The topology study evaluates newly simulated events on reduced networks.

## Cross-split event similarity

[event_split.py](event_split.py) characterizes exact and near-duplicate events
using directly shocked-firm sets and disruption characteristics.
[event_similarity.py](event_similarity.py) evaluates the same frozen model after
excluding increasingly similar test events. Report the retained event counts
and cascade metrics for each exclusion criterion.

## Subgraph topology perturbation

Connected node blocks are selected using pre-disruption topology, production and
supply-flow information. After removing a block, network attributes and the
operating equilibrium are recomputed. Each modified network generates 1,000
new disruption events. The original full-network MeCaSNet and DirGNN models are
then evaluated without additional training. Table 3 reports three moderate
perturbations; stronger perturbations are described in Supplementary Note S7.

| Analysis code | Scientific role |
|---|---|
| [create_topology_blocks.py](create_topology_blocks.py), [audit_topology_blocks.py](audit_topology_blocks.py) | Connected-block selection and comparison of retained network structure |
| [search_induced_subgraphs.py](search_induced_subgraphs.py), [sweep_subgraph_deletion.py](sweep_subgraph_deletion.py) | Examination of connected-block removal conditions |
| [build_severe_manifests.py](build_severe_manifests.py), [audit_subgraph_design.py](audit_subgraph_design.py) | Stronger topology perturbations and structural-preservation statistics |
| [evaluate_subgraph.py](evaluate_subgraph.py) | Frozen-model evaluation on newly simulated reduced-network events |

The reduced-network labels must come from new simulations on the modified graph.
The experiment concerns changes within the original network domain; it does not
establish transfer to an entirely unseen industrial network.

See [Supplementary Note S7](../../si/note_s7_robustness_and_uncertainty/README.md)
for Tables S12-S15 and the relationship to the other robustness studies.
