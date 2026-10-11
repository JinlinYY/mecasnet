# 4.5 Propagation-Depth and Network-Position Analysis

These analyses compare MeCaSNet and DirGNN across structurally distinct
cascade populations on the 500-event Henriet reference test set.

Cascade distance is the shortest directed supplier-to-customer path from any
initially shocked firm. Table 4 compares one-hop, two-hop and at least three-hop
cascade populations. Nodes must be initially unshocked, downstream reachable
and have a target peak loss greater than 0.05.

Table 5 compares feed-forward downstream nodes with bidirectionally reachable
nodes. A bidirectionally reachable node can be reached from a shocked firm and
can itself reach a shocked firm. These two paths may involve different shocked
firms; the classification does not require membership in a single cycle with
the same shock source.

| Analysis code | Scientific role |
|---|---|
| [propagation_depth.py](propagation_depth.py) | Prediction errors stratified by downstream distance |
| [network_propagation.py](network_propagation.py) | Matched-model distance and network-position comparisons |
| [audit_prediction_scope.py](audit_prediction_scope.py) | Event-firm counts, directed reachability and connected-component classifications |

Report sample counts for each stratum. Paired confidence intervals resample
complete events, retaining dependence among firms within an event. The
network-position comparison describes prediction difficulty associated with
directed reachability; it does not by itself identify a causal feedback effect.
