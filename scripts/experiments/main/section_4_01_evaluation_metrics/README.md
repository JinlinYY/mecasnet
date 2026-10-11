# 4.1 Evaluation Metrics

This section defines the populations and quantities used throughout the study.
The evaluated population is the event-specific five-hop downstream prediction
subgraph. Counts represent event-firm observations; the same firm may occur in
more than one disruption event.

| Quantity | Definition |
|---|---|
| Overall peak-loss R-squared | Peak-loss agreement over all prediction nodes |
| Cascade peak-loss MAE and R-squared | Peak-loss agreement for initially unshocked nodes with target peak loss greater than 0.05 |
| Cascade trajectory R-squared | Equal-weight mean of date-specific R-squared values over valid key dates |
| Valid key date | Pooled target standard deviation of at least 0.01 |

The target peak loss is calculated from the complete daily simulator trajectory.
The predicted peak is calculated from the ten predicted key-date values. This
sampling distinction is retained when interpreting peak-loss error.

The metric calculations are in [runner.py](../../../../src/mecasnet/runner.py)
and [evaluation.py](../../../../src/mecasnet/evaluation.py). Detailed definitions
and aggregation rules appear in the [evaluation protocol](../../../../docs/evaluation_protocol.md).
