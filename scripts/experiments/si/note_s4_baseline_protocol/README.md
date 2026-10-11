# Supplementary Note S4: Baseline Comparability and Training Protocol

Tables S7-S8 describe model capacities, available inputs, output heads and
supervision for MLP, GCN, GAT, STGNN, DirGNN and MeCaSNet.

All methods compared within a repeat use the same event partition, key dates,
training budget and validation-based selection rule. Graph comparators receive
the common graph, input-share and Day-0 information. MLP provides the graph-free
comparison. Any additional capacity comparison is identified separately from
the principal baseline definition.

The comparator definitions are in [model_v2.py](../../../../src/mecasnet/model_v2.py)
and [model_baselines_strong.py](../../../../src/mecasnet/model_baselines_strong.py).
[train.py](../../../../src/mecasnet/train.py) contains the common training and
model-selection procedure. The ten-repeat benchmark is described in
[Section 4.3](../../main/section_4_03_overall_performance/README.md).

Each model is selected on its own validation performance before test evaluation.
Means and sample standard deviations are calculated over matched repeats.
Pairing is defined by the event partition, rather than by individual firms.

The low-cost mechanistic comparisons answer an accuracy-cost question and are
reported separately in [Supplementary Note S8](../note_s8_computational_benchmark/README.md).
