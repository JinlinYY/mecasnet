# 4.3 Overall Cascade Prediction Performance

This experiment compares MeCaSNet with MLP, GCN, GAT, STGNN and DirGNN on the
Chemical and NEEQ benchmarks. Table 2 summarizes ten repeated event partitions
for each benchmark. Every repeat uses 4,000 training, 500 validation and 500
test events, with identical assignments across models.

Models are selected using validation performance. Cascade peak-loss MAE,
cascade peak-loss R-squared and cascade trajectory R-squared are then evaluated
on the test events. The mean and sample standard deviation across repeats
reflect both event-partition variation and training randomness. Chemical-domain
comparisons with DirGNN use paired two-sided t-tests across the same ten repeats.

| Analysis code | Scientific role |
|---|---|
| [repeated_split_experiment.py](repeated_split_experiment.py) | Seed-specific event partitions and matched model comparisons |
| [neeq_repeated_train.py](neeq_repeated_train.py) | Repeated-partition training procedures for the NEEQ benchmark |
| [threshold_multiseed.py](threshold_multiseed.py) | Matched-checkpoint evaluation at a common cascade threshold |
| [Model training](../../../../src/mecasnet/train.py) | Training and validation-based model selection |

The repeated-partition files include research procedures used with the original
training environment. The matched-checkpoint evaluator uses the documented
`compat` model settings. New `paper` model results are evaluated with their own
training-time model definition and event assignments. Checkpoint settings and
evaluation populations must agree before results are compared.

Comparator inputs, capacities and supervision are discussed in
[Supplementary Note S4](../../si/note_s4_baseline_protocol/README.md).
Data and software requirements appear in the
[reproduction guide](../../REPRODUCTION.md).
