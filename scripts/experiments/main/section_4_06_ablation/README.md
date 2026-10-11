# 4.6 Ablation Study

Table 6 examines propagation, output branches and auxiliary supervision on the
Henriet chemical benchmark. The full model and all variants are trained on
three matched event partitions, each containing 4,000 training, 500 validation
and 500 test events. Results are summarized as mean and sample standard
deviation across the three repeats.

| Variant | Scientific comparison |
|---|---|
| Full MeCaSNet | Reference model for this ablation study |
| Decoder-only output | Remove recurrent/direct output fusion |
| Direct rollout output | Remove the structured trajectory decoder |
| Regression-only training | Remove cascade-boundary auxiliary supervision |
| Unconditioned propagation | Remove Day-0 shock conditioning from the encoder |
| Forward-only propagation | Remove reverse demand-feedback messages |
| Gaussian decline front | Change the decline exponent from 6 to 2 |
| Single-exponential recovery | Set the recovery exponent to 1 |
| Plain GAT | Replace the supply-relation-state propagation backbone |

[paper_ablation_train.py](paper_ablation_train.py) defines the study variants
and their training procedures in the original research environment. Full-model
comparisons use the reference trained on these same three partitions, rather
than the ten-repeat average in Table 2.

Peak-loss and trajectory metrics are reported separately because output-branch
changes can affect the two tasks differently. The Plain GAT comparison evaluates
the combined propagation design, whereas the other comparisons target the
listed component or setting.

Temporal-regularization comparisons and trough-timing evaluation belong to
[Supplementary Note S3](../../si/note_s3_training_objective/README.md).
