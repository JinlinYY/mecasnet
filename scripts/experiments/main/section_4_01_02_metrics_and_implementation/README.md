# Sections 4.1–4.2: metrics and implementation details

These sections are implemented in the installable package rather than a
post-training script.

| Responsibility | Implementation |
|---|---|
| Cascade subsets and `R²`/MAE metrics | `src/mecasnet/runner.py` |
| Event-level training and validation selection | `src/mecasnet/train.py` |
| Optimizer and loss coefficients | `src/mecasnet/config.py` |
| Composite objective | `src/mecasnet/losses.py` |
| Threshold evaluation CLI | `src/mecasnet/evaluation.py` |

Inspect the effective command surface with:

```bash
mecasnet-train --help
mecasnet-thresholds --help
```

Metric definitions, undefined-stratum behavior, and aggregation rules are in
`docs/evaluation_protocol.md`. Training defaults and checkpoint selection are
in `docs/training.md`.
