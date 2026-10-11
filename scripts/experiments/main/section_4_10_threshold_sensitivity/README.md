# 4.10 Sensitivity to the Cascade-Definition Threshold

Table 10 evaluates whether cascade-specific prediction metrics depend strongly
on the threshold used to identify indirectly affected firms. The same frozen
model and the same 500 reference test events are retained throughout.

The thresholds are 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25 and
0.30. Only the subset of initially unshocked prediction nodes with a target peak
loss above the threshold changes. Model parameters and model selection remain
fixed.

The calculation is included in [evaluation.py](../../../../src/mecasnet/evaluation.py).
For a checkpoint with the documented `compat` settings:

```bash
mecasnet-thresholds \
  --data-root /path/to/authorized/data \
  --checkpoint /path/to/compat-checkpoint.pt \
  --output-dir runs/main/section_4_10 \
  --thresholds 0.01 0.02 0.03 0.05 0.075 0.10 0.15 0.20 0.25 0.30 \
  --device cuda
```

Report the threshold, cascade event-firm count, fraction of initially unshocked
prediction observations and the three cascade metrics. The reference denominator
is 22,865 initially unshocked event-firm observations within the five-hop
prediction scope. These are repeated observations across events, rather than
22,865 unique firms.
