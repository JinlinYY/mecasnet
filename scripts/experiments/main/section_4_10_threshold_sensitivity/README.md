# Section 4.10: cascade-threshold sensitivity

The installed `mecasnet-thresholds` entry point evaluates one frozen archived
checkpoint over a list of post-hoc cascade thresholds:

```bash
mecasnet-thresholds \
  --data-root /path/to/authorized/data \
  --checkpoint /path/to/legacy-y8-seed0.pt \
  --output-dir runs/main/section_4_10 \
  --thresholds 0.01 0.02 0.03 0.05 0.075 0.10 0.15 0.20 \
  --benchmark-events 100 --device cuda
```

Thresholding changes the evaluation subset; it does not retrain the model.
Report the threshold, node-event sample count, and any undefined strata for
every result. The primary manuscript threshold is `0.05`.
