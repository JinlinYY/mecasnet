# Section 4.7: damage-trajectory reconstruction

The trajectory predictions are emitted by the normal evaluation path as
`u_keyframes`. The analytical and fused branches are implemented in
`src/mecasnet/model_v3.py`; the final late correction is in
`src/mecasnet/model_v3_residual.py`.

`recovery_parameters.py` is a supporting archived-checkpoint diagnostic. It
captures the structured decoder's raw parameter head and reports transformed
recovery-parameter distributions:

```bash
python scripts/experiments/main/section_4_07_trajectory_reconstruction/recovery_parameters.py \
  --data-root /path/to/authorized/data \
  --checkpoint /path/to/legacy-y8-seed0.pt \
  --output-dir runs/main/section_4_7/recovery-parameters \
  --n-test 500 --device cuda
```

The repository does not include manuscript figure-result artifacts. When
rebuilding representative trajectory plots, use the same event-specific
cascade set for predictions and targets, and do not label ground-truth
cross-node spread as predictive uncertainty.
