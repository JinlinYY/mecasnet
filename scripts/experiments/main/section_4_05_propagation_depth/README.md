# Section 4.5: propagation depth and network position

`propagation_depth.py` analyzes one frozen MeCaSNet checkpoint. The matched-seed
`network_propagation.py` compares MeCaSNet and DirGNN across downstream-distance
and feedback-position strata.

```bash
python scripts/experiments/main/section_4_05_propagation_depth/propagation_depth.py \
  --data-root /path/to/authorized/data \
  --checkpoint /path/to/compat-seed0.pt \
  --output runs/main/section_4_5/depth.json \
  --threshold 0.05 --n-test 500 --device cuda

python scripts/experiments/main/section_4_05_propagation_depth/network_propagation.py \
  --checkpoint-root /path/to/matched-checkpoints \
  --data-root /path/to/authorized/data \
  --output runs/main/section_4_5/network-position.json \
  --threshold 0.05 --n-test 500 --device cuda
```

These are descriptive error stratifications. They do not establish a causal
transmission mechanism by themselves.
