# Section 4.3: overall cascade prediction performance

Train MeCaSNet and every comparator with `mecasnet-train`; use identical event
IDs, seed labels, optimization budgets, and validation selection rules.

Example full-model run:

```bash
mecasnet-train \
  --profile paper --variant MeCaSNet \
  --data-root /path/to/authorized/data \
  --n-train 4000 --n-val 500 --n-test 500 \
  --epochs 60 --seeds 5 --seed-start 0 --save-ckpt \
  --out-dir runs/main/section_4_3/mecasnet
```

`threshold_multiseed.py` re-evaluates a matched set of completed checkpoints at
one cascade threshold and computes paired model comparisons:

```bash
python scripts/experiments/main/section_4_03_overall_performance/threshold_multiseed.py \
  --root /path/to/matched-checkpoints \
  --data-root /path/to/authorized/data \
  --output-dir runs/main/section_4_3/table \
  --threshold 0.05 --n-test 500 --device cuda
```

The post-training script constructs the `compat` profile. For `paper`
checkpoints, use the metrics in each training summary unless the analysis
builder has been explicitly configured for that profile.
