# Supplementary Note S4: baseline comparability

All neural comparators are built by `src/mecasnet/train.py` and share the same
event inputs, split, optimizer budget, validation rule, and test timing.

Train each model with the same command while changing only `--variant`:

```bash
for model in MLP GCN GAT STGNN DirGNN MeCaSNet; do
  mecasnet-train \
    --profile paper --variant "$model" \
    --data-root /path/to/authorized/data \
    --n-train 4000 --n-val 500 --n-test 500 \
    --epochs 60 --seeds 5 --seed-start 0 --save-ckpt \
    --out-dir "runs/si/s4/$model"
done
```

Use the Section 4.3 matched-seed evaluator only with the archived checkpoint
family it reconstructs. Always report parameter counts and label deeper
capacity probes separately.
