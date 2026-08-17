# Reproducibility guide

## What can be reproduced publicly

The repository exposes the architecture, losses, training protocol, evaluation
metrics, and audit procedures. Exact manuscript numbers additionally require
the authorized simulator-generated data and are therefore outside the public
release boundary.

## Standard Chemical-domain run

```bash
mecasnet-train \
  --profile paper --variant MeCaSNet \
  --data-root /path/to/data \
  --n-train 4000 --n-val 500 --n-test 500 \
  --epochs 60 --warmup-epochs 3 --lr-schedule cosine \
  --event-scalars-mode minimal \
  --seeds 5 --seed-start 0 --save-ckpt \
  --out-dir runs/chemical/mecasnet
```

The optimizer is AdamW (`lr=3e-4`, weight decay `1e-5`), gradients accumulate
over four events and are clipped at norm 1.0, and the best checkpoint is selected
by validation `R²_pk,csc`.

## Baselines

Repeat the command with each of `MLP`, `GCN`, `GAT`, `STGNN`, and `DirGNN`.
Use the same event IDs, key dates, observable inputs, optimization budget,
selection metric, and cascade threshold. Report trainable parameter counts with
the results.

## Post-training analyses

Examples assume an editable installation (`python -m pip install -e .`).

```bash
mecasnet-thresholds \
  --data-root /path/to/data \
  --checkpoint /path/to/checkpoint.pt \
  --output-dir runs/thresholds --device cuda

python scripts/evaluation/predictive_uncertainty.py \
  --data-root /path/to/data \
  --checkpoints /path/to/seed0.pt /path/to/seed1.pt /path/to/seed2.pt \
  --output-dir runs/uncertainty --n-val 500 --n-test 500 --alpha 0.10

python scripts/evaluation/propagation_depth.py \
  --data-root /path/to/data --checkpoint /path/to/seed0.pt \
  --output runs/propagation-depth.json --threshold 0.05
```

The evaluation utilities named `Y8` load the archived `legacy-y8` profile for
checkpoint compatibility. Use `--profile paper` for all new training runs.

## Result record

Retain the following with every experiment:

- command and named profile;
- code commit and dirty-state indicator;
- ordered event IDs or split-manifest hash;
- static-network and event-data hashes;
- Python, NumPy, SciPy, PyTorch, CUDA, driver, and GPU versions;
- random seeds, epochs, selected epoch, parameter count, and wall time;
- cascade threshold and exact metric implementation;
- whether timing includes startup, I/O, transfer, and warm-up.

## Claim boundaries

The random event split is graph-transductive: event identities are unseen, but
the static firms and relations are shared. Reduced-graph tests modify a known
network and do not establish fully node-disjoint transfer. Simulator-generated
labels support predictive-surrogate evaluation, not automatic causal discovery
in real supply chains.
