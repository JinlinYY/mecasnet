# Training guide

This guide covers installation, the paper training profile, deterministic data
splits, comparator models, checkpoints, and generated files. The commands assume
the repository root as the current directory.

## 1. Environment setup

Python 3.10 or newer is required. Install the PyTorch build appropriate for the
local CPU/CUDA environment first, then install MeCaSNet in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate              # Linux/macOS
# .venv\Scripts\activate               # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -e ".[dev,analysis]"
pytest
```

Confirm the runtime before a long experiment:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
mecasnet-train --help
```

If CUDA is requested but unavailable, the training driver falls back to CPU.
This is useful for smoke tests but generally impractical for the full paper
budget.

## 2. Data preflight

Before training, verify:

- `DATA_ROOT/static_meta.pkl` exists;
- `DATA_ROOT/events/` contains consecutively named or otherwise uniquely
  numbered `event_XXXXXX.npz` files;
- all events use the key dates `[0,5,10,20,30,50,70,100,150,199]`;
- edge orientation is supplier to customer;
- event arrays use the same node ordering as `static_meta.pkl`;
- event IDs are sufficient for the requested train/validation/test counts.

The loader validates array shapes, index ranges, finite positive production and
flows, binary shock masks, finite damage, and exact key-date agreement. See
[data_contract.md](data_contract.md) for the complete schema.

Both `static_meta.pkl` and `shock_meta` use Python object deserialization. Load
only data created by a trusted source.

## 3. Paper-profile command

```bash
mecasnet-train \
  --profile paper \
  --variant MeCaSNet \
  --data-root /path/to/authorized/cascade-data \
  --n-train 4000 --n-val 500 --n-test 500 \
  --epochs 60 \
  --lr-schedule cosine --warmup-epochs 3 --min-lr-ratio 0.1 \
  --event-scalars-mode minimal \
  --seeds 5 --seed-start 0 \
  --save-ckpt \
  --out-dir runs/chemical/mecasnet
```

The `paper` profile locks the information and architecture boundary. In
particular, `--event-scalars-mode` must be `minimal`; architecture-changing
experimental environment variables do not override the final paper settings.

## 4. Default optimization protocol

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Base learning rate | `3e-4` |
| Weight decay | `1e-5` |
| Epochs | 60 from the public training CLI |
| Warm-up | 3 epochs, linear |
| Post-warm-up schedule | cosine decay |
| Minimum LR ratio | 0.1 of base LR |
| Gradient accumulation | 4 events |
| Gradient clipping | global norm 1.0 |
| Seeds | 5, starting at 0 |
| Validation frequency | every epoch |
| Selection metric | validation `R²_pk,csc` |
| DataLoader batch size | 1 variable-size event |

`Config.epochs` is a library-level default; the `mecasnet-train` CLI explicitly
sets the manuscript default to 60. For custom event counts, prefer a training
count divisible by four because optimizer updates occur after each complete
gradient-accumulation group.

## 5. Important CLI options

| Option | Default | Effect |
|---|---:|---|
| `--data-root` | required | Static network and event directory |
| `--profile` | `paper` | `paper` or archived `legacy-y8` |
| `--variant` | `MeCaSNet` | Architecture or comparator to train |
| `--n-train` | 4000 | Maximum events retained from the training partition |
| `--n-val` | 500 | Maximum events retained from validation |
| `--n-test` | 500 | Maximum events retained from internal test |
| `--epochs` | 60 | Training epochs per seed |
| `--device` | `cuda` | Requested torch device |
| `--seeds` | 5 | Number of consecutive seeds |
| `--seed-start` | 0 | First model seed |
| `--lr-schedule` | `cosine` | `cosine` or `const` |
| `--warmup-epochs` | 3 | Linear warm-up length |
| `--min-lr-ratio` | 0.1 | Cosine floor relative to base LR |
| `--event-scalars-mode` | `minimal` | Event-information exposure; paper requires minimal |
| `--save-ckpt` | off | Save the best state for every seed |
| `--out-dir` | `runs/chemical/mecasnet` | JSON summaries and checkpoints |
| `--resume-from` | empty | Initialize each seed from a checkpoint state dictionary |
| `--split-manifest` | empty | Use explicit event IDs instead of the random event split |
| `--strict-test-data-root` | empty | Evaluate a separate event pool after model selection |

Underscore aliases remain accepted for compatibility, but hyphenated options
are preferred in new commands.

## 6. Default event split

Without a manifest, the loader:

1. sorts all `event_*.npz` filenames;
2. extracts integer event IDs;
3. shuffles them with NumPy RNG seed 0;
4. assigns 80% to training, 10% to validation, and 10% to test;
5. truncates the three partitions to `n_train`, `n_val`, and `n_test`.

This is an event-disjoint but graph-transductive split. Every partition uses the
same static nodes and edges. It evaluates new disruptions on a known network,
not transfer to unseen firms.

For exact repeatability, preserve the ordered IDs or use a manifest:

```json
{
  "train_ids": [0, 7, 12],
  "validation_ids": [2],
  "internal_test_ids": [5],
  "strict_test_ids": [10001, 10002]
}
```

Run with:

```bash
mecasnet-train \
  --profile paper --variant MeCaSNet \
  --data-root /path/to/development-pool \
  --split-manifest splits/paper_split.json \
  --strict-test-data-root /path/to/locked-test-pool \
  --epochs 60 --seeds 5 --save-ckpt \
  --out-dir runs/preregistered
```

When a manifest is used, the `n_train/n_val/n_test` limits are ignored. The
driver rejects overlapping or empty development splits and missing event files.
The strict pool must use exactly the same static graph arrays and feature width;
it is a locked event pool, not a different network.

## 7. Validation and checkpoint selection

Every epoch is evaluated on validation events. The driver keeps the state with
the largest validation cascade-subset peak-loss R-squared:

```text
cascade subset = (shock_mask == 0) and (true peak loss > 0.05)
```

Only after selection is that state evaluated on internal test and optional
strict-test events. The test metric is never used to choose an epoch.

If the validation cascade subset is empty or its target standard deviation is
below `0.01`, the selection metric is `NaN`. Fix the split or dataset rather
than selecting by test performance.

## 8. Loss composition

The default objective is a sum of the following terms:

| Term | Weight | Purpose |
|---|---:|---|
| Key-date trajectory loss | 1.0 | Fit production ratios at all ten dates |
| Peak-loss regression | 2.0 | Fit node peak production loss |
| Recovery-rate regularizer | 0.1 | Limit overly rapid recovery |
| Late monotonicity | 0.5 | Penalize new decline after Day 70 |
| Reachability BCE | 0.15 | Separate affected from nearly inert nodes |
| Trough-date classification | 0.05 | Identify the key date of minimum production |
| Mass residual | 0.0 | Disabled in the paper defaults |
| PINN residual | 0.0 | Disabled in the paper defaults |

Key-date weights for Days `[0,5,10,20,30,50,70,100,150,199]` are:

```text
[1.0, 2.0, 1.5, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]
```

The data loss uses a severity-dependent node weight
`0.1 + 5 * true_peak_loss`. Directly shocked nodes additionally have a minimum
weight of `2.5`. Auxiliary labels influence training only; they are not fed to
the predictor.

## 9. Comparator models

Use the same split, event inputs, epochs, seeds, and selection rule for every
comparator:

```bash
for model in MLP GCN GAT DirGNN STGNN Physics; do
  mecasnet-train \
    --profile paper --variant "$model" \
    --data-root /path/to/data \
    --n-train 4000 --n-val 500 --n-test 500 \
    --epochs 60 --seeds 5 --seed-start 0 \
    --lr-schedule cosine --warmup-epochs 3 \
    --out-dir "runs/baselines/$model"
done
```

| Variant | Structural difference |
|---|---|
| `MLP` | No graph aggregation |
| `GCN` | Undirected/symmetric normalized graph convolution comparator |
| `GAT` | Attention-based graph comparator |
| `DirGNN` | Explicit forward/reverse directed aggregation |
| `STGNN` | Spatial graph encoder with recurrent temporal prediction |
| `Physics` | Analytical Leontief-style propagation without a learned GNN |

Capacity probes (`MLP_big`, `GAT_deep`, `GCN_deep`, `DirGNN_deep`) and named
`V3DE_*` variants are retained for ablations and checkpoint lineage. They should
not replace the primary matched comparator unless the deviation is reported.

## 10. Output files

For `--variant MeCaSNet --seeds 5 --seed-start 0 --save-ckpt`, the output
directory contains:

```text
runs/chemical/mecasnet/
├── mecasnet_seed0.pt
├── mecasnet_seed1.pt
├── mecasnet_seed2.pt
├── mecasnet_seed3.pt
├── mecasnet_seed4.pt
└── mecasnet_summary.json
```

Each checkpoint includes:

- `state_dict` selected by validation performance;
- `best_val` metrics;
- `internal_test` metrics;
- optional `strict_test` metrics;
- selected `epoch`;
- a small `config` record containing variant/profile lineage.

The summary JSON includes CLI arguments, seed range, per-seed validation and
test metrics, aggregate mean/standard deviation, and per-epoch histories.

Keep the full command, Git commit, data hashes, environment, and hardware next
to this output; the checkpoint payload is not a complete provenance record.

## 11. Resuming and checkpoint compatibility

`--resume-from` loads weights with `strict=False`, prints missing/unexpected
keys, and starts a fresh optimizer and LR schedule. It is initialization, not a
bit-exact continuation of an interrupted optimizer state.

Use `--profile legacy-y8 --require-y8-exact` only for the archived Y8-H model.
That guard checks the expected architecture and parameter count. New manuscript
runs should always use `--profile paper`.
