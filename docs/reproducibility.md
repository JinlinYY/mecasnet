# Reproducibility guide

## What can be reproduced publicly

The repository exposes the architecture, losses, training protocol, evaluation
metrics, and audit procedures. Exact manuscript numbers additionally require
the authorized simulator-generated data and are therefore outside the public
release boundary.

Reproducibility has three distinct levels in this project:

1. **implementation reproducibility** — inspect and execute the published model,
   losses, split logic, and metrics;
2. **protocol reproducibility** — repeat the training/evaluation procedure on a
   schema-compatible authorized dataset;
3. **numerical result reproducibility** — recover the manuscript tables using
   the exact protected graph, event files, simulator configuration, and software
   environment.

The repository supports the first two. The third additionally requires assets
that are not distributed publicly.

## Before the first experiment

Record the repository and environment state:

```bash
git rev-parse HEAD
git status --short
python --version
python -m pip freeze > environment.txt
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Also record the operating system, GPU driver, CPU model, RAM, GPU memory, and
filesystem location of the data. Do not assume that a package lock alone
captures CUDA kernels or driver behavior.

Run the repository tests before training:

```bash
python -m pip install -e ".[dev,analysis]"
pytest -q
```

The tests cover grouped tensor reduction, profile locking, Day-0 boundary
preservation, and basic data-contract behavior. They are smoke tests, not a
replacement for validating a scientific dataset.

## Data provenance and hashes

At minimum preserve:

- the cryptographic hash of `static_meta.pkl`;
- an ordered table of event ID, filename, size, and hash;
- the split manifest or the exact split seed/fractions;
- the event generator commit and simulator parameter file;
- a statement of whether `peak_loss_node` used daily or key-date trajectories;
- access-control and de-identification decisions for protected data.

Example static-file hashes:

```bash
sha256sum /path/to/data/static_meta.pkl                  # Linux/macOS
certutil -hashfile C:\path\to\data\static_meta.pkl SHA256  # Windows
```

Hash event files through a deterministic, sorted file list. Do not publish the
hash list if it can act as a sensitive dataset fingerprint without data-owner
approval.

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

The expected optimization settings are:

| Item | Setting |
|---|---|
| Event counts | 4000 train, 500 validation, 500 internal test |
| Epochs | 60 |
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Weight decay | `1e-5` |
| Schedule | 3-epoch linear warm-up, then cosine to 0.1× base LR |
| Gradient accumulation | 4 events |
| Gradient clipping | norm 1.0 |
| Seeds | 0–4 |
| Selection | largest validation `R²_pk,csc` |

See [training.md](training.md) for the complete CLI and checkpoint schema.

## Smoke run before the full budget

Use a separate output directory and a small event budget to catch environment or
shape errors:

```bash
mecasnet-train \
  --profile paper --variant MeCaSNet \
  --data-root /path/to/data \
  --n-train 8 --n-val 4 --n-test 4 \
  --epochs 1 --seeds 1 --device cpu \
  --out-dir runs/smoke
```

This verifies loading and a full optimization/evaluation path. It does not
validate performance or approximate the paper result.

## Baselines

Repeat the command with each of `MLP`, `GCN`, `GAT`, `STGNN`, and `DirGNN`.
Use the same event IDs, key dates, observable inputs, optimization budget,
selection metric, and cascade threshold. Report trainable parameter counts with
the results.

Do not reuse a MeCaSNet-selected epoch for a comparator. Each model is selected
independently on validation data, then evaluated once on test data. Keep the
same seed labels so comparisons can use paired seed-level differences.

Capacity variants must be labeled as such; a deeper comparator is an additional
analysis, not a silent replacement for the registered baseline.

## Post-training analyses

Examples assume an editable installation (`python -m pip install -e .`).

```bash
mecasnet-thresholds \
  --data-root /path/to/data \
  --checkpoint /path/to/legacy-y8-checkpoint.pt \
  --output-dir runs/thresholds --device cuda

python scripts/evaluation/predictive_uncertainty.py \
  --data-root /path/to/data \
  --checkpoints /path/to/legacy-y8-seed0.pt /path/to/legacy-y8-seed1.pt \
                /path/to/legacy-y8-seed2.pt \
  --output-dir runs/uncertainty --n-val 500 --n-test 500 --alpha 0.10

python scripts/evaluation/propagation_depth.py \
  --data-root /path/to/data --checkpoint /path/to/legacy-y8-seed0.pt \
  --output runs/propagation-depth.json --threshold 0.05
```

The evaluation utilities named `Y8` load the archived `legacy-y8` profile for
checkpoint compatibility. Use `--profile paper` for all new training runs.

Before applying an archived analysis script to a new checkpoint, read its model
builder. Many scripts intentionally require `legacy-y8` and strict state loading.
See [evaluation_protocol.md](evaluation_protocol.md) for a script-by-script
scope explanation.

## Expected output audit

After a five-seed run, verify:

1. the summary JSON records all five expected seeds;
2. each seed has a finite selected epoch;
3. checkpoint `config.profile` is `paper`;
4. validation selection and internal-test metrics are both present;
5. parameter counts are identical across seeds of the same variant;
6. per-key-date metrics have ten entries;
7. no test value appears in the epoch-selection history;
8. the output directory contains no raw event or protected graph data.

Archive the console log as well as the JSON. The log records effective profile,
device fallback, model variant, event counts, parameter count, and saved paths.

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

Suggested result manifest:

```json
{
  "repository_commit": "<40-character SHA>",
  "dirty_worktree": false,
  "profile": "paper",
  "variant": "MeCaSNet",
  "data": {
    "static_sha256": "<hash>",
    "split_manifest_sha256": "<hash>",
    "peak_definition": "maximum over full simulator trajectory"
  },
  "runtime": {
    "python": "3.11.x",
    "torch": "2.x",
    "cuda": "<version or null>",
    "gpu": "<model or cpu>"
  },
  "command_file": "train_command.sh",
  "summary_file": "mecasnet_summary.json"
}
```

Keep this manifest next to outputs, not inside a checkpoint alone.

## Randomness and determinism

The driver sets NumPy and PyTorch seeds for every model run and uses zero worker
processes by default. The event partition uses `Config.seed=0` unless a split
manifest is supplied.

The code does not globally enable `torch.use_deterministic_algorithms(True)`.
CUDA kernels, driver/library versions, and hardware may therefore produce small
run-to-run differences even with the same seed. Multi-seed mean and standard
deviation are the intended reporting unit; do not promise bitwise-identical GPU
weights across platforms.

For a strict engineering audit, enable deterministic PyTorch behavior in a
separate run and report any performance/runtime impact. Do not mix deterministic
and default runs in one aggregate.

## Checkpoint lineage

Two profiles are intentionally distinct:

- `paper`: final information boundary, `c=6`, learned `q_v`;
- `legacy-y8`: archived pre-final checkpoint compatibility, `c=2`, no learned
  `q_v`.

A successful `strict=False` load does not prove scientific compatibility.
Record the profile explicitly and use strict loading for frozen evaluations.

## Claim boundaries

The random event split is graph-transductive: event identities are unseen, but
the static firms and relations are shared. Reduced-graph tests modify a known
network and do not establish fully node-disjoint transfer. Simulator-generated
labels support predictive-surrogate evaluation, not automatic causal discovery
in real supply chains.

Additional boundaries:

- The primary split does not test unseen firms or an unseen industrial system.
- Induced-subgraph tests assess perturbations of a known graph, not fully
  node-disjoint induction unless nodes are separated during training.
- Input-perturbation intervals measure reconstruction sensitivity, not total
  predictive uncertainty.
- Raw ensemble spread is not calibrated coverage.
- Structured decoder parameters are predictive latent quantities and should not
  be reported as uniquely identified physical constants without a separate
  identifiability analysis.
- A surrogate's agreement with simulator labels does not validate the simulator
  as a causal model of real disruptions.

## Publication checklist

Before releasing a table, figure, or checkpoint:

- [ ] code commit and worktree state recorded;
- [ ] data/split hashes recorded;
- [ ] profile and variant stated;
- [ ] all five seeds completed or deviations explained;
- [ ] validation-only selection confirmed;
- [ ] subset threshold and sample count reported;
- [ ] undefined strata retained as `NaN`, not silently removed;
- [ ] timing scope and hardware disclosed;
- [ ] public/private asset boundary reviewed;
- [ ] checkpoint release approved for privacy and memorization risk;
- [ ] final DOI and citation metadata updated after publication.
