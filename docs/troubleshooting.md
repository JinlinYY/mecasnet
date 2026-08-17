# Troubleshooting

This page lists common failures in installation, data loading, training,
checkpoint evaluation, and metrics. Start with the first error in the traceback;
later errors are often consequences.

## Installation and imports

### `ModuleNotFoundError: No module named 'mecasnet'`

Install the repository from its root:

```bash
python -m pip install -e ".[dev,analysis]"
```

Confirm that `python` and `pip` refer to the same environment:

```bash
python -c "import sys; print(sys.executable)"
python -m pip --version
python -c "import mecasnet; print(mecasnet.__version__)"
```

Do not solve this by copying source files into the working directory or by
adding old internal directories to `PYTHONPATH`.

### PyTorch or CUDA cannot be imported

Install PyTorch using the official command for the local OS, Python, and CUDA
version. Then check:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The repository declares a minimum torch dependency but cannot choose the
correct CUDA wheel for every machine.

### Training requested CUDA but is running on CPU

The driver falls back to CPU when `torch.cuda.is_available()` is false. Inspect
the startup log and verify the driver, CUDA runtime, and torch build. Use CPU
only for smoke tests unless the dataset is very small.

## Static-network errors

### Missing `static_meta.pkl`

`--data-root` must point to the directory containing both `static_meta.pkl` and
`events/`, not to the `events/` directory itself.

```text
correct:   --data-root /project/data/cascade_set
incorrect: --data-root /project/data/cascade_set/events
```

### Invalid static-network array shapes

Check that node arrays have length `V`, edge arrays have length `E`, and the
declared `V/E` values reflect the exported arrays after filtering or edge
aggregation. A common cause is filtering an edge table without updating `E`.

### Edge index outside `[0,V)`

The exporter likely retained source-system IDs instead of mapping firms to
contiguous zero-based indices. Build one node-index mapping and use it for every
static and event array.

### `P_ini` or `A` must contain finite positive values

Remove or explicitly model inactive firms and zero-flow relations before
export. Do not replace invalid production with an arbitrary epsilon solely to
pass validation; input shares would become scientifically meaningless.

### Checkpoint feature-width mismatch

With sector one-hot features, `Fv=n_sec+5`. A checkpoint trained with one sector
taxonomy cannot be loaded into a graph with a different width. Options are:

- reproduce the original sector vocabulary and ordering;
- train a new checkpoint;
- deliberately set `use_sector_oh=False` for all compared networks and retrain.

Changing the width only at evaluation time is not checkpoint compatibility.

## Event-file errors

### No events found

Files must be under `DATA_ROOT/events/` and match `event_*.npz`. The part after
the underscore must parse as an integer.

### Event shape does not match the static graph

All global event vectors must have length `V`; `u_keyframes` must be `(10,V)`.
If events were generated before a node filter or reordering, regenerate them or
apply the exact same mapping to every field.

### `key_days` does not match `Config.key_days`

The paper profile requires:

```text
[0, 5, 10, 20, 30, 50, 70, 100, 150, 199]
```

Resampling or interpolating targets is a data-generation decision. Do it before
training and record the interpolation rule; do not silently relabel dates.

### `shock_meta` is not a mapping

Store it as a scalar object:

```python
shock_meta=np.array({"source": "my-generator"}, dtype=object)
```

Minimal mode still requires a mapping for schema compatibility, although it
does not read simulator recovery/mode/tier fields.

### The event subgraph is unexpectedly small or empty

Verify:

- at least one node has `shock_mask=1`;
- edges are oriented supplier to customer;
- shocked node indices use the static node order;
- the intended cascade travels downstream under that convention.

An event with no direct shocks yields an empty reach set and is not a useful
training example.

## Training behavior

### Out of memory

Event batches have variable `Nr` and `Er`; one unusually large reach subgraph
can dominate memory. Inspect reach sizes before training. Safe remedies include:

- reducing hidden width in an explicitly labeled capacity experiment;
- reducing reach hops with a scientific justification;
- using the existing event-level subgraph augmentation for a training study;
- moving to a larger-memory device.

Do not increase DataLoader batch size above one without implementing padded or
disjoint-graph batching.

### Loss is `NaN` or `inf`

Check source arrays for nonfinite target values, then inspect the first event and
loss component that fails. Also check for an empty local graph, extreme input
shares, and incompatible experimental environment variables. The paper profile
should be run without architecture-changing environment overrides.

### The last gradient-accumulation group is not stepped

The current loop updates after each complete group of four events. Use a
training count divisible by `grad_accum=4` for registered runs. The manuscript
count of 4000 satisfies this condition.

### Validation `R²_pk,csc` is `NaN`

The validation cascade subset is empty or its true peak-loss standard deviation
is below 0.01. Inspect `n_csc`, the shock masks, target distribution, and split.
Do not fall back to choosing an epoch on the test set.

### Results differ across identical GPU runs

The driver seeds NumPy and PyTorch but does not globally force deterministic
CUDA algorithms. Small differences across hardware/library versions are
expected. Record the environment and report the registered multi-seed aggregate.

## Checkpoint problems

### Missing or unexpected state-dictionary keys

First check the profile:

- new manuscript runs: `paper`;
- archived Y8-H analysis: `legacy-y8`.

Then check the model variant, sector feature width, hidden width, propagation
block count, and whether the file is a raw state dictionary or a checkpoint
payload containing `state_dict`.

For frozen evaluation, keep `strict=True`. Using `strict=False` can create a
partially random model while appearing to load successfully.

### `--require-y8-exact` fails

This option is only for `--profile legacy-y8 --variant MeCaSNet`. It verifies
the archived architecture and parameter count. It is expected to reject the
final `paper` profile.

### Resume does not continue the same optimization trajectory

`--resume-from` restores model weights only. Optimizer state, scheduler state,
epoch counter, and RNG state start fresh. Treat this as warm-start training, not
exact interrupted-run continuation.

## Metrics and analysis

### Sample counts are larger than the number of firms

Metrics pool node-event observations from many reach subgraphs. The same static
firm may appear in multiple events. `n_total` is not a unique-firm count.

### Per-key-date mean looks valid but some dates are missing

`r2_kf_csc_mean` averages finite date-level values. Inspect the ten entries in
`r2_kf_csc`; a constant or empty stratum becomes `NaN` and is excluded from the
mean.

### A post-training script rejects a new paper checkpoint

Many reviewer-analysis scripts intentionally reconstruct `legacy-y8`. They are
audit records for archived checkpoints. Use the training summary for new paper
runs or adapt the script's builder explicitly and document the change.

### A topology script rejects `n_sec` or `Fv`

Some topology workflows contain manuscript-network checks and are not generic
data utilities. Review their module docstring and expected feature contract
before adapting them to another graph.

### Sections 4.8 or 4.9 cannot import simulator modules

The repair and runtime scripts are under `scripts/experiments/main/` in their
article-aligned section folders. The ARIO/Henriet, Inoue-Todo, and FINDER
backends are not part of the public release. These scripts require an authorized
private backend on `PYTHONPATH` and the corresponding assets.

## Getting a useful bug report

Include:

- operating system and Python/torch/CUDA versions;
- repository commit and `git status --short`;
- the exact command;
- full traceback from the first error;
- model profile and variant;
- static `V/E/n_sec/Fv` and event `Nr/Er` (without sensitive identifiers);
- whether a minimal synthetic dataset reproduces the problem.

Do not attach protected graphs, event data, checkpoints, credentials, or firm
identifiers to a public issue without authorization.
