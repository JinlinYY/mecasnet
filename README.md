# MeCaSNet

Official research code for **Mechanism-Guided Graph Learning for Dynamic
Disruption Propagation and Recovery Prediction in Process-Industry Supply
Chains**.

MeCaSNet predicts firm-level production-ratio trajectories after a Day-0
disruption. It combines shock-conditioned node encoding, persistent
supply-relation states, direction-specific supply and demand messages, and
three complementary trajectory streams: a structured decline–trough–recovery
decoder, a recurrent rollout, and a direct key-date readout.

## Method at a glance

1. Represent the operating supply chain as a directed graph with input-share
   edge weights.
2. Encode pre-disruption firm attributes, the direct-shock mask, Day-0 damage,
   and the mean observed Day-0 shock intensity.
3. Propagate forward material-shortage messages and reverse demand-feedback
   messages through persistent edge states.
4. Reconstruct production ratios at Days 0, 5, 10, 20, 30, 50, 70, 100, 150,
   and 199.
5. Train with trajectory, peak-loss, cascade-extent, trough-timing, and
   recovery-regularity objectives.

The primary cascade subset is
`(not directly shocked) and (ground-truth peak loss > 0.05)`.

## Installation

Python 3.10 or newer is required. Install PyTorch from the channel appropriate
for your CPU/CUDA platform, then install this project:

```bash
git clone https://github.com/JinlinYY/mecasnet.git
cd mecasnet
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,analysis]"
pytest
```

For large experiments, a CUDA-capable GPU is recommended. Small smoke tests can
run on CPU.

## Documentation

Start with the [documentation guide](docs/README.md), which provides separate
reading paths for first-time users, experiment replication, method review, and
code maintenance. The main technical references are:

- [Architecture](docs/architecture.md): equations, tensor flow, profile
  differences, output semantics, and source locations.
- [Data contract](docs/data_contract.md): required files, array shapes, edge
  orientation, event semantics, validation, and a minimal example.
- [Training](docs/training.md): preflight checks, paper-profile settings,
  command-line options, losses, outputs, checkpoints, and resume behavior.
- [Evaluation protocol](docs/evaluation_protocol.md): subsets, metric
  definitions, aggregation rules, uncertainty analyses, and reporting checks.
- [Reproducibility](docs/reproducibility.md): environment capture, manifests,
  hashes, seeds, provenance, and publication checklist.
- [Troubleshooting](docs/troubleshooting.md): installation, data, memory,
  checkpoint, metric, and analysis-script diagnostics.

## Data contract

Point `--data-root` to a directory with this layout:

```text
DATA_ROOT/
├── static_meta.pkl
└── events/
    ├── event_000000.npz
    ├── event_000001.npz
    └── ...
```

The required arrays, shapes, edge orientation, normalization rules, and privacy
constraints are specified in [docs/data_contract.md](docs/data_contract.md).
The loader intentionally keeps the model input width fixed while the `paper`
profile exposes only the manuscript's Day-0 information set.

## Training on an event partition

```bash
mecasnet-train \
  --profile paper \
  --variant MeCaSNet \
  --data-root /path/to/authorized/chemical-cascade-data \
  --n-train 4000 --n-val 500 --n-test 500 \
  --epochs 60 --warmup-epochs 3 \
  --lr-schedule cosine --min-lr-ratio 0.1 \
  --event-scalars-mode minimal \
  --split-manifest /path/to/repeat_00.json \
  --seeds 1 --seed-start 0 --save-ckpt \
  --out-dir runs/chemical/mecasnet
```

The command above represents one specified event partition. The principal
comparison repeats training on ten seed-specific partitions; the architecture
ablation uses three matched partitions. See the [repeated-study procedure](docs/reproducibility.md)
for the full comparison design.

The `paper` profile fixes the documented information set and architecture: Day-0-only
event information, `u(0) = 1 - shock * damage`, four persistent edge-state
propagation steps, a three-component lower envelope with `c = 6`, learned
node-specific recovery exponent `q` in `[0.5, 2.0]`, and three-stream fusion.

To train a comparator under the same split and optimization protocol, replace
`MeCaSNet` with `MLP`, `GCN`, `GAT`, `STGNN`, or `DirGNN`.

The `compat` profile provides the fixed `c=2` configuration used by the
post-training analysis scripts. New studies should use the `paper` profile.

## Experimental studies

The [experimental guide](scripts/experiments/README.md) follows the current
manuscript and Supplementary Information. Each study describes its scientific
question, comparison conditions, evaluation population and available analyses.

| Study | Manuscript location | Further details |
|---|---|---|
| Metrics and common training conditions | Sections 4.1-4.2 | [Definitions and study design](scripts/experiments/main/README.md) |
| Repeated model comparison | Section 4.3; Table 2 | [Overall cascade prediction](scripts/experiments/main/section_4_03_overall_performance/README.md) |
| Event similarity and topology perturbation | Section 4.4; Table 3 | [Robustness evaluation](scripts/experiments/main/section_4_04_split_and_topology/README.md) |
| Propagation depth and network position | Section 4.5; Tables 4-5 | [Structural comparison](scripts/experiments/main/section_4_05_propagation_depth/README.md) |
| Architecture ablation | Section 4.6; Table 6 | [Matched component comparisons](scripts/experiments/main/section_4_06_ablation/README.md) |
| Trajectory and spatial damage reconstruction | Section 4.7; Figures 4-5 | [Temporal and spatial analysis](scripts/experiments/main/section_4_07_trajectory_reconstruction/README.md) |
| Repair-set screening | Section 4.8; Figure 6 | [Screening and resilience](scripts/experiments/main/section_4_08_repair_screening_and_resilience/README.md) |
| Accuracy-cost comparisons and cost amortization | Section 4.9; Tables 7-9 | [Computational study](scripts/experiments/main/section_4_09_efficiency/README.md) |
| Cascade-threshold sensitivity | Section 4.10; Table 10 | [Threshold analysis](scripts/experiments/main/section_4_10_threshold_sensitivity/README.md) |
| Input sensitivity and independent event-level coverage | Section 4.11; Tables 11-12 | [Sensitivity and predictive uncertainty](scripts/experiments/main/section_4_11_uncertainty/README.md) |

[Supplementary Notes S1-S9](scripts/experiments/si/README.md) describe the dataset,
objective, comparator definitions, decoder rationale, predictive information,
robustness, computational comparisons and repair evaluation. Temporal
regularization and trough-timing diagnostics are presented in Note S3;
computational cost and repair-set evaluation are presented in Notes S8 and S9.

The [experimental reproduction guide](scripts/experiments/REPRODUCTION.md)
describes the required data, simulator-dependent calculations and companion
research procedures. The code supports examination and reuse of the available
methods; numerical reproduction additionally requires the corresponding
research data and simulation environment.

## Repository layout

```text
src/mecasnet/        Model, data, loss, training and evaluation calculations
scripts/experiments/main/  Studies corresponding to manuscript Sections 4.1-4.11
scripts/experiments/si/    Studies corresponding to Supplementary Notes S1-S9
docs/                Data contract and reproducibility documentation
tests/               CPU-friendly unit and architecture smoke tests
```

## Reproducibility and interpretation

- Select checkpoints using validation `R²_pk,csc`; evaluate the test set only
  after selection.
- Keep event IDs, split manifests, data hashes, random seeds, software versions,
  and hardware metadata with every result.
- Do not interpret the standard event split as unseen-network transfer: all
  events share one static graph. The induced-subgraph analysis evaluates
  robustness to topology modification, not fully node-disjoint induction.
- Do not interpret ground-truth trajectory spread as predictive uncertainty.
- Do not compare surrogate and simulator wall-clock times without reporting
  startup, I/O, warm-up, event count, hardware, and simulator configuration.

## Code and data availability

This repository contains the model, training, evaluation, sensitivity-analysis,
and audit code. Confidential firm identities, supply relations, event files,
simulator implementations, trained checkpoints, and paper result artifacts are
not distributed in this repository. Exact reproduction of the manuscript's
numerical tables therefore requires an authorized dataset in the documented
format.

See [DATA_AND_SOFTWARE_AVAILABILITY.md](DATA_AND_SOFTWARE_AVAILABILITY.md) for
the data, simulator, and software availability statement.

## Citation

The bibliographic record is provided in [CITATION.cff](CITATION.cff).

## License

Released under the [MIT License](LICENSE).
