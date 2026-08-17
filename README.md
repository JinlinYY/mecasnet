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

## Training the manuscript profile

```bash
mecasnet-train \
  --profile paper \
  --variant MeCaSNet \
  --data-root /path/to/authorized/chemical-cascade-data \
  --n-train 4000 --n-val 500 --n-test 500 \
  --epochs 60 --warmup-epochs 3 \
  --lr-schedule cosine --min-lr-ratio 0.1 \
  --event-scalars-mode minimal \
  --seeds 5 --seed-start 0 --save-ckpt \
  --out-dir runs/chemical/mecasnet
```

The `paper` profile fixes the manuscript boundary and architecture: Day-0-only
event information, `u(0) = 1 - shock * damage`, four persistent edge-state
propagation steps, a three-component lower envelope with `c = 6`, learned
node-specific recovery exponent `q` in `[0.5, 2.0]`, and three-stream fusion.

To train a comparator under the same split and optimization protocol, replace
`MeCaSNet` with `MLP`, `GCN`, `GAT`, `STGNN`, or `DirGNN`.

The `legacy-y8` profile exists only to load archived pre-final-manuscript
checkpoints. It should not be presented as the manuscript architecture.

## Analysis workflows

| Manuscript/SI analysis | Entry point |
|---|---|
| Cascade-threshold sweep and runtime decomposition | `mecasnet-thresholds` |
| Matched-seed baseline evaluation | `scripts/evaluation/threshold_multiseed.py` |
| Input uncertainty | `scripts/evaluation/input_uncertainty.py` |
| Ensemble and conformal uncertainty | `scripts/evaluation/predictive_uncertainty.py` |
| Learned recovery parameters | `scripts/evaluation/recovery_parameters.py` |
| Propagation depth and feedback position | `scripts/evaluation/propagation_depth.py` and `network_propagation.py` |
| Cross-split event similarity | `scripts/audits/event_split.py` and `event_similarity.py` |
| Static topology blocks | `scripts/audits/create_topology_blocks.py` |
| Induced-subgraph evaluation | `scripts/topology/evaluate_subgraph.py` |

Scripts under `scripts/reference/` and the topology-generation scripts requiring
the ARIO/Henriet backend are audit records of the manuscript workflow. They are
not standalone public reproductions because the simulator backend is not part
of this release.

Detailed command examples and interpretation limits are in
[docs/reproducibility.md](docs/reproducibility.md). A section-by-section mapping
from the manuscript and SI to source files is in
[docs/method_to_code.md](docs/method_to_code.md).

## Repository layout

```text
src/mecasnet/        Installable model, data, loss, training, and evaluation code
scripts/evaluation/  Post-training metrics and uncertainty analyses
scripts/audits/      Split and topology audit utilities
scripts/topology/    Topology-shift search/evaluation workflows
scripts/reference/   Workflows requiring non-public simulator interfaces
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

See [OPEN_SOURCE_SCOPE.md](OPEN_SOURCE_SCOPE.md) for the complete release and
data-governance boundary.

## Citation

The bibliographic record is provided in [CITATION.cff](CITATION.cff). Replace
the provisional journal metadata with the final DOI after publication.

## License

Released under the [MIT License](LICENSE).
