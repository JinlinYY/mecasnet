# R3 experiment source import

The 20 Python scripts from `experiment(R3)(1).zip` were added to the existing
manuscript-section directories. Each script retains its original filename and
byte-for-byte contents. Existing packaged implementations and experiment scripts
are retained.

## Source-to-repository mapping

Paths in the first column are relative to the archive's `experiment(R3)/` folder.

| Archive folder | Repository destination | Scripts | Experiment |
|---|---|---:|---|
| `01_mechanistic_baselines/` | [main/section_4_03_overall_performance/](main/section_4_03_overall_performance/) | 2 | Mechanistic baselines |
| `02_repeated_splits_and_ablation/01_repeated_splits/` | [main/section_4_04_split_and_topology/](main/section_4_04_split_and_topology/) | 2 | Repeated event splits and NEEQ training |
| `02_repeated_splits_and_ablation/02_architecture_ablation/` | [main/section_4_06_ablation/](main/section_4_06_ablation/) | 1 | Architecture ablation |
| `02_repeated_splits_and_ablation/03_regularizer_ablation/` | [main/section_4_06_ablation/](main/section_4_06_ablation/) | 2 | Regularizer ablation and gradient audit |
| `03_runtime_and_breakeven/` | [main/section_4_09_efficiency/](main/section_4_09_efficiency/) | 4 | Matched runtime, batched simulation, generation cost and break-even |
| `04_network_scope_audit/` | [main/section_4_04_split_and_topology/](main/section_4_04_split_and_topology/) | 1 | Network scope and node/edge count audit |
| `05_share_and_missing_edge_sensitivity/` | [main/section_4_11_uncertainty/](main/section_4_11_uncertainty/) | 2 | Share and missing-edge sensitivity |
| `06_recovery_and_conformal/` | [main/section_4_11_uncertainty/](main/section_4_11_uncertainty/) | 2 | Recovery-floor and independent conformal evaluation |
| `07_repair_screening/` | [main/section_4_08_repair_decision_support/](main/section_4_08_repair_decision_support/) | 3 | Surrogate, oracle and diversity repair screening |
| `08_trough_timing/` | [main/section_4_07_trajectory_reconstruction/](main/section_4_07_trajectory_reconstruction/) | 1 | Trough-timing evaluation |

## Runtime requirements

These files preserve the original research-environment interfaces. They have
not been ported to the installable `mecasnet` package, and installing this
repository alone does not supply their complete runtime environment.

- Many scripts import the original `pi_dl` package, including its `train_v3de`
  and `_runner` modules. `mecasnet` is not asserted to be a drop-in substitute.
- Simulator workflows require the original `fix.sim_env`, `fix.finder_policy`,
  `stage3_ario_henriet2012`, and/or Henriet event-generation modules.
- Supporting modules referenced by the scripts but absent from this archive and
  repository include `y8_reviewer_experiments`, `comment12_common`,
  `independent_conformal_common`, and `evaluate_y8_propagation_depth`.
- Some files are selected procedure extracts with their original mixed experiment
  entry points omitted. They provide helper functions rather than standalone
  command-line programs; supplying dependencies alone does not add those entry
  points.
- Several scripts derive the original project root with fixed `parents[...]`
  offsets and refer to `flagship_y8h_download/`, `flagship_train/`,
  `cangzhou_pipeline/`, `fix/`, and `answer_result/`. These offsets do not locate
  that original environment after placement in this section-based tree.
- To execute these exact source snapshots, restore them to the corresponding
  locations in the original authorized environment or explicitly port their
  path/import setup first. Retain sibling scripts together where one imports
  another. Use explicit data, checkpoint and output arguments where available;
  `answer_result/` defaults belong to the original environment.

No external simulator implementations, datasets, checkpoints or result artifacts
were supplied by this archive. The import was checked for Python syntax and
byte-for-byte agreement with all 20 archive entries; full experiments require
the external modules, authorized data and appropriate hardware.

## Import validation

- All 20 imported files match their corresponding archive entries byte for byte
  and pass Python AST parsing; documentation links resolve.
- `ruff check .` reports 183 findings across existing and imported code. These
  include unused imports, import ordering and unresolved type annotation names
  (`torch` and `GenerationWorker`) in the supplied source. The import preserves
  the supplied files rather than applying lint fixes.
- The repository test suite could not complete collection in the import
  environment because PyTorch (`torch`) is not installed. Full experimental
  execution and numerical reproduction have not been validated by this import.
- The two dependency-independent profile tests pass (`tests/test_profiles.py`).
- Original CRLF line endings and trailing blank lines are retained with the
  source snapshot; Git whitespace checks flag these original formatting details.
