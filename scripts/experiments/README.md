# Article-aligned experiments

This tree maps code to the experimental structure of the manuscript and
Supplementary Information (SI):

- [main/](main/README.md): main-manuscript Sections 4.1–4.11;
- [si/](si/README.md): Supplementary Notes S1–S7.

Every section directory contains a README, even when the implementation lives
in `src/mecasnet/` rather than in a standalone script. This makes absent public
assets explicit instead of representing them with an empty or misleading
runner.

## Execution conventions

1. Run commands from the repository root.
2. Install with `python -m pip install -e ".[dev,analysis]"`.
3. Use `--profile paper` for new manuscript-profile training.
4. Treat scripts that reconstruct `legacy-y8` as archived-checkpoint analyses.
5. Store data outside the repository and outputs under `runs/`.

The section READMEs label workflows that require authorized datasets, archived
checkpoints, or non-public simulator interfaces.
