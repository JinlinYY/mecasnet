# Article-aligned experiments

This tree maps code to the experimental structure of the manuscript and
Supplementary Information (SI):

- [main/](main/README.md): main-manuscript Sections 4.1–4.11;
- [si/](si/README.md): Supplementary Notes S1–S7.

Each section README identifies its implementation entry points, expected
outputs, and any required external datasets or simulator interfaces.

## Execution conventions

1. Run commands from the repository root.
2. Install with `python -m pip install -e ".[dev,analysis]"`.
3. Use `--profile paper` for new manuscript-profile training.
4. Use `compat` only for checkpoints that require the analysis configuration.
5. Store data outside the repository and outputs under `runs/`.

The section READMEs identify required datasets, checkpoint profiles, and
external simulator interfaces.
