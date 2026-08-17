# Section 4.6: ablation study

The ablation implementation is distributed across the named model variants in
`src/mecasnet/train.py`, the profile controls in `src/mecasnet/factory.py`, and
the architecture flags consumed by `src/mecasnet/model_v3.py`.

There is no public one-command runner that exactly regenerates every row of the
submitted ablation table. The `paper` profile deliberately resets experimental
architecture overrides to protect the final model definition. Consequently,
generic CLI variants must not be relabeled as manuscript ablations without an
explicit row-to-configuration audit.

For a new ablation study:

1. copy the full-model command from Section 4.3;
2. change exactly one named variant or configuration factor;
3. keep event IDs, five seeds, optimizer, epochs, and selection metric fixed;
4. store the effective configuration and parameter count with the result;
5. report the run as a new ablation unless it matches the archived row exactly.

This directory exists to make the absence of an exact public matrix runner
visible rather than hiding it among general training utilities.
