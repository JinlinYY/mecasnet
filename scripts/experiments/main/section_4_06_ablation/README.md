# Section 4.6: ablation study

The ablation implementation is distributed across the named model variants in
`src/mecasnet/train.py`, the profile controls in `src/mecasnet/factory.py`, and
the architecture flags consumed by `src/mecasnet/model_v3.py`.

Each ablation is defined by a named model variant together with its effective
configuration. The `paper` profile resets architecture overrides to the article
definition, so an ablation command must specify its changed factor explicitly.

For a new ablation study:

1. copy the full-model command from Section 4.3;
2. change exactly one named variant or configuration factor;
3. keep event IDs, five seeds, optimizer, epochs, and selection metric fixed;
4. store the effective configuration and parameter count with the result;
5. report the variant name, changed factor, seed set, and parameter count.

This convention makes every ablation independently inspectable and prevents a
generic training variant from being mistaken for a reported experiment.
