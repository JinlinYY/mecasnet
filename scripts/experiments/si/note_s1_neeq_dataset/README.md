# Supplementary Note S1: NEEQ benchmark dataset

The public repository begins after a network and event pool have been exported
to the schema in `docs/data_contract.md`. It does not reconstruct NEEQ firms,
transactions, or simulator events.

For an authorized NEEQ export, validate:

- one stable zero-based node order across static and event arrays;
- supplier-to-customer edge orientation;
- sector vocabulary and checkpoint feature width;
- key dates and peak-loss definition;
- simulator/generator commit, seeds, and file hashes.

The schema smoke example in `docs/data_contract.md` can be used before loading
the protected benchmark.
