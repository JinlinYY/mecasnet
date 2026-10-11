# Supplementary Note S1: NEEQ Supply-Chain Benchmark Dataset

This note describes the supply-network construction, Inoue-style simulation
and event characterization of the NEEQ benchmark. Figure S1 summarizes event
quality and topology. The benchmark supplies a second network and simulator
domain for separately trained cascade-prediction models.

The available [data definitions](../../../../docs/data_contract.md) describe
firm ordering, supplier-to-customer edge direction, input shares, static
features, event targets and key dates. Scientific reproduction requires the
corresponding firm records, network reconstruction and simulation conditions.
Those underlying records and simulator implementations are subject to the
[data availability statement](../../../../DATA_AND_SOFTWARE_AVAILABILITY.md).

Use a consistent firm ordering across the static network and all events.
Record the sector definitions, event-generation settings and the distinction
between full-daily peak loss and key-date production ratios. Prediction scope
and full-network populations must be identified when reporting event counts
or descriptive statistics.
