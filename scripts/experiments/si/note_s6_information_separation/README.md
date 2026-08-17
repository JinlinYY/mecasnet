# Supplementary Note S6: training–inference separation

The predictive input boundary is enforced by:

- `src/mecasnet/data.py`: constructs the 12-slot masked event descriptor;
- `src/mecasnet/factory.py`: locks the paper profile to minimal Day-0 inputs;
- `src/mecasnet/model_v3.py`: forward prediction consumes only graph/static,
  shock, Day-0 damage, event descriptor, and key-date information;
- `src/mecasnet/losses.py`: introduces target-derived supervision only after
  the forward prediction exists.

Regression tests are in `tests/test_profiles.py`, `tests/test_data_contract.py`,
and `tests/test_model_smoke.py`. See `docs/data_contract.md` and
`docs/method_to_code.md` for the tensor-level audit.

True trajectories, peak labels, reach labels, trough labels, simulator recovery
parameters, shock mode, and severity tier are not paper-profile inference inputs.
