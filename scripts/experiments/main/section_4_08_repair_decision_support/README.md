# Section 4.8: repair-oriented resilience decision support

`repair_experiment.py` uses MeCaSNet to rank candidate repair actions and then
verifies reported outcomes with the full simulator.

```bash
python scripts/experiments/main/section_4_08_repair_decision_support/repair_experiment.py \
  --data-root /path/to/authorized/data \
  --checkpoint /path/to/compat-seed0.pt \
  --finder-checkpoint /path/to/finder_henriet.pt \
  --budget 10 \
  --output runs/main/section_4_8/repair-comparison.json
```

Running this analysis requires authorized ARIO/Henriet and FINDER interfaces.
The surrogate ranks actions; it does not replace the simulator used to verify
repair outcomes.

## R3 experiment scripts

Additional scripts from the updated experiment archive:

| Script | Purpose |
|---|---|
| [run_repair_diversity_experiment.py](run_repair_diversity_experiment.py) | Surrogate, oracle and diversity repair screening |
| [run_repair_oracle_experiment.py](run_repair_oracle_experiment.py) | Surrogate, oracle and diversity repair screening |
| [run_surrogate_repair_experiment.py](run_surrogate_repair_experiment.py) | Surrogate, oracle and diversity repair screening |

These source snapshots retain their original research-environment imports and
paths. See [R3 import requirements](../../R3_IMPORT.md) before running them.
