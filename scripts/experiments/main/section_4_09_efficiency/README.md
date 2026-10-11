# Section 4.9: computational efficiency

`runtime_benchmark.py` separates startup cost, warm-up, surrogate inference,
and simulator throughput for the Henriet and Inoue domains.

Audit expected assets without running the benchmark:

```bash
python scripts/experiments/main/section_4_09_efficiency/runtime_benchmark.py \
  --audit-only \
  --output runs/main/section_4_9/asset-audit.json
```

Run an authorized benchmark with:

```bash
python scripts/experiments/main/section_4_09_efficiency/runtime_benchmark.py \
  --domains henriet inoue \
  --surrogate-events 20 --simulator-events 5 --warmup-forwards 3 \
  --device cuda \
  --output runs/main/section_4_9/runtime.json
```

The simulator branches require authorized interfaces and protected domain
assets. Runtime comparisons must report hardware, event count, warm-up,
precision, and whether startup and I/O are included.

## R3 experiment scripts

Additional scripts from the updated experiment archive:

| Script | Purpose |
|---|---|
| [benchmark_henriet_batched_breakeven.py](benchmark_henriet_batched_breakeven.py) | Matched runtime, batched simulation, generation cost and break-even |
| [benchmark_henriet_matched_workflow.py](benchmark_henriet_matched_workflow.py) | Matched runtime, batched simulation, generation cost and break-even |
| [calculate_henriet_breakeven.py](calculate_henriet_breakeven.py) | Matched runtime, batched simulation, generation cost and break-even |
| [measure_henriet_generation_cost.py](measure_henriet_generation_cost.py) | Matched runtime, batched simulation, generation cost and break-even |

These source snapshots retain their original research-environment imports and
paths. See [R3 import requirements](../../R3_IMPORT.md) before running them.
