"""Map the largest feasible connected node-removal level using static constraints only."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANGZHOU = Path(os.environ.get("CANGZHOU_PIPELINE_ROOT", ROOT / "cangzhou_pipeline"))
for path in (ROOT, CANGZHOU):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import search_induced_subgraphs as search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static feasibility envelope for connected induced-subgraph removals."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[60, 65, 70, 75, 80, 85, 90, 95, 100])
    parser.add_argument("--attempts-per-strategy", type=int, default=30000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--min-largest-component-retention", type=float, default=0.80)
    parser.add_argument("--max-added-isolated-nodes", type=int, default=20)
    parser.add_argument("--min-retained-sector-fraction", type=float, default=0.85)
    return parser.parse_args()


def level_args(base: argparse.Namespace, level: int) -> argparse.Namespace:
    return argparse.Namespace(
        min_removed_nodes=level,
        target_removed_nodes=level,
        max_removed_nodes=level,
        attempts_per_strategy=base.attempts_per_strategy,
        candidates_per_strategy=1,
        min_largest_component_retention=base.min_largest_component_retention,
        max_added_isolated_nodes=base.max_added_isolated_nodes,
        min_retained_sector_fraction=base.min_retained_sector_fraction,
    )


def compact_summary(candidate: search.Candidate) -> dict[str, float | int | str]:
    summary = candidate.summary
    return {
        "strategy": candidate.strategy,
        "removed_nodes": summary["heldout_node_count"],
        "largest_component_retention": summary["largest_component_retention"],
        "added_isolated_node_count": summary["added_isolated_node_count"],
        "retained_edge_fraction": summary["retained_edge_fraction"],
        "retained_flow_fraction": summary["retained_flow_fraction"],
        "retained_bau_fraction": summary["retained_bau_fraction"],
        "retained_sector_fraction": summary["retained_nonempty_sector_count"] / summary["full_sector_count"],
    }


def search_task(
    level: int,
    level_index: int,
    strategy: str,
    strategy_index: int,
    args: argparse.Namespace,
) -> tuple[int, str, list[search.Candidate]]:
    full = search.build_network(search.Params())
    neighbours = search.undirected_neighbours(search.np.asarray(full.A, dtype=float))
    flow = search.np.asarray(full.A, dtype=float) * search.np.asarray(full.P_ini, dtype=float)[:, None]
    baseline = search.full_connectivity_baseline(full)
    candidates = search.collect_candidates(
        full, neighbours, flow, baseline, strategy, level_args(args, level),
        args.seed + 1000 * level_index + strategy_index,
    )
    return level, strategy, candidates


def main() -> None:
    args = parse_args()
    levels = sorted(set(args.levels))
    if not levels or levels[0] < 1:
        raise ValueError("--levels must contain positive integers")
    if args.attempts_per_strategy < 1:
        raise ValueError("--attempts-per-strategy must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    full = search.build_network(search.Params())
    baseline = search.full_connectivity_baseline(full)
    if any(level >= full.N for level in levels):
        raise ValueError(f"Removal levels must be below N={full.N}")
    tasks = [
        (level, level_index, strategy, strategy_index, args)
        for level_index, level in enumerate(levels)
        for strategy_index, strategy in enumerate(search.STRATEGIES)
    ]
    groups_by_level = {level: {} for level in levels}
    print(json.dumps({
        "parallel_tasks": len(tasks),
        "workers": min(args.workers, len(tasks)),
        "attempts_per_level_per_strategy": args.attempts_per_strategy,
    }, indent=2), flush=True)
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = [executor.submit(search_task, *task) for task in tasks]
        for future in as_completed(futures):
            level, strategy, candidates = future.result()
            groups_by_level[level][strategy] = candidates
            print(
                f"[complete] removed_nodes={level}; strategy={strategy}; "
                f"witnesses={len(candidates)}",
                flush=True,
            )
    rows = []
    for level in levels:
        groups = groups_by_level[level]
        candidates = [candidate for group in groups.values() for candidate in group]
        row = {
            "removed_nodes": level,
            "removed_node_fraction": level / full.N,
            "witness_candidate_count_by_strategy": {
                strategy: len(group) for strategy, group in groups.items()
            },
            "witness_candidate_count": len(candidates),
            "feasible": bool(candidates),
            "best_candidates": [compact_summary(candidate) for candidate in candidates],
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    feasible_levels = [row["removed_nodes"] for row in rows if row["feasible"]]
    report = {
        "status": "STATIC_FEASIBILITY_ENVELOPE_COMPLETE",
        "selection_guard": "static graph and BAU quantities only; no events, labels, checkpoints, predictions, or metrics",
        "network": {"nodes": full.N, "baseline": baseline},
        "fixed_hard_constraints": {
            "removed_block_induced_connected": True,
            "min_largest_component_retention": args.min_largest_component_retention,
            "max_added_isolated_nodes": args.max_added_isolated_nodes,
            "min_retained_sector_fraction": args.min_retained_sector_fraction,
        },
        "search_budget_per_level_per_strategy": args.attempts_per_strategy,
        "parallel_workers": min(args.workers, len(tasks)),
        "levels": rows,
        "largest_feasible_tested_removal_level": max(feasible_levels) if feasible_levels else None,
        "interpretation": (
            "Largest feasible tested level is a multi-start search result under fixed static constraints, "
            "not a mathematical proof that no other graph subset exists."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "deletion_feasibility_envelope.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "largest_feasible_tested_removal_level": report["largest_feasible_tested_removal_level"],
        "output": str(output),
    }, indent=2), flush=True)
    print("STATIC_FEASIBILITY_ENVELOPE_COMPLETE")


if __name__ == "__main__":
    main()
