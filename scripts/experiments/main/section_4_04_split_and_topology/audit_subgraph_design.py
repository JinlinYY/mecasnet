"""Pre-outcome audit for three comparable large induced-subgraph perturbations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

def repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Could not locate the repository root")


ROOT = repository_root()
CANGZHOU = ROOT / "cangzhou_pipeline"
for path in (ROOT, CANGZHOU):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stage3_ario_henriet2012 import Params, build_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--min-largest-component-retention", type=float, default=0.80)
    parser.add_argument("--max-added-isolated-nodes", type=int, default=20)
    parser.add_argument("--max-node-retention-spread", type=float, default=0.03)
    parser.add_argument("--max-edge-retention-spread", type=float, default=0.10)
    parser.add_argument("--max-flow-retention-spread", type=float, default=0.10)
    parser.add_argument("--max-bau-retention-spread", type=float, default=0.10)
    parser.add_argument("--min-retained-sector-fraction", type=float, default=0.85)
    parser.add_argument("--min-distinct-cut-strategies", type=int, default=2)
    parser.add_argument("--max-pairwise-removed-node-jaccard", type=float, default=0.25)
    return parser.parse_args()


def component_sizes(adjacency: np.ndarray) -> list[int]:
    neighbours = (adjacency != 0) | (adjacency.T != 0)
    unseen = set(range(adjacency.shape[0]))
    sizes: list[int] = []
    while unseen:
        seed = unseen.pop()
        seen = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbour in np.flatnonzero(neighbours[current]).astype(int):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    seen.add(neighbour)
                    stack.append(neighbour)
        sizes.append(len(seen))
    return sorted(sizes, reverse=True)


def full_connectivity_baseline(full: Any) -> dict[str, int]:
    adjacency = np.asarray(full.A, dtype=float)
    components = component_sizes(adjacency)
    edge_mask = adjacency != 0
    degrees = edge_mask.sum(axis=0) + edge_mask.sum(axis=1)
    return {
        "largest_component_nodes": int(components[0]),
        "isolated_node_count": int((degrees == 0).sum()),
    }


def summarize_block(full: Any, baseline: dict[str, int], manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    removed = set(map(int, manifest["heldout_node_ids"]))
    keep = np.asarray([node for node in range(full.N) if node not in removed], dtype=np.int64)
    removed_indices = np.asarray(sorted(removed), dtype=np.int64)
    adjacency = np.asarray(full.A[np.ix_(keep, keep)], dtype=float)
    removed_adjacency = np.asarray(full.A[np.ix_(removed_indices, removed_indices)], dtype=float)
    full_edge_mask = full.A != 0
    retained_edge_mask = adjacency != 0
    full_flow = np.asarray(full.A, dtype=float) * np.asarray(full.P_ini, dtype=float)[:, None]
    retained_flow = adjacency * np.asarray(full.P_ini, dtype=float)[keep, None]
    components = component_sizes(adjacency)
    degrees = retained_edge_mask.sum(axis=0) + retained_edge_mask.sum(axis=1)
    retained_sectors = {full.sector_of[node] for node in keep}
    return {
        "block": int(manifest["block"]),
        "cut_strategy": manifest.get("cut_strategy"),
        "removed_node_ids": sorted(removed),
        "removed_nodes": len(removed),
        "removed_block_induced_connected": len(component_sizes(removed_adjacency)) == 1,
        "retained_nodes": int(keep.size),
        "retained_node_fraction": float(keep.size / full.N),
        "retained_edges": int(retained_edge_mask.sum()),
        "retained_edge_fraction": float(retained_edge_mask.sum() / full_edge_mask.sum()),
        "retained_flow_fraction": float(retained_flow.sum() / full_flow.sum()),
        "largest_component_nodes": components[0],
        "largest_component_fraction": float(components[0] / keep.size),
        "largest_component_retention": float(components[0] / baseline["largest_component_nodes"]),
        "weak_component_count": len(components),
        "isolated_node_count": int((degrees == 0).sum()),
        "added_isolated_node_count": int((degrees == 0).sum() - baseline["isolated_node_count"]),
        "retained_nonempty_sector_count": len(retained_sectors),
        "full_sector_count": int(full.K),
        "retained_bau_fraction": float(np.asarray(full.P_ini)[keep].sum() / np.asarray(full.P_ini).sum()),
    }


def spread(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return max(values) - min(values)


def node_jaccard(left: list[int], right: list[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    return len(left_set & right_set) / len(left_set | right_set)


def main() -> None:
    args = parse_args()
    full = build_network(Params())
    baseline = full_connectivity_baseline(full)
    if len(args.blocks) < 2 or len(set(args.blocks)) != len(args.blocks):
        raise ValueError("--blocks must contain at least two unique block IDs")
    paths = [args.block_dir / f"heldout_topology_block{block}.json" for block in args.blocks]
    rows = [summarize_block(full, baseline, path) for path in paths]
    pairwise_jaccard = [
        node_jaccard(rows[left]["removed_node_ids"], rows[right]["removed_node_ids"])
        for left in range(len(rows)) for right in range(left + 1, len(rows))
    ]
    checks = {
        "distinct_cut_strategy_count": {
            "value": len({row["cut_strategy"] for row in rows}),
            "required_minimum": args.min_distinct_cut_strategies,
        },
        "max_pairwise_removed_node_jaccard": {
            "value": max(pairwise_jaccard),
            "allowed_maximum": args.max_pairwise_removed_node_jaccard,
        },
        "all_removed_blocks_induced_connected": {
            "value": all(row["removed_block_induced_connected"] for row in rows),
            "required": True,
        },
        "all_largest_component_retention": {
            "value": min(row["largest_component_retention"] for row in rows),
            "required_minimum": args.min_largest_component_retention,
        },
        "max_added_isolated_nodes": {
            "value": max(row["added_isolated_node_count"] for row in rows),
            "allowed_maximum": args.max_added_isolated_nodes,
        },
        "node_retention_spread": {
            "value": spread(rows, "retained_node_fraction"),
            "allowed_maximum": args.max_node_retention_spread,
        },
        "edge_retention_spread": {
            "value": spread(rows, "retained_edge_fraction"),
            "allowed_maximum": args.max_edge_retention_spread,
        },
        "flow_retention_spread": {
            "value": spread(rows, "retained_flow_fraction"),
            "allowed_maximum": args.max_flow_retention_spread,
        },
        "bau_retention_spread": {
            "value": spread(rows, "retained_bau_fraction"),
            "allowed_maximum": args.max_bau_retention_spread,
        },
        "all_retained_sector_fraction": {
            "value": min(
                row["retained_nonempty_sector_count"] / row["full_sector_count"]
                for row in rows
            ),
            "required_minimum": args.min_retained_sector_fraction,
        },
    }
    checks["distinct_cut_strategy_count"]["pass"] = (
        checks["distinct_cut_strategy_count"]["value"]
        >= checks["distinct_cut_strategy_count"]["required_minimum"]
    )
    checks["all_removed_blocks_induced_connected"]["pass"] = (
        checks["all_removed_blocks_induced_connected"]["value"]
        == checks["all_removed_blocks_induced_connected"]["required"]
    )
    checks["all_largest_component_retention"]["pass"] = (
        checks["all_largest_component_retention"]["value"]
        >= checks["all_largest_component_retention"]["required_minimum"]
    )
    checks["all_retained_sector_fraction"]["pass"] = (
        checks["all_retained_sector_fraction"]["value"]
        >= checks["all_retained_sector_fraction"]["required_minimum"]
    )
    for name in (
        "node_retention_spread", "edge_retention_spread", "flow_retention_spread",
        "bau_retention_spread", "max_added_isolated_nodes", "max_pairwise_removed_node_jaccard",
    ):
        checks[name]["pass"] = checks[name]["value"] <= checks[name]["allowed_maximum"]
    status = "PASS" if all(check["pass"] for check in checks.values()) else "FAIL"
    report = {
        "status": status,
        "selection_guard": "static graph and BAU quantities only; no events, labels, or predictions",
        "thresholds_preregistered_before_evaluation": True,
        "full_graph_connectivity_baseline": baseline,
        "blocks": rows,
        "pairwise_removed_node_jaccard": pairwise_jaccard,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit("LARGE_SUBGRAPH_DESIGN_FAIL")
    print("LARGE_SUBGRAPH_DESIGN_PASS")


if __name__ == "__main__":
    main()
