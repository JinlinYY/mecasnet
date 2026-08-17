"""Fail-fast audit for topology-block-constrained Henriet event generation."""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


TIER_TARGET_RANGES = {
    "medium": (2, 8),
    "heavy": (5, 15),
    "catastrophic": (8, 30),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--node-block-manifest", required=True)
    parser.add_argument("--expected-scope", choices=["nonheldout", "heldout"], required=True)
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--expected-modes", default="cluster,hub,random")
    return parser.parse_args()


def connected(nodes: np.ndarray, src: np.ndarray, dst: np.ndarray) -> bool:
    if nodes.size == 0:
        return False
    node_set = set(nodes.astype(int).tolist())
    neighbours: dict[int, set[int]] = {node: set() for node in node_set}
    for source, destination in zip(src.astype(int), dst.astype(int)):
        if source in node_set and destination in node_set:
            neighbours[source].add(destination)
            neighbours[destination].add(source)
    seen = {int(nodes[0])}
    stack = [int(nodes[0])]
    while stack:
        current = stack.pop()
        for neighbour in neighbours[current]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen == node_set


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    manifest = json.loads(Path(args.node_block_manifest).read_text(encoding="utf-8"))
    heldout = set(map(int, manifest["heldout_node_ids"]))
    expected_modes = tuple(item.strip() for item in args.expected_modes.split(",") if item.strip())
    if not expected_modes or len(set(expected_modes)) != len(expected_modes):
        raise ValueError("--expected-modes must contain distinct modes")

    with (data_root / "static_meta.pkl").open("rb") as handle:
        static = pickle.load(handle)
    node_count = int(static["V"])
    if any(node < 0 or node >= node_count for node in heldout):
        raise ValueError("Node-block manifest has invalid node IDs for static_meta")
    src = np.asarray(static["edge_src"], dtype=np.int64)
    dst = np.asarray(static["edge_dst"], dtype=np.int64)
    paths = sorted((data_root / "events").glob("event_*.npz"))
    if len(paths) != args.expected_events:
        raise ValueError(f"Expected {args.expected_events} events, found {len(paths)}")

    violations: list[dict[str, Any]] = []
    mode_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    tier_mode_counts: Counter[str] = Counter()
    target_counts: list[int] = []
    cluster_event_count = 0
    for path in paths:
        with np.load(path, allow_pickle=True) as event:
            targets = np.flatnonzero(np.asarray(event["shock_mask"]) > 0).astype(np.int64)
            metadata = event["shock_meta"].item()
        mode = str(metadata.get("mode"))
        tier = str(metadata.get("tier"))
        valid_scope = (set(targets.astype(int).tolist()).isdisjoint(heldout)
                       if args.expected_scope == "nonheldout"
                       else set(targets.astype(int).tolist()) <= heldout)
        if targets.size == 0 or not valid_scope:
            violations.append({"event": path.name, "reason": "targets violate node-block scope"})
        if metadata.get("target_scope") != args.expected_scope:
            violations.append({"event": path.name, "reason": "target_scope metadata mismatch"})
        if mode not in expected_modes:
            violations.append({"event": path.name, "reason": f"unexpected mode={mode!r}"})
        if tier not in TIER_TARGET_RANGES:
            violations.append({"event": path.name, "reason": f"unexpected tier={tier!r}"})
        else:
            lower, upper = TIER_TARGET_RANGES[tier]
            if not lower <= targets.size <= upper:
                violations.append({
                    "event": path.name,
                    "reason": f"target count {targets.size} outside {tier} range [{lower}, {upper}]",
                })
        if mode == "cluster":
            cluster_event_count += 1
            if not connected(targets, src, dst):
                violations.append({"event": path.name, "reason": "cluster targets are not connected"})
        mode_counts[mode] += 1
        tier_counts[tier] += 1
        tier_mode_counts[f"{tier}|{mode}"] += 1
        target_counts.append(int(targets.size))

    expected_combinations = {f"{tier}|{mode}" for tier in TIER_TARGET_RANGES for mode in expected_modes}
    missing_combinations = sorted(expected_combinations - set(tier_mode_counts))
    if missing_combinations:
        violations.append({"reason": f"missing tier/mode combinations: {missing_combinations}"})
    if violations:
        raise ValueError(f"Topology-block audit failed ({len(violations)} violations): {violations[:10]}")

    report = {
        "status": "PASS",
        "expected_scope": args.expected_scope,
        "event_count": len(paths),
        "heldout_node_count": len(heldout),
        "expected_modes": list(expected_modes),
        "mode_counts": dict(sorted(mode_counts.items())),
        "tier_counts": dict(sorted(tier_counts.items())),
        "tier_mode_counts": dict(sorted(tier_mode_counts.items())),
        "cluster_event_count": cluster_event_count,
        "target_count": {
            "min": min(target_counts),
            "median": float(np.median(target_counts)),
            "max": max(target_counts),
        },
        "constraint_violations": 0,
    }
    output = data_root / "topology_block_constraint_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
