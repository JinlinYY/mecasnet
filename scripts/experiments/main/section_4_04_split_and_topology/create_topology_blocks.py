"""Search static, economically comparable connected held-out graph blocks.

The search deliberately uses only the fixed graph and its static production/flow
metadata. It does not inspect simulated events, labels, model outputs, or
training results. Each accepted block is an induced connected node set suitable
for evaluating transfer to unseen directly shocked graph locations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-meta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--target-nodes", type=int, default=82)
    parser.add_argument("--min-nodes", type=int, default=65)
    parser.add_argument("--max-nodes", type=int, default=110)
    parser.add_argument("--target-bau-fraction", type=float, default=0.10)
    parser.add_argument("--min-bau-fraction", type=float, default=0.03)
    parser.add_argument("--max-bau-fraction", type=float, default=0.20)
    parser.add_argument("--max-single-node-bau-fraction", type=float, default=0.05,
                        help="Reject blocks containing a firm above this global BAU share.")
    parser.add_argument("--min-internal-flow-fraction", type=float, default=0.10,
                        help="Minimum internal-flow share among internal plus boundary flow.")
    parser.add_argument("--restarts", type=int, default=20000)
    return parser.parse_args()


def sha256_ids(values: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, values)).encode("ascii")).hexdigest()


def block_score(node_count: int, bau_fraction: float, args: argparse.Namespace) -> float:
    node_error = (node_count - args.target_nodes) / max(args.target_nodes, 1)
    bau_error = (bau_fraction - args.target_bau_fraction) / max(args.target_bau_fraction, 1e-12)
    return node_error * node_error + bau_error * bau_error


def make_adjacency(node_count: int, src: np.ndarray, dst: np.ndarray) -> list[np.ndarray]:
    neighbours: list[set[int]] = [set() for _ in range(node_count)]
    for source, destination in zip(src.astype(int), dst.astype(int)):
        if source != destination:
            neighbours[source].add(destination)
            neighbours[destination].add(source)
    return [np.fromiter(sorted(items), dtype=np.int64) for items in neighbours]


def grow_block(available: np.ndarray, neighbours: list[np.ndarray], bau: np.ndarray,
               rng: np.random.Generator, args: argparse.Namespace) -> list[int] | None:
    max_node_bau = args.max_single_node_bau_fraction * float(bau.sum())
    available = available[bau[available] <= max_node_bau]
    available_set = set(available.astype(int).tolist())
    if len(available_set) < args.min_nodes:
        return None
    weights = bau[available]
    seed = int(rng.choice(available, p=weights / weights.sum()))
    selected = {seed}
    frontier = set(int(node) for node in neighbours[seed] if int(node) in available_set)
    target_bau = args.target_bau_fraction * float(bau.sum())

    while frontier and len(selected) < args.max_nodes:
        candidates = np.fromiter(frontier, dtype=np.int64)
        candidate_bau = bau[candidates]
        current_bau = float(bau[list(selected)].sum())
        # Prefer candidates that bring the block toward its static BAU target,
        # while retaining stochasticity across restarts.
        score = np.abs((current_bau + candidate_bau) - target_bau) / max(target_bau, 1e-12)
        rank = np.argsort(score)
        rank_position = int(rng.integers(min(rank.size, 12)))
        choice = int(candidates[rank[rank_position]])
        selected.add(choice)
        frontier.remove(choice)
        frontier.update(int(node) for node in neighbours[choice]
                        if int(node) in available_set and int(node) not in selected)

        node_count = len(selected)
        bau_fraction = float(bau[list(selected)].sum() / bau.sum())
        if (node_count >= args.min_nodes and bau_fraction >= args.min_bau_fraction
                and bau_fraction <= args.max_bau_fraction):
            return sorted(selected)
    return None


def connected(nodes: list[int], neighbours: list[np.ndarray]) -> bool:
    node_set = set(nodes)
    visited = {nodes[0]}
    stack = [nodes[0]]
    while stack:
        current = stack.pop()
        for neighbour in neighbours[current]:
            neighbour_int = int(neighbour)
            if neighbour_int in node_set and neighbour_int not in visited:
                visited.add(neighbour_int)
                stack.append(neighbour_int)
    return visited == node_set


def summarize(nodes: list[int], bau: np.ndarray, src: np.ndarray, dst: np.ndarray,
              flow: np.ndarray, neighbours: list[np.ndarray]) -> dict[str, Any]:
    node_set = set(nodes)
    inside_src = np.isin(src, nodes)
    inside_dst = np.isin(dst, nodes)
    internal = inside_src & inside_dst
    boundary = inside_src ^ inside_dst
    inbound = (~inside_src) & inside_dst
    outbound = inside_src & (~inside_dst)
    degrees = np.asarray([len(neighbours[node]) for node in nodes], dtype=float)
    internal_flow = float(flow[internal].sum())
    inbound_flow = float(flow[inbound].sum())
    outbound_flow = float(flow[outbound].sum())
    incident_flow = internal_flow + inbound_flow + outbound_flow
    return {
        "heldout_node_ids": nodes,
        "heldout_node_count": len(nodes),
        "heldout_node_fraction": len(nodes) / bau.size,
        "heldout_bau_output": float(bau[nodes].sum()),
        "heldout_bau_output_fraction": float(bau[nodes].sum() / bau.sum()),
        "induced_connected": connected(nodes, neighbours),
        "internal_edge_count": int(internal.sum()),
        "boundary_edge_count": int(boundary.sum()),
        "inbound_boundary_edge_count": int(inbound.sum()),
        "outbound_boundary_edge_count": int(outbound.sum()),
        "internal_flow": internal_flow,
        "inbound_boundary_flow": inbound_flow,
        "outbound_boundary_flow": outbound_flow,
        "internal_flow_fraction": float(internal_flow / incident_flow) if incident_flow > 0 else 0.0,
        "mean_undirected_degree": float(degrees.mean()),
        "median_undirected_degree": float(np.median(degrees)),
        "max_single_node_bau_fraction": float(bau[nodes].max() / bau.sum()),
    }


def main() -> None:
    args = parse_args()
    if not (0 < args.min_nodes <= args.target_nodes <= args.max_nodes):
        raise ValueError("Require 0 < min-nodes <= target-nodes <= max-nodes")
    if not (0 < args.min_bau_fraction <= args.target_bau_fraction <= args.max_bau_fraction < 1):
        raise ValueError("Invalid BAU fraction bounds")
    if not (0 < args.max_single_node_bau_fraction < 1):
        raise ValueError("--max-single-node-bau-fraction must lie in (0, 1)")
    if not (0 < args.min_internal_flow_fraction <= 1):
        raise ValueError("--min-internal-flow-fraction must lie in (0, 1]")
    with Path(args.static_meta).open("rb") as handle:
        static = pickle.load(handle)
    bau = np.asarray(static["P_ini"], dtype=float)
    src = np.asarray(static["edge_src"], dtype=np.int64)
    dst = np.asarray(static["edge_dst"], dtype=np.int64)
    flow = np.asarray(static["A"], dtype=float)
    if bau.ndim != 1 or bau.size == 0 or bau.sum() <= 0:
        raise ValueError("static_meta P_ini must be a positive one-dimensional array")
    neighbours = make_adjacency(bau.size, src, dst)
    rng = np.random.default_rng(args.seed)
    best: list[list[int]] | None = None
    best_score = float("inf")

    for restart in range(1, args.restarts + 1):
        available = np.arange(bau.size, dtype=np.int64)
        blocks: list[list[int]] = []
        for _ in range(args.n_blocks):
            block = grow_block(available, neighbours, bau, rng, args)
            if block is None:
                break
            candidate_summary = summarize(block, bau, src, dst, flow, neighbours)
            if candidate_summary["internal_flow_fraction"] < args.min_internal_flow_fraction:
                break
            blocks.append(block)
            available = np.asarray([node for node in available if node not in set(block)], dtype=np.int64)
        if len(blocks) != args.n_blocks:
            continue
        score = sum(block_score(len(block), float(bau[block].sum() / bau.sum()), args)
                    for block in blocks)
        if score < best_score:
            best, best_score = blocks, score
        if restart % 1000 == 0 or restart == args.restarts:
            result = f"{best_score:.6f}" if best is not None else "no feasible design"
            print(f"[{time.strftime('%H:%M:%S')}] search {restart:,}/{args.restarts:,}; best_score={result}", flush=True)

    if best is None:
        raise RuntimeError("No connected, disjoint block design met the static bounds.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    used_nodes: set[int] = set()
    for index, nodes in enumerate(best, 1):
        if used_nodes & set(nodes):
            raise RuntimeError("Connected blocks unexpectedly overlap")
        used_nodes.update(nodes)
        summary = summarize(nodes, bau, src, dst, flow, neighbours)
        manifest = {
            "definition": {
                "design": "three disjoint topology-aware held-out connected node blocks; not unseen topology transfer",
                "selection_guard": "static node membership, fixed graph edges, P_ini BAU output, and static edge flow only; no events, labels, predictions, or metrics",
                "train_validation_rule": "all directly shocked firms must be outside the held-out block",
                "strict_test_rule": "all directly shocked firms must be inside the held-out block",
                "claim_scope": "generalization to unseen directly shocked subgraphs/locations on the same fixed graph",
                "selection_bounds": {
                    "min_nodes": args.min_nodes,
                    "max_nodes": args.max_nodes,
                    "min_bau_output_fraction": args.min_bau_fraction,
                    "max_bau_output_fraction": args.max_bau_fraction,
                    "max_single_node_bau_fraction": args.max_single_node_bau_fraction,
                    "min_internal_flow_fraction": args.min_internal_flow_fraction,
                },
            },
            "partition_seed": args.seed,
            "block": index,
            **summary,
            "id_sha256": {"heldout_nodes": sha256_ids(nodes)},
        }
        path = output_dir / f"heldout_topology_block{index}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        summaries.append({"block": index, "manifest": path.name, **summary})

    report = {
        "design": "static topology-aware held-out connected node blocks",
        "selection_method": "stochastic connected expansion using static BAU-guided frontier ranking",
        "partition_seed": args.seed,
        "restarts": args.restarts,
        "optimization_score": best_score,
        "parameters": vars(args),
        "network": {"nodes": int(bau.size), "edges": int(src.size)},
        "blocks": summaries,
        "checks": {
            "blocks_disjoint": True,
            "blocks_connected": all(summary["induced_connected"] for summary in summaries),
            "blocks_not_exhaustive_by_design": len(used_nodes) < bau.size,
        },
    }
    path = output_dir / "topology_heldout_blocks_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Created: {path}")
    for summary in summaries:
        print(f"Block {summary['block']}: nodes={summary['heldout_node_count']}; "
              f"BAU={100 * summary['heldout_bau_output_fraction']:.2f}%; "
              f"internal_flow_fraction={100 * summary['internal_flow_fraction']:.1f}%; "
              f"boundary_edges={summary['boundary_edge_count']}")


if __name__ == "__main__":
    main()
