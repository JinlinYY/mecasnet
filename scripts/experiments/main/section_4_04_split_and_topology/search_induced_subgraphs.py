"""Find static, connected removal blocks for induced-subgraph perturbation tests.

The search never reads events, labels, checkpoints, or metrics.  It searches three
connected-cut constructions, and accepts a triplet only when both the removed
blocks and each retained induced graph satisfy preregistered static constraints.
The resulting manifests are compatible with build_large_subgraph_events.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
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


STRATEGIES = (
    "random_frontier",
    "connectivity_preserving_compact",
    "low_flow_frontier",
)


@dataclass(frozen=True)
class Candidate:
    strategy: str
    nodes: tuple[int, ...]
    summary: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search matched connected node-removal blocks using static graph quantities only."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--attempts-per-strategy", type=int, default=30000)
    parser.add_argument("--candidates-per-strategy", type=int, default=120)
    parser.add_argument("--target-removed-nodes", type=int, default=65)
    parser.add_argument("--min-removed-nodes", type=int, default=50)
    parser.add_argument("--max-removed-nodes", type=int, default=80)
    parser.add_argument("--min-largest-component-retention", type=float, default=0.80)
    parser.add_argument("--max-added-isolated-nodes", type=int, default=20)
    parser.add_argument("--max-node-retention-spread", type=float, default=0.03)
    parser.add_argument("--max-edge-retention-spread", type=float, default=0.10)
    parser.add_argument("--max-flow-retention-spread", type=float, default=0.10)
    parser.add_argument("--max-bau-retention-spread", type=float, default=0.10)
    parser.add_argument("--min-retained-sector-fraction", type=float, default=0.85)
    parser.add_argument("--representatives-per-strategy", type=int, default=2)
    parser.add_argument("--representative-min-removed-nodes", type=int, default=55)
    parser.add_argument("--require-disjoint-blocks", action="store_true", default=True)
    parser.add_argument("--allow-overlapping-blocks", dest="require_disjoint_blocks", action="store_false")
    return parser.parse_args()


def undirected_neighbours(adjacency: np.ndarray) -> list[np.ndarray]:
    mask = (adjacency != 0) | (adjacency.T != 0)
    return [np.flatnonzero(mask[node]).astype(np.int64) for node in range(mask.shape[0])]


def is_connected(nodes: set[int], neighbours: list[np.ndarray]) -> bool:
    if not nodes:
        return False
    seen = {next(iter(nodes))}
    stack = list(seen)
    while stack:
        node = stack.pop()
        for neighbour in neighbours[node]:
            value = int(neighbour)
            if value in nodes and value not in seen:
                seen.add(value)
                stack.append(value)
    return seen == nodes


def component_sizes(adjacency: np.ndarray) -> list[int]:
    return_sizes: list[int] = []
    for_start = set(range(adjacency.shape[0]))
    neighbours = (adjacency != 0) | (adjacency.T != 0)
    while for_start:
        seed = for_start.pop()
        component = {seed}
        stack = [seed]
        while stack:
            node = stack.pop()
            for neighbour in np.flatnonzero(neighbours[node]).astype(int):
                if neighbour in for_start:
                    for_start.remove(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
        return_sizes.append(len(component))
    return sorted(return_sizes, reverse=True)


def full_connectivity_baseline(full: Any) -> dict[str, int]:
    adjacency = np.asarray(full.A, dtype=float)
    components = component_sizes(adjacency)
    edge_mask = adjacency != 0
    degrees = edge_mask.sum(axis=0) + edge_mask.sum(axis=1)
    return {
        "largest_component_nodes": int(components[0]),
        "isolated_node_count": int((degrees == 0).sum()),
    }


def summarize(
    full: Any, nodes: set[int], neighbours: list[np.ndarray], baseline: dict[str, int],
) -> dict[str, Any]:
    keep = np.asarray([node for node in range(full.N) if node not in nodes], dtype=np.int64)
    retained = np.asarray(full.A[np.ix_(keep, keep)], dtype=float)
    full_edges = np.asarray(full.A != 0)
    retained_edges = retained != 0
    full_flow = np.asarray(full.A, dtype=float) * np.asarray(full.P_ini, dtype=float)[:, None]
    retained_flow = retained * np.asarray(full.P_ini, dtype=float)[keep, None]
    retained_components = component_sizes(retained)
    degrees = retained_edges.sum(axis=0) + retained_edges.sum(axis=1)
    removed_sorted = sorted(nodes)
    retained_sectors = {full.sector_of[node] for node in keep}
    return {
        "heldout_node_ids": removed_sorted,
        "heldout_node_count": len(removed_sorted),
        "heldout_node_fraction": float(len(removed_sorted) / full.N),
        "heldout_induced_connected": is_connected(nodes, neighbours),
        "retained_nodes": int(keep.size),
        "retained_node_fraction": float(keep.size / full.N),
        "retained_edges": int(retained_edges.sum()),
        "retained_edge_fraction": float(retained_edges.sum() / full_edges.sum()),
        "retained_flow_fraction": float(retained_flow.sum() / full_flow.sum()),
        "retained_bau_fraction": float(np.asarray(full.P_ini)[keep].sum() / np.asarray(full.P_ini).sum()),
        "largest_component_nodes": int(retained_components[0]),
        "largest_component_fraction": float(retained_components[0] / keep.size),
        "largest_component_retention": float(
            retained_components[0] / baseline["largest_component_nodes"]
        ),
        "weak_component_count": len(retained_components),
        "isolated_node_count": int((degrees == 0).sum()),
        "added_isolated_node_count": int((degrees == 0).sum() - baseline["isolated_node_count"]),
        "retained_nonempty_sector_count": len(retained_sectors),
        "full_sector_count": int(full.K),
    }


def valid_candidate(summary: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        args.min_removed_nodes <= summary["heldout_node_count"] <= args.max_removed_nodes
        and summary["heldout_induced_connected"]
        and summary["largest_component_retention"] >= args.min_largest_component_retention
        and summary["added_isolated_node_count"] <= args.max_added_isolated_nodes
        and summary["retained_nonempty_sector_count"] / summary["full_sector_count"]
        >= args.min_retained_sector_fraction
    )


def frontier_scores(
    strategy: str, frontier: set[int], selected: set[int], adjacency: np.ndarray,
    flow: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    choices = np.asarray(sorted(frontier), dtype=np.int64)
    if strategy == "random_frontier":
        return rng.random(choices.size)
    selected_array = np.asarray(sorted(selected), dtype=np.int64)
    internal_links = ((adjacency[np.ix_(choices, selected_array)] != 0)
                      | (adjacency[np.ix_(selected_array, choices)].T != 0)).sum(axis=1)
    if strategy == "connectivity_preserving_compact":
        total_links = ((adjacency[choices] != 0) | (adjacency[:, choices].T != 0)).sum(axis=1)
        boundary_links = total_links - internal_links
        incident_flow = flow[choices].sum(axis=1) + flow[:, choices].sum(axis=0)
        flow_scale = max(float(np.median(incident_flow)), 1e-12)
        return (
            -internal_links.astype(float)
            + 0.35 * boundary_links.astype(float)
            + 0.15 * (incident_flow / flow_scale)
            + 1e-4 * rng.random(choices.size)
        )
    incident_flow = flow[choices].sum(axis=1) + flow[:, choices].sum(axis=0)
    return incident_flow + 1e-4 * rng.random(choices.size)


def grow_candidate(
    full: Any, neighbours: list[np.ndarray], flow: np.ndarray, strategy: str,
    rng: np.random.Generator, args: argparse.Namespace,
) -> set[int] | None:
    seed = int(rng.integers(full.N))
    selected = {seed}
    frontier = set(map(int, neighbours[seed]))
    target = int(np.clip(
        rng.integers(args.min_removed_nodes, args.max_removed_nodes + 1),
        args.min_removed_nodes, args.max_removed_nodes,
    ))
    while frontier and len(selected) < target:
        choices = np.asarray(sorted(frontier), dtype=np.int64)
        score = frontier_scores(strategy, frontier, selected, full.A, flow, rng)
        node = int(choices[int(np.argmin(score))])
        frontier.remove(node)
        selected.add(node)
        frontier.update(int(neighbour) for neighbour in neighbours[node] if neighbour not in selected)
    return selected if len(selected) >= args.min_removed_nodes else None


def collect_candidates(
    full: Any, neighbours: list[np.ndarray], flow: np.ndarray, baseline: dict[str, int], strategy: str,
    args: argparse.Namespace, seed: int,
) -> list[Candidate]:
    rng = np.random.default_rng(seed)
    found: dict[tuple[int, ...], Candidate] = {}
    seen: set[tuple[int, ...]] = set()
    for attempt in range(1, args.attempts_per_strategy + 1):
        nodes = grow_candidate(full, neighbours, flow, strategy, rng, args)
        if nodes is not None:
            key = tuple(sorted(nodes))
            if key not in seen:
                seen.add(key)
                summary = summarize(full, nodes, neighbours, baseline)
                if valid_candidate(summary, args):
                    found[key] = Candidate(strategy=strategy, nodes=key, summary=summary)
        if attempt % 5000 == 0:
            print(
                f"[{strategy}] attempts={attempt}; unique={len(seen)}; valid={len(found)}",
                flush=True,
            )
    return diversified_shortlist(
        list(found.values()), args.candidates_per_strategy, args.target_removed_nodes,
    )


def diversified_shortlist(
    candidates: list[Candidate], limit: int, target_removed_nodes: int,
) -> list[Candidate]:
    """Round-robin static rankings so matching-relevant extremes survive truncation."""
    if len(candidates) <= limit:
        return candidates
    rankings = [
        sorted(candidates, key=lambda candidate: (
            abs(candidate.summary["heldout_node_count"] - target_removed_nodes),
            -candidate.summary["largest_component_retention"],
        )),
    ]
    for key in (
        "retained_edge_fraction", "retained_flow_fraction", "retained_bau_fraction",
        "largest_component_retention",
    ):
        rankings.append(sorted(candidates, key=lambda candidate: candidate.summary[key]))
        rankings.append(sorted(candidates, key=lambda candidate: candidate.summary[key], reverse=True))
    selected: list[Candidate] = []
    selected_nodes: set[tuple[int, ...]] = set()
    rank = 0
    while len(selected) < limit:
        added = False
        for ranking in rankings:
            if rank >= len(ranking):
                continue
            candidate = ranking[rank]
            if candidate.nodes not in selected_nodes:
                selected.append(candidate)
                selected_nodes.add(candidate.nodes)
                added = True
                if len(selected) == limit:
                    break
        if not added and rank >= max(len(ranking) for ranking in rankings) - 1:
            break
        rank += 1
    return selected


def spread(items: list[Candidate], key: str) -> float:
    values = [float(item.summary[key]) for item in items]
    return max(values) - min(values)


def triplet_rejection_reason(items: list[Candidate], args: argparse.Namespace) -> str | None:
    if args.require_disjoint_blocks:
        node_sets = [set(item.nodes) for item in items]
        if any(node_sets[left] & node_sets[right] for left in range(3) for right in range(left + 1, 3)):
            return "overlapping_removed_nodes"
    node_spread = spread(items, "retained_node_fraction")
    edge_spread = spread(items, "retained_edge_fraction")
    flow_spread = spread(items, "retained_flow_fraction")
    bau_spread = spread(items, "retained_bau_fraction")
    if node_spread > args.max_node_retention_spread:
        return "node_retention_spread"
    if edge_spread > args.max_edge_retention_spread:
        return "edge_retention_spread"
    if flow_spread > args.max_flow_retention_spread:
        return "flow_retention_spread"
    if bau_spread > args.max_bau_retention_spread:
        return "bau_retention_spread"
    return None


def triplet_score(items: list[Candidate], args: argparse.Namespace) -> float | None:
    if triplet_rejection_reason(items, args) is not None:
        return None
    node_spread = spread(items, "retained_node_fraction")
    edge_spread = spread(items, "retained_edge_fraction")
    flow_spread = spread(items, "retained_flow_fraction")
    bau_spread = spread(items, "retained_bau_fraction")
    size_error = sum(abs(item.summary["heldout_node_count"] - args.target_removed_nodes)
                     for item in items) / max(3 * args.target_removed_nodes, 1)
    return 3.0 * node_spread + 3.0 * edge_spread + 3.0 * flow_spread + bau_spread + size_error


def select_best(
    groups: dict[str, list[Candidate]], args: argparse.Namespace,
) -> tuple[tuple[list[Candidate], float] | None, dict[str, int]]:
    best: tuple[list[Candidate], float] | None = None
    diagnostics = {"total_triplets": 0, "feasible_triplets": 0}
    for first in groups[STRATEGIES[0]]:
        for second in groups[STRATEGIES[1]]:
            for third in groups[STRATEGIES[2]]:
                items = [first, second, third]
                diagnostics["total_triplets"] += 1
                reason = triplet_rejection_reason(items, args)
                if reason is not None:
                    diagnostics[reason] = diagnostics.get(reason, 0) + 1
                    continue
                diagnostics["feasible_triplets"] += 1
                score = triplet_score(items, args)
                assert score is not None
                if best is None or score < best[1]:
                    best = (items, score)
    return best, diagnostics


def candidate_ranges(candidates: list[Candidate]) -> dict[str, dict[str, float]]:
    keys = (
        "heldout_node_count", "retained_node_fraction", "retained_edge_fraction",
        "retained_flow_fraction", "retained_bau_fraction", "largest_component_retention",
        "added_isolated_node_count",
    )
    return {
        key: {
            "min": float(min(candidate.summary[key] for candidate in candidates)),
            "max": float(max(candidate.summary[key] for candidate in candidates)),
        }
        for key in keys
    }


def node_jaccard(left: Candidate, right: Candidate) -> float:
    left_nodes = set(left.nodes)
    right_nodes = set(right.nodes)
    return len(left_nodes & right_nodes) / len(left_nodes | right_nodes)


def representative_score(candidate: Candidate, max_removed_nodes: int) -> float:
    summary = candidate.summary
    size = summary["heldout_node_count"] / max(max_removed_nodes, 1)
    isolation = summary["added_isolated_node_count"] / max(summary["retained_nodes"], 1)
    return (
        0.30 * size
        + 0.25 * summary["largest_component_retention"]
        + 0.15 * summary["retained_edge_fraction"]
        + 0.15 * summary["retained_flow_fraction"]
        + 0.15 * summary["retained_bau_fraction"]
        - 0.25 * isolation
    )


def select_representatives(
    groups: dict[str, list[Candidate]], args: argparse.Namespace,
) -> list[Candidate]:
    selected: list[Candidate] = []
    for strategy in STRATEGIES:
        eligible = [
            candidate for candidate in groups[strategy]
            if candidate.summary["heldout_node_count"] >= args.representative_min_removed_nodes
        ]
        if len(eligible) < args.representatives_per_strategy:
            raise RuntimeError(
                f"{strategy} has only {len(eligible)} candidates with at least "
                f"{args.representative_min_removed_nodes} removed nodes"
            )
        chosen: list[Candidate] = []
        while len(chosen) < args.representatives_per_strategy:
            candidate = max(
                (item for item in eligible if item not in chosen),
                key=lambda item: representative_score(item, args.max_removed_nodes)
                - 0.20 * max((node_jaccard(item, other) for other in chosen), default=0.0),
            )
            chosen.append(candidate)
        selected.extend(chosen)
    return selected


def write_representatives(
    output_dir: Path, candidates: list[Candidate], args: argparse.Namespace,
) -> Path:
    representative_dir = output_dir / "representative_large_subgraphs"
    representative_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, candidate in enumerate(candidates, 1):
        manifest = {
            "definition": {
                "design": "static representative large connected removal block",
                "selection_guard": "static graph, P_ini BAU, and static edge flow only; no events, labels, checkpoints, predictions, or metrics",
                "selection_rule": (
                    "two per cut strategy; minimum removal size followed by maximum static "
                    "core/edge/flow/BAU retention, minimum added isolation, and within-strategy "
                    "node-overlap penalty"
                ),
                "claim_scope": "candidate selection only; repeated blocks may overlap and are not independent replications",
            },
            "partition_seed": args.seed,
            "representative_index": index,
            "cut_strategy": candidate.strategy,
            "static_selection_score": representative_score(candidate, args.max_removed_nodes),
            **candidate.summary,
            "id_sha256": {"heldout_nodes": sha256_ids(candidate.nodes)},
        }
        path = representative_dir / f"representative_large_subgraph_{index}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        rows.append({"manifest": path.name, **manifest})
    report = {
        "status": "STATIC_REPRESENTATIVE_SELECTION_COMPLETE",
        "selection_guard": "static graph and BAU quantities only; no events, labels, checkpoints, predictions, or metrics",
        "selection_rule": "two top-scoring large candidates per strategy with within-strategy overlap penalty",
        "parameters": {
            "representatives_per_strategy": args.representatives_per_strategy,
            "representative_min_removed_nodes": args.representative_min_removed_nodes,
        },
        "blocks": rows,
        "pairwise_removed_node_jaccard": [
            {
                "left": left + 1,
                "right": right + 1,
                "value": node_jaccard(candidates[left], candidates[right]),
            }
            for left in range(len(candidates)) for right in range(left + 1, len(candidates))
        ],
    }
    report_path = representative_dir / "representative_large_subgraphs_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def sha256_ids(nodes: tuple[int, ...]) -> str:
    return hashlib.sha256(",".join(map(str, nodes)).encode("ascii")).hexdigest()


def main() -> None:
    args = parse_args()
    if not (0 < args.min_removed_nodes <= args.target_removed_nodes <= args.max_removed_nodes):
        raise ValueError("Require 0 < min-removed-nodes <= target-removed-nodes <= max-removed-nodes")
    if args.attempts_per_strategy < 1 or args.candidates_per_strategy < 1:
        raise ValueError("Search budgets must be positive")
    if args.representatives_per_strategy < 1:
        raise ValueError("--representatives-per-strategy must be positive")
    full = build_network(Params())
    neighbours = undirected_neighbours(np.asarray(full.A, dtype=float))
    flow = np.asarray(full.A, dtype=float) * np.asarray(full.P_ini, dtype=float)[:, None]
    baseline = full_connectivity_baseline(full)
    print(json.dumps({"full_graph_connectivity_baseline": baseline}, indent=2), flush=True)
    groups = {
        strategy: collect_candidates(full, neighbours, flow, baseline, strategy, args, args.seed + index)
        for index, strategy in enumerate(STRATEGIES)
    }
    counts = {strategy: len(candidates) for strategy, candidates in groups.items()}
    print(json.dumps({"valid_candidates": counts}, indent=2), flush=True)
    chosen, triplet_diagnostics = select_best(groups, args)
    diagnostic_report = {
        "selection_guard": "static graph and BAU quantities only; no events, labels, checkpoints, predictions, or metrics",
        "candidate_shortlist_counts": counts,
        "candidate_ranges_by_strategy": {
            strategy: candidate_ranges(candidates)
            for strategy, candidates in groups.items()
        },
        "triplet_diagnostics": triplet_diagnostics,
        "rejection_order": [
            "overlapping_removed_nodes", "node_retention_spread", "edge_retention_spread",
            "flow_retention_spread", "bau_retention_spread",
        ],
    }
    diagnostic_path = args.output_dir / "matched_induced_subgraphs_diagnostics.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(json.dumps(diagnostic_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"triplet_diagnostics": triplet_diagnostics}, indent=2), flush=True)
    representative_candidates = select_representatives(groups, args)
    representative_report = write_representatives(
        args.output_dir, representative_candidates, args,
    )
    print(f"STATIC_REPRESENTATIVE_SELECTION_COMPLETE: {representative_report}", flush=True)
    if chosen is None:
        print(
            "NO_MATCHED_INDUCED_SUBGRAPH_TRIPLET: representative blocks were written; "
            f"inspect {diagnostic_path} for static constraint rejection counts.",
            flush=True,
        )
        return
    blocks, score = chosen
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for block_index, candidate in enumerate(blocks, 1):
        manifest = {
            "definition": {
                "design": "connected removal block with retained-core-preserving induced graph",
                "selection_guard": "static graph, P_ini BAU, and static edge flow only; no events, labels, checkpoints, predictions, or metrics",
                "claim_scope": "frozen-checkpoint topology-perturbation robustness; not unseen-node or unseen-topology transfer",
                "hard_constraints": {
                    "removed_block_induced_connected": True,
                    "retained_largest_component_retention_min": args.min_largest_component_retention,
                    "max_added_isolated_nodes": args.max_added_isolated_nodes,
                    "retained_sector_fraction_min": args.min_retained_sector_fraction,
                    "cross_block_retention_spreads": {
                        "node_max": args.max_node_retention_spread,
                        "edge_max": args.max_edge_retention_spread,
                        "flow_max": args.max_flow_retention_spread,
                        "bau_max": args.max_bau_retention_spread,
                    },
                },
            },
            "partition_seed": args.seed,
            "block": block_index,
            "cut_strategy": candidate.strategy,
            **candidate.summary,
            "id_sha256": {"heldout_nodes": sha256_ids(candidate.nodes)},
        }
        path = output_dir / f"heldout_topology_block{block_index}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        summaries.append({"block": block_index, "cut_strategy": candidate.strategy, **candidate.summary})
    report = {
        "status": "PASS",
        "design": "three matched static connected removal blocks selected from three cut strategies",
        "selection_guard": "static graph and BAU quantities only; no events, labels, checkpoints, predictions, or metrics",
        "parameters": vars(args),
        "full_graph_connectivity_baseline": baseline,
        "candidate_counts": counts,
        "optimization_score": score,
        "optimization_scope": (
            "minimum objective among all feasible triplets formed from the retained "
            "per-strategy candidate shortlists; not a proof of the global graph optimum"
        ),
        "blocks": summaries,
        "cross_block_spreads": {
            key: spread(blocks, key)
            for key in ("retained_node_fraction", "retained_edge_fraction", "retained_flow_fraction", "retained_bau_fraction")
        },
    }
    (output_dir / "matched_induced_subgraphs_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("MATCHED_INDUCED_SUBGRAPH_SEARCH_PASS")


if __name__ == "__main__":
    main()
