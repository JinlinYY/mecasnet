"""Rebuild the two severe-tier 90-node static witnesses from the feasibility sweep."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

def repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Could not locate the repository root")


ROOT = repository_root()
CANGZHOU = Path(os.environ.get("CANGZHOU_PIPELINE_ROOT", ROOT / "cangzhou_pipeline"))
for path in (ROOT, CANGZHOU):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import search_induced_subgraphs as search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def digest(nodes: tuple[int, ...]) -> str:
    return hashlib.sha256(",".join(map(str, nodes)).encode("ascii")).hexdigest()


def main() -> None:
    args = parse_args()
    full = search.build_network(search.Params())
    neighbours = search.undirected_neighbours(search.np.asarray(full.A, dtype=float))
    flow = search.np.asarray(full.A, dtype=float) * search.np.asarray(full.P_ini, dtype=float)[:, None]
    baseline = search.full_connectivity_baseline(full)
    sweep_args = argparse.Namespace(
        min_removed_nodes=90,
        target_removed_nodes=90,
        max_removed_nodes=90,
        attempts_per_strategy=30000,
        candidates_per_strategy=1,
        min_largest_component_retention=0.70,
        max_added_isolated_nodes=40,
        min_retained_sector_fraction=0.80,
    )
    selections = (
        (1, "connectivity_preserving_compact", 20260817 + 1000 + 1),
        (2, "low_flow_frontier", 20260817 + 1000 + 2),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for block, strategy, seed in selections:
        candidates = search.collect_candidates(
            full, neighbours, flow, baseline, strategy, sweep_args, seed,
        )
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one severe 90-node witness for {strategy}, got {len(candidates)}")
        candidate = candidates[0]
        payload = {
            "definition": {
                "design": "formal severe-tier connected removal block",
                "selection_guard": "static graph, P_ini BAU, and static edge flow only; no events, labels, checkpoints, predictions, or metrics",
                "selection_rule": "exact 90-node witness reproduced from the severe feasibility sweep with its fixed seed and static constraints",
                "claim_scope": "frozen-checkpoint severe topology-perturbation stress test; not unseen-node or unseen-topology transfer",
            },
            "sweep_seed": seed,
            "block": block,
            "cut_strategy": strategy,
            **candidate.summary,
            "id_sha256": {"heldout_nodes": digest(candidate.nodes)},
        }
        path = args.output_dir / f"heldout_topology_block{block}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        rows.append({"manifest": path.name, **payload})
        print(f"SEVERE_MANIFEST_PASS block={block} strategy={strategy} nodes={len(candidate.nodes)}")
    (args.output_dir / "severe90_selection.json").write_text(
        json.dumps({"status": "PASS", "blocks": rows}, indent=2) + "\n", encoding="utf-8"
    )
    print("SEVERE90_MANIFESTS_COMPLETE")


if __name__ == "__main__":
    main()
