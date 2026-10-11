"""Audit saved Tables 4-5 counts without simulation or checkpoint inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "flagship_y8h_download"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

from pi_dl.config import Config, get_paths
from pi_dl.data import CascadeEventDataset, StaticNetwork, collate_single, split_event_ids
from evaluate_y8_propagation_depth import directed_distances


DISTANCE_KEYS = ("downstream_1_hop", "downstream_2_hop", "downstream_3plus_hop")
POSITION_KEYS = ("feedforward_downstream", "feedback_connected_downstream")


def graph_batch(node_count, sources, targets, shocked, peak):
    return {
        "Nr": node_count,
        "edge_src": torch.as_tensor(sources, dtype=torch.long),
        "edge_dst": torch.as_tensor(targets, dtype=torch.long),
        "shock_mask": torch.as_tensor(shocked, dtype=torch.float32),
        "peak_loss": torch.as_tensor(peak, dtype=torch.float32),
    }


def batch_statistics(batch, threshold):
    downstream, upstream = directed_distances(batch)
    shocked = batch["shock_mask"].numpy() > 0.5
    cascade = (~shocked) & (batch["peak_loss"].numpy() > threshold)
    forward = np.isfinite(downstream)
    backward = np.isfinite(upstream)
    masks = {
        "cascade_total": cascade,
        "downstream_1_hop": cascade & (downstream == 1),
        "downstream_2_hop": cascade & (downstream == 2),
        "downstream_3plus_hop": cascade & forward & (downstream >= 3),
        "feedforward_downstream": cascade & forward & ~backward,
        "feedback_connected_downstream": cascade & forward & backward,
        "upstream_only": cascade & ~forward & backward,
        "neither": cascade & ~forward & ~backward,
    }
    adjacency = csr_matrix(
        (np.ones(batch["edge_src"].numel()),
         (batch["edge_src"].numpy(), batch["edge_dst"].numpy())),
        shape=(int(batch["Nr"]), int(batch["Nr"])),
    )
    _, components = connected_components(adjacency, directed=True, connection="strong")
    sizes = np.bincount(components)
    nontrivial = sizes[components] > 1
    same_shock = np.isin(components, components[shocked]) & nontrivial
    feedback = masks["feedback_connected_downstream"]
    masks.update({
        "feedback_same_shock_cycle": feedback & same_shock,
        "feedback_different_shock_only": feedback & ~same_shock,
        "feedback_in_nontrivial_scc": feedback & nontrivial,
        "feedback_outside_nontrivial_scc": feedback & ~nontrivial,
    })
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    assert counts["cascade_total"] == sum(counts[name] for name in (
        *POSITION_KEYS, "upstream_only", "neither",
    ))
    assert counts["feedback_connected_downstream"] == (
        counts["feedback_same_shock_cycle"] + counts["feedback_different_shock_only"]
    )
    return counts, masks, components, sizes


def add_check(checks, label, actual, expected):
    checks.append({"name": label, "actual": actual, "expected": expected,
                   "passed": actual == expected})


def compare_saved_report(saved, counts, event_ids):
    checks = []
    add_check(checks, "saved_test_event_count", saved["protocol"]["test_events"], len(event_ids))
    add_check(checks, "saved_models", sorted(saved["per_seed"]), ["DirGNN", "PhyCaSNet"])
    for model, runs in saved["per_seed"].items():
        add_check(checks, f"{model}/seeds", sorted(run["seed"] for run in runs), list(range(5)))
        for run in runs:
            prefix = f"{model}/seed{run['seed']}"
            add_check(checks, prefix + "/event_ids",
                      sorted(map(int, run["event_peak_mae"])), sorted(event_ids))
            for group, names in (("by_distance", DISTANCE_KEYS), ("by_position", POSITION_KEYS)):
                for name in names:
                    add_check(checks, prefix + "/" + name,
                              counts[name], run[group][name]["n_event_nodes"])
        summary = saved["mean_std_across_seeds"][model]
        for group, names in (("by_distance", DISTANCE_KEYS), ("by_position", POSITION_KEYS)):
            for name in names:
                add_check(checks, model + "/summary/" + name,
                          counts[name], summary[group][name]["n_event_nodes"])
    return checks


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_audit(args):
    cfg = Config(data_root=str(args.data_root))
    cfg.seed = args.seed
    cfg.train_frac, cfg.val_frac = 0.80, 0.10
    cfg.event_scalars_mode = "minimal"
    paths = get_paths(cfg)
    network = StaticNetwork(cfg)
    _, _, event_ids = split_event_ids(paths["events"], cfg)
    event_ids = event_ids[:args.n_test]
    if len(event_ids) != args.n_test:
        raise ValueError(f"Requested {args.n_test} test events; found {len(event_ids)}")
    dataset = CascadeEventDataset(cfg, network, event_ids, train_mode=False)
    saved = json.loads(args.saved_report.read_text(encoding="utf-8"))
    totals, full_totals, outside_totals = Counter(), Counter(), Counter()
    rows, manifest = [], []
    feedback_firms, feedback_events = set(), set()
    graph_description = None
    for index, event_id in enumerate(event_ids):
        batch = collate_single([dataset[index]])
        local_counts, local_masks, _, _ = batch_statistics(batch, args.threshold)
        reach = batch["reach_idx"].numpy()
        event_path = paths["events"] / f"event_{event_id:06d}.npz"
        with np.load(event_path, allow_pickle=True) as event:
            shock = (event["shock_mask"] > 0.5).astype(np.float32)
            peak = event["peak_loss_node"].astype(np.float32)
        np.testing.assert_array_equal(batch["shock_mask"].numpy(), shock[reach])
        np.testing.assert_array_equal(batch["peak_loss"].numpy(), peak[reach])
        np.testing.assert_array_equal(reach, network.shock_reach(shock, cfg.reach_hops))
        full_batch = graph_batch(network.V, network.edge_src, network.edge_dst, shock, peak)
        full_counts, full_masks, components, sizes = batch_statistics(full_batch, args.threshold)
        outside = np.ones(network.V, dtype=bool)
        outside[reach] = False
        outside_counts = {
            name: int((mask & outside).sum())
            for name, mask in full_masks.items()
            if name in ("cascade_total", *POSITION_KEYS, "upstream_only", "neither")
        }
        assert full_counts["cascade_total"] == local_counts["cascade_total"] + outside_counts["cascade_total"]
        local_feedback = local_masks["feedback_connected_downstream"]
        local_counts["feedback_outside_full_nontrivial_scc"] = int(
            (local_feedback & (sizes[components[reach]] == 1)).sum()
        )
        global_same_shock = np.isin(components[reach], components[shock > 0.5]) & (sizes[components[reach]] > 1)
        local_counts["feedback_same_shock_cycle_full_graph"] = int((local_feedback & global_same_shock).sum())
        totals.update(local_counts)
        full_totals.update(full_counts)
        outside_totals.update(outside_counts)
        feedback_firms.update(reach[local_feedback].tolist())
        if local_feedback.any():
            feedback_events.add(event_id)
        rows.append({"event_id": event_id, "prediction_nodes": int(batch["Nr"]),
                     **{"prediction_" + name: value for name, value in local_counts.items()},
                     **{"full_" + name: value for name, value in full_counts.items()},
                     **{"outside_" + name: value for name, value in outside_counts.items()}})
        manifest.append({"event_id": event_id, "sha256": file_hash(event_path)})
        if graph_description is None:
            edges = set(zip(network.edge_src.tolist(), network.edge_dst.tolist()))
            graph_description = {
                "nodes": network.V, "edges": network.E,
                "largest_scc_size": int(sizes.max()),
                "scc_size_histogram": dict(Counter(sizes.tolist())),
                "nontrivial_scc_members_zero_based": [
                    np.flatnonzero(components == label).tolist()
                    for label in np.flatnonzero(sizes > 1)
                ],
                "reciprocal_pairs_zero_based": sorted(
                    [source, target] for source, target in edges
                    if source < target and (target, source) in edges
                ),
                "self_loops_zero_based": sorted(source for source, target in edges if source == target),
            }
        if (index + 1) % 50 == 0:
            print(f"Audited {index + 1}/{len(event_ids)} events", flush=True)
    checks = compare_saved_report(saved, totals, event_ids)
    add_check(checks, "saved_cascade_definition", saved["protocol"]["cascade_definition"],
              f"not directly shocked and ground-truth peak loss > {args.threshold}")
    add_check(checks, "distance_partition", sum(totals[name] for name in DISTANCE_KEYS), totals["cascade_total"])
    add_check(checks, "position_partition", sum(totals[name] for name in POSITION_KEYS), totals["cascade_total"])
    claims = {"cascade_total": 10849, "feedback_connected_downstream": 1072,
              "feedback_same_shock_cycle": 131, "feedback_different_shock_only": 941,
              "feedback_outside_full_nontrivial_scc": 768}
    for name, expected in claims.items():
        add_check(checks, "response_claim/" + name, totals[name], expected)
    add_check(checks, "response_claim/full_total", full_totals["cascade_total"], 17450)
    add_check(checks, "response_claim/upstream_only", full_totals["upstream_only"], 4991)
    add_check(checks, "response_claim/gscc_size", graph_description["largest_scc_size"], 2)
    sources = [Path(__file__), ROOT / "flagship_train/evaluate_multiseed_network_propagation.py",
               ROOT / "flagship_train/evaluate_y8_propagation_depth.py",
               ARCHIVE / "pi_dl/data.py", ARCHIVE / "pi_dl/config.py",
               paths["static"], args.saved_report]
    passed = all(check["passed"] for check in checks)
    report = {
        "status": "PASS" if passed else "FAIL",
        "scope": "Read-only label/count audit; no simulation, model loading, or inference.",
        "limitations": ["Topology does not establish exclusively demand-caused losses.",
                        "Prediction accuracy and reverse-message ablation metrics are not recomputed."],
        "protocol": {"data_root": str(args.data_root.resolve()), "split_seed": cfg.seed,
                     "test_events": len(event_ids), "threshold": args.threshold,
                     "reach_hops": cfg.reach_hops, "counting_unit": "event-node observation"},
        "graph": graph_description, "prediction_scope": dict(totals),
        "full_network": dict(full_totals), "outside_prediction_scope": dict(outside_totals),
        "distinct_feedback_firms": len(feedback_firms), "feedback_events": len(feedback_events),
        "checks": checks, "test_event_manifest": manifest,
        "source_sha256": {str(path.resolve()): file_hash(path) for path in sources},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "comment14_network_count_audit.json"
    summary_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output_dir / "comment14_event_counts.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": report["status"], "prediction_scope": dict(totals),
                      "full_network": dict(full_totals), "outside_prediction_scope": dict(outside_totals),
                      "failed_checks": [check for check in checks if not check["passed"]]}, indent=2))
    print(f"Saved: {summary_path}")
    return 0 if passed else 1


class SyntheticAuditTests(unittest.TestCase):
    def test_two_shocks_in_dag_are_not_a_cycle(self):
        counts, _, _, _ = batch_statistics(graph_batch(3, [0, 1], [1, 2], [1, 0, 1], [1, .2, 1]), .05)
        self.assertEqual(counts["feedback_connected_downstream"], 1)
        self.assertEqual(counts["feedback_same_shock_cycle"], 0)
        self.assertEqual(counts["feedback_outside_nontrivial_scc"], 1)

    def test_reciprocal_pair_is_same_shock_cycle(self):
        counts, _, _, _ = batch_statistics(graph_batch(2, [0, 1], [1, 0], [1, 0], [1, .2]), .05)
        self.assertEqual(counts["feedback_same_shock_cycle"], 1)

    def test_upstream_only_outside_downstream_population(self):
        counts, _, _, _ = batch_statistics(graph_batch(3, [0, 1], [1, 2], [0, 1, 0], [.2, 1, .2]), .05)
        self.assertEqual(counts["upstream_only"], 1)
        self.assertEqual(counts["downstream_1_hop"], 1)
        self.assertEqual(counts["cascade_total"], 2)

    def test_strict_threshold_and_shocked_exclusion(self):
        counts, _, _, _ = batch_statistics(graph_batch(3, [0, 0], [1, 2], [1, 0, 0], [1, .05, .05001]), .05)
        self.assertEqual(counts["cascade_total"], 1)

    def test_scc_size_not_number_of_cycles(self):
        _, _, _, sizes = batch_statistics(graph_batch(4, [0, 1, 2, 3], [1, 0, 3, 2], [1, 0, 0, 0], [1, .2, .2, .2]), .05)
        self.assertEqual(sizes.tolist(), [2, 2])

    def test_no_shock_and_disconnected_damage(self):
        counts, _, _, _ = batch_statistics(graph_batch(2, [], [], [0, 0], [.2, 0]), .05)
        self.assertEqual(counts["neither"], 1)
        self.assertEqual(counts["feedback_connected_downstream"], 0)

    def test_saved_count_tamper_is_detected(self):
        counts = {name: 1 for name in (*DISTANCE_KEYS, *POSITION_KEYS)}
        grouped = {"by_distance": {name: {"n_event_nodes": 1} for name in DISTANCE_KEYS},
                   "by_position": {name: {"n_event_nodes": 1} for name in POSITION_KEYS}}
        saved = {"protocol": {"test_events": 1},
                 "per_seed": {model: [{"seed": seed, "event_peak_mae": {"7": {}}, **grouped}
                                      for seed in range(5)] for model in ("PhyCaSNet", "DirGNN")},
                 "mean_std_across_seeds": {model: grouped for model in ("PhyCaSNet", "DirGNN")}}
        self.assertTrue(all(check["passed"] for check in compare_saved_report(saved, counts, [7])))
        counts["downstream_1_hop"] = 2
        self.assertFalse(all(check["passed"] for check in compare_saved_report(saved, counts, [7])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ARCHIVE / "data/cascade_henriet_5000")
    parser.add_argument("--saved-report", type=Path, default=ARCHIVE / "reviewer_experiments_local/multiseed_network_propagation.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "answer_result/comment14_network_audit")
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic checks only; do not read real event data.")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticAuditTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.n_test <= 0 or not np.isfinite(args.threshold) or args.threshold < 0:
        parser.error("n-test must be positive and threshold must be finite and nonnegative")
    return run_audit(args)


if __name__ == "__main__":
    raise SystemExit(main())