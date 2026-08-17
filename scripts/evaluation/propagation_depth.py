r"""Evaluate frozen Y8-H predictions by cascade distance from shocked firms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from torch.utils.data import DataLoader


from mecasnet.config import Config
from mecasnet.data import (
    CascadeEventDataset,
    StaticNetwork,
    collate_single,
    split_event_ids,
)
from mecasnet.evaluation import KEY_DAYS, load_y8, r2, to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def finite_mean(values: list[float]) -> float:
    valid = [value for value in values if np.isfinite(value)]
    return float(np.mean(valid)) if valid else float("nan")


def metrics(
    peak_prediction: np.ndarray,
    peak_target: np.ndarray,
    keyframe_prediction: np.ndarray,
    keyframe_target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    per_day = [
        r2(keyframe_prediction[index, mask], keyframe_target[index, mask])
        for index in range(keyframe_target.shape[0])
    ]
    flat_prediction = keyframe_prediction[:, mask].reshape(-1)
    flat_target = keyframe_target[:, mask].reshape(-1)
    return {
        "n_event_nodes": int(mask.sum()),
        "peak_mae": (
            float(np.abs(peak_prediction[mask] - peak_target[mask]).mean())
            if mask.any() else float("nan")
        ),
        "peak_r2": r2(peak_prediction[mask], peak_target[mask]),
        "keyframe_r2_mean": finite_mean(per_day),
        "keyframe_r2_by_day": dict(zip(map(str, KEY_DAYS), per_day)),
        "trajectory_mae": (
            float(np.abs(flat_prediction - flat_target).mean())
            if flat_target.size else float("nan")
        ),
        "trajectory_r2": r2(flat_prediction, flat_target),
        "ground_truth_peak_mean": (
            float(peak_target[mask].mean()) if mask.any() else float("nan")
        ),
    }


def directed_distances(batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    node_count = int(batch["Nr"])
    edge_src = batch["edge_src"].detach().cpu().numpy().astype(np.int64)
    edge_dst = batch["edge_dst"].detach().cpu().numpy().astype(np.int64)
    shock = batch["shock_mask"].detach().cpu().numpy() > 0.5
    shock_indices = np.flatnonzero(shock)
    if shock_indices.size == 0:
        missing = np.full(node_count, np.inf, dtype=np.float64)
        return missing, missing.copy()
    adjacency = csr_matrix(
        (np.ones(edge_src.size, dtype=np.float32), (edge_src, edge_dst)),
        shape=(node_count, node_count),
    )
    downstream = dijkstra(
        adjacency,
        directed=True,
        unweighted=True,
        indices=shock_indices,
        min_only=True,
    )
    upstream = dijkstra(
        adjacency.T.tocsr(),
        directed=True,
        unweighted=True,
        indices=shock_indices,
        min_only=True,
    )
    return np.asarray(downstream), np.asarray(upstream)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = Config(data_root=args.data_root)
    cfg.deq_iters = 1
    cfg.deq_grad_iters = 1
    cfg.event_scalars_mode = "minimal"
    network = StaticNetwork(cfg)
    _, _, test_ids = split_event_ids(Path(args.data_root) / cfg.events_dir, cfg)
    test_ids = test_ids[:args.n_test]
    dataset = CascadeEventDataset(cfg, network, test_ids, train_mode=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_single,
        pin_memory=device.type == "cuda",
    )
    model = load_y8(Path(args.checkpoint), cfg, network.Fv, device).eval()

    peak_predictions: list[np.ndarray] = []
    peak_targets: list[np.ndarray] = []
    keyframe_predictions: list[np.ndarray] = []
    keyframe_targets: list[np.ndarray] = []
    shocked: list[np.ndarray] = []
    hops: list[np.ndarray] = []
    downstream_distances: list[np.ndarray] = []
    upstream_distances: list[np.ndarray] = []
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        output = model(batch)
        peak_predictions.append(output["peak"].detach().cpu().numpy())
        peak_targets.append(batch["peak_loss"].detach().cpu().numpy())
        keyframe_predictions.append(
            output["u_keyframes"].detach().cpu().numpy()
        )
        keyframe_targets.append(batch["u_keyframes"].detach().cpu().numpy())
        shocked.append(batch["shock_mask"].detach().cpu().numpy() > 0.5)
        hops.append(
            batch["shock_hop_oh"].detach().cpu().numpy().argmax(axis=1)
        )
        downstream, upstream = directed_distances(batch)
        downstream_distances.append(downstream)
        upstream_distances.append(upstream)

    peak_prediction = np.concatenate(peak_predictions)
    peak_target = np.concatenate(peak_targets)
    keyframe_prediction = np.concatenate(keyframe_predictions, axis=1)
    keyframe_target = np.concatenate(keyframe_targets, axis=1)
    shock_mask = np.concatenate(shocked)
    hop = np.concatenate(hops)
    downstream_distance = np.concatenate(downstream_distances)
    upstream_distance = np.concatenate(upstream_distances)
    cascade = (~shock_mask) & (peak_target > args.threshold)

    strata = {
        "1_hop": cascade & (hop == 1),
        "2_hop": cascade & (hop == 2),
        "3_hop": cascade & (hop == 3),
    }
    downstream_reachable = np.isfinite(downstream_distance)
    upstream_reachable = np.isfinite(upstream_distance)
    position_strata = {
        "downstream_only": cascade & downstream_reachable & ~upstream_reachable,
        "bidirectionally_reachable": cascade & downstream_reachable & upstream_reachable,
    }
    directed_distance_strata = {
        "downstream_1_hop": cascade & (downstream_distance == 1),
        "downstream_2_hop": cascade & (downstream_distance == 2),
        "downstream_3plus_hop": cascade & (downstream_distance >= 3) & downstream_reachable,
        "upstream_1_hop": cascade & (upstream_distance == 1),
        "upstream_2_hop": cascade & (upstream_distance == 2),
        "upstream_3plus_hop": cascade & (upstream_distance >= 3) & upstream_reachable,
    }
    overall = metrics(
        peak_prediction,
        peak_target,
        keyframe_prediction,
        keyframe_target,
        cascade,
    )
    report = {
        "protocol": {
            "model": "Frozen Y8-H: MeCaSNet + triple_blend",
            "data_root": args.data_root,
            "checkpoint": args.checkpoint,
            "split": f"seed=0 test split; first {len(test_ids)} events",
            "cascade_definition": (
                "not directly shocked and ground-truth peak loss > "
                f"{args.threshold}"
            ),
            "distance_definition": (
                "shortest unweighted path from any directly shocked node on "
                "the undirected event reach subgraph; distances >=4 pooled"
            ),
            "directed_position_definition": (
                "downstream distance follows supplier-to-customer edges from "
                "the shock set; upstream distance follows reversed edges"
            ),
            "aggregation": (
                "pooled event-node predictions; a firm appearing in multiple "
                "events contributes once per event"
            ),
        },
        "overall_cascade": overall,
        "by_hop": {
            name: metrics(
                peak_prediction,
                peak_target,
                keyframe_prediction,
                keyframe_target,
                mask,
            )
            for name, mask in strata.items()
        },
        "by_directed_position": {
            name: metrics(
                peak_prediction,
                peak_target,
                keyframe_prediction,
                keyframe_target,
                mask,
            )
            for name, mask in position_strata.items()
        },
        "by_directed_distance": {
            name: metrics(
                peak_prediction,
                peak_target,
                keyframe_prediction,
                keyframe_target,
                mask,
            )
            for name, mask in directed_distance_strata.items()
        },
    }

    expected_peak_r2 = 0.9335831257977932
    expected_keyframe_r2 = 0.7783541516718401
    if abs(overall["peak_r2"] - expected_peak_r2) > 5e-4:
        raise RuntimeError(
            f"Pooled peak R2 mismatch: {overall['peak_r2']:.6f} "
            f"vs expected {expected_peak_r2:.6f}"
        )
    if abs(overall["keyframe_r2_mean"] - expected_keyframe_r2) > 5e-4:
        raise RuntimeError(
            "Pooled keyframe R2 mismatch: "
            f"{overall['keyframe_r2_mean']:.6f} "
            f"vs expected {expected_keyframe_r2:.6f}"
        )
    if sum(item["n_event_nodes"] for item in report["by_hop"].values()) != overall["n_event_nodes"]:
        raise RuntimeError("Hop strata do not partition the cascade subset.")
    if sum(item["n_event_nodes"] for item in report["by_directed_position"].values()) != overall["n_event_nodes"]:
        raise RuntimeError("Directed-position strata do not partition the cascade subset.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"Created: {output}")
    print(json.dumps(report["by_hop"], indent=2, allow_nan=True))
    print(json.dumps(report["by_directed_position"], indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
