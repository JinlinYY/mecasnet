"""Evaluate fixed compatibility-profile weights after excluding similar events.

This is a post-hoc sensitivity analysis only: it never trains, fine-tunes, or
selects a checkpoint. Event subsets are determined solely from the precomputed
train/test similarity audit, then evaluated with the frozen primary checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mecasnet.config import Config
from mecasnet.data import CascadeEventDataset, StaticNetwork, collate_single, split_event_ids
from mecasnet.evaluation import collect_predictions, load_compat, threshold_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def as_bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"Expected a Boolean CSV field, got {value!r}.")
    return value == "True"


def load_test_audit(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    audit: dict[int, dict[str, Any]] = {}
    for row in rows:
        event_id = int(row["event_id"])
        if event_id in audit:
            raise ValueError(f"Duplicate test event {event_id} in {path}.")
        audit[event_id] = {
            "strict_near_duplicate": as_bool(row["strict_near_duplicate"]),
            "relaxed_near_duplicate": as_bool(row["relaxed_near_duplicate"]),
            "has_same_target_set": as_bool(row["has_same_target_set"]),
            "max_target_jaccard": float(row["max_target_jaccard_any_reference"]),
        }
    if not audit:
        raise ValueError(f"No test rows found in similarity audit: {path}")
    return audit


def evaluate_subset(model: torch.nn.Module, cfg: Config, net: StaticNetwork,
                    event_ids: list[int], device: torch.device,
                    num_workers: int, threshold: float) -> dict[str, Any]:
    dataset = CascadeEventDataset(cfg, net, event_ids, train_mode=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=collate_single, pin_memory=device.type == "cuda",
    )
    rows, elapsed_seconds = collect_predictions(model, loader, device)
    metrics = threshold_metrics(rows, threshold)
    metrics["event_count"] = len(event_ids)
    metrics["event_ids"] = event_ids
    metrics["inference_wall_seconds"] = elapsed_seconds
    return metrics


def summarize_jaccard(event_ids: list[int], audit: dict[int, dict[str, Any]]) -> dict[str, float]:
    values = np.asarray([audit[event_id]["max_target_jaccard"] for event_id in event_ids])
    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    audit_path = Path(args.audit_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = Config(data_root=str(data_root))
    cfg.deq_iters = 1
    cfg.deq_grad_iters = 1
    cfg.event_scalars_mode = "minimal"
    _, _, test_ids = split_event_ids(data_root / cfg.events_dir, cfg)
    audit = load_test_audit(audit_path)
    if set(test_ids) != set(audit):
        missing = sorted(set(test_ids) - set(audit))
        unexpected = sorted(set(audit) - set(test_ids))
        raise ValueError(
            "Audit/test split mismatch. Re-run run_event_split_audit.ps1. "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    subsets = {
        "all_test_events": test_ids,
        "exclude_strict_near_duplicates": [
            event_id for event_id in test_ids
            if not audit[event_id]["strict_near_duplicate"]
        ],
        "exclude_relaxed_near_duplicates": [
            event_id for event_id in test_ids
            if not audit[event_id]["relaxed_near_duplicate"]
        ],
        "exclude_same_target_set": [
            event_id for event_id in test_ids
            if not audit[event_id]["has_same_target_set"]
        ],
        "exclude_jaccard_ge_0_80": [
            event_id for event_id in test_ids
            if audit[event_id]["max_target_jaccard"] < 0.80
        ],
        "only_jaccard_lt_0_50": [
            event_id for event_id in test_ids
            if audit[event_id]["max_target_jaccard"] < 0.50
        ],
    }
    if len(subsets["all_test_events"]) != 500:
        raise ValueError(f"Expected 500 test events, got {len(test_ids)}.")
    if any(not event_ids for event_ids in subsets.values()):
        raise ValueError("A requested similarity subset is empty.")

    net = StaticNetwork(cfg)
    model = load_compat(Path(args.checkpoint), cfg, net.Fv, device)
    results: dict[str, dict[str, Any]] = {}
    for name, event_ids in subsets.items():
        print(f"[evaluate] {name}: {len(event_ids)} events", flush=True)
        result = evaluate_subset(
            model, cfg, net, event_ids, device, args.num_workers, args.threshold,
        )
        result["max_train_target_jaccard"] = summarize_jaccard(event_ids, audit)
        results[name] = result

    baseline = results["all_test_events"]
    for name, result in results.items():
        result["delta_vs_all_test"] = {
            "r2_pk": result["r2_pk"] - baseline["r2_pk"],
            "r2_pk_csc": result["r2_pk_csc"] - baseline["r2_pk_csc"],
            "mae_pk": result["mae_pk"] - baseline["mae_pk"],
            "mae_pk_csc": result["mae_pk_csc"] - baseline["mae_pk_csc"],
            "r2_kf_csc_mean": result["r2_kf_csc_mean"] - baseline["r2_kf_csc_mean"],
        }

    report = {
        "protocol": {
            "model": "MeCaSNet compatibility profile",
            "checkpoint": str(Path(args.checkpoint)),
            "event_scalars_mode": "minimal",
            "threshold": args.threshold,
            "cascade_definition": "not_directly_shocked AND ground_truth_peak_loss > threshold",
            "similarity_reference": "For each test event, maximum target-set Jaccard over all 4,000 train events.",
            "no_retraining": True,
            "no_checkpoint_selection": True,
        },
        "audit_csv": str(audit_path),
        "subsets": results,
    }
    path = output_dir / "event_similarity_sensitivity.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    print("=== Event-Similarity Sensitivity ===")
    print(f"{'subset':<36} {'events':>6} {'R2pk':>8} {'R2pk,csc':>10} {'MAE,csc':>9} {'R2kf,csc':>10}")
    for name, result in results.items():
        print(f"{name:<36} {result['event_count']:6d} {result['r2_pk']:8.3f} "
              f"{result['r2_pk_csc']:10.3f} {result['mae_pk_csc']:9.4f} "
              f"{result['r2_kf_csc_mean']:10.3f}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
