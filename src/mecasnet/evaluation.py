"""Post-hoc reviewer experiments for the frozen Y8-H checkpoint.

This script does not retrain or alter model weights. It reports cascade-metric
threshold sensitivity and separates data-to-device, inference, and training
step costs on the exact Y8-H test split.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import CascadeEventDataset, StaticNetwork, collate_single, split_event_ids
from .factory import LEGACY_Y8_PROFILE, build_mecasnet
from .losses import total_loss


KEY_DAYS = (0, 5, 10, 20, 30, 50, 70, 100, 150, 199)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--benchmark-events", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.01, 0.02, 0.03, 0.05, 0.075,
                                 0.10, 0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def r2(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.size == 0 or float(target.std()) < 1e-2:
        return float("nan")
    residual = float(np.square(prediction - target).sum())
    total = float(np.square(target - target.mean()).sum()) + 1e-8
    return 1.0 - residual / total


def percentile(values: list[float], q: float) -> float | None:
    clean = [value for value in values if np.isfinite(value)]
    return float(np.percentile(clean, q)) if clean else None


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def build_y8(cfg: Config, feature_count: int) -> torch.nn.Module:
    """Build the archived Y8 checkpoint profile (not the paper default)."""
    return build_mecasnet(
        cfg, feature_count, profile=LEGACY_Y8_PROFILE, propagation_steps=4
    )


def load_y8(checkpoint: Path, cfg: Config, feature_count: int,
            device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload["state_dict"] if "state_dict" in payload else payload
    model = build_y8(cfg, feature_count)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device)


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader,
                        device: torch.device) -> tuple[list[dict[str, np.ndarray]], float]:
    model.eval()
    rows: list[dict[str, np.ndarray]] = []
    start = time.perf_counter()
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        output = model(batch)
        rows.append({
            "peak_pred": output["peak"].detach().cpu().numpy(),
            "peak_gt": batch["peak_loss"].detach().cpu().numpy(),
            "shock": batch["shock_mask"].detach().cpu().numpy() > 0.5,
            "kf_pred": output["u_keyframes"].detach().cpu().numpy(),
            "kf_gt": batch["u_keyframes"].detach().cpu().numpy(),
        })
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return rows, time.perf_counter() - start


def threshold_metrics(rows: list[dict[str, np.ndarray]], threshold: float) -> dict[str, Any]:
    peak_pred = np.concatenate([row["peak_pred"] for row in rows])
    peak_gt = np.concatenate([row["peak_gt"] for row in rows])
    shocked = np.concatenate([row["shock"] for row in rows])
    cascade = (~shocked) & (peak_gt > threshold)
    unshocked = ~shocked
    kf_pred = np.concatenate([row["kf_pred"] for row in rows], axis=1)
    kf_gt = np.concatenate([row["kf_gt"] for row in rows], axis=1)
    per_frame = [r2(kf_pred[index, cascade], kf_gt[index, cascade])
                 for index in range(kf_pred.shape[0])]
    valid = [value for value in per_frame if np.isfinite(value)]
    return {
        "threshold": threshold,
        "definition": "not_directly_shocked AND ground_truth_peak_loss > threshold",
        "capacity_ratio_cutoff": 1.0 - threshold,
        "n_total": int(peak_gt.size),
        "n_unshocked": int(unshocked.sum()),
        "n_cascade": int(cascade.sum()),
        "cascade_fraction_of_unshocked": float(cascade.sum() / max(unshocked.sum(), 1)),
        # These two all-node metrics are invariant to the cascade threshold.
        # They are included to reproduce every column in the main comparison table.
        "r2_pk": r2(peak_pred, peak_gt),
        "mae_pk": float(np.abs(peak_pred - peak_gt).mean()),
        "r2_pk_csc": r2(peak_pred[cascade], peak_gt[cascade]),
        "mae_pk_csc": float(np.abs(peak_pred[cascade] - peak_gt[cascade]).mean())
        if cascade.any() else float("nan"),
        "r2_kf_csc_mean": float(np.mean(valid)) if valid else float("nan"),
        "r2_kf_csc_by_day": dict(zip(map(str, KEY_DAYS), per_frame)),
    }


def cuda_time(fn, device: torch.device) -> float:
    if device.type != "cuda":
        start = time.perf_counter()
        fn()
        return (time.perf_counter() - start) * 1000
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end))


def benchmark(model: torch.nn.Module, raw_batches: list[dict[str, Any]], cfg: Config,
              device: torch.device) -> dict[str, Any]:
    if not raw_batches:
        raise RuntimeError("No test batches available for benchmark.")
    cached = [to_device(batch, device) for batch in raw_batches]
    warmup = cached[:min(10, len(cached))]
    model.eval()
    with torch.no_grad():
        for batch in warmup:
            model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    inference_ms = []
    with torch.no_grad():
        for batch in cached:
            inference_ms.append(cuda_time(lambda batch=batch: model(batch), device))
    peak_memory = (torch.cuda.max_memory_allocated(device) / 1024**2
                   if device.type == "cuda" else None)

    train_model = copy.deepcopy(model).train()
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_ms = []
    for batch in warmup:
        optimizer.zero_grad(set_to_none=True)
        output = train_model(batch)
        total_loss(output, batch, cfg)["total"].backward()
        optimizer.step()
    for batch in cached:
        def train_step(batch=batch):
            optimizer.zero_grad(set_to_none=True)
            output = train_model(batch)
            total_loss(output, batch, cfg)["total"].backward()
            optimizer.step()
        training_ms.append(cuda_time(train_step, device))
    peak_train_memory = (torch.cuda.max_memory_allocated(device) / 1024**2
                         if device.type == "cuda" else None)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "benchmark_events": len(cached),
        "parameters": parameter_count,
        "parameter_memory_mib_fp32": parameter_count * 4 / 1024**2,
        "inference_ms_per_event_median": percentile(inference_ms, 50),
        "inference_ms_per_event_p95": percentile(inference_ms, 95),
        "training_step_ms_per_event_median": percentile(training_ms, 50),
        "training_step_ms_per_event_p95": percentile(training_ms, 95),
        "peak_gpu_memory_mib_inference": peak_memory,
        "peak_gpu_memory_mib_training_step": peak_train_memory,
        "measurement": "CUDA-synchronised per-event timing; cached GPU batches exclude data loading",
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(data_root=args.data_root)
    cfg.deq_iters = 1
    cfg.deq_grad_iters = 1
    cfg.event_scalars_mode = "minimal"
    net = StaticNetwork(cfg)
    _, _, test_ids = split_event_ids(Path(args.data_root) / cfg.events_dir, cfg)
    test_ids = test_ids[:args.n_test]
    dataset = CascadeEventDataset(cfg, net, test_ids, train_mode=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=collate_single, pin_memory=device.type == "cuda")
    checkpoint_path = Path(args.checkpoint)
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = load_y8(checkpoint_path, cfg, net.Fv, device)

    rows, end_to_end_seconds = collect_predictions(model, loader, device)
    sensitivity = [threshold_metrics(rows, threshold) for threshold in args.thresholds]
    raw_batches = [dataset[index] for index in range(min(args.benchmark_events, len(dataset)))]
    cost = benchmark(model, raw_batches, cfg, device)
    cost["end_to_end_eval_seconds"] = end_to_end_seconds
    cost["end_to_end_eval_ms_per_event"] = end_to_end_seconds * 1000 / len(rows)
    cost["device"] = str(device)
    cost["gpu"] = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    cost["checkpoint_size_mib"] = checkpoint_path.stat().st_size / 1024**2
    trained_wall_seconds = checkpoint_payload.get("test", {}).get("wall_s")
    cost["full_training_wall_seconds"] = trained_wall_seconds
    cost["full_training_wall_minutes"] = (trained_wall_seconds / 60
                                           if trained_wall_seconds is not None else None)
    cost["best_epoch"] = checkpoint_payload.get("epoch")

    report = {
        "protocol": {
            "model": "Y8-H: MeCaSNet + triple_blend",
            "data_root": args.data_root,
            "split": "seed=0, 80/10/10, first 500 test events",
            "event_scalars_mode": "minimal",
            "threshold_roles": {
                "0.001": "reachability auxiliary BCE label only",
                "0.05": "trough-day auxiliary supervision and current headline cascade subset",
                "0.10_and_0.20": "post-hoc headline-metric sensitivity only; model weights unchanged",
            },
        },
        "threshold_sensitivity": sensitivity,
        "computational_cost": cost,
    }
    path = output_dir / "y8_reviewer_experiments.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    print("=== Threshold Sensitivity (frozen Y8-H checkpoint) ===")
    print(f"{'threshold':>9} {'N_cascade':>10} {'% unshocked':>13} {'MAE':>8} "
          f"{'R2pk':>8} {'R2pk,csc':>10} {'R2kf,csc':>10} {'MAE,csc':>9}")
    for row in sensitivity:
        print(f"{row['threshold']:9.2f} {row['n_cascade']:10d} "
              f"{100 * row['cascade_fraction_of_unshocked']:12.1f}% "
              f"{row['mae_pk']:8.4f} {row['r2_pk']:8.3f} "
              f"{row['r2_pk_csc']:10.3f} {row['r2_kf_csc_mean']:10.3f} "
              f"{row['mae_pk_csc']:9.4f}")
    print("\n=== Computational Cost ===")
    for key, value in cost.items():
        print(f"{key}: {value}")
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
