"""Predictive-uncertainty evaluation for the five-seed MeCaSNet ensemble.

The training protocol fixes the data split at Config.seed=0 while seed_start
changes model initialization. Validation events calibrate split-conformal
intervals; test events are used once for coverage and width evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from mecasnet.config import Config
from mecasnet.data import CascadeEventDataset, StaticNetwork, collate_single, split_event_ids

from mecasnet.evaluation import load_y8, r2, to_device


CASCADE_THRESHOLD = 0.05
SEVERE_THRESHOLD = 0.20


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoints", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--scale-floor", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_loader(dataset: CascadeEventDataset, num_workers: int,
                device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_single,
        pin_memory=device.type == "cuda",
    )


def materialise(loader: DataLoader, label: str) -> list[dict[str, Any]]:
    progress(f"Materialising {label} events from disk.")
    batches = list(loader)
    progress(f"Loaded {len(batches)} {label} events.")
    return batches


def targets_from_batches(batches: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "target": np.concatenate([batch["peak_loss"].numpy() for batch in batches]),
        "shock": np.concatenate([batch["shock_mask"].numpy() > 0.5 for batch in batches]),
        "event_id": np.concatenate([
            np.full(int(batch["Nr"]), int(batch["event_id"]), dtype=np.int64)
            for batch in batches
        ]),
    }


@torch.no_grad()
def predict(model: torch.nn.Module, batches: list[dict[str, Any]],
            device: torch.device, label: str) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    report_every = max(1, len(batches) // 10)
    started = time.perf_counter()
    for index, batch in enumerate(batches, 1):
        output = model(to_device(batch, device))
        predictions.append(output["peak"].detach().cpu().numpy())
        if index % report_every == 0 or index == len(batches):
            progress(f"{label}: {index}/{len(batches)} events complete.")
    progress(f"{label}: finished in {time.perf_counter() - started:.1f}s.")
    return np.concatenate(predictions)


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> tuple[float, float]:
    if scores.size == 0:
        raise ValueError("Cannot calibrate a conformal interval from zero scores.")
    level = min(math.ceil((scores.size + 1) * (1.0 - alpha)) / scores.size, 1.0)
    return float(np.quantile(scores, level, method="higher")), float(level)


def point_metrics(prediction: np.ndarray, target: np.ndarray,
                  shock: np.ndarray) -> dict[str, float | int]:
    cascade = (~shock) & (target > CASCADE_THRESHOLD)
    return {
        "n_nodes": int(target.size),
        "n_cascade_nodes": int(cascade.sum()),
        "mae_all": float(np.mean(np.abs(prediction - target))),
        "r2_peak_all": r2(prediction, target),
        "mae_peak_cascade": float(np.mean(np.abs(prediction[cascade] - target[cascade]))),
        "r2_peak_cascade": r2(prediction[cascade], target[cascade]),
    }


def stratum_masks(target: np.ndarray, shock: np.ndarray) -> dict[str, np.ndarray]:
    unshocked = ~shock
    return {
        "all_reach_nodes": np.ones(target.shape, dtype=bool),
        "directly_shocked": shock,
        "unshocked_no_or_small_cascade": unshocked & (target <= CASCADE_THRESHOLD),
        "cascade_all": unshocked & (target > CASCADE_THRESHOLD),
        "cascade_moderate": unshocked & (target > CASCADE_THRESHOLD)
                            & (target <= SEVERE_THRESHOLD),
        "cascade_severe": unshocked & (target > SEVERE_THRESHOLD),
    }


def interval_report(lower: np.ndarray, upper: np.ndarray, target: np.ndarray,
                    shock: np.ndarray) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, mask in stratum_masks(target, shock).items():
        if not bool(mask.any()):
            rows[name] = {"n": 0, "coverage": None, "mean_width": None,
                          "median_width": None}
            continue
        widths = upper[mask] - lower[mask]
        rows[name] = {
            "n": int(mask.sum()),
            "coverage": float(np.mean((target[mask] >= lower[mask])
                                      & (target[mask] <= upper[mask]))),
            "mean_width": float(np.mean(widths)),
            "median_width": float(np.median(widths)),
        }
    return rows


def clipped_interval(center: np.ndarray, radius: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    return np.clip(center - radius, 0.0, 1.0), np.clip(center + radius, 0.0, 1.0)


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    correlation = float(spearmanr(left, right).statistic)
    return correlation if np.isfinite(correlation) else None


def main() -> None:
    args = parse_args()
    if len(args.checkpoints) != 5:
        raise ValueError(f"Expected exactly five independently seeded checkpoints; got {len(args.checkpoints)}.")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must lie strictly between 0 and 1.")
    if args.scale_floor <= 0.0:
        raise ValueError("--scale-floor must be positive.")

    checkpoint_paths = [Path(path) for path in args.checkpoints]
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = Config(data_root=args.data_root, seed=0, event_scalars_mode="minimal")
    cfg.deq_iters = 1
    cfg.deq_grad_iters = 1
    net = StaticNetwork(cfg)
    progress("Preflighting all checkpoints with strict state-dict loading on CPU.")
    for checkpoint in checkpoint_paths:
        preflight_model = load_y8(checkpoint, cfg, net.Fv, torch.device("cpu"))
        del preflight_model
    progress("All five checkpoints are architecture-compatible.")
    events_dir = Path(args.data_root) / cfg.events_dir
    _, validation_ids, test_ids = split_event_ids(events_dir, cfg)
    validation_ids = validation_ids[:args.n_val]
    test_ids = test_ids[:args.n_test]
    if not validation_ids or not test_ids:
        raise ValueError("Validation and test splits must both be non-empty.")
    if set(validation_ids) & set(test_ids):
        raise RuntimeError("Validation and test event IDs overlap.")

    validation_batches = materialise(make_loader(
        CascadeEventDataset(cfg, net, validation_ids, train_mode=False),
        args.num_workers, device), "validation")
    test_batches = materialise(make_loader(
        CascadeEventDataset(cfg, net, test_ids, train_mode=False),
        args.num_workers, device), "test")
    validation_data = targets_from_batches(validation_batches)
    test_data = targets_from_batches(test_batches)

    validation_predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    for seed, checkpoint in enumerate(checkpoint_paths):
        progress(f"Loading seed {seed} checkpoint: {checkpoint}")
        model = load_y8(checkpoint, cfg, net.Fv, device)
        validation_predictions.append(predict(
            model, validation_batches, device, f"seed{seed} validation"))
        test_predictions.append(predict(model, test_batches, device, f"seed{seed} test"))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    validation_stack = np.stack(validation_predictions)
    test_stack = np.stack(test_predictions)
    validation_target = validation_data["target"]
    test_target = test_data["target"]
    test_shock = test_data["shock"]
    if validation_stack.shape[1] != validation_target.size:
        raise RuntimeError("Validation prediction/target alignment failed.")
    if test_stack.shape[1] != test_target.size:
        raise RuntimeError("Test prediction/target alignment failed.")

    seed0_validation = validation_stack[0]
    seed0_test = test_stack[0]
    ensemble_validation = validation_stack.mean(axis=0)
    ensemble_test = test_stack.mean(axis=0)
    validation_spread = validation_stack.std(axis=0, ddof=1)
    test_spread = test_stack.std(axis=0, ddof=1)

    seed0_radius, seed0_level = finite_sample_quantile(
        np.abs(seed0_validation - validation_target), args.alpha)
    ensemble_radius, ensemble_level = finite_sample_quantile(
        np.abs(ensemble_validation - validation_target), args.alpha)
    scaled_score = np.abs(ensemble_validation - validation_target) / (
        validation_spread + args.scale_floor)
    scaled_multiplier, scaled_level = finite_sample_quantile(scaled_score, args.alpha)

    seed0_lower, seed0_upper = clipped_interval(seed0_test, seed0_radius)
    ensemble_lower, ensemble_upper = clipped_interval(ensemble_test, ensemble_radius)
    scaled_lower, scaled_upper = clipped_interval(
        ensemble_test, scaled_multiplier * (test_spread + args.scale_floor))
    raw_lower = np.quantile(test_stack, args.alpha / 2.0, axis=0)
    raw_upper = np.quantile(test_stack, 1.0 - args.alpha / 2.0, axis=0)

    cascade = (~test_shock) & (test_target > CASCADE_THRESHOLD)
    absolute_error = np.abs(ensemble_test - test_target)
    report: dict[str, Any] = {
        "protocol": {
            "model": "MeCaSNet MeCaSNet + triple_blend",
            "checkpoint_count": len(checkpoint_paths),
            "checkpoints": [str(path) for path in checkpoint_paths],
            "split": "Config.seed=0 fixed for all training runs; model seed changes initialization only",
            "calibration_events": len(validation_ids),
            "test_events": len(test_ids),
            "nominal_coverage": 1.0 - args.alpha,
            "conformal_scope": "marginal node-level split conformal; validation and test events are disjoint",
            "severity_diagnostics": {
                "cascade": "unshocked node with ground-truth peak loss > 0.05",
                "moderate": "0.05 < ground-truth peak loss <= 0.20",
                "severe": "ground-truth peak loss > 0.20",
                "note": "ground-truth strata are diagnostic only and are not used to construct intervals",
            },
        },
        "point_performance": {
            "individual_seeds": [
                point_metrics(test_stack[index], test_target, test_shock)
                for index in range(test_stack.shape[0])
            ],
            "seed0": point_metrics(seed0_test, test_target, test_shock),
            "ensemble_mean": point_metrics(ensemble_test, test_target, test_shock),
        },
        "intervals": {
            "seed0_split_conformal": {
                "calibration_quantile": seed0_radius,
                "finite_sample_quantile_level": seed0_level,
                "by_severity": interval_report(seed0_lower, seed0_upper, test_target, test_shock),
            },
            "ensemble_mean_split_conformal": {
                "calibration_quantile": ensemble_radius,
                "finite_sample_quantile_level": ensemble_level,
                "by_severity": interval_report(
                    ensemble_lower, ensemble_upper, test_target, test_shock),
            },
            "ensemble_scaled_split_conformal": {
                "scale": "five-seed sample standard deviation + fixed scale floor",
                "scale_floor": args.scale_floor,
                "calibration_multiplier": scaled_multiplier,
                "finite_sample_quantile_level": scaled_level,
                "by_severity": interval_report(
                    scaled_lower, scaled_upper, test_target, test_shock),
            },
            "raw_ensemble_central_interval": {
                "warning": "descriptive ensemble spread only; five members do not provide calibrated 90% coverage",
                "by_severity": interval_report(raw_lower, raw_upper, test_target, test_shock),
            },
        },
        "ensemble_spread_diagnostics": {
            "mean_standard_deviation_all": float(test_spread.mean()),
            "mean_standard_deviation_cascade": float(test_spread[cascade].mean()),
            "spearman_spread_vs_absolute_error_all": safe_spearman(test_spread, absolute_error),
            "spearman_spread_vs_absolute_error_cascade": safe_spearman(
                test_spread[cascade], absolute_error[cascade]),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "y8_predictive_uncertainty.json"
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    progress(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
