"""Robustness analysis for uncertainty in reconstructed supply-network inputs.

The simulator labels and model weights stay fixed. Only deployment
inputs are perturbed: relative input shares, observed edges, and day-0 damage
fractions.  This separates input-reconstruction robustness from simulator or
retraining effects.

Run from the repository root with an authorized event set and checkpoint; use
``--help`` for the required paths and perturbation controls.
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

from mecasnet.config import Config
from mecasnet.data import CascadeEventDataset, StaticNetwork, collate_single, split_event_ids

from mecasnet.evaluation import load_compat, r2, to_device


KEY_DAYS = (0, 5, 10, 20, 30, 50, 70, 100, 150, 199)
THRESHOLD = 0.05


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in batch.items()}


def recompute_hops(batch: dict[str, Any]) -> None:
    """Recompute the observed-graph shock-hop feature after edge deletion."""
    node_count = int(batch["Nr"])
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for src, dst in zip(batch["edge_src"].tolist(), batch["edge_dst"].tolist()):
        adjacency[src].append(dst)
        adjacency[dst].append(src)
    distance = np.full(node_count, 5, dtype=np.int64)
    frontier = np.flatnonzero(batch["shock_mask"].numpy() > 0.5).tolist()
    for node in frontier:
        distance[node] = 0
    for step in range(4):
        next_frontier: list[int] = []
        for node in frontier:
            for neighbour in adjacency[node]:
                if distance[neighbour] == 5:
                    distance[neighbour] = step + 1
                    next_frontier.append(neighbour)
        frontier = next_frontier
        if not frontier:
            break
    batch["shock_hop_oh"] = torch.from_numpy(np.eye(5, dtype=np.float32)[np.minimum(distance, 4)])


def perturb_edge_shares(batch: dict[str, Any], log_sigma: float,
                        rng: np.random.Generator) -> None:
    """Perturb relative shares while preserving each buyer's observed input mass."""
    edge_a = batch["edge_a"]
    if log_sigma <= 0 or edge_a.numel() == 0:
        return
    source = edge_a.numpy()
    noisy = source * np.exp(rng.normal(0.0, log_sigma, size=source.size)).astype(np.float32)
    destination = batch["edge_dst"].numpy()
    for node in np.unique(destination):
        mask = destination == node
        original_total = float(source[mask].sum())
        noisy_total = float(noisy[mask].sum())
        if noisy_total > 0:
            noisy[mask] *= original_total / noisy_total
    batch["edge_a"] = torch.from_numpy(noisy.astype(np.float32))
    batch["edge_outshare"] = (
        batch["edge_a"] * batch["P_ini"][batch["edge_dst"]]
        / batch["P_ini"][batch["edge_src"]].clamp_min(1e-6)
    )


def remove_edges(batch: dict[str, Any], probability: float,
                 rng: np.random.Generator) -> None:
    """Model an unobserved supplier-buyer relation; shares are not reallocated."""
    edge_count = batch["edge_src"].numel()
    if probability <= 0 or edge_count == 0:
        return
    keep = torch.from_numpy(rng.random(edge_count) >= probability)
    # Retain at least one edge if the original reach graph had edges.
    if not bool(keep.any()):
        keep[int(rng.integers(edge_count))] = True
    for key in ("edge_src", "edge_dst", "edge_a", "edge_outshare"):
        batch[key] = batch[key][keep]
    recompute_hops(batch)


def perturb_delta0(batch: dict[str, Any], sigma: float,
                   rng: np.random.Generator) -> None:
    """Add bounded measurement noise only where a day-0 shock is observed."""
    if sigma <= 0:
        return
    damaged = batch["shock_mask"].numpy() > 0.5
    values = batch["delta0"].numpy().copy()
    values[damaged] = np.clip(values[damaged] + rng.normal(0.0, sigma, damaged.sum()), 0.0, 1.0)
    batch["delta0"] = torch.from_numpy(values.astype(np.float32))


def perturb_baseline_capacity(batch: dict[str, Any], log_sigma: float,
                              rng: np.random.Generator) -> None:
    """Perturb P_ini and every model input derived from it.

    The static node vector contains a globally standardized log(1 + P_ini)
    feature. Its affine transform is recovered from the clean batch before
    updating the feature, while edge_outshare is recomputed from the perturbed
    supplier and buyer capacities with edge_a held fixed.
    """
    if log_sigma <= 0:
        return
    capacity = batch["P_ini"].numpy().astype(np.float64)
    noisy_capacity = capacity * np.exp(rng.normal(0.0, log_sigma, size=capacity.size))
    noisy_capacity = np.clip(noisy_capacity, 1e-6, None).astype(np.float32)

    features = batch["x_v"].numpy().copy()
    capacity_feature_index = features.shape[1] - 3
    clean_log_capacity = np.log1p(capacity)
    feature_slope, feature_intercept = np.polyfit(
        clean_log_capacity, features[:, capacity_feature_index], 1
    )
    features[:, capacity_feature_index] = (
        feature_slope * np.log1p(noisy_capacity) + feature_intercept
    ).astype(np.float32)
    batch["x_v"] = torch.from_numpy(features)
    batch["P_ini"] = torch.from_numpy(noisy_capacity)
    batch["edge_outshare"] = (
        batch["edge_a"] * batch["P_ini"][batch["edge_dst"]]
        / batch["P_ini"][batch["edge_src"]].clamp_min(1e-6)
    )


def apply_scenario(batch: dict[str, Any], scenario: dict[str, float],
                   rng: np.random.Generator) -> dict[str, Any]:
    perturbed = clone_batch(batch)
    perturb_edge_shares(perturbed, scenario.get("edge_log_sigma", 0.0), rng)
    remove_edges(perturbed, scenario.get("edge_missing_probability", 0.0), rng)
    perturb_delta0(perturbed, scenario.get("delta0_sigma", 0.0), rng)
    perturb_baseline_capacity(perturbed, scenario.get("capacity_log_sigma", 0.0), rng)
    return perturbed


@torch.no_grad()
def predict(model: torch.nn.Module, raw_batches: list[dict[str, Any]], scenario: dict[str, float],
            replicate_seed: int, device: torch.device, label: str) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(replicate_seed)
    rows: list[dict[str, np.ndarray]] = []
    model.eval()
    total = len(raw_batches)
    report_every = max(1, total // 10)
    started = time.perf_counter()
    progress(f"{label}: starting {total} event forwards.")
    for event_index, batch in enumerate(raw_batches, 1):
        observed = apply_scenario(batch, scenario, rng)
        output = model(to_device(observed, device))
        rows.append({
            "event_id": np.asarray([batch["event_id"]]),
            "peak_pred": output["peak"].detach().cpu().numpy(),
            "peak_gt": batch["peak_loss"].numpy(),
            "shock": batch["shock_mask"].numpy() > 0.5,
            "kf_pred": output["u_keyframes"].detach().cpu().numpy(),
            "kf_gt": batch["u_keyframes"].numpy(),
        })
        if event_index % report_every == 0 or event_index == total:
            elapsed = time.perf_counter() - started
            rate = event_index / max(elapsed, 1e-9)
            remaining = (total - event_index) / max(rate, 1e-9)
            progress(f"{label}: {event_index}/{total} events ({elapsed:.1f}s elapsed, "
                     f"~{remaining:.1f}s remaining).")
    return rows


def metrics(rows: list[dict[str, np.ndarray]]) -> dict[str, float | int]:
    peak_pred = np.concatenate([row["peak_pred"] for row in rows])
    peak_gt = np.concatenate([row["peak_gt"] for row in rows])
    shocked = np.concatenate([row["shock"] for row in rows])
    cascade = (~shocked) & (peak_gt > THRESHOLD)
    kf_pred = np.concatenate([row["kf_pred"] for row in rows], axis=1)
    kf_gt = np.concatenate([row["kf_gt"] for row in rows], axis=1)
    frame_r2 = [r2(kf_pred[frame, cascade], kf_gt[frame, cascade])
                for frame in range(kf_pred.shape[0])]
    valid_r2 = [value for value in frame_r2 if np.isfinite(value)]
    return {
        "n_events": len(rows),
        "n_cascade_nodes": int(cascade.sum()),
        "mae_all": float(np.abs(peak_pred - peak_gt).mean()),
        "r2_peak_all": r2(peak_pred, peak_gt),
        "mae_peak_cascade": float(np.abs(peak_pred[cascade] - peak_gt[cascade]).mean()),
        "r2_peak_cascade": r2(peak_pred[cascade], peak_gt[cascade]),
        "r2_keyframe_cascade_mean": float(np.mean(valid_r2)),
    }


def peak_cascade_r2(rows: list[dict[str, np.ndarray]]) -> float:
    """Headline metric without keyframe concatenation; used only in bootstrap."""
    peak_pred = np.concatenate([row["peak_pred"] for row in rows])
    peak_gt = np.concatenate([row["peak_gt"] for row in rows])
    shocked = np.concatenate([row["shock"] for row in rows])
    cascade = (~shocked) & (peak_gt > THRESHOLD)
    return r2(peak_pred[cascade], peak_gt[cascade])


def bootstrap_delta(baseline: list[dict[str, np.ndarray]],
                    perturbation_replicates: list[list[dict[str, np.ndarray]]],
                    metric: str, samples: int, seed: int) -> list[float]:
    """Paired event bootstrap, with one reconstruction draw per resample.

    This estimates uncertainty in the mean scenario degradation while retaining
    the dependence between clean and perturbed predictions for each event.
    """
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    count = len(baseline)
    report_every = max(1, samples // 10)
    started = time.perf_counter()
    progress(f"Bootstrap {metric}: starting {samples:,} paired resamples.")
    for index in range(samples):
        selected = rng.integers(0, count, size=count)
        perturbed = perturbation_replicates[int(rng.integers(len(perturbation_replicates)))]
        if metric != "r2_peak_cascade":
            raise ValueError(f"Bootstrap does not support metric: {metric}")
        values[index] = float(peak_cascade_r2([perturbed[i] for i in selected])
                              - peak_cascade_r2([baseline[i] for i in selected]))
        completed = index + 1
        if completed % report_every == 0 or completed == samples:
            elapsed = time.perf_counter() - started
            rate = completed / max(elapsed, 1e-9)
            remaining = (samples - completed) / max(rate, 1e-9)
            progress(f"Bootstrap {metric}: {completed:,}/{samples:,} resamples "
                     f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining).")
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def input_interval(predictions: list[list[dict[str, np.ndarray]]]) -> dict[str, float]:
    """Empirical 90% interval induced by jointly perturbed input reconstructions."""
    stacked = np.stack([
        np.concatenate([row["peak_pred"] for row in replicate]) for replicate in predictions
    ])
    target = np.concatenate([row["peak_gt"] for row in predictions[0]])
    lower, upper = np.quantile(stacked, [0.05, 0.95], axis=0)
    return {
        "replicates": len(predictions),
        "nominal_coverage": 0.90,
        "empirical_coverage": float(np.mean((target >= lower) & (target <= upper))),
        "mean_interval_width": float(np.mean(upper - lower)),
        "median_interval_width": float(np.median(upper - lower)),
        "scope": "input-reconstruction uncertainty only; does not include model epistemic uncertainty",
    }


def main() -> None:
    args = parse_args()
    if args.replicates < 2:
        raise ValueError("--replicates must be at least 2 to quantify reconstruction uncertainty.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress(f"Using device={device}; requested test events={args.n_test}; "
             f"replicates={args.replicates}; bootstrap={args.bootstrap:,}.")

    progress("Loading static network and constructing the fixed test split.")
    cfg = Config(data_root=args.data_root, seed=0, event_scalars_mode="minimal")
    cfg.deq_iters = 1
    cfg.deq_grad_iters = 1
    net = StaticNetwork(cfg)
    _, _, test_ids = split_event_ids(Path(args.data_root) / cfg.events_dir, cfg)
    dataset = CascadeEventDataset(cfg, net, test_ids[:args.n_test], train_mode=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_single, pin_memory=device.type == "cuda")
    progress(f"Materialising {len(dataset)} test-event batches from disk.")
    raw_batches = list(loader)
    progress(f"Loaded {len(raw_batches)} batches; loading frozen checkpoint.")
    model = load_compat(Path(args.checkpoint), cfg, net.Fv, device)
    progress("Frozen checkpoint loaded; evaluating clean baseline.")

    scenarios: dict[str, dict[str, float]] = {
        "baseline": {},
        "edge_share_cv_10pct": {"edge_log_sigma": 0.10},
        "edge_share_cv_20pct": {"edge_log_sigma": 0.20},
        "edge_share_cv_30pct": {"edge_log_sigma": 0.30},
        "missing_edges_5pct": {"edge_missing_probability": 0.05},
        "missing_edges_10pct": {"edge_missing_probability": 0.10},
        "missing_edges_20pct": {"edge_missing_probability": 0.20},
        "delta0_sigma_005": {"delta0_sigma": 0.05},
        "delta0_sigma_010": {"delta0_sigma": 0.10},
        "delta0_sigma_020": {"delta0_sigma": 0.20},
        "baseline_capacity_cv_10pct": {"capacity_log_sigma": 0.10},
        "baseline_capacity_cv_20pct": {"capacity_log_sigma": 0.20},
        "baseline_capacity_cv_30pct": {"capacity_log_sigma": 0.30},
        "joint_plausible": {"edge_log_sigma": 0.20, "edge_missing_probability": 0.10,
                    "delta0_sigma": 0.10, "capacity_log_sigma": 0.20},
    }
    baseline_rows = predict(model, raw_batches, scenarios["baseline"], args.seed, device,
                            "Baseline")
    baseline_metrics = metrics(baseline_rows)
    progress(f"Baseline complete: R2_peak,cascade={baseline_metrics['r2_peak_cascade']:.4f}.")
    report: dict[str, Any] = {
        "protocol": {
            "model": "MeCaSNet compatibility profile",
            "split": "seed=0, first test events only; identical events for every scenario",
            "n_test": len(raw_batches),
            "event_scalars_mode": "minimal",
            "cascade_metric": "unshocked nodes with ground-truth peak loss > 0.05",
            "replicates_per_stochastic_scenario": args.replicates,
            "paired_bootstrap_resamples": args.bootstrap,
            "seed": args.seed,
            "edge_share": "multiplicative log-normal noise, re-normalised within each buyer's visible incoming edges",
            "missing_edges": "uniform independent edge omission, no share reallocation, observed hop encoding recomputed",
            "delta0": "zero-mean additive Gaussian measurement noise, clipped to [0,1], shocked firms only",
            "baseline_capacity": "multiplicative log-normal P_ini noise; standardized log(1 + P_ini) node feature and capacity-derived edge_outshare updated consistently while edge_a is held fixed",
        },
        "baseline": baseline_metrics,
        "scenarios": {},
        "predictive_uncertainty_discussion": {
            "current_checkpoint": "point prediction only; it does not represent parameter epistemic uncertainty",
            "reported_proxy": "joint input-perturbation ensemble yields intervals for reconstruction uncertainty only",
            "recommended_deployment_upgrade": "train 5 independently seeded checkpoints and calibrate conformal or quantile intervals on a held-out temporal/site split; report coverage and interval width by cascade severity",
        },
    }

    for scenario_index, (name, scenario) in enumerate(scenarios.items()):
        if name == "baseline":
            continue
        progress(f"Scenario {scenario_index}/{len(scenarios) - 1}: {name}; "
                 f"running {args.replicates} reconstruction replicates.")
        replicate_rows = []
        for replicate in range(args.replicates):
            replicate_rows.append(predict(
                model, raw_batches, scenario,
                args.seed + scenario_index * 10_000 + replicate, device,
                f"{name} replicate {replicate + 1}/{args.replicates}",
            ))
        progress(f"{name}: computing aggregate metrics and confidence interval.")
        replicate_metrics = [metrics(rows) for rows in replicate_rows]
        metric_summary: dict[str, Any] = {}
        for metric in ("mae_all", "r2_peak_all", "mae_peak_cascade",
                       "r2_peak_cascade", "r2_keyframe_cascade_mean"):
            samples = np.asarray([row[metric] for row in replicate_metrics], dtype=float)
            summary: dict[str, Any] = {
                "mean": float(samples.mean()),
                "std_across_reconstructions": float(samples.std(ddof=1)),
                "delta_vs_baseline_mean": float(samples.mean() - baseline_metrics[metric]),
            }
            if metric == "r2_peak_cascade":
                summary["paired_bootstrap_95ci_for_delta"] = bootstrap_delta(
                    baseline_rows, replicate_rows, metric, args.bootstrap,
                    args.seed + scenario_index * 100 + len(metric),
                )
            metric_summary[metric] = summary
        record: dict[str, Any] = {
            "perturbation": scenario,
            "metrics": metric_summary,
            "replicate_metrics": replicate_metrics,
        }
        if name == "joint_plausible":
            record["input_uncertainty_interval"] = input_interval(replicate_rows)
        report["scenarios"][name] = record
        progress(f"{name}: complete; R2_peak,cascade="
                 f"{metric_summary['r2_peak_cascade']['mean']:.4f} "
                 f"(baseline {baseline_metrics['r2_peak_cascade']:.4f}).")

    path = output_dir / "input_uncertainty.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    progress(f"Saved: {path}")


if __name__ == "__main__":
    main()
