"""Measure learned recovery parameters from fixed model weights.

Captures the trimodal decoder's raw 13-output param_head and applies the exact
forward mappings used by KSGATv3. Reports all-node and severity-stratified
statistics over the fixed test split without changing model weights.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mecasnet.config import Config
from mecasnet.data import CascadeEventDataset, StaticNetwork, collate_single, split_event_ids

from mecasnet.evaluation import load_compat, to_device


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def describe(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def describe_pair(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    """Summarize paired component values, not merely their marginal distributions."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"Paired arrays have different shapes: {left.shape} vs {right.shape}.")
    difference = left - right
    correlation = (float(np.corrcoef(left, right)[0, 1])
                   if left.size > 1 and left.std() > 0 and right.std() > 0 else None)
    absolute = np.abs(difference)
    return {
        "n": int(left.size),
        "pearson_correlation": correlation,
        "mean_signed_difference_days": float(difference.mean()),
        "mean_absolute_difference_days": float(absolute.mean()),
        "median_absolute_difference_days": float(np.median(absolute)),
        "fraction_within_5_days": float((absolute <= 5.0).mean()),
        "fraction_within_10_days": float((absolute <= 10.0).mean()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress(f"Loading network and frozen checkpoint on {device}.")

    cfg = Config(data_root=args.data_root, seed=0, event_scalars_mode="minimal")
    cfg.deq_iters = 1
    cfg.deq_grad_iters = 1
    net = StaticNetwork(cfg)
    _, _, test_ids = split_event_ids(Path(args.data_root) / cfg.events_dir, cfg)
    test_ids = test_ids[:args.n_test]
    dataset = CascadeEventDataset(cfg, net, test_ids, train_mode=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_single,
                        pin_memory=device.type == "cuda")
    model = load_compat(Path(args.checkpoint), cfg, net.Fv, device)
    if getattr(model, "decoder_mode", None) != "param3":
        raise RuntimeError(f"Expected trimodal decoder_mode='param3', got {model.decoder_mode!r}.")

    captured: dict[str, torch.Tensor] = {}

    def capture_param_head(_module, _inputs, output):
        captured["raw"] = output.detach()

    hook = model.param_head.register_forward_hook(capture_param_head)
    component_parameters = (
        "P1", "mu1", "sigma1", "T_r1",
        "P2", "mu2", "sigma2", "T_r2",
        "P3", "mu3", "sigma3", "T_r3",
    )
    groups: dict[str, dict[str, list[np.ndarray]]] = {
        name: {parameter: [] for parameter in ("tau_u", "c", *component_parameters)}
        for name in ("all", "shocked", "unshocked", "cascade")
    }
    component_effects: dict[str, list[np.ndarray]] = {"G1": [], "G2": [], "G3": []}
    report_every = max(1, len(dataset) // 10)
    started = time.perf_counter()
    model.eval()
    try:
        with torch.no_grad():
            for event_index, raw_batch in enumerate(loader, 1):
                batch = to_device(raw_batch, device)
                captured.clear()
                output = model(batch)
                raw = captured.get("raw")
                if raw is None:
                    raise RuntimeError("param_head hook did not capture an output.")
                if raw.ndim != 2 or raw.shape[1] != 13:
                    raise RuntimeError(
                        f"Expected param3 raw output shape (Nr, 13), got {tuple(raw.shape)}."
                    )

                values = {
                    "tau_u": output["tau_u_learned"].detach().cpu().numpy(),
                    "c": output["c_learned"].detach().cpu().numpy(),
                    "P1": torch.sigmoid(raw[:, 0]).cpu().numpy(),
                    "mu1": (model.p3_mu1_off + model.p3_mu1_scale * torch.sigmoid(raw[:, 1])).cpu().numpy(),
                    "sigma1": (model.p3_sigma1_off + model.p3_sigma1_scale * torch.sigmoid(raw[:, 2])).cpu().numpy(),
                    "T_r1": (50.0 + 250.0 * torch.sigmoid(raw[:, 3])).cpu().numpy(),
                    "P2": torch.sigmoid(raw[:, 4]).cpu().numpy(),
                    "mu2": (model.p3_mu2_off + model.p3_mu2_scale * torch.sigmoid(raw[:, 5])).cpu().numpy(),
                    "sigma2": (model.p3_sigma2_off + model.p3_sigma2_scale * torch.sigmoid(raw[:, 6])).cpu().numpy(),
                    "T_r2": (50.0 + 250.0 * torch.sigmoid(raw[:, 7])).cpu().numpy(),
                    "P3": torch.sigmoid(raw[:, 8]).cpu().numpy(),
                    "mu3": (model.p3_mu3_off + model.p3_mu3_scale * torch.sigmoid(raw[:, 9])).cpu().numpy(),
                    "sigma3": (model.p3_sigma3_off + model.p3_sigma3_scale * torch.sigmoid(raw[:, 10])).cpu().numpy(),
                    "T_r3": (50.0 + 250.0 * torch.sigmoid(raw[:, 11])).cpu().numpy(),
                }
                shock = raw_batch["shock_mask"].numpy() > 0.5
                peak_gt = raw_batch["peak_loss"].numpy()
                masks = {
                    "all": np.ones(shock.shape, dtype=bool),
                    "shocked": shock,
                    "unshocked": ~shock,
                    "cascade": (~shock) & (peak_gt > 0.05),
                }
                for group, mask in masks.items():
                    for parameter, array in values.items():
                        groups[group][parameter].append(array[mask])

                # Re-run with one amplitude logit forced near zero. The mean absolute
                # change over keyframes is a direct use test for each component.
                baseline_kf = output["u_keyframes"].detach()
                original_hook = hook
                original_hook.remove()
                try:
                    for component, amplitude_index in (("G1", 0), ("G2", 4), ("G3", 8)):
                        def disable_component(_module, _inputs, raw_output,
                                              amplitude_index=amplitude_index):
                            modified = raw_output.clone()
                            modified[:, amplitude_index] = -20.0
                            return modified

                        disable_hook = model.param_head.register_forward_hook(disable_component)
                        try:
                            ablated = model(batch)["u_keyframes"].detach()
                        finally:
                            disable_hook.remove()
                        component_effects[component].append(
                            (ablated - baseline_kf).abs().mean(dim=0).cpu().numpy()
                        )
                finally:
                    hook = model.param_head.register_forward_hook(capture_param_head)

                if event_index % report_every == 0 or event_index == len(dataset):
                    elapsed = time.perf_counter() - started
                    rate = event_index / max(elapsed, 1e-9)
                    remaining = (len(dataset) - event_index) / max(rate, 1e-9)
                    progress(f"{event_index}/{len(dataset)} events; {elapsed:.1f}s elapsed, "
                             f"~{remaining:.1f}s remaining.")
    finally:
        hook.remove()

    statistics: dict[str, Any] = {}
    paired_values: dict[str, dict[str, np.ndarray]] = {}
    for group, parameters in groups.items():
        paired_values[group] = {
            parameter: np.concatenate(chunks) if chunks else np.array([])
            for parameter, chunks in parameters.items()
        }
        statistics[group] = {
            parameter: describe(values) for parameter, values in paired_values[group].items()
        }
    paired_diagnostics = {
        group: {
            "T_r1_vs_T_r2": describe_pair(values["T_r1"], values["T_r2"]),
            "T_r1_vs_T_r3": describe_pair(values["T_r1"], values["T_r3"]),
            "T_r2_vs_T_r3": describe_pair(values["T_r2"], values["T_r3"]),
        }
        for group, values in paired_values.items()
    }
    active_component_statistics: dict[str, dict[str, float | int]] = {}
    for component_index, component in enumerate(("G1", "G2", "G3"), 1):
        amplitude = paired_values["all"][f"P{component_index}"]
        recovery = paired_values["all"][f"T_r{component_index}"]
        active = amplitude >= 0.10
        active_component_statistics[component] = {
            "amplitude_threshold": 0.10,
            "n_active": int(active.sum()),
            "active_fraction": float(active.mean()),
            "T_r_days": describe(recovery[active]),
            "mean_keyframe_change_when_disabled": describe(
                np.concatenate(component_effects[component]) if component_effects[component] else np.array([])
            ),
        }
    report = {
        "protocol": {
            "model": "MeCaSNet compatibility profile",
            "checkpoint": str(Path(args.checkpoint)),
            "data_root": str(Path(args.data_root)),
            "split": "seed=0, first 500 test events",
            "n_events": len(dataset),
            "event_scalars_mode": "minimal",
            "cascade_definition": "not directly shocked and ground-truth peak loss > 0.05",
            "aggregation": "event-node predictions; the same firm may appear in multiple events",
            "component_ablation": "Each component is disabled by setting its amplitude logit to -20; effect is mean absolute change over predicted keyframes per event-node.",
        },
        "definitions": {
            "tau_u": "0.5 + 4.5*sigmoid(z_tau), independent PhysicsParameterGenerator head, days",
            "c": "0.1 + 0.9*sigmoid(z_c), same PhysicsParameterGenerator head",
            "T_r1": "50 + 250*sigmoid(param_head[:,3]), early G1 recovery constant, days",
            "T_r2": "50 + 250*sigmoid(param_head[:,7]), middle G2 recovery constant, days",
            "T_r3": "50 + 250*sigmoid(param_head[:,11]), late G3 recovery constant, days",
            "mu1_mu2_mu3": "component timing locations; param3 ranges are G1 [0,40], G2 [20,90], G3 [100,220] days",
        },
        "statistics": statistics,
        "paired_recovery_diagnostics": paired_diagnostics,
        "active_component_statistics": active_component_statistics,
    }
    path = output_dir / "recovery_parameters.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    progress(f"Saved: {path}")
    for parameter, stats in statistics["all"].items():
        if parameter not in ("tau_u", "c", "T_r1", "T_r2", "T_r3"):
            continue
        progress(f"{parameter}: {stats['mean']:.4f} ± {stats['std']:.4f}; "
                 f"range {stats['min']:.4f}–{stats['max']:.4f}; n={stats['n']:,}.")
    comparison = paired_diagnostics["all"]["T_r1_vs_T_r3"]
    progress("T_r1 vs T_r3 paired check: "
             f"r={comparison['pearson_correlation']:.4f}, "
             f"median |Δ|={comparison['median_absolute_difference_days']:.2f} d, "
             f"within 10 d={100 * comparison['fraction_within_10_days']:.2f}%.")
    for component, stats in active_component_statistics.items():
        effect = stats["mean_keyframe_change_when_disabled"]
        progress(f"{component} active (P≥0.10): {100 * stats['active_fraction']:.2f}%; "
                 f"disable-effect |Δu_kf|={effect['mean']:.5f} "
                 f"(median {effect['median']:.5f}).")


if __name__ == "__main__":
    main()
