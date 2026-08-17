r"""Compare propagation-depth robustness across matched model seeds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


from mecasnet.config import Config
from mecasnet.data import (
    CascadeEventDataset,
    StaticNetwork,
    collate_single,
    split_event_ids,
)


def repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Could not locate the repository root")


OVERALL_EXPERIMENT = (
    repository_root()
    / "scripts"
    / "experiments"
    / "main"
    / "section_4_03_overall_performance"
)
if str(OVERALL_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(OVERALL_EXPERIMENT))

from threshold_multiseed import (
    build_model,
    checkpoint_paths,
)
from propagation_depth import directed_distances
from mecasnet.evaluation import r2, to_device


MODELS = ("MeCaSNet", "DirGNN")
DISTANCES = ("downstream_1_hop", "downstream_2_hop", "downstream_3plus_hop")
POSITIONS = ("feedforward_downstream", "feedback_connected_downstream")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | int]:
    valid = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return {
        "n": int(valid.size),
        "mean": float(valid.mean()) if valid.size else float("nan"),
        "std": float(valid.std(ddof=1)) if valid.size > 1 else 0.0,
    }


def pooled_metrics(records: dict[str, list[np.ndarray]]) -> dict[str, float | int]:
    peak_prediction = np.concatenate(records["peak_prediction"])
    peak_target = np.concatenate(records["peak_target"])
    keyframe_prediction = np.concatenate(records["keyframe_prediction"], axis=1)
    keyframe_target = np.concatenate(records["keyframe_target"], axis=1)
    flat_prediction = keyframe_prediction.reshape(-1)
    flat_target = keyframe_target.reshape(-1)
    absolute_peak_error = np.abs(peak_prediction - peak_target)
    return {
        "n_event_nodes": int(peak_target.size),
        "peak_mae": float(absolute_peak_error.mean()),
        "peak_relative_mae": float(
            (absolute_peak_error / peak_target.clip(min=0.05)).mean()
        ),
        "peak_r2": r2(peak_prediction, peak_target),
        "trajectory_mae": float(np.abs(flat_prediction - flat_target).mean()),
        "trajectory_r2": r2(flat_prediction, flat_target),
        "ground_truth_peak_mean": float(peak_target.mean()),
    }


def bootstrap_mean(values: np.ndarray, resamples: int, seed: int) -> dict[str, Any]:
    if values.size == 0:
        return {"n_events": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    return {
        "n_events": int(values.size),
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint: Path,
    model_name: str,
    cfg: Config,
    network: StaticNetwork,
    loader: DataLoader,
    threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    model = build_model(model_name, cfg, network.Fv).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    pooled = {
        key: {
            "peak_prediction": [],
            "peak_target": [],
            "keyframe_prediction": [],
            "keyframe_target": [],
        }
        for key in DISTANCES
    }
    pooled_positions = {
        key: {
            "peak_prediction": [],
            "peak_target": [],
            "keyframe_prediction": [],
            "keyframe_target": [],
        }
        for key in POSITIONS
    }
    event_mae: dict[str, dict[str, float]] = {}
    event_relative_mae: dict[str, dict[str, float]] = {}
    event_position_mae: dict[str, dict[str, float]] = {}

    for raw_batch in loader:
        event_id = str(int(raw_batch["event_id"]))
        batch = to_device(raw_batch, device)
        output = model(batch)
        peak_prediction = output["peak"].detach().cpu().numpy()
        peak_target = batch["peak_loss"].detach().cpu().numpy()
        keyframe_prediction = output["u_keyframes"].detach().cpu().numpy()
        keyframe_target = batch["u_keyframes"].detach().cpu().numpy()
        shocked = batch["shock_mask"].detach().cpu().numpy() > 0.5
        downstream, upstream = directed_distances(batch)
        cascade = (~shocked) & (peak_target > threshold)
        masks = {
            "downstream_1_hop": cascade & (downstream == 1),
            "downstream_2_hop": cascade & (downstream == 2),
            "downstream_3plus_hop": cascade & np.isfinite(downstream) & (downstream >= 3),
        }
        position_masks = {
            "feedforward_downstream": cascade & np.isfinite(downstream) & ~np.isfinite(upstream),
            "feedback_connected_downstream": cascade & np.isfinite(downstream) & np.isfinite(upstream),
        }
        event_mae[event_id] = {}
        event_relative_mae[event_id] = {}
        event_position_mae[event_id] = {}
        for key, mask in masks.items():
            if not mask.any():
                continue
            pooled[key]["peak_prediction"].append(peak_prediction[mask])
            pooled[key]["peak_target"].append(peak_target[mask])
            pooled[key]["keyframe_prediction"].append(keyframe_prediction[:, mask])
            pooled[key]["keyframe_target"].append(keyframe_target[:, mask])
            event_mae[event_id][key] = float(
                np.abs(peak_prediction[mask] - peak_target[mask]).mean()
            )
            event_relative_mae[event_id][key] = float(
                (
                    np.abs(peak_prediction[mask] - peak_target[mask])
                    / peak_target[mask].clip(min=threshold)
                ).mean()
            )
        for key, mask in position_masks.items():
            if not mask.any():
                continue
            pooled_positions[key]["peak_prediction"].append(peak_prediction[mask])
            pooled_positions[key]["peak_target"].append(peak_target[mask])
            pooled_positions[key]["keyframe_prediction"].append(keyframe_prediction[:, mask])
            pooled_positions[key]["keyframe_target"].append(keyframe_target[:, mask])
            event_position_mae[event_id][key] = float(
                np.abs(peak_prediction[mask] - peak_target[mask]).mean()
            )

    by_distance = {}
    for key in DISTANCES:
        if not pooled[key]["peak_target"]:
            raise RuntimeError(f"Empty distance stratum for {model_name}: {key}")
        by_distance[key] = pooled_metrics(pooled[key])
    by_position = {}
    for key in POSITIONS:
        if not pooled_positions[key]["peak_target"]:
            raise RuntimeError(f"Empty position stratum for {model_name}: {key}")
        by_position[key] = pooled_metrics(pooled_positions[key])
    seed = int(checkpoint.parent.name.removeprefix("seed"))
    return {
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_best_epoch": payload.get("epoch"),
        "by_distance": by_distance,
        "by_position": by_position,
        "event_peak_mae": event_mae,
        "event_peak_relative_mae": event_relative_mae,
        "event_position_peak_mae": event_position_mae,
    }


def seed_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"by_distance": {}, "by_position": {}}
    for distance in DISTANCES:
        result["by_distance"][distance] = {
            metric: summarize([
                float(run["by_distance"][distance][metric]) for run in runs
            ])
            for metric in (
                "peak_mae",
                "peak_relative_mae",
                "peak_r2",
                "trajectory_mae",
                "trajectory_r2",
            )
        }
        counts = {int(run["by_distance"][distance]["n_event_nodes"]) for run in runs}
        if len(counts) != 1:
            raise RuntimeError(f"Target stratum count changed across seeds: {distance}")
        result["by_distance"][distance]["n_event_nodes"] = counts.pop()
    for position in POSITIONS:
        result["by_position"][position] = {
            metric: summarize([
                float(run["by_position"][position][metric]) for run in runs
            ])
            for metric in (
                "peak_mae",
                "peak_relative_mae",
                "peak_r2",
                "trajectory_mae",
                "trajectory_r2",
            )
        }
        counts = {int(run["by_position"][position]["n_event_nodes"]) for run in runs}
        if len(counts) != 1:
            raise RuntimeError(f"Target position count changed across seeds: {position}")
        result["by_position"][position]["n_event_nodes"] = counts.pop()
    return result


def seed_averaged_event_metric(
    runs: list[dict[str, Any]],
    metric_key: str,
    strata: tuple[str, ...] = DISTANCES,
) -> dict[str, dict[str, float]]:
    event_ids = sorted({event_id for run in runs for event_id in run[metric_key]})
    result: dict[str, dict[str, float]] = {}
    for event_id in event_ids:
        result[event_id] = {}
        for distance in strata:
            values = [
                run[metric_key].get(event_id, {}).get(distance)
                for run in runs
            ]
            valid = [float(value) for value in values if value is not None]
            if valid:
                result[event_id][distance] = float(np.mean(valid))
    return result


def paired_event_analysis(
    per_model: dict[str, dict[str, dict[str, float]]],
    resamples: int,
    metric_name: str,
) -> dict[str, Any]:
    degradation = {}
    for model_name, events in per_model.items():
        values = np.asarray([
            row["downstream_2_hop"] - row["downstream_1_hop"]
            for row in events.values()
            if "downstream_1_hop" in row and "downstream_2_hop" in row
        ], dtype=float)
        degradation[model_name] = bootstrap_mean(values, resamples, 20260811)

    common_events = sorted(set(per_model["MeCaSNet"]) & set(per_model["DirGNN"]))
    two_hop_difference = []
    degradation_difference = []
    for event_id in common_events:
        phy = per_model["MeCaSNet"][event_id]
        baseline = per_model["DirGNN"][event_id]
        if "downstream_2_hop" in phy and "downstream_2_hop" in baseline:
            two_hop_difference.append(
                phy["downstream_2_hop"] - baseline["downstream_2_hop"]
            )
        needed = ("downstream_1_hop", "downstream_2_hop")
        if all(key in phy and key in baseline for key in needed):
            degradation_difference.append(
                (phy["downstream_2_hop"] - phy["downstream_1_hop"])
                - (baseline["downstream_2_hop"] - baseline["downstream_1_hop"])
            )
    return {
        "within_model_2hop_minus_1hop_peak_mae": degradation,
        "mecasnet_minus_dirgnn_2hop_peak_mae": bootstrap_mean(
            np.asarray(two_hop_difference, dtype=float), resamples, 20260812
        ),
        "difference_in_distance_degradation": bootstrap_mean(
            np.asarray(degradation_difference, dtype=float), resamples, 20260813
        ),
        "sign_convention": (
            "For model comparisons, negative values favor MeCaSNet. "
            "For within-model degradation, positive values indicate higher "
            f"{metric_name} at 2 hops than at 1 hop."
        ),
    }


def paired_position_analysis(
    per_model: dict[str, dict[str, dict[str, float]]],
    resamples: int,
) -> dict[str, Any]:
    degradation = {}
    for model_name, events in per_model.items():
        values = np.asarray([
            row["feedback_connected_downstream"] - row["feedforward_downstream"]
            for row in events.values()
            if "feedforward_downstream" in row and "feedback_connected_downstream" in row
        ], dtype=float)
        degradation[model_name] = bootstrap_mean(values, resamples, 20260814)

    feedback_difference = []
    degradation_difference = []
    common_events = sorted(set(per_model["MeCaSNet"]) & set(per_model["DirGNN"]))
    for event_id in common_events:
        phy = per_model["MeCaSNet"][event_id]
        baseline = per_model["DirGNN"][event_id]
        feedback_key = "feedback_connected_downstream"
        feedforward_key = "feedforward_downstream"
        if feedback_key in phy and feedback_key in baseline:
            feedback_difference.append(phy[feedback_key] - baseline[feedback_key])
        if all(key in phy and key in baseline for key in (feedforward_key, feedback_key)):
            degradation_difference.append(
                (phy[feedback_key] - phy[feedforward_key])
                - (baseline[feedback_key] - baseline[feedforward_key])
            )
    return {
        "within_model_feedback_minus_feedforward_peak_mae": degradation,
        "mecasnet_minus_dirgnn_feedback_peak_mae": bootstrap_mean(
            np.asarray(feedback_difference, dtype=float), resamples, 20260815
        ),
        "difference_in_position_degradation": bootstrap_mean(
            np.asarray(degradation_difference, dtype=float), resamples, 20260816
        ),
        "sign_convention": (
            "Negative model differences favor MeCaSNet; positive within-model "
            "values indicate higher MAE in feedback-connected positions."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(args.device)
    cfg = Config(data_root=args.data_root)
    cfg.train_frac = 0.80
    cfg.val_frac = 0.10
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

    checkpoint_root = Path(args.checkpoint_root)
    per_seed: dict[str, list[dict[str, Any]]] = {}
    for model_name in MODELS:
        paths = checkpoint_paths(checkpoint_root, model_name)
        if len(paths) != 5:
            raise RuntimeError(
                f"Expected five checkpoints for {model_name}, found {len(paths)}"
            )
        runs = []
        for checkpoint in paths:
            print(f"Evaluating {model_name}: {checkpoint}", flush=True)
            runs.append(evaluate_checkpoint(
                checkpoint,
                model_name,
                cfg,
                network,
                loader,
                args.threshold,
                device,
            ))
        per_seed[model_name] = runs

    summaries = {name: seed_summary(runs) for name, runs in per_seed.items()}
    event_mae = {
        name: seed_averaged_event_metric(runs, "event_peak_mae")
        for name, runs in per_seed.items()
    }
    event_relative_mae = {
        name: seed_averaged_event_metric(runs, "event_peak_relative_mae")
        for name, runs in per_seed.items()
    }
    event_position_mae = {
        name: seed_averaged_event_metric(
            runs, "event_position_peak_mae", POSITIONS
        )
        for name, runs in per_seed.items()
    }
    report = {
        "protocol": {
            "models": list(MODELS),
            "seeds": [0, 1, 2, 3, 4],
            "test_events": len(test_ids),
            "cascade_definition": (
                "not directly shocked and ground-truth peak loss > "
                f"{args.threshold}"
            ),
            "distance_definition": (
                "shortest directed supplier-to-customer distance from any "
                "directly shocked firm"
            ),
            "inference": "post-hoc frozen-checkpoint evaluation; no retraining",
            "event_inference": (
                "event-level peak MAE is averaged across model seeds before "
                "paired event bootstrap"
            ),
        },
        "per_seed": per_seed,
        "mean_std_across_seeds": summaries,
        "paired_event_bootstrap": {
            "peak_mae": paired_event_analysis(
                event_mae, args.bootstrap, "peak MAE"
            ),
            "peak_relative_mae": paired_event_analysis(
                event_relative_mae, args.bootstrap, "peak relative MAE"
            ),
            "network_position_peak_mae": paired_position_analysis(
                event_position_mae, args.bootstrap
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"Created: {output}")
    print(json.dumps(report["mean_std_across_seeds"], indent=2))
    print(json.dumps(report["paired_event_bootstrap"], indent=2))


if __name__ == "__main__":
    main()
