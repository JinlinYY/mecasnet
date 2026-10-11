"""Conditional prediction-error variation under unobserved recovery time."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import pickle
import sys
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "flagship_y8h_download"
for path in (ROOT, ARCHIVE, ROOT / "cangzhou_pipeline", ROOT / "flagship_train"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_henriet_training_data import TIERS
from stage3_ario_henriet2012 import Scenario, simulate
from fix.sim_env import HenrietEnv
from pi_dl.config import Config
from pi_dl.data import CascadeEventDataset, StaticNetwork, collate_single, split_event_ids
from y8_reviewer_experiments import load_y8, to_device


KEY_DAYS = np.asarray([0, 5, 10, 20, 30, 50, 70, 100, 150, 199], dtype=np.int64)


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)






def percentile_ci(values: np.ndarray, reps: int, seed: int) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(reps, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_ratio_ci(numerators: np.ndarray, denominators: np.ndarray, reps: int, seed: int) -> list[float]:
    if numerators.size == 0 or numerators.shape != denominators.shape:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, numerators.size, size=(reps, numerators.size))
    sampled_numerators = numerators[indices].mean(axis=1)
    sampled_denominators = denominators[indices].mean(axis=1)
    ratios = sampled_numerators / sampled_denominators
    return [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))]


def load_batches(dataset: CascadeEventDataset) -> list[dict[str, Any]]:
    return list(DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_single))


@torch.inference_mode()
def predict_ensemble(
    checkpoints: list[Path], cfg: Config, net: StaticNetwork, batches: list[dict[str, Any]], device: torch.device,
) -> dict[int, dict[str, Any]]:
    by_event = {
        int(batch["event_id"]): {
            "event_id": int(batch["event_id"]),
            "reach_idx": batch["reach_idx"].numpy(),
            "shock": batch["shock_mask"].numpy() > 0.5,
            "peak_true": batch["peak_loss"].numpy(),
            "trajectory_true": batch["u_keyframes"].numpy(),
            "peak_predictions": [],
            "trajectory_predictions": [],
            "minimal_scalars": batch["event_scalars"].numpy(),
        }
        for batch in batches
    }
    for seed, checkpoint in enumerate(checkpoints):
        progress(f"Predicting {len(batches)} events with frozen checkpoint seed {seed}.")
        model = load_y8(checkpoint, cfg, net.Fv, device).eval()
        for batch in batches:
            output = model(to_device(batch, device))
            record = by_event[int(batch["event_id"])]
            record["peak_predictions"].append(output["peak"].detach().cpu().numpy())
            record["trajectory_predictions"].append(output["u_keyframes"].detach().cpu().numpy())
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for record in by_event.values():
        record["peak_mean"] = np.mean(np.stack(record.pop("peak_predictions")), axis=0)
        record["trajectory_mean"] = np.mean(np.stack(record.pop("trajectory_predictions")), axis=0)
    return by_event






def event_metadata(data_root: Path, event_id: int) -> dict[str, Any]:
    with np.load(data_root / "events" / f"event_{event_id:06d}.npz", allow_pickle=True) as data:
        return dict(data["shock_meta"].item())


def stratified_event_ids(data_root: Path, test_ids: list[int], count: int, seed: int) -> list[int]:
    if count > len(test_ids):
        raise ValueError("floor-events exceeds the available test-event count.")
    groups: dict[tuple[str, str], list[int]] = {}
    for event_id in test_ids:
        meta = event_metadata(data_root, event_id)
        groups.setdefault((str(meta["mode"]), str(meta["tier"])), []).append(event_id)
    rng = np.random.default_rng(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[int] = []
    keys = sorted(groups)
    while len(selected) < count:
        advanced = False
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
                advanced = True
        if not advanced:
            raise RuntimeError("Unable to draw the requested stratified recovery families.")
    return sorted(selected)


def replay_recovery_family(
    environment: HenrietEnv,
    p_bau: np.ndarray,
    row: dict[str, Any],
    metadata: dict[str, Any],
    data_root: Path,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tier = str(metadata["tier"])
    if tier not in TIERS:
        raise ValueError(f"Unknown Henriet severity tier: {tier}")
    tau_low, tau_high = TIERS[tier]["tau_range"]
    with np.load(data_root / "events" / f"event_{row['event_id']:06d}.npz", allow_pickle=True) as data:
        shock_mask = data["shock_mask"].astype(bool)
        delta0 = data["delta0"].astype(np.float64)
    fixed_damage = {environment._net.nodes[index]: float(delta0[index]) for index in np.where(shock_mask)[0]}
    if not fixed_damage:
        raise RuntimeError(f"Event {row['event_id']} has no directly shocked firms.")
    taus = rng.uniform(tau_low, tau_high, size=draws)
    trajectories, peaks = [], []
    reach_idx = row["reach_idx"]
    for tau in taus:
        scenario = Scenario(name=f"recovery_floor_{row['event_id']}", initial_damage=fixed_damage)
        params = replace(environment._params, T=200, tau_recover=float(tau))
        with contextlib.redirect_stdout(io.StringIO()):
            result = simulate(environment._net, scenario, params)
        health = np.asarray(result["P_a"], dtype=np.float64) / np.maximum(p_bau[None, :], 1e-12)
        keyframes = health[KEY_DAYS][:, reach_idx].astype(np.float32)
        trajectories.append(keyframes)
        peaks.append(np.maximum(1.0 - health[:, reach_idx], 0.0).max(axis=0).astype(np.float32))
    return np.asarray(trajectories), np.asarray(peaks), taus.astype(np.float32)


def recovery_cache_path(output_dir: Path, event_id: int, draws: int, seed: int) -> Path:
    return output_dir / "recovery_families" / f"event_{event_id:06d}_seed{seed}_draws{draws}.npz"


def load_recovery_cache(path: Path, event_id: int, draws: int, node_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(data["event_id"]) != event_id or int(data["random_seed"]) != seed:
                return None
            trajectories = np.asarray(data["trajectories"], dtype=np.float32)
            peaks = np.asarray(data["peaks"], dtype=np.float32)
            taus = np.asarray(data["taus"], dtype=np.float32)
        if trajectories.shape != (draws, len(KEY_DAYS), node_count) or peaks.shape != (draws, node_count) or taus.shape != (draws,):
            return None
        if not (np.isfinite(trajectories).all() and np.isfinite(peaks).all() and np.isfinite(taus).all()):
            return None
    except (KeyError, OSError, ValueError, zipfile.BadZipFile):
        return None
    return trajectories, peaks, taus


def save_recovery_cache(path: Path, event_id: int, seed: int, trajectories: np.ndarray, peaks: np.ndarray, taus: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            event_id=np.asarray(event_id, dtype=np.int64),
            random_seed=np.asarray(seed, dtype=np.int64),
            trajectories=trajectories,
            peaks=peaks,
            taus=taus,
        )
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def family_statistics(values: list[np.ndarray], prediction: np.ndarray, mask: np.ndarray) -> dict[str, float] | None:
    if not np.any(mask):
        return None
    floors_mae, floors_mse, model_mae, model_mse = [], [], [], []
    for family in values:
        selected = family[:, ..., mask] if family.ndim == 3 else family[:, mask]
        median = np.median(selected, axis=0)
        conditional_mean = np.mean(selected, axis=0)
        floors_mae.append(float(np.mean(np.abs(selected - median))))
        floors_mse.append(float(np.mean((selected - conditional_mean) ** 2)))
        predicted = prediction[..., mask]
        model_mae.append(float(np.mean(np.abs(selected - predicted))))
        model_mse.append(float(np.mean((selected - predicted) ** 2)))
    mae_floor = float(np.mean(floors_mae))
    mse_floor = float(np.mean(floors_mse))
    return {
        "conditional_mae_floor": mae_floor,
        "conditional_mse_floor": mse_floor,
        "conditional_rmse_floor": float(np.sqrt(mse_floor)),
        "model_mae_against_resampled_truth": float(np.mean(model_mae)),
        "model_mse_against_resampled_truth": float(np.mean(model_mse)),
    }




