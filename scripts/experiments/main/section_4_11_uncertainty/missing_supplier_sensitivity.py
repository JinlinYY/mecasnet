"""Missing-supplier sensitivity with preserved buyer input-share mass."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

# NumPy 2 pickles reference ``numpy._core``. The deployment server uses the
# latest NumPy compatible with Python 3.8, where the same modules live under
# ``numpy.core``.
if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "flagship_y8h_download"
for path in (ROOT, ARCHIVE, ROOT / "flagship_train"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pi_dl.config import Config
from pi_dl.data import CascadeEventDataset, StaticNetwork, collate_single, split_event_ids

from y8_reviewer_experiments import build_y8, load_y8, r2, to_device


KEY_DAYS = (0, 5, 10, 20, 30, 50, 70, 100, 150, 199)
THRESHOLD = 0.05


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)




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




def remove_edges(batch: dict[str, Any], probability: float,
                 rng: np.random.Generator, reallocate_by_buyer: bool = False) -> None:
    """Model missing supplier-buyer relations, optionally preserving buyer input mass."""
    edge_count = batch["edge_src"].numel()
    if probability <= 0 or edge_count == 0:
        return
    edge_dst = batch["edge_dst"].numpy()
    clean_shares = batch["edge_a"].numpy()
    keep = rng.random(edge_count) >= probability
    # A retained supplier is required for every observed buyer. This also makes
    # the mass-preserving control well defined at high deletion probabilities.
    for buyer in np.unique(edge_dst):
        buyer_edges = np.flatnonzero(edge_dst == buyer)
        if not bool(keep[buyer_edges].any()):
            keep[int(rng.choice(buyer_edges))] = True
        if reallocate_by_buyer:
            original_mass = float(clean_shares[buyer_edges].sum())
            retained_edges = buyer_edges[keep[buyer_edges]]
            retained_mass = float(clean_shares[retained_edges].sum())
            if retained_mass <= 0.0:
                raise RuntimeError(f"Buyer {buyer} retained no positive input share.")
            clean_shares[retained_edges] *= original_mass / retained_mass
    keep_tensor = torch.from_numpy(keep)
    if reallocate_by_buyer:
        batch["edge_a"] = torch.from_numpy(clean_shares.astype(np.float32))
    for key in ("edge_src", "edge_dst", "edge_a", "edge_outshare"):
        batch[key] = batch[key][keep_tensor]
    if reallocate_by_buyer:
        batch["edge_outshare"] = (
            batch["edge_a"] * batch["P_ini"][batch["edge_dst"]]
            / batch["P_ini"][batch["edge_src"]].clamp_min(1e-6)
        )
    recompute_hops(batch)










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






