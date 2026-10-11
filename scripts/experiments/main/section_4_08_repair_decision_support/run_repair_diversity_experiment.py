"""Selected revision experiment procedures; mixed experiment entry points omitted."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
import time
from typing import Any, Callable

import numpy as np


SAMPLERS = ("uniform", "hub", "sector")
HEURISTICS = (
    "out_degree", "out_strength", "initial_damage", "bau_capacity",
    "damage_capacity", "damage_out_strength",
)
PRIMARY_POLICY = "mecasnet_greedy_swap"
PROTOCOL = "repair_diversity_v1"
ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "flagship_y8h_download"


def generate_shock(
    sectors: np.ndarray,
    out_degree: np.ndarray,
    seed: int,
    candidates: int,
    sampler: str,
    damage_level: float,
    heterogeneous: bool,
) -> tuple[np.ndarray, np.ndarray]:
    sectors = np.asarray(sectors)
    out_degree = np.asarray(out_degree, dtype=float)
    if sectors.ndim != 1 or sectors.shape != out_degree.shape:
        raise ValueError("Sector and degree arrays must have matching one-dimensional shapes.")
    if not 1 <= candidates <= sectors.size or sampler not in SAMPLERS:
        raise ValueError("Invalid candidate count or sampler.")
    if not 0 < damage_level <= 0.95 or np.any(out_degree < 0):
        raise ValueError("Invalid damage level or negative degree.")
    if not np.isfinite(out_degree).all():
        raise ValueError("Degrees must be finite.")
    topology_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    damage_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    if sampler == "hub":
        if np.count_nonzero(out_degree) < candidates:
            raise ValueError("Not enough positive-degree nodes for hub sampling.")
        chosen = topology_rng.choice(
            sectors.size, candidates, replace=False, p=out_degree / out_degree.sum()
        )
    elif sampler == "uniform":
        chosen = topology_rng.choice(sectors.size, candidates, replace=False)
    else:
        grouped = []
        for sector in topology_rng.permutation(np.unique(sectors)):
            grouped.extend(topology_rng.permutation(np.flatnonzero(sectors == sector)).tolist())
            if len(grouped) >= candidates:
                break
        chosen = np.asarray(grouped[:candidates], dtype=int)
    chosen = np.sort(chosen)
    damage = np.full(candidates, damage_level, dtype=float)
    if heterogeneous:
        damage *= damage_rng.uniform(0.5, 1.5, candidates)
        damage = np.clip(damage, 0.01, 0.95)
    return chosen, damage


def greedy_set(candidates: list[int], budget: int, score: Callable) -> list[int]:
    selected: list[int] = []
    for _ in range(budget):
        action = min(
            (node for node in candidates if node not in selected),
            key=lambda node: (score(selected + [node]), node),
        )
        selected.append(action)
    return selected


def improve_set(
    selected: list[int], candidates: list[int], score: Callable, sweeps: int,
) -> tuple[list[int], list[float]]:
    current = sorted(selected)
    losses = [score(current)]
    for _ in range(sweeps):
        trials = (
            sorted([node for node in current if node != removed] + [added])
            for removed in current for added in candidates if added not in current
        )
        best = min(trials, key=lambda trial: (score(trial), trial), default=current)
        best_loss = score(best)
        if best_loss >= losses[-1] - 1e-10:
            break
        current = best
        losses.append(best_loss)
    return current, losses


def heuristic_sets(
    candidates: list[int], budget: int, damage: np.ndarray,
    degree: np.ndarray, strength: np.ndarray, capacity: np.ndarray, seed: int,
) -> dict[str, list[int]]:
    scores = {
        "out_degree": degree,
        "out_strength": strength,
        "initial_damage": damage,
        "bau_capacity": capacity,
        "damage_capacity": damage * capacity,
        "damage_out_strength": damage * strength,
    }
    selected = {
        name: sorted(candidates, key=lambda node: (-float(values[node]), node))[:budget]
        for name, values in scores.items()
    }
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    selected["random"] = rng.permutation(candidates).tolist()[:budget]
    return selected










def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
















