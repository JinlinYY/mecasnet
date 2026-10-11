"""Selected revision experiment procedures; mixed experiment entry points omitted."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import heapq
import io
import json
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

for _thread_environment_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_environment_variable, "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "flagship_y8h_download"
for _path in (ROOT, ARCHIVE, ROOT / "cangzhou_pipeline", ROOT / "fix"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cangzhou_pipeline.generate_henriet_training_data import KEY_DAYS, sample_shock
from fix.sim_env import HenrietEnv
from stage3_ario_henriet2012 import Scenario, simulate


_WORKER: Optional["GenerationWorker"] = None




def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


@contextlib.contextmanager
def silence_stdout() -> Iterable[None]:
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)




def static_metadata(path: Path) -> Dict[str, Any]:
    with path.open("rb") as stream:
        try:
            return pickle.load(stream)
        except ModuleNotFoundError as error:
            if error.name != "numpy._core":
                raise
            import numpy.core as numpy_core
            import numpy.core.multiarray as numpy_core_multiarray
            sys.modules.setdefault("numpy._core", numpy_core)
            sys.modules.setdefault("numpy._core.multiarray", numpy_core_multiarray)
            stream.seek(0)
            return pickle.load(stream)




def stratum_key(tier: str, mode: str) -> str:
    return f"{tier}|{mode}"


def historical_strata(data_root: Path) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    paths = sorted((data_root / "events").glob("event_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No source events found under {data_root / 'events'}")
    for path in paths:
        with np.load(path, allow_pickle=True) as payload:
            meta = dict(payload["shock_meta"].item())
        counts[stratum_key(str(meta["tier"]), str(meta["mode"]))] += 1
    return dict(sorted(counts.items()))


def allocate_quotas(source_counts: Dict[str, int], n_events: int) -> Dict[str, int]:
    total = sum(source_counts.values())
    if total < 1:
        raise ValueError("Source stratum counts are empty.")
    exact = {key: n_events * count / total for key, count in source_counts.items()}
    quotas = {key: int(np.floor(value)) for key, value in exact.items()}
    remaining = n_events - sum(quotas.values())
    for key in sorted(exact, key=lambda item: (exact[item] - quotas[item], item), reverse=True)[:remaining]:
        quotas[key] += 1
    return {key: count for key, count in quotas.items() if count > 0}


def assignments_from_quotas(quotas: Dict[str, int], seed: int) -> Dict[int, Tuple[str, str]]:
    assignments: List[Tuple[str, str]] = []
    for key, count in sorted(quotas.items()):
        tier, mode = key.split("|", 1)
        assignments.extend([(tier, mode)] * count)
    np.random.default_rng(seed).shuffle(assignments)
    return {event_id: assignment for event_id, assignment in enumerate(assignments)}








def quantiles(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)), "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()), "max": float(array.max()),
    }


def stratified_extrapolation(records: Sequence[Dict[str, Any]], source_counts: Dict[str, int],
                             replicates: int, seed: int) -> Dict[str, Any]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in records:
        grouped[stratum_key(str(row["tier"]), str(row["mode"]))].append(float(row["wall_seconds"]))
    missing = sorted(set(source_counts) - set(grouped))
    if missing:
        raise RuntimeError(f"No measured events for source strata: {missing}")
    point = float(sum(source_counts[key] * np.mean(grouped[key]) for key in source_counts))
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(replicates):
        bootstrap.append(float(sum(
            source_counts[key] * rng.choice(grouped[key], size=len(grouped[key]), replace=True).mean()
            for key in source_counts
        )))
    return {
        "target_success_events": int(sum(source_counts.values())),
        "method": "stratified bootstrap over per-event CPU seconds by historical tier|mode composition",
        "point_seconds": point,
        "bootstrap_replicates": replicates,
        "bootstrap_seconds": bootstrap,
        "ci95_seconds": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "stratum_means_seconds": {key: float(np.mean(values)) for key, values in sorted(grouped.items())},
    }


def pool_makespan(values: Sequence[float], workers: int) -> float:
    loads = [0.0] * workers
    heapq.heapify(loads)
    for value in values:
        current = heapq.heappop(loads)
        heapq.heappush(loads, current + float(value))
    return float(max(loads))


def stratified_batched_wall_extrapolation(records: Sequence[Dict[str, Any]], source_counts: Dict[str, int],
                                          workers: int, measured_parent_seconds: float,
                                          replicates: int, seed: int) -> Dict[str, Any]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in records:
        grouped[stratum_key(str(row["tier"]), str(row["mode"]))].append(float(row["wall_seconds"]))
    missing = sorted(set(source_counts) - set(grouped))
    if missing:
        raise RuntimeError(f"No measured events for source strata: {missing}")
    measured_compute_wall = pool_makespan([float(row["wall_seconds"]) for row in records], workers)
    fixed_overhead = max(0.0, measured_parent_seconds - measured_compute_wall)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(replicates):
        sampled = np.concatenate([
            rng.choice(grouped[key], size=source_counts[key], replace=True)
            for key in sorted(source_counts)
        ])
        rng.shuffle(sampled)
        bootstrap.append(pool_makespan(sampled, workers) + fixed_overhead)
    return {
        "target_success_events": int(sum(source_counts.values())),
        "method": "stratified bootstrap with greedy dynamic worker-pool scheduling by historical tier|mode composition",
        "workers": workers,
        "point_seconds": float(np.mean(bootstrap)),
        "bootstrap_replicates": replicates,
        "bootstrap_seconds": bootstrap,
        "ci95_seconds": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "measured_parent_wall_seconds": measured_parent_seconds,
        "measured_compute_makespan_seconds": measured_compute_wall,
        "fixed_startup_and_parent_overhead_seconds": fixed_overhead,
    }


def stratified_call_extrapolation(records: Sequence[Dict[str, Any]], source_counts: Dict[str, int],
                                  replicates: int, seed: int) -> Dict[str, Any]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in records:
        grouped[stratum_key(str(row["tier"]), str(row["mode"]))].append(
            float(row["actual_simulator_calls"])
        )
    missing = sorted(set(source_counts) - set(grouped))
    if missing:
        raise RuntimeError(f"No measured events for source strata: {missing}")
    point = float(sum(source_counts[key] * np.mean(grouped[key]) for key in source_counts))
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(replicates):
        bootstrap.append(float(sum(
            source_counts[key] * rng.choice(grouped[key], size=len(grouped[key]), replace=True).mean()
            for key in source_counts
        )))
    return {
        "target_success_events": int(sum(source_counts.values())),
        "method": "stratified bootstrap over actual simulator calls by historical tier|mode composition",
        "point_calls": point,
        "bootstrap_replicates": replicates,
        "bootstrap_calls": bootstrap,
        "ci95_calls": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "stratum_means_calls": {key: float(np.mean(values)) for key, values in sorted(grouped.items())},
    }




