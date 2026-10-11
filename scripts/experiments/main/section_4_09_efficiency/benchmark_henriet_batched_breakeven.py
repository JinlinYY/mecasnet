"""Benchmark equivalent single-process and batched ARIO/Henriet simulation.

The script replays stored event shocks, verifies that process-pool execution
matches the single-process numerical core, and compares the best CPU throughput
with the frozen Y8-H GPU surrogate. It records actual simulator calls directly.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import multiprocessing as mp
import os
import platform
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

for _thread_environment_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_environment_variable, "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "flagship_y8h_download"
for path in (ROOT, ARCHIVE, ROOT / "cangzhou_pipeline", ROOT / "flagship_train", ROOT / "fix"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fix.sim_env import HenrietEnv
from stage3_ario_henriet2012 import Scenario, simulate


KEY_DAYS = np.asarray([0, 5, 10, 20, 30, 50, 70, 100, 150, 199], dtype=np.int64)
_WORKER: Optional["ReplayWorker"] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ARCHIVE / "data" / "cascade_henriet_5000")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "answer_result" / "multiseed" / "PhyCaSNet" / "seed0" / "v3de_g3_gns_seed0.pt")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "answer_result" / "batched_simulator_breakeven.json")
    parser.add_argument("--tune-events", type=int, default=100)
    parser.add_argument("--final-events", type=int, default=500)
    parser.add_argument("--correctness-events", type=int, default=100)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-events", type=int, default=5)
    parser.add_argument("--surrogate-warmups", type=int, default=30)
    parser.add_argument("--selection-seed", type=int, default=20260929)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--training-seconds", type=float,
                        help="One-checkpoint training wall-clock. Omit when only measuring throughput.")
    parser.add_argument("--generation-seconds", type=float,
                        help="Measured wall-clock for producing 5,000 training events with the selected CPU worker count.")
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--skip-surrogate", action="store_true")
    parser.add_argument("--checkpoint-file", type=Path,
                        help="Persistent checkpoint path. Defaults beside --output.")
    parser.add_argument("--reset-checkpoint", action="store_true",
                        help="Discard the existing benchmark checkpoint before running.")
    return parser.parse_args()


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


def stable_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def file_hash(path: Path) -> str:
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


def checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint_file is not None:
        return args.checkpoint_file
    return args.output.with_name(f"{args.output.stem}_checkpoint.json")


def checkpoint_fingerprint(args: argparse.Namespace, ids: Dict[str, List[int]]) -> Dict[str, Any]:
    manifest = args.data_root / "runtime_suite_manifest.json"
    return {
        "schema_version": 1,
        "data_root": str(args.data_root.resolve()),
        "runtime_suite_manifest_sha256": file_hash(manifest) if manifest.is_file() else None,
        "event_sets": ids,
        "workers": sorted(set(args.workers)),
        "repetitions": args.repetitions,
        "warmup_events": args.warmup_events,
        "surrogate_warmups": args.surrogate_warmups,
        "checkpoint": str(args.checkpoint.resolve()),
        "device": args.device,
    }


def load_checkpoint(path: Path, fingerprint: Dict[str, Any], reset: bool) -> Dict[str, Any]:
    if reset and path.exists():
        path.unlink()
    if path.exists():
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"Checkpoint {path} was created with different inputs or benchmark arguments. "
                "Use --reset-checkpoint to discard it."
            )
        if not isinstance(checkpoint.get("stages"), dict):
            raise RuntimeError(f"Checkpoint {path} has no valid stages object. Use --reset-checkpoint to discard it.")
        print(f"Resuming benchmark from {path}", flush=True)
        return checkpoint
    checkpoint = {"fingerprint": fingerprint, "stages": {}}
    atomic_json_write(path, checkpoint)
    return checkpoint


def save_stage(checkpoint: Dict[str, Any], path: Path, name: str, value: Any) -> None:
    checkpoint["stages"][name] = value
    atomic_json_write(path, checkpoint)
    print(f"Checkpointed {name} to {path}", flush=True)


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


class ReplayWorker:
    def __init__(self, data_root: str, horizon: int) -> None:
        self.data_root = Path(data_root)
        self.meta = static_metadata(self.data_root / "static_meta.pkl")
        self.environment = HenrietEnv(
            budget=1, horizon=horizon, meta_pkl=str(self.data_root / "static_meta.pkl")
        )
        self.base_p_ini = self.environment._net.P_ini.copy()
        self.base_n_opt_sector = self.environment._net.n_opt_sector.copy()
        self.p_bau = np.asarray(self.meta["P_ini"], dtype=np.float64)
        self.horizon = horizon
        self.calls = 0

    def replay(self, event_id: int, include_arrays: bool = False) -> Dict[str, Any]:
        self.environment._net.P_ini = self.base_p_ini.copy()
        self.environment._net.n_opt_sector = self.base_n_opt_sector.copy()
        event_path = self.data_root / "events" / f"event_{event_id:06d}.npz"
        with np.load(event_path, allow_pickle=True) as payload:
            shock_mask = payload["shock_mask"].astype(bool)
            delta0 = payload["delta0"].astype(np.float64)
            meta = dict(payload["shock_meta"].item())
        damage = {
            self.environment._net.nodes[index]: float(delta0[index])
            for index in np.flatnonzero(shock_mask)
        }
        if not damage:
            raise RuntimeError(f"Event {event_id} has no directly shocked firms.")
        recovery_days = float(meta["recovery_days"])
        scenario = Scenario(name=f"benchmark_{event_id}", initial_damage=damage)
        params = replace(self.environment._params, T=self.horizon, tau_recover=recovery_days)
        started = time.perf_counter()
        with silence_stdout():
            result = simulate(self.environment._net, scenario, params)
        elapsed = time.perf_counter() - started
        self.calls += 1
        health = np.asarray(result["P_a"], dtype=np.float64) / np.maximum(self.p_bau[None, :], 1e-12)
        health = np.clip(health, 0.0, 1.5)
        peak_loss = np.maximum(1.0 - health, 0.0).max(axis=0)
        aggregate_loss = np.maximum(
            1.0 - (health * self.p_bau[None, :]).sum(axis=1) / max(float(self.p_bau.sum()), 1e-12),
            0.0,
        )
        record: Dict[str, Any] = {
            "event_id": event_id,
            "seconds": elapsed,
            "calls": 1,
            "trajectory_hash": stable_hash(health),
            "peak_hash": stable_hash(peak_loss),
            "keyframe_hash": stable_hash(health[KEY_DAYS]),
            "aggregate_peak_loss": float(aggregate_loss.max()),
            "final_aggregate_loss": float(aggregate_loss[-1]),
        }
        if include_arrays:
            record.update({"health": health, "peak_loss": peak_loss, "aggregate_loss": aggregate_loss})
        return record


def worker_init(data_root: str, horizon: int) -> None:
    global _WORKER
    _WORKER = ReplayWorker(data_root, horizon)


def worker_replay(event_id: int) -> Dict[str, Any]:
    if _WORKER is None:
        raise RuntimeError("Worker was not initialized.")
    return _WORKER.replay(event_id, include_arrays=False)


def choose_ids(data_root: Path, tune_events: int, final_events: int, correctness_events: int, seed: int) -> Dict[str, List[int]]:
    event_ids = sorted(int(path.stem.split("_")[1]) for path in (data_root / "events").glob("event_*.npz"))
    required = tune_events + final_events
    if required > len(event_ids):
        raise ValueError(f"Need {required} runtime events but only {len(event_ids)} are available.")
    shuffled = list(event_ids)
    np.random.default_rng(seed).shuffle(shuffled)
    tune = sorted(shuffled[:tune_events])
    final = sorted(shuffled[tune_events:tune_events + final_events])
    return {"tune": tune, "final": final, "correctness": tune[:min(correctness_events, len(tune))]}


def validate_runtime_suite_precision(data_root: Path, event_ids: Sequence[int]) -> None:
    """Ensure a generated runtime suite can be replayed from its saved shocks."""
    manifest_path = data_root / "runtime_suite_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "henriet_runtime_suite_v4":
        raise RuntimeError(
            "This runtime suite predates the isolated-state replay contract. Regenerate it with "
            "create_henriet_runtime_suite.py --overwrite before benchmarking."
        )
    for event_id in event_ids:
        event_path = data_root / "events" / f"event_{event_id:06d}.npz"
        with np.load(event_path, allow_pickle=True) as payload:
            if payload["delta0"].dtype != np.dtype(np.float64):
                raise RuntimeError(f"Runtime event {event_id} does not store delta0 as float64.")


def summarize_times(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size), "mean_seconds": float(array.mean()),
        "median_seconds": float(np.median(array)), "p95_seconds": float(np.percentile(array, 95)),
        "min_seconds": float(array.min()), "max_seconds": float(array.max()),
    }


def run_single(data_root: Path, horizon: int, event_ids: Sequence[int], warmup_events: int) -> Dict[str, Any]:
    worker = ReplayWorker(str(data_root), horizon)
    for event_id in event_ids[:warmup_events]:
        worker.replay(event_id)
    started = time.perf_counter()
    records = [worker.replay(event_id) for event_id in event_ids]
    wall = time.perf_counter() - started
    return {"records": records, "wall_seconds": wall, "calls": worker.calls - min(warmup_events, len(event_ids)), "backend": "single_process"}


def run_pool(data_root: Path, horizon: int, event_ids: Sequence[int], workers: int, warmup_events: int) -> Dict[str, Any]:
    if workers == 1:
        return run_single(data_root, horizon, event_ids, warmup_events)
    context = mp.get_context("spawn" if os.name == "nt" else "fork")
    if os.name == "nt":
        mp.set_executable(sys.executable)
    with context.Pool(processes=workers, initializer=worker_init, initargs=(str(data_root), horizon)) as pool:
        list(pool.imap(worker_replay, event_ids[:warmup_events], chunksize=1))
        started = time.perf_counter()
        records = list(pool.imap(worker_replay, event_ids, chunksize=1))
        wall = time.perf_counter() - started
    return {"records": records, "wall_seconds": wall, "calls": len(records), "backend": "process_pool"}


def compare_records(left: Sequence[Dict[str, Any]], right: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_id_left = {row["event_id"]: row for row in left}
    by_id_right = {row["event_id"]: row for row in right}
    common = sorted(set(by_id_left) & set(by_id_right))
    mismatches = []
    for event_id in common:
        for key in ("trajectory_hash", "peak_hash", "keyframe_hash"):
            if by_id_left[event_id][key] != by_id_right[event_id][key]:
                mismatches.append({"event_id": event_id, "field": key})
    return {"events_compared": len(common), "hash_mismatches": mismatches, "passed": not mismatches}


def stored_label_check(data_root: Path, horizon: int, event_ids: Sequence[int]) -> Dict[str, Any]:
    worker = ReplayWorker(str(data_root), horizon)
    max_keyframe_error = 0.0
    max_peak_error = 0.0
    for event_id in event_ids:
        replay = worker.replay(event_id, include_arrays=True)
        with np.load(data_root / "events" / f"event_{event_id:06d}.npz", allow_pickle=True) as payload:
            stored_keyframes = np.asarray(payload["u_keyframes"], dtype=np.float64)
            stored_peak = np.asarray(payload["peak_loss_node"], dtype=np.float64)
        max_keyframe_error = max(max_keyframe_error, float(np.max(np.abs(replay["health"][KEY_DAYS] - stored_keyframes))))
        max_peak_error = max(max_peak_error, float(np.max(np.abs(replay["peak_loss"] - stored_peak))))
    return {"events_compared": len(event_ids), "max_keyframe_abs_error": max_keyframe_error, "max_peak_abs_error": max_peak_error}


def benchmark_surrogate(data_root: Path, checkpoint: Path, event_ids: Sequence[int], device: torch.device, warmups: int) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from pi_dl.config import Config
    from pi_dl.data import CascadeEventDataset, StaticNetwork, collate_single
    from y8_reviewer_experiments import load_y8, to_device
    cfg = Config(data_root=str(data_root), seed=0, event_scalars_mode="minimal")
    cfg.deq_iters = cfg.deq_grad_iters = 1
    network = StaticNetwork(cfg)
    model = load_y8(checkpoint, cfg, network.Fv, device).eval()
    dataset = CascadeEventDataset(cfg, network, list(event_ids), train_mode=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_single)
    batches = [to_device(batch, device) for batch in loader]
    with torch.inference_mode():
        for batch in batches[:min(warmups, len(batches))]:
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        event_times = []
        started = time.perf_counter()
        for batch in batches:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            event_start = time.perf_counter()
            model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            event_times.append(time.perf_counter() - event_start)
        wall = time.perf_counter() - started
    return {
        "wall_seconds": wall, "events_per_second": len(batches) / wall,
        "event_timing": summarize_times(event_times),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 1024 ** 2) if device.type == "cuda" else None,
    }


def system_metadata() -> Dict[str, Any]:
    numpy_config = np.__config__
    get_info = getattr(numpy_config, "get_info", None)
    if callable(get_info):
        blas = get_info("blas_opt_info")
    else:
        blas = {"numpy_config": str(getattr(numpy_config, "CONFIG", "unavailable"))}
    result = {
        "platform": platform.platform(), "python": sys.version,
        "numpy": np.__version__,
        "cpu_count_logical": os.cpu_count(), "pid": os.getpid(),
        "blas": blas,
        "environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }
    try:
        import torch
        result.update({"torch": torch.__version__, "torch_cuda": torch.version.cuda})
    except ModuleNotFoundError:
        result.update({"torch": None, "torch_cuda": None})
    return result


def break_even(generation_seconds: Optional[float], training_seconds: Optional[float], simulator_eps: float, surrogate_eps: float, ensemble_size: int) -> Optional[int]:
    if generation_seconds is None or training_seconds is None or simulator_eps <= surrogate_eps:
        return None
    return int(np.ceil((generation_seconds + ensemble_size * training_seconds) / (1.0 / simulator_eps - 1.0 / surrogate_eps)))


def main() -> None:
    args = parse_args()
    if args.tune_events < 1 or args.final_events < 1 or args.repetitions < 1:
        raise ValueError("Event counts and repetitions must be positive.")
    if not args.data_root.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("data-root or checkpoint is missing")
    ids = choose_ids(args.data_root, args.tune_events, args.final_events, args.correctness_events, args.selection_seed)
    validate_runtime_suite_precision(args.data_root, ids["correctness"])
    horizon = 200
    checkpoint = load_checkpoint(checkpoint_path(args), checkpoint_fingerprint(args, ids), args.reset_checkpoint)
    stages = checkpoint["stages"]
    if "correctness" not in stages:
        correctness_single = run_single(args.data_root, horizon, ids["correctness"], 0)["records"]
        correctness_pool = run_pool(args.data_root, horizon, ids["correctness"], 2, 0)["records"]
        equivalence = compare_records(correctness_single, correctness_pool)
        if not equivalence["passed"]:
            raise RuntimeError(f"Single and batched simulator results differ: {equivalence['hash_mismatches'][:5]}")
        label_replay = stored_label_check(args.data_root, horizon, ids["correctness"])
        if label_replay["max_keyframe_abs_error"] > 1e-6 or label_replay["max_peak_abs_error"] > 1e-6:
            raise RuntimeError(f"Replayed simulator does not match stored labels: {label_replay}")
        save_stage(checkpoint, checkpoint_path(args), "correctness", {
            "single_vs_pool": equivalence, "replay_vs_stored_labels": label_replay,
        })

    tune = stages.get("cpu_tuning", {})
    tune_repetitions = stages.get("cpu_tuning_repetitions", {})
    for workers in sorted(set(args.workers)):
        if str(workers) in tune:
            continue
        repetitions = list(tune_repetitions.get(str(workers), []))
        while len(repetitions) < args.repetitions:
            run = run_pool(args.data_root, horizon, ids["tune"], workers, args.warmup_events)
            repetitions.append(run["wall_seconds"])
            tune_repetitions[str(workers)] = repetitions
            save_stage(checkpoint, checkpoint_path(args), "cpu_tuning_repetitions", tune_repetitions)
        median_wall = float(np.median(repetitions))
        tune[str(workers)] = {
            "workers": workers, "wall_seconds": summarize_times(repetitions),
            "events_per_second": len(ids["tune"]) / median_wall,
            "cpu_core_seconds": workers * median_wall,
            "backend": run["backend"],
        }
        save_stage(checkpoint, checkpoint_path(args), "cpu_tuning", tune)
    best_workers = max(tune.values(), key=lambda row: row["events_per_second"])["workers"]

    if "cpu_final" not in stages:
        final_repetitions = list(stages.get("cpu_final_repetitions", []))
        while len(final_repetitions) < args.repetitions:
            run = run_pool(args.data_root, horizon, ids["final"], best_workers, args.warmup_events)
            final_repetitions.append(run["wall_seconds"])
            save_stage(checkpoint, checkpoint_path(args), "cpu_final_repetitions", final_repetitions)
        final_wall = float(np.median(final_repetitions))
        save_stage(checkpoint, checkpoint_path(args), "cpu_final", {
            "selected_workers_from_tune_set": best_workers,
            "final_events": len(ids["final"]),
            "wall_seconds": summarize_times(final_repetitions),
            "events_per_second": len(ids["final"]) / final_wall,
            "amortized_seconds_per_event": final_wall / len(ids["final"]),
            "actual_simulator_calls_per_final_repeat": len(ids["final"]),
            "backend": run["backend"],
        })
    simulator = stages["cpu_final"]
    surrogate = None
    if not args.skip_surrogate and "surrogate_final" in stages:
        surrogate = stages["surrogate_final"]
    elif not args.skip_surrogate:
        import torch
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        surrogate = benchmark_surrogate(args.data_root, args.checkpoint, ids["final"], device, args.surrogate_warmups)
        save_stage(checkpoint, checkpoint_path(args), "surrogate_final", surrogate)

    report: Dict[str, Any] = {
        "protocol": {
            "description": "Equivalent ARIO/Henriet replay; CPU worker count chosen on tune events and finalized on disjoint events.",
            "event_sets": ids, "selection_seed": args.selection_seed,
            "event_pool": "All pre-generated event files; disjoint from model-selection logic and used only for runtime measurement.",
            "workers_tested": sorted(set(args.workers)), "repetitions": args.repetitions,
            "training_simulator_calls": 0,
        },
        "system": system_metadata(),
        "correctness": stages["correctness"],
        "cpu_tuning": tune,
        "cpu_final": simulator,
        "surrogate_final": surrogate,
    }
    if surrogate is not None:
        report["throughput_speedup_surrogate_over_batched_cpu"] = surrogate["events_per_second"] / simulator["events_per_second"]
        report["break_even_scenarios"] = break_even(args.generation_seconds, args.training_seconds, simulator["events_per_second"], surrogate["events_per_second"], args.ensemble_size)
    atomic_json_write(args.output, report)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()