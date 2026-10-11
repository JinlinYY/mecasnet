"""Accuracy and computational cost of coarse-step ARIO cascade approximations."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Sequence

for _thread_environment_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_environment_variable, "1")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "flagship_y8h_download"
for path in (ROOT, ARCHIVE, ROOT / "cangzhou_pipeline", ROOT / "flagship_train", ROOT / "fix"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fix.sim_env import HenrietEnv
from pi_dl.config import Config
from pi_dl.data import CascadeEventDataset, StaticNetwork, split_event_ids
from stage3_ario_henriet2012 import Scenario, step
from y8_reviewer_experiments import load_y8, to_device


KEY_DAYS = np.asarray([0, 5, 10, 20, 30, 50, 70, 100, 150, 199], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "answer_result" / "henriet_runtime_suite_600")
    parser.add_argument("--data-protocol", choices=("runtime", "frozen-test"), default="runtime")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "answer_result" / "multiseed" / "PhyCaSNet" / "seed0" / "v3de_g3_gns_seed0.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "answer_result" / "coarse_ario_mechanistic_baseline")
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--recovery-mode", choices=("oracle", "fixed"), default="oracle")
    parser.add_argument("--fixed-recovery-days", type=float, default=60.0)
    parser.add_argument("--cascade-threshold", type=float, default=0.05)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


@contextlib.contextmanager
def silence_stdout() -> Any:
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def atomic_npz_write(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def r2(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.size == 0:
        return float("nan")
    total = float(np.square(target - target.mean()).sum())
    if total <= 1e-12:
        return float("nan")
    return float(1.0 - np.square(prediction - target).sum() / total)


class CoarseArio:
    def __init__(self, data_root: Path, step_days: int, recovery_mode: str, fixed_recovery_days: float) -> None:
        if step_days < 1:
            raise ValueError("--step-days must be positive.")
        if fixed_recovery_days <= 0:
            raise ValueError("--fixed-recovery-days must be positive.")
        self.environment = HenrietEnv(budget=1, horizon=200, meta_pkl=str(data_root / "static_meta.pkl"))
        self.base_p_ini = self.environment._net.P_ini.copy()
        self.base_n_opt_sector = self.environment._net.n_opt_sector.copy()
        self.p_bau = self.base_p_ini.copy()
        self.step_days = step_days
        self.recovery_mode = recovery_mode
        self.fixed_recovery_days = fixed_recovery_days

    def predict(self, event_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
        with np.load(event_path, allow_pickle=True) as payload:
            shock_mask = payload["shock_mask"].astype(bool)
            delta0 = payload["delta0"].astype(np.float64)
            metadata = dict(payload["shock_meta"].item())
        self.environment._net.P_ini = self.base_p_ini.copy()
        self.environment._net.n_opt_sector = self.base_n_opt_sector.copy()
        damage = {
            self.environment._net.nodes[index]: float(delta0[index])
            for index in np.flatnonzero(shock_mask)
        }
        scenario = Scenario(name=event_path.stem, initial_damage=damage)
        recovery_days = (
            float(metadata["recovery_days"])
            if self.recovery_mode == "oracle"
            else self.fixed_recovery_days
        )
        coarse_steps = int(np.ceil(int(KEY_DAYS[-1]) / self.step_days)) + 1
        params = replace(
            self.environment._params,
            T=coarse_steps,
            dt=float(self.step_days),
            tau_recover=recovery_days,
        )
        physical_days = np.arange(coarse_steps, dtype=np.float64) * self.step_days
        direct_damage = np.zeros((coarse_steps, self.environment._net.N), dtype=np.float64)
        for node_id, damage_value in scenario.initial_damage.items():
            node_index = self.environment._net.idx[node_id]
            direct_damage[:, node_index] = damage_value * np.exp(-physical_days / params.tau_recover)
        p0 = self.environment._P0_cached.copy()
        self.environment._net.P_ini = p0.copy()
        inventory = self.environment._net.n_opt_sector[None, :] * self.environment._net.A_tot * p0[:, None]
        demand = p0.copy()
        trajectory = []
        started = time.perf_counter()
        with silence_stdout():
            for step_index in range(coarse_steps):
                production, demand, inventory, _ = step(
                    self.environment._net, params, step_index, demand, inventory, direct_damage[step_index]
                )
                trajectory.append(production)
        elapsed = time.perf_counter() - started
        production_trace = np.asarray(trajectory, dtype=np.float64)
        keyframe_positions = KEY_DAYS / self.step_days
        lower = np.floor(keyframe_positions).astype(np.int64)
        upper = np.minimum(lower + 1, production_trace.shape[0] - 1)
        weight = (keyframe_positions - lower)[:, None]
        sampled = ((1.0 - weight) * production_trace[lower] + weight * production_trace[upper])
        sampled /= np.maximum(self.p_bau[None, :], 1e-12)
        health = np.clip(sampled, 0.0, 1.5)
        coarse_health = np.clip(
            production_trace / np.maximum(self.p_bau[None, :], 1e-12), 0.0, 1.5
        )
        peak_loss = np.maximum(1.0 - coarse_health, 0.0).max(axis=0)
        return peak_loss.astype(np.float32), health.astype(np.float32), elapsed


def event_ids(data_root: Path, max_events: int | None) -> list[int]:
    ids = sorted(int(path.stem.split("_")[1]) for path in (data_root / "events").glob("event_*.npz"))
    return ids if max_events is None else ids[:max_events]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluation_inputs(data_root: Path, max_events: int | None,
                      data_protocol: str) -> tuple[Config, list[int], Dict[str, Any]]:
    if max_events is not None and max_events < 1:
        raise ValueError("--max-events must be positive.")
    cfg = Config(data_root=str(data_root), event_scalars_mode="minimal",
                 seed=0, train_frac=0.8, val_frac=0.1)
    cfg.deq_iters = cfg.deq_grad_iters = 1
    if data_protocol == "runtime":
        manifest_path = data_root / "runtime_suite_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol") != "henriet_runtime_suite_v4":
            raise RuntimeError("Runtime mode requires henriet_runtime_suite_v4.")
        ids = event_ids(data_root, None)
        if ids != list(range(manifest["n_events"])):
            raise RuntimeError("Runtime event inventory does not match its manifest.")
        selection = "all runtime events"
    elif data_protocol == "frozen-test":
        if event_ids(data_root, None) != list(range(5000)):
            raise RuntimeError("Frozen-test mode requires all 5000 original event files.")
        _, _, ids = split_event_ids(data_root / cfg.events_dir, cfg)
        selection = "original seed=0 80/10/10 held-out test split"
    else:
        raise ValueError(f"Unsupported data protocol: {data_protocol}")
    if max_events is not None:
        ids = ids[:max_events]
    if not ids:
        raise RuntimeError("No evaluation events selected.")
    event_hashes = {
        str(event_id): file_sha256(data_root / "events" / f"event_{event_id:06d}.npz")
        for event_id in ids
    }
    evidence = {
        "data_protocol": data_protocol, "selection": selection,
        "split_seed": cfg.seed, "train_fraction": cfg.train_frac,
        "validation_fraction": cfg.val_frac, "event_ids": ids,
        "static_meta_sha256": file_sha256(data_root / "static_meta.pkl"),
        "event_file_sha256": event_hashes,
    }
    return cfg, ids, evidence


def metric_block(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    return {
        "r2": r2(predictions, targets),
        "mae": float(np.abs(predictions - targets).mean()),
        "rmse": float(np.sqrt(np.square(predictions - targets).mean())),
        "max_abs_error": float(np.abs(predictions - targets).max()),
    }


def summarize(records_dir: Path, ids: Sequence[int], threshold: float) -> Dict[str, Any]:
    fields: Dict[str, list[np.ndarray]] = {
        "peak_true": [], "peak_coarse": [], "peak_y8": [], "kf_true": [], "kf_coarse": [], "kf_y8": [], "shock": [],
    }
    coarse_seconds, y8_forward_seconds, y8_end_to_end_seconds = [], [], []
    for event_id in ids:
        with np.load(records_dir / f"event_{event_id:06d}.npz", allow_pickle=False) as record:
            for name in fields:
                fields[name].append(record[name])
            coarse_seconds.append(float(record["coarse_seconds"]))
            y8_forward_seconds.append(float(record["y8_forward_seconds"]))
            y8_end_to_end_seconds.append(float(record["y8_end_to_end_seconds"]))
    peak_true = np.concatenate(fields["peak_true"])
    shock = np.concatenate(fields["shock"]).astype(bool)
    cascade = (~shock) & (peak_true > threshold)
    keyframe_true = np.concatenate(fields["kf_true"], axis=1)
    keyframe_cascade = np.broadcast_to(cascade[None, :], keyframe_true.shape)
    result: Dict[str, Any] = {
        "event_count": len(ids),
        "cascade_definition": f"not directly shocked and full-step peak loss > {threshold}",
        "cascade_node_count": int(cascade.sum()),
        "all_node_peak": {},
        "all_node_keyframes": {},
        "cascade_node_peak": {},
        "cascade_node_keyframes": {},
    }
    for label, peak_key, kf_key in (("coarse_ario", "peak_coarse", "kf_coarse"), ("y8_h", "peak_y8", "kf_y8")):
        peak_prediction = np.concatenate(fields[peak_key])
        keyframe_prediction = np.concatenate(fields[kf_key], axis=1)
        result["all_node_peak"][label] = metric_block(peak_prediction, peak_true)
        result["all_node_keyframes"][label] = metric_block(keyframe_prediction, keyframe_true)
        result["cascade_node_peak"][label] = metric_block(peak_prediction[cascade], peak_true[cascade])
        per_day = {
            str(day): metric_block(keyframe_prediction[index, cascade], keyframe_true[index, cascade])
            for index, day in enumerate(KEY_DAYS)
        }
        result["cascade_node_keyframes"][label] = {
            "mean_r2_across_key_days": float(np.nanmean([value["r2"] for value in per_day.values()])),
            "by_day": per_day,
        }
    coarse = np.asarray(coarse_seconds)
    y8_forward = np.asarray(y8_forward_seconds)
    y8_end_to_end = np.asarray(y8_end_to_end_seconds)
    result["timing_seconds_per_event"] = {
        "coarse_ario": {"median": float(np.median(coarse)), "p95": float(np.percentile(coarse, 95)), "mean": float(coarse.mean())},
        "y8_h_forward_only": {"median": float(np.median(y8_forward)), "p95": float(np.percentile(y8_forward, 95)), "mean": float(y8_forward.mean())},
        "y8_h_end_to_end": {"median": float(np.median(y8_end_to_end)), "p95": float(np.percentile(y8_end_to_end, 95)), "mean": float(y8_end_to_end.mean())},
        "speedup_y8_end_to_end_over_coarse": float(np.median(coarse) / np.median(y8_end_to_end)),
    }
    return result


def main() -> None:
    args = parse_args()
    if not args.data_root.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("--data-root or --checkpoint is missing.")
    if args.step_days < 1:
        raise ValueError("--step-days must be positive.")
    cfg, ids, evaluation = evaluation_inputs(args.data_root, args.max_events, args.data_protocol)
    output_dir = args.output_dir.resolve()
    records_dir = output_dir / "event_records"
    run_manifest_path = output_dir / "run_manifest.json"
    if args.reset and output_dir.exists():
        for path in records_dir.glob("event_*.npz") if records_dir.is_dir() else []:
            path.unlink()
        summary = output_dir / "coarse_ario_mechanistic_baseline.json"
        if summary.exists():
            summary.unlink()
        if run_manifest_path.exists():
            run_manifest_path.unlink()
    run_manifest = {
        "integrator_version": 4,
        "step_days": args.step_days,
        "recovery_mode": args.recovery_mode,
        "fixed_recovery_days": args.fixed_recovery_days if args.recovery_mode == "fixed" else None,
        "keyframe_interpolation": "linear between bracketing coarse states",
        "inventory_restoration_scaling": "tau unchanged; dt is applied once in the inventory balance",
        "peak_sampling": "all coarse states",
        "macro_step_definition": "zero-order hold of order, production, rationing, receipts, and consumption rates within each macro step",
        "data_root": str(args.data_root.resolve()),
        "evaluation": evaluation,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "evaluator_sha256": file_sha256(Path(__file__)),
    }
    if run_manifest_path.exists():
        existing_run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if existing_run_manifest != run_manifest:
            raise RuntimeError("Existing event records use another coarse integrator. Re-run with --reset.")
    else:
        atomic_json_write(run_manifest_path, run_manifest)
    network = StaticNetwork(cfg)
    dataset = CascadeEventDataset(cfg, network, ids, train_mode=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_y8(args.checkpoint, cfg, network.Fv, device).eval()
    coarse = CoarseArio(args.data_root, args.step_days, args.recovery_mode, args.fixed_recovery_days)
    for index, event_id in enumerate(ids):
        record_path = records_dir / f"event_{event_id:06d}.npz"
        if record_path.exists():
            continue
        end_to_end_start = time.perf_counter()
        batch = dataset[index]
        with torch.inference_mode():
            device_batch = to_device(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_start = time.perf_counter()
            output = model(device_batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_seconds = time.perf_counter() - forward_start
        end_to_end_seconds = time.perf_counter() - end_to_end_start
        coarse_peak, coarse_kf, coarse_seconds = coarse.predict(args.data_root / "events" / f"event_{event_id:06d}.npz")
        reach_idx = batch["reach_idx"].numpy()
        atomic_npz_write(
            record_path,
            peak_true=batch["peak_loss"].numpy(),
            peak_coarse=coarse_peak[reach_idx],
            peak_y8=output["peak"].detach().cpu().numpy(),
            kf_true=batch["u_keyframes"].numpy(),
            kf_coarse=coarse_kf[:, reach_idx],
            kf_y8=output["u_keyframes"].detach().cpu().numpy(),
            shock=batch["shock_mask"].numpy().astype(np.uint8),
            coarse_seconds=np.asarray(coarse_seconds),
            y8_forward_seconds=np.asarray(forward_seconds),
            y8_end_to_end_seconds=np.asarray(end_to_end_seconds),
        )
        print(f"[{index + 1}/{len(ids)}] checkpointed event {event_id}", flush=True)
    report = {
        "protocol": {
            "baseline": "ARIO/Henriet with a 7-day explicit coarse time step".replace("7", str(args.step_days), 1),
            "audit_mode": args.step_days == 1,
            "reference": "Stored full-step ARIO/Henriet labels; " + evaluation["selection"],
            "evaluation": evaluation,
            "checkpoint_sha256": run_manifest["checkpoint_sha256"],
            "coarse_dynamics": "Explicit zero-order-hold macro steps: order, production, rationing, receipts, and consumption rates are evaluated at the macro-step start and held over dt=step_days for the inventory balance; tau is unchanged.",
            "recovery_information": (
                "Event-specific simulator recovery time (extra information, not the primary fair comparison)."
                if args.recovery_mode == "oracle"
                else f"Fixed recovery time of {args.fixed_recovery_days} days for every event; no event-specific future recovery information."
            ),
            "keyframe_sampling": "Linearly interpolate between the two bracketing coarse states at each original key day.",
            "peak_sampling": "Peak loss is computed over every coarse state, not only the reported keyframes.",
            "integrator_version": 4,
            "model": "Frozen Y8-H checkpoint; no retraining or model selection.",
            "data_root": str(args.data_root.resolve()),
            "event_ids": ids,
            "device": str(device),
        },
        "results": summarize(records_dir, ids, args.cascade_threshold),
    }
    atomic_json_write(output_dir / "coarse_ario_mechanistic_baseline.json", report)
    print(f"Saved {output_dir / 'coarse_ario_mechanistic_baseline.json'}")


if __name__ == "__main__":
    main()