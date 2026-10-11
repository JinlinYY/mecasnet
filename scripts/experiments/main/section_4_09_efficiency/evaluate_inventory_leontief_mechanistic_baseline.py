"""Evaluate an inventory-buffered sparse Leontief/IIM mechanistic baseline."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from pi_dl.data import CascadeEventDataset, StaticNetwork
from y8_reviewer_experiments import load_y8, to_device
from evaluate_coarse_ario_mechanistic_baseline import evaluation_inputs, file_sha256


KEY_DAYS = np.asarray([0, 5, 10, 20, 30, 50, 70, 100, 150, 199], dtype=np.int64)
HORIZON = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "answer_result" / "henriet_runtime_suite_600")
    parser.add_argument("--data-protocol", choices=("runtime", "frozen-test"), default="runtime")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "answer_result" / "multiseed" / "PhyCaSNet" / "seed0" / "v3de_g3_gns_seed0.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "answer_result" / "inventory_leontief_mechanistic_baseline")
    parser.add_argument("--recovery-mode", choices=("oracle", "fixed"), default="fixed")
    parser.add_argument("--fixed-recovery-days", type=float, default=60.0)
    parser.add_argument("--cascade-threshold", type=float, default=0.05)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


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


def metric_block(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    error = prediction - target
    return {
        "r2": r2(prediction, target),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "max_abs_error": float(np.abs(error).max()),
    }


class InventoryBufferedLeontief:
    def __init__(self, data_root: Path) -> None:
        environment = HenrietEnv(budget=1, horizon=HORIZON, meta_pkl=str(data_root / "static_meta.pkl"))
        network = environment._net
        buyer, supplier = np.nonzero(network.A > 0)
        supplier_sector = np.asarray(
            [network.sec_idx[network.sector_of[index]] for index in supplier], dtype=np.int64
        )
        pair_values = np.column_stack((buyer, supplier_sector))
        unique_pairs, edge_pair = np.unique(pair_values, axis=0, return_inverse=True)
        self.edge_supplier = supplier.astype(np.int64)
        self.edge_pair = edge_pair.astype(np.int64)
        self.edge_weight = network.A[buyer, supplier].astype(np.float64)
        self.pair_buyer = unique_pairs[:, 0].astype(np.int64)
        self.pair_sector = unique_pairs[:, 1].astype(np.int64)
        self.pair_denominator = np.bincount(
            self.edge_pair, weights=self.edge_weight, minlength=len(unique_pairs)
        ).astype(np.float64)
        self.initial_inventory_days = network.n_opt_sector[self.pair_sector].astype(np.float64)
        self.node_count = network.N
        no_shock_peak, no_shock_keyframes, _ = self.predict(
            np.zeros(self.node_count, dtype=np.float64), 60.0
        )
        no_shock_error = max(
            float(np.max(np.abs(no_shock_peak))),
            float(np.max(np.abs(no_shock_keyframes - 1.0))),
        )
        if no_shock_error > 1e-12:
            raise RuntimeError(f"No-shock inventory-Leontief self-check failed: error={no_shock_error}.")

    def predict(self, delta0: np.ndarray, recovery_days: float) -> tuple[np.ndarray, np.ndarray, float]:
        delta = np.asarray(delta0, dtype=np.float64)
        inventory_days = self.initial_inventory_days.copy()
        trajectory = np.empty((HORIZON, self.node_count), dtype=np.float64)
        started = time.perf_counter()
        for day in range(HORIZON):
            direct_capacity = np.clip(1.0 - delta * np.exp(-day / recovery_days), 0.0, 1.0)
            supply_capacity = np.full(self.node_count, np.inf, dtype=np.float64)
            np.minimum.at(supply_capacity, self.pair_buyer, inventory_days)
            health = np.minimum(direct_capacity, supply_capacity)
            health = np.clip(health, 0.0, 1.0)
            trajectory[day] = health

            delivered = np.bincount(
                self.edge_pair,
                weights=self.edge_weight * health[self.edge_supplier],
                minlength=inventory_days.size,
            )
            delivery_ratio = delivered / self.pair_denominator
            consumption_ratio = health[self.pair_buyer]
            inventory_days = np.maximum(
                inventory_days + delivery_ratio - consumption_ratio, 0.0
            )
        elapsed = time.perf_counter() - started
        if not np.isfinite(trajectory).all():
            raise RuntimeError("Inventory-Leontief produced non-finite health values.")
        keyframes = trajectory[KEY_DAYS].astype(np.float32)
        peak_loss = np.maximum(1.0 - trajectory, 0.0).max(axis=0).astype(np.float32)
        return peak_loss, keyframes, elapsed


def summarize(records_dir: Path, ids: Sequence[int], threshold: float) -> Dict[str, Any]:
    fields: Dict[str, list[np.ndarray]] = {
        "peak_true": [], "peak_mechanistic": [], "peak_y8": [],
        "kf_true": [], "kf_mechanistic": [], "kf_y8": [], "shock": [],
    }
    mechanistic_seconds, y8_forward_seconds, y8_end_to_end_seconds = [], [], []
    for event_id in ids:
        with np.load(records_dir / f"event_{event_id:06d}.npz", allow_pickle=False) as record:
            for name in fields:
                fields[name].append(record[name])
            mechanistic_seconds.append(float(record["mechanistic_seconds"]))
            y8_forward_seconds.append(float(record["y8_forward_seconds"]))
            y8_end_to_end_seconds.append(float(record["y8_end_to_end_seconds"]))
    peak_true = np.concatenate(fields["peak_true"])
    shock = np.concatenate(fields["shock"]).astype(bool)
    cascade = (~shock) & (peak_true > threshold)
    keyframe_true = np.concatenate(fields["kf_true"], axis=1)
    result: Dict[str, Any] = {
        "event_count": len(ids),
        "cascade_definition": f"not directly shocked and full-step peak loss > {threshold}",
        "cascade_node_count": int(cascade.sum()),
        "all_node_peak": {}, "cascade_node_peak": {}, "cascade_node_keyframes": {},
    }
    for label, peak_key, kf_key in (
        ("inventory_buffered_leontief", "peak_mechanistic", "kf_mechanistic"),
        ("y8_h", "peak_y8", "kf_y8"),
    ):
        peak_prediction = np.concatenate(fields[peak_key])
        keyframe_prediction = np.concatenate(fields[kf_key], axis=1)
        result["all_node_peak"][label] = metric_block(peak_prediction, peak_true)
        result["cascade_node_peak"][label] = metric_block(peak_prediction[cascade], peak_true[cascade])
        by_day = {
            str(day): metric_block(keyframe_prediction[index, cascade], keyframe_true[index, cascade])
            for index, day in enumerate(KEY_DAYS)
        }
        result["cascade_node_keyframes"][label] = {
            "mean_r2_across_key_days": float(np.nanmean([row["r2"] for row in by_day.values()])),
            "by_day": by_day,
        }
    mechanistic = np.asarray(mechanistic_seconds)
    y8_forward = np.asarray(y8_forward_seconds)
    y8_end_to_end = np.asarray(y8_end_to_end_seconds)
    result["timing_seconds_per_event"] = {
        "inventory_buffered_leontief": {
            "median": float(np.median(mechanistic)), "p95": float(np.percentile(mechanistic, 95)),
            "mean": float(mechanistic.mean()),
        },
        "y8_h_forward_only": {
            "median": float(np.median(y8_forward)), "p95": float(np.percentile(y8_forward, 95)),
            "mean": float(y8_forward.mean()),
        },
        "y8_h_end_to_end": {
            "median": float(np.median(y8_end_to_end)), "p95": float(np.percentile(y8_end_to_end, 95)),
            "mean": float(y8_end_to_end.mean()),
        },
        "speedup_y8_end_to_end_over_mechanistic": float(np.median(y8_end_to_end) / np.median(mechanistic)),
    }
    return result


def main() -> None:
    args = parse_args()
    if not args.data_root.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("--data-root or --checkpoint is missing.")
    if args.fixed_recovery_days <= 0:
        raise ValueError("--fixed-recovery-days must be positive.")
    cfg, ids, evaluation = evaluation_inputs(args.data_root, args.max_events, args.data_protocol)
    output_dir = args.output_dir.resolve()
    records_dir = output_dir / "event_records"
    run_manifest_path = output_dir / "run_manifest.json"
    if args.reset and output_dir.exists():
        for path in records_dir.glob("event_*.npz") if records_dir.is_dir() else []:
            path.unlink()
        for path in (run_manifest_path, output_dir / "inventory_leontief_mechanistic_baseline.json"):
            if path.exists():
                path.unlink()
    run_manifest = {
        "baseline_version": 1,
        "model": "inventory-buffered sparse Leontief with fixed BAU orders",
        "recovery_mode": args.recovery_mode,
        "fixed_recovery_days": args.fixed_recovery_days if args.recovery_mode == "fixed" else None,
        "horizon": HORIZON,
        "data_root": str(args.data_root.resolve()),
        "evaluation": evaluation,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "evaluator_sha256": file_sha256(Path(__file__)),
        "selection_source_sha256": file_sha256(Path(evaluation_inputs.__code__.co_filename)),
    }
    if run_manifest_path.exists():
        if json.loads(run_manifest_path.read_text(encoding="utf-8")) != run_manifest:
            raise RuntimeError("Existing records use another protocol. Re-run with --reset.")
    else:
        atomic_json_write(run_manifest_path, run_manifest)
    network = StaticNetwork(cfg)
    dataset = CascadeEventDataset(cfg, network, ids, train_mode=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_y8(args.checkpoint, cfg, network.Fv, device).eval()
    mechanistic = InventoryBufferedLeontief(args.data_root)
    for index, event_id in enumerate(ids):
        record_path = records_dir / f"event_{event_id:06d}.npz"
        if record_path.exists():
            continue
        y8_start = time.perf_counter()
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
        y8_end_to_end_seconds = time.perf_counter() - y8_start
        event_path = args.data_root / "events" / f"event_{event_id:06d}.npz"
        with np.load(event_path, allow_pickle=True) as payload:
            delta0 = payload["delta0"].astype(np.float64)
            metadata = dict(payload["shock_meta"].item())
        recovery_days = (
            float(metadata["recovery_days"])
            if args.recovery_mode == "oracle"
            else args.fixed_recovery_days
        )
        peak, keyframes, mechanistic_seconds = mechanistic.predict(delta0, recovery_days)
        reach_idx = batch["reach_idx"].numpy()
        atomic_npz_write(
            record_path,
            peak_true=batch["peak_loss"].numpy(), peak_mechanistic=peak[reach_idx],
            peak_y8=output["peak"].detach().cpu().numpy(),
            kf_true=batch["u_keyframes"].numpy(), kf_mechanistic=keyframes[:, reach_idx],
            kf_y8=output["u_keyframes"].detach().cpu().numpy(),
            shock=batch["shock_mask"].numpy().astype(np.uint8),
            mechanistic_seconds=np.asarray(mechanistic_seconds),
            y8_forward_seconds=np.asarray(forward_seconds),
            y8_end_to_end_seconds=np.asarray(y8_end_to_end_seconds),
        )
        print(f"[{index + 1}/{len(ids)}] checkpointed event {event_id}", flush=True)
    report = {
        "protocol": {
            "baseline": "Inventory-buffered sparse Leontief/IIM with fixed BAU orders",
            "mechanistic_link": "Retains ARIO sector-wise essential-input constraints, within-sector supplier substitution, proportional supplier delivery, daily direct recovery, and 1/10-day inventories.",
            "omitted_mechanisms": ["adaptive inventory-restoration orders", "endogenous demand feedback", "full demand-based proportional rationing"],
            "recovery_information": (
                "Event-specific simulator recovery time (extra information, not the primary fair comparison)."
                if args.recovery_mode == "oracle"
                else f"Fixed {args.fixed_recovery_days}-day recovery time; no event-specific future recovery information."
            ),
            "model": "Frozen Y8-H checkpoint with minimal Day-0 event scalars.",
            "evaluation": evaluation,
            "checkpoint_sha256": run_manifest["checkpoint_sha256"],
            "data_root": str(args.data_root.resolve()), "event_ids": ids, "device": str(device),
        },
        "results": summarize(records_dir, ids, args.cascade_threshold),
    }
    atomic_json_write(output_dir / "inventory_leontief_mechanistic_baseline.json", report)
    print(f"Saved {output_dir / 'inventory_leontief_mechanistic_baseline.json'}")


if __name__ == "__main__":
    main()