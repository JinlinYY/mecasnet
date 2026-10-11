"""Selected revision experiment procedures; mixed experiment entry points omitted."""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

for thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import numpy as np

from independent_conformal_common import (
    KEY_DAYS, ORIGINAL_PROTOCOL, PROTOCOL, atomic_json, atomic_npz, day0_fingerprint,
    event_score, exact_conformal_radius, interval_row, lock_json, sha256,
    original_distribution_plan, split_plan, summarize_intervals,
)


def progress(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def setup_paths(root):
    for path in (root, root / "cangzhou_pipeline", root / "flagship_train", root / "flagship_y8h_download"):
        sys.path.insert(0, str(path))


def source_hashes(root):
    sources = [root / "flagship_train" / "y8_reviewer_experiments.py",
               root / "flagship_y8h_download" / "generate_henriet_training_data_server.py"]
    for directory in (root / "flagship_y8h_download" / "pi_dl", root / "cangzhou_pipeline", root / "fix"):
        sources.extend(sorted(directory.glob("*.py")))
    if not (root / "flagship_y8h_download" / "pi_dl" / "model_v3_decoder_ac.py").is_file():
        raise FileNotFoundError("Archived Y8-H package not found.")
    result = {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in sources}
    for name in ("run_independent_conformal.py", "independent_conformal_common.py"):
        result[f"supplement/{name}"] = sha256(Path(__file__).parent / name)
    return result


def checkpoint_paths(root):
    return [root / "answer_result" / "multiseed" / "PhyCaSNet" / f"seed{seed}" /
            f"v3de_g3_gns_seed{seed}.pt" for seed in range(5)]


def runtime_versions():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch")}}


def original_inventory(data_root):
    from pi_dl.data import load_pickle_compat
    meta = load_pickle_compat(data_root / "static_meta.pkl")
    base_seed = int(meta["params"]["seed_base"])
    paths = sorted((data_root / "events").glob("event_*.npz"))
    if len(paths) != 5000:
        raise ValueError(f"Expected the original 5000-event dataset, got {len(paths)}.")
    records = []
    progress("Hashing original 5000 events and checking Day-0 identities (no inference).")
    for index, path in enumerate(paths, 1):
        event_id = int(path.stem.split("_")[-1])
        with np.load(path, allow_pickle=True) as data:
            fingerprint = day0_fingerprint(data["shock_mask"], data["delta0"])
            shock = dict(data["shock_meta"].item())
        records.append({"event_id": event_id, "seed": base_seed + event_id,
                        "sha256": sha256(path), "day0_fingerprint": fingerprint,
                        "tier": str(shock["tier"]), "mode": str(shock["mode"])})
        if index % 500 == 0:
            progress(f"Original inventory {index}/5000")
    return records, meta


def original_sampler(root):
    path = root / "flagship_y8h_download" / "generate_henriet_training_data_server.py"
    spec = importlib.util.spec_from_file_location("frozen_original_henriet_sampler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sample_shock


def excluded_inventory(paths):
    records, inventories = [], []
    for path in paths:
        path = Path(path).resolve()
        payload = read_json(path)
        rows = payload["events"]
        if not rows or any(any(field not in row for field in ("event_id", "seed", "day0_fingerprint")) for row in rows):
            raise ValueError(f"Invalid previously inspected event inventory: {path}")
        records.extend(rows)
        inventories.append({"path": str(path), "sha256": sha256(path), "events": len(rows)})
    return records, inventories


def check_disjoint(plan, original):
    for field in ("event_id", "seed"):
        proposed = [row[field] for row in plan]
        if len(set(proposed)) != len(proposed) or set(proposed) & {row[field] for row in original}:
            raise ValueError(f"Fresh {field} values overlap or repeat.")


def event_path(pool, event_id):
    return pool / "events" / f"event_{event_id:06d}.npz"


def require_dedicated_directory(directory, manifest_name):
    if directory.exists() and any(directory.iterdir()) and not (directory / manifest_name).is_file():
        raise ValueError(f"Refusing to reuse a nonempty unrelated directory: {directory}")


def complete_payload(pool, records):
    return {"protocol": read_json(pool / "protocol.json")["protocol"], "events": records,
            "simulation_config_sha256": sha256(pool / "simulation_config.json"),
            "original_inventory_sha256": sha256(pool / "original_inventory.json")}


def verify_event(pool, planned):
    path = event_path(pool, planned["event_id"])
    record = read_json(pool / "event_records" / f"event_{planned['event_id']:06d}.json")
    fields = ("event_id", "seed", "role") + tuple(field for field in ("tier", "mode") if field in planned)
    if any(record[field] != planned[field] for field in fields):
        raise ValueError(f"Event plan changed: {path}")
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Event content changed: {path}")
    with np.load(path, allow_pickle=False) as data:
        if day0_fingerprint(data["shock_mask"], data["delta0"]) != record["day0_fingerprint"]:
            raise ValueError(f"Day-0 identity mismatch: {path}")
        if data["key_days"].tolist() != list(KEY_DAYS):
            raise ValueError(f"Key dates changed: {path}")
    return record


def check_fingerprints(records, original):
    seen = {row["day0_fingerprint"] for row in original}
    for row in records:
        if row["day0_fingerprint"] in seen:
            raise ValueError("Day-0 input overlaps original/fresh events; abort without resampling.")
        seen.add(row["day0_fingerprint"])




def predict_role(args, protocol, records, cfg, net, role, device):
    import torch
    from torch.utils.data import DataLoader
    from pi_dl.data import CascadeEventDataset, collate_single
    from y8_reviewer_experiments import load_y8, to_device

    selected = [row for row in records if row["role"] == role]
    ids = [row["event_id"] for row in selected]
    batches = list(DataLoader(CascadeEventDataset(cfg, net, ids, train_mode=False),
                             batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_single))
    forbidden = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]
    if any(np.any(batch["event_scalars"].numpy()[forbidden] != 0) for batch in batches):
        raise ValueError("Recovery/sampling metadata leaked into predictive event scalars.")
    predictions = {event_id: [] for event_id in ids}
    for seed, checkpoint in enumerate(checkpoint_paths(args.project_root)):
        progress(f"{role}: frozen seed {seed}; {len(ids)} events (including verified cache)")
        model = None
        for index, (record, batch) in enumerate(zip(selected, batches), 1):
            event_id = record["event_id"]
            path = args.output_dir / "predictions" / role / f"seed{seed}" / f"event_{event_id:06d}.npz"
            tag_path = path.with_suffix(".json")
            expected = {"event_sha256": record["sha256"], "checkpoint_sha256": protocol["checkpoints"][seed]["sha256"]}
            if tag_path.exists():
                tag = read_json(tag_path)
                if any(tag[key] != value for key, value in expected.items()) or sha256(path) != tag["prediction_sha256"]:
                    raise ValueError(f"Prediction cache changed: {path}")
                with np.load(path, allow_pickle=False) as data:
                    peak, trajectory = data["peak"], data["trajectory"]
            else:
                if model is None:
                    model = load_y8(checkpoint, cfg, net.Fv, device).eval()
                    if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) != 289512:
                        raise ValueError("Not the archived 289512-parameter Y8-H model.")
                with torch.inference_mode():
                    output = model(to_device(batch, device))
                peak = output["peak"].detach().cpu().numpy()
                trajectory = output["u_keyframes"].detach().cpu().numpy()
                event_score(batch["peak_loss"].numpy(), peak)
                event_score(batch["u_keyframes"].numpy(), trajectory)
                atomic_npz(path, peak=peak, trajectory=trajectory)
                atomic_json(tag_path, {**expected, "prediction_sha256": sha256(path)})
            predictions[event_id].append((peak, trajectory))
            if index == 1 or index % 50 == 0 or index == len(ids):
                progress(f"{role} seed {seed}: {index}/{len(ids)}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return [{"event_id": event_id, "peak_true": batch["peak_loss"].numpy(),
             "trajectory_true": batch["u_keyframes"].numpy(),
             "peak_mean": np.mean(np.stack([pair[0] for pair in predictions[event_id]]), axis=0),
             "trajectory_mean": np.mean(np.stack([pair[1] for pair in predictions[event_id]]), axis=0)}
            for event_id, batch in zip(ids, batches)]


def evaluate(args):
    import torch
    from pi_dl.config import Config
    from pi_dl.data import StaticNetwork
    from y8_reviewer_experiments import load_y8

    protocol = read_json(args.pool_dir / "protocol.json")
    if protocol["protocol"] not in (PROTOCOL, ORIGINAL_PROTOCOL) or protocol["sources"] != source_hashes(args.project_root):
        raise ValueError("Protocol/source version changed; use a new pool.")
    if protocol["runtime"] != runtime_versions():
        raise ValueError("Python/package versions changed since protocol freeze.")
    if sha256(args.pool_dir / "static_meta.pkl") != protocol["static_meta_sha256"]:
        raise ValueError("Static metadata changed.")
    progress("Verifying frozen weights, original inventory, and all fresh-event hashes.")
    for checkpoint, frozen in zip(checkpoint_paths(args.project_root), protocol["checkpoints"]):
        if sha256(checkpoint) != frozen["sha256"]:
            raise ValueError(f"Frozen checkpoint changed: {checkpoint}")
    original, _ = original_inventory(args.original_data_root)
    if read_json(args.pool_dir / "original_inventory.json") != original:
        raise ValueError("Original data changed.")
    excluded, inventories = excluded_inventory([row["path"] for row in protocol["excluded_inventories"]])
    if inventories != protocol["excluded_inventories"] or read_json(args.pool_dir / "excluded_inventory.json") != excluded:
        raise ValueError("Previously inspected event inventory changed.")
    forbidden = original + excluded
    check_disjoint(protocol["plan"], forbidden)
    if protocol["protocol"] == ORIGINAL_PROTOCOL:
        planned, distribution = original_distribution_plan(protocol["plan"], original)
        if planned != protocol["plan"] or distribution != protocol["target_stratum_distribution"]:
            raise ValueError("Original-distribution sampling plan changed.")
    records = [verify_event(args.pool_dir, planned) for planned in protocol["plan"]]
    check_fingerprints(records, forbidden)
    if read_json(args.pool_dir / "complete.json") != complete_payload(args.pool_dir, records):
        raise ValueError("Fresh pool not complete/unaltered.")
    if args.output_dir in (args.pool_dir, args.original_data_root):
        raise ValueError("Evaluation output must be separate from both datasets.")
    require_dedicated_directory(args.output_dir, "evaluation_manifest.json")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly choose --device cpu for a separate evaluation directory.")
    torch.manual_seed(20261001)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config(data_root=str(args.pool_dir), seed=0, event_scalars_mode="minimal")
    cfg.deq_iters = cfg.deq_grad_iters = 1
    net = StaticNetwork(cfg)
    import pi_dl
    if Path(pi_dl.__file__).resolve().parent != (args.project_root / "flagship_y8h_download" / "pi_dl").resolve() or net.Fv != 274:
        raise ValueError("Wrong model package/static feature profile.")
    lock_json(args.output_dir / "evaluation_manifest.json", {
        "protocol_sha256": sha256(args.pool_dir / "protocol.json"),
        "complete_sha256": sha256(args.pool_dir / "complete.json"),
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "runtime": runtime_versions(), "bootstrap_reps": args.bootstrap_reps,
        "deterministic_algorithms": True, "tf32": False,
    })
    for checkpoint in checkpoint_paths(args.project_root):
        model = load_y8(checkpoint, cfg, net.Fv, torch.device("cpu"))
        if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) != 289512:
            raise ValueError("Wrong frozen model profile.")
        del model
    calibration = predict_role(args, protocol, records, cfg, net, "calibration", device)
    radii = {target: exact_conformal_radius([event_score(row[f"{target}_true"], row[f"{target}_mean"])
             for row in calibration], protocol["alpha"]) for target in ("peak", "trajectory")}
    calibration_record = {"protocol_sha256": sha256(args.pool_dir / "protocol.json"),
                          "alpha": protocol["alpha"], "event_ids": [row["event_id"] for row in calibration],
                          "scores": {target: [event_score(row[f"{target}_true"], row[f"{target}_mean"])
                                     for row in calibration] for target in radii}, "radii": radii}
    lock_json(args.output_dir / "calibration.json", calibration_record)
    progress("Calibration radii locked; now starting fresh-test inference/reporting.")
    test = predict_role(args, protocol, records, cfg, net, "test", device)
    results = {}
    metadata = {row["event_id"]: row for row in records}
    for target, upper_support in (("peak", 1.0), ("trajectory", 1.5)):
        rows = [{"event_id": row["event_id"], **interval_row(row[f"{target}_true"], row[f"{target}_mean"],
                 radii[target], upper_support)} for row in test]
        results[target] = {"radius": radii[target], "support": [0, upper_support],
                           **summarize_intervals(rows, args.bootstrap_reps), "event_rows": rows}
        results[target]["by_severity_diagnostic"] = {
            tier: summarize_intervals(selected, args.bootstrap_reps)
            for tier in sorted({metadata[row["event_id"]]["tier"] for row in rows})
            if (selected := [row for row in rows if metadata[row["event_id"]]["tier"] == tier])}
    realized = {}
    for role in ("calibration", "test"):
        counts = {}
        for row in records:
            if row["role"] == role:
                key = f"{row['tier']}|{row['mode']}"
                counts[key] = counts.get(key, 0) + 1
        realized[role] = counts
    report = {"protocol": protocol["protocol"], "nominal_coverage": 1 - protocol["alpha"],
              "sampling_distribution": protocol["sampling_distribution"],
              "target_stratum_distribution": protocol["target_stratum_distribution"],
              "realized_stratum_counts": realized,
              "calibration_events": len(calibration), "test_events": len(test),
              "independence_checks": {"new_event_ids_and_rng_seeds": True, "day0_overlap_count": 0,
                  "original_events_checked": len(original), "frozen_checkpoints": 5,
                  "previously_inspected_events_checked": len(excluded),
                  "used_original_validation_for_calibration": False,
                  "calibration_locked_before_test_prediction": True},
              "prediction_scope": protocol["prediction_scope"],
              "bootstrap_boundary": "Test events resampled with calibration radius fixed; intervals do not include calibration-set variation.",
              "guarantee_boundary": "Event-level marginal coverage under exchangeable calibration/test events from this fixed fresh simulator/sampler, conditional on frozen models. Not coverage conditional on every severity, not simultaneous coverage across both targets, not a guarantee on historical labels or other topologies.",
              "results": results}
    atomic_json(args.output_dir / "independent_event_conformal.json", report)
    for target, values in results.items():
        progress(f"{target}: event coverage={values['event_simultaneous_coverage']:.4f}; mean width={values['event_mean_width']:.4f}")
    progress(f"Saved {args.output_dir / 'independent_event_conformal.json'}")




