"""Matched retraining procedures for temporal regularization comparisons."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from temporal_regularization_diagnostics import context_from_batch, gradient_report, rate_diagnostics, to_device
from comment12_common import (
    KEY_DAYS, PROTOCOL, atomic_json, bind_manifest, build_model, data_manifest,
    event_statistics, load_runtime, make_config, make_splits, node_masks,
    scheduled_lr, seed_everything, source_manifest, summarize_events,
)


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def apply_variant(cfg, variant):
    if variant not in {"full", "no_late"}:
        raise ValueError(f"Unknown variant: {variant}")
    result = copy.deepcopy(cfg)
    if variant == "no_late":
        result.w_mono_late = 0.0
    return result


def save_prediction(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def prediction_metrics(chunks):
    def r2(prediction, target):
        if not target.size or target.std() < 1e-2:
            return None
        return float(1 - np.square(prediction - target).sum() / (np.square(target - target.mean()).sum() + 1e-8))
    peak_pred = np.concatenate([chunk["peak_pred"] for chunk in chunks])
    peak_true = np.concatenate([chunk["peak_true"] for chunk in chunks])
    shock = np.concatenate([chunk["shock"] for chunk in chunks])
    prediction = np.concatenate([chunk["health_pred"] for chunk in chunks], axis=1)
    truth = np.concatenate([chunk["health_true"] for chunk in chunks], axis=1)
    result = {}
    for subset, mask in node_masks(shock, peak_true).items():
        selected_pred = prediction[:, mask]
        selected_true = truth[:, mask]
        result[subset] = {
            "nodes": int(mask.sum()),
            "peak_r2": r2(peak_pred[mask], peak_true[mask]),
            "peak_mae": float(np.abs(peak_pred[mask] - peak_true[mask]).mean()) if mask.any() else None,
            "trajectory_mae": float(np.abs(selected_pred - selected_true).mean()) if mask.any() else None,
            "health_by_day": {str(day): {
                "r2": r2(selected_pred[index], selected_true[index]),
                "mae": float(np.abs(selected_pred[index] - selected_true[index]).mean()) if mask.any() else None,
                "signed_error": float((selected_pred[index] - selected_true[index]).mean()) if mask.any() else None,
            } for index, day in enumerate(KEY_DAYS)},
        }
        valid = [row["r2"] for row in result[subset]["health_by_day"].values() if row["r2"] is not None]
        result[subset]["mean_keyframe_r2"] = float(np.mean(valid)) if valid else None
    return result


def evaluate_test(model, cfg, runtime, net, ids, device, output_dir):
    model.eval()
    dataset = runtime[1].CascadeEventDataset(cfg, net, ids, train_mode=False)
    events = []
    chunks = []
    for index, event_id in enumerate(ids):
        raw = dataset[index]
        cache = output_dir / "test_predictions" / f"event_{event_id:06d}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                arrays = {name: saved[name] for name in saved.files}
            if int(arrays["event_id"]) != event_id or not np.array_equal(arrays["reach_idx"], raw["reach_idx"].numpy()):
                raise ValueError("Test prediction cache identity mismatch.")
        else:
            with torch.no_grad():
                output = model(to_device(raw, device))
            arrays = {
                "event_id": np.asarray(event_id), "reach_idx": raw["reach_idx"].numpy(),
                "key_days": raw["key_days"].numpy(), "shock": raw["shock_mask"].numpy(),
                "peak_true": raw["peak_loss"].numpy(), "peak_pred": output["peak"].cpu().numpy(),
                "health_true": raw["u_keyframes"].numpy(), "health_pred": output["u_keyframes"].cpu().numpy(),
            }
            save_prediction(cache, arrays)
        if tuple(arrays["key_days"]) != KEY_DAYS:
            raise ValueError("Unexpected test key dates.")
        if not np.array_equal(arrays["health_true"], raw["u_keyframes"].numpy()):
            raise ValueError("Cached labels differ from original test labels.")
        chunks.append(arrays)
        statistics = event_statistics(
            arrays["health_true"], arrays["key_days"], node_masks(arrays["shock"], arrays["peak_true"]),
            arrays["health_pred"], threshold=cfg.mono_late_threshold_day)
        events.append({"event_id": event_id, "statistics": statistics})
        if (index + 1) % 50 == 0 or index + 1 == len(ids):
            print(f"[test] {index + 1}/{len(ids)}", flush=True)
    atomic_json(output_dir / "test_event_statistics.json", events)
    return {"metrics": prediction_metrics(chunks),
            "late_statistics": summarize_events([event["statistics"] for event in events], bootstrap=0)}


def train_run(args, runtime, cfg, net, splits, split_seed, init_seed, variant, shared_manifest):
    cfg = apply_variant(cfg, variant)
    cfg.seed = split_seed
    output_dir = Path(args.output_dir) / f"split{split_seed}" / f"seed{init_seed}" / variant
    device = torch.device(args.device)
    seed_everything(init_seed)
    model = build_model(runtime, cfg, net.Fv).to(device)
    initial_sha256 = state_digest(model.state_dict())
    generator = torch.Generator().manual_seed(100000 + init_seed)
    train_dataset = runtime[1].CascadeEventDataset(cfg, net, splits["train"], train_mode=False)
    validation_dataset = runtime[1].CascadeEventDataset(cfg, net, splits["validation"], train_mode=False)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0,
                              collate_fn=runtime[1].collate_single, generator=generator)
    validation_generator = torch.Generator().manual_seed(200000 + init_seed)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0,
                                   collate_fn=runtime[1].collate_single, generator=validation_generator)
    manifest = {"protocol": PROTOCOL, "task": "matched_training", "shared": shared_manifest,
                "split_seed": split_seed, "init_seed": init_seed, "variant": variant,
                "effective_config": dataclasses.asdict(cfg), "splits": splits,
                "initial_state_sha256": initial_sha256,
                "data": data_manifest(cfg, [event_id for ids in splits.values() for event_id in ids])}
    bind_manifest(output_dir, manifest)
    result_path = output_dir / "result.json"
    if result_path.exists():
        print(f"[skip complete] {output_dir}", flush=True)
        return
    context_handle = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    resume_path = output_dir / "resume.pt"
    history = []
    best_score = -float("inf")
    best_epoch = None
    best_state = None
    best_validation = None
    start_epoch = 1
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved["manifest"] != manifest:
            raise ValueError("Checkpoint provenance mismatch.")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        best_score, best_epoch = saved["best_score"], saved["best_epoch"]
        best_state, best_validation = saved["best_state"], saved["best_validation"]
        start_epoch = saved["epoch"] + 1
        restore_rng(saved["rng"], generator)
        validation_generator.set_state(saved["validation_generator"])
        print(f"[resume] {output_dir} from epoch {start_epoch}", flush=True)
    evaluate = importlib.import_module("pi_dl._runner").evaluate
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            lr = scheduled_lr(epoch, args.epochs, cfg.lr, args.warmup_epochs, args.min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            optimizer.zero_grad(set_to_none=True)
            sums = {}
            rate_intervals = 0
            rate_violations = 0
            order = []
            diagnostic = None
            started = time.perf_counter()
            for index, raw in enumerate(train_loader):
                order.append(int(raw["event_id"]))
                batch = to_device(raw, device)
                output = model(batch)
                parts = runtime[2].total_loss(output, batch, cfg)
                rate = rate_diagnostics(output, batch, cfg, runtime[2])
                rate_intervals += rate["intervals"]
                rate_violations += rate["violations"]
                if index == 0:
                    diagnostic = {"event_id": int(raw["event_id"]), "rate": rate,
                                  "gradient_sample_scope": "one first shuffled training event per epoch"}
                    for name in ("l_mono", "l_mono_late", "total"):
                        diagnostic[name] = {
                            "trajectory": gradient_report(parts[name], [output["u_full"]]),
                            "auxiliary_parameters": gradient_report(parts[name], model.phys_param_gen.parameters()),
                        }
                for name, value in parts.items():
                    if name == "total" or name.startswith("l_"):
                        sums[name] = sums.get(name, 0.0) + float(value.detach())
                (parts["total"] / cfg.grad_accum).backward()
                if (index + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            validation = evaluate(model, validation_loader, cfg, device)
            score = validation["r2_pk_csc"]
            if not math_isfinite(score):
                raise RuntimeError("Validation selection metric is undefined; cannot select a valid checkpoint.")
            if score > best_score:
                best_score, best_epoch, best_validation = float(score), epoch, validation
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            row = {"epoch": epoch, "lr": lr, "wall_seconds": time.perf_counter() - started,
                   "training_loss_mean_per_event": {name: value / len(order) for name, value in sums.items()},
                   "rate_active_fraction": rate_violations / rate_intervals,
                   "rate_violations": rate_violations, "rate_intervals": rate_intervals,
                   "data_order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
                   "validation": validation, "diagnostic": diagnostic}
            history.append(row)
            atomic_torch(resume_path, {
                "manifest": manifest, "epoch": epoch, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(), "history": history,
                "best_score": best_score, "best_epoch": best_epoch, "best_state": best_state,
                "best_validation": best_validation, "rng": rng_state(generator),
                "validation_generator": validation_generator.get_state(),
            })
            atomic_json(output_dir / "history.json", history)
            print(f"[split={split_seed} seed={init_seed} {variant}] epoch {epoch}/{args.epochs} "
                  f"val_cascade_peak_r2={score:.6f} best_epoch={best_epoch}", flush=True)
        if best_state is None:
            raise RuntimeError("No valid validation-selected state.")
        model.load_state_dict(best_state, strict=True)
        atomic_torch(output_dir / "best.pt", {
            "state_dict": best_state, "effective_config": dataclasses.asdict(cfg),
            "manifest": manifest, "epoch": best_epoch,
            "seed": init_seed, "split_seed": split_seed, "best_val": best_validation,
        })
        test = evaluate_test(model, cfg, runtime, net, splits["test"], device, output_dir)
        atomic_json(result_path, {"manifest": manifest, "best_epoch": best_epoch,
                                  "best_validation": best_validation, "test": test})
    finally:
        if context_handle is not None:
            context_handle.remove()


def math_isfinite(value):
    return value is not None and bool(np.isfinite(value))






def save_split_manifest(path, splits, split_seed):
    payload = {"train_ids": splits["train"], "validation_ids": splits["validation"],
               "internal_test_ids": splits["test"], "split_seed": split_seed}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("Existing split manifest differs from the requested split.")
    else:
        atomic_json(path, payload)




