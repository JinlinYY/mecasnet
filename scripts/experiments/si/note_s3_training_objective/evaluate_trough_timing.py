from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class TimingAccumulator:
    def __init__(self, days, threshold=0.05, tie_tolerance=1e-6):
        self.days = np.asarray(days, dtype=np.float64)
        if self.days.ndim != 1 or len(self.days) < 2 or not np.all(np.diff(self.days) > 0):
            raise ValueError("Key days must be strictly increasing.")
        if not np.isfinite(threshold) or tie_tolerance < 0 or not np.isfinite(tie_tolerance):
            raise ValueError("Invalid evaluation threshold or tie tolerance.")
        self.threshold = threshold
        self.tie_tolerance = tie_tolerance
        self.confusion = {name: np.zeros((len(self.days), len(self.days)), dtype=np.int64)
                          for name in ("head_argmax", "ordinal_median")}
        self.sums = {name: {key: 0.0 for key in (
            "correct", "adjacent", "day_error", "rank_error", "near_min_correct",
        )} for name in self.confusion}
        self.count = 0
        self.empty_events = 0
        self.near_tie_count = 0
        self.score_sums = {key: 0.0 for key in (
            "cross_entropy", "ranked_probability_score", "time_weighted_rps",
            "uniform_ranked_probability_score", "uniform_time_weighted_rps",
        )}
        self.events = []

    def add(self, event_id, logits, truth, shock_mask, peak_loss, saved_target=None):
        logits = np.asarray(logits, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        shock_mask = np.asarray(shock_mask)
        peak_loss = np.asarray(peak_loss)
        classes = len(self.days)
        if logits.ndim != 2 or logits.shape[1] != classes or truth.shape != (classes, logits.shape[0]):
            raise ValueError("Expected logits[N,K] and truth[K,N].")
        if shock_mask.shape != (logits.shape[0],) or peak_loss.shape != shock_mask.shape:
            raise ValueError("Node mask shape mismatch.")
        if not all(np.isfinite(value).all() for value in (logits, truth, shock_mask, peak_loss)):
            raise ValueError("Nonfinite timing inputs are not accepted.")
        target_all = np.argmin(truth, axis=0)
        if saved_target is not None and not np.array_equal(target_all, np.asarray(saved_target)):
            raise ValueError("Dataset trough labels do not equal earliest keyframe argmin.")
        mask = (shock_mask < 0.5) & (peak_loss > self.threshold)
        count = int(mask.sum())
        record = {"event_id": int(event_id), "eligible_nodes": count}
        if count == 0:
            self.empty_events += 1
            self.events.append(record)
            return
        target = target_all[mask]
        selected = logits[mask]
        shifted = selected - selected.max(axis=1, keepdims=True)
        log_normalizer = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        log_probability = shifted - log_normalizer
        probability = np.exp(log_probability)
        cumulative = np.cumsum(probability, axis=1)[:, :-1]
        target_cumulative = (target[:, None] <= np.arange(classes - 1)[None, :]).astype(np.float64)
        squared_error = (cumulative - target_cumulative) ** 2
        uniform_error = ((np.arange(1, classes) / classes)[None, :] - target_cumulative) ** 2
        time_weights = np.diff(self.days) / (self.days[-1] - self.days[0])
        score_values = {
            "cross_entropy": -log_probability[np.arange(count), target],
            "ranked_probability_score": squared_error.mean(axis=1),
            "time_weighted_rps": squared_error @ time_weights,
            "uniform_ranked_probability_score": uniform_error.mean(axis=1),
            "uniform_time_weighted_rps": uniform_error @ time_weights,
        }
        for name, values in score_values.items():
            self.score_sums[name] += float(values.sum())
            record[name] = float(values.mean())
        selected_truth = truth[:, mask].T
        near_min = np.abs(selected_truth - selected_truth.min(axis=1, keepdims=True)) <= self.tie_tolerance
        self.near_tie_count += int((near_min.sum(axis=1) > 1).sum())
        predictions = {
            "head_argmax": np.argmax(probability, axis=1),
            "ordinal_median": np.argmax(np.cumsum(probability, axis=1) >= 0.5, axis=1),
        }
        for name, predicted in predictions.items():
            rank_error = np.abs(predicted - target)
            values = {
                "correct": (predicted == target).astype(np.float64),
                "adjacent": (rank_error <= 1).astype(np.float64),
                "day_error": np.abs(self.days[predicted] - self.days[target]),
                "rank_error": rank_error.astype(np.float64),
                "near_min_correct": near_min[np.arange(count), predicted].astype(np.float64),
            }
            np.add.at(self.confusion[name], (target, predicted), 1)
            record[name] = {}
            for key, value in values.items():
                self.sums[name][key] += float(value.sum())
                record[name][key] = float(value.mean())
        self.count += count
        self.events.append(record)

    def result(self):
        if self.count == 0:
            raise ValueError("No eligible cascade nodes; timing accuracy is undefined.")
        output = {
            "eligible_nodes": self.count, "events": len(self.events),
            "events_without_eligible_nodes": self.empty_events,
            "key_days": self.days.tolist(), "peak_threshold": self.threshold,
            "near_min_tolerance": self.tie_tolerance,
            "near_tie_node_fraction": self.near_tie_count / self.count,
            "score_means": {key: value / self.count for key, value in self.score_sums.items()},
            "events_detail": self.events,
        }
        for name, matrix in self.confusion.items():
            support = matrix.sum(axis=1)
            predicted_count = matrix.sum(axis=0)
            present = support > 0
            recall = np.divide(np.diag(matrix), support, out=np.zeros(len(support)), where=present)
            values = self.sums[name]
            output[name] = {
                "accuracy": values["correct"] / self.count,
                "balanced_accuracy_present_classes": float(recall[present].mean()),
                "adjacent_accuracy": values["adjacent"] / self.count,
                "day_mae": values["day_error"] / self.count,
                "rank_mae": values["rank_error"] / self.count,
                "near_min_accuracy": values["near_min_correct"] / self.count,
                "event_macro_accuracy": float(np.mean([
                    event[name]["correct"] for event in self.events if event["eligible_nodes"]
                ])),
                "confusion_true_rows_predicted_columns": matrix.tolist(),
                "true_class_counts": support.tolist(), "predicted_class_counts": predicted_count.tolist(),
                "per_class_recall": [float(recall[index]) if present[index] else None
                                     for index in range(len(support))],
                "boundary_last_keyframe_target_fraction": int(support[-1]) / self.count,
            }
        return output


def audit_manifest(path, repeat_id):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_sha256", None)
    if not claimed or canonical_hash(manifest) != claimed:
        raise ValueError("Manifest hash mismatch: {}".format(path))
    if manifest["repeat_id"] != repeat_id or manifest["split_seed"] != repeat_id:
        raise ValueError("Repeat/split identity mismatch.")
    groups = [manifest[key] for key in ("train_ids", "validation_ids", "internal_test_ids")]
    flattened = [int(value) for group in groups for value in group]
    if any(not group for group in groups) or len(set(flattened)) != len(flattened):
        raise ValueError("Empty or overlapping event pools.")
    if len(flattened) != manifest["event_count"]:
        raise ValueError("Manifest event count mismatch.")
    manifest["manifest_sha256"] = claimed
    return manifest


def make_plan(args, repeat_id, require_timing_head=True):
    run_dir = args.run_root / "repeat_{:02d}".format(repeat_id) / args.model_label
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed" or metadata.get("returncode") != 0:
        raise ValueError("Only completed runs may be evaluated: {}".format(run_dir))
    if metadata["repeat_id"] != repeat_id or metadata["initialization_seed"] != repeat_id:
        raise ValueError("Checkpoint initialization/repeat identity mismatch.")
    model = metadata["model"]
    variant = model["variant"]
    if require_timing_head and variant != "V3DE_G3_GNS":
        raise ValueError("This evaluator requires the manuscript GNS-backed MeCaSNet with a timing head.")
    summary_path = run_dir / (variant.lower() + "_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recorded = summary["args"]
    if summary["seed_range"] != [repeat_id] or recorded["variant"] != variant:
        raise ValueError("Summary variant/seed identity mismatch.")
    if recorded.get("split_seed", repeat_id) != repeat_id:
        raise ValueError("Recorded training configuration has a different split seed.")
    checkpoint = run_dir / (variant.lower() + "_seed{}.pt".format(repeat_id))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    command = metadata.get("command", [])
    if command:
        if "--save_ckpt" not in command or command[command.index("--seed_start") + 1] != str(repeat_id):
            raise ValueError("Recorded training command/checkpoint seed mismatch.")
    manifest_path = args.manifest_dir / "repeat_{:02d}.json".format(repeat_id)
    manifest = audit_manifest(manifest_path, repeat_id)
    if metadata["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Run and requested split differ.")
    original_dataset = manifest["data_root"]
    data_root = args.data_root or Path(recorded["data_root"])
    data_root = data_root.resolve()
    if str(data_root) != str(Path(original_dataset).resolve()) and not args.allow_data_relocation:
        raise ValueError("Relocated data requires --allow-data-relocation after verifying it is an exact copy.")
    for event_id in manifest["internal_test_ids"]:
        path = data_root / "events" / "event_{:06d}.npz".format(event_id)
        if not path.is_file():
            raise FileNotFoundError(path)
    trainer_args = []
    for key, value in recorded.items():
        if key in {"data_root", "out_dir", "split_manifest", "seed_start",
                   "seeds", "save_ckpt", "resume_from", "device"}:
            continue
        if value is None or value is False:
            continue
        trainer_args.append("--" + key)
        if value is not True:
            trainer_args.append(str(value))
    return {
        "repeat_id": repeat_id, "model_label": args.model_label, "variant": variant,
        "code_root": str(args.code_root), "data_root": str(data_root),
        "data_relocated": str(data_root) != str(Path(original_dataset).resolve()),
        "relocation_user_verified_exact_copy": bool(args.allow_data_relocation),
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
        "summary_sha256": file_hash(summary_path), "metadata_sha256": file_hash(metadata_path),
        "source_manifest_sha256": manifest["manifest_sha256"],
        "train_ids": manifest["train_ids"], "validation_ids": manifest["validation_ids"],
        "test_ids": manifest["internal_test_ids"], "environment": model.get("env", {}),
        "trainer_args": trainer_args, "device": args.device,
        "tie_tolerance": args.tie_tolerance,
        "best_epoch": summary["test_per_seed"][0]["best_epoch"],
        "expected_parameters": summary["test_per_seed"][0]["params"],
    }


class EvaluationComplete(Exception):
    pass


def worker(plan, output):
    code_root = Path(plan["code_root"])
    for key in list(os.environ):
        if key.startswith(("GNS_", "MECAS_")) or key in {
            "D_HIDDEN_OVERRIDE", "NEEQ_FAST_EXEC", "NEEQ_CACHE_DIR", "PAPER_ABLATION_KIND",
        }:
            os.environ.pop(key)
    os.environ.update({str(key): str(value) for key, value in plan["environment"].items()})
    sys.path.insert(0, str(code_root))
    import torch
    from pi_dl import train_v3de as trainer

    sources = {str(path.relative_to(code_root)): file_hash(path)
               for path in sorted((code_root / "pi_dl").glob("*.py"))}
    checkpoint = torch.load(plan["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("config", {}).get("variant") != plan["variant"]:
        raise ValueError("Checkpoint variant differs from recorded run.")
    if checkpoint.get("epoch") != plan["best_epoch"]:
        raise ValueError("Checkpoint is not the recorded validation-selected best epoch.")
    state = checkpoint.get("state_dict")
    if not state or not any("trough_day_head" in key for key in state):
        raise ValueError("Checkpoint has no trained trough-timing head.")
    outcome = {}
    data_root = Path(plan["data_root"])
    static_hash = file_hash(data_root / "static_meta.pkl")
    event_hashes = {str(event_id): file_hash(data_root / "events" / "event_{:06d}.npz".format(event_id))
                    for event_id in plan["test_ids"]}

    def inference_only(cfg, net, train_ids, val_ids, device, build_model_fn, **unused):
        if list(train_ids) != plan["train_ids"] or list(val_ids) != plan["validation_ids"]:
            raise ValueError("Model reconstruction changed the event pools.")
        if float(getattr(cfg, "w_trough_day", 0.0)) <= 0:
            raise ValueError("This checkpoint's configuration did not supervise the timing head.")
        model = build_model_fn().to(device)
        model.load_state_dict(state, strict=True)
        if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) != plan["expected_parameters"]:
            raise ValueError("Reconstructed model capacity differs from the recorded training run.")
        model.eval()
        dataset = trainer.CascadeEventDataset(cfg, net, plan["test_ids"])
        loader = trainer.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                    collate_fn=trainer.collate_single)
        accumulator = TimingAccumulator(cfg.key_days,
            float(getattr(cfg, "trough_day_peak_thresh", 0.05)), plan["tie_tolerance"])
        excluded = {"u_keyframes", "peak_loss", "is_reach", "trough_day_target"}
        with torch.inference_mode():
            for batch in loader:
                inputs = {key: value.to(device) if torch.is_tensor(value) else value
                          for key, value in batch.items() if key not in excluded}
                prediction = model(inputs)
                if "trough_day_logit" not in prediction:
                    raise ValueError("Model forward does not return the timing head.")
                accumulator.add(batch["event_id"], prediction["trough_day_logit"].cpu().numpy(),
                    batch["u_keyframes"].numpy(), batch["shock_mask"].numpy(),
                    batch["peak_loss"].numpy(), batch["trough_day_target"].numpy())
        outcome.update(accumulator.result())
        outcome["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        outcome["model_input_keys"] = sorted(inputs)
        outcome["supervision_fields_excluded_from_forward"] = sorted(excluded)
        raise EvaluationComplete()

    trainer.train_with_val = inference_only
    with tempfile.TemporaryDirectory(prefix="trough_timing_no_train_") as scratch:
        scratch = Path(scratch)
        reconstruction_manifest = scratch / "split.json"
        write_json(reconstruction_manifest, {
            "train_ids": plan["train_ids"], "validation_ids": plan["validation_ids"],
            "internal_test_ids": plan["test_ids"],
            "strict_test_ids": plan["test_ids"],
        })
        sys.argv = ["inference_only", *plan["trainer_args"],
            "--data_root", plan["data_root"], "--out_dir", str(scratch / "unused"),
            "--split_manifest", str(reconstruction_manifest), "--seed_start", str(plan["repeat_id"]),
            "--seeds", "1", "--device", plan["device"]]
        try:
            trainer.main()
        except EvaluationComplete:
            pass
        else:
            raise RuntimeError("Inference interception was not reached; no result accepted.")
    if not outcome:
        raise RuntimeError("No timing evaluation was produced.")
    if file_hash(plan["checkpoint"]) != plan["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed during evaluation.")
    if file_hash(data_root / "static_meta.pkl") != static_hash or any(
        file_hash(data_root / "events" / "event_{:06d}.npz".format(int(event_id))) != expected
        for event_id, expected in event_hashes.items()
    ):
        raise RuntimeError("Evaluation dataset changed during inference.")
    outcome.update({
        "repeat_id": plan["repeat_id"], "best_epoch": plan["best_epoch"],
        "audit": plan, "evaluated_source_sha256": sources,
        "evaluated_static_sha256": static_hash, "evaluated_event_sha256": event_hashes,
        "training_executed": False, "ordinal_loss_training_executed": False,
        "test_event_ids": plan["test_ids"],
        "scope": "internal held-out events on the existing topology; discrete keyframe timing",
    })
    write_json(output, outcome)


def aggregate_results(results):
    if not results or len({result["repeat_id"] for result in results}) != len(results):
        raise ValueError("Require nonempty, unique repeats.")
    reference = results[0]
    for result in results[1:]:
        if any(result[key] != reference[key] for key in (
            "key_days", "peak_threshold", "near_min_tolerance", "parameter_count",
        )):
            raise ValueError("Repeated evaluations use different timing protocols or capacities.")
        if result["evaluated_source_sha256"] != reference["evaluated_source_sha256"]:
            raise ValueError("Do not pool results evaluated by different source versions.")
        if result["evaluated_static_sha256"] != reference["evaluated_static_sha256"]:
            raise ValueError("Do not pool evaluations from different static networks.")
    def summary(values):
        return {"mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                "per_repeat": values}
    metrics = ("accuracy", "balanced_accuracy_present_classes", "adjacent_accuracy", "day_mae",
               "rank_mae", "near_min_accuracy", "event_macro_accuracy")
    report = {
        "protocol": "frozen_checkpoint_trough_timing_no_training_v1",
        "n_repeats": len(results), "repeat_ids": [result["repeat_id"] for result in results],
        "key_days": reference["key_days"], "peak_threshold": reference["peak_threshold"],
        "evaluation_population": "node-event observations inside the original predicted subgraph; unshocked and peak loss > threshold",
        "label": "earliest minimum production among sampled keyframes; not a continuous-time trough label",
        "training_executed": False, "ordinal_loss_training_executed": False,
        "head_argmax": {key: summary([result["head_argmax"][key] for result in results]) for key in metrics},
        "ordinal_median": {key: summary([result["ordinal_median"][key] for result in results]) for key in metrics},
        "ordinal_probability_scores": {key: summary([result["score_means"][key] for result in results])
                                       for key in reference["score_means"]},
        "near_tie_fraction": summary([result["near_tie_node_fraction"] for result in results]),
        "paired_posthoc_differences": {},
        "p_values_reported": False,
        "statistical_unit": "repeat, not individual nodes; sample SD is descriptive, not an independent-replicate confidence interval",
        "claim_boundary": "accuracy evaluates the existing CE-trained head; ordinal-aware decoding and probability scoring do not test ordinal-loss retraining",
        "per_repeat": results,
    }
    for key in ("accuracy", "day_mae", "rank_mae"):
        differences = [result["ordinal_median"][key] - result["head_argmax"][key] for result in results]
        report["paired_posthoc_differences"][key] = {
            **summary(differences), "definition": "ordinal_median minus head_argmax",
            "ordinal_better_repeats": sum(value > 0 if key == "accuracy" else value < 0 for value in differences),
            "ties": sum(abs(value) <= 1e-12 for value in differences),
        }
    return report


def write_reports(output, report):
    import csv

    write_json(output / "trough_timing_report.json", report)
    def fmt(item, percentage=False):
        scale = 100 if percentage else 1
        mean = item["mean"] * scale
        if item["sample_sd"] is None:
            return "{:.3f} (one repeat)".format(mean)
        return "{:.3f} +/- {:.3f}".format(mean, item["sample_sd"] * scale)
    lines = [
        "# Frozen-Checkpoint Trough-Timing Evaluation", "",
        "Existing CE-trained head; no retraining, optimizer steps, or checkpoint selection on test timing accuracy.",
        "Scope: held-out events on the existing topology, original eligible cascade-node subset.",
        "Key days: {}. Repeats: {}. Mean +/- sample SD across repeats.".format(report["key_days"], report["n_repeats"]),
        "", "| Decoding | Exact Accuracy (%) | Balanced Accuracy (%) | Day MAE | Adjacent Accuracy (%) | Event-Macro Accuracy (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in (("head_argmax", "Existing Head: Argmax"), ("ordinal_median", "Same Head: Ordinal Median (Post Hoc)")):
        metrics = report[name]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(label,
            fmt(metrics["accuracy"], True), fmt(metrics["balanced_accuracy_present_classes"], True),
            fmt(metrics["day_mae"]), fmt(metrics["adjacent_accuracy"], True), fmt(metrics["event_macro_accuracy"], True)))
    lines += ["", "## Ordinal Structure", "",
        "RPS is the mean squared error between cumulative predicted probabilities and cumulative one-hot labels over K-1 ordered thresholds.",
        "Time-weighted RPS weights thresholds by adjacent day gaps divided by the observed day span.",
        "These are post-hoc scores of the SAME CE-trained probability distribution, not results from ordinal-loss training.",
        "", "| Score | Existing Head | Uniform Probability Reference |", "| --- | ---: | ---: |",
        "| RPS | {} | {} |".format(fmt(report["ordinal_probability_scores"]["ranked_probability_score"]),
                                  fmt(report["ordinal_probability_scores"]["uniform_ranked_probability_score"])),
        "| Time-weighted RPS | {} | {} |".format(fmt(report["ordinal_probability_scores"]["time_weighted_rps"]),
                                                fmt(report["ordinal_probability_scores"]["uniform_time_weighted_rps"])),
        "", "## Boundaries", "",
        "Accuracy refers to sampled keyframes, not the unobserved continuous-time trough. Adjacent accuracy is not a fixed-day tolerance.",
        "Labels use the earliest argmin. Near-min accuracy is secondary and uses the stated production-value tolerance.",
        "Class recall is null for unsupported classes; balanced accuracy averages supported classes only.",
        "Last-keyframe targets do not establish that the physical trough occurred within the sampled window.",
        "Node-event observations are not treated as independent experimental replicates; no node-level p-values are reported.",
        "No statement that ordinal-loss retraining improves performance is supported by this evaluation.",
    ]
    (output / "trough_timing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "class_recall.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["repeat", "decoding", "class", "day", "true_count", "predicted_count", "recall"])
        for result in report["per_repeat"]:
            for name in ("head_argmax", "ordinal_median"):
                metrics = result[name]
                for index, day in enumerate(result["key_days"]):
                    writer.writerow([result["repeat_id"], name, index, day, metrics["true_class_counts"][index],
                                     metrics["predicted_class_counts"][index], metrics["per_class_recall"][index]])
    text = [
        "The trough-timing head was evaluated using the existing validation-selected checkpoints, without retraining, on the original held-out event splits.",
        "Evaluation was restricted to unshocked nodes in the prediction subgraph with true peak loss above {:.3f}, matching the auxiliary-supervision population.".format(report["peak_threshold"]),
        "Across {} repeats, its exact keyframe accuracy was {}%, balanced accuracy was {}%, and timing MAE was {} days.".format(
            report["n_repeats"], fmt(report["head_argmax"]["accuracy"], True),
            fmt(report["head_argmax"]["balanced_accuracy_present_classes"], True), fmt(report["head_argmax"]["day_mae"])),
        "Ordinal structure was examined using cumulative-probability scoring and a predefined posterior-median decoding rule applied to the same frozen head.",
        "Posterior-median decoding yielded an exact accuracy of {}% and a timing MAE of {} days; this is a post-hoc decision-rule comparison, not a comparison between CE-trained and ordinal-loss-trained models.".format(
            fmt(report["ordinal_median"]["accuracy"], True), fmt(report["ordinal_median"]["day_mae"])),
        "A cumulative-distribution ordinal loss is a viable alternative to CE because it accounts for ordered keyframes. The present scores characterize ordered errors, but do not establish the outcome of optimizing that alternative loss.",
        "Timing labels are the earliest minima among sampled keyframes; timing resolution and plateau ambiguity are therefore explicitly bounded. Sample variability is reported across repeats, not inferred from treating nodes as independent replicates.",
    ]
    (output / "reviewer_response_evidence.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen trough-timing heads; no training.")
    parser.add_argument("--worker-plan", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--model-label")
    parser.add_argument("--repeats", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--allow-data-relocation", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.worker_plan:
        if not args.worker_output:
            parser.error("Worker output required.")
        worker(json.loads(args.worker_plan.read_text(encoding="utf-8")), args.worker_output)
        return
    if any(value is None for value in (args.run_root, args.manifest_dir, args.code_root, args.model_label, args.output)):
        parser.error("Provide run-root, manifest-dir, code-root, model-label and output.")
    repeats = [int(value.strip()) for value in args.repeats.split(",") if value.strip()]
    if not repeats or len(set(repeats)) != len(repeats) or min(repeats) < 0:
        parser.error("Repeats must be unique nonnegative integers.")
    for key in ("run_root", "manifest_dir", "code_root", "output"):
        setattr(args, key, getattr(args, key).resolve())
    if not (args.code_root / "pi_dl" / "train_v3de.py").is_file():
        raise FileNotFoundError("Original pi_dl training source is required for faithful model reconstruction.")
    if args.output.exists():
        raise FileExistsError("Require a fresh report directory: {}".format(args.output))
    plans = [make_plan(args, repeat_id) for repeat_id in repeats]
    print("[preflight] {} complete checkpoints; frozen inference only; device={}; no training".format(
        len(plans), args.device), flush=True)
    if args.check_only:
        return
    args.output.mkdir(parents=True)
    results = []
    for plan in plans:
        repeat_id = plan["repeat_id"]
        plan_path = args.output / "plan_{:02d}.json".format(repeat_id)
        result_path = args.output / "repeat_{:02d}.json".format(repeat_id)
        write_json(plan_path, plan)
        environment = dict(os.environ)
        environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                            "CUDA_VISIBLE_DEVICES": "" if args.device == "cpu" else args.gpu,
                            "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONUNBUFFERED": "1"})
        print("[timing] evaluating repeat={} original held-out events".format(repeat_id), flush=True)
        with (args.output / "repeat_{:02d}.log".format(repeat_id)).open("x", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                            "--worker-plan", str(plan_path), "--worker-output", str(result_path)],
                           cwd=str(args.code_root), env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        results.append(json.loads(result_path.read_text(encoding="utf-8")))
    report = aggregate_results(results)
    write_reports(args.output, report)
    print("[done] {} frozen repeats; results={} ; no ordinal-loss training claim".format(len(results), args.output), flush=True)


if __name__ == "__main__":
    main()