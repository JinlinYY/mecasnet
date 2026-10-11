"""Paired model comparisons across seed-specific disruption-event partitions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


DEFAULT_SEEDS = tuple(range(10))
CONFLICTING_ARGUMENTS = {
    "--seeds", "--seed_start", "--out_dir", "--split_manifest", "--split_seed",
}


@dataclass(frozen=True)
class Split:
    repeat_id: int
    split_seed: int
    train_ids: List[int]
    validation_ids: List[int]
    internal_test_ids: List[int]


def parse_seed_list(value: str) -> List[int]:
    seeds = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique integer repeat IDs.")
    return seeds


def canonical_hash(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def event_ids(data_root: Path) -> List[int]:
    events_dir = data_root / "events"
    ids = sorted(int(path.stem.split("_")[1]) for path in events_dir.glob("event_*.npz"))
    if not ids:
        raise FileNotFoundError(f"No event_*.npz files found under {events_dir}")
    return ids


def make_split(ids: List[int], seed: int, train_fraction: float, validation_fraction: float) -> Split:
    shuffled = list(ids)
    np.random.default_rng(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * train_fraction)
    validation_count = int(len(shuffled) * validation_fraction)
    split = Split(
        repeat_id=seed,
        split_seed=seed,
        train_ids=shuffled[:train_count],
        validation_ids=shuffled[train_count:train_count + validation_count],
        internal_test_ids=shuffled[train_count + validation_count:],
    )
    sets = [set(split.train_ids), set(split.validation_ids), set(split.internal_test_ids)]
    if any(sets[left] & sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError(f"Repeat {seed} contains overlapping split IDs.")
    if not all((split.train_ids, split.validation_ids, split.internal_test_ids)):
        raise RuntimeError(f"Repeat {seed} contains an empty split.")
    return split


def write_manifests(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = event_ids(data_root)
    manifests = []
    for repeat_id in parse_seed_list(args.seeds):
        split = make_split(ids, repeat_id, args.train_fraction, args.validation_fraction)
        payload = {
            "protocol": "repeated_event_identity_split_v1",
            "repeat_id": split.repeat_id,
            "split_seed": split.split_seed,
            "data_root": str(data_root),
            "event_count": len(ids),
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "train_ids": split.train_ids,
            "validation_ids": split.validation_ids,
            "internal_test_ids": split.internal_test_ids,
        }
        payload["manifest_sha256"] = canonical_hash(payload)
        path = output_dir / f"repeat_{repeat_id:02d}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifests.append({
            "repeat_id": repeat_id,
            "path": str(path),
            "sha256": payload["manifest_sha256"],
            "n_train": len(split.train_ids),
            "n_validation": len(split.validation_ids),
            "n_test": len(split.internal_test_ids),
        })
    index = {
        "protocol": "repeated_event_identity_split_v1",
        "data_root": str(data_root),
        "event_count": len(ids),
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "repeats": manifests,
    }
    (output_dir / "manifest_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"Created {len(manifests)} immutable split manifests in {output_dir}")


def load_config(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Experiment config must contain a nonempty 'models' list.")
    labels = set()
    for model in models:
        if not isinstance(model.get("label"), str) or not isinstance(model.get("variant"), str):
            raise ValueError("Every model needs string fields 'label' and 'variant'.")
        if model["label"] in labels:
            raise ValueError(f"Duplicate model label: {model['label']}")
        labels.add(model["label"])
        if not isinstance(model.get("env", {}), dict):
            raise ValueError(f"env must be an object for {model['label']}")
        if not isinstance(model.get("train_args", []), list) or not all(
            isinstance(argument, str) for argument in model.get("train_args", [])
        ):
            raise ValueError(f"train_args must be a list of strings for {model['label']}")
    return models


def load_manifest(manifest_dir: Path, repeat_id: int) -> Tuple[Path, Dict[str, Any]]:
    path = manifest_dir / f"repeat_{repeat_id:02d}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed_hash = payload.pop("manifest_sha256", None)
    if claimed_hash != canonical_hash(payload):
        raise RuntimeError(f"Manifest hash mismatch: {path}")
    payload["manifest_sha256"] = claimed_hash
    return path, payload


def ensure_common_args_safe(common_args: List[str]) -> None:
    conflicting = sorted(set(common_args) & CONFLICTING_ARGUMENTS)
    if conflicting:
        raise ValueError(f"Do not pass {conflicting} in --common-train-args; the runner controls them.")


def select_models(models: List[Dict[str, Any]], labels: str) -> List[Dict[str, Any]]:
    if not labels:
        return models
    requested = {label.strip() for label in labels.split(",") if label.strip()}
    available = {model["label"] for model in models}
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(f"Unknown model labels: {unknown}")
    return [model for model in models if model["label"] in requested]


def run_experiments(args: argparse.Namespace) -> None:
    manifest_dir = Path(args.manifest_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.experiment_config).resolve()
    models = select_models(load_config(config_path), args.model_labels)
    common_args = shlex.split(args.common_train_args)
    ensure_common_args_safe(common_args)
    repeats = parse_seed_list(args.repeats)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    for repeat_id in repeats:
        manifest_path, manifest = load_manifest(manifest_dir, repeat_id)
        for model in models:
            run_dir = output_dir / f"repeat_{repeat_id:02d}" / model["label"]
            run_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = run_dir / "run_metadata.json"
            command = [
                sys.executable, "-m", args.trainer_module,
                "--variant", model["variant"],
                "--split_manifest", str(manifest_path),
                "--split_seed", str(repeat_id),
                "--seeds", "1", "--seed_start", str(repeat_id),
                "--out_dir", str(run_dir),
                "--save_ckpt",
                *model.get("train_args", []),
                *common_args,
            ]
            metadata = {
                "protocol": "paired_repeated_split_v1",
                "repeat_id": repeat_id,
                "split_seed": manifest["split_seed"],
                "initialization_seed": repeat_id,
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest["manifest_sha256"],
                "experiment_config": str(config_path),
                "experiment_config_sha256": config_hash,
                "model": model,
                "command": command,
                "status": "running",
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            print(f"\n=== repeat={repeat_id} model={model['label']} ===", flush=True)
            child_environment = dict(os.environ)
            for name in list(child_environment):
                if name.startswith(("GNS_", "MECAS_")) or name == "D_HIDDEN_OVERRIDE":
                    child_environment.pop(name)
            child_environment.update({str(k): str(v) for k, v in model.get("env", {}).items()})
            completed = subprocess.run(
                command,
                cwd=args.working_dir,
                env=child_environment,
            )
            metadata["returncode"] = completed.returncode
            metadata["status"] = "completed" if completed.returncode == 0 else "failed"
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"Training failed for repeat={repeat_id}, model={model['label']}")






def summary_path(run_root: Path, repeat_id: int, label: str, variant: str) -> Path:
    return run_root / f"repeat_{repeat_id:02d}" / label / f"{variant.lower()}_summary.json"


def prepare_json_report(report: Dict[str, Any]) -> Dict[str, Any]:
    non_finite_fields = []

    def clean(value: Any, path: str) -> Any:
        if isinstance(value, float) and not np.isfinite(value):
            non_finite_fields.append({"path": path, "original_value": str(value)})
            return None
        if isinstance(value, dict):
            return {key: clean(item, f"{path}.{key}") for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item, f"{path}[{index}]") for index, item in enumerate(value)]
        return value

    cleaned = clean(report, "$")
    cleaned["non_finite_fields"] = non_finite_fields
    return cleaned








