from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repeated_split_experiment import canonical_hash, load_manifest


def prepare_training(arguments, trainer):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--split_seed", type=int, required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--seed_start", type=int, required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    args, remaining = parser.parse_known_args(arguments)
    if args.seeds != 1 or args.seed_start != args.split_seed:
        raise ValueError("Each task must use one aligned initialization/split seed.")
    if "--resume_from" in remaining:
        raise ValueError("Repeated NEEQ runs must train from scratch.")
    original_path, manifest = load_manifest(Path(args.split_manifest).parent, args.split_seed)
    if original_path.resolve() != Path(args.split_manifest).resolve():
        raise ValueError("Unexpected manifest path.")
    if manifest["repeat_id"] != args.split_seed or manifest["split_seed"] != args.split_seed:
        raise ValueError("Manifest repeat identity mismatch.")
    converted = dict(manifest)
    converted.pop("manifest_sha256")
    converted["source_manifest_sha256"] = manifest["manifest_sha256"]
    converted["strict_test_ids"] = list(manifest["internal_test_ids"])
    converted["manifest_sha256"] = canonical_hash(converted)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    converted_path = output / "trainer_manifest.json"
    if converted_path.exists():
        raise FileExistsError(converted_path)
    converted_path.write_text(json.dumps(converted, indent=2) + "\n", encoding="utf-8")
    config_type = trainer.Config

    class RepeatConfig(config_type):
        def __init__(self, *values, **keywords):
            super().__init__(*values, **keywords)
            self.seed = args.split_seed

    trainer.Config = RepeatConfig
    return remaining + [
        "--split_manifest", str(converted_path), "--seed_start", str(args.seed_start),
        "--seeds", "1", "--out_dir", str(output),
    ]


def main():
    from pi_dl import train_v3de

    sys.argv = [sys.argv[0]] + prepare_training(sys.argv[1:], train_v3de)
    train_v3de.main()


if __name__ == "__main__":
    main()