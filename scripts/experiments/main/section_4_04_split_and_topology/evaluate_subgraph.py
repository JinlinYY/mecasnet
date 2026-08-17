"""Evaluate a frozen full-graph checkpoint on re-equilibrated induced subgraphs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mecasnet.config import Config
from mecasnet.data import CascadeEventDataset, StaticNetwork, collate_single
from mecasnet.factory import COMPAT_PROFILE, build_mecasnet
from mecasnet.model_baselines_strong import DirGNNBaseline
from mecasnet.runner import evaluate


EXPECTED_COMPAT_PARAMS = 289512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot frozen-checkpoint evaluation on one induced graph."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=("mecasnet", "dirgnn"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--event-scalars-mode", default="minimal",
                        choices=("full", "clean", "minimal"))
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint has no state_dict mapping: {path}")
    return state


def build_model(model_name: str, cfg: Config, fv: int) -> torch.nn.Module:
    if model_name == "mecasnet":
        model = build_mecasnet(
            cfg, fv, profile=COMPAT_PROFILE, propagation_steps=4
        )
        params = sum(parameter.numel() for parameter in model.parameters())
        if params != EXPECTED_COMPAT_PARAMS:
            raise RuntimeError(
                "Compatibility-profile architecture mismatch: "
                f"expected {EXPECTED_COMPAT_PARAMS}, got {params}"
            )
        return model
    return DirGNNBaseline(cfg, Fv=fv, n_layers=8, d_hidden=96)


def main() -> None:
    args = parse_args()
    manifest_path = args.data_root / "large_subgraph_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing induced-subgraph manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event_paths = sorted((args.data_root / "events").glob("event_*.npz"))
    if not event_paths:
        raise FileNotFoundError(f"No event files under {args.data_root / 'events'}")
    event_ids = [int(path.stem.split("_")[-1]) for path in event_paths]

    cfg = Config()
    cfg.data_root = str(args.data_root)
    cfg.event_scalars_mode = args.event_scalars_mode
    cfg.include_audit_metadata = True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    network = StaticNetwork(cfg)
    if network.n_sec != 269 or network.Fv != 274:
        raise RuntimeError(
            f"Checkpoint feature contract violated: expected n_sec=269/Fv=274, "
            f"got n_sec={network.n_sec}/Fv={network.Fv}"
        )

    model = build_model(args.model, cfg, network.Fv)
    state = load_checkpoint(args.checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    dataset = CascadeEventDataset(cfg, network, event_ids)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_single)
    metrics = evaluate(model, loader, cfg, device)
    output = {
        "status": "PASS",
        "protocol": manifest["protocol_version"],
        "claim_boundary": manifest["claim_boundary"],
        "block": manifest["block"],
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "event_scalars_mode": args.event_scalars_mode,
        "static_feature_policy": (
            "degrees, reverse PageRank, GSCC membership, and continuous-feature "
            "standardization are recomputed on each induced graph"
        ),
        "induced_graph": manifest["induced_graph"],
        "n_events": len(event_ids),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, default=float))
    print(f"LARGE_SUBGRAPH_EVAL_PASS model={args.model} block={manifest['block']}")


if __name__ == "__main__":
    main()
