"""Audit assets and benchmark MeCaSNet against both cascade simulators.

The benchmark separates one-time startup from steady-state per-event latency.
Simulator events and surrogate events follow the same domain protocol but are
not paired event-by-event; this is a wall-clock throughput comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Could not locate the repository root")


ROOT = repository_root()
RELEASE = ROOT
FIX_DIR = ROOT / "fix"
for path in (ROOT, RELEASE, FIX_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mecasnet.config import Config
from mecasnet.data import CascadeEventDataset, StaticNetwork, collate_single
from mecasnet.factory import LEGACY_Y8_PROFILE, build_mecasnet


DOMAINS = {
    "henriet": {
        "checkpoint": ROOT / "checkpoints/legacy_y8_henriet_seed0.pt",
        "data": ROOT / "data/reference/henriet",
        "simulator_inputs": [
            ROOT / "private_backend/henriet/nodes_attr.csv",
            ROOT / "private_backend/henriet/edges_static.csv",
        ],
        "simulator": "ARIO/Henriet",
    },
    "inoue": {
        "checkpoint": ROOT / "checkpoints/legacy_inoue_seed0.pt",
        "data": ROOT / "data/reference/inoue",
        "simulator_inputs": [
            ROOT / "private_backend/inoue/nodes_attr.csv",
            ROOT / "private_backend/inoue/edges_static.csv",
        ],
        "simulator": "Inoue-Todo",
        "expected_parameters": 229873,
        "reference_static_meta": ROOT / "data/reference/inoue/static_meta_reference.pkl",
        "simulator_env": {
            "SUBNET_DIR": str(ROOT / "private_backend/inoue"),
            "DELTA_PER_FIRM_BETA": "1",
            "SECTOR_DELTA_MULT": "0.61",
            "HUB_DELTA_MULT": "1.11",
            "RANDOM_DELTA_MULT": "1.11",
            "TIER_DELTA_MULT": "1.0",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--surrogate-events", type=int, default=20)
    parser.add_argument("--simulator-events", type=int, default=5)
    parser.add_argument("--warmup-forwards", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=70000)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/reference/dual_scale_runtime.json",
    )
    return parser.parse_args()


def event_ids(data_root: Path) -> list[int]:
    result = []
    for path in (data_root / "events").glob("event_*.npz"):
        try:
            result.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(result)


def compare_static_networks(local_path: Path, reference_path: Path) -> dict[str, Any]:
    with local_path.open("rb") as stream:
        local = pickle.load(stream)
    with reference_path.open("rb") as stream:
        reference = pickle.load(stream)
    checks = {
        "nodes": int(local["V"]) == int(reference["V"]),
        "edges": int(local["E"]) == int(reference["E"]),
        "firms": np.array_equal(np.asarray(local["firms"]), np.asarray(reference["firms"])),
        "edge_src": np.array_equal(np.asarray(local["edge_src"]), np.asarray(reference["edge_src"])),
        "edge_dst": np.array_equal(np.asarray(local["edge_dst"]), np.asarray(reference["edge_dst"])),
    }
    return {
        "local_nodes": int(local["V"]),
        "local_edges": int(local["E"]),
        "reference_nodes": int(reference["V"]),
        "reference_edges": int(reference["E"]),
        "checks": checks,
        "matches": all(checks.values()),
    }


def audit_domain(domain: str) -> dict[str, Any]:
    spec = DOMAINS[domain]
    ids = event_ids(spec["data"])
    required = {
        "checkpoint": spec["checkpoint"],
        "static_meta": spec["data"] / "static_meta.pkl",
        "events_dir": spec["data"] / "events",
        **{f"simulator_input_{index}": path
           for index, path in enumerate(spec["simulator_inputs"], 1)},
    }
    if "reference_static_meta" in spec:
        required["reference_static_meta"] = spec["reference_static_meta"]
    files = {name: {"path": str(path), "exists": path.exists()}
             for name, path in required.items()}
    static_meta = None
    if required["static_meta"].exists():
        with required["static_meta"].open("rb") as stream:
            payload = pickle.load(stream)
        params = payload.get("params", {})
        static_meta = {
            "nodes": int(payload["V"]),
            "edges": int(payload["E"]),
            "seed_base": params.get("seed_base"),
            "horizon": params.get("T"),
            "warmup": params.get("warmup"),
        }
    static_alignment = None
    if domain == "inoue" and required["static_meta"].exists() and required["reference_static_meta"].exists():
        static_alignment = compare_static_networks(
            required["static_meta"], required["reference_static_meta"]
        )
    ready = all(record["exists"] for record in files.values()) and bool(ids)
    if static_alignment is not None:
        ready = ready and static_alignment["matches"]
    return {
        "domain": domain,
        "simulator": spec["simulator"],
        "training_data": spec.get("training_data"),
        "simulator_env": spec.get("simulator_env"),
        "files": files,
        "event_count": len(ids),
        "event_id_min": min(ids) if ids else None,
        "event_id_max": max(ids) if ids else None,
        "static_meta": static_meta,
        "static_alignment_with_i_a1_d_v1": static_alignment,
        "repair_replay": (
            "directly reconstructable from full delta0 and global recovery_days"
            if domain == "henriet"
            else "must regenerate from seed_base + event_id and verify shock_mask/delta0"
        ),
        "ready": ready,
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean_seconds": float(array.mean()),
        "median_seconds": float(np.median(array)),
        "p95_seconds": float(np.percentile(array, 95)),
        "std_seconds": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def load_i_a1_surrogate(spec: dict[str, Any], device: torch.device):
    """Rebuild I_A1 with the exact residual-model lineage used in training."""
    import torch.nn as nn
    from mecasnet.config import Config as InoueConfig
    from mecasnet.data import CascadeEventDataset as InoueDataset
    from mecasnet.data import StaticNetwork as InoueStaticNetwork
    from mecasnet.model_v3_residual import KSGATv3Residual

    cfg = InoueConfig(data_root=str(spec["data"]), seed=0,
                      event_scalars_mode="minimal")
    net = InoueStaticNetwork(cfg)
    ids = event_ids(spec["data"])
    dataset = InoueDataset(cfg, net, ids, train_mode=False)
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = KSGATv3Residual(
        cfg, Fv=net.Fv, peak_mode="traj", prewarm_layers=3,
        decoder_mode="param2", param2_preset="g3_narrow",
    )

    # I_A1 predates the later d -> 2d -> K residual-head widening.
    head_w0 = state_dict["late_residual_head.0.weight"]
    head_w2 = state_dict["late_residual_head.2.weight"]
    current_head = model.late_residual_head
    if current_head[0].out_features != head_w0.shape[0]:
        model.late_residual_head = nn.Sequential(
            nn.Linear(head_w0.shape[1], head_w0.shape[0]),
            type(current_head[1])(),
            nn.Linear(head_w0.shape[0], head_w2.shape[0]),
        )
    model.load_state_dict(state_dict, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != spec["expected_parameters"]:
        raise RuntimeError(
            f"I_A1 reconstructed {parameter_count} parameters; expected "
            f"{spec['expected_parameters']}."
        )
    return cfg, net, dataset, model.to(device).eval()


def load_surrogate(domain: str, device: torch.device):
    spec = DOMAINS[domain]
    if domain == "inoue":
        return load_i_a1_surrogate(spec, device)
    cfg = Config(data_root=str(spec["data"]), seed=0, event_scalars_mode="minimal")
    net = StaticNetwork(cfg)
    ids = event_ids(spec["data"])
    dataset = CascadeEventDataset(cfg, net, ids, train_mode=False)
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = build_mecasnet(
        cfg, net.Fv, profile=LEGACY_Y8_PROFILE, propagation_steps=4
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return cfg, net, dataset, model


@torch.inference_mode()
def benchmark_surrogate(domain: str, device: torch.device, n_events: int,
                        warmup_forwards: int) -> dict[str, Any]:
    start = time.perf_counter()
    cfg, net, dataset, model = load_surrogate(domain, device)
    startup = time.perf_counter() - start
    if not dataset:
        raise RuntimeError(f"No events found for {domain}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_single)
    batches = []
    for batch in loader:
        batches.append(to_device(batch, device))
        if len(batches) >= max(n_events, warmup_forwards):
            break
    for batch in batches[:warmup_forwards]:
        model(batch)
    synchronize(device)
    times = []
    for batch in batches[:n_events]:
        synchronize(device)
        event_start = time.perf_counter()
        model(batch)
        synchronize(device)
        times.append(time.perf_counter() - event_start)
    result = summarize(times)
    result.update({
        "startup_seconds": startup,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "nodes": net.V,
        "edges": net.E,
        "available_events": len(dataset),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "expected_parameter_count": DOMAINS[domain].get("expected_parameters"),
    })
    return result


def timed_simulator_factory(domain: str) -> tuple[Any, float]:
    from sim_env import HenrietEnv, InoueEnv

    start = time.perf_counter()
    if domain == "henriet":
        env = HenrietEnv(
            budget=5,
            horizon=200,
            meta_pkl=str(DOMAINS[domain]["data"] / "static_meta.pkl"),
        )
    else:
        # `simulate_inoue_on_real_network` reads these at import time.
        # Set them before InoueEnv imports that module.
        os.environ.update(DOMAINS[domain]["simulator_env"])
        env = InoueEnv(budget=25, horizon=365, warmup=100,
                       subgraph_hops=None)
    return env, time.perf_counter() - start


def benchmark_simulator(domain: str, n_events: int, seed_start: int) -> dict[str, Any]:
    env, startup = timed_simulator_factory(domain)
    times = []
    state = None
    for offset in range(n_events):
        event_start = time.perf_counter()
        state = env.reset(seed_start + offset)
        times.append(time.perf_counter() - event_start)
    if state is None:
        raise RuntimeError("simulator-events must be at least 1")
    result = summarize(times)
    result.update({
        "startup_seconds": startup,
        "nodes": state.n_nodes,
        "edges": int(state.edge_src.size),
        "horizon_days": 200 if domain == "henriet" else 365,
        "warmup_days": 0 if domain == "henriet" else 100,
    })
    return result


def main() -> None:
    args = parse_args()
    if args.surrogate_events < 1 or args.simulator_events < 1:
        raise ValueError("event counts must be positive")
    audits = {domain: audit_domain(domain) for domain in args.domains}
    report: dict[str, Any] = {
        "methodology": {
            "timing": "wall-clock, batch_size=1, model/data startup reported separately",
            "comparison": "same-domain protocol; simulator and surrogate events are not paired",
            "surrogate_warmup_forwards": args.warmup_forwards,
        },
        "audit": audits,
        "benchmarks": {},
    }
    print(json.dumps({"audit": audits}, indent=2))
    if args.audit_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved audit: {args.output}")
        return

    not_ready = [domain for domain, record in audits.items() if not record["ready"]]
    if not_ready:
        raise FileNotFoundError(f"Missing assets for domains: {', '.join(not_ready)}")

    device = torch.device(args.device)
    for domain in args.domains:
        print(f"\n=== {domain}: MeCaSNet ===", flush=True)
        surrogate = benchmark_surrogate(
            domain, device, args.surrogate_events, args.warmup_forwards
        )
        print(json.dumps(surrogate, indent=2), flush=True)
        print(f"=== {domain}: {DOMAINS[domain]['simulator']} ===", flush=True)
        simulator = benchmark_simulator(domain, args.simulator_events, args.seed_start)
        print(json.dumps(simulator, indent=2), flush=True)
        report["benchmarks"][domain] = {
            "surrogate": surrogate,
            "simulator": simulator,
            "speedup_median": (
                simulator["median_seconds"] / surrogate["median_seconds"]
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved benchmark: {args.output}")


if __name__ == "__main__":
    main()
