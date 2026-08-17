"""Re-evaluate completed Table-2 checkpoints using the paper cascade threshold.

Paper cascade subset: unshocked nodes with ground-truth peak loss p_v > 0.05.
This is a post-hoc evaluation only: no retraining, no weight updates.

Run from the repository root with an authorized event set and the corresponding
matched-seed checkpoints. Use ``--help`` for the path and threshold options.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

from mecasnet.config import Config
from mecasnet.data import CascadeEventDataset, StaticNetwork, collate_single, split_event_ids
from mecasnet.factory import COMPAT_PROFILE, build_mecasnet
from mecasnet.model_v2 import PlainGATBaseline, PlainGCNBaseline, PlainMLPBaseline
from mecasnet.model_baselines_strong import DirGNNBaseline, STGNNBaseline


KEY_DAYS = (0, 5, 10, 20, 30, 50, 70, 100, 150, 199)
MODELS = ("MLP", "GCN", "GAT", "STGNN", "DirGNN", "MeCaSNet")
CHECKPOINT_BASE = {
    "MLP": "mlp", "GCN": "gcn", "GAT": "gat", "STGNN": "stgnn",
    "DirGNN": "dirgnn", "MeCaSNet": "mecasnet",
}
REFERENCE = "MeCaSNet"
SIGNIFICANCE_METRICS = ("mae_pk_csc", "r2_pk_csc", "r2_kf_csc_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="multiseed checkpoint root")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--models", default=",".join(MODELS),
        help="comma-separated model names to evaluate; default evaluates all models",
    )
    return parser.parse_args()


def r2(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.size == 0 or float(target.std()) < 1e-2:
        return float("nan")
    residual = float(np.square(prediction - target).sum())
    total = float(np.square(target - target.mean()).sum()) + 1e-8
    return 1.0 - residual / total


def mean_std(values: list[float]) -> dict[str, float]:
    valid = [value for value in values if np.isfinite(value)]
    if not valid:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
    }


def paired_ttest(reference: list[dict[str, Any]], baseline: list[dict[str, Any]],
                 metric: str) -> dict[str, Any]:
    """Test matched seeds; a positive mean difference favors MeCaSNet."""
    reference_by_seed = {run["seed"]: run[metric] for run in reference}
    baseline_by_seed = {run["seed"]: run[metric] for run in baseline}
    common_seeds = sorted(set(reference_by_seed) & set(baseline_by_seed))
    differences = [reference_by_seed[seed] - baseline_by_seed[seed]
                   for seed in common_seeds
                   if np.isfinite(reference_by_seed[seed]) and np.isfinite(baseline_by_seed[seed])]
    if len(differences) < 2:
        return {"n": len(differences), "seeds": common_seeds, "t": None, "p": None,
                "cohens_d": None, "mean_diff_mecasnet_minus_baseline": None}
    differences_array = np.asarray(differences, dtype=float)
    std_diff = float(differences_array.std(ddof=1))
    mean_diff = float(differences_array.mean())
    if std_diff == 0.0:
        if mean_diff == 0.0:
            return {"n": len(differences), "seeds": common_seeds, "t": 0.0, "p": 1.0,
                "cohens_d": 0.0, "mean_diff_mecasnet_minus_baseline": 0.0}
        return {"n": len(differences), "seeds": common_seeds, "t": float("inf"), "p": 0.0,
                "cohens_d": float("inf"), "mean_diff_mecasnet_minus_baseline": mean_diff}
    t_stat = mean_diff / (std_diff / math.sqrt(len(differences)))
    p_value = (float(2 * scipy_stats.t.sf(abs(t_stat), df=len(differences) - 1))
               if scipy_stats is not None else None)
    return {"n": len(differences), "seeds": common_seeds, "t": float(t_stat), "p": p_value,
            "cohens_d": float(mean_diff / std_diff),
            "mean_diff_mecasnet_minus_baseline": mean_diff}


def build_model(name: str, cfg: Config, feature_count: int) -> torch.nn.Module:
    if name == "MLP":
        return PlainMLPBaseline(cfg, Fv=feature_count, n_layers=3)
    if name == "GCN":
        return PlainGCNBaseline(cfg, Fv=feature_count, n_layers=3)
    if name == "GAT":
        return PlainGATBaseline(cfg, Fv=feature_count, n_layers=3)
    if name == "STGNN":
        return STGNNBaseline(cfg, Fv=feature_count, n_spatial=2, d_hidden=160)
    if name == "DirGNN":
        return DirGNNBaseline(cfg, Fv=feature_count, n_layers=8, d_hidden=96)
    if name == "MeCaSNet":
        return build_mecasnet(
            cfg, feature_count, profile=COMPAT_PROFILE, propagation_steps=4
        )
    raise ValueError(f"Unknown model: {name}")


def checkpoint_paths(root: Path, name: str) -> list[Path]:
    base = CHECKPOINT_BASE[name]
    return sorted((root / name).glob(f"seed*/{base}_seed*.pt"))


@torch.no_grad()
def evaluate_checkpoint(checkpoint: Path, name: str, cfg: Config, feature_count: int,
                        loader: DataLoader, threshold: float,
                        device: torch.device) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    model = build_model(name, cfg, feature_count).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    peaks_pred, peaks_gt, shocks, kf_pred, kf_gt = [], [], [], [], []
    for raw_batch in loader:
        batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                 for key, value in raw_batch.items()}
        output = model(batch)
        peaks_pred.append(output["peak"].detach().cpu().numpy())
        peaks_gt.append(batch["peak_loss"].detach().cpu().numpy())
        shocks.append(batch["shock_mask"].detach().cpu().numpy() > 0.5)
        kf_pred.append(output["u_keyframes"].detach().cpu().numpy())
        kf_gt.append(batch["u_keyframes"].detach().cpu().numpy())

    prediction = np.concatenate(peaks_pred)
    target = np.concatenate(peaks_gt)
    shocked = np.concatenate(shocks)
    cascade = (~shocked) & (target > threshold)
    predicted_kf = np.concatenate(kf_pred, axis=1)
    target_kf = np.concatenate(kf_gt, axis=1)
    by_day = [r2(predicted_kf[index, cascade], target_kf[index, cascade])
              for index in range(predicted_kf.shape[0])]
    valid = [value for value in by_day if np.isfinite(value)]
    return {
        "checkpoint": str(checkpoint),
        "seed": int(checkpoint.parent.name.removeprefix("seed")),
        "checkpoint_best_epoch": payload.get("epoch"),
        "threshold": threshold,
        "definition": "not_directly_shocked AND ground_truth_peak_loss > threshold",
        "n_total": int(target.size),
        "n_shocked": int(shocked.sum()),
        "n_cascade": int(cascade.sum()),
        "cascade_fraction_of_unshocked": float(cascade.sum() / max((~shocked).sum(), 1)),
        "mae_pk": float(np.abs(prediction - target).mean()),
        "r2_pk": r2(prediction, target),
        "mae_pk_csc": float(np.abs(prediction[cascade] - target[cascade]).mean()),
        "r2_pk_csc": r2(prediction[cascade], target[cascade]),
        "r2_kf_csc_mean": float(np.mean(valid)),
        "r2_kf_csc_by_day": dict(zip(map(str, KEY_DAYS), by_day)),
    }


def main() -> None:
    args = parse_args()
    models = tuple(name.strip() for name in args.models.split(",") if name.strip())
    unknown_models = sorted(set(models) - set(MODELS))
    if unknown_models:
        raise ValueError(f"Unknown model names: {', '.join(unknown_models)}")
    if not models:
        raise ValueError("At least one model must be selected.")
    if REFERENCE not in models:
        raise ValueError(f"--models must include the reference model {REFERENCE}.")

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable in this Python environment. "
            "Use a CUDA-enabled PyTorch installation or explicitly pass --device cpu."
        )
    device = torch.device(args.device)
    print(f"Using device: {device}", flush=True)

    cfg = Config(data_root=args.data_root)
    cfg.train_frac = 0.80
    cfg.val_frac = 0.10
    cfg.deq_iters = 1
    cfg.deq_grad_iters = 1
    cfg.event_scalars_mode = "minimal"
    network = StaticNetwork(cfg)
    _, _, test_ids = split_event_ids(Path(args.data_root) / cfg.events_dir, cfg)
    dataset = CascadeEventDataset(cfg, network, test_ids[:args.n_test], train_mode=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                        collate_fn=collate_single, pin_memory=device.type == "cuda")

    report: dict[str, Any] = {
        "protocol": {
            "cascade_threshold": args.threshold,
            "definition": "not_directly_shocked AND ground_truth_peak_loss > threshold",
            "data_root": args.data_root,
            "test_split": "seed=0, 80/10/10; 500 test events",
            "event_scalars_mode": "minimal",
            "note": "Post-hoc checkpoint evaluation; no model retraining.",
            "models": list(models),
        },
        "per_seed": {},
        "mean_std": {},
        "significance_vs_mecasnet": {},
        "scipy_available": scipy_stats is not None,
    }

    for name in models:
        runs = []
        paths = checkpoint_paths(root, name)
        if not paths:
            raise FileNotFoundError(
                f"No checkpoints found for {name} under {root / name}. "
                "Download or train every model before running the paper table evaluation."
            )
        print(f"=== {name}: {len(paths)} checkpoints ===", flush=True)
        for checkpoint in paths:
            print(f"Evaluating {checkpoint.name}", flush=True)
            metrics = evaluate_checkpoint(checkpoint, name, cfg, network.Fv, loader,
                                          args.threshold, device)
            runs.append(metrics)
            print(f"  csc={metrics['r2_pk_csc']:+.4f} "
                  f"kf={metrics['r2_kf_csc_mean']:+.4f} "
                  f"n={metrics['n_cascade']}", flush=True)
        report["per_seed"][name] = runs
        report["mean_std"][name] = {
            metric: mean_std([run[metric] for run in runs])
            for metric in ("mae_pk", "r2_pk", "mae_pk_csc", "r2_pk_csc", "r2_kf_csc_mean")
        }

    reference_runs = report["per_seed"][REFERENCE]
    for name in models:
        if name == REFERENCE:
            continue
        report["significance_vs_mecasnet"][name] = {
            metric: paired_ttest(reference_runs, report["per_seed"][name], metric)
            for metric in SIGNIFICANCE_METRICS
        }

    threshold_tag = f"{args.threshold:.2f}".replace(".", "")
    out = output_dir / f"paper_threshold_{threshold_tag}_multiseed.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print("\n=== Paper threshold p_v > %.2f (mean +/- std) ===" % args.threshold)
    print(f"{'Model':<11}{'MAE':>16}{'R2pk':>16}{'MAE_csc':>16}{'R2pk,csc':>16}{'R2kf,csc':>16}")
    for name in models:
        row = report["mean_std"][name]
        def cell(metric: str) -> str:
            value = row[metric]
            return f"{value['mean']:.4f}+/-{value['std']:.4f}"
        print(f"{name:<11}{cell('mae_pk'):>16}{cell('r2_pk'):>16}"
              f"{cell('mae_pk_csc'):>16}{cell('r2_pk_csc'):>16}"
              f"{cell('r2_kf_csc_mean'):>16}")
    print("\nPaired two-sided t-tests: MeCaSNet minus baseline (matched seeds)")
    for name, tests in report["significance_vs_mecasnet"].items():
        for metric, test in tests.items():
            p_text = "n/a" if test["p"] is None else f"{test['p']:.4g}"
            print(f"  {name:<11} {metric:<18} n={test['n']} p={p_text}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
