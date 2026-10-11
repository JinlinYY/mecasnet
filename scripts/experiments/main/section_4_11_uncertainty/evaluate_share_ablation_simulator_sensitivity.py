"""Evaluate uniform-share model ablation and ARIO share sensitivity.

The two analyses are deliberately separated. The frozen-model ablation changes
only model inputs while retaining the original labels. The simulator analysis
holds Day-0 disruptions fixed, perturbs the full ARIO coefficient matrix, and
regenerates labels from an internally consistent equilibrium for each matrix.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader


# NumPy 2 pickles reference ``numpy._core``. The deployment server uses the
# latest NumPy compatible with Python 3.8, where the same modules live under
# ``numpy.core``.
if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "flagship_y8h_download"
for path in (ROOT, ARCHIVE, ROOT / "cangzhou_pipeline", ROOT / "flagship_train"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import stage3_ario_henriet2012 as ario
from fix.sim_env import HenrietEnv
from pi_dl.config import Config
from pi_dl.data import CascadeEventDataset, StaticNetwork, collate_single, split_event_ids
from y8_reviewer_experiments import load_y8, to_device


KEY_DAYS = np.asarray([0, 5, 10, 20, 30, 50, 70, 100, 150, 199], dtype=np.int64)
CASCADE_THRESHOLD = 0.05


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=ARCHIVE / "data" / "cascade_henriet_5000")
    parser.add_argument(
        "--checkpoints", type=Path, nargs="+",
        default=[ROOT / "answer_result" / "multiseed" / "PhyCaSNet" / f"seed{seed}" /
                 f"v3de_g3_gns_seed{seed}.pt" for seed in range(5)],
    )
    parser.add_argument("--output", type=Path,
                        default=ROOT / "answer_result" / "share_ablation_simulator_sensitivity.json")
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--simulator-events", type=int, default=100)
    parser.add_argument("--simulator-audit-events", type=int, default=500,
                        help="Maximum test events screened for exact clean ARIO replay.")
    parser.add_argument("--simulator-replicates", type=int, default=10)
    parser.add_argument("--share-log-sigmas", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260928)
    parser.add_argument("--replay-tolerance", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in batch.items()}


def uniform_model_shares(batch: dict[str, Any]) -> dict[str, Any]:
    """Equalize visible incoming shares per buyer while preserving row sums."""
    altered = clone_batch(batch)
    edge_a = altered["edge_a"].numpy().astype(np.float64)
    edge_dst = altered["edge_dst"].numpy()
    uniform = edge_a.copy()
    for buyer in np.unique(edge_dst):
        mask = edge_dst == buyer
        uniform[mask] = float(edge_a[mask].sum()) / int(mask.sum())
    altered["edge_a"] = torch.from_numpy(uniform.astype(np.float32))
    altered["edge_outshare"] = (
        altered["edge_a"] * altered["P_ini"][altered["edge_dst"]]
        / altered["P_ini"][altered["edge_src"]].clamp_min(1e-6)
    )
    return altered


@torch.inference_mode()
def ensemble_predictions(
    checkpoints: list[Path], cfg: Config, net: StaticNetwork,
    batches: list[dict[str, Any]], transform: Callable[[dict[str, Any]], dict[str, Any]],
    device: torch.device, label: str,
) -> dict[int, dict[str, np.ndarray]]:
    records: dict[int, dict[str, Any]] = {
        int(batch["event_id"]): {
            "peak_true": batch["peak_loss"].numpy(),
            "trajectory_true": batch["u_keyframes"].numpy(),
            "shock": batch["shock_mask"].numpy() > 0.5,
            "peak_predictions": [],
            "trajectory_predictions": [],
        }
        for batch in batches
    }
    for seed, checkpoint in enumerate(checkpoints):
        progress(f"{label}: checkpoint {seed + 1}/{len(checkpoints)}.")
        model = load_y8(checkpoint, cfg, net.Fv, device).eval()
        for batch in batches:
            observed = transform(batch)
            output = model(to_device(observed, device))
            record = records[int(batch["event_id"])]
            record["peak_predictions"].append(output["peak"].detach().cpu().numpy())
            record["trajectory_predictions"].append(
                output["u_keyframes"].detach().cpu().numpy()
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result: dict[int, dict[str, np.ndarray]] = {}
    for event_id, raw in records.items():
        result[event_id] = {
            "peak_true": raw["peak_true"],
            "trajectory_true": raw["trajectory_true"],
            "shock": raw["shock"],
            "peak_prediction": np.mean(np.stack(raw["peak_predictions"]), axis=0),
            "trajectory_prediction": np.mean(
                np.stack(raw["trajectory_predictions"]), axis=0
            ),
        }
    return result


def r2(prediction: np.ndarray, target: np.ndarray) -> float | None:
    if target.size < 2 or float(np.std(target)) < 1e-2:
        return None
    value = 1.0 - float(np.square(prediction - target).sum()) / (
        float(np.square(target - target.mean()).sum()) + 1e-12
    )
    return value if np.isfinite(value) else None


def model_metrics(records: dict[int, dict[str, np.ndarray]]) -> dict[str, float | int | None]:
    peak_true = np.concatenate([row["peak_true"] for row in records.values()])
    peak_prediction = np.concatenate([row["peak_prediction"] for row in records.values()])
    shock = np.concatenate([row["shock"] for row in records.values()])
    cascade = (~shock) & (peak_true > CASCADE_THRESHOLD)
    trajectory_true = np.concatenate(
        [row["trajectory_true"] for row in records.values()], axis=1
    )
    trajectory_prediction = np.concatenate(
        [row["trajectory_prediction"] for row in records.values()], axis=1
    )
    frame_r2 = [
        r2(trajectory_prediction[index, cascade], trajectory_true[index, cascade])
        for index in range(trajectory_true.shape[0])
    ]
    valid_frame_r2 = [value for value in frame_r2 if value is not None]
    return {
        "event_count": len(records),
        "cascade_node_count": int(cascade.sum()),
        "peak_mae_all": float(np.mean(np.abs(peak_prediction - peak_true))),
        "peak_r2_all": r2(peak_prediction, peak_true),
        "peak_mae_cascade": float(np.mean(np.abs(peak_prediction[cascade] - peak_true[cascade]))),
        "peak_r2_cascade": r2(peak_prediction[cascade], peak_true[cascade]),
        "trajectory_r2_cascade_mean": (
            float(np.mean(valid_frame_r2)) if valid_frame_r2 else None
        ),
    }


def paired_model_bootstrap(
    clean: dict[int, dict[str, np.ndarray]], uniform: dict[int, dict[str, np.ndarray]],
    reps: int, seed: int,
) -> dict[str, list[float]]:
    event_ids = sorted(clean)
    rng = np.random.default_rng(seed)
    deltas = {"peak_mae_cascade": [], "peak_r2_cascade": []}
    for _ in range(reps):
        sampled = rng.integers(0, len(event_ids), size=len(event_ids))
        clean_sample = {index: clean[event_ids[position]] for index, position in enumerate(sampled)}
        uniform_sample = {index: uniform[event_ids[position]] for index, position in enumerate(sampled)}
        clean_metrics = model_metrics(clean_sample)
        uniform_metrics = model_metrics(uniform_sample)
        for metric in deltas:
            left, right = uniform_metrics[metric], clean_metrics[metric]
            if left is not None and right is not None:
                deltas[metric].append(float(left) - float(right))
    return {
        metric: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for metric, values in deltas.items() if values
    }


def event_metadata(data_root: Path, event_id: int) -> dict[str, Any]:
    with np.load(data_root / "events" / f"event_{event_id:06d}.npz", allow_pickle=True) as data:
        return dict(data["shock_meta"].item())


def stratified_event_ids(data_root: Path, event_ids: list[int], count: int, seed: int) -> list[int]:
    if count > len(event_ids):
        raise ValueError("simulator-events exceeds the available test-event count.")
    groups: dict[tuple[str, str], list[int]] = {}
    for event_id in event_ids:
        metadata = event_metadata(data_root, event_id)
        groups.setdefault((str(metadata["mode"]), str(metadata["tier"])), []).append(event_id)
    rng = np.random.default_rng(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[int] = []
    keys = sorted(groups)
    while len(selected) < count:
        advanced = False
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
                advanced = True
        if not advanced:
            raise RuntimeError("Unable to complete stratified event selection.")
    return sorted(selected)


def transformed_matrix(base: np.ndarray, mode: str, rng: np.random.Generator,
                       log_sigma: float = 0.0) -> np.ndarray:
    """Transform nonzero coefficients and preserve every buyer's row sum."""
    altered = np.zeros_like(base, dtype=np.float64)
    for buyer in range(base.shape[0]):
        suppliers = np.flatnonzero(base[buyer] > 0.0)
        if suppliers.size == 0:
            continue
        total = float(base[buyer, suppliers].sum())
        if mode == "uniform":
            values = np.full(suppliers.size, total / suppliers.size)
        elif mode == "lognormal":
            values = base[buyer, suppliers] * np.exp(
                rng.normal(0.0, log_sigma, size=suppliers.size)
            )
            values *= total / float(values.sum())
        elif mode == "clean":
            values = base[buyer, suppliers]
        else:
            raise ValueError(f"Unknown matrix transformation: {mode}")
        altered[buyer, suppliers] = values
    return altered


def network_with_matrix(base_network: Any, matrix: np.ndarray) -> Any:
    network = copy.deepcopy(base_network)
    network.A = matrix.copy()
    network.A_tot = np.zeros((network.N, network.K), dtype=np.float64)
    for sector, members in enumerate(network.sector_members):
        if members.size:
            network.A_tot[:, sector] = network.A[:, members].sum(axis=1)
    return network


def simulator_params(static_meta: dict[str, Any]) -> ario.Params:
    params = ario.Params()
    stored = static_meta.get("params", {})
    for name in ("tau", "n_opt_default", "n_opt_nonstockable", "dt", "T",
                 "tau_recover_default"):
        if name not in stored:
            continue
        target = "tau_recover" if name == "tau_recover_default" else name
        setattr(params, target, stored[name])
    return params


def replay_variant(
    base_network: Any, matrix: np.ndarray, params: ario.Params,
    data_root: Path, event_ids: list[int], label: str,
) -> dict[int, dict[str, np.ndarray]]:
    network = network_with_matrix(base_network, matrix)
    original_solver = ario.solve_equilibrium
    with contextlib.redirect_stdout(io.StringIO()):
        equilibrium = original_solver(network).copy()
    ario.solve_equilibrium = lambda _: equilibrium.copy()
    records: dict[int, dict[str, np.ndarray]] = {}
    try:
        for index, event_id in enumerate(event_ids, 1):
            event_path = data_root / "events" / f"event_{event_id:06d}.npz"
            with np.load(event_path, allow_pickle=True) as data:
                shock_mask = data["shock_mask"].astype(bool)
                delta0 = data["delta0"].astype(np.float64)
                metadata = dict(data["shock_meta"].item())
            damage = {
                network.nodes[node]: float(delta0[node])
                for node in np.flatnonzero(shock_mask)
            }
            scenario = ario.Scenario(name=f"{label}_{event_id}", initial_damage=damage)
            event_params = replace(
                params, T=200, tau_recover=float(metadata["recovery_days"])
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = ario.simulate(network, scenario, event_params)
            bau = np.maximum(np.asarray(result["P_BAU"], dtype=np.float64), 1e-12)
            health = np.asarray(result["P_a"], dtype=np.float64) / bau[None, :]
            loss = np.maximum(1.0 - health, 0.0)
            aggregate_loss = np.maximum(
                1.0 - np.asarray(result["P_a"]).sum(axis=1) / float(bau.sum()), 0.0
            )
            records[event_id] = {
                "peak": loss.max(axis=0).astype(np.float32),
                "trajectory": health[KEY_DAYS].astype(np.float32),
                "aggregate_loss": aggregate_loss.astype(np.float32),
                "shock": shock_mask,
            }
            if index % max(1, len(event_ids) // 10) == 0 or index == len(event_ids):
                progress(f"{label}: simulator event {index}/{len(event_ids)}.")
    finally:
        ario.solve_equilibrium = original_solver
    return records


def simulator_agreement(
    clean: dict[int, dict[str, np.ndarray]], altered: dict[int, dict[str, np.ndarray]],
) -> dict[str, float | int | None]:
    clean_peak = np.concatenate([clean[event_id]["peak"] for event_id in clean])
    altered_peak = np.concatenate([altered[event_id]["peak"] for event_id in clean])
    shock = np.concatenate([clean[event_id]["shock"] for event_id in clean])
    cascade = (~shock) & (clean_peak > CASCADE_THRESHOLD)
    trajectory_differences = [
        float(np.mean(np.abs(altered[event_id]["trajectory"] - clean[event_id]["trajectory"])))
        for event_id in clean
    ]
    aggregate_curve_differences = [
        float(np.mean(np.abs(altered[event_id]["aggregate_loss"] - clean[event_id]["aggregate_loss"])))
        for event_id in clean
    ]
    aggregate_peak_differences = [
        abs(float(altered[event_id]["aggregate_loss"].max())
            - float(clean[event_id]["aggregate_loss"].max()))
        for event_id in clean
    ]
    jaccards = []
    for event_id in clean:
        eligible = ~clean[event_id]["shock"]
        clean_set = eligible & (clean[event_id]["peak"] > CASCADE_THRESHOLD)
        altered_set = eligible & (altered[event_id]["peak"] > CASCADE_THRESHOLD)
        union = clean_set | altered_set
        jaccards.append(float(np.sum(clean_set & altered_set) / np.sum(union)) if union.any() else 1.0)
    return {
        "event_count": len(clean),
        "cascade_node_count_clean_defined": int(cascade.sum()),
        "peak_mae_all_vs_clean": float(np.mean(np.abs(altered_peak - clean_peak))),
        "peak_r2_all_vs_clean": r2(altered_peak, clean_peak),
        "peak_mae_clean_cascade_vs_clean": float(
            np.mean(np.abs(altered_peak[cascade] - clean_peak[cascade]))
        ),
        "peak_r2_clean_cascade_vs_clean": r2(altered_peak[cascade], clean_peak[cascade]),
        "trajectory_mae_vs_clean": float(np.mean(trajectory_differences)),
        "aggregate_curve_mae_vs_clean": float(np.mean(aggregate_curve_differences)),
        "aggregate_peak_absolute_delta_mean": float(np.mean(aggregate_peak_differences)),
        "cascade_set_jaccard_mean": float(np.mean(jaccards)),
    }


def percentile_ci(values: np.ndarray, reps: int, rng: np.random.Generator) -> list[float]:
    indices = rng.integers(0, values.size, size=(reps, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def simulator_event_cis(
    clean: dict[int, dict[str, np.ndarray]], replicates: list[dict[int, dict[str, np.ndarray]]],
    reps: int, seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    event_ids = sorted(clean)
    bootstrap_means: dict[str, list[float]] = {
        "event_peak_mae": [], "aggregate_curve_mae": [],
        "aggregate_peak_absolute_delta": [],
    }
    for _ in range(reps):
        sampled_events = rng.integers(0, len(event_ids), size=len(event_ids))
        altered = replicates[int(rng.integers(len(replicates)))]
        sampled_values = {name: [] for name in bootstrap_means}
        for event_position in sampled_events:
            event_id = event_ids[int(event_position)]
            sampled_values["event_peak_mae"].append(float(np.mean(np.abs(
                altered[event_id]["peak"] - clean[event_id]["peak"]
            ))))
            sampled_values["aggregate_curve_mae"].append(float(np.mean(np.abs(
                altered[event_id]["aggregate_loss"] - clean[event_id]["aggregate_loss"]
            ))))
            sampled_values["aggregate_peak_absolute_delta"].append(abs(
                float(altered[event_id]["aggregate_loss"].max())
                - float(clean[event_id]["aggregate_loss"].max())
            ))
        for name, values in sampled_values.items():
            bootstrap_means[name].append(float(np.mean(values)))
    return {
        name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for name, values in bootstrap_means.items()
    }


def replay_audit(data_root: Path, replayed: dict[int, dict[str, np.ndarray]],
                 tolerance: float) -> tuple[list[int], dict[str, Any]]:
    peak_errors, trajectory_errors = [], []
    eligible, excluded = [], []
    for event_id, row in replayed.items():
        with np.load(data_root / "events" / f"event_{event_id:06d}.npz", allow_pickle=True) as data:
            peak_error = float(np.max(np.abs(row["peak"] - data["peak_loss_node"])))
            trajectory_error = float(np.max(np.abs(row["trajectory"] - data["u_keyframes"])))
        peak_errors.append(peak_error)
        trajectory_errors.append(trajectory_error)
        if max(peak_error, trajectory_error) <= tolerance:
            eligible.append(event_id)
        else:
            excluded.append({
                "event_id": event_id,
                "maximum_peak_label_absolute_error": peak_error,
                "maximum_trajectory_label_absolute_error": trajectory_error,
            })
    audit = {
        "screened_event_count": len(replayed),
        "eligible_event_count": len(eligible),
        "excluded_event_count": len(excluded),
        "excluded_events": excluded,
        "maximum_peak_label_absolute_error": max(peak_errors),
        "maximum_trajectory_label_absolute_error": max(trajectory_errors),
        "required_tolerance": tolerance,
    }
    return eligible, audit


def main() -> None:
    args = parse_args()
    if len(args.checkpoints) != 5 or any(not path.is_file() for path in args.checkpoints):
        raise FileNotFoundError("Exactly five existing primary checkpoints are required.")
    if args.simulator_events < 1 or args.simulator_replicates < 1 or args.bootstrap_reps < 1:
        raise ValueError("Event, replicate, and bootstrap counts must be positive.")
    if any(value <= 0.0 for value in args.share_log_sigmas):
        raise ValueError("All share-log-sigmas must be positive.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = Config(data_root=str(args.data_root), seed=0, event_scalars_mode="minimal")
    cfg.deq_iters = cfg.deq_grad_iters = 1
    net = StaticNetwork(cfg)
    _, _, test_ids = split_event_ids(args.data_root / cfg.events_dir, cfg)
    test_ids = test_ids[:args.n_test]
    batches = list(DataLoader(
        CascadeEventDataset(cfg, net, test_ids, train_mode=False), batch_size=1,
        shuffle=False, collate_fn=collate_single,
    ))

    progress("Running clean five-checkpoint ensemble.")
    clean_model = ensemble_predictions(
        args.checkpoints, cfg, net, batches, clone_batch, device, "clean model input"
    )
    progress("Running uniform-share five-checkpoint ensemble.")
    uniform_model = ensemble_predictions(
        args.checkpoints, cfg, net, batches, uniform_model_shares, device,
        "uniform-share model input",
    )
    clean_model_metrics = model_metrics(clean_model)
    uniform_model_metrics = model_metrics(uniform_model)
    model_delta = {
        key: (float(uniform_model_metrics[key]) - float(clean_model_metrics[key]))
        for key in clean_model_metrics
        if isinstance(clean_model_metrics[key], float) and uniform_model_metrics[key] is not None
    }

    audit_ids = test_ids[:min(args.simulator_audit_events, len(test_ids))]
    if args.simulator_events > len(audit_ids):
        raise ValueError("simulator-events exceeds the screened test-event count.")
    with (args.data_root / "static_meta.pkl").open("rb") as stream:
        static_meta = pickle.load(stream)
    base_network = HenrietEnv._network_from_meta(str(args.data_root / "static_meta.pkl"))
    params = simulator_params(static_meta)
    clean_matrix = np.asarray(base_network.A, dtype=np.float64).copy()
    clean_replay_candidates = replay_variant(
        base_network, clean_matrix, params, args.data_root, audit_ids, "clean_audit"
    )
    eligible_ids, audit = replay_audit(
        args.data_root, clean_replay_candidates, args.replay_tolerance
    )
    if len(eligible_ids) < args.simulator_events:
        raise RuntimeError(
            f"Only {len(eligible_ids)} test events passed exact clean-label replay; "
            f"{args.simulator_events} are required."
        )
    simulator_ids = stratified_event_ids(
        args.data_root, eligible_ids, args.simulator_events, args.random_seed
    )
    clean_simulator = {event_id: clean_replay_candidates[event_id] for event_id in simulator_ids}

    simulator_scenarios: dict[str, Any] = {}
    scenario_specs = [("uniform_shares", "uniform", 0.0)] + [
        (f"share_log_sigma_{sigma:.2f}", "lognormal", sigma)
        for sigma in args.share_log_sigmas
    ]
    for scenario_index, (name, mode, sigma) in enumerate(scenario_specs):
        replicate_count = 1 if mode == "uniform" else args.simulator_replicates
        replicate_outputs = []
        replicate_metrics = []
        for replicate_index in range(replicate_count):
            rng = np.random.default_rng(
                args.random_seed + 100_000 * (scenario_index + 1) + replicate_index
            )
            matrix = transformed_matrix(clean_matrix, mode, rng, sigma)
            row_sum_error = float(np.max(np.abs(matrix.sum(axis=1) - clean_matrix.sum(axis=1))))
            if row_sum_error > 1e-10:
                raise RuntimeError(f"Buyer input-mass invariance failed: {row_sum_error}")
            output = replay_variant(
                base_network, matrix, params, args.data_root, simulator_ids,
                f"{name}_replicate_{replicate_index + 1}",
            )
            replicate_outputs.append(output)
            replicate_metrics.append(simulator_agreement(clean_simulator, output))
        numeric_keys = [
            key for key, value in replicate_metrics[0].items()
            if isinstance(value, float)
        ]
        simulator_scenarios[name] = {
            "transformation": mode,
            "log_sigma": sigma if mode == "lognormal" else None,
            "replicate_count": replicate_count,
            "replicate_metrics": replicate_metrics,
            "metric_mean_across_network_reconstructions": {
                key: float(np.mean([float(row[key]) for row in replicate_metrics]))
                for key in numeric_keys if all(row[key] is not None for row in replicate_metrics)
            },
            "event_bootstrap_95ci": simulator_event_cis(
                clean_simulator, replicate_outputs, args.bootstrap_reps,
                args.random_seed + scenario_index,
            ),
        }

    report = {
        "protocol": {
            "checkpoint_paths": [str(path.resolve()) for path in args.checkpoints],
            "test_event_ids": test_ids,
            "simulator_event_ids": simulator_ids,
            "simulator_audit_candidate_event_ids": audit_ids,
            "model_ablation_scope": "Frozen-ensemble inference ablation; original labels retained.",
            "uniform_share_definition": "Equal incoming coefficients per buyer while preserving that buyer's total incoming coefficient mass.",
            "simulator_scope": "Fixed topology, sectors, final demand, Day-0 damage, recovery time, inventory parameters, and horizon; altered A, A_tot, equilibrium, and BAU are internally recomputed. Only events whose clean ARIO replay matches the stored label within the stated tolerance are eligible.",
            "simulator_normalization": "Each altered network is normalized to its own internally consistent no-shock BAU.",
            "cascade_definition": "Day-0-unshocked nodes with clean peak loss > 0.05.",
            "random_seed": args.random_seed,
        },
        "frozen_model_uniform_share_ablation": {
            "clean": clean_model_metrics,
            "uniform_shares": uniform_model_metrics,
            "uniform_minus_clean": model_delta,
            "paired_event_bootstrap_95ci_for_uniform_minus_clean": paired_model_bootstrap(
                clean_model, uniform_model, args.bootstrap_reps, args.random_seed
            ),
            "interpretation_boundary": "This tests whether the frozen ensemble uses precise relative shares at inference; it is not a retrained uniform-share model.",
        },
        "simulator_share_sensitivity": {
            "clean_replay_audit": audit,
            "scenarios": simulator_scenarios,
            "interpretation_boundary": "Changes quantify ARIO label sensitivity to buyer-mass-preserving share reallocations, conditional on the tested network, events, and simulator specification.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    progress(f"Saved {args.output}")


if __name__ == "__main__":
    main()