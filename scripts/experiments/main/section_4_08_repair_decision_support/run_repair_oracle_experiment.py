"""Exact small-set repair benchmark for the MeCaSNet screening analysis.

The repair environment applies every intervention at Day 0, so the final
simulator state depends on the repaired set rather than its action order. For
five directly shocked firms and a budget of three repairs, all C(5, 3) = 10
sets can therefore be evaluated exactly. The ARIO minimum is the true oracle;
the MeCaSNet minimum is only a surrogate-objective oracle.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "flagship_y8h_download"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

from fix.sim_env import HenrietEnv
from pi_dl.config import Config
from pi_dl.data import StaticNetwork
from run_surrogate_repair_experiment import (
    assert_network_alignment,
    load_static_meta,
    scenario_delta0,
    simulator_greedy,
    simulator_loss,
    surrogate_greedy,
    surrogate_system_loss,
)
from y8_reviewer_experiments import load_y8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact ARIO oracle for small Day-0 repair sets."
    )
    parser.add_argument(
        "--data-root", default=str(ARCHIVE / "data" / "cascade_henriet_5000")
    )
    parser.add_argument(
        "--checkpoint", default=str(ARCHIVE / "artifacts" / "v3de_g3_gns_seed0.pt")
    )
    parser.add_argument(
        "--output",
        default=str(ARCHIVE / "reviewer_experiments_local" / "repair_oracle_smallset.json"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--damage-levels", type=float, nargs="+", default=[0.3, 0.6, 0.7])
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def bootstrap_mean_ci(values: np.ndarray, resamples: int, seed: int) -> list[float] | None:
    if values.size == 0:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize_rows(rows: list[dict[str, Any]], resamples: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for policy in sorted({row["policy"] for row in rows}):
        policy_rows = [row for row in rows if row["policy"] == policy]
        values: dict[str, Any] = {"n_events": len(policy_rows)}
        for metric in (
            "final_loss",
            "absolute_reduction",
            "relative_reduction",
            "optimality_gap",
            "oracle_efficiency",
            "decision_seconds",
        ):
            data = np.asarray(
                [row[metric] for row in policy_rows if row.get(metric) is not None],
                dtype=float,
            )
            if data.size:
                values[metric] = {
                    "mean": float(data.mean()),
                    "median": float(np.median(data)),
                    "std": float(data.std(ddof=1)) if data.size > 1 else 0.0,
                    "bootstrap_95ci_for_mean": bootstrap_mean_ci(
                        data, resamples, 20260928 + len(metric)
                    ),
                }
        summary[policy] = values
    return summary


def rank_selection(
    candidates: np.ndarray,
    budget: int,
    score: Callable[[int], float],
) -> tuple[int, ...]:
    ranked = sorted((int(node) for node in candidates), key=lambda node: (-score(node), node))
    return tuple(sorted(ranked[:budget]))


def policy_row(
    policy: str,
    selected: tuple[int, ...] | None,
    final_loss: float,
    initial_loss: float,
    oracle_loss: float,
    decision_seconds: float,
) -> dict[str, Any]:
    achievable_reduction = initial_loss - oracle_loss
    reduction = initial_loss - final_loss
    efficiency = None
    if achievable_reduction > 1e-12:
        efficiency = reduction / achievable_reduction
    return {
        "policy": policy,
        "selected": list(selected) if selected is not None else None,
        "final_loss": float(final_loss),
        "absolute_reduction": float(reduction),
        "relative_reduction": float(reduction / max(initial_loss, 1e-12)),
        "optimality_gap": float(final_loss - oracle_loss),
        "oracle_efficiency": None if efficiency is None else float(efficiency),
        "decision_seconds": float(decision_seconds),
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.budget <= args.candidates:
        raise ValueError("Require 1 <= --budget <= --candidates.")
    if not 0 < args.alpha <= 1:
        raise ValueError("--alpha must be in (0, 1].")

    data_root = Path(args.data_root)
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    static_path = data_root / "static_meta.pkl"
    for required in (static_path, checkpoint):
        if not required.exists():
            raise FileNotFoundError(required)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = Config(data_root=str(data_root))
    cfg.deq_iters = 1
    cfg.deq_grad_iters = 1
    cfg.event_scalars_mode = "minimal"
    net = StaticNetwork(cfg)
    model = load_y8(checkpoint, cfg, net.Fv, device).eval()
    static_meta = load_static_meta(static_path)

    combinations_per_event = int(
        len(list(itertools.combinations(range(args.candidates), args.budget)))
    )
    report: dict[str, Any] = {
        "protocol": {
            "scope": "Henriet small-set Day-0 repair; full-simulator exact oracle",
            "data_root": str(data_root),
            "phycasnet_checkpoint": str(checkpoint),
            "seeds": args.seeds,
            "damage_levels": args.damage_levels,
            "candidates": args.candidates,
            "budget": args.budget,
            "combinations_per_event": combinations_per_event,
            "alpha": args.alpha,
            "horizon": args.horizon,
            "surrogate_objective": "BAU-output-weighted minimum aggregate production over predicted keyframes",
            "true_oracle": "minimum full ARIO/Henriet loss among all repair sets",
            "order_invariance": "All repairs are applied at Day 0; the final state depends on the repaired set, not action order.",
            "limitations": [
                "Repair is instantaneous at Day 0 and is represented only by scaling initial damage.",
                "The experiment does not model repair duration, scheduling, crews, costs, precedence, or time-varying resources.",
                "Candidates are restricted to directly shocked firms in the sampled small shock set.",
                "The result is limited to the Henriet domain and this small-set protocol.",
            ],
        },
        "events": [],
    }

    alignment_checked = False
    for damage in args.damage_levels:
        for seed in args.seeds:
            label = f"damage={damage:.3f}, seed={seed}"
            progress(f"Starting {label}")
            env = HenrietEnv(
                budget=args.budget,
                alpha=args.alpha,
                default_damage=damage,
                k_hubs=args.candidates,
                horizon=args.horizon,
                meta_pkl=str(static_path),
            )
            state = env.reset(seed=seed)
            if not alignment_checked:
                assert_network_alignment(static_meta, env, net)
                alignment_checked = True
            candidates = state.candidates
            if len(candidates) != args.candidates:
                raise RuntimeError(f"Expected {args.candidates} candidates, got {len(candidates)}.")

            initial_loss = simulator_loss(env, [])
            combinations = [
                tuple(sorted(int(node) for node in combo))
                for combo in itertools.combinations(candidates, args.budget)
            ]
            enumerated: dict[tuple[int, ...], dict[str, float]] = {}
            surrogate_enumeration_seconds = 0.0
            simulator_enumeration_seconds = 0.0
            for selected in combinations:
                delta0 = scenario_delta0(env, list(selected))
                started = time.perf_counter()
                predicted_loss = surrogate_system_loss(
                    model, cfg, net, delta0, damage, device
                )
                surrogate_enumeration_seconds += time.perf_counter() - started
                started = time.perf_counter()
                true_loss = simulator_loss(env, list(selected))
                simulator_enumeration_seconds += time.perf_counter() - started
                enumerated[selected] = {
                    "surrogate_loss": float(predicted_loss),
                    "simulator_loss": float(true_loss),
                }

            surrogate_oracle = min(
                combinations, key=lambda selected: (enumerated[selected]["surrogate_loss"], selected)
            )
            simulator_oracle = min(
                combinations, key=lambda selected: (enumerated[selected]["simulator_loss"], selected)
            )
            out_degree = np.bincount(state.edge_src, minlength=state.n_nodes)
            initial_damage = state.node_feat[:, 0]
            bau_capacity = env._net.P_ini

            surrogate_greedy_result = surrogate_greedy(
                env, model, cfg, net, damage, device, args.budget
            )
            simulator_greedy_result = simulator_greedy(env, args.budget)
            selections = {
                "surrogate_greedy": tuple(sorted(surrogate_greedy_result["selected"])),
                "simulator_greedy": tuple(sorted(simulator_greedy_result["selected"])),
                "surrogate_exhaustive": surrogate_oracle,
                "simulator_oracle": simulator_oracle,
                "out_degree": rank_selection(candidates, args.budget, lambda node: float(out_degree[node])),
                "initial_damage": rank_selection(candidates, args.budget, lambda node: float(initial_damage[node])),
                "bau_capacity": rank_selection(candidates, args.budget, lambda node: float(bau_capacity[node])),
            }

            rows = []
            oracle_loss = enumerated[simulator_oracle]["simulator_loss"]
            decision_times = {
                "surrogate_greedy": surrogate_greedy_result["decision_seconds"],
                "simulator_greedy": simulator_greedy_result["decision_seconds"],
                "surrogate_exhaustive": surrogate_enumeration_seconds,
                "simulator_oracle": simulator_enumeration_seconds,
                "out_degree": 0.0,
                "initial_damage": 0.0,
                "bau_capacity": 0.0,
            }
            for policy, selected in selections.items():
                rows.append(
                    policy_row(
                        policy,
                        selected,
                        enumerated[selected]["simulator_loss"],
                        initial_loss,
                        oracle_loss,
                        decision_times[policy],
                    )
                )
            random_losses = np.asarray(
                [enumerated[selected]["simulator_loss"] for selected in combinations], dtype=float
            )
            rows.append(
                policy_row(
                    "random_uniform_set_expectation",
                    None,
                    float(random_losses.mean()),
                    initial_loss,
                    oracle_loss,
                    0.0,
                )
            )
            report["events"].append(
                {
                    "seed": seed,
                    "damage": damage,
                    "candidate_nodes": [int(node) for node in candidates],
                    "initial_loss": float(initial_loss),
                    "simulator_oracle_set": list(simulator_oracle),
                    "surrogate_oracle_set": list(surrogate_oracle),
                    "surrogate_enumeration_seconds": surrogate_enumeration_seconds,
                    "simulator_enumeration_seconds": simulator_enumeration_seconds,
                    "sets": [
                        {
                            "selected": list(selected),
                            **enumerated[selected],
                        }
                        for selected in combinations
                    ],
                    "policies": rows,
                }
            )
            report["summary"] = summarize_rows(
                [row for event in report["events"] for row in event["policies"]], args.bootstrap
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            progress(f"Completed {label}; checkpointed {output}")

    report["summary"] = summarize_rows(
        [row for event in report["events"] for row in event["policies"]], args.bootstrap
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    progress(f"Saved final report: {output}")


if __name__ == "__main__":
    main()