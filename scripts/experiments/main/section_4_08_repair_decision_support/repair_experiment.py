"""Compare MeCaSNet-driven and simulator-driven repair decisions.

The experiment uses MeCaSNet only to rank hypothetical repair actions. Every
reported repair outcome is independently verified by the full ARIO/Henriet
simulator. The analysis requires authorized simulator and FINDER interfaces on
``PYTHONPATH``.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from fix.finder_policy import FinderAgent
    from fix.sim_env import HenrietEnv
else:
    FinderAgent = Any
    HenrietEnv = Any

import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra


def repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Could not locate the repository root")


ROOT = repository_root()
PROJECT_ROOT = ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mecasnet.config import Config  # noqa: E402
from mecasnet.data import StaticNetwork  # noqa: E402
from mecasnet.evaluation import load_compat, to_device  # noqa: E402


KEY_DAYS = (0, 5, 10, 20, 30, 50, 70, 100, 150, 199)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-simulator verification of MeCaSNet-driven repair decisions."
    )
    parser.add_argument(
        "--data-root",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--finder-checkpoint",
        default=str(ROOT / "checkpoints" / "finder_henriet.pt"),
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT / "runs" / "repair" / "surrogate_repair_comparison.json"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument(
        "--damage-levels", type=float, nargs="+", default=[0.3, 0.6, 0.7, 0.9]
    )
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--k-hubs", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-simulator-greedy",
        "--skip-oracle",
        dest="skip_simulator_greedy",
        action="store_true",
        help="Skip the full-simulator greedy reference to reduce runtime.",
    )
    parser.add_argument(
        "--skip-finder",
        action="store_true",
        help="Skip the pretrained FINDER policy even if its checkpoint exists.",
    )
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_static_meta(path: Path) -> dict[str, Any]:
    import pickle

    with path.open("rb") as handle:
        return pickle.load(handle)


def assert_network_alignment(
    static_meta: dict[str, Any], env: HenrietEnv, net: StaticNetwork
) -> None:
    firms = list(static_meta["firms"])
    if firms != list(env._net.nodes):
        raise ValueError("FINDER environment and MeCaSNet firm order do not match.")
    checks = {
        "nodes": int(static_meta["V"]) == net.V == len(env._net.nodes),
        "edges": int(static_meta["E"]) == net.E,
        "edge_src": np.array_equal(np.asarray(static_meta["edge_src"]), net.edge_src),
        "edge_dst": np.array_equal(np.asarray(static_meta["edge_dst"]), net.edge_dst),
        "P_ini": np.allclose(np.asarray(static_meta["P_ini"]), net.P_ini),
    }
    failed = [name for name, value in checks.items() if not value]
    if failed:
        raise ValueError(f"Static-network alignment failed: {failed}")


def scenario_delta0(env: HenrietEnv, selected: list[int]) -> np.ndarray:
    delta0 = np.zeros(len(env._net.nodes), dtype=np.float32)
    selected_set = set(selected)
    for node_id, damage in env._scn_base.initial_damage.items():
        index = env._net.idx[node_id]
        delta0[index] = float(damage) * (env._alpha if index in selected_set else 1.0)
    return delta0


def make_batch(
    cfg: Config,
    net: StaticNetwork,
    delta0: np.ndarray,
    nominal_damage: float,
) -> dict[str, Any]:
    shock_mask = (delta0 > 0).astype(np.uint8)
    reach_idx = net.shock_reach(shock_mask, cfg.reach_hops)
    sub_A = net.A_csr[reach_idx][:, reach_idx]
    coo = sub_A.tocoo()
    edge_dst = torch.as_tensor(coo.row, dtype=torch.long)
    edge_src = torch.as_tensor(coo.col, dtype=torch.long)
    edge_a = torch.as_tensor(coo.data, dtype=torch.float32)
    sub_P_ini = torch.as_tensor(net.P_ini[reach_idx], dtype=torch.float32)
    edge_outshare = (
        edge_a * sub_P_ini[edge_dst] / sub_P_ini[edge_src].clamp_min(1e-6)
    )

    sub_shock = shock_mask[reach_idx].astype(np.float32)
    shock_local = np.flatnonzero(sub_shock > 0.5)
    if shock_local.size:
        undirected = (sub_A + sub_A.T).tocsr()
        distances = dijkstra(
            undirected,
            indices=shock_local,
            unweighted=True,
            directed=False,
            limit=4.0,
            min_only=True,
        )
        distances = np.where(np.isfinite(distances), distances, 5).astype(np.int64)
    else:
        distances = np.full(reach_idx.size, 5, dtype=np.int64)
    shock_hop = np.eye(5, dtype=np.float32)[np.minimum(distances, 4)]

    damaged_values = delta0[shock_mask > 0]
    event_scalars = np.zeros(12, dtype=np.float32)
    event_scalars[2] = float(damaged_values.mean()) if damaged_values.size else 0.0
    # The nominal event magnitude describes the original event; firm-level
    # delta0 and its mean carry the action-specific repair update.
    event_scalars[3] = float(nominal_damage)

    return {
        "event_id": -1,
        "reach_idx": torch.as_tensor(reach_idx, dtype=torch.long),
        "Nr": int(reach_idx.size),
        "x_v": torch.as_tensor(net.x_v_static[reach_idx], dtype=torch.float32),
        "shock_mask": torch.as_tensor(sub_shock, dtype=torch.float32),
        "delta0": torch.as_tensor(delta0[reach_idx], dtype=torch.float32),
        "P_ini": sub_P_ini,
        "sectors": torch.as_tensor(net.sectors[reach_idx], dtype=torch.long),
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_a": edge_a,
        "edge_outshare": edge_outshare,
        "key_days": torch.as_tensor(KEY_DAYS, dtype=torch.long),
        "event_scalars": torch.as_tensor(event_scalars, dtype=torch.float32),
        "shock_hop_oh": torch.as_tensor(shock_hop, dtype=torch.float32),
    }


@torch.no_grad()
def surrogate_system_loss(
    model: torch.nn.Module,
    cfg: Config,
    net: StaticNetwork,
    delta0: np.ndarray,
    nominal_damage: float,
    device: torch.device,
) -> float:
    batch = make_batch(cfg, net, delta0, nominal_damage)
    output = model(to_device(batch, device))
    u_keyframes = output["u_keyframes"].detach()
    reach_idx = batch["reach_idx"].to(device)
    reach_weights = torch.as_tensor(net.P_ini, device=device)[reach_idx]
    outside_weight = float(net.P_ini.sum() - net.P_ini[batch["reach_idx"].numpy()].sum())
    production = (u_keyframes * reach_weights.unsqueeze(0)).sum(dim=1) + outside_weight
    return float((1.0 - production.min() / float(net.P_ini.sum())).cpu().item())


def simulator_loss(env: HenrietEnv, selected: list[int]) -> float:
    scenario = env._apply_attn(env._scn_base, selected)
    return env._run_peak(scenario)


def surrogate_greedy(
    env: HenrietEnv,
    model: torch.nn.Module,
    cfg: Config,
    net: StaticNetwork,
    nominal_damage: float,
    device: torch.device,
    budget: int,
) -> dict[str, Any]:
    selected: list[int] = []
    candidates = [env._net.idx[node] for node in env._scn_base.initial_damage]
    predicted_curve = [
        surrogate_system_loss(
            model, cfg, net, scenario_delta0(env, selected), nominal_damage, device
        )
    ]
    candidate_calls = 0
    synchronize(device)
    started = time.perf_counter()
    for _ in range(min(budget, len(candidates))):
        scored = []
        for candidate in candidates:
            if candidate in selected:
                continue
            trial = selected + [candidate]
            predicted = surrogate_system_loss(
                model, cfg, net, scenario_delta0(env, trial), nominal_damage, device
            )
            scored.append((predicted, candidate))
            candidate_calls += 1
        if not scored:
            break
        predicted, action = min(scored, key=lambda item: (item[0], item[1]))
        selected.append(action)
        predicted_curve.append(predicted)
    synchronize(device)
    return {
        "selected": selected,
        "decision_seconds": time.perf_counter() - started,
        "candidate_evaluations": candidate_calls,
        "predicted_loss_curve": predicted_curve,
    }


def simulator_greedy(env: HenrietEnv, budget: int) -> dict[str, Any]:
    selected: list[int] = []
    candidates = [env._net.idx[node] for node in env._scn_base.initial_damage]
    true_curve = [simulator_loss(env, selected)]
    candidate_calls = 0
    started = time.perf_counter()
    for _ in range(min(budget, len(candidates))):
        scored = []
        for candidate in candidates:
            if candidate in selected:
                continue
            trial = selected + [candidate]
            scored.append((simulator_loss(env, trial), candidate))
            candidate_calls += 1
        if not scored:
            break
        loss, action = min(scored, key=lambda item: (item[0], item[1]))
        selected.append(action)
        true_curve.append(loss)
    return {
        "selected": selected,
        "decision_seconds": time.perf_counter() - started,
        "candidate_evaluations": candidate_calls,
        "true_loss_curve_during_search": true_curve,
    }


def load_finder_agent(
    state: Any,
    checkpoint: Path,
    budget: int,
) -> FinderAgent:
    try:
        from fix.finder_policy import AUX_DIM, FinderAgent
    except ImportError as exc:
        raise RuntimeError(
            "The FINDER backend is an external dependency. "
            "Install an authorized backend before enabling FINDER."
        ) from exc
    agent = FinderAgent(
        node_in=state.node_feat.shape[1],
        budget=budget,
        hidden=64,
        reg_hidden=32,
        aux_dim=AUX_DIM,
        n_iters=3,
        device="cpu",
    )
    agent.load(str(checkpoint))
    return agent


def finder_simulator_feedback(
    env: HenrietEnv,
    checkpoint: Path,
    budget: int,
) -> dict[str, Any]:
    state = env.reset(seed=env._experiment_seed)
    agent = load_finder_agent(state, checkpoint, budget)
    rng = random.Random(0)
    selected = []
    started = time.perf_counter()
    while len(selected) < budget and len(state.candidates):
        action = agent.select(state, eps=0.0, rng=rng)
        if action < 0:
            break
        _, state, done = env.step(action)
        selected.append(int(action))
        if done:
            break
    return {
        "selected": selected,
        "decision_seconds": time.perf_counter() - started,
        "candidate_evaluations": len(selected),
        "note": "FINDER policy loop: one full simulator rerun updates the state after each action.",
    }


def finder_mecasnet_feedback(
    env: HenrietEnv,
    checkpoint: Path,
    model: torch.nn.Module,
    cfg: Config,
    net: StaticNetwork,
    nominal_damage: float,
    device: torch.device,
    budget: int,
) -> dict[str, Any]:
    state = env.reset(seed=env._experiment_seed)
    agent = load_finder_agent(state, checkpoint, budget)
    rng = random.Random(0)
    selected: list[int] = []
    predicted_curve = [
        surrogate_system_loss(
            model, cfg, net, scenario_delta0(env, selected), nominal_damage, device
        )
    ]
    state = env._build_state(peak_cur=predicted_curve[0])
    state.peak_init = predicted_curve[0]
    synchronize(device)
    started = time.perf_counter()
    while len(selected) < budget and len(state.candidates):
        action = agent.select(state, eps=0.0, rng=rng)
        if action < 0:
            break
        selected.append(int(action))
        env._selected = list(selected)
        predicted_loss = surrogate_system_loss(
            model, cfg, net, scenario_delta0(env, selected), nominal_damage, device
        )
        predicted_curve.append(predicted_loss)
        state = env._build_state(peak_cur=predicted_loss)
        state.peak_init = predicted_curve[0]
    synchronize(device)
    return {
        "selected": selected,
        "decision_seconds": time.perf_counter() - started,
        "candidate_evaluations": len(selected),
        "predicted_loss_curve": predicted_curve,
        "note": "Same pretrained FINDER policy; MeCaSNet replaces the simulator for sequential state feedback.",
    }


def ordered_policy(
    env: HenrietEnv,
    budget: int,
    score: Callable[[int], float],
) -> dict[str, Any]:
    candidates = [env._net.idx[node] for node in env._scn_base.initial_damage]
    started = time.perf_counter()
    selected = sorted(candidates, key=lambda node: (-score(node), node))[:budget]
    return {
        "selected": selected,
        "decision_seconds": time.perf_counter() - started,
        "candidate_evaluations": len(candidates),
    }


def verify_policy(
    env: HenrietEnv,
    selected: list[int],
    model: torch.nn.Module,
    cfg: Config,
    net: StaticNetwork,
    nominal_damage: float,
    device: torch.device,
) -> dict[str, Any]:
    true_curve = []
    predicted_curve = []
    started = time.perf_counter()
    for step in range(len(selected) + 1):
        prefix = selected[:step]
        true_curve.append(simulator_loss(env, prefix))
        predicted_curve.append(
            surrogate_system_loss(
                model, cfg, net, scenario_delta0(env, prefix), nominal_damage, device
            )
        )
    elapsed = time.perf_counter() - started
    true_values = np.asarray(true_curve)
    predicted_values = np.asarray(predicted_curve)
    return {
        "true_loss_curve": true_curve,
        "surrogate_loss_curve": predicted_curve,
        "verification_seconds": elapsed,
        "initial_loss": true_curve[0],
        "final_loss": true_curve[-1],
        "absolute_reduction": true_curve[0] - true_curve[-1],
        "relative_reduction": (true_curve[0] - true_curve[-1]) / max(true_curve[0], 1e-12),
        "repair_state_mae": float(np.abs(true_values - predicted_values).mean()),
        "repair_state_max_abs_error": float(np.abs(true_values - predicted_values).max()),
    }


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    policies = sorted({name for event in events for name in event["policies"]})
    summary: dict[str, Any] = {}
    for policy in policies:
        rows = [event["policies"][policy] for event in events if policy in event["policies"]]
        keys = (
            "decision_seconds",
            "verification_seconds",
            "initial_loss",
            "final_loss",
            "absolute_reduction",
            "relative_reduction",
            "repair_state_mae",
            "repair_state_max_abs_error",
            "final_loss_delta_vs_simulator_greedy",
            "topk_overlap_with_simulator_greedy",
        )
        values: dict[str, Any] = {"n_events": len(rows)}
        for key in keys:
            data = [float(row[key]) for row in rows if key in row and row[key] is not None]
            if data:
                values[key] = {
                    "mean": float(np.mean(data)),
                    "median": float(np.median(data)),
                    "p95": float(np.percentile(data, 95)),
                    "std": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
                }
        summary[policy] = values
    return summary


def bootstrap_mean_ci(
    values: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> list[float] | None:
    if values.size == 0:
        return None
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_comparison(
    events: list[dict[str, Any]],
    policy: str,
    reference: str,
    resamples: int,
) -> dict[str, Any] | None:
    pairs = [
        (event["policies"][policy], event["policies"][reference])
        for event in events
        if policy in event["policies"] and reference in event["policies"]
    ]
    if not pairs:
        return None
    final_delta = np.asarray(
        [candidate["final_loss"] - baseline["final_loss"] for candidate, baseline in pairs]
    )
    reduction_delta = np.asarray(
        [
            candidate["absolute_reduction"] - baseline["absolute_reduction"]
            for candidate, baseline in pairs
        ]
    )
    decision_speedup = np.asarray(
        [
            baseline["decision_seconds"] / max(candidate["decision_seconds"], 1e-12)
            for candidate, baseline in pairs
        ]
    )
    overlap = np.asarray([
        len(set(candidate["selected"]) & set(baseline["selected"]))
        / max(len(set(baseline["selected"])), 1)
        for candidate, baseline in pairs
    ])
    rng = np.random.default_rng(20260808)
    return {
        "policy": policy,
        "reference": reference,
        "n_paired_events": len(pairs),
        "final_loss_delta": {
            "definition": "policy final full-simulator loss minus reference final full-simulator loss",
            "mean": float(final_delta.mean()),
            "median": float(np.median(final_delta)),
            "paired_bootstrap_95ci_for_mean": bootstrap_mean_ci(
                final_delta, resamples, rng
            ),
        },
        "absolute_reduction_delta": {
            "definition": "policy reduction minus reference reduction; positive favors policy",
            "mean": float(reduction_delta.mean()),
            "median": float(np.median(reduction_delta)),
            "paired_bootstrap_95ci_for_mean": bootstrap_mean_ci(
                reduction_delta, resamples, rng
            ),
        },
        "decision_time_speedup": {
            "definition": "reference decision time divided by policy decision time",
            "median": float(np.median(decision_speedup)),
            "p05": float(np.percentile(decision_speedup, 5)),
            "p95": float(np.percentile(decision_speedup, 95)),
        },
        "topk_action_overlap": {
            "mean": float(overlap.mean()),
            "median": float(np.median(overlap)),
        },
    }


def build_comparisons(events: list[dict[str, Any]], resamples: int) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for policy, reference in (
        ("mecasnet_greedy", "simulator_greedy"),
        ("finder_mecasnet_feedback", "finder_simulator_feedback"),
        ("finder_mecasnet_feedback", "simulator_greedy"),
        ("finder_simulator_feedback", "simulator_greedy"),
        ("mecasnet_greedy", "finder_simulator_feedback"),
    ):
        result = paired_comparison(events, policy, reference, resamples)
        if result is not None:
            comparisons[f"{policy}_vs_{reference}"] = result
    return comparisons


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        from fix.sim_env import HenrietEnv
    except ImportError as exc:
        raise RuntimeError(
            "The ARIO/Henriet simulator is an external dependency. "
            "Install an authorized backend before running this analysis."
        ) from exc
    if not 0 < args.alpha <= 1:
        raise ValueError("--alpha must be in (0, 1].")
    if args.budget < 1 or args.k_hubs < args.budget:
        raise ValueError("Require 1 <= budget <= k_hubs.")
    data_root = Path(args.data_root)
    checkpoint = Path(args.checkpoint)
    finder_checkpoint = Path(args.finder_checkpoint)
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
    model = load_compat(checkpoint, cfg, net.Fv, device).eval()
    static_meta = load_static_meta(static_path)

    report: dict[str, Any] = {
        "protocol": {
            "scope": "Henriet network; sequential repair with full-simulator verification",
            "data_root": str(data_root),
            "mecasnet_checkpoint": str(checkpoint),
            "finder_checkpoint": str(finder_checkpoint),
            "device": str(device),
            "seeds": args.seeds,
            "damage_levels": args.damage_levels,
            "budget": args.budget,
            "alpha": args.alpha,
            "k_hubs": args.k_hubs,
            "horizon": args.horizon,
            "paired_bootstrap_resamples": args.bootstrap,
            "surrogate_objective": "BAU-output-weighted minimum aggregate production over predicted keyframes",
            "final_outcome_source": "full ARIO/Henriet simulator for every policy and repair prefix",
            "simulator_greedy_reference": not args.skip_simulator_greedy,
            "finder": not args.skip_finder and finder_checkpoint.exists(),
            "important_limitations": [
                "Surrogate decisions use ten predicted keyframes; the simulator may attain its minimum between keyframes.",
                "Repair creates heterogeneous damage patterns that may differ from the training distribution.",
                "The event-level nominal damage scalar remains the original event magnitude; firm-level delta0 and mean damage reflect repair.",
                "This experiment evaluates Henriet only and does not establish an Inoue repair-loop result.",
                "The FINDER checkpoint was trained with simulator feedback; replacing that feedback at inference tests integration without retraining FINDER.",
                "Uniform-damage events make the initial-damage ranking baseline degenerate to deterministic tie breaking.",
            ],
        },
        "events": [],
    }

    alignment_checked = False
    for damage in args.damage_levels:
        for seed in args.seeds:
            event_label = f"damage={damage:.3f}, seed={seed}"
            progress(f"Starting {event_label}")
            env = HenrietEnv(
                budget=args.budget,
                alpha=args.alpha,
                default_damage=damage,
                k_hubs=args.k_hubs,
                horizon=args.horizon,
                meta_pkl=str(static_path),
            )
            if not alignment_checked:
                assert_network_alignment(static_meta, env, net)
                alignment_checked = True
            env._experiment_seed = int(seed)
            state = env.reset(seed=int(seed))
            out_degree = np.bincount(state.edge_src, minlength=state.n_nodes)
            rng = np.random.default_rng(seed + 100000)
            random_scores = {int(node): float(rng.random()) for node in state.candidates}

            raw_policies: dict[str, dict[str, Any]] = {}
            raw_policies["mecasnet_greedy"] = surrogate_greedy(
                env, model, cfg, net, damage, device, args.budget
            )
            if not args.skip_simulator_greedy:
                progress(f"{event_label}: running expensive simulator-greedy reference")
                raw_policies["simulator_greedy"] = simulator_greedy(env, args.budget)
            raw_policies["out_degree"] = ordered_policy(
                env, args.budget, lambda node: float(out_degree[node])
            )
            raw_policies["initial_damage"] = ordered_policy(
                env, args.budget, lambda node: float(state.node_feat[node, 0])
            )
            raw_policies["random"] = ordered_policy(
                env, args.budget, lambda node: random_scores[node]
            )
            if not args.skip_finder and finder_checkpoint.exists():
                raw_policies["finder_simulator_feedback"] = finder_simulator_feedback(
                    env, finder_checkpoint, args.budget
                )
                raw_policies["finder_mecasnet_feedback"] = finder_mecasnet_feedback(
                    env,
                    finder_checkpoint,
                    model,
                    cfg,
                    net,
                    damage,
                    device,
                    args.budget,
                )

            simulator_greedy_selected = (
                set(raw_policies["simulator_greedy"]["selected"])
                if "simulator_greedy" in raw_policies
                else None
            )
            simulator_greedy_final = None
            event_policies: dict[str, Any] = {}
            # Verify the simulator-greedy reference first for paired deltas.
            policy_order = sorted(
                raw_policies,
                key=lambda name: (name != "simulator_greedy", name),
            )
            for name in policy_order:
                progress(f"{event_label}: full-simulator verification of {name}")
                verified = verify_policy(
                    env,
                    raw_policies[name]["selected"],
                    model,
                    cfg,
                    net,
                    damage,
                    device,
                )
                row = copy.deepcopy(raw_policies[name])
                row.update(verified)
                if name == "simulator_greedy":
                    simulator_greedy_final = verified["final_loss"]
                if simulator_greedy_selected is not None:
                    selected_set = set(row["selected"])
                    row["topk_overlap_with_simulator_greedy"] = len(
                        selected_set & simulator_greedy_selected
                    ) / max(len(simulator_greedy_selected), 1)
                event_policies[name] = row
            if simulator_greedy_final is not None:
                for row in event_policies.values():
                    row["final_loss_delta_vs_simulator_greedy"] = (
                        row["final_loss"] - simulator_greedy_final
                    )

            event = {
                "seed": int(seed),
                "damage": float(damage),
                "shock_nodes": [int(node) for node in state.candidates],
                "policies": event_policies,
            }
            report["events"].append(event)
            report["summary"] = summarize(report["events"])
            report["paired_comparisons"] = build_comparisons(
                report["events"], args.bootstrap
            )
            save_report(output, report)
            progress(f"Completed {event_label}; checkpointed {output}")

    report["summary"] = summarize(report["events"])
    report["paired_comparisons"] = build_comparisons(report["events"], args.bootstrap)
    save_report(output, report)
    progress(f"Saved final report: {output}")


if __name__ == "__main__":
    main()
