"""Matched warm-service timings and measured multi-round repair workflows."""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

import benchmark_henriet_batched_breakeven as baseline


ROOT = baseline.ROOT
WORKER = None


def make_trial_batch(cfg, network, delta, nominal_damage):
    import torch
    from scipy.sparse.csgraph import dijkstra

    shocked = (delta > 0).astype(np.uint8)
    indices = network.shock_reach(shocked, cfg.reach_hops)
    matrix = network.A_csr[indices][:, indices]
    edges = matrix.tocoo()
    destination = torch.as_tensor(edges.row, dtype=torch.long)
    source = torch.as_tensor(edges.col, dtype=torch.long)
    shares = torch.as_tensor(edges.data, dtype=torch.float32)
    production = torch.as_tensor(network.P_ini[indices], dtype=torch.float32)
    local_shocked = shocked[indices].astype(np.float32)
    shock_indices = np.flatnonzero(local_shocked > 0.5)
    if shock_indices.size:
        distances = dijkstra((matrix + matrix.T).tocsr(), indices=shock_indices,
                             unweighted=True, directed=False, limit=4.0, min_only=True)
        distances = np.where(np.isfinite(distances), distances, 5).astype(np.int64)
    else:
        distances = np.full(indices.size, 5, dtype=np.int64)
    scalars = np.zeros(12, dtype=np.float32)
    damaged = delta[shocked > 0]
    scalars[2] = float(damaged.mean()) if damaged.size else 0.0
    scalars[3] = float(nominal_damage)
    return {
        "event_id": -1, "reach_idx": torch.as_tensor(indices, dtype=torch.long), "Nr": int(indices.size),
        "x_v": torch.as_tensor(network.x_v_static[indices], dtype=torch.float32),
        "shock_mask": torch.as_tensor(local_shocked, dtype=torch.float32),
        "delta0": torch.as_tensor(delta[indices], dtype=torch.float32), "P_ini": production,
        "sectors": torch.as_tensor(network.sectors[indices], dtype=torch.long),
        "edge_src": source, "edge_dst": destination, "edge_a": shares,
        "edge_outshare": shares * production[destination] / production[source].clamp_min(1e-6),
        "key_days": torch.as_tensor(baseline.KEY_DAYS, dtype=torch.long),
        "event_scalars": torch.as_tensor(scalars, dtype=torch.float32),
        "shock_hop_oh": torch.as_tensor(np.eye(5, dtype=np.float32)[np.minimum(distances, 4)], dtype=torch.float32),
    }


def load_event(data_root, event_id):
    with np.load(Path(data_root) / "events" / f"event_{event_id:06d}.npz", allow_pickle=True) as payload:
        metadata = dict(payload["shock_meta"].item())
        delta = payload["delta0"].astype(np.float64)
        shocked = payload["shock_mask"].astype(bool)
    return {
        "delta0": np.where(shocked, delta, 0.0),
        "recovery_days": float(metadata["recovery_days"]),
        "nominal_damage": float(metadata.get("delta", 0.5)),
    }


class MatchedWorker(baseline.ReplayWorker):
    def evaluate(self, spec, selected=(), alpha=1.0, audit=False):
        network = self.environment._net
        network.P_ini = self.base_p_ini.copy()
        network.n_opt_sector = self.base_n_opt_sector.copy()
        delta = np.asarray(spec["delta0"], dtype=np.float64).copy()
        delta[list(selected)] *= alpha
        damage = {network.nodes[index]: float(delta[index]) for index in np.flatnonzero(delta > 0)}
        scenario = baseline.Scenario(name="matched_benchmark", initial_damage=damage)
        params = replace(self.environment._params, T=self.horizon, tau_recover=spec["recovery_days"])
        with baseline.silence_stdout():
            result = baseline.simulate(network, scenario, params)
        self.calls += 1
        production = np.asarray(result["P_a"], dtype=np.float64)
        bau = np.asarray(result["P_BAU"], dtype=np.float64)
        row = {"loss": float(1.0 - (production.sum(axis=1) / bau.sum()).min()), "calls": 1}
        if audit:
            health = np.clip(production / np.maximum(bau[None, :], 1e-12), 0.0, 1.5)
            peak = np.maximum(1.0 - health, 0.0).max(axis=0)
            row.update(trajectory_hash=baseline.stable_hash(health), peak_hash=baseline.stable_hash(peak),
                       keyframe_hash=baseline.stable_hash(health[baseline.KEY_DAYS]),
                       keyframes=health[baseline.KEY_DAYS], peak=peak)
        return row


def worker_init(data_root, event_specs, repair_specs, warmups, ready):
    global WORKER
    WORKER = MatchedWorker(data_root, 200)
    WORKER.event_specs = event_specs
    WORKER.repair_specs = repair_specs
    for spec in list(event_specs.values())[:warmups]:
        WORKER.evaluate(spec)
    ready.put({"pid": os.getpid(), "warmup_simulator_calls": WORKER.calls})


def worker_job(task):
    if task["kind"] == "event":
        spec = (load_event(WORKER.data_root, task["event_id"]) if task["mode"] == "workflow"
                else WORKER.event_specs[task["event_id"]])
    else:
        spec = WORKER.repair_specs[task["case_id"]]
    return WORKER.evaluate(spec, task.get("selected", ()), task.get("alpha", 1.0), task.get("audit", False))


class Predictor:
    def __init__(self, data_root, checkpoint, device):
        import torch
        from pi_dl.config import Config
        from pi_dl.data import StaticNetwork
        from y8_reviewer_experiments import load_y8

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; CPU fallback would invalidate this GPU benchmark.")
        self.torch = torch
        self.device = torch.device(device)
        self.cfg = Config(data_root=str(data_root), seed=0, event_scalars_mode="minimal")
        self.cfg.deq_iters = self.cfg.deq_grad_iters = 1
        self.network = StaticNetwork(self.cfg)
        self.model = load_y8(checkpoint, self.cfg, self.network.Fv, self.device).eval()
        self.total_bau = float(self.network.P_ini.sum())

    def prepare(self, spec, selected=(), alpha=1.0):
        from y8_reviewer_experiments import to_device

        delta = np.asarray(spec["delta0"], dtype=np.float32).copy()
        delta[list(selected)] *= alpha
        batch = make_trial_batch(self.cfg, self.network, delta, spec["nominal_damage"])
        outside = self.total_bau - float(self.network.P_ini[batch["reach_idx"].numpy()].sum())
        return to_device(batch, self.device), outside

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def score(self, prepared):
        batch, outside = prepared
        with self.torch.inference_mode():
            health = self.model(batch)["u_keyframes"]
            production = (health * batch["P_ini"].unsqueeze(0)).sum(dim=1) + outside
            return float((1.0 - production.min() / self.total_bau).cpu().item())


def greedy(candidates, budget, score_many):
    selected = []
    initial = score_many([[]])
    if len(initial) != 1:
        raise ValueError("Baseline scoring must return one result.")
    curve = [float(initial[0])]
    calls = 1
    rounds = []
    for step in range(min(budget, len(candidates))):
        remaining = [candidate for candidate in candidates if candidate not in selected]
        trials = [selected + [candidate] for candidate in remaining]
        started = time.perf_counter()
        values = score_many(trials)
        if len(values) != len(trials) or not np.isfinite(values).all():
            raise ValueError("Missing or non-finite candidate scores.")
        loss, action = min(zip(values, remaining), key=lambda item: (item[0], item[1]))
        selected.append(int(action))
        curve.append(float(loss))
        calls += len(trials)
        rounds.append({"round": step + 1, "candidate_evaluations": len(trials),
                       "seconds": time.perf_counter() - started, "selected": list(selected)})
    if not np.isfinite(curve).all():
        raise ValueError("Non-finite greedy loss curve.")
    return {"selected": selected, "loss_curve": curve, "scoring_evaluations_including_baseline": calls,
            "candidate_evaluations": calls - 1, "rounds": rounds}


def repair_run(pool, predictor, case_id, spec, candidates, budget, alpha, policy):
    simulator_calls = 0

    def simulate_many(trials):
        nonlocal simulator_calls
        tasks = [{"kind": "repair", "case_id": case_id, "selected": trial, "alpha": alpha} for trial in trials]
        rows = pool.map(worker_job, tasks, chunksize=1)
        simulator_calls += sum(row["calls"] for row in rows)
        return [row["loss"] for row in rows]

    def predict_many(trials):
        return [predictor.score(predictor.prepare(spec, trial, alpha)) for trial in trials]

    predictor.synchronize()
    started = time.perf_counter()
    result = greedy(candidates, budget, simulate_many if policy == "cpu_greedy" else predict_many)
    predictor.synchronize()
    search_seconds = time.perf_counter() - started
    if policy == "surrogate_verified":
        verification_started = time.perf_counter()
        true_curve = simulate_many([result["selected"][:length] for length in range(len(result["selected"]) + 1)])
        verification_seconds = time.perf_counter() - verification_started
    else:
        true_curve = list(result["loss_curve"])
        verification_seconds = 0.0
    result.update(policy=policy, search_seconds=search_seconds, verification_seconds=verification_seconds,
                  workflow_seconds=time.perf_counter() - started, actual_simulator_calls=simulator_calls,
                  surrogate_evaluations=result["scoring_evaluations_including_baseline"] if policy == "surrogate_verified" else 0,
                  simulator_verified_loss_curve=true_curve)
    expected = sum(len(candidates) - step for step in range(budget))
    assert result["candidate_evaluations"] == expected
    assert simulator_calls == (expected + 1 if policy == "cpu_greedy" else budget + 1)
    return result


def timing_summary(values):
    return baseline.summarize_times([float(value) for value in values])


def count_break_even(frontload, savings):
    return max(0, int(math.ceil(frontload / savings))) if savings > 0 else None


def summarize(timings, repairs, event_count, generation_seconds, training_seconds, setup):
    runtime = {}
    for mode in ("preloaded", "workflow"):
        backends = {}
        for backend in ("cpu", "gpu"):
            rows = [row for row in timings if row["mode"] == mode and row["backend"] == backend]
            wall = timing_summary([row["seconds"] for row in rows])
            backends[backend] = {"wall_seconds": wall, "seconds_per_event": wall["median_seconds"] / event_count,
                                 "events_per_second": event_count / wall["median_seconds"]}
        savings = backends["cpu"]["seconds_per_event"] - backends["gpu"]["seconds_per_event"]
        backends.update(speedup_gpu_over_cpu=backends["gpu"]["events_per_second"] / backends["cpu"]["events_per_second"],
                        break_even_queries=count_break_even(generation_seconds + training_seconds, savings),
                        break_even_queries_including_measured_service_setup=count_break_even(
                            generation_seconds + training_seconds + setup["gpu"] - setup["cpu_pool"], savings))
        runtime[mode] = backends
    cases = sorted({row["case_id"] for row in repairs})
    paired = []
    for case_id in cases:
        cpu = [row for row in repairs if row["case_id"] == case_id and row["policy"] == "cpu_greedy"]
        gpu = [row for row in repairs if row["case_id"] == case_id and row["policy"] == "surrogate_verified"]
        cpu_seconds = float(np.median([row["workflow_seconds"] for row in cpu]))
        gpu_seconds = float(np.median([row["workflow_seconds"] for row in gpu]))
        paired.append({"case_id": case_id, "cpu_median_seconds": cpu_seconds, "surrogate_verified_median_seconds": gpu_seconds,
                       "savings_seconds": cpu_seconds - gpu_seconds,
                       "cpu_selected": cpu[0]["selected"], "surrogate_selected": gpu[0]["selected"],
                       "cpu_final_true_loss": cpu[0]["simulator_verified_loss_curve"][-1],
                       "surrogate_final_true_loss": gpu[0]["simulator_verified_loss_curve"][-1]})
    repair = {"cases": paired, "description": "Each policy timed directly; per-case median across repetitions, then equal-case mean. No large-batch throughput extrapolation."}
    if paired:
        savings = float(np.mean([row["savings_seconds"] for row in paired]))
        repair.update(mean_savings_seconds=savings,
                      break_even_repair_events=count_break_even(generation_seconds + training_seconds, savings),
                      break_even_repair_events_including_gpu_service_setup=count_break_even(
                          generation_seconds + training_seconds + setup["gpu"], savings))
    return {"scenario": runtime, "repair": repair}


def self_test():
    import torch
    from scipy import sparse

    feature_predictor = Predictor.__new__(Predictor)
    feature_predictor.cfg = SimpleNamespace(reach_hops=5)
    feature_predictor.device = torch.device("cpu")
    feature_predictor.network = SimpleNamespace(
        shock_reach=lambda mask, hops: np.array([0, 1], dtype=np.int64),
        A_csr=sparse.csr_matrix([[0.0, 0.0], [0.2, 0.0]], dtype=np.float32),
        P_ini=np.array([10.0, 20.0], dtype=np.float32),
        x_v_static=np.ones((2, 3), dtype=np.float32), sectors=np.array([0, 1]))
    feature_predictor.total_bau = 30.0
    batch, outside = feature_predictor.prepare({"delta0": [0.6, 0.0], "nominal_damage": 0.6}, [0], 0.1)
    assert outside == 0.0 and batch["Nr"] == 2
    assert torch.allclose(batch["delta0"], torch.tensor([0.06, 0.0]))
    assert torch.allclose(batch["edge_outshare"], torch.tensor([0.4]))
    assert torch.allclose(batch["event_scalars"][[2, 3]], torch.tensor([0.06, 0.6]))
    assert batch["shock_hop_oh"].argmax(dim=1).tolist() == [0, 1]
    calls = []

    def fake_scores(trials):
        calls.append([list(trial) for trial in trials])
        return [20.0 - sum(node + 1 for node in trial) for trial in trials]

    result = greedy(list(range(20)), 10, fake_scores)
    assert result["selected"] == list(range(19, 9, -1))
    assert result["scoring_evaluations_including_baseline"] == 156
    assert [len(batch) for batch in calls] == [1] + list(range(20, 10, -1))
    assert all(all(trial[:-1] == result["selected"][:step] for trial in batch)
               for step, batch in enumerate(calls[1:]))
    tied = greedy([5, 2, 7], 2, lambda trials: [0.0] * len(trials))
    assert tied["selected"] == [2, 5]
    try:
        greedy([1], 1, lambda trials: [float("nan")] * len(trials))
    except ValueError:
        pass
    else:
        raise AssertionError("Non-finite scores accepted")
    assert count_break_even(100, 2) == 50 and count_break_even(100, 0) is None
    assert count_break_even(100, -1) is None and count_break_even(-1, 1) == 0
    simulated = []

    class FakePool:
        def map(self, function, tasks, chunksize):
            simulated.extend(tasks)
            return [{"loss": 1.0 - len(task["selected"]) / 10, "calls": 1} for task in tasks]

    fake_predictor = SimpleNamespace(synchronize=lambda: None, prepare=lambda spec, selected, alpha: selected,
                                    score=lambda selected: 1.0 - len(selected) / 10)
    cpu = repair_run(FakePool(), fake_predictor, "fake", {}, list(range(4)), 2, 0.1, "cpu_greedy")
    gpu = repair_run(FakePool(), fake_predictor, "fake", {}, list(range(4)), 2, 0.1, "surrogate_verified")
    assert cpu["actual_simulator_calls"] == 8 and gpu["actual_simulator_calls"] == 3
    assert gpu["surrogate_evaluations"] == 8 and gpu["simulator_verified_loss_curve"] == [1.0, 0.9, 0.8]
    assert [task["selected"] for task in simulated[-3:]] == [[], [0], [0, 1]]
    cpu.update(case_id="fake")
    gpu.update(case_id="fake")
    cpu["workflow_seconds"], gpu["workflow_seconds"] = 10.0, 4.0
    worker = MatchedWorker.__new__(MatchedWorker)
    worker.horizon, worker.calls = 200, 0
    worker.base_p_ini = np.ones(2)
    worker.base_n_opt_sector = np.ones(1)
    from stage3_ario_henriet2012 import Params
    worker.environment = SimpleNamespace(_net=SimpleNamespace(nodes=["a", "b"], P_ini=np.ones(2), n_opt_sector=np.ones(1)), _params=Params())
    original = baseline.simulate

    def fake_simulate(network, scenario, params):
        assert np.array_equal(network.P_ini, np.ones(2)) and np.array_equal(network.n_opt_sector, np.ones(1))
        network.P_ini[:] = 99
        network.n_opt_sector[:] = 99
        return {"P_a": np.full((200, 2), 0.5), "P_BAU": np.ones(2)}

    try:
        baseline.simulate = fake_simulate
        for _ in range(2):
            row = worker.evaluate({"delta0": [0.3, 0], "recovery_days": 100}, audit=True)
            assert row["loss"] == 0.5 and row["calls"] == 1
        assert worker.calls == 2
    finally:
        baseline.simulate = original
    timings = [{"mode": mode, "backend": backend, "seconds": seconds}
               for mode in ("preloaded", "workflow") for backend, seconds in (("cpu", 20), ("gpu", 10))]
    summary = summarize(timings, [], 10, 50, 50, {"gpu": 2, "cpu_pool": 1})
    assert summary["scenario"]["workflow"]["break_even_queries"] == 100
    paired = summarize(timings, [cpu, gpu], 10, 50, 50, {"gpu": 2, "cpu_pool": 1})
    assert paired["repair"]["break_even_repair_events"] == 17
    assert paired["repair"]["cases"][0]["cpu_final_true_loss"] == 0.8
    print("PASS: real two-node feature preparation, synthetic greedy rounds/ties, call accounting, prefix verification, state isolation, non-finite rejection and break-even checks; no real simulation or model forward.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT / "answer_result/henriet_runtime_suite_600")
    parser.add_argument("--throughput-json", type=Path, default=ROOT / "answer_result/batched_simulator_throughput.json")
    parser.add_argument("--generation-json", type=Path, default=ROOT / "answer_result/server_4090_benchmark/generation_cost_report.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "answer_result/multiseed/PhyCaSNet/seed0/v3de_g3_gns_seed0.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "answer_result/henriet_matched_workflow.json")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--final-events", type=int, default=500)
    parser.add_argument("--correctness-events", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--cpu-warmups-per-worker", type=int, default=1)
    parser.add_argument("--gpu-warmups", type=int, default=30)
    parser.add_argument("--repair-seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--damage-levels", type=float, nargs="+", default=[0.3, 0.6, 0.7])
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--training-seconds", type=float, default=47435.6)
    parser.add_argument("--training-provenance", default="Historical single-checkpoint wall-clock; not a retraining measurement on this benchmark host.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--order-seed", type=int, default=20260930)
    parser.add_argument("--skip-repair", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if min(args.final_events, args.correctness_events, args.repetitions, args.candidates, args.budget) < 1:
        raise ValueError("Counts and repetitions must be positive.")
    if args.budget > args.candidates or not 0 < args.alpha < 1 or not math.isfinite(args.training_seconds) or args.training_seconds <= 0:
        raise ValueError("Require budget <= candidates, 0 < alpha < 1 and positive training cost.")
    if min(args.cpu_warmups_per_worker, args.gpu_warmups) < 0 or any(not 0 < damage < 1 for damage in args.damage_levels):
        raise ValueError("Invalid warmup count or damage level.")
    if len(set(args.repair_seeds)) != len(args.repair_seeds) or len(set(args.damage_levels)) != len(args.damage_levels):
        raise ValueError("Repair seeds and damage levels must be unique.")
    previous = json.loads(args.throughput_json.read_text(encoding="utf-8"))
    generation = json.loads(args.generation_json.read_text(encoding="utf-8"))
    generation_seconds = float(generation["extrapolated_generation_5000"]["point_seconds"])
    if not math.isfinite(generation_seconds) or generation_seconds < 0:
        raise ValueError("Generation cost must be finite and non-negative.")
    workers = int(previous["cpu_final"]["selected_workers_from_tune_set"]) if args.workers is None else args.workers
    if not 1 <= workers <= (os.cpu_count() or 1):
        raise ValueError("Worker count must fit the visible logical CPU count.")
    final_ids = previous["protocol"]["event_sets"]["final"][:args.final_events]
    audit_ids = previous["protocol"]["event_sets"]["correctness"][:args.correctness_events]
    if len(final_ids) != args.final_events or len(audit_ids) != args.correctness_events or set(final_ids) & set(audit_ids):
        raise ValueError("Need sufficient disjoint final and correctness events from the existing benchmark.")
    baseline.validate_runtime_suite_precision(args.data_root, final_ids + audit_ids)
    event_specs = {event_id: load_event(args.data_root, event_id) for event_id in sorted(set(final_ids + audit_ids))}
    setup_started = time.perf_counter()
    predictor = Predictor(args.data_root, args.checkpoint, args.device)
    prepared = {event_id: predictor.prepare(event_specs[event_id]) for event_id in final_ids}
    for index in range(args.gpu_warmups):
        predictor.score(prepared[final_ids[index % len(final_ids)]])
    predictor.synchronize()
    gpu_setup = time.perf_counter() - setup_started
    reference = MatchedWorker(str(args.data_root), 200)
    repair_specs, candidate_sets = {}, {}
    if not args.skip_repair:
        for seed in args.repair_seeds:
            for damage in args.damage_levels:
                case_id = f"seed{seed}_damage{damage:g}"
                reference.environment._k_hubs = args.candidates
                reference.environment._default_damage = damage
                scenario = reference.environment._build_scenario(seed)
                delta = np.zeros(reference.environment._net.N, dtype=np.float64)
                candidates = []
                for node, value in scenario.initial_damage.items():
                    index = reference.environment._net.idx[node]
                    delta[index] = value
                    candidates.append(index)
                repair_specs[case_id] = {"delta0": delta, "nominal_damage": damage,
                                         "recovery_days": float(reference.environment._params.tau_recover)}
                candidate_sets[case_id] = candidates
    input_paths = [args.throughput_json, args.generation_json, args.checkpoint, args.data_root / "static_meta.pkl"]
    input_paths += [args.data_root / "events" / f"event_{event_id:06d}.npz" for event_id in sorted(event_specs)]
    input_paths += [Path(__file__), Path(baseline.__file__), ROOT / "fix/sim_env.py", ROOT / "cangzhou_pipeline/stage3_ario_henriet2012.py",
                    ROOT / "flagship_train/y8_reviewer_experiments.py"]
    input_paths += sorted((ROOT / "flagship_y8h_download/pi_dl").glob("*.py"))
    machine = baseline.system_metadata()
    machine.pop("pid", None)
    fingerprint = {"settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                   "workers": workers, "inputs": {str(path.resolve()): baseline.file_hash(path) for path in input_paths},
                   "machine": machine}
    state_path = args.output.with_name(args.output.stem + "_checkpoint.json")
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["fingerprint"] != fingerprint:
            raise RuntimeError("Inputs, source, hardware or arguments changed. Use a new output path; old results will not be mixed.")
    else:
        state = {"fingerprint": fingerprint, "stages": {}}
        baseline.atomic_json_write(state_path, state)

    def save(name, row):
        state["stages"][name] = row
        baseline.atomic_json_write(state_path, state)
        print(f"[saved] {name}", flush=True)

    context = mp.get_context("spawn")
    ready = context.Queue()
    pool_started = time.perf_counter()
    pool = context.Pool(workers, initializer=worker_init,
                        initargs=(str(args.data_root), event_specs, repair_specs, args.cpu_warmups_per_worker, ready))
    completed = False
    try:
        initialized = [ready.get(timeout=600) for _ in range(workers)]
        cpu_setup = time.perf_counter() - pool_started
        print(f"[ready] workers={workers}; final_events={len(final_ids)}; repair_cases={len(repair_specs)}", flush=True)
        if "correctness" not in state["stages"]:
            serial = [reference.evaluate(event_specs[event_id], audit=True) for event_id in audit_ids]
            parallel = pool.map(worker_job, [{"kind": "event", "mode": "preloaded", "event_id": event_id, "audit": True}
                                            for event_id in audit_ids], chunksize=1)
            maximum_label_error = 0.0
            for event_id, left, right in zip(audit_ids, serial, parallel):
                for key in ("trajectory_hash", "keyframe_hash", "peak_hash", "loss"):
                    if left[key] != right[key]:
                        raise RuntimeError(f"Serial/pool mismatch at event {event_id}: {key}")
                with np.load(args.data_root / "events" / f"event_{event_id:06d}.npz", allow_pickle=True) as payload:
                    maximum_label_error = max(maximum_label_error, float(np.max(np.abs(left["keyframes"] - payload["u_keyframes"]))),
                                              float(np.max(np.abs(left["peak"] - payload["peak_loss_node"]))))
            if maximum_label_error > 1e-6:
                raise RuntimeError(f"Stored-label replay error {maximum_label_error} exceeds 1e-6.")
            if repair_specs:
                case_id = next(iter(repair_specs))
                candidates = candidate_sets[case_id]
                trials = [[], [candidates[0]]]
                if len(candidates) > 1:
                    trials += [[candidates[1]], [candidates[0], candidates[1]]]
                checked = pool.map(worker_job, [{"kind": "repair", "case_id": case_id, "selected": trial,
                                                 "alpha": args.alpha, "audit": True} for trial in trials], chunksize=1)
                for trial, right in zip(trials, checked):
                    left = reference.evaluate(repair_specs[case_id], trial, args.alpha, audit=True)
                    if any(left[key] != right[key] for key in ("trajectory_hash", "peak_hash", "loss")):
                        raise RuntimeError("Serial/pool repaired-scenario mismatch.")
            save("correctness", {"events": len(audit_ids), "serial_pool_equal": True, "maximum_label_error": maximum_label_error})
        for repetition in range(args.repetitions):
            combinations = [(mode, backend) for mode in ("preloaded", "workflow") for backend in ("cpu", "gpu")]
            np.random.default_rng(args.order_seed + repetition).shuffle(combinations)
            for mode, backend in combinations:
                name = f"runtime_{repetition}_{mode}_{backend}"
                if name in state["stages"]:
                    continue
                predictor.synchronize()
                started = time.perf_counter()
                if backend == "cpu":
                    rows = pool.map(worker_job, [{"kind": "event", "mode": mode, "event_id": event_id} for event_id in final_ids], chunksize=1)
                    losses = [row["loss"] for row in rows]
                    calls = sum(row["calls"] for row in rows)
                else:
                    losses = [predictor.score(prepared[event_id] if mode == "preloaded" else
                                               predictor.prepare(load_event(args.data_root, event_id))) for event_id in final_ids]
                    calls = 0
                predictor.synchronize()
                elapsed = time.perf_counter() - started
                save(name, {"repetition": repetition, "mode": mode, "backend": backend, "seconds": elapsed,
                            "actual_simulator_calls": calls, "losses": losses})
        for case_index, (case_id, spec) in enumerate(repair_specs.items()):
            for repetition in range(args.repetitions):
                policies = ["cpu_greedy", "surrogate_verified"]
                if (case_index + repetition) % 2:
                    policies.reverse()
                for policy in policies:
                    name = f"repair_{case_id}_{repetition}_{policy}"
                    if name in state["stages"]:
                        continue
                    result = repair_run(pool, predictor, case_id, spec, candidate_sets[case_id], args.budget, args.alpha, policy)
                    result.update(case_id=case_id, repetition=repetition)
                    save(name, result)
        timings = [row for name, row in state["stages"].items() if name.startswith("runtime_")]
        repairs = [row for name, row in state["stages"].items() if name.startswith("repair_")]
        assert len(timings) == args.repetitions * 4 and len(repairs) == args.repetitions * len(repair_specs) * 2
        for backend in ("cpu", "gpu"):
            records = [row for row in timings if row["backend"] == backend]
            if not all(np.allclose(row["losses"], records[0]["losses"], atol=1e-7, rtol=1e-7) for row in records):
                raise RuntimeError(f"{backend} outputs differ across repetitions or loading modes.")
        for case_id in repair_specs:
            for policy in ("cpu_greedy", "surrogate_verified"):
                records = [row for row in repairs if row["case_id"] == case_id and row["policy"] == policy]
                if any(row["selected"] != records[0]["selected"] or
                       not np.allclose(row["simulator_verified_loss_curve"], records[0]["simulator_verified_loss_curve"], atol=1e-10, rtol=1e-10)
                       for row in records):
                    raise RuntimeError("Repair decisions or verification outcomes changed across repetitions.")
        setup = {"gpu": gpu_setup, "cpu_pool": cpu_setup, "cpu_worker_initialization": initialized}
        protocol = {
            "workers": workers, "final_event_ids": final_ids, "correctness_event_ids": audit_ids,
            "repetitions": args.repetitions, "formal_size_defaults_met": args.repetitions >= 5 and len(final_ids) >= 500 and len(audit_ids) >= 100,
            "preloaded_boundary": "Both inputs and backend state preloaded. CPU includes dispatch, simulation and scalar reduction; GPU includes forward, scalar reduction and synchronization.",
            "workflow_boundary": "Both backends load the saved event inside timing. GPU also includes feature construction and host-to-device transfer; CPU includes dispatch and scenario construction. Persistent worker/model startup excluded and separately reported.",
            "cache_state": "Warm services; file hashes, correctness checks and previous repetitions warm OS caches. Not a cold-disk benchmark.",
            "repair_boundary": "In-memory Day-0 state; CPU scores only the current greedy round in parallel. GPU constructs features and transfers every trial, including the unrepaired baseline. Full-simulator verification of all prefixes included, with no feedback to surrogate selection.",
            "output_scope": "Simulator: full-network 200-day aggregate peak. Surrogate: five-hop prediction nodes at ten key dates, BAU-weighted with outside production fixed to BAU. No assertion of equal model/simulator accuracy or resolution.",
            "generation_seconds": generation_seconds, "generation_cost_source": str(args.generation_json.resolve()),
            "generation_scope": "External measured/extrapolated cost; hardware and worker settings remain those declared by the generation source.",
            "training_seconds": args.training_seconds, "training_provenance": args.training_provenance,
            "break_even_scope": "Point estimates conditional on supplied generation/training costs and measured warm-service medians. No confidence interval or claim of jointly measured same-host training.",
        }
        baseline.atomic_json_write(args.output, {"protocol": protocol, "fingerprint": fingerprint, "setup": setup,
                                               "correctness": state["stages"]["correctness"], "timing_records": timings,
                                               "repair_records": repairs, "summary": summarize(timings, repairs, len(final_ids), generation_seconds, args.training_seconds, setup)})
        print(f"[complete] {args.output}", flush=True)
        completed = True
    finally:
        if completed:
            pool.close()
        else:
            pool.terminate()
        pool.join()
        ready.close()


if __name__ == "__main__":
    main()