"""Audit event-split leakage and create a leave-sector-out experiment manifest.

This script answers three distinct questions without using outcome labels to
construct the split:
  1. Reproduce the seeded random 80/10/10 event-identity split exactly.
  2. Quantify exact and near-duplicate shocks from train to validation/test.
  3. Pre-register held-out sectors and event IDs for a leave-sector-out retrain.

The current event split is i.i.d. and transductive on one fixed graph; it is not
a chronological forecast split.  The leave-sector-out manifest tests unseen
shock locations/sectors, not a wholly unseen graph.  A new-graph claim requires
training on one static network and evaluation on another compatible network.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EventRecord:
    event_id: int
    mode: str
    tier: str
    targets: frozenset[int]
    target_sectors: tuple[int, ...]
    mean_delta: float
    recovery_days: float
    n_targets: int


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0,
                        help="Exact event split seed used by Config.seed.")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--holdout-sector-seed", type=int, default=20260808)
    parser.add_argument("--holdout-node-frac", type=float, default=0.2,
                        help="Target fraction of graph nodes assigned to held-out sectors.")
    parser.add_argument("--near-jaccard", type=float, default=0.8)
    parser.add_argument("--near-delta", type=float, default=0.05)
    parser.add_argument("--near-tau-days", type=float, default=10.0)
    return parser.parse_args()


def sha256_ids(ids: list[int]) -> str:
    text = ",".join(map(str, ids)).encode("ascii")
    return hashlib.sha256(text).hexdigest()


def load_static(data_root: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with (data_root / "static_meta.pkl").open("rb") as handle:
        static = pickle.load(handle)
    sectors = np.asarray(static["sectors"], dtype=np.int64)
    graph = {
        "nodes": int(static["V"]),
        "edges": int(static["E"]),
        "sector_count": int(np.unique(sectors).size),
    }
    return sectors, graph


def load_records(events_dir: Path, sectors: np.ndarray) -> dict[int, EventRecord]:
    files = sorted(events_dir.glob("event_*.npz"))
    records: dict[int, EventRecord] = {}
    report_every = max(1, len(files) // 20)
    progress(f"Loading metadata for {len(files):,} events.")
    for index, path in enumerate(files, 1):
        event_id = int(path.stem.split("_")[1])
        with np.load(path, allow_pickle=True) as event:
            meta = event["shock_meta"].item()
            shock_mask = np.asarray(event["shock_mask"]) > 0
        targets = frozenset(np.flatnonzero(shock_mask).astype(int).tolist())
        delta_values = np.asarray(meta.get("delta_per_target", []), dtype=float)
        mean_delta = (float(delta_values.mean()) if delta_values.size
                      else float(meta.get("delta", 0.0)))
        target_sectors = tuple(sorted(set(sectors[list(targets)].astype(int).tolist())))
        records[event_id] = EventRecord(
            event_id=event_id,
            mode=str(meta.get("mode", "unknown")),
            tier=str(meta.get("tier", "unknown")),
            targets=targets,
            target_sectors=target_sectors,
            mean_delta=mean_delta,
            recovery_days=float(meta.get("recovery_days", np.nan)),
            n_targets=len(targets),
        )
        if index % report_every == 0 or index == len(files):
            progress(f"Loaded {index:,}/{len(files):,} event metadata records.")
    return records


def random_identity_split(ids: list[int], seed: int, train_frac: float,
                          val_frac: float) -> dict[str, list[int]]:
    shuffled = ids.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    count = len(shuffled)
    train_end = int(count * train_frac)
    val_end = train_end + int(count * val_frac)
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 1.0


def distribution_summary(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def nearest_split_audit(reference: list[EventRecord], query: list[EventRecord],
                        near_jaccard: float, near_delta: float,
                        near_tau: float, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    report_every = max(1, len(query) // 10)
    progress(f"Auditing {label}: {len(query):,} events against {len(reference):,} reference events.")
    for query_index, current in enumerate(query, 1):
        best: EventRecord | None = None
        best_score = -1.0
        best_jaccard = 0.0
        best_delta = float("inf")
        best_tau = float("inf")
        max_jaccard = -1.0
        max_jaccard_delta = float("inf")
        max_jaccard_tau = float("inf")
        closest_same_target_delta = float("inf")
        closest_same_target_tau = float("inf")
        closest_same_target_distance = float("inf")
        exact_signature = False
        strict_duplicate = False
        relaxed_duplicate = False
        same_target_set = False
        threshold_hits = {0.25: False, 0.50: False, 0.80: False, 1.00: False}

        for candidate in reference:
            target_similarity = jaccard(current.targets, candidate.targets)
            delta_difference = abs(current.mean_delta - candidate.mean_delta)
            tau_difference = abs(current.recovery_days - candidate.recovery_days)
            same_categories = current.mode == candidate.mode and current.tier == candidate.tier
            if target_similarity > max_jaccard:
                max_jaccard = target_similarity
                max_jaccard_delta = delta_difference
                max_jaccard_tau = tau_difference
            if target_similarity == 1.0:
                parameter_distance = np.hypot(
                    delta_difference / max(near_delta, 1e-9),
                    tau_difference / max(near_tau, 1e-9),
                )
                if parameter_distance < closest_same_target_distance:
                    closest_same_target_distance = parameter_distance
                    closest_same_target_delta = delta_difference
                    closest_same_target_tau = tau_difference
            for threshold in threshold_hits:
                threshold_hits[threshold] |= target_similarity >= threshold
            same_target_set |= target_similarity == 1.0
            exact_signature |= bool(same_categories
                                    and target_similarity == 1.0
                                    and current.mean_delta == candidate.mean_delta
                                    and current.recovery_days == candidate.recovery_days)
            strict_duplicate |= bool(same_categories
                                     and target_similarity == 1.0
                                     and delta_difference <= 0.01
                                     and tau_difference <= 2.0)
            relaxed_duplicate |= bool(same_categories
                                      and target_similarity >= near_jaccard
                                      and delta_difference <= near_delta
                                      and tau_difference <= near_tau)
            # Ranking is only for nearest-neighbour diagnostics. Duplicate flags
            # below remain interpretable threshold rules, not this composite score.
            score = (0.70 * target_similarity
                     + 0.15 * float(same_categories)
                     + 0.075 * np.exp(-delta_difference / max(near_delta, 1e-9))
                     + 0.075 * np.exp(-tau_difference / max(near_tau, 1e-9)))
            if score > best_score:
                best = candidate
                best_score = float(score)
                best_jaccard = target_similarity
                best_delta = delta_difference
                best_tau = tau_difference

        assert best is not None
        rows.append({
            "split": label,
            "event_id": current.event_id,
            "nearest_train_event_id": best.event_id,
            "nearest_composite_score": best_score,
            "target_jaccard": best_jaccard,
            "delta_abs_difference": best_delta,
            "tau_abs_difference_days": best_tau,
            "max_target_jaccard_any_reference": max_jaccard,
            "delta_abs_difference_at_max_jaccard": max_jaccard_delta,
            "tau_abs_difference_at_max_jaccard_days": max_jaccard_tau,
            "delta_abs_difference_closest_same_target": closest_same_target_delta,
            "tau_abs_difference_closest_same_target_days": closest_same_target_tau,
            "normalized_distance_closest_same_target": closest_same_target_distance,
            "has_same_target_set": same_target_set,
            "has_target_jaccard_ge_0_25": threshold_hits[0.25],
            "has_target_jaccard_ge_0_50": threshold_hits[0.50],
            "has_target_jaccard_ge_0_80": threshold_hits[0.80],
            "has_target_jaccard_eq_1_00": threshold_hits[1.00],
            "same_mode": current.mode == best.mode,
            "same_tier": current.tier == best.tier,
            "exact_full_signature": exact_signature,
            "strict_near_duplicate": strict_duplicate,
            "relaxed_near_duplicate": relaxed_duplicate,
        })
        if query_index % report_every == 0 or query_index == len(query):
            progress(f"{label}: audited {query_index:,}/{len(query):,} events.")

    max_jaccards = np.asarray(
        [row["max_target_jaccard_any_reference"] for row in rows], dtype=float
    )
    scores = np.asarray([row["nearest_composite_score"] for row in rows], dtype=float)
    same_target_rows = [row for row in rows if row["has_same_target_set"]]

    def fraction(field: str) -> float:
        return float(np.mean([row[field] for row in rows]))

    summary = {
        "n_events": len(rows),
        "exact_full_signature_count": sum(row["exact_full_signature"] for row in rows),
        "strict_near_duplicate_count": sum(row["strict_near_duplicate"] for row in rows),
        "relaxed_near_duplicate_count": sum(row["relaxed_near_duplicate"] for row in rows),
        "exact_full_signature_fraction": float(np.mean([row["exact_full_signature"] for row in rows])),
        "strict_near_duplicate_fraction": float(np.mean([row["strict_near_duplicate"] for row in rows])),
        "relaxed_near_duplicate_fraction": float(np.mean([row["relaxed_near_duplicate"] for row in rows])),
        "max_target_jaccard_any_reference": distribution_summary(max_jaccards),
        "target_overlap_existence_fraction": {
            "jaccard_ge_0_25": fraction("has_target_jaccard_ge_0_25"),
            "jaccard_ge_0_50": fraction("has_target_jaccard_ge_0_50"),
            "jaccard_ge_0_80": fraction("has_target_jaccard_ge_0_80"),
            "jaccard_eq_1_00": fraction("has_target_jaccard_eq_1_00"),
        },
        "same_target_set_parameter_difference": (
            {
                "n_events": len(same_target_rows),
                "delta_abs_difference": distribution_summary(
                    [row["delta_abs_difference_closest_same_target"] for row in same_target_rows]
                ),
                "tau_abs_difference_days": distribution_summary(
                    [row["tau_abs_difference_closest_same_target_days"] for row in same_target_rows]
                ),
                "normalized_parameter_distance": distribution_summary(
                    [row["normalized_distance_closest_same_target"] for row in same_target_rows]
                ),
            } if same_target_rows else {"n_events": 0}
        ),
        "nearest_composite_score": distribution_summary(scores),
    }
    return summary, rows


def categorical_distribution(records: list[EventRecord], field: str) -> dict[str, float]:
    counts = Counter(getattr(record, field) for record in records)
    return {str(key): value / max(len(records), 1) for key, value in sorted(counts.items())}


def choose_heldout_sectors(sectors: np.ndarray, node_fraction: float,
                           seed: int) -> list[int]:
    """Choose sectors using static topology only, never event labels or outcomes."""
    sector_ids, counts = np.unique(sectors, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sector_ids))
    target_nodes = max(1, int(round(len(sectors) * node_fraction)))
    chosen: list[int] = []
    covered = 0
    for position in order:
        chosen.append(int(sector_ids[position]))
        covered += int(counts[position])
        if covered >= target_nodes:
            break
    return sorted(chosen)


def create_leave_sector_out(records: dict[int, EventRecord], sectors: np.ndarray,
                            holdout_sectors: list[int], seed: int) -> dict[str, Any]:
    heldout = set(holdout_sectors)
    eligible_train: list[int] = []
    strict_test: list[int] = []
    boundary_test: list[int] = []
    mixed_excluded: list[int] = []
    for event_id, record in records.items():
        target_sector_set = set(record.target_sectors)
        if target_sector_set and target_sector_set <= heldout:
            strict_test.append(event_id)
        elif target_sector_set & heldout:
            boundary_test.append(event_id)
            mixed_excluded.append(event_id)
        else:
            eligible_train.append(event_id)

    shuffled = sorted(eligible_train)
    np.random.default_rng(seed).shuffle(shuffled)
    n_validation = max(1, int(round(0.1 * len(shuffled))))
    validation = shuffled[:n_validation]
    train = shuffled[n_validation:]
    strict_test.sort()
    boundary_test.sort()
    heldout_nodes = np.flatnonzero(np.isin(sectors, holdout_sectors)).astype(int).tolist()
    return {
        "definition": {
            "train": "all shocked firms are outside held-out sectors",
            "validation": "seeded 10% subset of eligible non-heldout-sector events",
            "strict_test": "all shocked firms are in held-out sectors",
            "boundary_test": "at least one, but not all, shocked firms are in held-out sectors",
            "claim_scope": "unseen shock sectors/locations on the same fixed graph; not unseen-topology generalization",
            "selection_guard": "held-out sectors selected from static sector/node counts only; no trajectory or loss labels used",
        },
        "heldout_sector_ids": holdout_sectors,
        "heldout_node_ids": heldout_nodes,
        "heldout_node_fraction": len(heldout_nodes) / len(sectors),
        "train_ids": train,
        "validation_ids": validation,
        "strict_test_ids": strict_test,
        "boundary_test_ids": boundary_test,
        "mixed_events_excluded_from_strict_test": mixed_excluded,
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "strict_test": len(strict_test),
            "boundary_test": len(boundary_test),
        },
        "id_sha256": {
            "train": sha256_ids(train),
            "validation": sha256_ids(validation),
            "strict_test": sha256_ids(strict_test),
            "boundary_test": sha256_ids(boundary_test),
        },
    }


def write_neighbour_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_dir = data_root / "events"

    sectors, graph = load_static(data_root)
    records = load_records(events_dir, sectors)
    ids = sorted(records)
    split = random_identity_split(ids, args.seed, args.train_frac, args.val_frac)
    if set(split["train"]) & set(split["validation"]) or set(split["train"]) & set(split["test"]):
        raise RuntimeError("Event identity overlap detected between splits.")
    progress(f"Reproduced random identity split: train={len(split['train'])}, "
             f"validation={len(split['validation'])}, test={len(split['test'])}.")

    train_records = [records[event_id] for event_id in split["train"]]
    validation_records = [records[event_id] for event_id in split["validation"]]
    test_records = [records[event_id] for event_id in split["test"]]
    val_summary, val_rows = nearest_split_audit(
        train_records, [records[event_id] for event_id in split["validation"]],
        args.near_jaccard, args.near_delta, args.near_tau_days, "validation",
    )
    test_summary, test_rows = nearest_split_audit(
        train_records, [records[event_id] for event_id in split["test"]],
        args.near_jaccard, args.near_delta, args.near_tau_days, "test",
    )
    validation_test_summary, validation_test_rows = nearest_split_audit(
        validation_records, test_records,
        args.near_jaccard, args.near_delta, args.near_tau_days,
        "test_against_validation",
    )

    heldout_sectors = choose_heldout_sectors(
        sectors, args.holdout_node_frac, args.holdout_sector_seed,
    )
    leave_sector_out = create_leave_sector_out(
        records, sectors, heldout_sectors, args.holdout_sector_seed,
    )
    manifest_path = output_dir / "leave_sector_out_manifest.json"
    manifest_path.write_text(json.dumps(leave_sector_out, indent=2) + "\n", encoding="utf-8")

    split_records = {
        name: [records[event_id] for event_id in event_ids]
        for name, event_ids in split.items()
    }
    report = {
        "protocol": {
            "assignment": "sorted event IDs are shuffled by numpy.default_rng(seed).shuffle, then sliced",
            "seed": args.seed,
            "fractions": {"train": args.train_frac, "validation": args.val_frac,
                          "test": 1.0 - args.train_frac - args.val_frac},
            "chronological_split": False,
            "event_generation": "event seed = seed_base + event_id; events are independent simulator draws on one fixed graph",
            "future_information_statement": "No event history is passed between samples. The minimal scalar mode retains only mean and nominal day-0 damage; recovery time is zeroed.",
            "evaluation_scope": "transductive event generalization on a fixed graph; firms and edges are shared across splits",
            "near_duplicate_rule_strict": "same target set, mode and tier; abs(delta)<=0.01; abs(tau)<=2 days",
            "near_duplicate_rule_relaxed": (
                f"same mode and tier; target Jaccard>={args.near_jaccard}; "
                f"abs(delta)<={args.near_delta}; abs(tau)<={args.near_tau_days} days"
            ),
        },
        "graph": graph,
        "split_counts": {name: len(event_ids) for name, event_ids in split.items()},
        "split_id_sha256": {name: sha256_ids(event_ids) for name, event_ids in split.items()},
        "identity_intersection_counts": {
            "train_validation": len(set(split["train"]) & set(split["validation"])),
            "train_test": len(set(split["train"]) & set(split["test"])),
            "validation_test": len(set(split["validation"]) & set(split["test"])),
        },
        "mode_distribution": {
            name: categorical_distribution(items, "mode") for name, items in split_records.items()
        },
        "tier_distribution": {
            name: categorical_distribution(items, "tier") for name, items in split_records.items()
        },
        "pairwise_event_similarity_audit": {
            "train_to_validation": val_summary,
            "train_to_test": test_summary,
            "validation_to_test": validation_test_summary,
        },
        "leave_sector_out_manifest": str(manifest_path),
        "leave_sector_out_counts": leave_sector_out["counts"],
        "limitations": [
            "Random event identity splitting is not a prospective chronological forecast because simulator events have no temporal ordering.",
            "The standard test is transductive with respect to the static graph.",
            "The leave-sector-out manifest tests unseen shock sectors on the same graph; it does not establish transfer to an entirely new topology.",
            "A true unseen-topology claim requires a separately generated compatible graph and training/evaluation without node-identity alignment.",
        ],
    }
    report_path = output_dir / "event_split_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_neighbour_csv(
        output_dir / "event_pair_nearest_neighbors.csv",
        val_rows + test_rows + validation_test_rows,
    )

    progress(f"Test exact duplicates: {test_summary['exact_full_signature_count']}/{test_summary['n_events']}; "
             f"strict near duplicates: {test_summary['strict_near_duplicate_count']}/{test_summary['n_events']}; "
             f"relaxed near duplicates: {test_summary['relaxed_near_duplicate_count']}/{test_summary['n_events']}.")
    progress(f"Leave-sector-out counts: {leave_sector_out['counts']}.")
    progress(f"Saved audit: {report_path}")
    progress(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
