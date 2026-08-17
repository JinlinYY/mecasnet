"""Shared train/eval helpers for V3DE training, ablation, and baselines.

All three scripts (train_v3de.py, ablate_v3de.py, compare_baselines.py)
import from here so the evaluation protocol is identical and any
reported number is directly comparable across experiments.

Conventions (locked):
  - cascade subset       : (~shock_mask) & (peak_loss > 0.05)
  - r2 returns NaN if    : empty mask OR ground-truth std < 1e-2
  - per-keyframe R²pk_csc: NaN frames excluded from r2_kf_csc_mean
  - paired delta vs base : aligned per-seed, t-stat = mean / (std/sqrt(n))
"""
from __future__ import annotations
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CascadeEventDataset, collate_single
from .losses import total_loss


# ---------------------------------------------------------------------------
# R² and evaluation
# ---------------------------------------------------------------------------
def r2(p: np.ndarray, g: np.ndarray) -> float:
    if p.size == 0 or float(g.std()) < 1e-2:
        return float("nan")
    ss_res = float(((p - g) ** 2).sum())
    ss_tot = float(((g - g.mean()) ** 2).sum()) + 1e-8
    return 1.0 - ss_res / ss_tot


def _bau_bin(value: float) -> str:
    if value < 0.005:
        return "[0.001,0.005)"
    if value < 0.02:
        return "[0.005,0.02)"
    if value < 0.10:
        return "[0.02,0.10)"
    return "[0.10,inf)"


def _metrics_from_event_chunks(chunks: List[Dict]) -> Dict:
    p = np.concatenate([chunk["peak_pred"] for chunk in chunks])
    g = np.concatenate([chunk["peak_true"] for chunk in chunks])
    s = np.concatenate([chunk["shock"] for chunk in chunks])
    csc = (~s) & (g > 0.05)
    metrics = {
        "n_events": len(chunks),
        "n_nodes": int(p.size),
        "n_csc": int(csc.sum()),
        "r2_pk": r2(p, g),
        "r2_pk_csc": r2(p[csc], g[csc]) if csc.any() else float("nan"),
        "mae_pk_csc": float(np.abs(p[csc] - g[csc]).mean()) if csc.any() else float("nan"),
    }
    if all(chunk["kf_pred"] is not None for chunk in chunks):
        kfp = np.concatenate([chunk["kf_pred"] for chunk in chunks], axis=1)
        kfg = np.concatenate([chunk["kf_true"] for chunk in chunks], axis=1)
        kf_r2 = [r2(kfp[index][csc], kfg[index][csc]) for index in range(kfp.shape[0])]
        valid = [value for value in kf_r2 if not np.isnan(value)]
        metrics["r2_kf_csc"] = kf_r2
        metrics["r2_kf_csc_mean"] = float(np.mean(valid)) if valid else float("nan")
    return metrics


def _stratified_metrics(chunks: List[Dict]) -> Dict:
    dimensions = {
        "tier": lambda chunk: chunk["tier"],
        "mode": lambda chunk: chunk["mode"],
        "tier_by_mode": lambda chunk: f"{chunk['tier']}|{chunk['mode']}",
        "peak_aggregate_bau_bin": lambda chunk: _bau_bin(chunk["peak_aggregate_bau_loss"]),
    }
    result = {}
    for dimension, key_fn in dimensions.items():
        groups = {}
        for chunk in chunks:
            groups.setdefault(key_fn(chunk), []).append(chunk)
        result[dimension] = {
            key: _metrics_from_event_chunks(group)
            for key, group in sorted(groups.items())
        }
    return result


@torch.no_grad()
def evaluate(model, loader, cfg, device) -> Dict:
    model.eval()
    pks_p, pks_g, shks = [], [], []
    kf_p, kf_g = [], []
    losses = []
    event_chunks = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        out = model(batch)
        parts = total_loss(out, batch, cfg)
        losses.append(parts["total"].item())
        pks_p.append(out["peak"].cpu().numpy())
        pks_g.append(batch["peak_loss"].cpu().numpy())
        shks.append(batch["shock_mask"].cpu().numpy())
        event_kf_pred = None
        event_kf_true = None
        if "u_keyframes" in out:
            event_kf_pred = out["u_keyframes"].cpu().numpy()
            event_kf_true = batch["u_keyframes"].cpu().numpy()
            kf_p.append(event_kf_pred)
            kf_g.append(event_kf_true)
        if "audit_tier" in batch:
            event_chunks.append({
                "event_id": int(batch["event_id"]),
                "tier": str(batch["audit_tier"]),
                "mode": str(batch["audit_mode"]),
                "peak_aggregate_bau_loss": float(
                    batch["audit_peak_aggregate_bau_loss"]),
                "peak_pred": out["peak"].cpu().numpy(),
                "peak_true": batch["peak_loss"].cpu().numpy(),
                "shock": batch["shock_mask"].cpu().numpy() > 0.5,
                "kf_pred": event_kf_pred,
                "kf_true": event_kf_true,
            })
    p = np.concatenate(pks_p); g = np.concatenate(pks_g)
    s = np.concatenate(shks) > 0.5
    csc = (~s) & (g > 0.05)
    res = dict(
        loss=float(np.mean(losses)),
        r2_pk=r2(p, g),
        r2_pk_shk=r2(p[s], g[s]) if s.any() else float("nan"),
        r2_pk_csc=r2(p[csc], g[csc]) if csc.any() else float("nan"),
        mae_pk_csc=float(np.abs(p[csc] - g[csc]).mean()) if csc.any() else float("nan"),
        n_csc=int(csc.sum()), n_shk=int(s.sum()), n_total=int(p.shape[0]),
    )
    if kf_p:
        kfp = np.concatenate(kf_p, axis=1)
        kfg = np.concatenate(kf_g, axis=1)
        K = kfp.shape[0]
        kf_r2 = [r2(kfp[k][csc], kfg[k][csc]) for k in range(K)]
        res["r2_kf_csc"] = kf_r2
        valid = [x for x in kf_r2 if not np.isnan(x)]
        res["r2_kf_csc_mean"] = float(np.mean(valid)) if valid else float("nan")
    if event_chunks:
        res["stratified"] = _stratified_metrics(event_chunks)
    return res


# ---------------------------------------------------------------------------
# Training: single fixed-epoch run (for ablation / comparison)
# ---------------------------------------------------------------------------
def train_fixed(cfg, net, train_ids, val_ids, device,
                build_model_fn: Callable, n_epochs: int) -> Dict:
    """Train a model for n_epochs, evaluate once on val at the end.

    Mirrors ablate_v3_check.train_short.  Used by ablate_v3de.py and
    compare_baselines.py where we want apples-to-apples short runs.
    """
    ds_tr = CascadeEventDataset(cfg, net, train_ids, train_mode=True)
    ds_va = CascadeEventDataset(cfg, net, val_ids)
    dl_tr = DataLoader(ds_tr, batch_size=1, shuffle=True,
                       num_workers=getattr(cfg, "train_num_workers", 0),
                       collate_fn=collate_single)
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False,
                       num_workers=getattr(cfg, "eval_num_workers", 0),
                       collate_fn=collate_single)
    model = build_model_fn().to(device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        for i, b in enumerate(dl_tr):
            b = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in b.items()}
            out = model(b)
            parts = total_loss(out, b, cfg)
            (parts["total"] / cfg.grad_accum).backward()
            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step(); opt.zero_grad()
    metrics = evaluate(model, dl_va, cfg, device)
    metrics["params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return metrics


# ---------------------------------------------------------------------------
# Training: per-epoch eval + best-checkpoint (for train_v3de.py)
# ---------------------------------------------------------------------------
def train_with_val(cfg, net, train_ids, val_ids, device,
                   build_model_fn: Callable, n_epochs: int,
                   verbose: bool = True,
                   eval_every: int = 1,
                   lr_schedule: str = "const",
                   warmup_epochs: int = 0,
                   min_lr_ratio: float = 0.1) -> Tuple[Dict, List[Dict], dict]:
    """Train n_epochs, eval on val every `eval_every` epochs, track best.

    Optional LR schedule (per-epoch):
      lr_schedule='const'   : keep cfg.lr fixed.
      lr_schedule='cosine'  : linear warmup for `warmup_epochs` from 0 -> cfg.lr,
                              then cosine decay from cfg.lr -> cfg.lr*min_lr_ratio
                              over the remaining epochs.

    Returns
    -------
    best_metrics : dict
        eval metrics at the epoch with best val R²pk_csc
    history : list[dict]
        per-eval metrics including {epoch, train_loss, val_*}
    best_state : dict
        state_dict copy of the best checkpoint (CPU tensors)
    """
    import math
    ds_tr = CascadeEventDataset(cfg, net, train_ids, train_mode=True)
    ds_va = CascadeEventDataset(cfg, net, val_ids)
    dl_tr = DataLoader(ds_tr, batch_size=1, shuffle=True,
                       num_workers=getattr(cfg, "train_num_workers", 0),
                       collate_fn=collate_single)
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False,
                       num_workers=getattr(cfg, "eval_num_workers", 0),
                       collate_fn=collate_single)
    model = build_model_fn().to(device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ----- LR schedule helper (per-epoch) -----
    base_lr = cfg.lr
    def _lr_for_epoch(ep_1based: int) -> float:
        if lr_schedule == "const":
            return base_lr
        # cosine with linear warmup
        if warmup_epochs > 0 and ep_1based <= warmup_epochs:
            return base_lr * ep_1based / max(warmup_epochs, 1)
        # remaining epochs after warmup
        decay_total = max(n_epochs - warmup_epochs, 1)
        t = (ep_1based - warmup_epochs) / decay_total      # 0..1
        t = min(max(t, 0.0), 1.0)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * t))    # 1 -> 0
        return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos_factor)

    history: List[Dict] = []
    best = {"r2_pk_csc": -float("inf")}
    best_state: dict = {k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()}
    for ep in range(1, n_epochs + 1):
        # apply per-epoch LR
        cur_lr = _lr_for_epoch(ep)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        model.train()
        opt.zero_grad()
        ep_losses = []
        t0 = time.time()
        for i, b in enumerate(dl_tr):
            b = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in b.items()}
            out = model(b)
            parts = total_loss(out, b, cfg)
            (parts["total"] / cfg.grad_accum).backward()
            ep_losses.append(parts["total"].item())
            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step(); opt.zero_grad()
        train_loss = float(np.mean(ep_losses))
        if ep % eval_every == 0 or ep == n_epochs:
            v_metrics = evaluate(model, dl_va, cfg, device)
            row = {"epoch": ep, "train_loss": train_loss,
                   "lr": cur_lr,
                   "wall_s": round(time.time() - t0, 1)}
            row.update({f"val_{k}": v for k, v in v_metrics.items()
                        if k != "r2_kf_csc"})
            history.append(row)
            if v_metrics["r2_pk_csc"] > best["r2_pk_csc"]:
                best = v_metrics
                best["epoch"] = ep
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            if verbose:
                print(f"  [ep {ep:3d}/{n_epochs}] tr_loss={train_loss:.4f}  "
                      f"lr={cur_lr:.2e}  "
                      f"val: R²pk={v_metrics['r2_pk']:+.3f}  "
                      f"shk={v_metrics['r2_pk_shk']:+.3f}  "
                      f"csc={v_metrics['r2_pk_csc']:+.3f}  "
                      f"kf̄={v_metrics.get('r2_kf_csc_mean', float('nan')):+.3f}  "
                      f"({row['wall_s']}s)"
                      + ("  [BEST]" if v_metrics["r2_pk_csc"] >= best["r2_pk_csc"] - 1e-9
                         and ep == best.get("epoch") else ""))
    best["params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return best, history, best_state


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------
def agg_seeds(runs: List[Dict], keys=("loss", "r2_pk", "r2_pk_shk",
                                       "r2_pk_csc", "mae_pk_csc",
                                       "r2_kf_csc_mean")) -> Dict:
    """Mean ± std (ddof=1) across seeds for each scalar key."""
    if not runs:
        return {}
    out = {"n_seeds": len(runs), "params": runs[0].get("params", 0)}
    for k in keys:
        vals = [r[k] for r in runs
                if k in r and not (isinstance(r[k], float) and np.isnan(r[k]))]
        if not vals:
            out[k], out[k + "_std"] = float("nan"), float("nan")
            continue
        out[k] = float(np.mean(vals))
        out[k + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    if all("r2_kf_csc" in r for r in runs):
        kf = np.array([r["r2_kf_csc"] for r in runs], dtype=float)
        out["r2_kf_csc"] = list(np.nanmean(kf, axis=0))
        out["r2_kf_csc_std"] = (list(np.nanstd(kf, axis=0, ddof=1))
                                if len(runs) > 1 else list(np.zeros(kf.shape[1])))
    return out


def paired_delta(var_runs: List[Dict], base_runs: List[Dict],
                 key: str = "r2_pk_csc") -> Dict:
    """Paired per-seed Δ stat for `var - base`."""
    n = min(len(var_runs), len(base_runs))
    d = np.array([var_runs[i][key] - base_runs[i][key] for i in range(n)])
    m = float(d.mean())
    s = float(d.std(ddof=1)) if n > 1 else 0.0
    t = m / (s / np.sqrt(n)) if n > 1 and s > 1e-9 else float("inf")
    return {"mean": m, "std": s, "t": t, "n": n,
            "sig": (n > 1 and abs(t) > 2.0)}
