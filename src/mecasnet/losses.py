"""Loss functions for MeCaSNet.

Total loss = L_data (focal-weighted) + L_mono + L_mass [+ L_phys, L_risk]

L_data : Huber on u_keyframes + peak_loss, with focal weighting
         - directly shocked nodes (shock_mask=1) ×5
         - inert nodes (label u≈1) ×0.1
         - everything else ×1
L_mono : penalise du/dt above a cap (recovery rate ceiling)
L_mass : ∑_v cap_v ≥ ∑_v consumption_v (mass conservation slack)
L_phys : Inoue residual on neural-correction term (drives correction → 0 unless needed)
L_risk : CVaR head supervision (v2; off by default)
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Dict
from .config import Config


def focal_weights(shock_mask: torch.Tensor, peak_target: torch.Tensor,
                  cfg: Config) -> torch.Tensor:
    """Per-node continuous focal weights for data loss.  shape (Nr,)

    w_v = inert_w + shock_w * peak_loss_gt[v]               (severity-proportional)
    With a floor for directly-shocked nodes so they cannot drop below
    0.5 * shock_w even if their peak happens to be small.

    Peak-loss targets are heavy-tailed, with many near-zero nodes. Continuous
    weighting gives each node a learning signal proportional to target severity
    while retaining a floor for directly shocked nodes.
    """
    Nr = shock_mask.shape[0]
    inert_w = cfg.focal_inert_weight
    shock_w = cfg.focal_shock_weight
    pk = peak_target.clamp(0.0, 1.0)
    w = inert_w + shock_w * pk
    # safety floor for direct hits (rare: shocked node with tiny peak)
    direct_floor = torch.full_like(w, 0.5 * shock_w)
    w = torch.where(shock_mask > 0.5, torch.maximum(w, direct_floor), w)
    return w


def loss_data(pred_kf: torch.Tensor, gt_kf: torch.Tensor,
              pred_peak: torch.Tensor, gt_peak: torch.Tensor,
              shock_mask: torch.Tensor, cfg: Config,
              domain: str | None = None) -> Dict[str, torch.Tensor]:
    # weight by ground-truth peak severity (severity-proportional focal)
    w = focal_weights(shock_mask, gt_peak, cfg)                 # (Nr,)
    # keyframe loss (K, Nr)
    err_kf = F.huber_loss(pred_kf, gt_kf, reduction="none", delta=0.1)
    # per-keyframe day weights — domain-specific override if available.
    K = err_kf.shape[0]
    kfw_by_dom = getattr(cfg, "kf_weights_by_domain", {}) or {}
    if domain is not None and domain in kfw_by_dom:
        kfw_list = list(kfw_by_dom[domain])
    elif hasattr(cfg, "kf_weights"):
        kfw_list = list(cfg.kf_weights)
    else:
        kfw_list = [1.0] * K
    if len(kfw_list) != K:
        kfw_list = [1.0] * K
    kf_w = torch.tensor(kfw_list, dtype=err_kf.dtype, device=err_kf.device)
    err_kf = err_kf * w.unsqueeze(0) * kf_w.unsqueeze(1)
    err_kf = err_kf.mean()
    # peak loss
    err_pk = F.huber_loss(pred_peak, gt_peak, reduction="none", delta=0.1)
    err_pk = (err_pk * w).mean()
    return dict(
        l_kf=cfg.w_data_keyframes * err_kf,
        l_peak=cfg.w_data_peak * err_pk,
    )


def loss_reach(reach_logit: torch.Tensor, is_reach: torch.Tensor,
               cfg: Config) -> torch.Tensor:
    """BCE on reachability label (peak > reach_thresh).
    Helps the model exploit the 49% strict-zero structure (DATA_CHARACTERISTICS §5.3).
    """
    if cfg.w_reach == 0:
        return torch.zeros((), device=reach_logit.device)
    # class-balanced pos_weight: pos:neg roughly 51:49, so weight ≈ 1
    n_pos = is_reach.sum().clamp_min(1.0)
    n_neg = (1.0 - is_reach).sum().clamp_min(1.0)
    pos_w = (n_neg / n_pos).clamp(0.1, 10.0)
    return cfg.w_reach * F.binary_cross_entropy_with_logits(
        reach_logit, is_reach, pos_weight=pos_w
    )


def loss_trough_day(trough_day_logit: torch.Tensor,
                    trough_day_target: torch.Tensor,
                    shock_mask: torch.Tensor,
                    peak_loss: torch.Tensor,
                    cfg: Config) -> torch.Tensor:
    """Cross-entropy on per-node trough_day (which keyframe is u-trough).

    Active only on cascade nodes (peak > trough_day_peak_thresh, not shocked).
    This auxiliary objective teaches the encoder when each node reaches its
    trough, improving temporal identifiability around d30/d50.
    """
    w = float(getattr(cfg, "w_trough_day", 0.0))
    if w == 0.0:
        return torch.zeros((), device=trough_day_logit.device)
    thresh = float(getattr(cfg, "trough_day_peak_thresh", 0.05))
    mask = (shock_mask < 0.5) & (peak_loss > thresh)            # (Nr,)
    n = int(mask.sum().item())
    if n == 0:
        return torch.zeros((), device=trough_day_logit.device)
    logits = trough_day_logit[mask]                              # (n, K)
    target = trough_day_target[mask].long()                      # (n,)
    return w * F.cross_entropy(logits, target)


def _kf_to_regime(kf_idx: torch.Tensor, key_days_list) -> torch.Tensor:
    """Map keyframe index → 3-way regime bucket.
        regime 0 = early   (key_day <= 20)
        regime 1 = mid     (30 <= key_day <= 70)
        regime 2 = late    (key_day >= 100)
    """
    boundaries = torch.tensor(
        [(0 if d <= 20 else (1 if d <= 70 else 2)) for d in key_days_list],
        device=kf_idx.device, dtype=torch.long,
    )
    return boundaries[kf_idx.long()]


def loss_regime_router(route_logits: torch.Tensor,
                       u_kf_gt: torch.Tensor,
                       shock_mask: torch.Tensor,
                       peak_loss: torch.Tensor,
                       key_days,
                       cfg: Config) -> torch.Tensor:
    """CE on per-node 3-way regime (early/mid/late) routing for MoE param head.
    GT regime = bucket(argmin(u_keyframes_gt)) over cascade-pool nodes.
    Plus load-balance term to prevent any regime from being starved.
    """
    w = float(getattr(cfg, "w_regime_router", 0.0))
    if w == 0.0:
        return torch.zeros((), device=route_logits.device)
    thresh = float(getattr(cfg, "trough_day_peak_thresh", 0.05))
    mask = (shock_mask < 0.5) & (peak_loss > thresh)
    n = int(mask.sum().item())
    if n == 0:
        return torch.zeros((), device=route_logits.device)
    # u_kf_gt is (K, Nr); per-node trough kf = argmin over K
    gt_kf = torch.argmin(u_kf_gt, dim=0)                          # (Nr,)
    if isinstance(key_days, torch.Tensor):
        kd_list = key_days.detach().cpu().tolist()
    else:
        kd_list = list(key_days)
    gt_regime = _kf_to_regime(gt_kf, kd_list)                     # (Nr,)
    ce = F.cross_entropy(route_logits[mask], gt_regime[mask])
    # load balance: encourage non-degenerate route distribution on this batch
    probs = torch.softmax(route_logits[mask], dim=-1).mean(dim=0)  # (3,)
    lb_w = float(getattr(cfg, "w_regime_loadbal", 0.05))
    # KL(uniform || mean_probs) — high when probs concentrate
    lb = -(torch.log(torch.tensor(3.0, device=probs.device))
           + (probs.clamp_min(1e-9).log()).mean())
    return w * ce + lb_w * lb


def loss_mono(u_full: torch.Tensor, params: tuple, cfg: Config,
              key_days: torch.Tensor | None = None) -> torch.Tensor:
    """Cap dot{u}: forbid recovery faster than 1/τ_u(v) per day.

    u_full is aliased to u_keyframes (K=10 frames at variable Δt). The per-keyframe
    raw Δu must be divided by Δt_days before comparing to the per-day cap 1/τ_u,
    otherwise a 5-day window's accumulated rise is wrongly compared to a 1-day rate
    and the constraint is silently never triggered (mono_viol=0 forever).
    """
    if cfg.w_mono == 0:
        return torch.zeros((), device=u_full.device)
    tau_u = params[2].clamp_min(0.5)                             # (Nr,)
    du = u_full[1:] - u_full[:-1]                                # (K-1, Nr)
    if key_days is not None:
        kd = key_days if torch.is_tensor(key_days) else torch.tensor(
            list(key_days), device=u_full.device, dtype=u_full.dtype)
        kd = kd.to(device=u_full.device, dtype=u_full.dtype)
        dt = (kd[1:] - kd[:-1]).clamp_min(1.0).unsqueeze(1)      # (K-1, 1)
        du = du / dt                                             # per-day rate
    cap = (1.0 / tau_u).unsqueeze(0)                             # (1, Nr)
    excess = (du - cap).clamp_min(0.0)
    return cfg.w_mono * excess.pow(2).mean()


def loss_mono_late(u_kf: torch.Tensor, key_days: torch.Tensor,
                   cfg: Config) -> torch.Tensor:
    """Penalise u decreasing between consecutive keyframes after the late
    threshold (default day 70). Physical invariant: after the initial-damage
    + early-cascade window has passed, production should be non-decreasing
    on its way back to steady state. Holds for both Inoue (monotone tails)
    and Henriet (recovery overshoot is >=1 not <previous)."""
    if getattr(cfg, "w_mono_late", 0.0) == 0.0:
        return torch.zeros((), device=u_kf.device)
    K = u_kf.shape[0]
    if K < 2:
        return torch.zeros((), device=u_kf.device)
    if torch.is_tensor(key_days):
        kd = key_days.detach().cpu().tolist()
    else:
        kd = list(key_days)
    thresh = getattr(cfg, "mono_late_threshold_day", 70)
    # iterate consecutive keyframe pairs where the EARLIER day >= thresh
    contribs = []
    for k in range(K - 1):
        if kd[k] >= thresh:
            dt = u_kf[k + 1] - u_kf[k]                # (Nr,)
            contribs.append(F.relu(-dt).pow(2).mean())
    if not contribs:
        return torch.zeros((), device=u_kf.device)
    return cfg.w_mono_late * torch.stack(contribs).mean()


def loss_mass(u_full: torch.Tensor, params: tuple,
              edge_src: torch.Tensor, edge_dst: torch.Tensor,
              edge_a: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Soft mass conservation: total arrivals ≥ total consumption (slack penalty)."""
    if cfg.w_mass == 0:
        return torch.zeros((), device=u_full.device)
    # use steady-state snapshot (mean over trajectory)
    u_mean = u_full.mean(dim=0)                                  # (Nr,)
    c = params[3]                                                # (Nr,) - learned from graph stats
    consumption = (c * u_mean).sum()
    # arrivals = sum over edges of u_src / a (raw, not min — for mass aggregate)
    arrivals = (u_mean[edge_src] / edge_a.clamp_min(1e-3)).sum()
    deficit = (consumption - arrivals).clamp_min(0.0)
    return cfg.w_mass * deficit / (consumption + 1e-6)


def loss_phys(model, out: Dict, cfg: Config) -> torch.Tensor:
    """L2 penalty on the neural-correction magnitude (PINN-style prior on physics).
    Effectively: 'use physics unless data forces a correction'."""
    if cfg.w_phys == 0 or not cfg.use_neural_correction:
        return torch.zeros((), device=out["u_full"].device)
    # correction is computed inside ODE; cheap proxy: penalise final-state deviation
    # from pure-physics one-shot (we recompute physics-only u≈1-delta0 baseline)
    # Skip exact recomputation here; placeholder zero. (Implement if needed.)
    return torch.zeros((), device=out["u_full"].device)


def total_loss(out: Dict, batch: Dict, cfg: Config) -> Dict[str, torch.Tensor]:
    parts = {}
    domain = batch.get("domain", None)
    d = loss_data(out["u_keyframes"], batch["u_keyframes"],
                  out["peak"], batch["peak_loss"],
                  batch["shock_mask"], cfg, domain=domain)
    parts.update(d)
    parts["l_mono"] = loss_mono(out["u_full"], out["params"], cfg,
                                key_days=batch.get("key_days"))
    parts["l_mono_late"] = loss_mono_late(out["u_keyframes"],
                                          batch["key_days"], cfg)
    parts["l_mass"] = loss_mass(out["u_full"], out["params"],
                                batch["edge_src"], batch["edge_dst"],
                                batch["edge_a"], cfg)
    # reach BCE aux (opt-in via cfg.w_reach>0; requires Decoder.reach_head & batch[is_reach])
    if cfg.w_reach > 0 and "reach_logit" in out and "is_reach" in batch:
        parts["l_reach"] = loss_reach(out["reach_logit"], batch["is_reach"], cfg)
    # cascade-depth aux (opt-in via cfg.w_trough_day>0)
    if (getattr(cfg, "w_trough_day", 0.0) > 0
            and "trough_day_logit" in out
            and "trough_day_target" in batch):
        parts["l_trough_day"] = loss_trough_day(
            out["trough_day_logit"], batch["trough_day_target"],
            batch["shock_mask"], batch["peak_loss"], cfg)
    # MoE regime router (opt-in via cfg.w_regime_router>0; requires
    # decoder_mode=param3 + moe_router=True)
    if (getattr(cfg, "w_regime_router", 0.0) > 0
            and "regime_route_logits" in out):
        parts["l_regime_router"] = loss_regime_router(
            out["regime_route_logits"], batch["u_keyframes"],
            batch["shock_mask"], batch["peak_loss"],
            batch.get("key_days"), cfg)
    raw_total = sum(parts.values())
    # ---- per-domain loss multiplier (event level) ----
    dlw = getattr(cfg, "domain_loss_weights", {}) or {}
    mult = float(dlw.get(domain, 1.0)) if domain is not None else 1.0
    parts["total"] = mult * raw_total

    # ---- learned-parameter monitoring outputs ----
    if "tau_u_learned" in out and "c_learned" in out:
        tau_u = out["tau_u_learned"]
        c = out["c_learned"]
        # Statistics for learned parameters
        parts["param_tau_u_mean"] = tau_u.mean()
        parts["param_tau_u_std"] = tau_u.std()
        parts["param_tau_u_min"] = tau_u.min()
        parts["param_tau_u_max"] = tau_u.max()
        parts["param_c_mean"] = c.mean()
        parts["param_c_std"] = c.std()
        # Check loss_mono constraint satisfaction (same time-norm as loss_mono)
        du = out["u_full"][1:] - out["u_full"][:-1]
        kd = batch.get("key_days")
        if kd is not None:
            kd_t = kd if torch.is_tensor(kd) else torch.tensor(
                list(kd), device=du.device, dtype=du.dtype)
            kd_t = kd_t.to(device=du.device, dtype=du.dtype)
            dt = (kd_t[1:] - kd_t[:-1]).clamp_min(1.0).unsqueeze(1)
            du = du / dt
        cap = (1.0 / tau_u.clamp_min(0.5)).unsqueeze(0)
        violations = (du > cap).sum().float()
        parts["param_mono_violations"] = violations / (du.numel() + 1e-6)

    return parts
