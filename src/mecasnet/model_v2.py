"""Legacy recurrent rollout and graph-free/graph-based comparators.

These implementations are retained for matched-protocol baselines and archived
checkpoint compatibility. New MeCaSNet models should be constructed through
the public package factory.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .config import Config
from .model import scatter_logsumexp  # reuse stable scatter ops


# ---------------------------------------------------------------------------
# NodeEncoder — same shape contract as v1 (static + shock + δ₀ + event scalars)
# ---------------------------------------------------------------------------
class NodeEncoderV2(nn.Module):
    EVT_DIM = 12

    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 2 + self.EVT_DIM, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

    def forward(self, x_v, shock_mask, delta0, event_scalars):
        Nr = x_v.shape[0]
        evt = event_scalars.view(1, -1).expand(Nr, -1)
        feat = torch.cat([x_v, shock_mask.unsqueeze(-1), delta0.unsqueeze(-1), evt], dim=-1)
        return self.net(feat)


# ---------------------------------------------------------------------------
# GRCSStep — one discrete-time GAT + GRU update producing (h_{t+1}, u_{t+1})
# ---------------------------------------------------------------------------
class GRCSStep(nn.Module):
    """One recurrent step.

    Components (all neural):
      forward_msg   : edge MLP over [h_src, h_dst, u_src, log a_rv] → m_e
      attention     : softmax over suppliers per dst node            → α_e
      msg_in_v      : Σ_e α_e · m_e                                  (GAT-like)
      msg_out_v     : index_add Σ_{e: src=v} g(h_dst, u_dst)          (reverse sum)
      gru           : h_v ← GRU(h_v, [msg_in, msg_out, u_v, shock_v, δ_t,v])
      du_head       : du_v = tanh(W·h_new) · STEP_BOUND
      u_v           ← clamp(u_v + du_v, 0, 1)
    """

    STEP_BOUND = 0.10  # max |Δu| per recurrent step

    def __init__(self, d: int):
        super().__init__()
        # forward (supplier→consumer) edge feature: h_src, h_dst, u_src, log_a
        self.fwd_msg = nn.Sequential(
            nn.Linear(2 * d + 2, d), nn.GELU(),
            nn.Linear(d, d),
        )
        self.fwd_score = nn.Linear(d, 1)
        # reverse (consumer→supplier) edge feature: h_dst, u_dst
        self.rev_msg = nn.Sequential(
            nn.Linear(d + 1, d), nn.GELU(),
        )
        # GRU input: msg_in (d), msg_out (d), u (1), shock (1), δ_t (1)
        self.gru = nn.GRUCell(2 * d + 3, d)
        # bounded Δu head
        self.du_head = nn.Linear(d, 1)
        # init du_head bias to 0 (no spurious drift at init)
        with torch.no_grad():
            self.du_head.bias.zero_()
            self.du_head.weight.mul_(0.1)

    def forward(self, h, u, shock, delta_t,
                edge_src, edge_dst, edge_a, Nr):
        """h: (Nr,d)  u: (Nr,)  shock: (Nr,)  delta_t: (Nr,)
        edge_src/dst: (E,) edge_a: (E,) edge weight a_{r→v}
        """
        # ---- forward message pass: supplier → consumer with attention ----
        log_a = edge_a.clamp_min(1e-4).log().unsqueeze(-1)        # (E,1)
        ef = torch.cat([h[edge_src], h[edge_dst],
                        u[edge_src].unsqueeze(-1), log_a], dim=-1)
        m = self.fwd_msg(ef)                                       # (E,d)
        s = self.fwd_score(m).squeeze(-1)                          # (E,)
        # softmax over suppliers per dst (numerically stable via logsumexp)
        log_norm = scatter_logsumexp(s, edge_dst, Nr)              # (Nr,)
        # nodes with no inbound edge → log_norm == 0 (neutral); avoid 0-div
        alpha = (s - log_norm[edge_dst]).exp().unsqueeze(-1)       # (E,1)
        msg_in = torch.zeros(Nr, m.shape[-1], device=h.device, dtype=h.dtype)
        msg_in.index_add_(0, edge_dst, alpha * m)

        # ---- reverse message pass: consumer → supplier sum ----
        rev_in = torch.cat([h[edge_dst], u[edge_dst].unsqueeze(-1)], dim=-1)
        rm = self.rev_msg(rev_in)                                  # (E,d)
        msg_out = torch.zeros_like(h).index_add_(0, edge_src, rm)

        # ---- GRU update ----
        gru_in = torch.cat([
            msg_in, msg_out,
            u.unsqueeze(-1), shock.unsqueeze(-1), delta_t.unsqueeze(-1),
        ], dim=-1)                                                 # (Nr, 2d+3)
        h_new = self.gru(gru_in, h)

        # ---- bounded du ----
        du = torch.tanh(self.du_head(h_new).squeeze(-1)) * self.STEP_BOUND
        u_new = (u + du).clamp(0.0, 1.0)
        return h_new, u_new


# ---------------------------------------------------------------------------
# GRCSRollout — runs T_steps, produces u_full + final h
# ---------------------------------------------------------------------------
class GRCSRollout(nn.Module):
    """Discrete-time recurrent cascade rollout.

    The integration grid uses fixed `stride_days` to keep memory bounded.
    For horizon=200 with stride=5 → 40 steps.  Key-day readouts use nearest
    step index (good enough for losses given Huber δ=0.1).
    """

    def __init__(self, d: int, horizon: int, stride_days: int = 5,
                 use_decay_signal: bool = True):
        super().__init__()
        self.step = GRCSStep(d)
        self.horizon = horizon
        self.stride = stride_days
        self.T_steps = max(1, horizon // stride_days)
        self.use_decay_signal = use_decay_signal
        # per-node recovery time T_r predicted from h₀: softplus·50 + 30 ∈ [30, ∞)
        # (matches v1 InventoryODE init range — mean ≈ 130 d, Inoue: 150)
        self.t_r_head = nn.Linear(d, 1)
        with torch.no_grad():
            # softplus(0) ≈ 0.69 → T_r ≈ 30 + 0.69·50 ≈ 65 d (close to learned-global ≈66)
            self.t_r_head.bias.zero_()
            self.t_r_head.weight.mul_(0.1)

    def forward(self, h0, shock, delta0,
                edge_src, edge_dst, edge_a,
                key_days):
        Nr = h0.shape[0]
        # T_r ∈ [30, ~∞) per node
        T_r = F.softplus(self.t_r_head(h0).squeeze(-1)) * 50.0 + 30.0   # (Nr,)

        # initial state
        u = (1.0 - delta0).clamp(0.0, 1.0)
        h = h0
        u_full = []
        for k in range(1, self.T_steps + 1):
            t_day = k * self.stride
            if self.use_decay_signal:
                delta_t = delta0 * torch.exp(-t_day / T_r)
            else:
                delta_t = torch.zeros_like(delta0)
            h, u = self.step(h, u, shock, delta_t,
                             edge_src, edge_dst, edge_a, Nr)
            u_full.append(u)
        u_full_t = torch.stack(u_full, dim=0)                 # (T_steps, Nr)

        # ---- key_days readout: pick nearest step ----
        u_kf = []
        T = u_full_t.shape[0]
        for kd in key_days.tolist():
            if kd == 0:
                u_kf.append((1.0 - delta0).clamp(0.0, 1.0))
            else:
                idx = max(0, min(T - 1, round(kd / self.stride) - 1))
                u_kf.append(u_full_t[idx])
        u_kf = torch.stack(u_kf, dim=0)                       # (K, Nr)

        # stub params tuple for loss_mono / loss_mass compatibility
        # (tau, N_inv, c, tau_u, T_r) — only T_r is real; others are sane defaults
        ones = torch.ones(Nr, device=h0.device, dtype=h0.dtype)
        params = (ones * 6.0, ones, ones * 0.5, ones * 1.0, T_r)
        return dict(u_full=u_full_t, u_keyframes=u_kf, params=params,
                    h_final=h)


# ---------------------------------------------------------------------------
# Decoder v2 — minimal: only reach classifier (peak is closed-form on neural u)
# ---------------------------------------------------------------------------
class DecoderV2(nn.Module):
    """Decoder with three peak modes (path β rescue: 2026-05-21).

        peak_mode="traj"   : peak = 1 - min_t u_t                    (original v2)
                             pure rollout — peak gradient must travel through
                             40 GRU steps; verified to fail in 5-epoch short
                             training (R²pk=−2.27 in ablate_v2 run).

        peak_mode="direct" : peak = sigmoid(MLP(h_final, traj_summary))
                             rollout still runs to produce u_keyframes (kf
                             loss supervises it), but peak takes a 1-hop
                             gradient path. Cleanest test of "are 40-step
                             recurrent features richer than single-pass GAT?"

        peak_mode="blend"  : peak = λ·peak_traj + (1−λ)·peak_direct
                             learnable scalar λ ∈ (0,1), init at 0.1 (favor
                             direct early; let the model up-weight traj if
                             rollout proves useful). DEFAULT.

    Note: this is NOT v1's rejected "phys + neural" blend. Both branches are
    fully neural. The blend only chooses between two neural readout paths.
    """

    TRAJ_DIM = 6

    def __init__(self, d: int, stride_days: int = 5,
                 peak_mode: str = "blend"):
        super().__init__()
        assert peak_mode in ("traj", "direct", "blend"), peak_mode
        self.peak_mode = peak_mode
        self.stride_days = stride_days
        # reach head — always present
        self.reach_head = nn.Sequential(
            nn.Linear(d + self.TRAJ_DIM, d), nn.GELU(),
            nn.Linear(d, 1),
        )
        # direct peak head — used by direct/blend modes
        self.peak_direct_head = nn.Sequential(
            nn.Linear(d + self.TRAJ_DIM, d), nn.GELU(),
            nn.Linear(d, 1),
        )
        with torch.no_grad():
            # init bias so sigmoid output ≈ 0.05 (matches gt_peak mean per
            # CASCADE_DATA_SPEC §C: 75.6% nodes have peak < 0.05)
            self.peak_direct_head[-1].bias.fill_(math.log(0.05 / 0.95))
        # blend λ — init logit(0.1) ≈ -2.197 → favor direct strongly at start
        self.blend_logit = nn.Parameter(torch.tensor(-2.197))

    def forward(self, h_final, u_full):
        T = u_full.shape[0]
        min_u = u_full.min(dim=0).values
        argmin_t = u_full.argmin(dim=0).float() / max(T - 1, 1)
        # Day-5 snapshot: with stride=5 → index 0 (end of step 1).
        day5_idx = max(0, min(round(5 / self.stride_days) - 1, T - 1))
        u_d5 = u_full[day5_idx]
        traj = torch.stack([
            min_u,
            u_full.mean(dim=0),
            u_full[-1],
            argmin_t,
            u_d5,
            u_full.std(dim=0),
        ], dim=-1)                                            # (Nr, 6)
        cat = torch.cat([h_final, traj], dim=-1)

        peak_traj = (1.0 - min_u).clamp(0.0, 1.0)
        peak_direct = torch.sigmoid(
            self.peak_direct_head(cat).squeeze(-1)
        )

        if self.peak_mode == "traj":
            peak = peak_traj
            lam = torch.tensor(1.0, device=peak.device)
        elif self.peak_mode == "direct":
            peak = peak_direct
            lam = torch.tensor(0.0, device=peak.device)
        else:  # blend
            lam = torch.sigmoid(self.blend_logit)
            peak = (lam * peak_traj + (1.0 - lam) * peak_direct).clamp(0.0, 1.0)

        reach_logit = self.reach_head(cat).squeeze(-1)
        return dict(
            peak=peak,
            peak_traj=peak_traj,                               # diagnostic
            peak_direct=peak_direct,                           # diagnostic
            peak_pred=peak,                                    # alias for ablation patches
            peak_phys=peak,                                    # alias (loss compat)
            peak_phys_raw=peak_traj,
            reach_logit=reach_logit,
            blend_lambda=lam.detach() if torch.is_tensor(lam) else lam,
            phys_scale=torch.ones_like(peak),                  # stub
        )


# ---------------------------------------------------------------------------
# Top-level v2 model
# ---------------------------------------------------------------------------
class PIDeepLeontiefV2(nn.Module):
    """Graph Recurrent Cascade Simulator.

    Args mirror v1 PIDeepLeontief so train.py can swap models with one line.
    """

    def __init__(self, cfg: Config, Fv: int, stride_days: int = 5,
                 use_decay_signal: bool = True,
                 peak_mode: str = "blend"):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_hidden
        self.encoder = NodeEncoderV2(Fv, d)
        self.rollout = GRCSRollout(
            d, horizon=cfg.horizon, stride_days=stride_days,
            use_decay_signal=use_decay_signal,
        )
        self.decoder = DecoderV2(d, stride_days=stride_days, peak_mode=peak_mode)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        h0 = self.encoder(batch["x_v"], batch["shock_mask"], batch["delta0"],
                          batch["event_scalars"])
        roll = self.rollout(h0, batch["shock_mask"], batch["delta0"],
                            batch["edge_src"], batch["edge_dst"], batch["edge_a"],
                            batch["key_days"])
        dec = self.decoder(roll["h_final"], roll["u_full"])
        return {
            **roll, **dec,
            "h_star": roll["h_final"],     # alias for v1 callers
        }

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# Plain-GAT baseline (variant E for ablate_v2_check.py)
# ---------------------------------------------------------------------------
# Single-pass message-passing model. NO time dimension, NO rollout. Predicts
# peak / reach / u_keyframes directly from h after L GAT layers. This is the
# strongest "is the trajectory rollout actually useful?" baseline:
#   - if v2 (full rollout) does NOT beat this on cascade R², the trajectory
#     model is not earning its compute and we should retreat to plain GAT.
# ===========================================================================
class GATLayer(nn.Module):
    """Single-head GAT layer (Velickovic et al., 2018).

    Aggregates supplier→consumer messages with softmax attention over each
    consumer's incoming edges.  Edge weight a_{rv} (Leontief input share)
    is fed as a log-feature into the attention score MLP, mirroring the v2
    GRCSStep so the comparison is fair (same edge information available).
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W = nn.Linear(d_in, d_out)
        # attention: a^T [W h_src || W h_dst || log_a]  →  scalar
        self.attn = nn.Linear(2 * d_out + 1, 1)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, h, edge_src, edge_dst, edge_a, Nr):
        Wh = self.W(h)                                              # (Nr, d_out)
        log_a = edge_a.clamp_min(1e-4).log().unsqueeze(-1)          # (E, 1)
        e_in = torch.cat([Wh[edge_src], Wh[edge_dst], log_a], dim=-1)
        e = F.leaky_relu(self.attn(e_in).squeeze(-1), 0.2)           # (E,)
        # softmax over suppliers per dst (numerically stable)
        log_norm = scatter_logsumexp(e, edge_dst, Nr)                # (Nr,)
        alpha = (e - log_norm[edge_dst]).exp().unsqueeze(-1)         # (E, 1)
        msg = alpha * Wh[edge_src]                                   # (E, d_out)
        agg = torch.zeros(Nr, Wh.shape[-1], device=h.device, dtype=h.dtype)
        agg.index_add_(0, edge_dst, msg)
        # residual + norm + GELU
        h_new = self.norm(Wh + agg)
        return F.gelu(h_new)


class PlainGATBaseline(nn.Module):
    """Single-pass GAT predictor.

    Architecture:
        Encoder(static, shock, δ₀, evt) → h₀
        for L layers: h ← GATLayer(h, edges)
        peak = sigmoid(MLP_peak(h))                ∈ (0,1) per node
        u_kf = sigmoid(MLP_kf(h))   reshaped to (K, Nr)
        reach_logit = MLP_reach(h)

    Output dict matches PIDeepLeontiefV2 so losses.py / train.py work unchanged.
    """

    def __init__(self, cfg: Config, Fv: int, n_layers: int = 3, K: int = 10,
                 d_hidden: int | None = None):
        super().__init__()
        self.cfg = cfg
        d = int(d_hidden) if d_hidden is not None else cfg.d_hidden
        self.K = K
        self.encoder = NodeEncoderV2(Fv, d)
        self.gat_layers = nn.ModuleList([GATLayer(d, d) for _ in range(n_layers)])
        # heads
        self.peak_head = nn.Linear(d, 1)
        with torch.no_grad():
            self.peak_head.bias.fill_(math.log(0.05 / 0.95))   # init ≈ 0.05
        self.kf_head = nn.Linear(d, K)
        with torch.no_grad():
            self.kf_head.bias.fill_(math.log(0.95 / 0.05))     # init kf ≈ 0.95 (most nodes ≈ 1)
        self.reach_head = nn.Linear(d, 1)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        h = self.encoder(batch["x_v"], batch["shock_mask"], batch["delta0"],
                         batch["event_scalars"])
        Nr = h.shape[0]
        for layer in self.gat_layers:
            h = layer(h, batch["edge_src"], batch["edge_dst"],
                      batch["edge_a"], Nr)
        peak = torch.sigmoid(self.peak_head(h).squeeze(-1))           # (Nr,)
        # kf head outputs K values per node; transpose to (K, Nr) to match data
        u_kf_logits = self.kf_head(h)                                  # (Nr, K)
        u_kf = torch.sigmoid(u_kf_logits).transpose(0, 1).contiguous() # (K, Nr)
        reach_logit = self.reach_head(h).squeeze(-1)                   # (Nr,)
        # u_full stub for traj-summary tools (use kf as the trajectory)
        # not used in losses, but train.evaluate's some metrics may peek
        u_full = u_kf
        # params stub (loss_mono / loss_mass with default weights ignore this)
        ones = torch.ones(Nr, device=h.device, dtype=h.dtype)
        params = (ones * 6.0, ones, ones * 0.5, ones * 1.0, ones * 150.0)
        return dict(
            peak=peak, peak_pred=peak, peak_phys=peak, peak_phys_raw=peak,
            u_keyframes=u_kf, u_full=u_full, params=params,
            reach_logit=reach_logit, h_star=h, h_final=h,
            blend_lambda=torch.zeros((), device=peak.device),
            phys_scale=torch.ones_like(peak),
        )

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PlainMLPBaseline(nn.Module):
    """Per-node MLP baseline (NO graph aggregation).

    Tests how much of the cascade prediction signal can be recovered from
    purely-local node features (static features + shock indicator + delta0
    + event scalars).  Any GNN variant should beat this if topology helps.
    """

    def __init__(self, cfg: Config, Fv: int, n_layers: int = 3, K: int = 10,
                 d_hidden: int | None = None):
        super().__init__()
        self.cfg = cfg
        d = int(d_hidden) if d_hidden is not None else cfg.d_hidden
        self.K = K
        self.encoder = NodeEncoderV2(Fv, d)
        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d)]
        self.mlp = nn.Sequential(*layers)
        self.peak_head = nn.Linear(d, 1)
        with torch.no_grad():
            self.peak_head.bias.fill_(math.log(0.05 / 0.95))
        self.kf_head = nn.Linear(d, K)
        with torch.no_grad():
            self.kf_head.bias.fill_(math.log(0.95 / 0.05))
        self.reach_head = nn.Linear(d, 1)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        h = self.encoder(batch["x_v"], batch["shock_mask"], batch["delta0"],
                         batch["event_scalars"])
        h = self.mlp(h)
        Nr = h.shape[0]
        peak = torch.sigmoid(self.peak_head(h).squeeze(-1))
        u_kf = torch.sigmoid(self.kf_head(h)).transpose(0, 1).contiguous()
        reach_logit = self.reach_head(h).squeeze(-1)
        u_full = u_kf
        ones = torch.ones(Nr, device=h.device, dtype=h.dtype)
        params = (ones * 6.0, ones, ones * 0.5, ones * 1.0, ones * 150.0)
        return dict(
            peak=peak, peak_pred=peak, peak_phys=peak, peak_phys_raw=peak,
            u_keyframes=u_kf, u_full=u_full, params=params,
            reach_logit=reach_logit, h_star=h, h_final=h,
            blend_lambda=torch.zeros((), device=peak.device),
            phys_scale=torch.ones_like(peak),
        )

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Canonical GCN baseline (Kipf & Welling 2017, adapted for directed weighted)
# ---------------------------------------------------------------------------
class GCNLayer(nn.Module):
    """Symmetric-normalized graph convolution with edge weights.

        h_v^{l+1} = GELU(LN(W h_v + W * sum_{u->v} (a_{uv} / sqrt(d_out(u)*d_in(v))) * h_u))

    Uses the same edge_a (Leontief input share) as GATLayer, so the edge
    information available to each baseline is identical; only the
    aggregation rule differs (fixed degree-normalised weights vs learned
    attention).
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, h, edge_src, edge_dst, edge_a, Nr):
        Wh = self.W(h)                                              # (Nr, d_out)
        deg_out = torch.zeros(Nr, device=h.device, dtype=h.dtype)
        deg_in = torch.zeros_like(deg_out)
        deg_out.index_add_(0, edge_src, edge_a)
        deg_in.index_add_(0, edge_dst, edge_a)
        norm_w = edge_a / (deg_out[edge_src].clamp_min(1e-6).sqrt() *
                            deg_in[edge_dst].clamp_min(1e-6).sqrt())
        msg = norm_w.unsqueeze(-1) * Wh[edge_src]                   # (E, d_out)
        agg = torch.zeros_like(Wh)
        agg.index_add_(0, edge_dst, msg)
        h_new = self.norm(Wh + agg)
        return F.gelu(h_new)


class PlainGCNBaseline(nn.Module):
    """Single-pass GCN predictor (Kipf-Welling-style).

    Same encoder + head structure as PlainGATBaseline; differs only in the
    aggregation rule (GCNLayer instead of GATLayer).  Provides a fair
    "GAT vs GCN" baseline pair under identical training conditions.
    """

    def __init__(self, cfg: Config, Fv: int, n_layers: int = 3, K: int = 10,
                 d_hidden: int | None = None):
        super().__init__()
        self.cfg = cfg
        d = int(d_hidden) if d_hidden is not None else cfg.d_hidden
        self.K = K
        self.encoder = NodeEncoderV2(Fv, d)
        self.gcn_layers = nn.ModuleList([GCNLayer(d, d) for _ in range(n_layers)])
        self.peak_head = nn.Linear(d, 1)
        with torch.no_grad():
            self.peak_head.bias.fill_(math.log(0.05 / 0.95))
        self.kf_head = nn.Linear(d, K)
        with torch.no_grad():
            self.kf_head.bias.fill_(math.log(0.95 / 0.05))
        self.reach_head = nn.Linear(d, 1)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        h = self.encoder(batch["x_v"], batch["shock_mask"], batch["delta0"],
                         batch["event_scalars"])
        Nr = h.shape[0]
        for layer in self.gcn_layers:
            h = layer(h, batch["edge_src"], batch["edge_dst"],
                      batch["edge_a"], Nr)
        peak = torch.sigmoid(self.peak_head(h).squeeze(-1))
        u_kf = torch.sigmoid(self.kf_head(h)).transpose(0, 1).contiguous()
        reach_logit = self.reach_head(h).squeeze(-1)
        u_full = u_kf
        ones = torch.ones(Nr, device=h.device, dtype=h.dtype)
        params = (ones * 6.0, ones, ones * 0.5, ones * 1.0, ones * 150.0)
        return dict(
            peak=peak, peak_pred=peak, peak_phys=peak, peak_phys_raw=peak,
            u_keyframes=u_kf, u_full=u_full, params=params,
            reach_logit=reach_logit, h_star=h, h_final=h,
            blend_lambda=torch.zeros((), device=peak.device),
            phys_scale=torch.ones_like(peak),
        )

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
