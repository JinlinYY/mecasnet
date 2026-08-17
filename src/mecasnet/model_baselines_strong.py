"""Strong comparator architectures used in the manuscript protocol.

The directed GNN, spatiotemporal GNN, and analytical physics comparator share
MeCaSNet's data and evaluation interfaces so they can be trained on the same
event splits and observable inputs.
"""
from __future__ import annotations
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .model_v2 import NodeEncoderV2, GCNLayer


# ---------------------------------------------------------------------------
# scatter helpers (degree-normalised sum / segment max-min)
# ---------------------------------------------------------------------------
def _scatter_sum(src: torch.Tensor, index: torch.Tensor, N: int) -> torch.Tensor:
    """src: (E, d) or (E,) -> out: (N, d) or (N,) summed over index."""
    if src.dim() == 1:
        out = torch.zeros(N, device=src.device, dtype=src.dtype)
    else:
        out = torch.zeros(N, src.shape[-1], device=src.device, dtype=src.dtype)
    return out.index_add_(0, index, src)


def _segment_reduce(values: torch.Tensor, index: torch.Tensor, N: int,
                    reduce: str, init: float) -> torch.Tensor:
    """1-D segment amax/amin with a constant initialisation (include_self)."""
    out = torch.full((N,), init, device=values.device, dtype=values.dtype)
    out.scatter_reduce_(0, index, values, reduce=reduce, include_self=True)
    return out


# ===========================================================================
# 1. Directed GNN (Dir-GNN, Rossi et al. 2023)
# ===========================================================================
class DirGNNLayer(nn.Module):
    """Direction-aware aggregation with separate in/out weight matrices.

        h_v^{l+1} = GELU(LN(
            W_self h_v
          + W_in  · mean_{u->v} a_{uv} h_u          (supplier -> consumer)
          + W_out · mean_{v->w} a_{vw} h_w ))        (consumer -> supplier)
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin_self = nn.Linear(d_in, d_out)
        self.lin_in = nn.Linear(d_in, d_out)
        self.lin_out = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, h, edge_src, edge_dst, edge_a, Nr):
        a = edge_a.unsqueeze(-1)
        # forward (in-edges of consumer): suppliers -> consumer (dst)
        deg_in = _scatter_sum(edge_a, edge_dst, Nr).clamp_min(1e-6)
        agg_in = _scatter_sum(a * h[edge_src], edge_dst, Nr) / deg_in.unsqueeze(-1)
        # reverse (out-edges of supplier): consumers -> supplier (src)
        deg_out = _scatter_sum(edge_a, edge_src, Nr).clamp_min(1e-6)
        agg_out = _scatter_sum(a * h[edge_dst], edge_src, Nr) / deg_out.unsqueeze(-1)
        h_new = self.lin_self(h) + self.lin_in(agg_in) + self.lin_out(agg_out)
        return F.gelu(self.norm(h_new))


class DirGNNBaseline(nn.Module):
    """Single-pass directed-GNN predictor (peak / u_keyframes / reach)."""

    def __init__(self, cfg: Config, Fv: int, n_layers: int = 6, K: int = 10,
                 d_hidden: int | None = None):
        super().__init__()
        self.cfg = cfg
        d = int(d_hidden) if d_hidden is not None else cfg.d_hidden
        self.K = K
        self.encoder = NodeEncoderV2(Fv, d)
        self.layers = nn.ModuleList([DirGNNLayer(d, d) for _ in range(n_layers)])
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
        for layer in self.layers:
            h = layer(h, batch["edge_src"], batch["edge_dst"],
                      batch["edge_a"], Nr)
        peak = torch.sigmoid(self.peak_head(h).squeeze(-1))
        u_kf = torch.sigmoid(self.kf_head(h)).transpose(0, 1).contiguous()
        reach_logit = self.reach_head(h).squeeze(-1)
        ones = torch.ones(Nr, device=h.device, dtype=h.dtype)
        params = (ones * 6.0, ones, ones * 0.5, ones * 1.0, ones * 150.0)
        return dict(
            peak=peak, peak_pred=peak, peak_phys=peak, peak_phys_raw=peak,
            u_keyframes=u_kf, u_full=u_kf, params=params,
            reach_logit=reach_logit, h_star=h, h_final=h,
            blend_lambda=torch.zeros((), device=peak.device),
            phys_scale=torch.ones_like(peak),
        )

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# 2. Spatio-temporal GNN (A3T-GCN / DCRNN style)
# ===========================================================================
class STGNNBaseline(nn.Module):
    """GCN spatial aggregation wrapped in a GRU, rolled out over key days.

    At each key-day step the model performs one GCN message pass to gather
    spatial context, then updates a per-node GRU state and reads out the
    production ratio u_k for that key day. The trajectory therefore emerges
    from a generic spatio-temporal recurrence, with NO analytical
    decline-trough-recovery prior (the MeCaSNet decoder ablation as a
    standalone competitor).
    """

    def __init__(self, cfg: Config, Fv: int, n_spatial: int = 2, K: int = 10,
                 d_hidden: int | None = None):
        super().__init__()
        self.cfg = cfg
        d = int(d_hidden) if d_hidden is not None else cfg.d_hidden
        self.K = K
        self.encoder = NodeEncoderV2(Fv, d)
        # stacked GCN for per-step spatial context
        self.spatial = nn.ModuleList([GCNLayer(d, d) for _ in range(n_spatial)])
        self.gru = nn.GRUCell(d, d)
        self.out_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        with torch.no_grad():
            self.out_head[-1].bias.fill_(math.log(0.95 / 0.05))  # u init ~0.95
        self.reach_head = nn.Linear(d, 1)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        h0 = self.encoder(batch["x_v"], batch["shock_mask"], batch["delta0"],
                          batch["event_scalars"])
        Nr = h0.shape[0]
        src, dst, a = batch["edge_src"], batch["edge_dst"], batch["edge_a"]
        state = h0
        us = []
        for _ in range(self.K):
            ctx = state
            for layer in self.spatial:
                ctx = layer(ctx, src, dst, a, Nr)
            state = self.gru(ctx, state)
            us.append(torch.sigmoid(self.out_head(state).squeeze(-1)))
        u_kf = torch.stack(us, dim=0)                              # (K, Nr)
        peak = (1.0 - u_kf.min(dim=0).values).clamp(0.0, 1.0)
        reach_logit = self.reach_head(state).squeeze(-1)
        ones = torch.ones(Nr, device=h0.device, dtype=h0.dtype)
        params = (ones * 6.0, ones, ones * 0.5, ones * 1.0, ones * 150.0)
        return dict(
            peak=peak, peak_pred=peak, peak_phys=peak, peak_phys_raw=peak,
            u_keyframes=u_kf, u_full=u_kf, params=params,
            reach_logit=reach_logit, h_star=state, h_final=state,
            blend_lambda=torch.zeros((), device=peak.device),
            phys_scale=torch.ones_like(peak),
        )

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# 3. Pure analytical physics (NO graph learning)
# ===========================================================================
class PhysicsAnalyticalBaseline(nn.Module):
    """Leontief-bottleneck peak propagation + global analytical trajectory.

    Mechanism (no neural network on the graph):
      * peak loss propagates over the directed graph by the Leontief min-rule:
        a consumer's loss is bounded below by the worst loss among its
        *essential* suppliers (input share a >= tau), attenuated by a learned
        transmission factor and an inventory buffer.
      * the per-node trajectory follows the SAME analytical decline-trough-
        recovery form as the MeCaSNet physics-prior decoder, but with a
        single GLOBAL set of shape parameters (trough day mu, decline width
        sigma, recovery time T_r, residual loss) rather than node-conditioned
        parameters predicted by a network.

    The model has only a handful of learnable global scalars, so it is a
    genuine "physics-only" arm for the fusion ablation.
    """

    def __init__(self, cfg: Config, Fv: int | None = None, K: int = 10,
                 n_hops: int = 6):
        super().__init__()
        self.cfg = cfg
        self.K = K
        self.n_hops = n_hops
        # learnable global physics scalars (raw, mapped through sigmoid/softplus)
        self.tau_raw = nn.Parameter(torch.tensor(-1.0))     # essential-input share thresh
        self.transmit_raw = nn.Parameter(torch.tensor(2.0))  # loss transmission factor
        self.buffer_raw = nn.Parameter(torch.tensor(-2.0))   # inventory buffer (abs loss)
        self.mu_raw = nn.Parameter(torch.tensor(0.0))        # trough day (-> [5, 60])
        self.sigma_raw = nn.Parameter(torch.tensor(0.0))     # decline width (-> [3, 40])
        self.tr_raw = nn.Parameter(torch.tensor(0.0))        # recovery T_r (-> [20, 200])
        self.resid_raw = nn.Parameter(torch.tensor(-2.0))    # residual loss fraction
        # 2-param affine readout so reach BCE is trainable from peak
        self.reach_w = nn.Parameter(torch.tensor(8.0))
        self.reach_b = nn.Parameter(torch.tensor(-2.0))

    def _propagate_peak(self, shock, delta0, edge_src, edge_dst, edge_a, Nr):
        tau = torch.sigmoid(self.tau_raw)                    # (0,1)
        transmit = torch.sigmoid(self.transmit_raw)          # (0,1)
        buffer = F.softplus(self.buffer_raw) * 0.5           # >=0 small
        essential = (edge_a >= tau).float()
        pk = (shock * delta0).clamp(0.0, 1.0)                # direct shock loss
        for _ in range(self.n_hops):
            # loss arriving at each consumer from its essential suppliers
            contrib = (pk[edge_src] * transmit - buffer).clamp_min(0.0) * essential
            inc = _segment_reduce(contrib, edge_dst, Nr, reduce="amax", init=0.0)
            pk = torch.maximum(pk, inc).clamp(0.0, 1.0)
        return pk

    def _trajectory(self, peak, key_days):
        mu = 5.0 + torch.sigmoid(self.mu_raw) * 55.0
        sigma = 3.0 + torch.sigmoid(self.sigma_raw) * 37.0
        Tr = 20.0 + torch.sigmoid(self.tr_raw) * 180.0
        resid = torch.sigmoid(self.resid_raw)               # residual fraction
        t = key_days.to(peak.dtype)                          # (K,)
        u_inf = 1.0 - peak.unsqueeze(0) * resid             # (1, Nr)
        p = peak.unsqueeze(0)                                # (1, Nr)
        tt = t.unsqueeze(1)                                  # (K, 1)
        decline = 1.0 - p * torch.exp(-(tt - mu) ** 2 / (2.0 * sigma ** 2))
        recover = u_inf + (1.0 - p - u_inf) * torch.exp(-(tt - mu) / Tr)
        u_kf = torch.where(tt <= mu, decline, recover).clamp(0.0, 1.0)
        return u_kf                                          # (K, Nr)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        shock = batch["shock_mask"]
        delta0 = batch["delta0"]
        Nr = shock.shape[0]
        peak = self._propagate_peak(shock, delta0, batch["edge_src"],
                                    batch["edge_dst"], batch["edge_a"], Nr)
        u_kf = self._trajectory(peak, batch["key_days"])
        reach_logit = self.reach_w * peak + self.reach_b
        ones = torch.ones(Nr, device=shock.device, dtype=shock.dtype)
        params = (ones * 6.0, ones, ones * 0.5, ones * 1.0, ones * 150.0)
        return dict(
            peak=peak, peak_pred=peak, peak_phys=peak, peak_phys_raw=peak,
            u_keyframes=u_kf, u_full=u_kf, params=params,
            reach_logit=reach_logit, h_star=u_kf.t(), h_final=u_kf.t(),
            blend_lambda=torch.zeros((), device=peak.device),
            phys_scale=torch.ones_like(peak),
        )

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
