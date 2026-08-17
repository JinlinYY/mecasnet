"""Keyframe-aligned spatiotemporal graph-attention model.

The model predicts ten supervised production keyframes sampled at
``[0, 5, 10, 20, 30, 50, 70, 100, 150, 199]``. Spatial updates use directed,
edge-aware graph attention; temporal updates use a GRU conditioned on the
elapsed interval between adjacent keyframes.

Architecture per step k → k+1:
  (a) spatial: msg_in_v ← Σ_e softmax(score_e) · m(h_src, h_dst, u_src, log a)
              msg_out_v ← Σ_e index_add (reverse direction)
  (b) temporal: h_{k+1} = GRU(h_k, [msg_in, msg_out, u_k, shock, δ₀, Δt_k])
  (c) readout:  u_{k+1} = sigmoid(MLP_u(h_{k+1}))           ← supervised by u_kf[k+1]

Outputs:
  u_keyframes (K, Nr) — directly the K predictions
  peak = (1 - min_k u_k).clamp(0, 1)
  cum_pred (Nr,) — auxiliary head (not yet used in losses by default)
  active_logit (Nr,) — affected/non-affected classification (peak > 0.05)
  reach_logit (Nr,) — required by losses.loss_reach (peak > 0.001)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .config import Config
from .model import scatter_logsumexp


# ---------------------------------------------------------------------------
# Encoder (matches v2 contract)
# ---------------------------------------------------------------------------
class NodeEncoderV3(nn.Module):
    EVT_DIM = 12

    def __init__(self, in_dim: int, d: int, hop_dim: int = 0):
        super().__init__()
        self.hop_dim = hop_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim + 2 + self.EVT_DIM + hop_dim, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

    def forward(self, x_v, shock_mask, delta0, event_scalars, hop_feat=None):
        Nr = x_v.shape[0]
        evt = event_scalars.view(1, -1).expand(Nr, -1)
        feats = [x_v, shock_mask.unsqueeze(-1),
                 delta0.unsqueeze(-1), evt]
        if self.hop_dim > 0:
            assert hop_feat is not None, \
                "hop_encoding != 'none' but no hop feature provided"
            assert hop_feat.shape[-1] == self.hop_dim, \
                f"hop_feat dim {hop_feat.shape[-1]} ≠ expected {self.hop_dim}"
            feats.append(hop_feat)
        feat = torch.cat(feats, dim=-1)
        return self.net(feat)


# ---------------------------------------------------------------------------
# Spatial GAT block (forward + reverse aggregation, edge-aware)
# ---------------------------------------------------------------------------
class DirectedGATBlock(nn.Module):
    """One spatial pass: forward attention (supplier→consumer) + reverse sum.

    Returns msg_in (attention-weighted forward) and msg_out (reverse sum).
    Both have shape (Nr, d). Used inside KSGATStep, NOT chained alone.
    """

    def __init__(self, d: int):
        super().__init__()
        # forward edge feature: [h_src, h_dst, u_src, log a] → m_e (d)
        self.fwd_msg = nn.Sequential(
            nn.Linear(2 * d + 2, d), nn.GELU(),
            nn.Linear(d, d),
        )
        self.fwd_score = nn.Linear(d, 1)
        # reverse edge feature: [h_dst, u_dst] → m_e (d)
        self.rev_msg = nn.Sequential(
            nn.Linear(d + 1, d), nn.GELU(),
        )

    def forward(self, h, u, edge_src, edge_dst, edge_a, Nr):
        log_a = edge_a.clamp_min(1e-4).log().unsqueeze(-1)         # (E,1)
        ef = torch.cat([h[edge_src], h[edge_dst],
                        u[edge_src].unsqueeze(-1), log_a], dim=-1)
        m = self.fwd_msg(ef)                                        # (E,d)
        s = self.fwd_score(m).squeeze(-1)                           # (E,)
        log_norm = scatter_logsumexp(s, edge_dst, Nr)               # (Nr,)
        alpha = (s - log_norm[edge_dst]).exp().unsqueeze(-1)        # (E,1)
        msg_in = torch.zeros(Nr, m.shape[-1], device=h.device, dtype=h.dtype)
        msg_in.index_add_(0, edge_dst, alpha * m)

        rev_in = torch.cat([h[edge_dst], u[edge_dst].unsqueeze(-1)], dim=-1)
        rm = self.rev_msg(rev_in)                                   # (E,d)
        msg_out = torch.zeros_like(h).index_add_(0, edge_src, rm)
        return msg_in, msg_out


# ---------------------------------------------------------------------------
# Min-Plus GAT block (Variant B: tropical / Leontief-aligned aggregation)
# ---------------------------------------------------------------------------
class MinPlusGATBlock(nn.Module):
    """Tropical (min-plus semiring) message passing aligned with Leontief min.

    Forward: h_v[d] = -tau * logsumexp_e(-m_e[d]/tau)  over edges e: dst[e]=v
    where  m_e = MLP([h_src, h_dst, u_src, log a]).

    Element-wise soft-min over MESSAGE VECTORS, NOT a softmax-weighted sum.
    For tau → 0: hard element-wise min (Inoue-Todo Leontief bottleneck).
    For tau → ∞: arithmetic mean. Default tau=0.5 sits between.

    Reverse aggregation kept as plain index_add (sum) — only forward
    direction carries the bottleneck physics.
    """

    def __init__(self, d: int, tau: float = 0.5):
        super().__init__()
        self.tau = float(tau)
        self.fwd_msg = nn.Sequential(
            nn.Linear(2 * d + 2, d), nn.GELU(),
            nn.Linear(d, d),
        )
        self.rev_msg = nn.Sequential(
            nn.Linear(d + 1, d), nn.GELU(),
        )

    def forward(self, h, u, edge_src, edge_dst, edge_a, Nr):
        log_a = edge_a.clamp_min(1e-4).log().unsqueeze(-1)             # (E,1)
        ef = torch.cat([h[edge_src], h[edge_dst],
                        u[edge_src].unsqueeze(-1), log_a], dim=-1)
        m = self.fwd_msg(ef)                                            # (E,d)
        # element-wise soft-min over edges grouped by dst, per hidden dim:
        #   logsumexp on (E,d) tensor returns (Nr,d) with empty-group → 0
        log_norm = scatter_logsumexp(-m / self.tau, edge_dst, Nr)       # (Nr,d)
        msg_in = -self.tau * log_norm                                   # (Nr,d)

        rev_in = torch.cat([h[edge_dst], u[edge_dst].unsqueeze(-1)], dim=-1)
        rm = self.rev_msg(rev_in)                                       # (E,d)
        msg_out = torch.zeros_like(h).index_add_(0, edge_src, rm)
        return msg_in, msg_out


# ---------------------------------------------------------------------------
# Physics parameter generator
# ---------------------------------------------------------------------------
class PhysicsParameterGenerator(nn.Module):
    """Learns physics parameters (τ_u recovery time, c consumption rate) from
    graph statistics + node embeddings. Replaces hardcoded stub params.

    Inputs:
      h_v: node embeddings (Nr, d)
      g_stats: graph statistics (5,) = [log V, log E, log E/V, shock_frac, mean_log_in]

    Outputs:
      params tuple: (τ_u, c) where:
        τ_u (Nr,): per-node recovery time constant [0.5, 5.0] days
        c (Nr,):   per-node consumption rate [0.1, 1.0]

    Philosophy: By conditioning on observable graph stats (not domain labels),
    the model learns that "dense networks recover faster" (Henriet) vs "sparse
    networks recover slower" (Inoue) *from the data*, not hardcoded.
    """

    def __init__(self, d_hidden: int, graph_stats_dim: int = 5):
        super().__init__()
        self.d_hidden = d_hidden
        # Encode graph-level statistics into a fixed representation.
        self.stats_encoder = nn.Sequential(
            nn.Linear(graph_stats_dim, 64), nn.GELU(),
            nn.Linear(64, d_hidden),
        )
        # Combine node embedding + graph context to predict per-node τ_u and c
        self.param_head = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 2),  # outputs: [τ_u_logit, c_logit]
        )

    def forward(self, h_v: torch.Tensor, g_stats: torch.Tensor) -> tuple:
        """
        Args:
            h_v: (Nr, d_hidden) node embeddings
            g_stats: (5,) graph statistics

        Returns:
            tuple: (tau_u, c) each (Nr,)
        """
        # Encode graph stats to get a global context vector
        g_enc = self.stats_encoder(g_stats)  # (d_hidden,)

        # Broadcast graph encoding to every node and concatenate with node embedding
        Nr = h_v.shape[0]
        g_enc_bcast = g_enc.unsqueeze(0).expand(Nr, -1)  # (Nr, d_hidden)
        h_aug = torch.cat([h_v, g_enc_bcast], dim=-1)    # (Nr, 2*d_hidden)

        # Predict per-node parameters
        raw = self.param_head(h_aug)  # (Nr, 2)

        # Decode logits to valid ranges
        tau_u_logit = raw[:, 0]
        c_logit = raw[:, 1]

        # τ_u ∈ [0.5, 5.0] days (recovery time).
        tau_u = 0.5 + 4.5 * torch.sigmoid(tau_u_logit)

        # c ∈ [0.1, 1.0] (consumption rate / capacity ratio)
        c = 0.1 + 0.9 * torch.sigmoid(c_logit)

        return (tau_u, c)

def _collapse_shape(delta, sigma, collapse_p=2.0):
    """Decline-front shape, returns value in [0,1] that is 1 at delta=0 (trough).

    collapse_p == 2.0  -> half-Gaussian exp(-delta^2/(2 sigma^2)).
    collapse_p  > 2.0  -> flat-top super-Gaussian: a PLATEAU for delta << 0
                          (inventory-buffer phase, u stays ~1) followed by a
                          STEEP onset toward the trough at delta=0. This matches
                          the real inventory-depletion front of BOTH the Henriet
                          ARIO and Inoue-Todo simulators (P_prop = P_ini*min_s
                          S/A only drops once a sector buffer is exhausted),
                          unlike the symmetric Gaussian which starts dropping at
                          t=0. The left half of a super-Gaussian is a
                          logistic-like depletion front.
    """
    if collapse_p == 2.0:
        # Half-Gaussian decline front.
        return torch.exp(-(delta * delta) / (2.0 * sigma * sigma + 1e-6))
    z = delta.abs() / (sigma + 1e-3)
    return torch.exp(-0.5 * z.pow(collapse_p))


def _recovery_shape(delta, T_r, recovery_p=1.0, recovery_gate=None, recovery_q=None):
    """Recovery remaining-gap fraction r(t): 1 at the trough (delta=0), 0 at
    full recovery (delta -> inf).  u_recovery = u_ss + (1-P-u_ss) * r.

    When ``recovery_gate`` is supplied, it overrides the scalar ``recovery_p``
    with a differentiable per-node mixture of two recovery mechanisms:
        r_v(z) = e^{-z} * (1 + (1 - g_v) * z),   z = delta/T_r
      g_v = 1 -> e^{-z}            (DIRECT node: instant rebound, single-exp)
      g_v = 0 -> (1+z) e^{-z}      (CASCADE node: lagged Erlang-2 refill)
    The backbone predicts g_v so direct nodes may retain a fast rebound while
    cascade nodes use a lagged refill. For any g in [0,1], r_v is monotone
    decreasing with r(0)=1 and r(inf)=0.

    recovery_p == 1.0 -> single-exponential exp(-delta/T_r).
                         Slope is MAXIMAL right at the trough (instant rebound).
                         This is exact for Henriet DIRECTLY-damaged nodes whose
                         capacity recovers as Delta(t)=Delta_0*exp(-t/tau).
    recovery_p  > 1.0 -> Erlang-k / Gamma-k survival function
                         S_k(z) = exp(-z) * sum_{j=0}^{k-1} z^j / j!,  z=delta/T_r.
                         Slope at delta=0 is ZERO -> a LAGGED, concave startup
                         that then accelerates.  This matches the inventory
                         REFILL dynamics of CASCADE (non-directly-damaged) nodes
                         in BOTH simulators: u recovers only after the stock
                         S(t+1)=S+dt*(receipts-consumption) integral rebuilds
                         (a convolution, not a clean exponential), so production
                         lags then catches up.  k=2 gives (1+z)exp(-z); k=3 gives
                         (1+z+z^2/2)exp(-z).  Each S_k is monotone-decreasing in
                         z with S_k(0)=1, S_k(inf)=0, so u stays a valid envelope.
                         Non-integer recovery_p linearly blends S_floor and S_ceil.
    """
    d = delta.clamp_min(0.0)
    z = d / (T_r + 1e-3)
    # Optional per-node sharpness q on r(z) = exp(-z^q). q == 1 gives a
    # single exponential; q < 1 gives front-loaded recovery and q > 1 gives a
    # lagged S-curve onset. r is monotone-decreasing
    # in z for any q > 0 with r(0)=1, r(inf)=0, so u stays a valid envelope.
    # Spans BOTH faster- and slower-than-single-exp (unlike recovery_gate, which
    # only goes slower) and is non-redundant with T_r (T_r scales z linearly,
    # q changes curvature). Takes precedence over recovery_gate/recovery_p.
    if recovery_q is not None:
        q = recovery_q
        if q.dim() == 1:
            q = q.view(1, -1)                       # (Nr,) -> (1,Nr) broadcast
        return torch.exp(-(z.clamp_min(1e-4)).pow(q))
    e = torch.exp(-z)
    if recovery_gate is not None:
        g = recovery_gate
        if g.dim() == 1:
            g = g.view(1, -1)                       # (Nr,) -> (1,Nr) broadcast
        return e * (1.0 + (1.0 - g) * z)
    if recovery_p == 1.0:
        return e

    def _erlang(k):
        # S_k(z) = exp(-z) * sum_{j=0}^{k-1} z^j / j!
        s = torch.ones_like(z)
        term = torch.ones_like(z)
        for j in range(1, k):
            term = term * z / float(j)
            s = s + term
        return s * e

    lo = int(math.floor(recovery_p))
    hi = int(math.ceil(recovery_p))
    if lo < 1:
        lo = 1
    s_lo = _erlang(lo)
    if hi == lo:
        return s_lo
    w = recovery_p - lo
    s_hi = _erlang(hi)
    return (1.0 - w) * s_lo + w * s_hi


def _reconstruct_trajectory(P, mu, sigma, T_r, days, u_ss=None, collapse_p=2.0,
                            recovery_p=1.0, recovery_gate=None, recovery_q=None):
    """Closed-form trajectory with learnable steady state.

    For t <= mu (collapse):  u(t) = 1 - P * collapse(t)              ; goes 1 -> 1-P
    For t >  mu (recovery):  u(t) = u_ss + (1-P - u_ss) * recovery(t); goes 1-P -> u_ss

    Reduces to `1 - P*shape` when u_ss=1 (rebound regime, Inoue).
    For Henriet's absorbing regime u_ss can drop to 0.

    Args:
      P, mu, sigma, T_r, u_ss : (Nr,) per-node parameters (already in valid ranges)
      days                    : (K,) keyframe days
      collapse_p              : decline-front exponent (2.0 = Gaussian; >2 = flat-top front)
    Returns:
      u : (K, Nr) in [0,1]
    """
    t = days.view(-1, 1)                                                # (K,1)
    P = P.view(1, -1)                                                   # (1,Nr)
    mu = mu.view(1, -1)
    sigma = sigma.view(1, -1)
    T_r = T_r.view(1, -1)
    if u_ss is None:
        u_ss_b = torch.ones_like(P)
    else:
        u_ss_b = u_ss.view(1, -1)
    delta = t - mu                                                      # (K,Nr)
    collapse = _collapse_shape(delta, sigma, collapse_p)
    recovery = _recovery_shape(delta, T_r, recovery_p, recovery_gate=recovery_gate, recovery_q=recovery_q)
    u_collapse = 1.0 - P * collapse
    u_recovery = u_ss_b + (1.0 - P - u_ss_b) * recovery
    u = torch.where(delta <= 0, u_collapse, u_recovery)
    return u.clamp(0.0, 1.0)


def _component_shape(P, mu, sigma, T_r, days, collapse_p=2.0, recovery_p=1.0,
                     recovery_gate=None, recovery_q=None):
    """Single-component shape g(t) = P * (collapse | recovery), returns (K, Nr) in [0, P]."""
    t = days.view(-1, 1)
    P = P.view(1, -1)
    mu = mu.view(1, -1)
    sigma = sigma.view(1, -1)
    T_r = T_r.view(1, -1)
    delta = t - mu
    collapse = _collapse_shape(delta, sigma, collapse_p)
    recovery = _recovery_shape(delta, T_r, recovery_p, recovery_gate=recovery_gate, recovery_q=recovery_q)
    s = torch.where(delta <= 0, collapse, recovery)
    return P * s


def _reconstruct_trajectory_bimodal(P1, mu1, sigma1, T_r1,
                                    P2, mu2, sigma2, T_r2, days,
                                    u_ss=None, tau: float = 0.0, collapse_p=2.0,
                                    recovery_p=1.0, recovery_gate=None, recovery_q=None):
    """Bimodal trajectory as (soft-)min of two single-component trajectories.

    Each component i produces a curve u_i(t) that:
      - collapses to 1-P_i at t=mu_i
      - recovers to u_ss with time constant T_r,i

    Composite:
      tau == 0  : hard min  -> gradient flows ONLY to the deeper component
      tau >  0  : soft min  -> -tau * logsumexp(-u/tau); gradient flows to
                  both components proportionally to softmax(-u/tau).
                  As tau -> 0 it recovers hard min; as tau grows it
                  approaches mean. Use small tau (~0.03) so absorbing
                  (u1≈0, u2>>0) still yields composite ≈ 0, but Inoue
                  mid-trajectory (u1≈u2) gets both gradients.

    Algebraic property preserved under soft-min for small tau:
      When P + u_ss ≈ 1 (Henriet absorbing) one component pins at u_ss≈0
      and dominates softmin → composite ≈ u_ss. The Inoue rebound case
      benefits because both components can now jointly shape the mid-wave.
    """
    u1 = _reconstruct_trajectory(P1, mu1, sigma1, T_r1, days, u_ss=u_ss, collapse_p=collapse_p, recovery_p=recovery_p, recovery_gate=recovery_gate, recovery_q=recovery_q)
    u2 = _reconstruct_trajectory(P2, mu2, sigma2, T_r2, days, u_ss=u_ss, collapse_p=collapse_p, recovery_p=recovery_p, recovery_gate=recovery_gate, recovery_q=recovery_q)
    # tau may be a float OR a 0-d Tensor (learnable case). Tensor branch is
    # always taken with the soft path because softplus+floor guarantees > 0.
    if isinstance(tau, torch.Tensor) or tau > 0.0:
        stacked = torch.stack([u1, u2], dim=0)                # (2, K, Nr)
        u = -tau * torch.logsumexp(-stacked / tau, dim=0)     # (K, Nr)
    else:
        u = torch.minimum(u1, u2)
    return u.clamp(0.0, 1.0)


def _reconstruct_trajectory_trimodal(P1, mu1, sigma1, T_r1,
                                     P2, mu2, sigma2, T_r2,
                                     P3, mu3, sigma3, T_r3, days,
                                     u_ss=None, tau: float = 0.0, collapse_p=2.0,
                                     recovery_p=1.0, recovery_gate=None, recovery_q=None,
                                     tau2=None):
    """Trimodal trajectory: soft-min of three single-component trajectories.

    Same composition rule as bimodal (hard min / softmin via -tau·logsumexp).
    Designed for range-partitioned mu ranges (early / mid / late) so each
    component has its own slot and they cannot collapse onto the same mode.

    Safety: when P3 init is small (e.g. sigmoid(-3)≈0.05), the third
    component contributes only ε ≈ 0 to the trajectory at init, so the
    overall behaviour starts essentially bimodal. Optimizer grows P3 only
    if late-trough nodes need it (the Inoue case); for monotone domains
    (Henriet) P3 should stay near zero, leaving the model bimodal in
    effect.

    With hierarchical dual temperatures, when ``tau2`` is provided the
    composition becomes a two-LEVEL nested softmin with INDEPENDENT sharpness
    at each seam:
        u = softmin_{tau2}( softmin_{tau}(u1, u2),  u3 )
    so `tau` controls ONLY the G1/G2 seam (d50) and `tau2` controls ONLY the
    G2/G3 seam (d70). The single-tau version forced both seams to share one
    sharpness, which is why per-node single-tau B drove d50 to 0.73 but
    collapsed d70 to -0.30 (a node's large tau, needed to blend d50, over-mixed
    G3 into d70). Decoupling lets each seam pick its own per-node sharpness.
    """
    u1 = _reconstruct_trajectory(P1, mu1, sigma1, T_r1, days, u_ss=u_ss, collapse_p=collapse_p, recovery_p=recovery_p, recovery_gate=recovery_gate, recovery_q=recovery_q)
    u2 = _reconstruct_trajectory(P2, mu2, sigma2, T_r2, days, u_ss=u_ss, collapse_p=collapse_p, recovery_p=recovery_p, recovery_gate=recovery_gate, recovery_q=recovery_q)
    u3 = _reconstruct_trajectory(P3, mu3, sigma3, T_r3, days, u_ss=u_ss, collapse_p=collapse_p, recovery_p=recovery_p, recovery_gate=recovery_gate, recovery_q=recovery_q)
    if tau2 is not None:
        # Two-level nested softmin: tau owns the G1/G2 seam, tau2 the G2/G3 seam.
        s12 = torch.stack([u1, u2], dim=0)                    # (2, K, Nr)
        u12 = -tau * torch.logsumexp(-s12 / tau, dim=0)       # (K, Nr)
        s = torch.stack([u12, u3], dim=0)                     # (2, K, Nr)
        u = -tau2 * torch.logsumexp(-s / tau2, dim=0)         # (K, Nr)
    elif isinstance(tau, torch.Tensor) or tau > 0.0:
        stacked = torch.stack([u1, u2, u3], dim=0)            # (3, K, Nr)
        u = -tau * torch.logsumexp(-stacked / tau, dim=0)     # (K, Nr)
    else:
        u = torch.minimum(torch.minimum(u1, u2), u3)
    return u.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Helper: build a spatial block matching aggr_mode and tau
# ---------------------------------------------------------------------------
def _build_spatial_block(d: int, mode: str, tropical_tau: float = 0.5):
    if mode == "softmin":
        return MinPlusGATBlock(d, tau=tropical_tau)
    return DirectedGATBlock(d)


# ---------------------------------------------------------------------------
# KS-GAT step: spatial(GAT) + temporal(GRU with Δt) + direct u readout
# ---------------------------------------------------------------------------
class KSGATStep(nn.Module):
    """One supervised step k → k+1.

    Inputs:
      h, u    : current node hidden + u state, (Nr,d) (Nr,)
      shock   : shock_mask (Nr,)
      delta0  : initial damage (Nr,)
      dt      : scalar (or Nr,) day-gap to the next keyframe
      edges   : edge_src, edge_dst, edge_a, Nr
    """

    def __init__(self, d: int, u_head_bias_init: float = 0.95,
                 aggr_mode: str = "softmax", tropical_tau: float = 0.5,
                 external_spatial: bool = False,
                 use_inv_state: bool = True,
                 tau_inv_init_days: float = 5.0,
                 inner_dt_days: float = 1.0):
        super().__init__()
        assert aggr_mode in ("softmax", "softmin"), aggr_mode
        self.external_spatial = external_spatial
        if external_spatial:
            self.spatial = None      # caller must pass spatial_block to forward()
        elif aggr_mode == "softmin":
            self.spatial = MinPlusGATBlock(d, tau=tropical_tau)
        else:
            self.spatial = DirectedGATBlock(d)
        # GRU input: msg_in (d), msg_out (d), u (1), shock (1), delta0 (1), Δt_norm (1),
        # plus (S, O) state (2) when use_inv_state
        self.use_inv_state = use_inv_state
        self.inner_dt_days = inner_dt_days
        extra = 2 if use_inv_state else 0
        self.gru = nn.GRUCell(2 * d + 4 + extra, d)
        if use_inv_state:
            # Inventory target S* ∈ (0,1) per node — conditions on (h, shock, delta0)
            # so the steady-state can react directly to the initial damage signal,
            # not just to whatever survives the GRU memory bottleneck.
            self.s_target_head = nn.Linear(d + 2, 1)
            # Init so initial output ≈ 1 (matches s_prev=1-delta0≈1) → zero initial drive
            with torch.no_grad():
                nn.init.zeros_(self.s_target_head.weight)
                self.s_target_head.bias.fill_(5.0)   # sigmoid(5) ≈ 0.993
            # log τ_inv: natural period 2π·τ ≈ 30d → τ ≈ 5d (matches Inoue TAU=6)
            self.log_tau_inv = nn.Parameter(
                torch.tensor(math.log(tau_inv_init_days), dtype=torch.float32))
            # Damping ratio ζ ∈ (0.05, 0.95); init overdamped (logit=+3 → ζ≈0.86)
            # → initial dynamics quiet; model can lower ζ later to enable oscillation.
            self.damping_logit = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))
        # u readout — direct sigmoid prediction; receives (h, S, O) when use_inv_state
        u_head_in = d + (2 if use_inv_state else 0)
        self.u_head = nn.Sequential(
            nn.Linear(u_head_in, d), nn.GELU(),
            nn.Linear(d, 1),
        )
        with torch.no_grad():
            # init bias ≈ logit(u_head_bias_init); 0.95 default, 0.99 for V3tB
            p = float(u_head_bias_init)
            p = min(max(p, 1e-4), 1 - 1e-4)
            self.u_head[-1].bias.fill_(math.log(p / (1 - p)))

    def forward(self, h, u, shock, delta0, dt_norm,
                edge_src, edge_dst, edge_a, Nr,
                spatial_block: "nn.Module | None" = None,
                s_prev=None, o_prev=None, dt_days=None):
        if spatial_block is not None:
            msg_in, msg_out = spatial_block(h, u, edge_src, edge_dst, edge_a, Nr)
        else:
            assert self.spatial is not None, \
                "KSGATStep was built with external_spatial=True but no spatial_block passed"
            msg_in, msg_out = self.spatial(h, u, edge_src, edge_dst, edge_a, Nr)
        if dt_norm.dim() == 0:
            dt_feat = dt_norm.expand(Nr).unsqueeze(-1)
        else:
            dt_feat = dt_norm.unsqueeze(-1)
        if self.use_inv_state:
            assert s_prev is not None and o_prev is not None and dt_days is not None, \
                "use_inv_state=True requires s_prev, o_prev and dt_days"
            gru_in = torch.cat([
                msg_in, msg_out,
                u.unsqueeze(-1), shock.unsqueeze(-1),
                delta0.unsqueeze(-1), dt_feat,
                s_prev.unsqueeze(-1), o_prev.unsqueeze(-1),
            ], dim=-1)
            h_new = self.gru(gru_in, h)
            # Damped 2nd-order oscillator (Inoue bullwhip mechanism):
            #   dS/dt = O
            #   dO/dt = -ω²(S - S*) - 2ζω O,    ω = 1/τ_inv
            # Sub-step with dt_inner ≈ 1 day for stable integration over Δt up to 50d.
            s_tgt_in = torch.cat(
                [h_new, shock.unsqueeze(-1), delta0.unsqueeze(-1)], dim=-1)
            s_target = torch.sigmoid(self.s_target_head(s_tgt_in).squeeze(-1))
            tau_inv = torch.exp(self.log_tau_inv).clamp_min(1.0)
            omega = 1.0 / tau_inv
            zeta = 0.05 + 0.9 * torch.sigmoid(self.damping_logit)
            dt_total = float(dt_days)
            K_inner = max(1, int(math.ceil(dt_total / self.inner_dt_days)))
            dt_in = dt_total / K_inner
            s_cur, o_cur = s_prev, o_prev
            for _ in range(K_inner):
                accel = -(omega * omega) * (s_cur - s_target) - 2.0 * zeta * omega * o_cur
                o_cur = o_cur + dt_in * accel
                s_cur = s_cur + dt_in * o_cur
            s_new, o_new = s_cur, o_cur
            u_in = torch.cat([h_new, (s_new - 1.0).unsqueeze(-1), o_new.unsqueeze(-1)], dim=-1)
            u_new = torch.sigmoid(self.u_head(u_in).squeeze(-1))
            return h_new, u_new, s_new, o_new
        gru_in = torch.cat([
            msg_in, msg_out,
            u.unsqueeze(-1), shock.unsqueeze(-1),
            delta0.unsqueeze(-1), dt_feat,
        ], dim=-1)
        h_new = self.gru(gru_in, h)
        u_new = torch.sigmoid(self.u_head(h_new).squeeze(-1))
        return h_new, u_new


# ---------------------------------------------------------------------------
# Top-level v3 model
# ---------------------------------------------------------------------------
class KSGATv3(nn.Module):
    """Keyframe-Aligned Spatiotemporal GAT.

    Peak readout supports three modes:
      "traj"   : peak = 1 - min_k u_k                   (closed form on rollout)
      "direct" : peak = sigmoid(MLP(h_final, traj_summary))   (low-bias prior)
      "blend"  : per-node gate g_v ∈ [0,1] mixes the two       (DEFAULT)

    The blend is structurally pinned at g=1 for shocked nodes (forcing them
    to use the trajectory branch, which v3 already nails — R²pk_shk≈+0.85).
    For unshocked nodes the gate is learned, biased toward direct at init
    (g_neural ≈ 0.30 → 70% direct), since cascade R² is the dominant problem.

    forward(batch) returns dict compatible with mecasnet.losses:
      u_keyframes (K, Nr)   — main supervision
      peak        (Nr,)     — see peak_mode above
      peak_traj, peak_direct, gate (Nr,) — diagnostics
      reach_logit (Nr,)     — required by loss_reach
      u_full, params, h_star, h_final, blend_lambda, phys_scale  — stubs / aliases
    """

    DT_NORM = 50.0   # divide Δt (days) by this so GRU sees O(1) magnitudes
    TRAJ_DIM = 5     # min, mean, last, argmin/K, std
    # hop_encoding options:
    #   "none"     : no hop feature added                              (0 dim)
    #   "onehot5"  : one-hot 0/1/2/3/>=4                               (5 dim)
    #   "graded2"  : [is_hop1, 1/(1+min(hop,4))]                        (2 dim)
    #                 — captures hop=1 outlier + monotone gradient per
    #                   characterise_dynamics.py Q7 (1000 events).
    HOP_DIMS = {"none": 0, "onehot5": 5, "graded2": 2}
    # graded factor per hop index 0..4:  1/(1+hop)
    _GRADED_TABLE = (1.0, 0.5, 1.0 / 3.0, 0.25, 0.2)

    def __init__(self, cfg: Config, Fv: int,
                 peak_mode: str = "traj",
                 u_head_bias_high: bool = False,
                 hop_encoding: str = "none",
                 softmin_tau: float = 0.0,
                 decoder_mode: str = "rollout",
                 aggr_mode: str = "softmax",
                 tropical_tau: float = 0.5,
                 prewarm_layers: int = 0,
                 phase_switch_k: int = 5,
                 param2_preset: str = "default",
                 bimodal_tau: float = 0.0,
                 learnable_bimodal_tau: bool = False,
                 use_film: bool = False,
                 use_inv_state: bool = True,
                 moe_router: bool = False,
                 blend_rollout: bool = False,
                 triple_blend: bool = False,
                 blend_init=(2.0, 0.0, 0.0),
                 free_residual: bool = False):
        super().__init__()
        # Initial logits for softmax fusion over analytical, rollout, and direct
        # streams. The default maps to approximately [0.79, 0.11, 0.11].
        self.blend_init = tuple(float(x) for x in blend_init)
        self.use_inv_state = bool(use_inv_state)
        self.moe_router = bool(moe_router)
        # Run rollout in parallel and learn per-keyframe fusion weights.
        self.blend_rollout = bool(blend_rollout)
        # Add a direct Linear(d, K) readout and learn per-keyframe softmax
        # weights across analytical, rollout, and direct streams. Enabling the
        # three-stream blend also enables the rollout stream.
        self.triple_blend = bool(triple_blend)
        if self.triple_blend:
            self.blend_rollout = True
        # Optional zero-initialized residual: u = u_analytical + delta_free.
        # It is a standalone path and the output is clamped to [0, 1].
        self.free_residual = bool(free_residual)
        # Exponent for the decline front. decline_p == 2 gives a Gaussian;
        # decline_p  > 2.0 -> super-Gaussian plateau-then-steep onset that
        # matches the inventory-depletion front of the Henriet/Inoue sims and
        # represents an inventory-depletion plateau followed by a steep onset.
        self.decline_p = float(getattr(cfg, "decline_p", 2.0)) if cfg is not None else 2.0
        # Exponent for the rebound shape.
        # recovery_p == 1.0 -> single-exp exp(-delta/T_r) (instant
        #   rebound at the trough; exact for Henriet directly-damaged nodes).
        # recovery_p  > 1.0 -> Erlang-k/Gamma-k survival fn = LAGGED concave
        #   startup matching the inventory-REFILL convolution of cascade nodes
        #   (S(t+1)=S+dt(receipts-consumption)) in both simulators.
        self.recovery_p = float(getattr(cfg, "recovery_p", 1.0)) if cfg is not None else 1.0
        # Optional per-node recovery gate. When cfg.recovery_gate is
        # True a Linear(d,1)->sigmoid head predicts g_v in [0,1] per node that
        # mixes DIRECT single-exp (g=1) and CASCADE Erlang-2 (g=0) recovery,
        # overriding the global scalar recovery_p. Built below once d is known.
        self.recovery_gate_enabled = bool(getattr(cfg, "recovery_gate", False)) if cfg is not None else False
        # Optional per-node recovery sharpness q_v.
        # When cfg.recovery_q is True a Linear(d,1) head predicts q_v in
        # [recovery_q_min, recovery_q_max] mapping the recovery shape to
        # r(z)=exp(-z^q). q<1 speeds recovery and q>1 delays it.
        self.recovery_q_enabled = bool(getattr(cfg, "recovery_q", False)) if cfg is not None else False
        # Single-component ablation switches.
        #   ablate_uncond      : zero shock_mask & delta0 fed to the NODE ENCODER
        #     so the learned representation/propagation is "unconditioned" (no
        #     Day-0 shock identity). u0's physical IC (1 - delta0*shock) and the
        #     FiLM shock_frac are kept, matching the paper's modest-degradation
        #     "unconditioned propagation" variant (not a catastrophic full wipe).
        #   ablate_forward_only: drop the REVERSE (downstream->upstream demand
        #     feedback) message msg_out in the prewarm + param-decoder spatial
        #     loops, leaving only forward (supplier->consumer) attention msg_in.
        self.ablate_uncond = bool(getattr(cfg, "ablate_uncond", False)) if cfg is not None else False
        self.ablate_forward_only = bool(getattr(cfg, "ablate_forward_only", False)) if cfg is not None else False
        # Optional graph-coupled inventory recovery integrator.
        # When cfg.physrecover is True the closed-form recovery branch (t > mu_v)
        # is REPLACED by an on-graph relaxation that faithfully mirrors BOTH
        # simulators' production rule  P_act = min(P_cap, P_prop=Leontief(S), D):
        #   u_v(t+dt) = u_v(t) + (1-exp(-kappa_v*dt)) * (target_v(t) - u_v(t))
        #   target_v(t) = min( own_recovery_v(t),  supply_v(t) )
        #   supply_v(t) = ( sum_{r in suppliers(v)} a_rv * u_r(t) ) / sum a_rv
        # i.e. a node cannot recover faster than its UPSTREAM suppliers recover
        # ("my recovery waits for my suppliers' recovery") — the graph+time
        # coupling that a per-node closed-form envelope structurally cannot
        # express. own_recovery_v(t) is the
        # decoder's existing closed-form recovery (so the interpretable
        # mu/sigma/A physics baseline is preserved); the integrator only imposes
        # the supply ceiling + a per-node relaxation rate kappa_v. This is the
        # sole recovery path when enabled.
        self.physrecover_enabled = bool(getattr(cfg, "physrecover", False)) if cfg is not None else False
        # During warmup, router output is overridden to uniform [1/3,1/3,1/3]
        # so all 3 Gaussian amplitudes receive equal gradient.
        self.moe_warmup_active = False
        assert peak_mode in ("traj", "direct", "blend", "hitgate"), peak_mode
        assert hop_encoding in self.HOP_DIMS, \
            f"hop_encoding must be in {list(self.HOP_DIMS)}"
        assert softmin_tau >= 0.0, "softmin_tau must be >= 0 (0 = hard min)"
        assert decoder_mode in ("rollout", "param", "param2", "param2_blend", "param3"), decoder_mode
        assert aggr_mode in ("softmax", "softmin", "phase"), aggr_mode
        assert prewarm_layers >= 0, prewarm_layers
        assert param2_preset in ("default", "deep_mid", "g3_narrow"), param2_preset
        assert bimodal_tau >= 0.0, "bimodal_tau must be >= 0 (0 = hard min)"
        self.cfg = cfg
        self.day0_demand_pullback = bool(
            getattr(cfg, "day0_demand_pullback", False)
        ) if cfg is not None else False
        self.peak_mode = peak_mode
        self.hop_encoding = hop_encoding
        self.softmin_tau = float(softmin_tau)
        self.decoder_mode = decoder_mode
        self.aggr_mode = aggr_mode
        self.prewarm_layers = int(prewarm_layers)
        self.phase_switch_k = int(phase_switch_k)
        self.param2_preset = param2_preset
        # Learnable softmin temperature for the trimodal envelope. When enabled,
        # replace fixed tau with a
        # softplus-parameterised scalar (>=1e-4 floor) so the model can learn
        # how much to soften the hard min at seam points like d10/d70.
        # The default uses the fixed scalar value.
        self.learnable_bimodal_tau = bool(learnable_bimodal_tau)
        if self.learnable_bimodal_tau:
            init = max(float(bimodal_tau), 0.02)        # avoid log(0)
            # softplus(p) = init  =>  p = log(exp(init) - 1)
            p_init = math.log(math.expm1(init))
            self._log_bimodal_tau = nn.Parameter(torch.tensor(float(p_init)))
            self.bimodal_tau = float(init)              # sentinel; real value via _get_bimodal_tau()
        else:
            self._log_bimodal_tau = None
            self.bimodal_tau = float(bimodal_tau)
        self.use_film = bool(use_film)
        # ----- graph-statistics FiLM conditioning -----
        # 5 SCALE-INVARIANT stats per event (no domain_id, no log V, no log E):
        #   [log(E/V), shock_frac, mean(log1p in_deg), std(log1p in_deg),
        #    max(log1p in_deg) - mean]
        # Mapped to (γ, β) ∈ R^(2d) which scales/shifts encoder output h.
        # Substrate-agnostic: model must infer physics regime from degree-distribution
        # shape, not from graph size or an explicit domain label.
        if self.use_film:
            self.film_net = nn.Sequential(
                nn.Linear(5, 32), nn.GELU(),
                nn.Linear(32, 2 * cfg.d_hidden),
            )
            with torch.no_grad():
                # init γ→1, β→0 (identity FiLM) so untrained model behaves like no-FiLM
                self.film_net[-1].weight.zero_()
                bias = torch.zeros(2 * cfg.d_hidden)
                bias[:cfg.d_hidden] = 1.0   # γ default = 1
                self.film_net[-1].bias.copy_(bias)
        # ----- learned physical parameters (τ_u, c) -----
        # Learn per-node parameters conditioned on graph statistics.
        self.phys_param_gen = PhysicsParameterGenerator(cfg.d_hidden, graph_stats_dim=5)
        # bimodal range params (used by param2 / param2_blend)
        # default:    mu1, mu2 ∈ [0, 50],  sigma ∈ [2, 30]
        # deep_mid:   mu1, mu2 ∈ [2, 82],  sigma ∈ [2, 52]   (G1: align with d30 trough)
        # g3_narrow:  mu1 ∈ [0, 50] sigma1 ∈ [2, 30]   (early-cascade Gaussian)
        #             mu2 ∈ [0, 200] sigma2 ∈ [2, 60]  (late recovery)
        if param2_preset == "deep_mid":
            self.p2_mu1_off, self.p2_mu1_scale = 2.0, 80.0
            self.p2_sigma1_off, self.p2_sigma1_scale = 2.0, 50.0
            self.p2_mu2_off, self.p2_mu2_scale = 2.0, 80.0
            self.p2_sigma2_off, self.p2_sigma2_scale = 2.0, 50.0
        elif param2_preset == "g3_narrow":
            self.p2_mu1_off, self.p2_mu1_scale = 0.0, 50.0
            self.p2_sigma1_off, self.p2_sigma1_scale = 2.0, 28.0
            self.p2_mu2_off, self.p2_mu2_scale = 0.0, 50.0
            self.p2_sigma2_off, self.p2_sigma2_scale = 2.0, 60.0
        else:
            self.p2_mu1_off, self.p2_mu1_scale = 0.0, 50.0
            self.p2_sigma1_off, self.p2_sigma1_scale = 2.0, 28.0
            self.p2_mu2_off, self.p2_mu2_scale = 0.0, 50.0
            self.p2_sigma2_off, self.p2_sigma2_scale = 2.0, 28.0
        # ----- param3 (trimodal) ranges. Used only when decoder_mode="param3".
        # Range partitioning: each Gaussian owns a region so they cannot
        # collapse onto the same mode.
        #   mu1 ∈ [0, 40],    sigma1 ∈ [2, 20]   (early: d0-d20)
        #   mu2 ∈ [20, 90],   sigma2 ∈ [3, 35]   (mid: d20-d70)
        #   mu3 ∈ [70, 220],  sigma3 ∈ [10, 70]  (late: d70-d199)
        # Overlapping bounds allow smooth boundary shift if a mode wants to
        # straddle two regions, while range centers stay clearly separated.
        self.p3_mu1_off, self.p3_mu1_scale = 0.0, 40.0
        self.p3_sigma1_off, self.p3_sigma1_scale = 2.0, 18.0
        self.p3_mu2_off, self.p3_mu2_scale = 20.0, 70.0
        self.p3_sigma2_off, self.p3_sigma2_scale = 3.0, 32.0
        # Optional override for the G2 center range. The default remains
        # [20, 90]; the override can place the middle component near d70.
        if cfg is not None and bool(getattr(cfg, "p3_g2_repos", False)):
            self.p3_mu2_off = float(getattr(cfg, "p3_mu2_off_override", 45.0))
            self.p3_mu2_scale = float(getattr(cfg, "p3_mu2_scale_override", 40.0))
        # The third component covers late recovery with center [100, 220] and
        # width [15, 60].
        self.p3_mu3_off, self.p3_mu3_scale = 100.0, 120.0
        self.p3_sigma3_off, self.p3_sigma3_scale = 15.0, 45.0
        d = cfg.d_hidden
        hop_dim = self.HOP_DIMS[hop_encoding]
        self.encoder = NodeEncoderV3(Fv, d, hop_dim=hop_dim)
        u_init = 0.99 if u_head_bias_high else 0.95

        # ----- phase-aware aggregation -----
        # When aggr_mode="phase":
        #   - rollout step k: minplus iff k <= phase_switch_k (collapse phase)
        #   - prewarm + param-decoder internal: first 2/3 layers minplus,
        #     last 1/3 softmax (collapse-y reasoning then recovery refinement)
        # The rollout's GRU/u_head is shared across phases (only spatial swaps).
        # Both branches expose softmax-only behaviour when aggr_mode="softmax"
        # (default) and minplus-only behaviour when aggr_mode="softmin".
        if aggr_mode == "phase":
            self.collapse_block = MinPlusGATBlock(d, tau=tropical_tau)
            self.recovery_block = DirectedGATBlock(d)
            self.step = KSGATStep(d, u_head_bias_init=u_init,
                                  external_spatial=True,
                                  use_inv_state=self.use_inv_state)
        else:
            self.step = KSGATStep(d, u_head_bias_init=u_init,
                                  aggr_mode=aggr_mode, tropical_tau=tropical_tau,
                                  use_inv_state=self.use_inv_state)

        # ----- interval-aware inter-keyframe spatial diffusion -----
        # Apply n_iter-1 extra GAT passes before each KSGATStep, where
        # n_iter = round(dt_days / D_GAT_DAYS).
        self.D_GAT_DAYS = 5.0
        self.diffusion_max_extra = 6
        if aggr_mode == "softmin":
            self.diffusion_gat = MinPlusGATBlock(d, tau=tropical_tau)
        else:
            self.diffusion_gat = DirectedGATBlock(d)
        self.diffusion_norm = nn.LayerNorm(d)

        # ----- encoder-side spatial prewarm -----
        # N spatial-only GAT layers with residual + LayerNorm, run once after
        # the encoder with constant u=u0, giving h0 multi-hop context before
        # the rollout begins.
        if prewarm_layers > 0:
            blocks = []
            for i in range(prewarm_layers):
                if aggr_mode == "phase":
                    use_minplus = i < (prewarm_layers * 2 + 2) // 3   # ~2/3
                    blocks.append(MinPlusGATBlock(d, tau=tropical_tau)
                                  if use_minplus else DirectedGATBlock(d))
                elif aggr_mode == "softmin":
                    blocks.append(MinPlusGATBlock(d, tau=tropical_tau))
                else:
                    blocks.append(DirectedGATBlock(d))
            self.prewarm_gat = nn.ModuleList(blocks)
            self.prewarm_norms = nn.ModuleList([
                nn.LayerNorm(d) for _ in range(prewarm_layers)
            ])
        # Variant A (single-mode parametric): 3 internal spatial layers + 4-param head
        # Variant E (param2, bimodal): 3 internal spatial layers + 8-param head
        # Variant E+blend (param2_blend): same param2 stack + per-frame learned
        #   blend gate alpha_kf against the rollout (G2).
        if decoder_mode in ("param", "param2", "param2_blend", "param3"):
            n_internal = 3
            blocks = []
            for i in range(n_internal):
                if aggr_mode == "phase":
                    use_minplus = i < (n_internal * 2 + 2) // 3   # 2 of 3 minplus
                    blocks.append(MinPlusGATBlock(d, tau=tropical_tau)
                                  if use_minplus else DirectedGATBlock(d))
                elif aggr_mode == "softmin":
                    blocks.append(MinPlusGATBlock(d, tau=tropical_tau))
                else:
                    blocks.append(DirectedGATBlock(d))
            self.param_gat = nn.ModuleList(blocks)
            self.param_gat_norms = nn.ModuleList([
                nn.LayerNorm(d) for _ in range(n_internal)
            ])
        # u_ss bias: sigmoid(4.595) ≈ 0.99 (rebound regime default, backward-compat)
        _U_SS_BIAS = math.log(0.99 / 0.01)
        if decoder_mode == "param":
            self.param_head = nn.Sequential(
                nn.Linear(d, d), nn.GELU(),
                nn.Linear(d, 5),
            )
            with torch.no_grad():
                self.param_head[-1].bias.copy_(torch.tensor([
                    math.log(0.05 / 0.95),                   # P  -> 0.05
                    math.log(0.4 / 0.6),                     # mu -> 20d
                    math.log((8.0 / 28.0) / (1.0 - 8.0 / 28.0)),  # sigma -> 10d
                    math.log(0.4 / 0.6),                     # T_r -> 150d
                    _U_SS_BIAS,                              # u_ss -> 0.99
                ]))
        elif decoder_mode in ("param2", "param2_blend"):
            # 9 outputs: (P1, mu1, sigma1, T_r1,  P2, mu2, sigma2, T_r2, u_ss)
            self.param_head = nn.Sequential(
                nn.Linear(d, d), nn.GELU(),
                nn.Linear(d, 9),
            )
            with torch.no_grad():
                if param2_preset == "deep_mid":
                    # G1: deep middle trough at d30 + wider second component.
                    bias = torch.tensor([
                        math.log(0.05 / 0.95),
                        math.log(0.10 / 0.90),
                        math.log((4.0 / 50.0) / (1.0 - 4.0 / 50.0)),
                        math.log(0.20 / 0.80),
                        math.log(0.25 / 0.75),
                        math.log(0.35 / 0.65),
                        math.log((18.0 / 50.0) / (1.0 - 18.0 / 50.0)),
                        math.log(0.28 / 0.72),
                        _U_SS_BIAS,
                    ])
                elif param2_preset == "g3_narrow":
                    bias = torch.tensor([
                        math.log(0.05 / 0.95),
                        math.log(0.20 / 0.80),
                        math.log((4.0 / 28.0) / (1.0 - 4.0 / 28.0)),
                        math.log(0.20 / 0.80),
                        math.log(0.25 / 0.75),
                        math.log(0.20 / 0.80),                          # mu2 → d40
                        math.log((14.0 / 60.0) / (1.0 - 14.0 / 60.0)),  # sigma2 → d14
                        math.log(0.28 / 0.72),
                        _U_SS_BIAS,
                    ])
                else:
                    bias = torch.tensor([
                        math.log(0.05 / 0.95),
                        math.log(0.30 / 0.70),
                        math.log((6.0 / 28.0) / (1.0 - 6.0 / 28.0)),
                        math.log(0.4 / 0.6),
                        math.log(0.03 / 0.97),
                        math.log(0.80 / 0.20),
                        math.log((13.0 / 28.0) / (1.0 - 13.0 / 28.0)),
                        math.log(0.4 / 0.6),
                        _U_SS_BIAS,
                    ])
                self.param_head[-1].bias.copy_(bias)

        elif decoder_mode == "param3":
            # 13 outputs:
            #   P1, mu1, sigma1, T_r1,  P2, mu2, sigma2, T_r2,
            #   P3, mu3, sigma3, T_r3,  u_ss
            # Range partitioning enforced by p3_mu*_off/scale (early/mid/late).
            # P3 starts at a moderate amplitude and remains independently learned.
            self.param_head = nn.Sequential(
                nn.Linear(d, d), nn.GELU(),
                nn.Linear(d, 13),
            )
            with torch.no_grad():
                # mu1=15, mu2=50, mu3=130 (in tightened range [100,220])
                # sigma1=8, sigma2=15, sigma3=30
                # P3 initialization and range partition keep all components active.
                bias = torch.tensor([
                    math.log(0.20 / 0.80),                                  # P1 → 0.20
                    math.log(0.375 / 0.625),                                # mu1 → 15
                    math.log((8.0 / 18.0) / (1.0 - 8.0 / 18.0)),            # sigma1 → 8
                    math.log(0.20 / 0.80),                                  # T_r1 → ~100
                    math.log(0.20 / 0.80),                                  # P2 → 0.20
                    math.log(0.429 / 0.571),                                # mu2 → 50
                    math.log((12.0 / 32.0) / (1.0 - 12.0 / 32.0)),          # sigma2 → 15
                    math.log(0.28 / 0.72),                                  # T_r2 → ~120
                    math.log(0.20 / 0.80),                                  # P3 → 0.20
                    math.log(0.25 / 0.75),                                  # mu3 → 130 (in [100,220])
                    math.log((15.0 / 45.0) / (1.0 - 15.0 / 45.0)),          # sigma3 → 30 (in [15,60])
                    math.log(0.40 / 0.60),                                  # T_r3 → ~150
                    _U_SS_BIAS,
                ])
                self.param_head[-1].bias.copy_(bias)
        # alpha_kf shape (K,); blended u_kf = sigmoid(alpha_k)*u_rollout + (1-...)*u_param.
        # Init zero → 50/50 blend; let model learn which path wins per frame.
        # Active whenever blend_rollout=True (param2_blend OR param3_moe_blend).
        if decoder_mode == "param2_blend":
            self.blend_rollout = True
        if self.blend_rollout:
            K = len(cfg.key_days)
            self.alpha_kf = nn.Parameter(torch.zeros(K))
        # Three-stream fusion heads. Mixing logits have shape (K, 3); a
        # softmax over the last dim yields per-keyframe weights summing to 1
        # for {trimodal, rollout, direct}. Initial logits [2, 0, 0] -> softmax
        # approximately [0.79, 0.11, 0.11] at initialization.
        if self.triple_blend:
            K = len(cfg.key_days)
            self.kf_direct_head = nn.Linear(d, K)
            with torch.no_grad():
                # Match pure-edge-state baseline prior: most (node, day) cells have
                # u close to 1 (no damage), so init bias = logit(0.95).
                self.kf_direct_head.bias.fill_(math.log(0.95 / 0.05))
                nn.init.zeros_(self.kf_direct_head.weight)
            init_logits = torch.zeros(K, 3)
            _bi = self.blend_init  # (physics, rollout, direct)
            init_logits[:, 0] = _bi[0]
            init_logits[:, 1] = _bi[1]
            init_logits[:, 2] = _bi[2]
            self.blend_logits = nn.Parameter(init_logits)
        # Zero-initialized residual head for the analytical trajectory.
        if self.free_residual:
            K = len(cfg.key_days)
            # Optionally replace Linear(d,K) with a d->2d->K GELU MLP. The last
            # layer is zero-initialized so the initial output is unchanged.
            _res_wide = bool(getattr(cfg, "residual_wide", False)) if cfg is not None else False
            if _res_wide:
                self.kf_residual_head = nn.Sequential(
                    nn.Linear(d, 2 * d), nn.GELU(),
                    nn.Linear(2 * d, K),
                )
                with torch.no_grad():
                    nn.init.zeros_(self.kf_residual_head[-1].weight)
                    nn.init.zeros_(self.kf_residual_head[-1].bias)
            else:
                self.kf_residual_head = nn.Linear(d, K)
                with torch.no_grad():
                    nn.init.zeros_(self.kf_residual_head.weight)
                    nn.init.zeros_(self.kf_residual_head.bias)
            # The optional time mask limits the residual to days through
            # resid_mask_day and leaves the analytical recovery tail unchanged.
            self.resid_tail_mask = bool(getattr(cfg, "resid_tail_mask", False)) if cfg is not None else False
            self.resid_mask_day = float(getattr(cfg, "resid_mask_day", 50.0)) if cfg is not None else 50.0
        # Optional per-node softmin temperature. This keeps the analytical
        # components explicit while allowing node-conditional seam smoothness.
        self.seam_tau_pernode = bool(getattr(cfg, "seam_tau_pernode", False)) if cfg is not None else False
        # Bias controlling the initial per-node seam temperature. The default
        # maps to tau=0.02.
        _tau_b = float(getattr(cfg, "seam_tau_init_bias", -3.954)) if cfg is not None else -3.954
        if self.seam_tau_pernode:
            self.seam_tau_head = nn.Linear(d, 1)
            with torch.no_grad():
                nn.init.zeros_(self.seam_tau_head.weight)
                # softplus(b)+1e-3 = 0.02 -> softplus(b)=0.019 -> b≈-3.954
                self.seam_tau_head.bias.fill_(_tau_b)
        # Optional second temperature head for the nested softmin
        #   u = softmin_{tau2_v}( softmin_{tau1_v}(u1,u2), u3 ),
        # where tau1_v controls G1/G2 and tau2_v controls their fusion with G3.
        self.seam_tau_dual = bool(getattr(cfg, "seam_tau_dual", False)) if cfg is not None else False
        if self.seam_tau_dual:
            assert self.seam_tau_pernode, "seam_tau_dual requires seam_tau_pernode (head 1)"
            self.seam_tau2_head = nn.Linear(d, 1)
            with torch.no_grad():
                nn.init.zeros_(self.seam_tau2_head.weight)
                self.seam_tau2_head.bias.fill_(_tau_b)
        # Per-node recovery gate head. Zero-initialized weight and
        # zero bias => sigmoid(0)=0.5 at start (neutral mix of direct/cascade)
        # so the backbone learns g_v per node from the start. Adds (d+1) params.
        if self.recovery_gate_enabled:
            self.recovery_gate_head = nn.Linear(d, 1)
            with torch.no_grad():
                nn.init.zeros_(self.recovery_gate_head.weight)
                nn.init.zeros_(self.recovery_gate_head.bias)
        # Per-node graph-recovery rate head kappa_v. Zero-initialized
        # weight + zero bias => sigmoid(0)=0.5 => kappa starts mid-range so the
        # supply-constrained relaxation is active from step 1 and the backbone
        # learns per-node recovery speed. Adds (d+1) params. kappa is mapped to
        # [PHYSREC_KAPPA_MIN, PHYSREC_KAPPA_MAX] per day in the forward pass.
        if self.physrecover_enabled:
            self.physrec_kappa_head = nn.Linear(d, 1)
            with torch.no_grad():
                nn.init.zeros_(self.physrec_kappa_head.weight)
                nn.init.zeros_(self.physrec_kappa_head.bias)
            # kappa_v range (per day). At gain=1-exp(-kappa*dt) a dt=30d gap
            # gives gain 0.14 (kappa=0.005) .. 0.99 (kappa=0.15): spans slow
            # supply-limited refill to near-instant rebound. Overridable via env.
            self.physrec_kappa_min = float(getattr(cfg, "physrec_kappa_min", 0.005)) if cfg is not None else 0.005
            self.physrec_kappa_max = float(getattr(cfg, "physrec_kappa_max", 0.15)) if cfg is not None else 0.15
        # Per-node recovery-sharpness head q_v. The bias maps to q==1.0 at init,
        # yielding a single-exponential recovery before learning
        # to deviate per node. Adds (d+1) params. q is mapped to
        # [recovery_q_min, recovery_q_max] in the forward pass.
        if self.recovery_q_enabled:
            self.recovery_q_min = float(getattr(cfg, "recovery_q_min", 0.5)) if cfg is not None else 0.5
            self.recovery_q_max = float(getattr(cfg, "recovery_q_max", 2.0)) if cfg is not None else 2.0
            self.recovery_q_head = nn.Linear(d, 1)
            with torch.no_grad():
                nn.init.zeros_(self.recovery_q_head.weight)
                # solve sigmoid(b) = (1 - q_min)/(q_max - q_min) so q starts at 1.0
                _span = max(self.recovery_q_max - self.recovery_q_min, 1e-6)
                _frac = min(max((1.0 - self.recovery_q_min) / _span, 1e-4), 1 - 1e-4)
                self.recovery_q_head.bias.fill_(math.log(_frac / (1.0 - _frac)))
        # Variant C: hit-gate head
        if peak_mode == "hitgate":
            self.hit_head = nn.Sequential(
                nn.Linear(d, d), nn.GELU(),
                nn.Linear(d, 1),
            )
            with torch.no_grad():
                # data prior: ~25% of unshocked nodes hit (peak>0.05)
                self.hit_head[-1].bias.fill_(math.log(0.25 / 0.75))
        # peak heads (used by direct/blend)
        self.peak_direct_head = nn.Sequential(
            nn.Linear(d + self.TRAJ_DIM, d), nn.GELU(),
            nn.Linear(d, 1),
        )
        with torch.no_grad():
            # init bias so sigmoid output ≈ 0.05 (data prior: 75% nodes peak<0.05)
            self.peak_direct_head[-1].bias.fill_(math.log(0.05 / 0.95))
        # per-node gate. Input: [h, shock, delta0]
        # Init bias = logit(0.30) → unshocked starts at g≈0.30 (70% direct)
        self.gate_head = nn.Sequential(
            nn.Linear(d + 2, d), nn.GELU(),
            nn.Linear(d, 1),
        )
        with torch.no_grad():
            self.gate_head[-1].bias.fill_(math.log(0.30 / 0.70))
        # auxiliary heads
        self.reach_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, 1),
        )
        self.active_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, 1),
        )
        # Cascade-depth aux head: predict per-node trough_day (K-class CE).
        # Forces encoder features to encode "when does this node bottom out"
        # → improves per-node identifiability at d30/d50 trough zone.
        K_kf = len(getattr(cfg, "key_days", [0, 5, 10, 20, 30, 50, 70, 100, 150, 199]))
        self.trough_day_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, K_kf),
        )
        # ----- MoE: 3-way regime router (early/mid/late) -----
        # Only active when moe_router=True (intended with decoder_mode="param3").
        # Output: softmax(3) per node → scales (P1, P2, P3) via amplitude gate.
        # Per-node routing decouples amplitude allocation across regimes.
        if self.moe_router:
            self.regime_router_head = nn.Sequential(
                nn.Linear(d, d), nn.GELU(),
                nn.Linear(d, 3),
            )
            with torch.no_grad():
                # Init uniform: bias=0 → softmax ≈ [1/3,1/3,1/3]. Last-layer
                # weight kept default-init so the routing has to LEARN
                # discrimination rather than starting biased.
                self.regime_router_head[-1].bias.zero_()

    def _physrecover_rollout(self, u_kf_param, h_for_route, days,
                             edge_src, edge_dst, edge_a):
        """Graph-coupled inventory recovery integrator.

        Replaces the closed-form recovery branch (t > trough) of u_kf_param with
        an on-graph relaxation faithful to the simulators' production rule
        P_act = min(P_cap, P_prop=Leontief(S), D):

            supply_v(t) = ( Σ_{e: dst=v} a_e · u[src_e](t) ) / ( Σ_{e: dst=v} a_e )
            target_v(t) = min( own_recovery_v(t),  supply_v(t) )
            u_v(t+dt)   = u_v(t) + (1 - exp(-kappa_v·dt)) · (target_v(t) - u_v(t))

        own_recovery_v(t) = the decoder's existing closed-form recovery value
        (keeps the interpretable mu/sigma/A baseline). supply_v(t) caps a node's
        recovery by its UPSTREAM suppliers' current production (the "wait for my
        suppliers" coupling). The (1-exp(-kappa·dt)) gain is the exact solution
        of du/dt = kappa(target-u) over a keyframe gap dt, unconditionally stable
        and bounded in [u_prev, target]. Only frames strictly after each node's
        trough (argmin of the closed-form curve) are overridden; decline frames
        are left identical to the closed form. SOLE recovery path (no blend).

        Args:
          u_kf_param  : (K, Nr) closed-form trajectory (decline + own recovery)
          h_for_route : (Nr, d) post-GAT hidden for the kappa head
          days        : (K,) keyframe days
          edge_src    : (E,) supplier index   (src = supplier, dst = buyer)
          edge_dst    : (E,) buyer index
          edge_a      : (E,) positive edge weight (economic input share proxy)
        Returns:
          u : (K, Nr) with the recovery branch replaced by the integrator.
        """
        K, Nr = u_kf_param.shape
        device = u_kf_param.device
        eps = 1e-6
        # per-node recovery rate kappa_v in [k_min, k_max] / day
        k_min, k_max = self.physrec_kappa_min, self.physrec_kappa_max
        kappa = k_min + (k_max - k_min) * torch.sigmoid(
            self.physrec_kappa_head(h_for_route).squeeze(-1))           # (Nr,)
        # denominator Σ a_e over suppliers of each buyer (constant across t)
        den = torch.zeros(Nr, device=device, dtype=u_kf_param.dtype)
        den = den.index_add_(0, edge_dst, edge_a)                       # (Nr,)
        has_sup = den > eps
        # per-node trough day (detached; just selects which frames to override)
        with torch.no_grad():
            k_tr = torch.argmin(u_kf_param, dim=0)                      # (Nr,)
            mu_eff = days[k_tr]                                         # (Nr,)
        u_rows = [u_kf_param[0]]
        for k in range(1, K):
            dt = (days[k] - days[k - 1]).clamp_min(0.0)
            u_prev = u_rows[-1]                                         # (Nr,)
            # supply ceiling = input-share-weighted mean of suppliers' u_prev
            num = torch.zeros(Nr, device=device, dtype=u_kf_param.dtype)
            num = num.index_add_(0, edge_dst, edge_a * u_prev[edge_src])
            supply = torch.where(has_sup, num / (den + eps),
                                 torch.ones_like(num))                  # (Nr,)
            own = u_kf_param[k]                                         # own closed-form recovery target
            target = torch.minimum(own, supply)
            gain = 1.0 - torch.exp(-kappa * dt)
            u_new = (u_prev + gain * (target - u_prev)).clamp(0.0, 1.0)
            rec_mask = (days[k] > mu_eff).to(u_kf_param.dtype)         # (Nr,) 1 after trough
            u_k = rec_mask * u_new + (1.0 - rec_mask) * u_kf_param[k]
            u_rows.append(u_k)
        return torch.stack(u_rows, dim=0)                              # (K, Nr)

    def _predict_u0(self, h0, shock, delta0,
                    edge_src=None, edge_dst=None, edge_outshare=None, Nr=None):
        """Compute the Day-0 production-ratio boundary.

        The manuscript profile uses the capacity boundary
        u0 = 1 - shock * delta0. ``day0_demand_pullback`` is an optional
        compatibility setting; when enabled, it also
        applies an output-share-weighted pullback from damaged customers. Both
        paths are deterministic and contain no learned parameters.
        """
        u_cap = (1.0 - delta0 * shock).clamp(0.0, 1.0)
        if (not self.day0_demand_pullback or edge_outshare is None
                or edge_src is None or edge_dst is None):
            return u_cap
        if Nr is None:
            Nr = h0.shape[0]
        s_dmg = delta0 * shock                                   # (Nr,)
        # Pullback from damaged buyers (j = edge_dst) to suppliers (i = edge_src).
        # NOTE: only damaged buyers contribute (s_dmg[j] = δ_j when shock_j=1,
        # else 0), so the sum is naturally restricted to the right set.
        pullback = torch.zeros(Nr, device=h0.device, dtype=h0.dtype)
        pullback.index_add_(0, edge_src, edge_outshare * s_dmg[edge_dst])
        u_dem = (1.0 - pullback).clamp(0.0, 1.0)
        return torch.minimum(u_cap, u_dem)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        x_v = batch["x_v"]
        shock = batch["shock_mask"]
        delta0 = batch["delta0"]
        evt = batch["event_scalars"]
        edge_src = batch["edge_src"]
        edge_dst = batch["edge_dst"]
        edge_a = batch["edge_a"]
        # Henriet eq-4 supplier-output share (data.py provides; older batches
        # without this field gracefully degrade _predict_u0 to capacity-only).
        edge_outshare = batch.get("edge_outshare", None)
        key_days = batch["key_days"]                        # (K,) int

        Nr = x_v.shape[0]
        # Build hop feature in the chosen encoding (computed from one-hot input)
        hop_feat = None
        if self.hop_encoding != "none":
            hop_oh = batch.get("shock_hop_oh", None)
            assert hop_oh is not None, \
                "hop_encoding != 'none' requires batch['shock_hop_oh']"
            if self.hop_encoding == "onehot5":
                hop_feat = hop_oh
            elif self.hop_encoding == "graded2":
                # is_hop1  = column 1 of one-hot (Nr,)
                # graded   = sum_k oh[:,k] * 1/(1+k)
                is_hop1 = hop_oh[:, 1:2]                       # (Nr,1)
                table = torch.tensor(self._GRADED_TABLE,
                                     device=hop_oh.device,
                                     dtype=hop_oh.dtype)         # (5,)
                graded = (hop_oh * table).sum(dim=-1, keepdim=True)  # (Nr,1)
                hop_feat = torch.cat([is_hop1, graded], dim=-1)     # (Nr,2)
        # Ablation: unconditioned representation — strip Day-0 shock identity
        # (shock_mask & delta0) from the ENCODER input only. u0 physical IC and
        # FiLM shock_frac below still see the true shock.
        if self.ablate_uncond:
            enc_shock = torch.zeros_like(shock)
            enc_delta0 = torch.zeros_like(delta0)
        else:
            enc_shock = shock
            enc_delta0 = delta0
        h = self.encoder(x_v, enc_shock, enc_delta0, evt, hop_feat=hop_feat)  # (Nr,d)

        # ---- FiLM conditioning on scale-invariant graph statistics ----
        # Uses dimensionless degree-distribution shape features only.
        if self.use_film:
            Nr_f = float(x_v.shape[0])
            E_f = float(edge_src.shape[0])
            in_deg_local = torch.zeros(int(Nr_f), device=h.device, dtype=h.dtype)
            in_deg_local.index_add_(0, edge_dst,
                                    torch.ones_like(edge_dst, dtype=h.dtype))
            log_in = torch.log1p(in_deg_local)
            mean_log_in = float(log_in.mean().item())
            std_log_in = float(log_in.std(unbiased=False).item())
            max_log_in = float(log_in.max().item())
            shock_frac = float(shock.mean().item())
            stats = torch.tensor([
                math.log(max(E_f, 1.0) / max(Nr_f, 1.0) + 1e-6),  # log mean in_deg
                shock_frac,
                mean_log_in,
                std_log_in,
                max_log_in - mean_log_in,                          # hub skewness
            ], device=h.device, dtype=h.dtype)
            gb = self.film_net(stats)                                 # (2d,)
            d_h = h.shape[-1]
            gamma = gb[:d_h].view(1, -1)                              # (1, d)
            beta = gb[d_h:].view(1, -1)
            h = gamma * h + beta

        # ---- step 0 (day 0) ----
        u0 = self._predict_u0(h, shock, delta0,
                              edge_src=edge_src, edge_dst=edge_dst,
                              edge_outshare=edge_outshare, Nr=Nr)

        K = key_days.shape[0]
        days = key_days.to(h.dtype)
        # Defaults make every decoder branch well defined and keep diagnostics
        # available for rollout-only ablations.
        h_param = h
        u_ss_v = torch.ones(Nr, device=h.device, dtype=h.dtype)

        # MoE outputs (only set when decoder_mode=param3 AND moe_router=True).
        moe_route_logits = None
        moe_route = None

        # ---- Variant D: spatial pre-warm (run once, before rollout/decoder) ----
        if self.prewarm_layers > 0:
            for gat, norm in zip(self.prewarm_gat, self.prewarm_norms):
                msg_in, msg_out = gat(h, u0, edge_src, edge_dst, edge_a, Nr)
                if self.ablate_forward_only:
                    msg_out = torch.zeros_like(msg_out)
                h = norm(h + msg_in + msg_out)

        # =====================================================================
        # Variant A / E / E+blend: parametric decoder (with optional rollout blend)
        # =====================================================================
        if self.decoder_mode in ("param", "param2", "param2_blend", "param3"):
            # For blend_rollout we MUST keep `h` pristine because the parallel
            # rollout below consumes it; the param branch runs on a clone.
            # For pure param/param2/param3 (no blend) we update `h` in place so
            # downstream heads see the three-layer spatial features.
            if self.blend_rollout:
                h_param = h
                for gat, norm in zip(self.param_gat, self.param_gat_norms):
                    msg_in, msg_out = gat(h_param, u0, edge_src, edge_dst, edge_a, Nr)
                    if self.ablate_forward_only:
                        msg_out = torch.zeros_like(msg_out)
                    h_param = norm(h_param + msg_in + msg_out)
                raw = self.param_head(h_param)
                h_for_route = h_param
            else:
                for gat, norm in zip(self.param_gat, self.param_gat_norms):
                    msg_in, msg_out = gat(h, u0, edge_src, edge_dst, edge_a, Nr)
                    if self.ablate_forward_only:
                        msg_out = torch.zeros_like(msg_out)
                    h = norm(h + msg_in + msg_out)
                raw = self.param_head(h)
                h_for_route = h
            # Per-node recovery gate g_v in [0,1] (None when
            # disabled => decoder falls back to the scalar recovery_p path).
            rec_gate = None
            if getattr(self, "recovery_gate_enabled", False):
                rec_gate = torch.sigmoid(self.recovery_gate_head(h_for_route)).squeeze(-1)  # (Nr,)
            # Per-node recovery sharpness q_v in
            # [q_min, q_max] => r(z)=exp(-z^q). None when disabled (single-exp).
            rec_q = None
            if getattr(self, "recovery_q_enabled", False):
                _qs = torch.sigmoid(self.recovery_q_head(h_for_route)).squeeze(-1)  # (Nr,)
                rec_q = self.recovery_q_min + (self.recovery_q_max - self.recovery_q_min) * _qs
            if self.decoder_mode == "param":
                P_v = torch.sigmoid(raw[:, 0])
                mu_v = 50.0 * torch.sigmoid(raw[:, 1])
                sigma_v = 2.0 + 28.0 * torch.sigmoid(raw[:, 2])
                T_r_v = 50.0 + 250.0 * torch.sigmoid(raw[:, 3])
                u_ss_v = torch.sigmoid(raw[:, 4])
                u_kf_param = _reconstruct_trajectory(P_v, mu_v, sigma_v, T_r_v,
                                                     days, u_ss=u_ss_v,
                                                     collapse_p=self.decline_p,
                                                     recovery_p=self.recovery_p,
                                                     recovery_gate=rec_gate,
                                                     recovery_q=rec_q)
                P_max_v = P_v
            elif self.decoder_mode == "param3":
                P1_v = torch.sigmoid(raw[:, 0])
                mu1_v = self.p3_mu1_off + self.p3_mu1_scale * torch.sigmoid(raw[:, 1])
                sigma1_v = self.p3_sigma1_off + self.p3_sigma1_scale * torch.sigmoid(raw[:, 2])
                T_r1_v = 50.0 + 250.0 * torch.sigmoid(raw[:, 3])
                P2_v = torch.sigmoid(raw[:, 4])
                mu2_v = self.p3_mu2_off + self.p3_mu2_scale * torch.sigmoid(raw[:, 5])
                sigma2_v = self.p3_sigma2_off + self.p3_sigma2_scale * torch.sigmoid(raw[:, 6])
                T_r2_v = 50.0 + 250.0 * torch.sigmoid(raw[:, 7])
                P3_v = torch.sigmoid(raw[:, 8])
                mu3_v = self.p3_mu3_off + self.p3_mu3_scale * torch.sigmoid(raw[:, 9])
                sigma3_v = self.p3_sigma3_off + self.p3_sigma3_scale * torch.sigmoid(raw[:, 10])
                T_r3_v = 50.0 + 250.0 * torch.sigmoid(raw[:, 11])
                u_ss_v = torch.sigmoid(raw[:, 12])
                # ----- MoE amplitude gating -----
                # When moe_router=True, scale (P1, P2, P3) by 3-way route weights
                # so each regime owns a Gaussian. During warmup, route is forced
                # uniform so all 3 Gaussians receive equal gradient.
                if self.moe_router:
                    route_logits = self.regime_router_head(h_for_route)  # (Nr, 3)
                    if self.moe_warmup_active:
                        route = torch.full_like(route_logits, 1.0 / 3.0)
                    else:
                        route = torch.softmax(route_logits, dim=-1)
                    # Per-node amplitude gate: P_eff = P × route[k], bounded
                    # in [0, 1]. During uniform warmup, P_eff = P/3.
                    P1_v = P1_v * route[:, 0]
                    P2_v = P2_v * route[:, 1]
                    P3_v = P3_v * route[:, 2]
                    moe_route_logits = route_logits
                    moe_route = route
                else:
                    moe_route_logits = None
                    moe_route = None
                u_kf_param = _reconstruct_trajectory_trimodal(
                    P1_v, mu1_v, sigma1_v, T_r1_v,
                    P2_v, mu2_v, sigma2_v, T_r2_v,
                    P3_v, mu3_v, sigma3_v, T_r3_v, days,
                    u_ss=u_ss_v, tau=self._get_seam_tau(h_for_route),
                    collapse_p=self.decline_p,
                    recovery_p=self.recovery_p,
                    recovery_gate=rec_gate,
                    recovery_q=rec_q,
                    tau2=self._get_seam_tau2(h_for_route),
                )
                P_max_v = torch.maximum(torch.maximum(P1_v, P2_v), P3_v)
            else:    # param2 or param2_blend
                P1_v = torch.sigmoid(raw[:, 0])
                mu1_v = self.p2_mu1_off + self.p2_mu1_scale * torch.sigmoid(raw[:, 1])
                sigma1_v = self.p2_sigma1_off + self.p2_sigma1_scale * torch.sigmoid(raw[:, 2])
                T_r1_v = 50.0 + 250.0 * torch.sigmoid(raw[:, 3])
                P2_v = torch.sigmoid(raw[:, 4])
                mu2_v = self.p2_mu2_off + self.p2_mu2_scale * torch.sigmoid(raw[:, 5])
                sigma2_v = self.p2_sigma2_off + self.p2_sigma2_scale * torch.sigmoid(raw[:, 6])
                T_r2_v = 50.0 + 250.0 * torch.sigmoid(raw[:, 7])
                u_ss_v = torch.sigmoid(raw[:, 8])
                u_kf_param = _reconstruct_trajectory_bimodal(
                    P1_v, mu1_v, sigma1_v, T_r1_v,
                    P2_v, mu2_v, sigma2_v, T_r2_v, days,
                    u_ss=u_ss_v, tau=self._get_bimodal_tau(),
                    collapse_p=self.decline_p,
                    recovery_p=self.recovery_p,
                    recovery_gate=rec_gate,
                    recovery_q=rec_q,
                )
                P_max_v = torch.maximum(P1_v, P2_v)
            # Zero-initialized residual: u = u_analytical + delta_free. The
            # analytical component parameters remain available in the output.
            if getattr(self, "free_residual", False):
                delta_free = self.kf_residual_head(h_for_route).t()    # (K, Nr)
                if getattr(self, "resid_tail_mask", False):
                    # Zero the residual on the recovery tail (days>resid_mask_day)
                    # so it cannot overfit/cannibalize d70+ (see __init__ note).
                    day_mask = (days <= self.resid_mask_day).to(delta_free.dtype).view(-1, 1)  # (K,1)
                    delta_free = delta_free * day_mask
                u_kf_param = (u_kf_param + delta_free).clamp(0.0, 1.0)
            # ---- graph-coupled recovery integrator ----
            # Replaces the recovery branch (frames after each node's trough) with
            # an on-graph relaxation u_v(t+dt)=u_v+(1-e^{-kappa_v dt})(min(own,
            # supply_v)-u_v), supply_v = input-share-weighted mean of suppliers'
            # u. This is the SOLE recovery path (no blend), addressing the d70+
            # collapse the per-node closed form structurally cannot express.
            # Decline frames are left identical to the closed form. Applied after
            # free_residual (so the trough it seeds from reflects any residual)
            # and only in the non-blend decoder path (BD / edge-state+free_residual).
            if getattr(self, "physrecover_enabled", False):
                u_kf_param = self._physrecover_rollout(
                    u_kf_param, h_for_route, days,
                    edge_src, edge_dst, edge_a)
            if self.blend_rollout:
                # Run rollout in parallel (uses original h, NOT h_param to keep
                # the two paths independent so alpha_k blend is meaningful).
                u_kf_roll = [u0]
                if self.step.use_inv_state:
                    s_prev = (1.0 - delta0).clamp(0.0, 1.0)
                    o_prev = torch.zeros_like(s_prev)
                else:
                    s_prev = o_prev = None
                for k in range(1, K):
                    dt_days = days[k] - days[k - 1]
                    dt_norm = dt_days / self.DT_NORM
                    # Extra spatial diffusion proportional to the interval.
                    n_extra = min(self.diffusion_max_extra,
                                  max(0, int(round(float(dt_days) / self.D_GAT_DAYS)) - 1))
                    u_prev = u_kf_roll[-1]
                    for _ in range(n_extra):
                        m_in_d, m_out_d = self.diffusion_gat(h, u_prev, edge_src, edge_dst, edge_a, Nr)
                        h = self.diffusion_norm(h + m_in_d + m_out_d)
                    if self.aggr_mode == "phase":
                        block = (self.collapse_block if k <= self.phase_switch_k
                                 else self.recovery_block)
                        if self.step.use_inv_state:
                            h, u, s_prev, o_prev = self.step(h, u_kf_roll[-1], shock, delta0, dt_norm,
                                             edge_src, edge_dst, edge_a, Nr,
                                             spatial_block=block,
                                             s_prev=s_prev, o_prev=o_prev, dt_days=dt_days)
                        else:
                            h, u = self.step(h, u_kf_roll[-1], shock, delta0, dt_norm,
                                             edge_src, edge_dst, edge_a, Nr,
                                             spatial_block=block)
                    else:
                        if self.step.use_inv_state:
                            h, u, s_prev, o_prev = self.step(h, u_kf_roll[-1], shock, delta0, dt_norm,
                                             edge_src, edge_dst, edge_a, Nr,
                                             s_prev=s_prev, o_prev=o_prev, dt_days=dt_days)
                        else:
                            h, u = self.step(h, u_kf_roll[-1], shock, delta0, dt_norm,
                                             edge_src, edge_dst, edge_a, Nr)
                    u_kf_roll.append(u)
                u_kf_roll_t = torch.stack(u_kf_roll, dim=0)            # (K, Nr)
                if self.triple_blend:
                    # ---- three-stream per-keyframe softmax fusion ----
                    # Direct head consumes h_param (same latent as trimodal
                    # decoder) so all three paths share the SAME post-encoder
                    # view of the graph — the only difference is their
                    # inductive bias on u(t) shape.
                    u_kf_direct = torch.sigmoid(
                        self.kf_direct_head(h_param)).t()              # (K, Nr)
                    # Enforce boundary u(t=0) = u0 for all paths (matches the
                    # 2-way blend's `alpha[0] = 1.0` convention).
                    w = torch.softmax(self.blend_logits, dim=-1)        # (K, 3)
                    w_param  = w[:, 0:1]
                    w_roll   = w[:, 1:2]
                    w_direct = w[:, 2:3]
                    u_kf_blend = (w_param  * u_kf_param +
                                  w_roll   * u_kf_roll_t +
                                  w_direct * u_kf_direct)
                    u_kf_t = torch.cat(
                        [u0.unsqueeze(0), u_kf_blend[1:]], dim=0)
                else:
                    # Per-frame learnable gate.  alpha[0] forced to 1.0 to keep u0
                    # purely from the shock-derived predictor.
                    alpha = torch.sigmoid(self.alpha_kf).view(-1, 1)        # (K, 1)
                    alpha = alpha.clone()
                    alpha[0] = 1.0
                    u_kf_t = alpha * u_kf_roll_t + (1.0 - alpha) * u_kf_param
            else:
                u_kf_t = torch.cat([u0.unsqueeze(0), u_kf_param[1:]], dim=0)
        else:
            # =================================================================
            # Rollout path: K-step GAT + GRU + per-step u readout
            # phase mode: minplus for k <= phase_switch_k, softmax otherwise
            # =================================================================
            u_kf = [u0]
            if self.step.use_inv_state:
                s_prev = (1.0 - delta0).clamp(0.0, 1.0)
                o_prev = torch.zeros_like(s_prev)
            else:
                s_prev = o_prev = None
            for k in range(1, K):
                dt_days = days[k] - days[k - 1]
                dt_norm = dt_days / self.DT_NORM
                # Extra spatial diffusion proportional to the interval.
                n_extra = min(self.diffusion_max_extra,
                              max(0, int(round(float(dt_days) / self.D_GAT_DAYS)) - 1))
                u_prev = u_kf[-1]
                for _ in range(n_extra):
                    m_in_d, m_out_d = self.diffusion_gat(h, u_prev, edge_src, edge_dst, edge_a, Nr)
                    h = self.diffusion_norm(h + m_in_d + m_out_d)
                if self.aggr_mode == "phase":
                    block = (self.collapse_block if k <= self.phase_switch_k
                             else self.recovery_block)
                    if self.step.use_inv_state:
                        h, u, s_prev, o_prev = self.step(h, u_kf[-1], shock, delta0, dt_norm,
                                         edge_src, edge_dst, edge_a, Nr,
                                         spatial_block=block,
                                         s_prev=s_prev, o_prev=o_prev, dt_days=dt_days)
                    else:
                        h, u = self.step(h, u_kf[-1], shock, delta0, dt_norm,
                                         edge_src, edge_dst, edge_a, Nr,
                                         spatial_block=block)
                else:
                    if self.step.use_inv_state:
                        h, u, s_prev, o_prev = self.step(h, u_kf[-1], shock, delta0, dt_norm,
                                         edge_src, edge_dst, edge_a, Nr,
                                         s_prev=s_prev, o_prev=o_prev, dt_days=dt_days)
                    else:
                        h, u = self.step(h, u_kf[-1], shock, delta0, dt_norm,
                                         edge_src, edge_dst, edge_a, Nr)
                u_kf.append(u)
            u_kf_t = torch.stack(u_kf, dim=0)                   # (K, Nr)

        # ---- traj branch: (soft-)min peak on rollout ----
        # hard min: gradient flows ONLY to argmin keyframe, leaving d5/d199
        # unsupervised by peak → they overshoot and pollute R²kf.
        # soft-min (tau>0): -tau * logsumexp(-u/tau, dim=0).
        # gradient flows to all frames ∝ softmax(-u/tau); recovers hard min as tau→0.
        if self.softmin_tau > 0.0:
            tau = self.softmin_tau
            min_u = -tau * torch.logsumexp(-u_kf_t / tau, dim=0)
        else:
            min_u = u_kf_t.min(dim=0).values
        peak_traj = (1.0 - min_u).clamp(0.0, 1.0)

        # ---- direct branch: peak from h_final + trajectory summary ----
        traj_summary = torch.stack([
            min_u,
            u_kf_t.mean(dim=0),
            u_kf_t[-1],
            u_kf_t.argmin(dim=0).to(h.dtype) / max(K - 1, 1),
            u_kf_t.std(dim=0),
        ], dim=-1)                                          # (Nr, 5)
        peak_direct = torch.sigmoid(
            self.peak_direct_head(torch.cat([h, traj_summary], dim=-1)).squeeze(-1)
        )

        # ---- per-node gate (structurally pinned at 1 for shocked) ----
        gate_in = torch.cat([h, shock.unsqueeze(-1), delta0.unsqueeze(-1)], dim=-1)
        g_neural = torch.sigmoid(self.gate_head(gate_in).squeeze(-1))
        gate = shock + (1.0 - shock) * g_neural             # shocked → 1.0

        # ---- final peak by mode ----
        if self.peak_mode == "traj":
            peak = peak_traj
            gate_out = torch.ones_like(peak)
        elif self.peak_mode == "direct":
            peak = peak_direct
            gate_out = torch.zeros_like(peak)
        elif self.peak_mode == "hitgate":
            # Variant C: per-node hit probability gates the trajectory peak.
            # Shocked nodes are pinned to p_hit=1 (always trust traj branch).
            p_hit = torch.sigmoid(self.hit_head(h).squeeze(-1))
            p_hit_eff = shock + (1.0 - shock) * p_hit
            peak = (p_hit_eff * peak_traj).clamp(0.0, 1.0)
            gate_out = p_hit_eff
        else:  # blend
            peak = (gate * peak_traj + (1.0 - gate) * peak_direct).clamp(0.0, 1.0)
            gate_out = gate

        # In param-decoder mode the peak comes directly from P (overrides traj).
        # Physics prior: shocked nodes have peak >= delta0 (initial damage is
        # an immediate observed loss).  Without this guard the param_head
        # would need many epochs to lift P_v from its 0.05 init for shocked
        # nodes, transiently crashing R²pk_shk.
        if self.decoder_mode in ("param", "param2", "param3"):
            # Pure parametric decode: peak from closed-form trajectory min
            # (accounts for both collapse depth P and absorbing steady state 1-u_ss).
            peak_param = (1.0 - u_kf_t.min(dim=0).values).clamp(0.0, 1.0)
            peak = torch.maximum(peak_param, delta0 * shock).clamp(0.0, 1.0)
            gate_out = torch.ones_like(peak)
        elif self.decoder_mode == "param2_blend":
            # Blended decode: peak comes from the blended u_kf min (traj-mode).
            # Still apply the shock floor so R^2pk_shk does not regress.
            peak_traj_blend = (1.0 - u_kf_t.min(dim=0).values).clamp(0.0, 1.0)
            peak = torch.maximum(peak_traj_blend, delta0 * shock).clamp(0.0, 1.0)
            gate_out = torch.ones_like(peak)

        # ---- aux heads ----
        reach_logit = self.reach_head(h).squeeze(-1)
        active_logit = self.active_head(h).squeeze(-1)

        # ---- Compute SCALE-INVARIANT graph statistics for PhysicsParameterGenerator ----
        Nr_f = float(x_v.shape[0])
        E_f = float(edge_src.shape[0])
        in_deg_local = torch.zeros(int(Nr_f), device=h.device, dtype=h.dtype)
        in_deg_local.index_add_(0, edge_dst,
                                torch.ones_like(edge_dst, dtype=h.dtype))
        log_in = torch.log1p(in_deg_local)
        mean_log_in = float(log_in.mean().item())
        std_log_in = float(log_in.std(unbiased=False).item())
        max_log_in = float(log_in.max().item())
        shock_frac = float(shock.mean().item())
        g_stats = torch.tensor([
            math.log(max(E_f, 1.0) / max(Nr_f, 1.0) + 1e-6),  # log mean in_deg
            shock_frac,
            mean_log_in,
            std_log_in,
            max_log_in - mean_log_in,                          # hub skewness
        ], device=h.device, dtype=h.dtype)

        # ---- Learn physics parameters (τ_u, c) from graph context ----
        tau_u, c = self.phys_param_gen(h, g_stats)

        # params tuple for compatibility with loss functions:
        # (P_v, mu_v, tau_u, c, T_r_v) - shape (Nr,) each
        # For now we use stubs for P/mu/T_r since they come from param decoder
        # but tau_u and c are learned from graph statistics.
        ones = torch.ones(Nr, device=h.device, dtype=h.dtype)
        params = (ones * 6.0, ones, tau_u, c, ones * 150.0)

        # Diagnostics: store parameter statistics for monitoring
        u_ss_diag = u_ss_v
        # Cascade-depth aux: per-node K-way classification of trough_day.
        # Use post-param-GAT h (richest features). For param2_blend mode the
        # param branch ran on h_param (a clone) — but the main `h` here has
        # been updated by either the param GATs (param/param2) or the rollout
        # path (param2_blend), both reflect post-spatial features.
        trough_day_logit = self.trough_day_head(h)                 # (Nr, K)
        out_dict = dict(
            u_keyframes=u_kf_t,
            u_full=u_kf_t,                                  # alias (no daily grid in v3)
            peak=peak, peak_pred=peak, peak_phys=peak,
            peak_phys_raw=peak_traj,
            peak_traj=peak_traj,
            peak_direct=peak_direct,
            gate=gate_out,
            reach_logit=reach_logit,
            active_logit=active_logit,
            trough_day_logit=trough_day_logit,
            params=params,
            h_star=h, h_final=h,
            blend_lambda=torch.zeros((), device=h.device),
            phys_scale=torch.ones_like(peak),
            # Learned-parameter monitoring outputs.
            tau_u_learned=tau_u,
            c_learned=c,
            u_ss_learned=u_ss_diag,
            g_stats_computed=g_stats,
        )
        if moe_route_logits is not None:
            out_dict["regime_route_logits"] = moe_route_logits
            out_dict["regime_route"] = moe_route
        return out_dict

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _get_bimodal_tau(self):
        """Return the current bimodal/trimodal softmin temperature.

        - Fixed mode (default):    Python float (0 = hard min, >0 = soft min).
        - Learnable mode: 0-d Tensor via softplus(_log_bimodal_tau)
          with a 1e-4 floor; gradients flow back to the parameter so the
          model can self-tune the smoothness of the d10/d70 seam.
        """
        if self.learnable_bimodal_tau:
            return F.softplus(self._log_bimodal_tau) + 1e-4
        return self.bimodal_tau

    def _get_seam_tau(self, h):
        """Per-node softmin temperature τ_v from the node embedding.

        Returns a (Nr,) tensor τ_v = softplus(W·h + b) + 1e-3 so each node
        controls how sharply its trimodal Gaussians compose at the d50/d70
        seams (large τ -> graceful blend preserving node rank; small τ ->
        hard min). Broadcasts over (3, K, Nr) in the softmin (last-dim align).
        Falls back to the scalar bimodal τ if the head is disabled.
        """
        if getattr(self, "seam_tau_pernode", False):
            raw = self.seam_tau_head(h).squeeze(-1)        # (Nr,)
            return F.softplus(raw) + 1e-3
        return self._get_bimodal_tau()

    def _get_seam_tau2(self, h):
        """Second per-node softmin temperature τ2_v for the G2/G3 (d70) seam.

        Only meaningful when seam_tau_dual is on; returns None otherwise so the
        trimodal reconstruction falls back to the single-tau path. Same
        softplus+floor parameterisation as the first head.
        """
        if getattr(self, "seam_tau_dual", False):
            raw = self.seam_tau2_head(h).squeeze(-1)       # (Nr,)
            return F.softplus(raw) + 1e-3
        return None

    def set_moe_warmup(self, active: bool) -> None:
        """Toggle MoE router warmup. When active, route is forced uniform
        [1/3,1/3,1/3] so all 3 Gaussian amplitudes receive equal gradient.
        Called by training loop based on current epoch."""
        self.moe_warmup_active = bool(active)
