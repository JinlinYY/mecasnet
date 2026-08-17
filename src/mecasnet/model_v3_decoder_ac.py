"""Experimental MeCaSNet decoder variants targeting the R^2_kf transition-frame
weakness exposed by the DirGNN baseline (DirGNN beats MeCaSNet on R2_kf_csc,
driven almost entirely by keyframes d10 and d70 — the early-decline slope and
the early->late Gaussian seam, where the closed-form bimodal trajectory is
structurally rigid).

Two variants, both reusing the trained-flagship recipe (g3_narrow param2 +
3-layer prewarm + residual idea) so the only thing that changes is HOW the
transition frames are corrected:

  A. KSGATv3ResidualA  — same late-residual head as the flagship V3DE_G3_RES,
     but (1) the fixed alpha(d) gate opens earlier at d10 (0.05 -> 0.40) so the
     residual can fix the early-decline slope, and (2) the residual head is
     widened (d -> 2d -> K) so it has the capacity to learn per-node offsets in
     the transition band. Everything else identical to V3DE_G3_RES.

  C. KSGATv3BlendC     — physics-prior bimodal decoder run in PARALLEL with a
     free GRU rollout, fused per-keyframe by a learnable gate alpha_k:
         u_k = sigmoid(alpha_k) * u_rollout_k + (1 - .) * u_param_k
     This is the existing `blend_rollout` path in KSGATv3 (param2_blend), simply
     surfaced as a named variant. The physics prior owns the peak frames; the
     free rollout owns the transition frames; the gate learns which per frame.
     Directly turns the DirGNN finding into an architecture: keep MeCaSNet's
     peak accuracy, borrow DirGNN's trajectory flexibility.

Both keep the exact forward I/O contract (drop into train_v3de / _runner /
losses). Param budgets stay ~230k (csmar) so comparison vs flagship is fair.
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

from .model_v3 import KSGATv3


# ===========================================================================
# Variant A — earlier-gated, wider residual head
# ===========================================================================
class KSGATv3ResidualA(KSGATv3):
    """V3DE_G3_RES with an earlier-opening alpha(d) gate + wider residual head.

    Motivation: the flagship gate is alpha(d10)=0.05 (residual barely active at
    d10) and alpha(d70)=0.97 (active, but the shared d->d->K head lacks capacity
    for the seam). DirGNN beats us by +0.60 at d10 and +0.55 at d70. We open d10
    to 0.40 and widen the head to d->2d->K.
    """

    def __init__(self, cfg, Fv: int, **kwargs):
        super().__init__(cfg, Fv=Fv, **kwargs)
        d = int(cfg.d_hidden)
        K = len(cfg.key_days)
        assert K == 10, f"alpha_kf_residual hard-coded for K=10, got {K}"

        # WIDER residual head: d -> 2d -> K (flagship was d -> d -> K)
        self.late_residual_head = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(),
            nn.Linear(2 * d, K),
        )
        nn.init.zeros_(self.late_residual_head[-1].weight)
        nn.init.zeros_(self.late_residual_head[-1].bias)

        # EARLIER-opening gate: d10 0.05 -> 0.40 (was [.,.,0.05,0.5,...]).
        # Keep d0,d5 = 0 to preserve the early-peak win (identity there).
        self.register_buffer(
            "alpha_kf_residual",
            torch.tensor([0.0, 0.0, 0.40, 0.65, 0.90, 0.95, 0.97, 0.98, 0.99, 0.99]),
        )

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        out = super().forward(batch)
        h = out["h_final"]                                   # (Nr, d)
        u_kf = out["u_keyframes"]                            # (K, Nr)

        residual = self.late_residual_head(h).t()            # (K, Nr)
        alpha = self.alpha_kf_residual.to(u_kf.dtype).view(-1, 1)
        u_kf_corrected = (u_kf + alpha * residual).clamp(0.0, 1.0)

        out["u_keyframes"] = u_kf_corrected
        out["u_full"] = u_kf_corrected

        if self.decoder_mode in ("param", "param2", "param3"):
            shock = batch["shock_mask"].to(h.dtype)
            delta0 = batch["delta0"].to(h.dtype)
            peak_param = (1.0 - u_kf_corrected.min(dim=0).values).clamp(0.0, 1.0)
            peak = torch.maximum(peak_param, delta0 * shock).clamp(0.0, 1.0)
            out["peak"] = peak
            out["peak_pred"] = peak
            out["peak_phys"] = peak
        return out


# ===========================================================================
# Variant C — physics-prior decoder + free rollout, per-keyframe gated fusion
# ===========================================================================
class KSGATv3BlendC(KSGATv3):
    """MeCaSNet bimodal decoder fused per-keyframe with a free GRU rollout.

    Thin wrapper that forces `blend_rollout=True` so KSGATv3's existing
    param2_blend path runs the parametric branch AND the rollout branch in
    parallel and mixes them with the learnable per-keyframe gate `alpha_kf`.
    No new modules beyond what KSGATv3 already builds for blend_rollout.
    """

    def __init__(self, cfg, Fv: int, **kwargs):
        # ensure blend is on regardless of caller kwargs
        kwargs["blend_rollout"] = True
        super().__init__(cfg, Fv=Fv, **kwargs)


# ===========================================================================
# Variant D — symmetric (DirGNN-style) reverse aggregation in the backbone
# ===========================================================================
#
# Root-cause finding: DirGNN beats MeCaSNet at the transition frames (d10, d70)
# NOT because of its free per-keyframe head (PlainGAT/PlainGCN have the same head
# and do NOT win), but because it is the only baseline with a STRONG reverse
# (consumer->supplier) message pass: separate W_out, degree-normalised, edge-
# weight-aware. The transition-frame timing depends on the round-trip propagation
# of the demand-feedback wave, which needs both directions.
#
# MeCaSNet's DirectedGATBlock nominally has a reverse path, but it is crippled:
#   - reverse message ignores the edge weight a (no Leontief share on feedback)
#   - reverse aggregation is an UNNORMALISED sum (high-degree blow-up)
#   - reverse message is a thin 1-layer MLP seeing only [h_dst, u_dst]
#
# Variant D fixes ONLY the reverse path to mirror DirGNN (edge-weight aware,
# degree-normalised, 2-layer MLP). The forward attention path is byte-for-byte
# identical, so peak / early-frame behaviour is preserved. The bimodal closed-
# form physics decoder is UNTOUCHED — full physics interpretability retained;
# we only feed it better-informed latents. This is the most physics-preserving
# way to borrow DirGNN's transition-frame strength.

from .model import scatter_logsumexp
from .model_v3 import DirectedGATBlock
from .model_v3_residual import KSGATv3Residual


def _scatter_sum_1d(src, index, N):
    out = torch.zeros(N, device=src.device, dtype=src.dtype)
    return out.index_add_(0, index, src)


class SymmetricDirectedGATBlock(nn.Module):
    """DirectedGATBlock with a DirGNN-strength reverse path.

    Forward (supplier->consumer): IDENTICAL to DirectedGATBlock (attention).
    Reverse (consumer->supplier): edge-weight-aware, degree-normalised, 2-layer
    MLP — matches DirGNN's W_out branch. Returns (msg_in, msg_out), same I/O
    contract as DirectedGATBlock so it is a drop-in replacement.
    """

    def __init__(self, d: int):
        super().__init__()
        # --- forward path: identical structure to DirectedGATBlock ---
        self.fwd_msg = nn.Sequential(
            nn.Linear(2 * d + 2, d), nn.GELU(),
            nn.Linear(d, d),
        )
        self.fwd_score = nn.Linear(d, 1)
        # --- reverse path: strengthened (edge-weight + 2-layer MLP) ---
        # input per edge: [h_dst, u_dst, log a]  (d + 2)
        self.rev_msg = nn.Sequential(
            nn.Linear(d + 2, d), nn.GELU(),
            nn.Linear(d, d),
        )

    def forward(self, h, u, edge_src, edge_dst, edge_a, Nr, edge_mask=None):
        if edge_mask is not None:
            edge_a = edge_a * edge_mask
        log_a = edge_a.clamp_min(1e-4).log().unsqueeze(-1)         # (E,1)
        # ----- forward attention (identical to DirectedGATBlock) -----
        ef = torch.cat([h[edge_src], h[edge_dst],
                        u[edge_src].unsqueeze(-1), log_a], dim=-1)
        m = self.fwd_msg(ef)                                        # (E,d)
        s = self.fwd_score(m).squeeze(-1)                          # (E,)
        if edge_mask is not None:
            s = s + (edge_mask - 1.0) * 1e6
        log_norm = scatter_logsumexp(s, edge_dst, Nr)             # (Nr,)
        alpha = (s - log_norm[edge_dst]).exp().unsqueeze(-1)       # (E,1)
        msg_in = torch.zeros(Nr, m.shape[-1], device=h.device, dtype=h.dtype)
        if edge_mask is not None:
            msg_in.index_add_(0, edge_dst, alpha * m * edge_mask.unsqueeze(-1))
        else:
            msg_in.index_add_(0, edge_dst, alpha * m)
        # ----- reverse: edge-weighted, degree-normalised, 2-layer MLP -----
        rev_in = torch.cat([h[edge_dst], u[edge_dst].unsqueeze(-1), log_a], dim=-1)
        rm = self.rev_msg(rev_in)                                  # (E,d)
        if edge_mask is not None:
            rm = rm * edge_mask.unsqueeze(-1)
        a = edge_a.unsqueeze(-1)                                   # (E,1) already masked
        deg_out = _scatter_sum_1d(edge_a, edge_src, Nr).clamp_min(1e-6).unsqueeze(-1)
        msg_out = torch.zeros_like(h).index_add_(0, edge_src, a * rm) / deg_out
        return msg_in, msg_out


def _swap_directed_blocks(module: nn.Module, d: int) -> int:
    """Recursively replace every DirectedGATBlock child with a
    SymmetricDirectedGATBlock. Returns the number of blocks swapped."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, DirectedGATBlock) and not isinstance(child, SymmetricDirectedGATBlock):
            setattr(module, name, SymmetricDirectedGATBlock(d))
            n += 1
        else:
            n += _swap_directed_blocks(child, d)
    return n


class KSGATv3SymRev(KSGATv3Residual):
    """Flagship (V3DE_G3_RES, bimodal + residual head) with every backbone
    DirectedGATBlock swapped for SymmetricDirectedGATBlock.

    Identical to the flagship in every other respect (param2 g3_narrow decoder,
    3-layer prewarm, late-residual head + fixed alpha gate). Only the reverse
    message-passing capacity changes. Physics decoder untouched.
    """

    def __init__(self, cfg, Fv: int, **kwargs):
        super().__init__(cfg, Fv=Fv, **kwargs)
        d = int(cfg.d_hidden)
        n_swapped = _swap_directed_blocks(self, d)
        assert n_swapped > 0, "no DirectedGATBlock found to swap"
        self._n_symrev_blocks = n_swapped


# ===========================================================================
# Variant BD — D's symmetric-reverse backbone + B's trimodal decoder
# ===========================================================================
#
# Theory of synergy (NOT a simple sum):
#   D supplies the round-trip demand-feedback signal in the latent h
#     ("WHEN/WHERE a second shock hits") — the mechanism DirGNN wins on.
#   B supplies the closed-form 3rd Gaussian shape ("express that second
#     trough at d100+") that the bimodal decoder structurally cannot.
#   Single-handed, each is half-crippled: D knows there is a late trough but
#   the bimodal decoder has no shape for it; B has the shape but, with a
#   feedback-blind backbone, no signal to place it. Together they close the
#   loop — and P3 (3rd-Gaussian amplitude) grows ONLY when D's reverse signal
#   indicates a genuine late trough (Inoue), staying ~0 for monotone domains
#   (Henriet), so the late Gaussian is self-gating and domain-safe.
#
# Implementation: KSGATv3SymRev already forwards decoder_mode via **kwargs to
# KSGATv3, and KSGATv3Residual.forward recomputes peak for param3. So the
# fusion needs NO new wiring — just pass decoder_mode="param3". This subclass
# only fixes that default and gives the fusion an explicit name for the server.

class KSGATv3SymRevTri(KSGATv3SymRev):
    """B+D fusion: symmetric (DirGNN-style) reverse backbone + trimodal
    (param3) closed-form decoder + the flagship late-residual head.

    Fully physics-interpretable: the trajectory is still a closed-form
    min of three Gaussians; only the latent that drives their parameters is
    better informed by the strengthened reverse path.
    """

    def __init__(self, cfg, Fv: int, **kwargs):
        kwargs.setdefault("decoder_mode", "param3")
        super().__init__(cfg, Fv=Fv, **kwargs)


# ===========================================================================
# Variant E — edge-state-style edge-state backbone (KSGATv3EdgeState)
# ===========================================================================
#
# Diagnosis (2026-06-11): edge-state baseline reached val kf̄=0.75 / R²pk_csc=0.89
# at ep32 on Henriet, ~0.10 above our flagship+BD+KFW (0.64). Reading the
# edge-state code (model_baselines_strong.py:209-271) vs ours shows the gap is
# NOT in the decoder but in the BACKBONE:
#
#   our DirectedGATBlock        : edge has NO hidden state. Each layer recomputes
#                                  m_e = MLP([h_src, h_dst, u_src, log a]) from
#                                  current h. Edge "memory" of the cascade is
#                                  only encoded implicitly in node h.
#   edge-state processor               : edge has d-dim hidden state e_ij that
#                                  PERSISTS across layers, updated as
#                                    e <- LN(e + MLP([e, h_src, h_dst]))
#                                  and the SAME e drives both forward (in)
#                                  and reverse (out) aggregation. After 6
#                                  layers e accumulates 12-hop bidirectional
#                                  message history.
#   our reverse                 : 1-layer MLP on (h_dst, u_dst), index_add sum
#   edge-state reverse                 : same e as forward, sum aggregation - fully
#                                  symmetric strength
#
# This block (`EdgeStateGATBlock`) brings edge-state's edge state into our prewarm
# WITHOUT touching the physics decoder, residual head, MoE router, KSGATStep
# inv-state oscillator, or the u₀ = 1 - δ₀·shock physics hard-code. We KEEP
# our forward attention (Leontief essential-supplier weighting — physics) as
# an extra weighting on top of edge messages, because in supply chains the
# consumer should attend more to large-share suppliers (a_ij > τ).
#
# Implementation trick: edge state e must persist ACROSS the prewarm layers.
# We avoid re-overriding the parent forward (250 lines) by giving each block
# a shared mutable `EdgeStateCache` that is reset at the start of every model
# forward(). The block list signature stays (h, u, src, dst, edge_a, Nr) →
# (msg_in, msg_out) so the parent's prewarm loop is untouched.
#
# Param budget (d=64, n_proc=4): + ~100K vs flagship 280K BD = 380K total.
#   per block: edge_update (3d→d→d) ~25K + node_update (3d→d→d) ~25K +
#   LayerNorms ~256 = ~50K. n_proc=4 → +200K-ish minus the 30K saved by
#   replacing the 3 DirectedGATBlocks → net ~+170K.

class EdgeStateCache:
    """Mutable cache shared across all EdgeStateGATBlocks in a single forward.

    Holds the per-edge hidden state e (E, d). Reset to None at the start of
    every KSGATv3EdgeState.forward() so the chain re-initialises e from h
    on the first block, then persists e across the remaining n_proc-1 layers.
    """

    def __init__(self):
        self.e = None


class EdgeStateGATBlock(nn.Module):
    """edge-state-style spatial block with persistent edge state.

    Drops into KSGATv3.prewarm_gat (signature compatible with DirectedGATBlock).

    Single MLP design (no dead weights):
      - edge state e ∈ R^d persists across blocks via shared EdgeStateCache
      - input to edge_update: [e_prev, h_src, h_dst, log a]   (3d+1 → d)
      - first block in chain (cache.e is None): we feed e_prev = 0 so the
        SAME MLP both initialises e_0 and updates it on subsequent calls.
        Residual connection skipped on first call (since e_0 = 0).
      - LayerNorm applied uniformly.
      - log a injected EVERY layer so edge weight (Leontief share) keeps
        influencing message passing throughout the chain (not only at init).

    Forward attention preserved as Leontief weighting on top of the edge-state
    bidirectional aggregation.
    """

    def __init__(self, d: int):
        super().__init__()
        # Unified edge update: [e_prev (d), h_src (d), h_dst (d), log a (1)] -> Δe (d)
        self.edge_update = nn.Sequential(
            nn.Linear(3 * d + 1, d), nn.GELU(),
            nn.Linear(d, d),
        )
        self.edge_norm = nn.LayerNorm(d)
        # Forward attention score (Leontief essential-supplier weighting)
        self.fwd_score = nn.Linear(d, 1)
        # Cache reference; injected by parent KSGATv3EdgeState before forward
        self._cache: "EdgeStateCache | None" = None

    def set_cache(self, cache) -> None:
        """Inject (or clear) the shared EdgeStateCache. Pass None to clear."""
        self._cache = cache

    def forward(self, h, u, edge_src, edge_dst, edge_a, Nr, edge_mask=None):
        cache = self._cache
        assert cache is not None, \
            "EdgeStateGATBlock used without EdgeStateCache injection"
        E = edge_src.shape[0]
        # Stage-2 repair masks out invalid edges; align with DirectedGATBlock.
        if edge_mask is not None:
            edge_a = edge_a * edge_mask
        log_a = edge_a.clamp_min(1e-4).log().unsqueeze(-1)             # (E,1)
        # ----- (re-)compute edge state e via the SAME MLP -----
        if cache.e is None:
            # First block in chain: feed e_prev = 0 so MLP synthesises e_0.
            # No residual since e_0 = 0 anyway.
            e_prev = torch.zeros(E, h.shape[-1], device=h.device, dtype=h.dtype)
            e_in = torch.cat([e_prev, h[edge_src], h[edge_dst], log_a], dim=-1)
            e = self.edge_norm(self.edge_update(e_in))
        else:
            e_in = torch.cat([cache.e, h[edge_src], h[edge_dst], log_a], dim=-1)
            e = self.edge_norm(cache.e + self.edge_update(e_in))
        cache.e = e                                                    # persist

        # ----- forward aggregation: attention-weighted sum of e into dst -----
        s = self.fwd_score(e).squeeze(-1)                              # (E,)
        if edge_mask is not None:
            # push masked edges to -inf in softmax (matches DirectedGATBlock)
            s = s + (edge_mask - 1.0) * 1e6
        log_norm = scatter_logsumexp(s, edge_dst, Nr)                  # (Nr,)
        alpha = (s - log_norm[edge_dst]).exp().unsqueeze(-1)           # (E,1)
        msg_in = torch.zeros(Nr, e.shape[-1], device=h.device, dtype=h.dtype)
        if edge_mask is not None:
            msg_in.index_add_(0, edge_dst, alpha * e * edge_mask.unsqueeze(-1))
        else:
            msg_in.index_add_(0, edge_dst, alpha * e)

        # ----- reverse aggregation: SAME e, plain sum into src -----
        if edge_mask is not None:
            msg_out = torch.zeros_like(h).index_add_(0, edge_src, e * edge_mask.unsqueeze(-1))
        else:
            msg_out = torch.zeros_like(h).index_add_(0, edge_src, e)
        return msg_in, msg_out


class KSGATv3EdgeState(KSGATv3SymRevTri):
    """B+D+edge-state fusion variant.

    Stacks the edge-state-style edge-state backbone on top of the BD recipe:
      backbone : 4-layer EdgeStateGATBlock chain with persistent edge state
                  (replaces the 3-layer DirectedGATBlock prewarm)
      KSGATStep: keeps SymmetricDirectedGATBlock (D's strong reverse) for
                  intra-rollout spatial passes and the inv-state oscillator
      decoder  : trimodal param3 (B's late Gaussian) — UNTOUCHED
      head     : late-residual head + α gate — UNTOUCHED
      u₀       : 1 - δ₀·shock physics hard-code — UNTOUCHED

    The edge-state backbone provides the latent richness edge-state uses to learn
    non-typical trajectories (mid-cascade rebounds, late demand spikes), but
    the closed-form trimodal decoder still consumes that latent and emits
    physics-readable (μ, σ, A) triples per node — interpretability preserved.

    Param budget vs flagship 280K BD: ~+170K (4 prewarm blocks @ ~50K each
    minus 3 saved DirectedGATBlock @ ~10K each); total ≈ 420K. Cut
    `n_edge_blocks` to 3 if a tighter budget is needed.
    """

    def __init__(self, cfg, Fv: int, n_edge_blocks: int = 4, **kwargs):
        super().__init__(cfg, Fv=Fv, **kwargs)
        d = int(cfg.d_hidden)
        # Replace the prewarm chain with EdgeStateGATBlocks. We KEEP the same
        # ModuleList attribute name (prewarm_gat) so the parent forward loop
        # picks them up unchanged.
        self.prewarm_gat = nn.ModuleList(
            [EdgeStateGATBlock(d) for _ in range(n_edge_blocks)]
        )
        self.prewarm_norms = nn.ModuleList(
            [nn.LayerNorm(d) for _ in range(n_edge_blocks)]
        )
        self.prewarm_layers = n_edge_blocks
        self._n_edge_blocks = n_edge_blocks

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        # Fresh cache per forward call: edge state initialises in block 0,
        # persists across blocks 1..n-1, discarded at end of forward.
        cache = EdgeStateCache()
        for block in self.prewarm_gat:
            if isinstance(block, EdgeStateGATBlock):
                block.set_cache(cache)
        try:
            return super().forward(batch)
        finally:
            # Defensive: clear the cache so refs don't leak across batches
            for block in self.prewarm_gat:
                if isinstance(block, EdgeStateGATBlock):
                    block.set_cache(None)
