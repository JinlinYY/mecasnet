"""Alternative MeCaSNet decoders and directed graph backbones.

The module provides residual, rollout-fusion, symmetric-reverse, trimodal, and
persistent edge-state variants. All variants preserve the ``KSGATv3`` forward
contract and can be selected through the training interface.
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
    """Residual variant with an earlier gate and a wider ``d -> 2d -> K`` head."""

    def __init__(self, cfg, Fv: int, **kwargs):
        super().__init__(cfg, Fv=Fv, **kwargs)
        d = int(cfg.d_hidden)
        K = len(cfg.key_days)
        assert K == 10, f"alpha_kf_residual hard-coded for K=10, got {K}"

        # Wider residual head: d -> 2d -> K.
        self.late_residual_head = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(),
            nn.Linear(2 * d, K),
        )
        nn.init.zeros_(self.late_residual_head[-1].weight)
        nn.init.zeros_(self.late_residual_head[-1].bias)

        # Keep d0 and d5 at identity; open the residual path from d10 onward.
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
# The symmetric block keeps supplier-to-consumer attention and adds an
# edge-weight-aware, degree-normalized consumer-to-supplier message path.

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
    """Residual decoder with symmetric directed message-passing blocks.

    The param2 decoder, three-layer prewarm, late-residual head, and fixed alpha
    gate are retained; only reverse message-passing capacity changes.
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
# This variant combines the symmetric reverse path with the trimodal analytical
# decoder. The third component represents late troughs while the reverse path
# supplies demand-feedback information to the parameter-generating latent.

class KSGATv3SymRevTri(KSGATv3SymRev):
    """Symmetric-reverse backbone with a trimodal closed-form decoder.

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
# Persistent edge-state backbone
# Persistent edge states provide explicit relation memory across layers:
#
#   e <- LN(e + MLP([e, h_src, h_dst, log_a]))
# The same edge representation drives forward attention and reverse aggregation.
#
# EdgeStateGATBlock adds the persistent state to the prewarm chain without
# changing the physics decoder, residual head, router, recurrent state, or
# deterministic Day-0 boundary. It retains
# our forward attention (Leontief essential-supplier weighting — physics) as
# an extra weighting on top of edge messages, because in supply chains the
# consumer should attend more to large-share suppliers (a_ij > τ).
#
# A shared EdgeStateCache persists e across the prewarm layers and is reset at
# the start of each forward pass. The block I/O contract is unchanged.

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
    """Trimodal decoder with a persistent edge-state prewarm backbone.

    Stacks the edge-state-style edge-state backbone on top of the BD recipe:
      backbone : EdgeStateGATBlock chain with persistent edge state
                  (replaces the 3-layer DirectedGATBlock prewarm)
      KSGATStep: keeps SymmetricDirectedGATBlock (D's strong reverse) for
                  intra-rollout spatial passes and the inv-state oscillator
      decoder  : trimodal param3 analytical trajectory
      head     : late-residual head with fixed alpha gate
      u₀       : deterministic 1 - delta0 * shock boundary

    The edge-state backbone provides the latent richness edge-state uses to learn
    non-typical trajectories (mid-cascade rebounds, late demand spikes), but
    the closed-form trimodal decoder still consumes that latent and emits
    physics-readable (μ, σ, A) triples per node — interpretability preserved.

    ``n_edge_blocks`` controls the depth and parameter budget of the prewarm
    chain.
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
