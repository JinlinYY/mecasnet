"""Named, auditable model profiles and construction helpers."""

from __future__ import annotations

from .config import Config


PAPER_PROFILE = "paper"
COMPAT_PROFILE = "compat"
PROFILES = (PAPER_PROFILE, COMPAT_PROFILE)


def apply_profile(cfg: Config, profile: str = PAPER_PROFILE) -> Config:
    """Apply a complete model/input profile to ``cfg`` in place.

    The paper profile follows the manuscript and SI: Day-0-only event input,
    capacity-based Day-0 boundary, a fixed ``c=6`` decline front, learned
    ``q_v`` in ``[0.5, 2.0]``, and the three prediction streams. The ``compat``
    profile provides the fixed ``c=2`` architecture required by the supplied
    post-training analysis utilities.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose one of {PROFILES}")

    cfg.profile = profile
    cfg.event_scalars_mode = "minimal"
    cfg.day0_demand_pullback = False
    cfg.recovery_p = 1.0
    cfg.recovery_gate = False
    cfg.physrecover = False
    cfg.ablate_uncond = False
    cfg.ablate_forward_only = False
    cfg.residual_wide = False
    cfg.seam_tau_pernode = False
    cfg.seam_tau_dual = False
    cfg.p3_g2_repos = False
    cfg.resid_tail_mask = False

    if profile == PAPER_PROFILE:
        cfg.decline_p = 6.0
        cfg.recovery_q = True
        cfg.recovery_q_min = 0.5
        cfg.recovery_q_max = 2.0
    else:
        cfg.decline_p = 2.0
        cfg.recovery_q = False
    return cfg


def build_mecasnet(
    cfg: Config,
    feature_count: int,
    *,
    profile: str = PAPER_PROFILE,
    propagation_steps: int = 4,
):
    """Build the manuscript architecture using an explicit named profile."""
    from .model_v3_decoder_ac import KSGATv3EdgeState

    apply_profile(cfg, profile)
    return KSGATv3EdgeState(
        cfg,
        Fv=feature_count,
        peak_mode="traj",
        prewarm_layers=3,
        n_edge_blocks=propagation_steps,
        bimodal_tau=0.02,
        learnable_bimodal_tau=False,
        blend_rollout=False,
        triple_blend=True,
        blend_init=(2.0, 0.0, 0.0),
        free_residual=False,
    )
