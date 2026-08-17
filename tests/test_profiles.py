from mecasnet import Config, apply_profile
from mecasnet.factory import LEGACY_Y8_PROFILE, PAPER_PROFILE


def test_paper_profile_matches_manuscript_information_boundary() -> None:
    cfg = apply_profile(Config(), PAPER_PROFILE)

    assert cfg.event_scalars_mode == "minimal"
    assert cfg.day0_demand_pullback is False
    assert cfg.decline_p == 6.0
    assert cfg.recovery_q is True
    assert (cfg.recovery_q_min, cfg.recovery_q_max) == (0.5, 2.0)
    assert cfg.seam_tau_pernode is False
    assert cfg.seam_tau_dual is False
    assert cfg.p3_g2_repos is False


def test_legacy_profile_is_explicitly_separate() -> None:
    cfg = apply_profile(Config(), LEGACY_Y8_PROFILE)

    assert cfg.profile == LEGACY_Y8_PROFILE
    assert cfg.decline_p == 2.0
    assert cfg.recovery_q is False
