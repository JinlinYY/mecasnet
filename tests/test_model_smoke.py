import torch

from mecasnet import Config, build_mecasnet


def test_paper_model_forward_preserves_day0_boundary() -> None:
    cfg = Config(d_hidden=16)
    model = build_mecasnet(cfg, feature_count=6, profile="paper", propagation_steps=2)
    model.eval()

    batch = {
        "x_v": torch.randn(4, 6),
        "shock_mask": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "delta0": torch.tensor([0.4, 0.0, 0.0, 0.0]),
        "event_scalars": torch.tensor([0.0, 0.0, 0.4] + [0.0] * 9),
        "edge_src": torch.tensor([0, 1, 2]),
        "edge_dst": torch.tensor([1, 2, 3]),
        "edge_a": torch.tensor([0.5, 0.4, 0.3]),
        "edge_outshare": torch.tensor([0.5, 0.4, 0.3]),
        "key_days": torch.tensor(cfg.key_days),
        "shock_hop_oh": torch.eye(5)[:4],
    }

    with torch.no_grad():
        output = model(batch)

    assert output["u_keyframes"].shape == (len(cfg.key_days), 4)
    expected_day0 = 1.0 - batch["shock_mask"] * batch["delta0"]
    torch.testing.assert_close(output["u_keyframes"][0], expected_day0)
    torch.testing.assert_close(output["peak"], 1.0 - output["u_keyframes"].min(dim=0).values)

