import pickle

import numpy as np

from mecasnet import Config
from mecasnet.data import CascadeEventDataset, StaticNetwork


def _write_static(root) -> None:
    payload = {
        "V": 3,
        "E": 2,
        "n_sec": 1,
        "sectors": np.array([0, 0, 0]),
        "P_ini": np.array([10.0, 10.0, 10.0]),
        "edge_src": np.array([0, 1]),
        "edge_dst": np.array([1, 2]),
        "A": np.array([2.0, 3.0]),
    }
    with (root / "static_meta.pkl").open("wb") as stream:
        pickle.dump(payload, stream)


def test_minimal_profile_does_not_read_simulator_only_metadata(tmp_path) -> None:
    _write_static(tmp_path)
    events = tmp_path / "events"
    events.mkdir()
    key_days = np.array([0, 5, 10, 20, 30, 50, 70, 100, 150, 199])
    u = np.ones((len(key_days), 3), dtype=np.float32)
    np.savez(
        events / "event_000000.npz",
        shock_mask=np.array([1, 0, 0], dtype=np.uint8),
        delta0=np.array([0.4, 0.0, 0.0], dtype=np.float32),
        u_keyframes=u,
        key_days=key_days,
        peak_loss_node=np.zeros(3, dtype=np.float32),
        cum_loss_node=np.zeros(3, dtype=np.float32),
        shock_meta=np.array({"recovery_days": "not-observable"}, dtype=object),
    )

    cfg = Config(data_root=str(tmp_path), event_scalars_mode="minimal")
    net = StaticNetwork(cfg)
    item = CascadeEventDataset(cfg, net, [0])[0]

    expected = np.zeros(12, dtype=np.float32)
    expected[2] = 0.4
    np.testing.assert_allclose(item["event_scalars"].numpy(), expected)


def test_static_network_rejects_nonpositive_production(tmp_path) -> None:
    _write_static(tmp_path)
    static_path = tmp_path / "static_meta.pkl"
    with static_path.open("rb") as stream:
        payload = pickle.load(stream)
    payload["P_ini"][1] = 0.0
    with static_path.open("wb") as stream:
        pickle.dump(payload, stream)

    cfg = Config(data_root=str(tmp_path))
    try:
        StaticNetwork(cfg)
    except ValueError as exc:
        assert "P_ini" in str(exc)
    else:
        raise AssertionError("StaticNetwork accepted nonpositive P_ini")
