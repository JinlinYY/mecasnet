"""Static-network preprocessing and cascade-event dataset.

Each event is evaluated on the directed k-hop neighborhood reachable from the
directly shocked firms. Nodes outside that event subgraph are omitted from the
model batch and from the metrics implemented in ``mecasnet.runner``.
"""
from __future__ import annotations
import pickle
import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List

from .config import Config, get_paths


# ---------------------------------------------------------------------------
# Static network (loaded once, shared)
# ---------------------------------------------------------------------------
class StaticNetwork:
    """Holds A (sparse, edge-indexed), P_ini, sectors. ~few MB.

    Real data format from generate_training_data.py / simulate_inoue_on_real_network.py:
      - `A` in static_meta is an (E,) array of absolute steady-state flows (currency).
        For edge e: src[e] supplies dst[e] with flow A[e].
        Column-normalised so   ∑_{e:dst=v} A[e] = A_COL_NORM · P_ini[v]   (=0.6·P_ini)
      - We need the input share  a_{rv} = A[e] / P_ini[dst[e]]
        which feeds the Leontief min as edge importance weight.
    """

    def __init__(self, cfg: Config):
        paths = get_paths(cfg)
        with open(paths["static"], "rb") as f:
            sm = pickle.load(f)
        self.V: int = int(sm["V"])
        self.E: int = int(sm["E"])
        self.n_sec: int = int(sm["n_sec"])
        self.sectors = np.asarray(sm["sectors"], dtype=np.int64)
        self.P_ini = np.asarray(sm["P_ini"], dtype=np.float32)
        self.edge_src = np.asarray(sm["edge_src"], dtype=np.int64)   # supplier
        self.edge_dst = np.asarray(sm["edge_dst"], dtype=np.int64)   # consumer
        A_flow = np.asarray(sm["A"], dtype=np.float32)

        expected_shapes = {
            "sectors": (self.V,),
            "P_ini": (self.V,),
            "edge_src": (self.E,),
            "edge_dst": (self.E,),
            "A": (self.E,),
        }
        actual_arrays = {
            "sectors": self.sectors,
            "P_ini": self.P_ini,
            "edge_src": self.edge_src,
            "edge_dst": self.edge_dst,
            "A": A_flow,
        }
        bad_shapes = {
            name: (array.shape, expected_shapes[name])
            for name, array in actual_arrays.items()
            if array.shape != expected_shapes[name]
        }
        if self.V <= 0 or self.E < 0 or self.n_sec <= 0:
            raise ValueError("V and n_sec must be positive; E must be non-negative")
        if bad_shapes:
            raise ValueError(f"Invalid static-network array shapes: {bad_shapes}")
        if np.any(self.sectors < 0) or np.any(self.sectors >= self.n_sec):
            raise ValueError("sectors contains an index outside [0, n_sec)")
        if not np.all(np.isfinite(self.P_ini)) or np.any(self.P_ini <= 0):
            raise ValueError("P_ini must contain finite positive values")
        if (np.any(self.edge_src < 0) or np.any(self.edge_src >= self.V)
                or np.any(self.edge_dst < 0) or np.any(self.edge_dst >= self.V)):
            raise ValueError("edge_src/edge_dst contains an index outside [0, V)")
        if not np.all(np.isfinite(A_flow)) or np.any(A_flow <= 0):
            raise ValueError("A must contain finite positive flows")

        # ---- edge share a_{rv} = A[e] / P_ini[dst[e]]  ∈ (0, A_COL_NORM] ----
        pini_dst = np.maximum(self.P_ini[self.edge_dst], 1e-6)
        edge_share = (A_flow / pini_dst).astype(np.float32)         # (E,)
        self.edge_share = edge_share                                # used per-event

        # ---- A as sparse (V,V): rows = consumer, cols = supplier ----
        # entry A_csr[v, r] = a_{rv}  (input share of supplier r in consumer v)
        self.A_csr = sp.csr_matrix(
            (edge_share, (self.edge_dst, self.edge_src)), shape=(self.V, self.V)
        )
        # forward propagation (supplier r → customers v): rows=supplier
        # used for shock-reach BFS
        self.fwd_csr = sp.csr_matrix(
            (edge_share, (self.edge_src, self.edge_dst)), shape=(self.V, self.V)
        )

        # ---- degrees (in/out) ----
        in_deg = np.zeros(self.V, dtype=np.float32)
        out_deg = np.zeros(self.V, dtype=np.float32)
        np.add.at(in_deg, self.edge_dst, 1.0)
        np.add.at(out_deg, self.edge_src, 1.0)
        self.in_deg = in_deg
        self.out_deg = out_deg

        # ---- GSCC membership (per DATA_CHARACTERISTICS §1: D subnet GSCC=30%) ----
        # connected_components on the directed graph
        from scipy.sparse.csgraph import connected_components
        n_comp, comp_lbl = connected_components(self.fwd_csr, directed=True,
                                                connection="strong")
        # largest SCC
        sizes = np.bincount(comp_lbl)
        gscc_lbl = int(sizes.argmax())
        in_gscc = (comp_lbl == gscc_lbl).astype(np.float32)   # (V,)
        self.in_gscc = in_gscc

        # ---- reverse-pagerank (exposure to upstream shocks) ----
        # Use power iteration on row-normalised reverse graph.
        # High score = node is downstream of many big suppliers ⇒ susceptible.
        bwd_row_norm = sp.csr_matrix(
            (np.ones_like(self.edge_src, dtype=np.float32),
             (self.edge_dst, self.edge_src)),
            shape=(self.V, self.V),
        )
        row_sum = np.asarray(bwd_row_norm.sum(axis=1)).ravel() + 1e-9
        bwd_row_norm = sp.diags(1.0 / row_sum) @ bwd_row_norm
        pr = np.full(self.V, 1.0 / self.V, dtype=np.float32)
        damp = 0.85
        for _ in range(40):
            pr_new = damp * (bwd_row_norm.T @ pr) + (1 - damp) / self.V
            if np.abs(pr_new - pr).sum() < 1e-7:
                pr = pr_new
                break
            pr = pr_new
        self.pagerank_rev = pr.astype(np.float32)

        # ---- node static features ----
        log_indeg = np.log1p(in_deg)[:, None]
        log_outdeg = np.log1p(out_deg)[:, None]
        log_Pini = np.log1p(self.P_ini)[:, None]
        log_pr = np.log(self.pagerank_rev + 1e-10)[:, None]
        for col in (log_indeg, log_outdeg, log_Pini, log_pr):
            col -= col.mean()
            col /= (col.std() + 1e-6)
        feats = [log_indeg, log_outdeg, log_Pini, log_pr, in_gscc[:, None]]
        if getattr(cfg, "use_sector_oh", True):
            sec_oh = np.eye(self.n_sec, dtype=np.float32)[self.sectors]
            feats = [sec_oh] + feats
        self.x_v_static = np.concatenate(feats, axis=1).astype(np.float32)
        self.Fv = self.x_v_static.shape[1]

    # ----- ShockReach BFS -----
    def shock_reach(self, shock_mask: np.ndarray, k_hops: int) -> np.ndarray:
        """Return node indices reachable from shocked nodes within k_hops (along
        supplier→customer). Includes the shocked nodes themselves."""
        frontier = np.where(shock_mask > 0)[0]
        seen = np.zeros(self.V, dtype=bool)
        seen[frontier] = True
        for _ in range(k_hops):
            if len(frontier) == 0:
                break
            rows = self.fwd_csr[frontier]
            nxt = np.unique(rows.indices)
            new = nxt[~seen[nxt]]
            seen[new] = True
            frontier = new
        return np.where(seen)[0].astype(np.int64)


# ---------------------------------------------------------------------------
# Per-event dataset
# ---------------------------------------------------------------------------
class CascadeEventDataset(Dataset):
    """Each item = one cascade event (.npz). Static net is shared via reference."""

    def __init__(self, cfg: Config, net: StaticNetwork, event_ids: List[int],
                 train_mode: bool = False):
        self.cfg = cfg
        self.net = net
        self.events_dir = get_paths(cfg)["events"]
        self.event_ids = event_ids
        self.train_mode = train_mode

    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, idx: int) -> Dict:
        eid = self.event_ids[idx]
        path = self.events_dir / f"event_{eid:06d}.npz"
        with np.load(path, allow_pickle=True) as z:
            shock_mask = z["shock_mask"].astype(np.uint8)             # (V,)
            delta0 = z["delta0"].astype(np.float32)                   # (V,)
            u_kf = z["u_keyframes"].astype(np.float32)                # (K, V)
            key_days = z["key_days"].astype(np.int32)                 # (K,)
            peak = z["peak_loss_node"].astype(np.float32)             # (V,)
            cum = z["cum_loss_node"].astype(np.float32)               # (V,)
            meta = z["shock_meta"].item()                             # dict
            include_audit_metadata = bool(getattr(
                self.cfg, "include_audit_metadata", False,
            ))
            aggregate_bau_peak = (
                float(np.asarray(z["aggregate_bau_loss_curve"], dtype=np.float32).max())
                if include_audit_metadata and "aggregate_bau_loss_curve" in z.files
                else float("nan")
            )

        expected_days = np.asarray(self.cfg.key_days, dtype=np.int32)
        event_shapes = {
            "shock_mask": shock_mask.shape,
            "delta0": delta0.shape,
            "u_keyframes": u_kf.shape,
            "key_days": key_days.shape,
            "peak_loss_node": peak.shape,
            "cum_loss_node": cum.shape,
        }
        expected_event_shapes = {
            "shock_mask": (self.net.V,),
            "delta0": (self.net.V,),
            "u_keyframes": (len(expected_days), self.net.V),
            "key_days": (len(expected_days),),
            "peak_loss_node": (self.net.V,),
            "cum_loss_node": (self.net.V,),
        }
        bad_event_shapes = {
            name: (shape, expected_event_shapes[name])
            for name, shape in event_shapes.items()
            if shape != expected_event_shapes[name]
        }
        if bad_event_shapes:
            raise ValueError(f"Invalid shapes in {path.name}: {bad_event_shapes}")
        if not np.array_equal(key_days, expected_days):
            raise ValueError(
                f"{path.name} key_days={key_days.tolist()} does not match "
                f"Config.key_days={expected_days.tolist()}"
            )
        if not np.all((shock_mask == 0) | (shock_mask == 1)):
            raise ValueError(f"{path.name} shock_mask must be binary")
        if (not np.all(np.isfinite(delta0)) or np.any(delta0 < 0)
                or np.any(delta0 > 1)):
            raise ValueError(f"{path.name} delta0 must be finite and in [0, 1]")
        for name, array in (("u_keyframes", u_kf), ("peak_loss_node", peak),
                            ("cum_loss_node", cum)):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{path.name} {name} contains non-finite values")
        if not hasattr(meta, "get"):
            raise ValueError(f"{path.name} shock_meta must be a mapping")

        # ----- event-level scalars (broadcast to every reach node) -----
        # Keep a fixed width for checkpoint compatibility, but do not even read
        # simulator-only recovery/mode/tier metadata in the manuscript profile.
        scalar_mode = getattr(self.cfg, "event_scalars_mode", "minimal")
        event_scalars = np.zeros(12, dtype=np.float32)
        if scalar_mode == "minimal":
            directly_shocked = shock_mask > 0
            event_scalars[2] = (
                float(delta0[directly_shocked].mean())
                if directly_shocked.any()
                else 0.0
            )
        elif scalar_mode in ("clean", "full"):
            rec_arr = np.asarray(
                meta.get("recovery_per_target", [150.0]), dtype=np.float32
            )
            if rec_arr.size == 0:
                rec_arr = np.array([150.0], dtype=np.float32)
            log_rec = np.log(np.clip(rec_arr, 1.0, None))
            delta_arr = np.asarray(
                meta.get("delta_per_target", [0.5]), dtype=np.float32
            )
            if delta_arr.size == 0:
                delta_arr = np.array([0.5], dtype=np.float32)
            modes = ("sector", "hub", "random")
            tiers = ("light", "medium", "heavy", "catastrophic")
            event_mode = str(meta.get("mode", "random"))
            tier = str(meta.get("tier", "medium"))
            event_scalars[:] = np.concatenate([
                np.array([
                    float(log_rec.mean()),
                    float(log_rec.std() + 1e-6),
                    float(delta_arr.mean()),
                    float(meta.get("delta", 0.5)),
                    float(np.log(meta.get("recovery_days", 150.0) + 1e-6)),
                ], dtype=np.float32),
                np.array([event_mode == item for item in modes], dtype=np.float32),
                np.array([tier == item for item in tiers], dtype=np.float32),
            ])
            if scalar_mode == "clean":
                # Remove per-firm recovery and simulator shock-mode fields.
                event_scalars[[0, 1, 5, 6, 7]] = 0.0
        else:
            raise ValueError(
                f"Unknown event_scalars_mode={scalar_mode!r}; "
                "choose 'minimal', 'clean', or 'full'"
            )

        # ----- shock-reach mask -----
        reach_idx = self.net.shock_reach(shock_mask, self.cfg.reach_hops)

        # ----- ① train-time subgraph subsample augmentation -----
        # With prob p, keep all shocked nodes + random sample of non-shock
        # reach nodes; resulting size ∈ [subgraph_aug_min, |reach|]. Exposes
        # the model to a continuous size spectrum so the absolute scale of
        # the network is not an identifying signal.
        sub_aug_p = float(getattr(self.cfg, "subgraph_aug_p", 0.0))
        if self.train_mode and sub_aug_p > 0.0 and np.random.rand() < sub_aug_p:
            N_min = int(getattr(self.cfg, "subgraph_aug_min", 400))
            if len(reach_idx) > N_min:
                shock_local = np.where(shock_mask[reach_idx] > 0)[0]
                non_shock_local = np.where(shock_mask[reach_idx] == 0)[0]
                lo = max(N_min, shock_local.size + 1)
                hi = len(reach_idx)
                if hi > lo:
                    target = np.random.randint(lo, hi)
                    n_keep = target - shock_local.size
                    if 0 < n_keep < non_shock_local.size:
                        pick = np.random.choice(non_shock_local, n_keep,
                                                replace=False)
                        keep_local = np.concatenate([shock_local, pick])
                        keep_local.sort()
                        reach_idx = reach_idx[keep_local]

        # restrict everything to reach subgraph
        sub_x_v = self.net.x_v_static[reach_idx]                      # (Nr, Fv)

        # ----- ② per-event re-standardization of x_v (dimensionless) -----
        if getattr(self.cfg, "dimensionless_inputs", False) and sub_x_v.shape[0] > 1:
            mu = sub_x_v.mean(axis=0, keepdims=True)
            sd = sub_x_v.std(axis=0, keepdims=True) + 1e-6
            sub_x_v = ((sub_x_v - mu) / sd).astype(np.float32)
        sub_shock = shock_mask[reach_idx].astype(np.float32)          # (Nr,)
        sub_delta0 = delta0[reach_idx]                                # (Nr,)
        sub_u_kf = u_kf[:, reach_idx]                                 # (K, Nr)
        sub_peak = peak[reach_idx]                                    # (Nr,)
        sub_P_ini = self.net.P_ini[reach_idx]                         # (Nr,)
        sub_sec = self.net.sectors[reach_idx]                         # (Nr,)
        # is_reach label for BCE aux loss (DATA_CHARACTERISTICS §5.3:
        #   49% of nodes have strictly zero peak_loss; expose this as
        #   classification subtask so the head can learn the 0/non-0 partition)
        sub_is_reach = (sub_peak > self.cfg.reach_thresh).astype(np.float32)

        # Sub-A: rows=consumer in reach, cols=supplier in reach
        # supplier of a reach node may be outside reach (then capacity stays 1 at
        # supplier → no effect on consumer). We include only intra-reach edges.
        sub_A = self.net.A_csr[reach_idx][:, reach_idx]               # csr (Nr,Nr)

        # COO edges (consumer ← supplier) for message passing
        coo = sub_A.tocoo()
        edge_dst = torch.as_tensor(coo.row, dtype=torch.long)         # consumer
        edge_src = torch.as_tensor(coo.col, dtype=torch.long)         # supplier
        edge_a = torch.as_tensor(coo.data, dtype=torch.float32)       # a_{v,r}

        # ----- train-time edge dropout for cross-domain augmentation -----
        # Random drop a fraction p of edges; preserves shocked-node connectivity
        # by exempting edges whose endpoints are both shocked.
        edge_drop_p = float(getattr(self.cfg, "edge_dropout_p", 0.0))
        if self.train_mode and edge_drop_p > 0.0 and edge_src.numel() > 0:
            keep_prob = 1.0 - edge_drop_p
            mask = (torch.rand(edge_src.shape[0]) < keep_prob)
            edge_src = edge_src[mask]
            edge_dst = edge_dst[mask]
            edge_a = edge_a[mask]

        # ----- edge OUTSHARE (Henriet eq-4 / eq-9 physics) -----
        # `edge_a` above is INSHARE  a_{v,r} = A_flow[e] / P_ini[dst]   (Leontief input share for the consumer).
        # `edge_outshare` is the COMPLEMENT view  o_{r,v} = A_flow[e] / P_ini[src]
        # = the fraction of supplier r's pre-event output that goes to consumer v.
        # Identity:  A_flow[e] = edge_a[e] * P_ini[dst[e]]  ⇒  outshare = edge_a · P_ini[dst] / P_ini[src].
        # This is the exact weight that appears in the demand-pullback term of
        # Henriet 2012 eq 4 (orders rescaled by buyer P_cap fraction) and is
        # consumed by KSGATv3._predict_u0 to evaluate the eq-9 day-0 initial
        # condition  u₀(i) = min(1 − δ_self_i,  1 − Σ_j o_{i→j} · δ_j · shock_j).
        sub_P_ini_t = torch.as_tensor(sub_P_ini, dtype=torch.float32)
        edge_outshare = (edge_a * sub_P_ini_t[edge_dst]
                         / sub_P_ini_t[edge_src].clamp_min(1e-6))

        # ----- train-time node-feature dropout -----
        feat_drop_p = float(getattr(self.cfg, "node_feat_dropout_p", 0.0))
        if self.train_mode and feat_drop_p > 0.0:
            non_shock = (sub_shock < 0.5)
            drop = (np.random.rand(len(reach_idx)) < feat_drop_p) & non_shock
            if drop.any():
                sub_x_v = sub_x_v.copy()
                sub_x_v[drop] = 0.0

        # ----- shock hop distance (undirected BFS from shocked nodes) -----
        # Used by V3tH variant. Capped at 4 (>=4 lumped). One-hot 5-dim.
        # BFS on the undirected reach subgraph; cheap (~Nr+E) per event.
        Nr = len(reach_idx)
        shock_idx_local = np.where(sub_shock > 0.5)[0]
        if shock_idx_local.size > 0:
            from scipy.sparse.csgraph import dijkstra
            sub_und = (sub_A + sub_A.T).tocsr()
            dist_f = dijkstra(sub_und, indices=shock_idx_local,
                              unweighted=True, directed=False,
                              limit=4.0, min_only=True)
            dist = np.where(np.isfinite(dist_f), dist_f, 5).astype(np.int64)
        else:
            dist = np.full(Nr, 5, dtype=np.int64)
        hop_capped = np.minimum(dist, 4).astype(np.int64)
        hop_oh = np.eye(5, dtype=np.float32)[hop_capped]              # (Nr, 5)

        item = dict(
            event_id=eid,
            reach_idx=torch.as_tensor(reach_idx, dtype=torch.long),
            Nr=Nr,
            x_v=torch.as_tensor(sub_x_v),
            shock_mask=torch.as_tensor(sub_shock),
            delta0=torch.as_tensor(sub_delta0),
            P_ini=torch.as_tensor(sub_P_ini),
            sectors=torch.as_tensor(sub_sec),
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_a=edge_a,
            edge_outshare=edge_outshare,
            key_days=torch.as_tensor(key_days, dtype=torch.long),
            u_keyframes=torch.as_tensor(sub_u_kf),                    # (K, Nr)
            peak_loss=torch.as_tensor(sub_peak),                      # (Nr,)
            is_reach=torch.as_tensor(sub_is_reach),                   # (Nr,)
            # Cascade-depth aux target: which keyframe is u-trough (argmin over K)
            trough_day_target=torch.as_tensor(
                np.argmin(sub_u_kf, axis=0).astype(np.int64)),        # (Nr,) int64
            event_scalars=torch.as_tensor(event_scalars),             # (12,)
            shock_hop_oh=torch.as_tensor(hop_oh),                     # (Nr, 5)
        )
        if include_audit_metadata:
            item.update(
                audit_mode=str(meta.get("mode", "unknown")),
                audit_tier=str(meta.get("tier", "unknown")),
                audit_peak_aggregate_bau_loss=aggregate_bau_peak,
            )
        return item


def collate_single(batch):
    """Batch size = 1 (variable Nr per event). Just unwrap."""
    assert len(batch) == 1
    return batch[0]


def split_event_ids(events_dir: Path, cfg: Config):
    """80/10/10 split by event_id (CASCADE_DATA_SPEC §5)."""
    files = sorted(events_dir.glob("event_*.npz"))
    ids = [int(f.stem.split("_")[1]) for f in files]
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(n * cfg.train_frac)
    n_va = int(n * cfg.val_frac)
    return ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]
