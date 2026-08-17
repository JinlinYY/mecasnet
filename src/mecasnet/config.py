"""Configuration shared by MeCaSNet training, evaluation, and audits."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    # --- paths ---
    data_root: str = "data/chemical"
    static_meta: str = "static_meta.pkl"
    events_dir: str = "events"
    out_dir: str = "runs/chemical/mecasnet"

    # --- data ---
    key_days: List[int] = field(
        default_factory=lambda: [0, 5, 10, 20, 30, 50, 70, 100, 150, 199]
    )
    reach_hops: int = 5            # k-hop BFS ball around shock for ShockReach mask
    train_frac: float = 0.8        # default 80/10/10 train/validation/test split
    val_frac: float = 0.1
    seed: int = 0
    # Cross-domain knob: when False, drop sector one-hot from x_v_static so
    # Fv is identical across networks with different n_sec (Cangzhou=269 vs
    # CSMAR D≈100). Required for joint training across simulators.
    use_sector_oh: bool = True
    include_audit_metadata: bool = False

    # --- model ---
    # Per-node latent width used by the graph backbone and parameter heads.
    d_hidden: int = 64
    deq_iters: int = 6             # fixed-point iterations (Jacobian-free backprop)
    deq_grad_iters: int = 1        # last K iterations with gradient
    alpha_init: float = 8.0        # smooth-min temperature init (learnable);
    inv_steps_per_day: int = 1     # RK4 fixed-step integration granularity
    horizon: int = 200             # T = 200 days (last key_day = 199)
    use_neural_correction: bool = True   # add small MLP correction on top of physics rhs
    profile: str = "paper"
    decline_p: float = 6.0
    recovery_p: float = 1.0
    recovery_gate: bool = False
    recovery_q: bool = True
    recovery_q_min: float = 0.5
    recovery_q_max: float = 2.0
    physrecover: bool = False
    day0_demand_pullback: bool = False
    ablate_uncond: bool = False
    ablate_forward_only: bool = False

    # --- losses ---
    # Composite-objective weights for trajectory, peak, and auxiliary tasks.
    w_data_keyframes: float = 1.0
    w_data_peak: float = 2.0
    w_mono: float = 0.1            # cap on dot{u}: recovery rate ≤ 1/T_r
    # Late-phase monotone penalty: penalise u(t+1) < u(t) on keyframes after
    # mono_late_threshold_day. Targets v4's day-100/150 regression where the
    # damped-oscillation basis injects spurious dips. Holds for both Inoue
    # (monotone tails) and Henriet (recovery overshoot is non-negative).
    w_mono_late: float = 0.5
    mono_late_threshold_day: int = 70
    w_mass: float = 0.0            # mass-conservation residual (off by default — physics-structured rhs already conserves)
    w_phys: float = 0.0            # PINN residual; >0 only if rhs has neural part
    focal_inert_weight: float = 0.1  # u≈1 nodes down-weighted in data loss
    focal_shock_weight: float = 5.0  # directly shocked nodes up-weighted
    # Reachability auxiliary loss: BCE on is_reach = peak > reach_thresh.
    w_reach: float = 0.15
    reach_thresh: float = 0.001    # peak_loss > thresh ⇒ is_reach=1
    # Cascade-depth aux: predict per-node trough_day (which kf is argmin of u_kf).
    # Forces encoder to learn "when does this node bottom out" → improves per-node
    # identifiability at d30/d50 trough zone.  Active for cascade nodes (peak>0.05).
    w_trough_day: float = 0.05
    trough_day_peak_thresh: float = 0.05
    # Per-keyframe weights emphasize the principal trough windows.
    # length must equal len(key_days)
    kf_weights: List[float] = field(
        default_factory=lambda: [1.0, 2.0, 1.5, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]
        #                          d=0  5    10   20   30   50   70  100  150  199
    )

    # --- train ---
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_accum: int = 4            # events per optimizer step
    grad_clip: float = 1.0
    log_every: int = 20
    val_every: int = 1
    device: str = "cuda"
    train_num_workers: int = 0
    eval_num_workers: int = 0

    # --- risk head (v2; off by default) ---
    risk_enabled: bool = False
    risk_mc_samples: int = 0

    # --- leakage audit ---
    # event_scalars layout (12 dims, see data.py):
    #   0: log_rec.mean     [LEAK   per-firm recovery time mean]
    #   1: log_rec.std      [LEAK   per-firm recovery time std]
    #   2: delta_arr.mean   [OK    day-0 observable]
    #   3: meta.delta       [OK    tier-nominal initial damage]
    #   4: log recovery_days[SOFT  tier-nominal recovery prior]
    #   5-7: mode_oh        [SOFT  shock-sampling artifact]
    #   8-11: tier_oh       [SOFT  severity class]
    # event_scalars_mode:
    #   "full"    - keep all 12 dims (possible target-information leakage)
    #   "clean"   - zero out 0,1,5,6,7 (drop hard-leaky per-firm recovery
    #               + drop simulation-artifact mode); keep tier prior + tier_oh
    #   "minimal" - keep only mean observed Day-0 damage at index 2
    event_scalars_mode: str = "minimal"

    # --- cross-domain augmentation ---
    # When > 0, randomly drop edges at training time with probability p (per
    # event, applied AFTER reach-subgraph extraction). Forces the GNN to
    # learn size/density-invariant aggregation by exposing it to topologies
    # of varying connectivity. Eval is always p=0.
    edge_dropout_p: float = 0.0
    # When > 0, also randomly mask non-shock node FEATURES (zero them out)
    # to simulate sparser feature observability.
    node_feat_dropout_p: float = 0.0

    # --- universal-aggregation knobs (subgraph-aug + dimensionless inputs) ---
    # ① train-time random subgraph subsampling: with prob p, after k-hop reach
    #    extraction, keep all shocked nodes + random sample of non-shocked
    #    reach nodes down to size U[subgraph_aug_min, Nr]. Exposes the model
    #    to a continuous size spectrum (e.g. V=400..12.9k in Inoue).
    subgraph_aug_p: float = 0.0
    subgraph_aug_min: int = 400
    # ② per-event re-standardization of x_v_static: subtract column mean,
    #    divide by column std over THIS event's reach subgraph. Removes the
    #    last absolute-scale signal in node features (StaticNetwork already
    #    does per-network standardization).
    dimensionless_inputs: bool = False

    # --- mixed-domain training weights ---
    # Per-domain loss multiplier applied at the event level inside total_loss.
    # Compensates for Henriet's smaller absolute error magnitude (peaks ~3% vs
    # Inoue ~20%) so gradient contributions equalize. {} → no boost.
    domain_loss_weights: dict = field(default_factory=dict)
    # Per-domain kf_weights override (length must equal len(key_days)). Lets
    # Henriet use earlier-peaking weights instead of Inoue's day-30/50 bimodal
    # profile. Keys missing → falls back to cfg.kf_weights.
    kf_weights_by_domain: dict = field(default_factory=dict)


def get_paths(cfg: Config):
    root = Path(cfg.data_root)
    return {
        "static": root / cfg.static_meta,
        "events": root / cfg.events_dir,
        "out": Path(cfg.out_dir),
    }
