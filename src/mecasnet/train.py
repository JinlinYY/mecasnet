"""Train MeCaSNet and the manuscript comparators.

The driver performs per-epoch validation, selects the best checkpoint using
validation cascade R-squared, evaluates the selected model on the test split,
and writes aggregate and per-seed metrics. The default `paper` profile uses
only information available at Day 0 and matches the final manuscript
architecture; `legacy-y8` exists only for archived checkpoints.

Example:
  mecasnet-train --data-root /path/to/chemical-cascade-data
"""
from __future__ import annotations
import argparse
import copy
import json
import time
from pathlib import Path

import os as _os
def _dh_override(default):
    v = _os.environ.get("D_HIDDEN_OVERRIDE")
    return int(v) if v else default

import numpy as np
import torch

from .config import Config
from .data import StaticNetwork, split_event_ids
from .model_v3 import KSGATv3
from .model_v2 import PlainMLPBaseline, PlainGATBaseline, PlainGCNBaseline
from .model_baselines_strong import (
    DirGNNBaseline, STGNNBaseline, PhysicsAnalyticalBaseline,
)
from .factory import LEGACY_Y8_PROFILE, PAPER_PROFILE, PROFILES, apply_profile
from .runner import (
    train_with_val, evaluate, agg_seeds,
)
from torch.utils.data import DataLoader
from .data import CascadeEventDataset, collate_single


def _eval_on_split(state_dict, build_fn, ids, cfg, net, device):
    """Load `state_dict` into a fresh model and evaluate on the given event ids."""
    model = build_fn().to(device)
    model.load_state_dict(state_dict)
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.include_audit_metadata = True
    ds = CascadeEventDataset(eval_cfg, net, ids)
    dl = DataLoader(ds, batch_size=1, shuffle=False,
                    num_workers=0, collate_fn=collate_single)
    return evaluate(model, dl, eval_cfg, device)


def main():
    # ------- A800 free speedups (TF32 + cudnn.benchmark) -------
    # Henriet graph is fixed shape (V=429, E=877, K=10) so cudnn.benchmark
    # is safe. TF32 on Ampere typically gives 1.2-1.4x on matmul-bound
    # workloads (GAT projections, decoder heads). No semantic change.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", "--data_root", dest="data_root", required=True)
    ap.add_argument("--n-train", "--n_train", dest="n_train", type=int, default=4000)
    ap.add_argument("--n-val", "--n_val", dest="n_val", type=int, default=500)
    ap.add_argument("--n-test", "--n_test", dest="n_test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", "--seed_start", dest="seed_start",
                    type=int, default=0)
    ap.add_argument("--profile", choices=PROFILES, default=PAPER_PROFILE,
                    help="named architecture/input profile (default: paper)")
    ap.add_argument("--variant", type=str, default="MeCaSNet",
                    choices=["V3DE", "V3DE_G1", "V3DE_G3", "V3DE_G3_RES",
                             "V3DE_G3_RESA", "V3DE_G3_BLEND",
                             "V3DE_G3_TRI", "V3DE_G3_SYMREV", "V3DE_G3_BD",
                             "V3DE_G3_WIDE", "MeCaSNet",
                             "V3DE_NOBM", "V3DE_NOBM_NOOSC",
                             "MLP", "MLP_big",
                             "GAT", "GAT_deep",
                             "GCN", "GCN_deep",
                             "DirGNN", "DirGNN_deep",
                             "STGNN",
                             "Physics"],
                    help="which model variant to train (default: MeCaSNet). "
                         "Weak baselines: MLP/MLP_big (no graph), "
                         "GAT/GAT_deep (PlainGAT), GCN/GCN_deep (Kipf-Welling). "
                         "Strong baselines: DirGNN (directed GNN, Rossi 2023), "
                         "STGNN (A3T-GCN/DCRNN spatio-temporal), "
                         "Physics (pure analytical Leontief min-rule, no GNN).")
    ap.add_argument("--lr-schedule", "--lr_schedule", dest="lr_schedule",
                    type=str, default="cosine",
                    choices=["const", "cosine"],
                    help="'const' = fixed cfg.lr (legacy); 'cosine' = linear "
                         "warmup then cosine decay (default)")
    ap.add_argument("--warmup-epochs", "--warmup_epochs", dest="warmup_epochs",
                    type=int, default=3,
                    help="linear warmup length for cosine schedule (epochs)")
    ap.add_argument("--min-lr-ratio", "--min_lr_ratio", dest="min_lr_ratio",
                    type=float, default=0.1,
                    help="cosine decay floor as a fraction of cfg.lr")
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir", type=str,
                    default="runs/chemical/mecasnet")
    ap.add_argument("--save-ckpt", "--save_ckpt", dest="save_ckpt",
                    action="store_true",
                    help="save best checkpoint per seed under out_dir/")
    ap.add_argument("--event-scalars-mode", "--event_scalars_mode",
                    dest="event_scalars_mode", type=str, default="minimal",
                    choices=["full", "clean", "minimal"],
                    help="leakage audit: 'full'=12 dims as-is; "
                         "'clean'=drop per-firm recovery + mode_oh "
                         "(removes hard leak); 'minimal'=keep only the observed "
                         "mean Day-0 damage among directly shocked firms")
    ap.add_argument("--kf-trans-weight", "--kf_trans_weight",
                    dest="kf_trans_weight", type=float, default=1.0,
                    help="loss weight multiplier for the transition keyframes "
                         "d10 and d70 (key_days idx 2 and 6). 1.0 = off. "
                         "Used to counter peak-frame gradient dilution.")
    ap.add_argument("--resume-from", "--resume_from", dest="resume_from",
                    type=str, default="",
                    help="path to a saved ckpt (.pt) whose state_dict will "
                         "be loaded into each seed's model before training. "
                         "Optimizer and LR scheduler start fresh.")
    ap.add_argument("--split-manifest", "--split_manifest", dest="split_manifest",
                    type=str, default="",
                    help="optional JSON manifest with train_ids, validation_ids, "
                         "and internal_test_ids; bypasses the default random split")
    ap.add_argument("--strict-test-data-root", "--strict_test_data_root",
                    dest="strict_test_data_root", type=str, default="",
                    help="external held-out pool used only after validation has selected "
                         "the best checkpoint; requires --split_manifest with "
                         "strict_test_ids")
    ap.add_argument("--require-y8-exact", "--require_y8_exact",
                    dest="require_y8_exact", action="store_true",
                    help="fail before training unless the archived Y8-H triple-blend "
                        "architecture and 289,512-parameter profile are active")
    args = ap.parse_args()
    if args.profile == PAPER_PROFILE and args.event_scalars_mode != "minimal":
        ap.error("--profile paper requires --event-scalars-mode minimal")

    cfg = Config()
    apply_profile(cfg, args.profile)
    cfg.data_root = args.data_root
    cfg.epochs = args.epochs
    cfg.device = args.device
    cfg.deq_iters = 1
    cfg.deq_grad_iters = 1
    cfg.event_scalars_mode = args.event_scalars_mode
    print(f"[init] event_scalars_mode = {cfg.event_scalars_mode}")

    # Flat-top decline front (2026-06-14): MECAS_DECLINE_P sets the collapse
    # exponent on the physics decoder. 2.0 = historical Gaussian (no change);
    # >2 (e.g. 6) = super-Gaussian plateau-then-steep onset matching the
    # inventory-depletion front of the Henriet/Inoue simulators, to fix the
    # early-keyframe (d5/d10) over-early-decline. Applies to all param decoders.
    import os as _os_dp
    cfg.decline_p = float(_os_dp.environ.get("MECAS_DECLINE_P", str(cfg.decline_p)))
    if cfg.decline_p != 2.0:
        print(f"[init] MECAS_DECLINE_P = {cfg.decline_p}  (flat-top super-Gaussian decline front)")

    # Recovery-branch family (2026-06-14): MECAS_RECOVERY_P sets the rebound
    # exponent on the physics decoder. 1.0 = historical single-exp (no change);
    # >1 (e.g. 2) = Erlang/Gamma-k survival fn = lagged concave refill matching
    # the inventory-refill convolution of cascade nodes in both simulators, to
    # fix the too-fast mid-recovery (d30/d50). Applies to all param decoders.
    cfg.recovery_p = float(_os_dp.environ.get("MECAS_RECOVERY_P", str(cfg.recovery_p)))
    if cfg.recovery_p != 1.0:
        print(f"[init] MECAS_RECOVERY_P = {cfg.recovery_p}  (Erlang/Gamma lagged-refill recovery branch)")

    # Plan A (2026-06-15): per-node recovery gate. MECAS_RECOVERY_GATE=1 makes the
    # backbone predict g_v in [0,1] per node that mixes DIRECT single-exp (g=1)
    # and CASCADE Erlang-2 (g=0) recovery, overriding the global recovery_p.
    # Resolves the mixed-distribution wash (d30/d50 up but d70 collapse) that a
    # single global recovery_p caused. Adds (d+1) params. Applies to all param
    # decoders that go through the base KSGATv3.forward (incl. V3DE_G3_BD).
    cfg.recovery_gate = _os_dp.environ.get(
        "MECAS_RECOVERY_GATE", "1" if cfg.recovery_gate else "0"
    ) == "1"
    if cfg.recovery_gate:
        print("[init] MECAS_RECOVERY_GATE=1  (per-node direct/cascade recovery mix gate)")

    # PhysRecover (2026-06-15): MECAS_PHYSRECOVER=1 replaces the closed-form
    # recovery branch with an on-graph inventory relaxation that caps each
    # node's recovery by its upstream suppliers' current production
    # (u_v += (1-e^{-kappa_v dt})(min(own_recovery, supply_v) - u_v)). Targets
    # the d70+ recovery-seam collapse that a per-node closed form cannot express
    # (the graph+time coupling of P_act=min(P_cap,Leontief(S),D) in both sims).
    # SOLE recovery path, ~ (d+1) params (kappa head). Applies to all param
    # decoders through the base KSGATv3.forward (incl. V3DE_G3_BD / edge-state+free_res).
    cfg.physrecover = _os_dp.environ.get(
        "MECAS_PHYSRECOVER", "1" if cfg.physrecover else "0"
    ) == "1"
    if cfg.physrecover:
        cfg.physrec_kappa_min = float(_os_dp.environ.get("MECAS_PHYSREC_KMIN", "0.005"))
        cfg.physrec_kappa_max = float(_os_dp.environ.get("MECAS_PHYSREC_KMAX", "0.15"))
        print(f"[init] MECAS_PHYSRECOVER=1  (graph-coupled recovery integrator, "
              f"kappa in [{cfg.physrec_kappa_min},{cfg.physrec_kappa_max}]/day)")

    # Recovery self-capacity (2026-06-15): MECAS_RECOVERY_Q=1 makes the backbone
    # predict a per-node recovery-sharpness q_v in [MECAS_RECOVERY_QMIN,
    # MECAS_RECOVERY_QMAX] => recovery shape r(z)=exp(-z^q). q<1 speeds recovery,
    # q>1 lags it; q=1 (init) is the exact single-exp. Targets the d70 recovery
    # LAG the root-cause diagnostic localized to the decoder's own-state recovery
    # rigidity (NOT graph coupling). Adds (d+1) params. Applies to all param
    # decoders through the base KSGATv3.forward (incl. V3DE_G3_BD / edge-state+free_res).
    cfg.recovery_q = _os_dp.environ.get(
        "MECAS_RECOVERY_Q", "1" if cfg.recovery_q else "0"
    ) == "1"
    if cfg.recovery_q:
        cfg.recovery_q_min = float(_os_dp.environ.get("MECAS_RECOVERY_QMIN", "0.5"))
        cfg.recovery_q_max = float(_os_dp.environ.get("MECAS_RECOVERY_QMAX", "2.0"))
        print(f"[init] MECAS_RECOVERY_Q=1  (per-node recovery sharpness q in "
              f"[{cfg.recovery_q_min},{cfg.recovery_q_max}], r(z)=exp(-z^q))")

    # Wider free-residual corrector (2026-06-17): MECAS_RESIDUAL_WIDE=1 swaps the
    # flat Linear(d,K) free-residual head for a d->2d->K GELU MLP. Targets the
    # d50/d70 seam deficit the R2 decomposition attributed ~50% to calibration
    # (the linear head cannot express the localized per-node correction the
    # param3 G1/G2/G3 seams need). Requires MECAS_FREE_RESIDUAL=1. Adds ~2d*d+2d*K
    # params; last layer zero-init so it still boots at the exact physics traj.
    cfg.residual_wide = _os_dp.environ.get("MECAS_RESIDUAL_WIDE", "0") == "1"
    if cfg.residual_wide:
        print("[init] MECAS_RESIDUAL_WIDE=1  (free-residual head d->2d->K GELU MLP)")

    # Per-node seam temperature (2026-06-17): MECAS_SEAM_TAU=1 makes the trimodal
    # decoder predict a per-node softmin temperature τ_v = softplus(W·h+b)+1e-3
    # instead of the global scalar τ. Targets the d50/d70 seam r-drop (node
    # ranking loss) the R2 decomposition localized to the param3 G1/G2 & G2/G3
    # crossovers: the global hard min clamps adjacent nodes toward the crossing
    # value, amplifying μ/σ errors into rank noise. τ_v lets cascade-buffered
    # nodes blend gracefully (rank preserved) while shocked nodes stay sharp.
    # The ONLY seam fix that keeps a pure closed-form physics decoder. Adds
    # (d+1) params; bias-init so τ_v=0.02 at start (zero behavioural change).
    cfg.seam_tau_pernode = _os_dp.environ.get("MECAS_SEAM_TAU", "0") == "1"
    if cfg.seam_tau_pernode:
        print("[init] MECAS_SEAM_TAU=1  (per-node trimodal softmin temperature τ_v)")

    # d70 G2-collapse fix (2026-06-17). The d70 root-cause diagnostic + 5 failed
    # warm-start arms proved d70 collapses because the mid Gaussian G2 is
    # gradient-starved: under near-hard-min softmin (tau~0.02) it is never the
    # deepest curve on any day, so T_r2 pins to its floor and d70 R2 falls
    # 0.63 -> 0.015. These two CONSTRUCTION-LEVEL gates (from-scratch only) attack
    # the two halves of that root cause:
    #   MECAS_G2_REPOS=1     -> narrow mu2 [20,90]->[MECAS_P3_MU2_OFF, +SCALE] (def
    #                          45..85) so G2's trough lands in the d70 window and
    #                          can WIN the softmin there (positioning).
    #   MECAS_SEAM_TAU_INITB -> seam_tau head init bias (def -3.954 = tau 0.02).
    #                          Raise (e.g. -2.0 => tau~0.12) so the softmin passes
    #                          gradient to G2 from the start (anti-starvation).
    cfg.p3_g2_repos = _os_dp.environ.get("MECAS_G2_REPOS", "0") == "1"
    if cfg.p3_g2_repos:
        cfg.p3_mu2_off_override = float(_os_dp.environ.get("MECAS_P3_MU2_OFF", "45"))
        cfg.p3_mu2_scale_override = float(_os_dp.environ.get("MECAS_P3_MU2_SCALE", "40"))
        print(f"[init] MECAS_G2_REPOS=1  (mu2 -> [{cfg.p3_mu2_off_override:.0f}, "
              f"{cfg.p3_mu2_off_override + cfg.p3_mu2_scale_override:.0f}] to force G2 into d70 window)")
    cfg.seam_tau_init_bias = float(_os_dp.environ.get("MECAS_SEAM_TAU_INITB", "-3.954"))
    if cfg.seam_tau_init_bias != -3.954:
        import math as _math_tau
        _tau0 = _math_tau.log1p(_math_tau.exp(cfg.seam_tau_init_bias)) + 1e-3
        print(f"[init] MECAS_SEAM_TAU_INITB={cfg.seam_tau_init_bias}  (seam tau init = {_tau0:.3f}, vs default 0.02)")

    # Hierarchical dual seam temperature (2026-06-17): MECAS_SEAM_TAU2=1 adds a
    # SECOND per-node softmin temperature so the trimodal composition becomes a
    # two-level nested softmin (tau1_v owns the d50 G1/G2 seam, tau2_v the d70
    # G2/G3 seam INDEPENDENTLY). Fixes the single-tau B failure mode (d50->0.73
    # but d70->-0.30: one tau cannot serve both seams). Implies MECAS_SEAM_TAU=1.
    cfg.seam_tau_dual = _os_dp.environ.get("MECAS_SEAM_TAU2", "0") == "1"
    if cfg.seam_tau_dual:
        cfg.seam_tau_pernode = True
        print("[init] MECAS_SEAM_TAU2=1  (dual per-node seam temperatures τ1_v/τ2_v, nested softmin)")

    # Time-masked PROTECTED residual (2026-06-17): MECAS_RESID_TAIL_MASK=1 ->
    # cfg.resid_tail_mask. Masks the free residual to the decline/peak window
    # (days<=MECAS_RESID_MASK_DAY, default 50) and forces it to ZERO on the
    # recovery tail (d70+). The residual trade-off ceiling showed the free
    # residual helps decline/peak (d5..d30, peak) but overfits the recovery tail
    # (clean d199 0.155->-0.258), and in real training this tail overfit
    # transfers to d70 (-0.30). Masking keeps the peak/bulk gain, restores the
    # tail, and structurally prevents the d70 collapse. Requires MECAS_FREE_RESIDUAL=1.
    cfg.resid_tail_mask = _os_dp.environ.get("MECAS_RESID_TAIL_MASK", "0") == "1"
    cfg.resid_mask_day = float(_os_dp.environ.get("MECAS_RESID_MASK_DAY", "50"))
    if cfg.resid_tail_mask:
        print(f"[init] MECAS_RESID_TAIL_MASK=1  (free residual active only on days<={cfg.resid_mask_day:.0f}, zero on recovery tail)")

    # ----- Henriet ablation toggles (2026-06-16) -----------------------------
    # Clean single-component removals for the flagship ablation table. All three
    # default OFF so every existing run is byte-for-byte unchanged.
    #   MECAS_ABLATE_UNCOND=1       -> cfg.ablate_uncond: strip Day-0 shock identity
    #     (shock_mask & delta0) from the node-encoder input (unconditioned
    #     representation). u0 physical IC + FiLM shock_frac kept.
    #   MECAS_ABLATE_FORWARD_ONLY=1 -> cfg.ablate_forward_only: drop the reverse
    #     (downstream->upstream demand-feedback) message in the prewarm +
    #     param-decoder spatial loops (forward-only propagation).
    #   MECAS_NO_REACH=1            -> cfg.w_reach=0: disable the cascade-boundary
    #     reachability BCE auxiliary loss (regression-only training).
    cfg.ablate_uncond = _os_dp.environ.get("MECAS_ABLATE_UNCOND", "0") == "1"
    if cfg.ablate_uncond:
        print("[init] MECAS_ABLATE_UNCOND=1  (unconditioned representation: "
              "shock_mask & delta0 zeroed at encoder)")
    cfg.ablate_forward_only = _os_dp.environ.get("MECAS_ABLATE_FORWARD_ONLY", "0") == "1"
    if cfg.ablate_forward_only:
        print("[init] MECAS_ABLATE_FORWARD_ONLY=1  (forward-only propagation: "
              "reverse demand-feedback message dropped)")
    if _os_dp.environ.get("MECAS_NO_REACH", "0") == "1":
        cfg.w_reach = 0.0
        print("[init] MECAS_NO_REACH=1  (regression-only: reachability BCE aux disabled, w_reach=0)")

    if args.kf_trans_weight != 1.0:
        # multiply the EXISTING per-keyframe weights (NOT reset to 1.0) so the
        # established d5/d30/d50 trough weighting is preserved; only the two
        # transition frames d10/d70 get the extra boost.
        kfw = list(cfg.kf_weights)
        for i, d in enumerate(cfg.key_days):
            if d in (10, 70):
                kfw[i] = kfw[i] * args.kf_trans_weight
        cfg.kf_weights = kfw
        print(f"[init] kf_weights = {kfw}  (transition d10/d70 x{args.kf_trans_weight})")

    # Seam loss reweighting (2026-06-17): MECAS_SEAM_KFW=<w> multiplies the d50 &
    # d70 keyframe loss weights by w. The R2 decomposition localized the only
    # flagship deficit vs the edge-state baseline to the param3 G1/G2 (d50) and G2/G3
    # (d70) seams; this is the LOSS-side lever (orthogonal to the wide-residual
    # and triple-blend architecture fixes): force the optimizer to allocate
    # capacity to the two under-fit seam frames instead of letting the early
    # frames dominate the gradient. Cheapest possible test of whether the seam
    # gap is an optimization-allocation issue vs a structural ceiling.
    _seam_w = float(_os_dp.environ.get("MECAS_SEAM_KFW", "1.0"))
    if _seam_w != 1.0:
        kfw = list(cfg.kf_weights)
        for i, d in enumerate(cfg.key_days):
            if d in (50, 70):
                kfw[i] = kfw[i] * _seam_w
        cfg.kf_weights = kfw
        print(f"[init] MECAS_SEAM_KFW={_seam_w}  kf_weights = {kfw}  (seam d50/d70 x{_seam_w})")

    # d70-only reweighting (2026-06-17): MECAS_D70_KFW=<w> multiplies ONLY the d70
    # keyframe loss weight by w. Source audit confirmed the root cause of the
    # kf̄ plateau: the default kf_weights have d5/d30/d50 boosted to 2.0 but the
    # sole recovery seam frame d70 left at 1.0 (HALF of d50). Combined with the
    # masked residual (which zeroes the residual path for day>50), d70 is driven
    # purely by the physics G2/G3 form yet receives half the per-frame gradient
    # of d50 -> systematically under-trained, well below its own ceiling (~0.63
    # established by the frozen-h fresh-head probe). Unlike MECAS_SEAM_KFW this
    # does NOT touch d50 (already at 2.0). This is the targeted lever: lift d70's
    # gradient share so the optimizer drives the physics form toward its ceiling.
    # NOTE (honest): this nudges d70 toward ~0.63, it does NOT surpass the edge-state
    # baseline (0.646) or the d70 ceiling -- the masked-residual design keeps
    # aggregate kf̄ roughly flat (seam is only 2/10 frames).
    _d70_w = float(_os_dp.environ.get("MECAS_D70_KFW", "1.0"))
    if _d70_w != 1.0:
        kfw = list(cfg.kf_weights)
        for i, d in enumerate(cfg.key_days):
            if d == 70:
                kfw[i] = kfw[i] * _d70_w
        cfg.kf_weights = kfw
        print(f"[init] MECAS_D70_KFW={_d70_w}  kf_weights = {kfw}  (d70 x{_d70_w})")

    # Experimental environment variables remain available to archived/custom
    # variants, but the named manuscript profile has an immutable information
    # and architecture boundary.
    if args.profile == PAPER_PROFILE:
        apply_profile(cfg, PAPER_PROFILE)
        print("[init] paper profile locked (Day-0 inputs and final architecture)")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] loading static network from {args.data_root}")
    net = StaticNetwork(cfg)
    print(f"       V={net.V}  E={net.E}  Fv={net.Fv}")
    events_dir = Path(cfg.data_root) / cfg.events_dir
    strict_cfg = None
    strict_net = None
    strict_test_ids = []
    if args.split_manifest:
        manifest_path = Path(args.split_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_ids = [int(event_id) for event_id in manifest["train_ids"]]
        val_ids = [int(event_id) for event_id in manifest["validation_ids"]]
        test_ids = [int(event_id) for event_id in manifest["internal_test_ids"]]
        split_sets = [set(train_ids), set(val_ids), set(test_ids)]
        if any(split_sets[left] & split_sets[right]
               for left in range(3) for right in range(left + 1, 3)):
            raise ValueError(f"Overlapping event IDs in split manifest: {manifest_path}")
        if not train_ids or not val_ids or not test_ids:
            raise ValueError(f"Empty train/validation/internal-test split: {manifest_path}")
        missing_ids = [
            event_id for event_id in train_ids + val_ids + test_ids
            if not (events_dir / f"event_{event_id:06d}.npz").exists()
        ]
        if missing_ids:
            raise FileNotFoundError(
                f"Manifest references {len(missing_ids)} missing events; first={missing_ids[:5]}"
            )
        if args.strict_test_data_root:
            strict_test_ids = [int(event_id) for event_id in manifest["strict_test_ids"]]
            if not strict_test_ids:
                raise ValueError(f"Empty strict-test split: {manifest_path}")
            strict_cfg = copy.deepcopy(cfg)
            strict_cfg.data_root = args.strict_test_data_root
            strict_events_dir = Path(strict_cfg.data_root) / strict_cfg.events_dir
            missing_strict_ids = [
                event_id for event_id in strict_test_ids
                if not (strict_events_dir / f"event_{event_id:06d}.npz").exists()
            ]
            if missing_strict_ids:
                raise FileNotFoundError(
                    f"Strict-test manifest references {len(missing_strict_ids)} missing "
                    f"events; first={missing_strict_ids[:5]}"
                )
            strict_net = StaticNetwork(strict_cfg)
            if (strict_net.V, strict_net.E, strict_net.Fv) != (net.V, net.E, net.Fv):
                raise ValueError(
                    "Training and strict-test static networks are incompatible: "
                    f"train={(net.V, net.E, net.Fv)} strict="
                    f"{(strict_net.V, strict_net.E, strict_net.Fv)}"
                )
            static_arrays = ("sectors", "P_ini", "edge_src", "edge_dst", "edge_share")
            mismatched_arrays = [
                name for name in static_arrays
                if not np.array_equal(getattr(net, name), getattr(strict_net, name))
            ]
            if mismatched_arrays:
                raise ValueError(
                    "Training and strict-test pools do not share the exact fixed graph; "
                    f"mismatched arrays={mismatched_arrays}"
                )
        print(f"[init] split_manifest = {manifest_path}")
        print("       using pre-registered held-out-subgraph IDs; n_train/n_val/n_test ignored")
    else:
        train_ids, val_ids, test_ids = split_event_ids(events_dir, cfg)
        train_ids = train_ids[:args.n_train]
        val_ids = val_ids[:args.n_val]
        test_ids = test_ids[:args.n_test]
    print(f"[init] {len(train_ids)} train / {len(val_ids)} val / "
            f"{len(test_ids)} internal-test / {len(strict_test_ids)} strict-test events")

    Fv = net.Fv

    # Variant builder —V3DE_G3 default (SOTA per round-7 ablation).
    # V3DE_G1: G1 wide-range bimodal init.  V3DE: original (no preset).
    # Baselines (MLP/GAT/GCN) use identical encoder + heads, only the
    # spatial aggregation differs; trained under the same protocol.
    _PRESET = {"V3DE": "default",
               "V3DE_G1": "deep_mid",
               "V3DE_G3": "g3_narrow"}
    _preset_name = _PRESET.get(args.variant, None)

    if args.variant in _PRESET:
        def build_v3de():
            return KSGATv3(cfg, Fv=Fv, peak_mode="traj",
                           prewarm_layers=3, decoder_mode="param2",
                           param2_preset=_preset_name)
        print(f"[init] variant = {args.variant}  (param2_preset='{_preset_name}')")
    elif args.variant == "V3DE_G3_RES":
        from .model_v3_residual import KSGATv3Residual
        def build_v3de():
            return KSGATv3Residual(cfg, Fv=Fv, peak_mode="traj",
                                   prewarm_layers=3, decoder_mode="param2",
                                   param2_preset="g3_narrow")
        print(f"[init] variant = V3DE_G3_RES  (g3_narrow + late_residual head, "
              f"alpha gated d>=20)")
    elif args.variant == "V3DE_G3_RESA":
        from .model_v3_decoder_ac import KSGATv3ResidualA
        def build_v3de():
            return KSGATv3ResidualA(cfg, Fv=Fv, peak_mode="traj",
                                    prewarm_layers=3, decoder_mode="param2",
                                    param2_preset="g3_narrow")
        print(f"[init] variant = V3DE_G3_RESA  (variant A: earlier alpha gate at "
              f"d10=0.40 + wider residual head d->2d->K; targets d10/d70 transition)")
    elif args.variant == "V3DE_G3_BLEND":
        from .model_v3_decoder_ac import KSGATv3BlendC
        def build_v3de():
            return KSGATv3BlendC(cfg, Fv=Fv, peak_mode="traj",
                                 prewarm_layers=3, decoder_mode="param2",
                                 param2_preset="g3_narrow")
        print(f"[init] variant = V3DE_G3_BLEND  (variant C: bimodal physics decoder "
              f"+ free GRU rollout, per-keyframe learnable gate alpha_kf)")
    elif args.variant == "V3DE_G3_TRI":
        from .model_v3_residual import KSGATv3Residual
        def build_v3de():
            return KSGATv3Residual(cfg, Fv=Fv, peak_mode="traj",
                                   prewarm_layers=3, decoder_mode="param3")
        print(f"[init] variant = V3DE_G3_TRI  (variant B: trimodal param3 decoder + "
              f"residual head; 3rd Gaussian fills the d70+ vacuum, fully closed-form)")
    elif args.variant == "V3DE_G3_SYMREV":
        from .model_v3_decoder_ac import KSGATv3SymRev
        def build_v3de():
            return KSGATv3SymRev(cfg, Fv=Fv, peak_mode="traj",
                                 prewarm_layers=3, decoder_mode="param2",
                                 param2_preset="g3_narrow")
        print(f"[init] variant = V3DE_G3_SYMREV  (variant D: DirGNN-style symmetric "
              f"reverse backbone, bimodal decoder UNCHANGED; targets transition frames)")
    elif args.variant == "V3DE_G3_BD":
        from .model_v3_decoder_ac import KSGATv3SymRevTri
        import os as _os
        # CLOSED-LOOP ABLATION (2026-06-14): allow triple_blend on the BD
        # (SymRevTri, NO edge-state backbone) variant, to isolate whether the +170K
        # edge-state edge-state backbone is needed GIVEN triblend, or whether triblend
        # alone (on plain physics backbone) already reaches the Y8 kf̄.
        # MECAS_TRIPLE_BLEND=1 / MECAS_BLEND_ROLLOUT=1 / MECAS_BLEND_INIT enable it.
        _bd_triple = _os.environ.get("MECAS_TRIPLE_BLEND", "0") == "1"
        _bd_roll = _os.environ.get("MECAS_BLEND_ROLLOUT", "0") == "1"
        _bd_bi_str = _os.environ.get("MECAS_BLEND_INIT", "2,0,0")
        try:
            _bd_bi = tuple(float(x) for x in _bd_bi_str.split(","))
            assert len(_bd_bi) == 3
        except Exception:
            _bd_bi = (2.0, 0.0, 0.0)
        # Plan A: physics-prior free residual on the BD (no-backbone) variant.
        _bd_free = _os.environ.get("MECAS_FREE_RESIDUAL", "0") == "1"
        if _bd_free:
            assert not (_bd_triple or _bd_roll), \
                "MECAS_FREE_RESIDUAL is standalone; unset MECAS_TRIPLE_BLEND/MECAS_BLEND_ROLLOUT"
        def build_v3de():
            return KSGATv3SymRevTri(cfg, Fv=Fv, peak_mode="traj",
                                    prewarm_layers=3,
                                    blend_rollout=_bd_roll,
                                    triple_blend=_bd_triple,
                                    blend_init=_bd_bi,
                                    free_residual=_bd_free)
        print(f"[init] variant = V3DE_G3_BD  (B+D fusion: symmetric reverse backbone "
              f"+ trimodal param3 decoder + residual head; self-gating 3rd Gaussian"
              f" | triple_blend={_bd_triple} blend_rollout={_bd_roll} "
              f"blend_init={_bd_bi} free_residual={_bd_free})")
    elif args.variant == "MeCaSNet":
        from .model_v3_decoder_ac import KSGATv3EdgeState
        # Read optional n_edge_blocks from env so we don't have to add a CLI flag
        # (default 4 for ~+170K params; cut to 3 for tighter param budget)
        import os as _os
        _np = (4 if args.profile == PAPER_PROFILE
               else int(_os.environ.get("MECAS_N_PROC", "4")))
        # Plan 6 (2026-06-12): optional learnable softmin temperature for the
        # trimodal envelope. MECAS_LEARN_TAU=1 turns on a learnable scalar τ
        # (softplus-parameterised, init via MECAS_TAU_INIT, default 0.02) so
        # the d10/d70 seam between Gaussians is differentiable. Adds 1 param.
        _learn_tau = (False if args.profile == PAPER_PROFILE
                      else _os.environ.get("MECAS_LEARN_TAU", "0") == "1")
        _tau_init = (0.02 if args.profile == PAPER_PROFILE
                     else float(_os.environ.get("MECAS_TAU_INIT", "0.02")))
        # Plan 7 (2026-06-12): blend_rollout —run rollout in parallel and
        # learn per-keyframe α to blend rollout-u_k with the trimodal
        # parametric u_k. Preserves physics narrative (decoder still emits
        # (μ,σ,A) triples) and lets the model lean on rollout where the
        # trimodal envelope is weak (d10/d70 seams, late tail).
        # Adds K=10 alpha params (negligible). MECAS_BLEND_ROLLOUT=1 enables.
        _blend_rollout = (False if args.profile == PAPER_PROFILE
                          else _os.environ.get("MECAS_BLEND_ROLLOUT", "0") == "1")
        # Plan 3 (2026-06-12): triple_blend —adds a 3rd path (direct-readout
        # head, Linear(d,K)->sigmoid; mirrors the pure-edge-state baseline) and
        # learns per-keyframe softmax mixing weights across
        #   { trimodal, rollout, direct }.
        # Implicitly turns on blend_rollout. Preserves physics interpretability
        # (trimodal still emits μ,σ,A) but gives the model the unconstrained
        # flexibility of pure-edge-state as a third option per frame.
        # MECAS_TRIPLE_BLEND=1 enables.
        _triple_blend = (True if args.profile == PAPER_PROFILE else
                         _os.environ.get("MECAS_TRIPLE_BLEND", "0") == "1")
        # Blend-init sweep knob: initial logits for the triple_blend softmax
        # over {physics, rollout, direct}. "physics,rollout,direct".
        _blend_init_str = ("2,0,0" if args.profile == PAPER_PROFILE
                           else _os.environ.get("MECAS_BLEND_INIT", "2,0,0"))
        try:
            _blend_init = tuple(float(x) for x in _blend_init_str.split(","))
            assert len(_blend_init) == 3
        except Exception:
            _blend_init = (2.0, 0.0, 0.0)
        print(f"[init] MECAS_BLEND_INIT (physics,rollout,direct) = {_blend_init}")
        # Plan A (2026-06-14): physics-prior free residual. u = u_phys + Δ_free,
        # Δ = zero-init Linear(d,K) => boots at exact physics traj (BD) and learns
        # an unconstrained correction with gradient from step 1. No blend/softmax
        # => no init-trap. MECAS_FREE_RESIDUAL=1 enables (standalone; exclusive with
        # triple_blend/blend_rollout/nobm).
        _free_residual = (False if args.profile == PAPER_PROFILE else
                          _os.environ.get("MECAS_FREE_RESIDUAL", "0") == "1")
        if _free_residual:
            assert not (_triple_blend or _blend_rollout), \
                "MECAS_FREE_RESIDUAL is standalone; unset MECAS_TRIPLE_BLEND/MECAS_BLEND_ROLLOUT"
            print("[init] MECAS_FREE_RESIDUAL=1  (u = u_physics + Δ_free, zero-init Δ)")
        # the residual head entirely; use the pure rollout (decoder_mode=
        # "rollout") for u_keyframes. This is the apples-to-apples comparison
        # to pure-edge-state (same encoder depth, no physics decoder bias). Useful
        # diagnostic: tells us whether the trimodal envelope itself is the
        # ceiling, vs the edge-state backbone.
        # NOTE: this is INCOMPATIBLE with blend_rollout / triple_blend because
        # they all require decoder_mode in {param,param2,param3}. MECAS_NOBM=1
        # silently overrides the two blend flags.
        _nobm = (False if args.profile == PAPER_PROFILE else
                 _os.environ.get("MECAS_NOBM", "0") == "1")

        if args.require_y8_exact:
            y8_checks = {
                "n_edge_blocks=4": _np == 4,
                "triple_blend=true": _triple_blend,
                "standalone_blend_rollout=false": not _blend_rollout,
                "blend_init=(2,0,0)": _blend_init == (2.0, 0.0, 0.0),
                "learnable_tau=false": not _learn_tau,
                "nobm=false": not _nobm,
                "free_residual=false": not _free_residual,
                "decline_p=2": cfg.decline_p == 2.0,
                "recovery_p=1": cfg.recovery_p == 1.0,
                "recovery_gate=false": not cfg.recovery_gate,
                "physrecover=false": not cfg.physrecover,
                "recovery_q=false": not cfg.recovery_q,
            }
            failed_checks = [name for name, passed in y8_checks.items() if not passed]
            if failed_checks:
                raise RuntimeError(f"Y8-H exact-profile check failed: {failed_checks}")

        if _nobm:
            # Pure rollout path on edge-state encoder. The trimodal decoder is not
            # built; the residual head is bypassed. Result matches the
            # historical V3DE_NOBM variant but on the edge-state backbone.
            assert not (_blend_rollout or _triple_blend), \
                "MECAS_NOBM=1 is exclusive with MECAS_BLEND_ROLLOUT / MECAS_TRIPLE_BLEND"
            def build_v3de():
                return KSGATv3EdgeState(cfg, Fv=Fv, peak_mode="traj",
                                          prewarm_layers=3,
                                          n_edge_blocks=_np,
                                          decoder_mode="rollout",
                                          param2_preset="default",
                                          skip_residual_head=True)
            print(f"[init] variant = MeCaSNet+NOBM  (rollout-only on edge-state "
                  f"backbone n_proc={_np}; trimodal decoder + residual head "
                  f"BYPASSED; apples-to-apples vs pure-edge-state baseline)")
        else:
            def build_v3de():
                return KSGATv3EdgeState(cfg, Fv=Fv, peak_mode="traj",
                                          prewarm_layers=3,  # overridden inside
                                          n_edge_blocks=_np,
                                          bimodal_tau=_tau_init,
                                          learnable_bimodal_tau=_learn_tau,
                                          blend_rollout=_blend_rollout,
                                          triple_blend=_triple_blend,
                                          blend_init=_blend_init,
                                          free_residual=_free_residual)
            print(f"[init] variant = MeCaSNet  (B+D + edge-state-style edge-state backbone, "
                  f"n_proc={_np}; trimodal decoder + residual head + α gate UNCHANGED; "
                  f"adds persistent edge hidden state across prewarm layers; "
                  f"learnable_bimodal_tau={_learn_tau}, tau_init={_tau_init}, "
                  f"blend_rollout={_blend_rollout}, triple_blend={_triple_blend}, "
                  f"free_residual={_free_residual})")
    elif args.variant == "V3DE_G3_WIDE":
        from .model_v3_residual import KSGATv3Residual
        def build_v3de():
            return KSGATv3Residual(cfg, Fv=Fv, peak_mode="traj",
                                   prewarm_layers=3, decoder_mode="param2",
                                   param2_preset="g3_wide")
        print(f"[init] variant = V3DE_G3_WIDE  (flagship + mu2 range widened to "
              f"[0,200] —fixes the d70+ structural blindness of g3_narrow; "
              f"zero new params)")
    elif args.variant == "V3DE_NOBM":
        def build_v3de():
            return KSGATv3(cfg, Fv=Fv, peak_mode="traj",
                           prewarm_layers=3,
                           decoder_mode="rollout",
                           param2_preset="default")
        print(f"[init] variant = V3DE_NOBM  (decoder_mode='rollout', no bimodal head)")
    elif args.variant == "V3DE_NOBM_NOOSC":
        def build_v3de():
            return KSGATv3(cfg, Fv=Fv, peak_mode="traj",
                           prewarm_layers=3,
                           decoder_mode="rollout",
                           param2_preset="default",
                           use_inv_state=False)
        print(f"[init] variant = V3DE_NOBM_NOOSC  (no bimodal head, no S/O oscillator)")
    elif args.variant == "MLP":
        def build_v3de():
            return PlainMLPBaseline(cfg, Fv=Fv, n_layers=3)
        print(f"[init] variant = MLP  (n_layers=3, d={cfg.d_hidden}; no graph)")
    elif args.variant == "MLP_big":
        def build_v3de():
            return PlainMLPBaseline(cfg, Fv=Fv, n_layers=6, d_hidden=_dh_override(128))
        print(f"[init] variant = MLP_big  (n_layers=6, d=128; no graph)")
    elif args.variant == "GAT":
        def build_v3de():
            return PlainGATBaseline(cfg, Fv=Fv, n_layers=3)
        print(f"[init] variant = GAT  (PlainGAT n_layers=3, d={cfg.d_hidden})")
    elif args.variant == "GAT_deep":
        def build_v3de():
            return PlainGATBaseline(cfg, Fv=Fv, n_layers=8, d_hidden=_dh_override(96))
        print(f"[init] variant = GAT_deep  (PlainGAT n_layers=8, d=96)")
    elif args.variant == "GCN":
        def build_v3de():
            return PlainGCNBaseline(cfg, Fv=Fv, n_layers=3)
        print(f"[init] variant = GCN  (Kipf-Welling n_layers=3, d={cfg.d_hidden})")
    elif args.variant == "GCN_deep":
        def build_v3de():
            return PlainGCNBaseline(cfg, Fv=Fv, n_layers=8, d_hidden=_dh_override(96))
        print(f"[init] variant = GCN_deep  (Kipf-Welling n_layers=8, d=96)")
    elif args.variant == "DirGNN":
        def build_v3de():
            return DirGNNBaseline(cfg, Fv=Fv, n_layers=8, d_hidden=_dh_override(96))
        print(f"[init] variant = DirGNN  (directed GNN n_layers=8, d=96, ~240k matched)")
    elif args.variant == "DirGNN_deep":
        def build_v3de():
            return DirGNNBaseline(cfg, Fv=Fv, n_layers=12, d_hidden=_dh_override(96))
        print(f"[init] variant = DirGNN_deep  (directed GNN n_layers=12, d=96, capacity probe)")
    elif args.variant == "STGNN":
        def build_v3de():
            return STGNNBaseline(cfg, Fv=Fv, n_spatial=2, d_hidden=_dh_override(160))
        print(f"[init] variant = STGNN  (A3T-GCN/DCRNN rollout, d=160, ~265k matched)")
    elif args.variant == "Physics":
        def build_v3de():
            return PhysicsAnalyticalBaseline(cfg, Fv=Fv, n_hops=6)
        print(f"[init] variant = Physics  (pure analytical Leontief min-rule, no GNN)")
    else:
        raise ValueError(f"unknown variant: {args.variant}")
    build_v3de.__name__ = f"build_{args.variant.lower()}"

    if args.require_y8_exact:
        if args.profile != LEGACY_Y8_PROFILE:
            raise ValueError(
                "--require-y8-exact is only valid with --profile legacy-y8"
            )
        if args.variant != "MeCaSNet":
            raise ValueError("--require_y8_exact requires --variant MeCaSNet")
        profile_model = build_v3de()
        profile_params = sum(parameter.numel() for parameter in profile_model.parameters())
        del profile_model
        if profile_params != 289512:
            raise RuntimeError(
                f"Y8-H parameter-count check failed: expected 289512, got {profile_params}"
            )
        print("[init] Y8-H EXACT PROFILE PASS  params=289512")

    if args.resume_from:
        print(f"[resume] loading weights from {args.resume_from}")
        _ckpt = torch.load(args.resume_from, map_location="cpu")
        _init_sd = _ckpt["state_dict"] if "state_dict" in _ckpt else _ckpt
        _orig_build = build_v3de
        def build_v3de():
            m = _orig_build()
            missing, unexpected = m.load_state_dict(_init_sd, strict=False)
            if missing:    print(f"[resume] missing keys: {len(missing)} (first: {missing[:3]})")
            if unexpected: print(f"[resume] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
            return m
        build_v3de.__name__ = f"build_{args.variant.lower()}_resume"

    seed_range = list(range(args.seed_start, args.seed_start + args.seeds))
    val_runs, test_runs, strict_test_runs, histories = [], [], [], []
    t_global = time.time()
    for s_idx, seed in enumerate(seed_range):
        print()
        print("#" * 70)
        print(f"# SEED {seed}  ({s_idx + 1}/{args.seeds})")
        print("#" * 70)
        torch.manual_seed(seed); np.random.seed(seed)
        t0 = time.time()
        best_val, history, best_state = train_with_val(
            cfg, net, train_ids, val_ids, device,
            build_model_fn=build_v3de, n_epochs=args.epochs,
            verbose=True,
            lr_schedule=args.lr_schedule,
            warmup_epochs=args.warmup_epochs,
            min_lr_ratio=args.min_lr_ratio,
        )
        wall = time.time() - t0
        # final test eval on best checkpoint
        test_metrics = _eval_on_split(best_state, build_v3de,
                                      test_ids, cfg, net, device)
        test_metrics["params"] = best_val["params"]
        test_metrics["best_epoch"] = best_val["epoch"]
        test_metrics["wall_s"] = round(wall, 1)
        strict_test_metrics = None
        if strict_cfg is not None and strict_net is not None:
            strict_test_metrics = _eval_on_split(
                best_state, build_v3de, strict_test_ids,
                strict_cfg, strict_net, device,
            )
            strict_test_metrics["params"] = best_val["params"]
            strict_test_metrics["best_epoch"] = best_val["epoch"]
        val_runs.append(best_val)
        test_runs.append(test_metrics)
        if strict_test_metrics is not None:
            strict_test_runs.append(strict_test_metrics)
        histories.append(history)
        print(f"  [seed {seed}] BEST @ ep {best_val['epoch']:3d}  "
              f"val csc={best_val['r2_pk_csc']:+.3f}  "
              f"-> test csc={test_metrics['r2_pk_csc']:+.3f}  "
              f"shk={test_metrics['r2_pk_shk']:+.3f}  "
              f"kf̄={test_metrics.get('r2_kf_csc_mean', float('nan')):+.3f}  "
              f"({wall/60:.1f}m)")
        if strict_test_metrics is not None:
            print(f"             strict-heldout: csc="
                  f"{strict_test_metrics['r2_pk_csc']:+.3f}  "
                  f"shk={strict_test_metrics['r2_pk_shk']:+.3f}  "
                  f"kf̄={strict_test_metrics.get('r2_kf_csc_mean', float('nan')):+.3f}")
        if args.save_ckpt:
            ckpt_path = out_dir / f"{args.variant.lower()}_seed{seed}.pt"
            ckpt_cfg = {"variant": args.variant, "profile": args.profile}
            if args.variant in _PRESET:
                ckpt_cfg.update({"prewarm_layers": 3,
                                 "decoder_mode": "param2",
                                 "param2_preset": _preset_name,
                                 "peak_mode": "traj"})
            torch.save({"state_dict": best_state,
                        "best_val": best_val,
                        "internal_test": test_metrics,
                        "strict_test": strict_test_metrics,
                        "epoch": best_val["epoch"],
                        "config": ckpt_cfg},
                       ckpt_path)
            print(f"             saved -> {ckpt_path}")

    val_agg = agg_seeds(val_runs)
    test_agg = agg_seeds(test_runs)
    strict_test_agg = agg_seeds(strict_test_runs)

    print()
    print("=" * 70)
    print(f"{args.variant} FINAL ({args.seeds} seeds, {args.epochs}-epoch, "
          f"n_train={len(train_ids)}, total {(time.time()-t_global)/60:.1f}m)")
    print("=" * 70)

    def _fmt(d, key, plus=True):
        mu, sd = d.get(key, float("nan")), d.get(key + "_std", float("nan"))
        if isinstance(mu, float) and np.isnan(mu):
            return "  nan"
        return f"{mu:+.3f}±{sd:.3f}" if plus else f"{mu:.4f}±{sd:.4f}"

    print(f"{'split':<8}{'R²pk':>16}{'R²shk':>16}{'R²csc':>16}"
          f"{'MAE_csc':>14}{'R²kf̄':>16}")
    displayed_splits = [("val", val_agg), ("internal", test_agg)]
    if strict_test_runs:
        displayed_splits.append(("strict", strict_test_agg))
    for tag, agg in displayed_splits:
        print(f"{tag:<8}{_fmt(agg, 'r2_pk'):>16}{_fmt(agg, 'r2_pk_shk'):>16}"
              f"{_fmt(agg, 'r2_pk_csc'):>16}"
              f"{_fmt(agg, 'mae_pk_csc', plus=False):>14}"
              f"{_fmt(agg, 'r2_kf_csc_mean'):>16}")

    if "r2_kf_csc" in test_agg:
        print()
        print("Per-keyframe R²pk_csc on TEST split (mean ± std across seeds):")
        kdays = cfg.key_days
        means = test_agg["r2_kf_csc"]; stds = test_agg["r2_kf_csc_std"]
        for d, m, s in zip(kdays, means, stds):
            sm = "  nan" if np.isnan(m) else f"{m:+.3f}±{s:.3f}"
            print(f"  d{d:>3}  {sm}")

    out_json = out_dir / f"{args.variant.lower()}_summary.json"
    with open(out_json, "w") as fp:
        json.dump({
            "args": vars(args),
            "seed_range": seed_range,
            "val_per_seed": val_runs,
            "test_per_seed": test_runs,
            "strict_test_per_seed": strict_test_runs,
            "val_agg": val_agg,
            "test_agg": test_agg,
            "strict_test_agg": strict_test_agg,
            "histories": histories,
        }, fp, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print()
    print(f"[done] summary -> {out_json}")


if __name__ == "__main__":
    main()
