"""Matched architecture ablations for the Henriet chemical benchmark."""
from __future__ import annotations

import json
import hashlib
import os
import sys
from pathlib import Path

BASE_ENV = {
    "GNS_TRIPLE_BLEND": "1", "GNS_N_PROC": "4", "GNS_BLEND_ROLLOUT": "0",
    "GNS_BLEND_INIT": "2,0,0", "GNS_NOBM": "0", "GNS_LEARN_TAU": "0",
    "GNS_FREE_RESIDUAL": "0", "GNS_DECLINE_P": "6.0", "GNS_RECOVERY_P": "1.0",
    "GNS_RECOVERY_GATE": "0", "GNS_PHYSRECOVER": "0", "GNS_RECOVERY_Q": "1",
    "GNS_RECOVERY_QMIN": "0.5", "GNS_RECOVERY_QMAX": "2.0",
    "GNS_ABLATE_UNCOND": "0", "GNS_ABLATE_FORWARD_ONLY": "0", "GNS_NO_REACH": "0",
}

PROFILES = [
    ("Full_MeCaSNet", "Full MeCaSNet", "None", {}),
    ("Decoder_only_output", "Decoder-only output", "Rollout/direct fusion branches", {"GNS_TRIPLE_BLEND": "0"}),
    ("Direct_rollout_output", "Direct rollout output", "Structured trajectory decoder", {"GNS_TRIPLE_BLEND": "0", "GNS_NOBM": "1"}),
    ("Regression_only_training", "Regression-only training", "Cascade-boundary auxiliary loss", {"GNS_NO_REACH": "1"}),
    ("Unconditioned_propagation", "Unconditioned propagation", "Day-0 shock-conditioned encoder inputs", {"GNS_ABLATE_UNCOND": "1"}),
    ("Forward_only_propagation", "Forward-only propagation", "Reverse demand-feedback messages in all spatial/rollout blocks", {"GNS_ABLATE_FORWARD_ONLY": "1"}),
    ("Gaussian_decline_front", "Gaussian decline front", "Flat-front decline exponent (6 to 2)", {"GNS_DECLINE_P": "2.0"}),
    ("Single_exponential_recovery", "Single-exponential recovery", "Node-specific recovery exponent", {"GNS_RECOVERY_Q": "0"}),
    ("Plain_GAT", "Plain GAT", "Complete supply-relation-state propagation backbone", {}),
]


def experiment_config():
    models = []
    for label, display_name, component, changes in PROFILES:
        environment = {**BASE_ENV, **changes, "PAPER_ABLATION_KIND": label}
        models.append({
            "label": label, "display_name": display_name, "component": component,
            "variant": "GAT" if label == "Plain_GAT" else "V3DE_G3_GNS",
            "env": environment, "train_args": [],
        })
    return {"protocol": "paper_table6_paired_ablation_v1", "models": models}


def spatial_blocks(model):
    blocks = []
    for name in ("prewarm_gat", "param_gat"):
        blocks.extend((f"{name}.{index}", block) for index, block in enumerate(getattr(model, name, [])))
    for name in ("diffusion_gat", "collapse_block", "recovery_block"):
        block = getattr(model, name, None)
        if block is not None:
            blocks.append((name, block))
    block = getattr(getattr(model, "step", None), "spatial", None)
    if block is not None:
        blocks.append(("step.spatial", block))
    unique = {}
    for name, block in blocks:
        unique.setdefault(id(block), (name, block))
    return list(unique.values())


def mask_reverse(block, inputs, outputs):
    import torch

    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise RuntimeError("Expected a forward/reverse spatial message pair.")
    forward, reverse = outputs
    return forward, torch.zeros_like(reverse)


def install_model_adapter(kind):
    import pi_dl.model_v3_decoder_ac as architecture

    original = architecture.KSGATv3GNSBackbone

    class PaperAlignedBackbone(original):
        def __init__(self, cfg, Fv, **kwargs):
            kwargs["skip_residual_head"] = True
            super().__init__(cfg, Fv=Fv, **kwargs)
            self.paper_ablation_kind = kind
            self.paper_forward_only_blocks = []
            if kind == "Forward_only_propagation":
                for name, block in spatial_blocks(self):
                    block.register_forward_hook(mask_reverse)
                    self.paper_forward_only_blocks.append(name)
                if not self.paper_forward_only_blocks:
                    raise RuntimeError("No spatial blocks found for forward-only removal.")
            if kind == "Decoder_only_output" and (self.blend_rollout or self.triple_blend):
                raise RuntimeError("Decoder-only output still has fusion branches.")
            if kind == "Direct_rollout_output" and self.decoder_mode != "rollout":
                raise RuntimeError("Rollout-only output still has an analytical decoder.")
            print("[paper-ablation] model={} params={} reverse_masks={} late_residual=false".format(
                kind, sum(parameter.numel() for parameter in self.parameters()),
                self.paper_forward_only_blocks), flush=True)

    architecture.KSGATv3GNSBackbone = PaperAlignedBackbone
    return original


def main():
    kind = os.environ.get("PAPER_ABLATION_KIND")
    expected = {model["label"]: model for model in experiment_config()["models"]}
    if kind not in expected:
        raise ValueError("PAPER_ABLATION_KIND must select an explicit Table 6 row.")
    profile = expected[kind]
    for key, value in profile["env"].items():
        if os.environ.get(key) != value:
            raise ValueError("Unexpected profile setting {} for {}".format(key, kind))
    if "--require_y8_exact" in sys.argv or "--resume_from" in sys.argv:
        raise ValueError("Paper ablations require fresh training without the historical exact-profile guard.")
    variant_position = sys.argv.index("--variant")
    if sys.argv[variant_position + 1] != profile["variant"]:
        raise ValueError("Trainer variant does not match the paper row.")
    install_model_adapter(kind)
    from pi_dl.train_v3de import main as train

    train()
    output_position = sys.argv.index("--out_dir")
    output = Path(sys.argv[output_position + 1]) / "paper_profile.json"
    output.write_text(json.dumps({
        "protocol": "paper_table6_paired_ablation_v1", "profile": profile,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "unreported_late_residual_head": False,
        "forward_only_scope": "all spatial and rollout blocks" if kind == "Forward_only_propagation" else None,
        "unconditioned_scope": "encoder shock identity/intensity only; physical Day-0 initial condition retained",
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()