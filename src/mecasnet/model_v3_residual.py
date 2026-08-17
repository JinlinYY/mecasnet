"""KSGATv3 with a late-keyframe residual correction head."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .model_v3 import KSGATv3


class KSGATv3Residual(KSGATv3):
    def __init__(self, cfg, Fv: int, **kwargs):
        self.skip_residual_head = bool(kwargs.pop("skip_residual_head", False))
        super().__init__(cfg, Fv=Fv, **kwargs)
        if self.skip_residual_head:
            return

        d_hidden = int(cfg.d_hidden)
        n_keyframes = len(cfg.key_days)
        if n_keyframes != 10:
            raise ValueError(
                f"KSGATv3Residual requires 10 keyframes, got {n_keyframes}"
            )

        self.late_residual_head = nn.Sequential(
            nn.Linear(d_hidden, 2 * d_hidden),
            nn.GELU(),
            nn.Linear(2 * d_hidden, n_keyframes),
        )
        nn.init.zeros_(self.late_residual_head[-1].weight)
        nn.init.zeros_(self.late_residual_head[-1].bias)
        self.register_buffer(
            "alpha_kf_residual",
            torch.tensor([0.0, 0.0, 0.40, 0.65, 0.90, 0.95, 0.97, 0.98, 0.99, 0.99]),
        )

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        out = super().forward(batch)
        if self.skip_residual_head:
            return out

        h_final = out["h_final"]
        u_keyframes = out["u_keyframes"]
        residual = self.late_residual_head(h_final).t()
        alpha = self.alpha_kf_residual.to(u_keyframes.dtype).view(-1, 1)
        u_keyframes = (u_keyframes + alpha * residual).clamp(0.0, 1.0)

        out["u_keyframes"] = u_keyframes
        out["u_full"] = u_keyframes

        if self.decoder_mode in ("param", "param2", "param3"):
            shock_mask = batch["shock_mask"].to(h_final.dtype)
            delta0 = batch["delta0"].to(h_final.dtype)
            peak_param = (1.0 - u_keyframes.min(dim=0).values).clamp(0.0, 1.0)
            peak = torch.maximum(peak_param, delta0 * shock_mask).clamp(0.0, 1.0)
            out["peak"] = peak
            out["peak_pred"] = peak
            out["peak_phys"] = peak
        return out
