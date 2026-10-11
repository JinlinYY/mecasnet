"""Recovery-rate, event-context and late-stage trajectory diagnostics."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch

from comment12_common import (
    CONTEXT_FIELDS, EPSILONS, KEY_DAYS, PROTOCOL, atomic_json, bind_manifest,
    build_model, data_manifest, event_statistics, late_indices, load_runtime,
    make_config, make_splits, node_masks, sha256, source_manifest, summarize_events,
)


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def context_from_batch(batch):
    count = int(batch["Nr"])
    if count < 1:
        raise ValueError("Empty reach subgraph.")
    destination = batch["edge_dst"]
    degrees = torch.zeros(count, dtype=batch["x_v"].dtype, device=destination.device)
    degrees.index_add_(0, destination, torch.ones_like(destination, dtype=degrees.dtype))
    log_degrees = torch.log1p(degrees)
    return torch.tensor([
        math.log(max(destination.numel(), 1) / count + 1e-6),
        float(batch["shock_mask"].mean()), float(log_degrees.mean()),
        float(log_degrees.std(unbiased=False)), float(log_degrees.max() - log_degrees.mean()),
    ], dtype=degrees.dtype, device=destination.device)


def gradient_report(loss, tensors):
    tensors = list(tensors)
    if not loss.requires_grad:
        return {"loss_requires_grad": False, "connected_tensors": 0,
                "nonzero_tensors": 0, "norm": None, "max_abs": None}
    eligible = [value for value in tensors if value.requires_grad]
    if not eligible:
        return {"loss_requires_grad": True, "connected_tensors": 0,
                "nonzero_tensors": 0, "norm": None, "max_abs": None}
    gradients = torch.autograd.grad(loss, eligible, allow_unused=True, retain_graph=True)
    connected = [value for value in gradients if value is not None]
    return {
        "loss_requires_grad": True, "connected_tensors": len(connected),
        "nonzero_tensors": sum(int(bool(torch.any(value != 0))) for value in connected),
        "norm": math.sqrt(sum(float(value.detach().double().square().sum()) for value in connected)) if connected else None,
        "max_abs": max(float(value.detach().abs().max()) for value in connected) if connected else None,
    }


def quantiles(values):
    values = np.asarray(values, dtype=float).ravel()
    return dict(zip(("min", "p05", "median", "p95", "max"),
                    np.percentile(values, [0, 5, 50, 95, 100]).tolist()))


def rate_diagnostics(output, batch, cfg, losses):
    health = output["u_full"]
    tau = output["params"][2]
    days = batch["key_days"].to(health)
    gaps = days[1:] - days[:-1]
    if bool(torch.any(gaps <= 0)):
        raise ValueError("Nonpositive keyframe gap.")
    slopes = (health[1:] - health[:-1]) / gaps[:, None]
    cap = 1 / tau.clamp_min(0.5)
    excess = torch.relu(slopes - cap[None, :])
    expected_loss = cfg.w_mono * excess.square().mean()
    actual_loss = losses.loss_mono(health, output["params"], cfg, batch["key_days"])
    if not torch.allclose(actual_loss, expected_loss, atol=1e-12, rtol=1e-5):
        raise RuntimeError("Audited formula differs from imported loss implementation.")
    return {
        "intervals": excess.numel(), "violations": int((excess > 0).sum()),
        "violations_over_1e_minus8_per_day": int((excess > 1e-8).sum()),
        "positive_slope_intervals": int((slopes > 0).sum()),
        "weighted_loss": float(actual_loss.detach()),
        "unweighted_loss": float(excess.square().mean().detach()),
        "max_excess_per_day": float(excess.max().detach()),
        "tau_days": quantiles(tau.detach().cpu().numpy()),
        "cap_per_day": quantiles(cap.detach().cpu().numpy()),
        "margin_cap_minus_slope": quantiles((cap[None, :] - slopes).detach().cpu().numpy()),
        "health_min": float(health.min().detach()), "health_max": float(health.max().detach()),
        "minimum_gap_days": float(gaps.min()),
        "structural_inactivity_conditions_observed": bool(
            health.min() >= 0 and health.max() <= 1 and tau.min() >= 0.5
            and tau.max() <= 5 and gaps.min() >= 5),
        "u_full_is_keyframe_alias": bool(torch.equal(health, output["u_keyframes"])),
    }


def intervene(model, batch, replacement):
    def replace_context(module, inputs):
        return inputs[0], replacement.to(inputs[1])
    handle = model.phys_param_gen.register_forward_pre_hook(replace_context)
    try:
        with torch.no_grad():
            return model(batch)
    finally:
        handle.remove()


def context_effects(model, batch, reference, mean_context, donor):
    result = {}
    for name, replacement in {"zero": torch.zeros_like(mean_context),
                              "training_mean": mean_context, "other_training_event": donor}.items():
        output = intervene(model, batch, replacement)
        result[name] = {
            "health_max_abs_change": float((output["u_keyframes"] - reference["u_keyframes"]).abs().max()),
            "peak_max_abs_change": float((output["peak"] - reference["peak"]).abs().max()),
            "tau_max_abs_change": float((output["params"][2] - reference["params"][2]).abs().max()),
            "c_max_abs_change": float((output["params"][3] - reference["params"][3]).abs().max()),
        }
    return result


def gradient_diagnostics(model, batch, cfg, losses, update_control=False):
    captured = {}
    def differentiable_context(module, inputs):
        context = inputs[1].detach().clone().requires_grad_(True)
        captured["context"] = context
        return inputs[0], context
    handle = model.phys_param_gen.register_forward_pre_hook(differentiable_context)
    try:
        output = model(batch)
    finally:
        handle.remove()
    parts = losses.total_loss(output, batch, cfg)
    parts["data"] = parts["l_kf"] + parts["l_peak"]
    named = list(model.named_parameters())
    shared = [parameter for name, parameter in named if not name.startswith("phys_param_gen.")]
    head = list(model.phys_param_gen.parameters())
    result = {}
    for name in ("l_mono", "l_mono_late", "data", "total"):
        result[name] = {
            "value": float(parts[name].detach()),
            "trajectory": gradient_report(parts[name], [output["u_full"]]),
            "tau": gradient_report(parts[name], [output["params"][2]]),
            "context": gradient_report(parts[name], [captured["context"]]),
            "shared_parameters": gradient_report(parts[name], shared),
            "auxiliary_parameters": gradient_report(parts[name], head),
        }
    disabled = copy.deepcopy(cfg)
    disabled.w_mono = 0.0
    off_loss = losses.total_loss(output, batch, disabled)["total"]
    parameters = [parameter for _, parameter in named]
    full_grad = torch.autograd.grad(parts["total"], parameters, allow_unused=True, retain_graph=True)
    off_grad = torch.autograd.grad(off_loss, parameters, allow_unused=True)
    differences = []
    for parameter, full, off in zip(parameters, full_grad, off_grad):
        full_value = torch.zeros_like(parameter) if full is None else full
        off_value = torch.zeros_like(parameter) if off is None else off
        differences.append(float((full_value - off_value).abs().max()))
    control = {
        "eval_mode": True, "full_minus_off_loss": float((parts["total"] - off_loss).detach()),
        "all_gradient_max_abs_difference_treating_none_as_zero": max(differences),
        "gradient_connection_changed_parameters": [name for (name, _), full, off in zip(named, full_grad, off_grad)
                                                   if (full is None) != (off is None)],
    }
    if update_control:
        models = [copy.deepcopy(model), copy.deepcopy(model)]
        for copied, gradients in zip(models, (full_grad, off_grad)):
            optimizer = torch.optim.AdamW(copied.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            for parameter, gradient in zip(copied.parameters(), gradients):
                parameter.grad = None if gradient is None else gradient.detach().clone()
            torch.nn.utils.clip_grad_norm_(copied.parameters(), cfg.grad_clip)
            optimizer.step()
        parameter_delta = {}
        for (name, first), (_, second) in zip(models[0].named_parameters(), models[1].named_parameters()):
            parameter_delta[name] = float((first - second).detach().abs().max())
        with torch.no_grad():
            outputs = [copied(batch) for copied in models]
        control.update(
            one_fresh_adamw_step_all_parameter_max_abs_difference=max(parameter_delta.values()),
            one_fresh_adamw_step_predictive_parameter_max_abs_difference=max(
                value for name, value in parameter_delta.items() if not name.startswith("phys_param_gen.")),
            one_fresh_adamw_step_health_max_abs_difference=float(
                (outputs[0]["u_keyframes"] - outputs[1]["u_keyframes"]).abs().max()),
            one_fresh_adamw_step_peak_max_abs_difference=float((outputs[0]["peak"] - outputs[1]["peak"]).abs().max()),
        )
    result["rate_disabled_control"] = control
    return result




def context_summary(vectors, tolerance=1e-6):
    values = np.asarray(vectors, dtype=float)
    return {name: {"std": float(values[:, index].std()),
                   "min": float(values[:, index].min()), "max": float(values[:, index].max()),
                   "unique_exact": int(np.unique(values[:, index]).size),
                   "unique_rounded_tolerance": int(np.unique(np.rint(values[:, index] / tolerance)).size)}
            for index, name in enumerate(CONTEXT_FIELDS)}




