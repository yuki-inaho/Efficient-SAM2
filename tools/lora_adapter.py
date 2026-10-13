"""Explicit image-encoder LoRA injection, gradient evidence and strict restoration."""

from contextlib import contextmanager
from types import MethodType

import torch
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.tuners_utils import BaseTunerLayer


def inject_image_lora(predictor, targets, training=True):
    trunk = predictor.image_encoder.trunk
    if any("lora_" in name for name, _ in trunk.named_parameters()):
        raise ValueError("image LoRA already injected")
    modules = dict(trunk.named_modules())
    if not targets or any(not isinstance(modules.get(t), torch.nn.Linear) for t in targets):
        raise ValueError("all explicit LoRA targets must be existing Linear modules")
    config = {"targets": list(targets), "r": 8, "alpha": 16}
    predictor.image_encoder.trunk = inject_adapter_in_model(
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", target_modules=targets),
        trunk,
    )
    initial = {}
    evidence = {"gradient_calls": 0, "nonzero_gradient_calls": 0}

    def record_gradient(grad):
        if not torch.isfinite(grad).all():
            raise FloatingPointError("nonfinite LoRA gradient")
        evidence["gradient_calls"] += 1
        evidence["nonzero_gradient_calls"] += int(torch.count_nonzero(grad).item() > 0)
        return grad

    for name, param in trunk.named_parameters():
        if "lora_" in name:
            param.requires_grad_(training)
            initial[name] = param.detach().cpu().clone()
            if training:
                param.register_hook(record_gradient)
        elif "bypass_branch" in name:
            # PEFT freezes base weights; preserve the explicitly trained bypass branch.
            param.requires_grad_(training)
    predictor._image_lora = {"config": config, "initial": initial, "evidence": evidence}
    return config


@contextmanager
def frozen_reference(trunk):
    adapters = [m for m in trunk.modules() if isinstance(m, BaseTunerLayer)]
    states = [not m.disable_adapters for m in adapters]
    try:
        for m in adapters:
            m.enable_adapters(False)
        with torch.no_grad():
            yield
    finally:
        for m, enabled in zip(adapters, states):
            m.enable_adapters(enabled)


def enable_fixed_teacher(encoder):
    def forward(self, sample):
        with frozen_reference(self.trunk):
            features, pos = self.neck(self.trunk.backbone_forward(sample))
        features_student, pos_student = self.neck(self.trunk.bypass_forward(sample))
        if self.scalp > 0:
            features, pos = features[: -self.scalp], pos[: -self.scalp]
            features_student = features_student[: -self.scalp]
            pos_student = pos_student[: -self.scalp]
        return (
            {"vision_features": features[-1], "vision_pos_enc": pos, "backbone_fpn": features},
            {
                "vision_features": features_student[-1],
                "vision_pos_enc": pos_student,
                "backbone_fpn": features_student,
            },
        )

    encoder.bypass_train_forward = MethodType(forward, encoder)


def save_image_lora(predictor, path):
    meta = predictor._image_lora
    state = {
        n: p.detach().cpu().clone()
        for n, p in predictor.image_encoder.trunk.named_parameters()
        if "lora_" in n
    }
    if not state or any(not torch.isfinite(p).all() for p in state.values()):
        raise ValueError("adapter tensors missing or nonfinite")
    changed = sum(not torch.equal(p, meta["initial"][n]) for n, p in state.items())
    info = dict(meta["evidence"], changed_tensors=changed, tensors=len(state))
    if not changed or not info["nonzero_gradient_calls"]:
        raise RuntimeError("LoRA did not receive nonzero gradients and update")
    torch.save({"config": meta["config"], "adapter_state": state, "update_evidence": info}, path)
    return info


def load_image_lora(predictor, path):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    config = blob["config"]
    if config["r"] != 8 or config["alpha"] != 16:
        raise ValueError("unsupported adapter configuration")
    inject_image_lora(predictor, config["targets"], training=False)
    params = dict(predictor.image_encoder.trunk.named_parameters())
    expected = {n for n in params if "lora_" in n}
    state = blob["adapter_state"]
    if set(state) != expected:
        raise ValueError("adapter keys differ from the injected model")
    for name, value in state.items():
        if value.shape != params[name].shape or not torch.isfinite(value).all():
            raise ValueError("invalid adapter tensor: " + name)
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)
    return blob["update_evidence"]
