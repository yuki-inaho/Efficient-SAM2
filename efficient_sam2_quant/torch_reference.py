"""PyTorch CPU correctness references, not optimized integer GEMM kernels."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from fnmatch import fnmatchcase

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .quantization import quantize_weight, QuantizedLinear, validate_mode
from .rotation import regular_hadamard


class ConvRotLinear(nn.Module):
    """Frozen INT8 weights with online rotation and floating-point GEMM.

    ``w8a16`` means weight-only quantization; on an FP32 CPU model activations
    remain FP32 (W8A32 arithmetic). ``w8a8`` simulates dynamic per-token INT8
    activations by quantize/dequantize before FP32 GEMM. Neither is a native
    INT8 CPU acceleration path. Outputs are cast back to the input dtype.
    """

    def __init__(self, quantized: QuantizedLinear, mode: str = "w8a16"):
        super().__init__()
        self.mode = validate_mode(mode)
        self.in_features = quantized.in_features
        self.out_features = quantized.out_features
        self.group_size = quantized.group_size
        self.original_dtype = quantized.original_dtype
        self.register_buffer(
            "weight_int8", torch.from_numpy(quantized.weight_int8.copy())
        )
        self.register_buffer(
            "weight_scale", torch.from_numpy(quantized.weight_scale.copy())
        )
        self.register_buffer(
            "bias",
            None if quantized.bias is None else torch.from_numpy(quantized.bias.copy()),
        )
        self.register_buffer(
            "rotation",
            torch.from_numpy(regular_hadamard(self.group_size)),
            persistent=False,
        )

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        mode: str = "w8a16",
        *,
        max_group_size: int = 256,
        group_size: int | None = None,
    ):
        if not isinstance(module, nn.Linear):
            raise TypeError("only nn.Linear is supported")
        quantized = quantize_weight(
            module.weight.detach().float().cpu().numpy(),
            None if module.bias is None else module.bias.detach().float().cpu().numpy(),
            max_group_size=max_group_size,
            group_size=group_size,
        )
        quantized.original_dtype = str(module.weight.dtype).removeprefix("torch.")
        result = cls(quantized, mode).to(module.weight.device)
        result.train(module.training)
        return result

    def _apply(self, fn, recurse=True):
        # nn.Module.half()/to(dtype=...) must not silently round or underflow the
        # serialized FP32 scales. Move them to the requested device, preserving
        # their original FP32 values, including the zero-row floor of 1e-30.
        precise = {
            name: self._buffers[name] for name in ("weight_scale", "bias", "rotation")
        }
        result = super()._apply(fn, recurse=recurse)
        for name, original in precise.items():
            current = self._buffers[name]
            if original is not None and current.dtype != torch.float32:
                self._buffers[name] = original.to(
                    device=current.device, dtype=torch.float32
                )
        return result

    def quantized(self) -> QuantizedLinear:
        return QuantizedLinear(
            self.weight_int8.detach().cpu().numpy().copy(),
            self.weight_scale.detach().float().cpu().numpy().copy(),
            self.group_size,
            (
                None
                if self.bias is None
                else self.bias.detach().float().cpu().numpy().copy()
            ),
            self.original_dtype,
        )

    def _rotate(self, value):
        if self.group_size == 1:
            return value
        shape = (
            *value.shape[:-1],
            self.in_features // self.group_size,
            self.group_size,
        )
        return (value.reshape(shape) @ self.rotation.float().T).reshape(value.shape)

    def dequantized_weight(self):
        return self._rotate(self.weight_int8.float() * self.weight_scale.float())

    @property
    def weight(self):
        """Original-coordinate dequantized weight for read-only inspection."""
        return self.dequantized_weight()

    def forward(self, value):
        if not value.is_floating_point():
            raise TypeError("activations must be floating point")
        if value.ndim < 1 or value.shape[-1] != self.in_features:
            raise ValueError("activation feature width does not match the weight")
        dtype = value.dtype
        # Disable surrounding autocast so the documented FP32 reference remains
        # reproducible; only the returned activation is restored to caller dtype.
        with torch.autocast(device_type=value.device.type, enabled=False):
            rotated = self._rotate(value.float())
            if self.mode == "w8a8":
                if not torch.isfinite(rotated).all():
                    raise ValueError("rotated activations must be finite for W8A8")
                scale = (rotated.abs().amax(dim=-1, keepdim=True) / 127).clamp_min(
                    1e-30
                )
                quantized = (rotated / scale).round().clamp(-128, 127).to(torch.int8)
                rotated = quantized.float() * scale
            result = F.linear(
                rotated,
                self.weight_int8.float() * self.weight_scale.float(),
                None if self.bias is None else self.bias.float(),
            )
        return result.to(dtype)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, mode={self.mode}, float_reference=True"
        )


def quantize_linear_modules(
    model: nn.Module,
    mode: str = "w8a16",
    *,
    max_group_size: int = 256,
    inplace: bool = False,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> nn.Module:
    """Replace selected Linear modules; preserve Conv2d and all other modules.

    Include/exclude are case-sensitive shell globs over full module names. A
    shared Linear is replaced only if all aliases have the same selection.
    This avoids silently breaking weight/module sharing.
    """
    validate_mode(mode)
    model = model if inplace else copy.deepcopy(model)
    selected = {}
    for name, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, nn.Linear):
            choose = (
                include is None or any(fnmatchcase(name, p) for p in include)
            ) and not any(fnmatchcase(name, p) for p in exclude or ())
            if id(module) in selected and selected[id(module)] != choose:
                raise ValueError(
                    "include/exclude select shared Linear aliases inconsistently"
                )
            selected[id(module)] = choose
    replacements = {}
    for name, module in list(model.named_modules(remove_duplicate=False)):
        if isinstance(module, nn.Linear) and selected[id(module)]:
            if id(module) not in replacements:
                replacements[id(module)] = ConvRotLinear.from_linear(
                    module, mode, max_group_size=max_group_size
                )
            replacement = replacements[id(module)]
            if not name:
                model = replacement
            else:
                parent_name, _, child_name = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, replacement)
    return model
