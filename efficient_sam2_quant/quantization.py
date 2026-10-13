"""Portable numerical reference for per-output-channel ConvRot INT8.

Numeric convention follows Comfy Quants INT8 W8A8: FP32 rotation, scale
max(abs(row))/127 clamped to 1e-30, ties-to-even rounding, signed INT8. Unlike
its fixed group-256 policy, adaptive groups retain rotation for Hiera widths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotation import choose_group_size, rotate_numpy, validate_group_size

MODES = ("w8a16", "w8a8")


def validate_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return mode


def _float32_finite(value, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype.kind != "f":
        raise TypeError(f"{name} must be floating point")
    value = value.astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite and representable as float32")
    return value


def quantize_per_token(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic INT8 activation quantization along the last (feature) axis."""
    value = _float32_finite(value, "activations")
    if value.ndim < 1 or value.shape[-1] < 1:
        raise ValueError("activations require a nonempty feature axis")
    scale = np.maximum(
        np.max(np.abs(value), axis=-1, keepdims=True) / np.float32(127),
        np.float32(1e-30),
    )
    quantized = np.clip(np.rint(value / scale), -128, 127).astype(np.int8)
    return quantized, scale


@dataclass
class QuantizedLinear:
    """Owned CPU arrays. The INT8 weight is stored in rotated coordinates."""

    weight_int8: np.ndarray
    weight_scale: np.ndarray
    group_size: int
    bias: np.ndarray | None = None
    original_dtype: str = "float32"

    def __post_init__(self):
        validate_group_size(self.group_size)
        self.weight_int8 = np.asarray(self.weight_int8)
        self.weight_scale = np.asarray(self.weight_scale)
        if (
            self.weight_int8.dtype != np.int8
            or self.weight_int8.ndim != 2
            or min(self.weight_int8.shape) < 1
        ):
            raise ValueError(
                "weight_int8 must have dtype int8 and positive [out, in] shape"
            )
        if self.weight_int8.shape[1] % self.group_size:
            raise ValueError("group_size must divide in_features")
        if self.weight_scale.dtype != np.float32 or self.weight_scale.shape != (
            self.out_features,
            1,
        ):
            raise ValueError("weight_scale must have dtype float32 and shape [out, 1]")
        if (
            not np.isfinite(self.weight_scale).all()
            or not (self.weight_scale > 0).all()
        ):
            raise ValueError("weight_scale must be finite and strictly positive")
        if self.bias is not None:
            self.bias = _float32_finite(self.bias, "bias")
            if self.bias.shape != (self.out_features,):
                raise ValueError("bias must have shape [out]")

    @property
    def in_features(self) -> int:
        return self.weight_int8.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight_int8.shape[0]

    def rotated_weight(self) -> np.ndarray:
        return self.weight_int8.astype(np.float32) * self.weight_scale

    def dequantized_weight(self) -> np.ndarray:
        """Undo rotation for stock PyTorch Linear; this is still lossy INT8."""
        return rotate_numpy(self.rotated_weight(), self.group_size)

    def marker(self) -> dict:
        result = {"format": "int8_tensorwise", "convrot": self.group_size > 1}
        if self.group_size > 1:
            result["convrot_groupsize"] = self.group_size
        result["per_row"] = True
        return result

    def forward(self, value: np.ndarray, mode: str = "w8a16") -> np.ndarray:
        """FP32 correctness reference, never a claim of CPU acceleration.

        w8a16 is weight-only: activation dtype follows the caller in the Torch
        wrapper and is FP32 here. w8a8 additionally dynamically quantizes each
        rotated token. Its NumPy reference accumulates exact integer products
        in INT64 to avoid overflow for unusual widths, then rescales to FP32.
        """
        validate_mode(mode)
        value = _float32_finite(value, "activations")
        if value.ndim < 1 or value.shape[-1] != self.in_features:
            raise ValueError("activation feature width does not match the weight")
        rotated = rotate_numpy(value, self.group_size)
        if mode == "w8a8":
            quantized, scale = quantize_per_token(rotated)
            accumulated = quantized.astype(np.int64) @ self.weight_int8.T.astype(
                np.int64
            )
            result = accumulated.astype(np.float32) * scale * self.weight_scale.T
        else:
            result = rotated @ self.rotated_weight().T
        if self.bias is not None:
            result = result + self.bias
        return result


def quantize_weight(
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    max_group_size: int = 256,
    group_size: int | None = None,
) -> QuantizedLinear:
    original_dtype = str(np.asarray(weight).dtype)
    weight = _float32_finite(weight, "weight")
    if weight.ndim != 2 or min(weight.shape) < 1:
        raise ValueError("weight must have positive [out, in] shape")
    group = (
        choose_group_size(weight.shape[1], max_group_size)
        if group_size is None
        else group_size
    )
    rotated = rotate_numpy(weight, group)
    if not np.isfinite(rotated).all():
        raise ValueError("rotation overflowed float32")
    quantized, scale = quantize_per_token(rotated)
    return QuantizedLinear(quantized, scale, group, bias, original_dtype)
