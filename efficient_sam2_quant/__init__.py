"""ConvRot INT8 reference and versioned GGUF storage for Efficient-SAM2."""

from .quantization import quantize_per_token, quantize_weight, QuantizedLinear
from .rotation import choose_group_size, regular_hadamard, rotate_numpy

__all__ = [
    "choose_group_size",
    "regular_hadamard",
    "rotate_numpy",
    "QuantizedLinear",
    "quantize_per_token",
    "quantize_weight",
    "ConvRotLinear",
    "quantize_linear_modules",
    "export_gguf",
    "load_gguf",
    "load_gguf_model",
]


def __getattr__(name):
    if name in {"ConvRotLinear", "quantize_linear_modules"}:
        from . import torch_reference

        return getattr(torch_reference, name)
    if name in {"export_gguf", "load_gguf", "load_gguf_model"}:
        from . import gguf_io

        return getattr(gguf_io, name)
    raise AttributeError(name)
