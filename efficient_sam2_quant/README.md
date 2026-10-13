# ConvRot INT8 CPU reference and GGUF storage

This package implements a correctness reference for Efficient-SAM2 Linear
quantization. It does not supply accelerated CPU INT8 matrix kernels. Conv2d,
normalization, nonlinearities, embedding tables and all other model components
are left unchanged.

## Numerical contract

The rotation is the normalized regular Hadamard matrix from
[ConvRot, arXiv:2512.03673, equations 9–13](https://arxiv.org/html/2512.03673v1),
with H4 rows `(1,1,1,-1)`, `(1,1,-1,1)`, `(1,-1,1,1)`, `(-1,1,1,1)`.
Repeated Kronecker products give sizes `4^k`, normalized by `1/sqrt(size)`.
This basis is symmetric and orthogonal, so applying the same group rotation to
inputs and weights preserves an unquantized Linear operation.

The default group is the largest power of four, at most 256, dividing the input
width. Hiera widths 112/224/448/896 use groups 16/16/64/64. A width not divisible
by four uses identity (group 1). An explicitly requested non-divisor is rejected;
no feature tail is silently dropped or padded. This adaptive policy differs from
Comfy Quants' fixed group-256 policy, which skips rotation for non-divisors.

Weights are first rotated in FP32, then quantized per output channel using:

- `scale = max(max(abs(rotated_weight), axis=1, keepdim=True)/127, 1e-30)`
- `q = round_to_even(rotated_weight/scale).clip(-128,127).astype(int8)`
- Stored scales have shape `[out_features,1]` and dtype FP32
- Biases are kept as FP32, without quantization

This matches the numerical convention documented in the
[Comfy Quants INT8 W8A8 format](https://github.com/Comfy-Org/comfy-quants/blob/main/docs/formats/int8_w8a8.md).
Floating-point rotation results can differ slightly between NumPy, PyTorch, BLAS
implementations and devices. No cross-backend bit-identical quantization claim is
made. Serialization of the already-quantized arrays is bit-exact.

Two separately labeled reference modes are available:

- `w8a16`: weight-only quantization; activation input/output dtype follows the
  caller. When the source model and inputs are FP32 on CPU, this is W8A32
  arithmetic, not a claim that activations became FP16. Internal reference
  rotation and matrix multiplication use FP32.
- `w8a8`: adds dynamic symmetric per-token INT8 quantization after input
  rotation, with a scale over the last feature axis. The Torch implementation
  dequantizes these activations and uses FP32 `linear`. The NumPy oracle uses
  INT64 accumulation of integer products to avoid overflow in unusual widths.
  Neither implementation demonstrates native INT8 CPU acceleration.

Calling `.half()` or `.to(dtype=...)` on the reference module preserves its FP32
scales, bias and rotation buffers; it never underflows zero-row scales to zero.
The wrapper is frozen inference code, not a quantization-aware training module.

## Python API

```python
from efficient_sam2_quant import (
    quantize_linear_modules, export_gguf, load_gguf, load_gguf_model,
)

reference = quantize_linear_modules(
    model.eval(), mode="w8a8", max_group_size=256, inplace=False,
    # Optional shell-glob filters over full module names:
    # include=["image_encoder.*"], exclude=["*.sensitive_projection"],
)
report = export_gguf(
    reference, "/path/outside/repository/model.gguf",
    model_config={
        "variant": "base_plus", "sam2_version": "2.1", "image_size": 1024,
        # Also provide exact Efficient-SAM2 SWR/SMR/bypass policy here.
    },
    source_sha256="...64-character lowercase checkpoint SHA-256...",
    source_metadata={"description": "public provenance only"},
)

# W8A8 reference reload, without a second quantization:
restored_reference = load_gguf_model(fresh_stock_model, report["path"], mode="w8a8")

# Fast stock-PyTorch weight-only accuracy path, without online rotation:
artifact = load_gguf(report["path"])
fresh_stock_model.load_state_dict(artifact.state_dict(), strict=True)
```

The last path dequantizes the rotated weight and applies the inverse rotation
once, recovering an original-coordinate, lossy-INT8-derived FP32 weight. It does
not reconstruct the original pre-quantization weight. This is mathematically the
weight-only reference path, with small floating-point evaluation differences
possible from operation order. It uses ordinary floating-point model execution.
Use the Torch reference modules to evaluate dynamic W8A8 activation error.

`export_gguf` writes atomically, refuses to overwrite by default, checks its
output using the standard `gguf.GGUFReader`, and verifies every tensor payload
against the source arrays before publishing. Checkpoints, images and GGUF
artifacts should remain outside the source repository. Private paths and inputs
are not automatically written into exported metadata.

## GGUF schema version 1

This is a standard little-endian GGUF v3 container, using standard I8/F32 tensor
types. It is **not** GGML Q8_0, and a generic GGUF reader does not provide model
execution. An architecture-aware loader must interpret the following schema.

- `general.architecture = "efficientsam2"`
- `general.alignment = 32`
- `general.quantization_version = 1`
- `efficient_sam2.storage_version = 1`
- `efficient_sam2.convrot_basis = "regular-h4-kron"`
- `efficient_sam2.model_config`: JSON object describing architecture/variant and
  the exact model/inference policy
- `efficient_sam2.source.sha256`: lowercase SHA-256; `.source.hash_kind` identifies
  `checkpoint-file` or a hash of the exported logical tensor stream
- `efficient_sam2.source.metadata`: caller-supplied JSON object
- `efficient_sam2.test_fixture`: boolean; synthetic files must not be mistaken
  for trained models
- `efficient_sam2.quantization`: JSON object with a `scheme` and `layers` map;
  every layer records `in_features`, `out_features`, `group_size`, `has_bias`,
  `source_dtype`, and `activation_mode`
- `efficient_sam2.int8_layers` and `.convrot_layers`: validated counts

Each quantized Linear `<layer>` contributes logical tensors:

| Name | Logical dtype | Shape |
| --- | --- | --- |
| `<layer>.weight` | I8 | `[out,in]`, rotated coordinates |
| `<layer>.weight_scale` | F32 | `[out,1]` |
| `<layer>.comfy_quant` | U8 | UTF-8 JSON bytes |
| `<layer>.bias` (if present) | F32 | `[out]` |

The Comfy marker is
`{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":G,"per_row":true}`.
With identity rotation it has `convrot:false` and omits `convrot_groupsize`.
The root module uses `weight`, `weight_scale`, `comfy_quant`, and `bias` without a
leading dot. Quantization covers Linear weights only.

The physical tensor name is `t_` plus the first 60 hexadecimal characters of
SHA-256 of the UTF-8 logical tensor name. This fits ggml's tensor-name buffer.
`efficient_sam2.tensor_map` is a JSON object keyed by physical name, with values
`{"name":logical_name,"dtype":logical_dtype,"shape":[logical_dimensions]}`.
Logical tensors of rank 1–4 keep their shape; scalars and rank >4 tensors are
flattened physically, with original shape preserved in the map. GGUF dimension
order is reversed relative to NumPy/PyTorch. All tensors retain C-order bytes.
U8/BOOL use standard I8 physical storage with their logical interpretation in the
map; BF16 uses standard BF16. Tensor-map and marker conventions intentionally
match the generic storage design in sam3.cpp's SAM 3.1 adapter; the architecture
and namespace remain distinct.

The loader validates architecture, version, basis, tensor names/types/shapes,
alignment, payload boundaries, positive finite scales, bias presence, marker
agreement, quantized layer counts and strict full-model state-dict compatibility.
No pickle is used to read GGUF. Shared Linear module aliases are preserved.

Native sam3.cpp integration must separately validate architecture/config support,
restore or execute these weights correctly, and pass the relevant accuracy gate.
Container readability alone proves neither Efficient-SAM2 SWR/SMR execution nor
full-graph INT8 CPU acceleration.

## Tests

```sh
python -m pytest -q tests/test_convrot_reference.py tests/test_convrot_gguf.py
```

Tests cover the exact regular basis, orthogonality, rotation invariance, adaptive
Hiera groups, zero rows/tokens, feature tails, empty/non-contiguous inputs,
rounding, quantization error bounds, Torch-versus-integer-oracle agreement,
selection/alias behavior, FP32 scale preservation, standard GGUFReader inspection,
exact payload/reload parity, residual dtypes/shapes, stock-model inverse rotation,
no-clobber writes, malformed metadata, invalid scales and truncated input.
