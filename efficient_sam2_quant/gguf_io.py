"""Standard GGUF v3 container with a versioned Efficient-SAM2 storage schema.

This is I8 + per-row F32 scale storage, NOT GGML Q8_0 and NOT a declaration of
native GGML INT8 graph support. The tensor map/Comfy markers intentionally use
the same layout as sam3.cpp's sam31 storage adapter, under a distinct architecture
and namespace. Every non-Linear tensor is preserved, including Conv2d weights.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import struct
import tempfile
from collections.abc import Mapping

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .quantization import QuantizedLinear, validate_mode

ARCHITECTURE = "efficientsam2"
PREFIX = "efficient_sam2."
STORAGE_VERSION = 1
# Logical dtype -> (NumPy representation, standard ggml tensor type).
DTYPES = {
    "F32": ("<f4", 0),
    "F16": ("<f2", 1),
    "I8": ("i1", 24),
    "U8": ("u1", 24),
    "BOOL": ("?", 24),
    "I16": ("<i2", 25),
    "I32": ("<i4", 26),
    "I64": ("<i8", 27),
    "F64": ("<f8", 28),
    "BF16": ("<u2", 30),
}
TORCH_DTYPES = {
    "float32": "F32",
    "float16": "F16",
    "float64": "F64",
    "bfloat16": "BF16",
    "int8": "I8",
    "uint8": "U8",
    "bool": "BOOL",
    "int16": "I16",
    "int32": "I32",
    "int64": "I64",
}


def _json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _decode(text: str):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)


def _key(module: str, suffix: str) -> str:
    return f"{module}.{suffix}" if module else suffix


def stored_name(name: str) -> str:
    return "t_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:60]


def storage_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    return shape if 1 <= len(shape) <= 4 else (math.prod(shape),)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numpy_tensor(tensor):
    import torch

    tensor = tensor.detach().cpu().contiguous()
    logical = TORCH_DTYPES.get(str(tensor.dtype).removeprefix("torch."))
    if logical is None:
        raise ValueError(f"unsupported tensor dtype {tensor.dtype}")
    array = tensor.view(torch.uint16).numpy() if logical == "BF16" else tensor.numpy()
    # Standard container is explicitly little-endian.
    return array.astype(np.dtype(DTYPES[logical][0]), copy=False), logical


def _validate_shape(raw) -> tuple[int, ...]:
    if not isinstance(raw, list) or len(raw) > 16:
        raise ValueError("logical shape must be a list with at most 16 axes")
    if any(type(n) is not int or n < 1 for n in raw) or math.prod(raw) > (1 << 60):
        raise ValueError("invalid or empty logical tensor shape")
    return tuple(raw)


@dataclass
class GGUFArtifact:
    """Validated, owned logical tensors; no memory-map or pickle dependency."""

    tensors: dict[str, np.ndarray]
    dtypes: dict[str, str]
    quantized: dict[str, QuantizedLinear]
    metadata: dict[str, Any]
    path: Path

    def numpy_state_dict(self) -> dict[str, np.ndarray]:
        """Original-coordinate quantized weights; BF16 residuals remain raw U16."""
        quant_keys = {
            _key(name, suffix)
            for name in self.quantized
            for suffix in ("weight", "weight_scale", "comfy_quant")
        }
        result = {
            name: value.copy()
            for name, value in self.tensors.items()
            if name not in quant_keys
        }
        for name, quantized in self.quantized.items():
            result[_key(name, "weight")] = quantized.dequantized_weight()
        return result

    def state_dict(self):
        """Stock PyTorch state dict, restoring BF16 residuals and inverse rotation."""
        import torch

        result = {}
        for name, value in self.numpy_state_dict().items():
            tensor = torch.from_numpy(value)
            if self.dtypes.get(name) == "BF16":
                tensor = tensor.view(torch.bfloat16)
            result[name] = tensor
        return result


def export_gguf(
    model,
    path: str | Path,
    *,
    model_config: Mapping | str | None = None,
    source_sha256: str | None = None,
    source_metadata: Mapping | None = None,
    overwrite: bool = False,
    test_fixture: bool = False,
) -> dict[str, Any]:
    """Export an already-quantized full model atomically and verify every payload.

    ``model_config`` should identify variant, image_size, sam2_version and the
    enabled Efficient-SAM2 policy. No checkpoint path or private inputs are
    automatically copied into metadata. Pass only metadata intended for sharing.
    """
    import gguf

    from .torch_reference import ConvRotLinear

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    modules = {
        name: module
        for name, module in model.named_modules(remove_duplicate=False)
        if isinstance(module, ConvRotLinear)
    }
    if not modules:
        raise ValueError(
            "model contains no ConvRotLinear modules; quantize Linear modules first"
        )
    skip = {
        _key(name, suffix)
        for name in modules
        for suffix in ("weight_int8", "weight_scale", "bias")
    }
    arrays, dtypes = {}, {}
    for name, tensor in model.state_dict().items():
        if name not in skip:
            arrays[name], dtypes[name] = _numpy_tensor(tensor)
    layers = {}
    for name, module in modules.items():
        record = module.quantized()
        components = {
            "weight": (record.weight_int8, "I8"),
            "weight_scale": (record.weight_scale, "F32"),
            "comfy_quant": (
                np.frombuffer(_json(record.marker()).encode("utf-8"), dtype=np.uint8),
                "U8",
            ),
        }
        if record.bias is not None:
            components["bias"] = (record.bias, "F32")
        for suffix, (value, dtype) in components.items():
            key = _key(name, suffix)
            if key in arrays:
                raise ValueError(f"quantization side-tensor name collision: {key}")
            arrays[key], dtypes[key] = value, dtype
        layers[name] = {
            "in_features": record.in_features,
            "out_features": record.out_features,
            "group_size": record.group_size,
            "has_bias": record.bias is not None,
            "source_dtype": record.original_dtype,
            "activation_mode": module.mode,
        }
    mapping = {}
    for name in sorted(arrays):
        if not name or "\0" in name or len(name.encode("utf-8")) > 4096:
            raise ValueError("invalid logical tensor name")
        _validate_shape(list(arrays[name].shape))
        physical = stored_name(name)
        if physical in mapping:
            raise ValueError("hashed tensor-name collision")
        mapping[physical] = {
            "name": name,
            "dtype": dtypes[name],
            "shape": list(arrays[name].shape),
        }
    config = (
        {"config": model_config}
        if isinstance(model_config, str)
        else dict(model_config or {})
    )
    logical_hash = hashlib.sha256()
    for name in sorted(arrays):
        logical_hash.update(
            _json([name, dtypes[name], list(arrays[name].shape)]).encode("utf-8")
        )
        logical_hash.update(arrays[name].tobytes())
    if source_sha256 is not None and (
        len(source_sha256) != 64
        or any(c not in "0123456789abcdef" for c in source_sha256)
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    metadata = {
        "general.architecture": ARCHITECTURE,
        "general.name": "Efficient-SAM2 ConvRot INT8 reference",
        "general.alignment": 32,
        "general.quantization_version": 1,
        PREFIX + "storage_version": STORAGE_VERSION,
        PREFIX + "storage": "typed-tensors-convrot-int8",
        PREFIX + "convrot_basis": "regular-h4-kron",
        PREFIX
        + "quantization": _json(
            {"scheme": "symmetric-per-output-channel-int8", "layers": layers}
        ),
        PREFIX + "tensor_map": _json(mapping),
        PREFIX + "model_config": _json(config),
        PREFIX + "source.metadata": _json(dict(source_metadata or {})),
        PREFIX + "source.sha256": source_sha256 or logical_hash.hexdigest(),
        PREFIX
        + "source.hash_kind": "checkpoint-file" if source_sha256 else "logical-tensors",
        PREFIX + "test_fixture": bool(test_fixture),
        PREFIX + "int8_layers": len(modules),
        PREFIX + "convrot_layers": sum(m.group_size > 1 for m in modules.values()),
        PREFIX + "runtime": "requires-schema-aware-loader; no-native-int8-graph-claim",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    writer = gguf.GGUFWriter(temporary, ARCHITECTURE)
    try:
        for key, value in metadata.items():
            if key == "general.architecture":
                continue  # Written by GGUFWriter's constructor.
            if isinstance(value, str):
                writer.add_string(key, value)
            elif isinstance(value, bool):
                writer.add_bool(key, value)
            else:
                writer.add_uint32(key, value)
        for name in sorted(arrays):
            value = np.ascontiguousarray(
                arrays[name].reshape(storage_shape(arrays[name].shape))
            )
            # U8/BOOL occupy the standard I8 storage type, interpreted through the
            # explicit logical dtype in the reversible tensor map.
            if dtypes[name] in {"U8", "BOOL"}:
                value = value.view(np.int8)
            writer.add_tensor(
                stored_name(name),
                value,
                raw_dtype=gguf.GGMLQuantizationType(DTYPES[dtypes[name]][1]),
            )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        checked = load_gguf(temporary)
        for name, value in arrays.items():
            if (
                checked.dtypes[name] != dtypes[name]
                or checked.tensors[name].tobytes() != value.tobytes()
            ):
                raise ValueError(f"GGUF round-trip payload mismatch: {name}")
        del checked
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(
                temporary, path
            )  # Atomic no-clobber even with concurrent exporters.
            temporary.unlink()
    finally:
        writer.close()
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "tensor_count": len(arrays),
        "quantized_layers": len(modules),
        "rotated_layers": metadata[PREFIX + "convrot_layers"],
        "stored_tensor_bytes": sum(value.nbytes for value in arrays.values()),
        "group_sizes": sorted({m.group_size for m in modules.values()}),
        "architecture": ARCHITECTURE,
        "storage_version": STORAGE_VERSION,
        "verified_payload_roundtrip": True,
    }


def load_gguf(path: str | Path) -> GGUFArtifact:
    """Validate with the standard GGUFReader plus the full custom schema."""
    import gguf

    path = Path(path)
    with path.open("rb") as source:
        header = source.read(24)
    if len(header) != 24:
        raise ValueError("truncated GGUF header")
    magic, version, tensors_count, metadata_count = struct.unpack("<4sIQQ", header)
    if magic != b"GGUF" or version != 3:
        raise ValueError("expected little-endian GGUF version 3")
    if not 1 <= tensors_count <= 100_000 or not 1 <= metadata_count <= 16_384:
        raise ValueError("invalid GGUF directory counts")
    reader = gguf.GGUFReader(path, "r")
    metadata = {
        name: field.contents()
        for name, field in reader.fields.items()
        if not name.startswith("GGUF.")
    }
    if metadata.get("general.architecture") != ARCHITECTURE:
        raise ValueError("GGUF architecture is not efficientsam2")
    if metadata.get(PREFIX + "storage_version") != STORAGE_VERSION:
        raise ValueError("unsupported Efficient-SAM2 GGUF storage version")
    if (
        metadata.get("general.alignment", 32) != 32
        or reader.data_offset > 64 * 1024 * 1024
    ):
        raise ValueError("invalid alignment or excessive header")
    if metadata.get(PREFIX + "convrot_basis") != "regular-h4-kron":
        raise ValueError("unsupported ConvRot basis")
    required_json = ("tensor_map", "model_config", "source.metadata", "quantization")
    decoded = {}
    for field in required_json:
        value = metadata.get(PREFIX + field)
        if not isinstance(value, str) or len(value) > 64 * 1024 * 1024:
            raise ValueError(f"missing or invalid metadata: {field}")
        decoded[field] = _decode(value)
        if not isinstance(decoded[field], dict):
            raise ValueError(f"metadata must be an object: {field}")
    source_hash = metadata.get(PREFIX + "source.sha256")
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(c not in "0123456789abcdef" for c in source_hash)
    ):
        raise ValueError("invalid source SHA-256")
    mapping = decoded["tensor_map"]
    if set(mapping) != {tensor.name for tensor in reader.tensors}:
        raise ValueError("tensor map does not match the physical directory")
    arrays, dtypes = {}, {}
    end = reader.data_offset
    file_size = path.stat().st_size
    for tensor in sorted(reader.tensors, key=lambda t: t.data_offset):
        if (
            tensor.data_offset < end
            or tensor.data_offset % 32
            or tensor.data_offset + tensor.n_bytes > file_size
        ):
            raise ValueError("misaligned, overlapping, or truncated GGUF payload")
        end = tensor.data_offset + tensor.n_bytes
        entry = mapping[tensor.name]
        if not isinstance(entry, dict):
            raise ValueError("invalid tensor-map entry")
        name, dtype = entry.get("name"), entry.get("dtype")
        if (
            not isinstance(name, str)
            or not name
            or "\0" in name
            or name in arrays
            or tensor.name != stored_name(name)
            or dtype not in DTYPES
        ):
            raise ValueError("invalid tensor-map name or dtype")
        shape = _validate_shape(entry.get("shape"))
        if tuple(reversed(tensor.shape.tolist())) != storage_shape(shape):
            raise ValueError(f"physical/logical shape mismatch: {name}")
        numpy_dtype, ggml_dtype = DTYPES[dtype]
        if (
            int(tensor.tensor_type) != ggml_dtype
            or tensor.n_bytes != math.prod(shape) * np.dtype(numpy_dtype).itemsize
        ):
            raise ValueError(f"physical/logical dtype mismatch: {name}")
        arrays[name] = tensor.data.view(np.dtype(numpy_dtype)).reshape(shape).copy()
        dtypes[name] = dtype
    specs = decoded["quantization"]
    if (
        specs.get("scheme") != "symmetric-per-output-channel-int8"
        or not isinstance(specs.get("layers"), dict)
        or not specs["layers"]
    ):
        raise ValueError("unsupported or empty quantization schema")
    quantized = {}
    for name, spec in specs["layers"].items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            raise ValueError("invalid quantized layer metadata")
        required = {
            "in_features",
            "out_features",
            "group_size",
            "has_bias",
            "source_dtype",
            "activation_mode",
        }
        if not required <= set(spec) or type(spec["has_bias"]) is not bool:
            raise ValueError("incomplete quantized layer metadata")
        if any(
            type(spec[field]) is not int or spec[field] < 1
            for field in ("in_features", "out_features", "group_size")
        ):
            raise ValueError("invalid quantized layer dimensions")
        validate_mode(spec["activation_mode"])
        for suffix, dtype in (
            ("weight", "I8"),
            ("weight_scale", "F32"),
            ("comfy_quant", "U8"),
        ):
            if dtypes.get(_key(name, suffix)) != dtype:
                raise ValueError(
                    f"missing or invalid quantization tensor: {_key(name, suffix)}"
                )
        bias_key = _key(name, "bias")
        if (bias_key in arrays) != spec["has_bias"] or (
            spec["has_bias"] and dtypes[bias_key] != "F32"
        ):
            raise ValueError(f"bias metadata mismatch: {name}")
        record = QuantizedLinear(
            arrays[_key(name, "weight")],
            arrays[_key(name, "weight_scale")],
            spec["group_size"],
            arrays.get(bias_key),
            spec["source_dtype"],
        )
        if (record.in_features, record.out_features) != (
            spec["in_features"],
            spec["out_features"],
        ):
            raise ValueError(f"quantized layer shape mismatch: {name}")
        marker = arrays[_key(name, "comfy_quant")]
        if (
            marker.ndim != 1
            or marker.nbytes > 4096
            or _json(_decode(marker.tobytes().decode("utf-8")))
            != _json(record.marker())
        ):
            raise ValueError(f"quantized marker mismatch: {name}")
        quantized[name] = record
    if metadata.get(PREFIX + "int8_layers") != len(quantized) or metadata.get(
        PREFIX + "convrot_layers"
    ) != sum(q.group_size > 1 for q in quantized.values()):
        raise ValueError("quantized layer count metadata mismatch")
    expected_markers = {_key(name, "comfy_quant") for name in quantized}
    if {
        name
        for name in arrays
        if name == "comfy_quant" or name.endswith(".comfy_quant")
    } != expected_markers:
        raise ValueError("unindexed quantization marker")
    return GGUFArtifact(arrays, dtypes, quantized, metadata, path)


def load_gguf_model(
    model, path: str | Path, mode: str = "w8a16", *, inplace: bool = False
):
    """Strictly load full state into a matching stock model, then install refs.

    Inverse rotation into stock weights is available separately through
    ``load_gguf(path).state_dict()``. Loading a GGUF does not quantize again.
    """
    import torch
    from torch import nn

    from .torch_reference import ConvRotLinear

    validate_mode(mode)
    artifact = load_gguf(path)
    model = model if inplace else copy.deepcopy(model)
    # Validate every target before changing the model, including partial loading.
    expected = model.state_dict()
    restored = artifact.state_dict()
    if set(expected) != set(restored):
        raise ValueError("model and GGUF state-dict names differ")
    if any(expected[name].shape != restored[name].shape for name in expected):
        raise ValueError("model and GGUF state-dict shapes differ")
    targets = {}
    for name, record in artifact.quantized.items():
        target = model.get_submodule(name) if name else model
        if not isinstance(target, nn.Linear) or (
            target.in_features,
            target.out_features,
        ) != (record.in_features, record.out_features):
            raise ValueError(
                f"GGUF quantized target is not a matching nn.Linear: {name}"
            )
        targets[name] = target
    aliases = {}
    for name, target in model.named_modules(remove_duplicate=False):
        if isinstance(target, nn.Linear):
            aliases.setdefault(id(target), []).append(name)
    for names in aliases.values():
        records = [artifact.quantized.get(name) for name in names]
        if not any(record is not None for record in records):
            continue
        if any(record is None for record in records):
            raise ValueError("GGUF selects shared Linear aliases inconsistently")
        first = records[0]
        for record in records[1:]:
            if (
                record.group_size != first.group_size
                or not np.array_equal(record.weight_int8, first.weight_int8)
                or not np.array_equal(record.weight_scale, first.weight_scale)
                or (record.bias is None) != (first.bias is None)
                or (
                    record.bias is not None
                    and not np.array_equal(record.bias, first.bias)
                )
            ):
                raise ValueError("GGUF has conflicting data for shared Linear aliases")
    model.load_state_dict(restored, strict=True)
    replacements = {}
    for name, record in artifact.quantized.items():
        target = targets[name]
        if id(target) not in replacements:
            replacement = ConvRotLinear(record, mode).to(target.weight.device)
            replacement.train(target.training)
            replacements[id(target)] = replacement
        replacement = replacements[id(target)]
        if name:
            parent_name, _, child_name = name.rpartition(".")
            setattr(
                model.get_submodule(parent_name) if parent_name else model,
                child_name,
                replacement,
            )
        else:
            model = replacement
    return model
