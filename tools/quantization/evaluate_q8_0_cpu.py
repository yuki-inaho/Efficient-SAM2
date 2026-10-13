"""Evaluate the separate, experimental ggml Q8_0 second-quantization candidate.

Uses a kernel-validated emulator from the matching native-runtime checkout.
This is a CPU FP32 mathematical simulation, never a speed benchmark. It does
not modify the input GGUF. All results and fixture files stay outside git.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from efficient_sam2_quant import load_gguf
from tools.quantization.evaluate_cpu import external_path, prepare_fixture, run, sha256
from torch import nn
from torch.nn import functional as F


def import_file(path):
    spec = importlib.util.spec_from_file_location(f"q8_candidate_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import candidate module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_torch_fixtures(emulator, directory):
    """Fail closed unless Torch matches native packed values and kernel outputs."""
    packed_dtype = np.dtype([("scale", "<f2"), ("q", "i1", (32,))])
    rows: list[dict[str, Any]] = []
    expected_shapes = {
        (112, 336, 64),
        (224, 672, 64),
        (448, 1344, 64),
        (448, 1792, 256),
        (896, 2688, 64),
        (1792, 448, 256),
    }
    for path in sorted(directory.glob("*_w.f32")):
        k, n, m = map(int, path.stem.removesuffix("_w").split("_"))
        prefix = str(path).removesuffix("_w.f32")
        weight = torch.from_numpy(np.fromfile(path, "<f4").reshape(n, k))
        value = torch.from_numpy(np.fromfile(prefix + "_x.f32", "<f4").reshape(m, k))
        native_output = np.fromfile(prefix + "_y.f32", "<f4").reshape(m, n)
        checks = {}
        for data, suffix, weights in [(weight, "w", True), (value, "x", False)]:
            if k % 32:
                continue
            native = np.fromfile(prefix + f"_{suffix}.packed", packed_dtype)
            expected = (
                native["q"].astype(np.float32)
                * native["scale"].astype(np.float32)[:, None]
            ).reshape(data.shape)
            actual = emulator.fake_q8_0(data, weights=weights).numpy()
            checks[suffix + "_dequant_exact"] = bool(np.array_equal(actual, expected))
            if not checks[suffix + "_dequant_exact"]:
                raise ValueError(
                    f"Torch/native packed dequant mismatch: {path.name}/{suffix}"
                )
        actual_output = emulator.linear(value, weight).numpy()
        if k % 32 == 0:
            cached = Q8ReferenceLinear(
                SimpleNamespace(
                    in_features=k, out_features=n, weight=weight, bias=None
                ),
                emulator,
            )(value).numpy()
            if not np.array_equal(cached, actual_output):
                raise ValueError(
                    "Cached evaluation path differs from validated emulator"
                )
        error = float(
            np.linalg.norm(actual_output - native_output)
            / max(float(np.linalg.norm(native_output)), 1e-30)
        )
        if not np.isfinite(actual_output).all() or error > 1e-6:
            raise ValueError(f"Torch/native kernel relative L2 {error}: {path.name}")
        rows.append({"shape_k_n_m": [k, n, m], **checks, "kernel_relative_l2": error})
    if {tuple(row["shape_k_n_m"]) for row in rows} != expected_shapes:
        raise ValueError("The six preregistered kernel fixtures are required")
    return {
        "passed": True,
        "relative_l2_limit": 1e-6,
        "cases": rows,
        "fixture_sha256": {
            p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file()
        },
    }


def weight_fallback_reason(weight):
    if weight.shape[-1] % 32:
        return "input-width-not-block-aligned"
    maximum = weight.float().reshape(-1, 32).abs().amax(dim=-1)
    if not bool(torch.isfinite(maximum).all()):
        raise ValueError("Non-finite restored weight")
    scale = maximum / 127.0
    if bool(((maximum > 0) & (scale.half() == 0)).any()):
        return "nonzero-weight-scale-fp16-underflow"
    if bool((maximum > 65504.0 * 127.0).any()):
        return "weight-scale-fp16-overflow"
    return None


class Q8ReferenceLinear(nn.Module):
    """Cached dequantized Q8 weights with per-call dynamic activation Q8."""

    def __init__(self, source, emulator):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.register_buffer(
            "weight", emulator.fake_q8_0(source.weight.detach(), weights=True)
        )
        self.register_buffer(
            "bias", source.bias.detach().clone() if source.bias is not None else None
        )
        self.emulator = emulator

    def forward(self, value):
        return F.linear(
            self.emulator.fake_q8_0(value, weights=False), self.weight, self.bias
        )


def install_candidate(model, artifact, emulator, manifest_module):
    specification = json.loads(artifact.metadata["efficient_sam2.quantization"])
    if any(
        row["activation_mode"] != "w8a16" for row in specification["layers"].values()
    ):
        raise ValueError("This candidate requires a W8A16 source artifact")
    model.load_state_dict(artifact.state_dict(), strict=True)
    manifest = manifest_module.build_manifest(model, artifact)
    for row in manifest["layers"]:
        module = model.get_submodule(row["module"])
        reason = weight_fallback_reason(module.weight)
        if reason is not None:
            row.update(candidate="f32", reason=reason)
            continue
        replacement = Q8ReferenceLinear(module, emulator)
        parent_name, _, attribute = row["module"].rpartition(".")
        setattr(
            model.get_submodule(parent_name) if parent_name else model,
            attribute,
            replacement,
        )
    manifest["selected_count"] = sum(
        row["candidate"] != "f32" for row in manifest["layers"]
    )
    manifest["fallback_count"] = len(manifest["layers"]) - manifest["selected_count"]
    return manifest


def apply_restore_policy(artifact, policy, module=None):
    """Keep dense restoration as default; opt in explicitly to native ordering."""
    if policy == "dense":
        return artifact, None
    if policy != "native-h4-f32-v1" or module is None:
        raise ValueError("Unknown or unavailable restoration policy")
    if module.POLICY.get("restore_policy") != policy:
        raise ValueError("Restoration module policy does not match the requested mode")
    return module.native_h4_artifact(artifact), dict(module.POLICY)


def validate_box_protocol(protocol, args, actual_frame_hashes):
    """Bind a separate box oracle to its predeclared native parity inputs."""
    if args.initial_box is None or len(args.objects) != 1:
        raise ValueError("Native box protocol requires one object and an initial box")
    object_id = args.objects[0]
    required = {
        "raw_frames": list(range(17)),
        "cpu_threads": args.threads,
        "image_size": 1024,
        "multimask_output": False,
        "logit_threshold": 0.0,
        "artifact_sha256": sha256(external_path(args.gguf)),
        "frame_sha256": actual_frame_hashes,
    }
    if any(protocol.get(key) != value for key, value in required.items()):
        raise ValueError("Native box protocol inputs differ from evaluation inputs")
    if (
        object_id not in protocol["objects"]
        or protocol["boxes_xyxy_original_pixels"].get(str(object_id))
        != args.initial_box
    ):
        raise ValueError(
            "Object or initial box differs from the frozen native protocol"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--name", default="protocol-v1-q8-0-emulated")
    parser.add_argument("--reference-dir")
    parser.add_argument(
        "--emulator-dir",
        required=True,
        help="Matching native checkout's efficient_sam2/cpu_quant directory",
    )
    parser.add_argument("--kernel-fixtures", required=True)
    parser.add_argument(
        "--restore-policy",
        choices=["dense", "native-h4-f32-v1"],
        default="dense",
        help="Optional inverse-ConvRot arithmetic order; native mode requires restore_policy.py in --emulator-dir",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--objects", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument(
        "--initial-box",
        type=float,
        nargs=4,
        help="Separate single-object box-prompt parity protocol",
    )
    parser.add_argument("--dump-temporal", action="store_true")
    parser.add_argument(
        "--parity-protocol",
        help="Frozen native box protocol JSON; verifies inputs and records its hash",
    )
    parser.add_argument(
        "--manifest-reference",
        help="Prior accepted candidate manifest JSON; require exact layer/fallback equality",
    )
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("--name must be a simple directory name")
    if args.initial_box is None and not args.reference_dir:
        parser.error("Full-mask evaluation requires --reference-dir")
    if args.initial_box is not None and len(args.objects) != 1:
        parser.error("--initial-box requires one --objects ID")
    parity_path = external_path(args.parity_protocol) if args.parity_protocol else None
    if parity_path is not None:
        _, _, frame_protocol = prepare_fixture(
            external_path(args.work_dir), 17, args.objects
        )
        validate_box_protocol(
            json.loads(parity_path.read_text()), args, frame_protocol["frame_sha256"]
        )
    output = external_path(args.work_dir) / "results" / args.name
    if (output / "report.json").exists():
        parser.error("Output report already exists; choose a new --name")
    emulator_dir = Path(args.emulator_dir).expanduser().resolve()
    emulator = import_file(emulator_dir / "emulator.py")
    manifest_module = import_file(emulator_dir / "manifest.py")
    restore_module = (
        import_file(emulator_dir / "restore_policy.py")
        if args.restore_policy != "dense"
        else None
    )
    torch.set_num_threads(args.threads)
    with torch.inference_mode():
        fixture_report = verify_torch_fixtures(
            emulator, external_path(args.kernel_fixtures)
        )
    source_hashes = {
        name: sha256(emulator_dir / name) for name in ["emulator.py", "manifest.py"]
    }
    if restore_module is not None:
        source_hashes["restore_policy.py"] = sha256(emulator_dir / "restore_policy.py")

    def transform(model, output):
        artifact = load_gguf(external_path(args.gguf))
        artifact, restore_policy = apply_restore_policy(
            artifact, args.restore_policy, restore_module
        )
        with torch.inference_mode():
            manifest = install_candidate(model, artifact, emulator, manifest_module)
        if restore_policy is not None:
            manifest.update(
                restore_policy=restore_policy["restore_policy"],
                weight_quantizer=restore_policy["weight_quantizer"],
                activation_quantizer=restore_policy["activation_quantizer"],
            )
            (output / "compute_policy.json").write_text(
                json.dumps(restore_policy, indent=2) + "\n"
            )
        if args.manifest_reference:
            expected_manifest = json.loads(
                external_path(args.manifest_reference).read_text()
            )
            if manifest != expected_manifest:
                raise ValueError(
                    "Candidate layer/fallback manifest differs from the accepted reference"
                )
        (output / "candidate_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        (output / "torch_kernel_fixture_verification.json").write_text(
            json.dumps(fixture_report, indent=2) + "\n"
        )
        metadata = {
            "variant": "ggml-q8_0-second-quantization-emulated",
            "execution": "CPU FP32 dequantized mathematical reference; not native INT8 execution or a speed benchmark",
            "gguf_loaded_sha256": sha256(external_path(args.gguf)),
            "strict_state_dict_load": True,
            "candidate_manifest_sha256": sha256(output / "candidate_manifest.json"),
            "emulator_source_sha256": source_hashes,
            "torch_kernel_fixture_verification_sha256": sha256(
                output / "torch_kernel_fixture_verification.json"
            ),
            "weights": "Inverse-ConvRot FP32, then second lossy Q8_0 quantization; block32 maxabs/127 scale stored FP16; ties away from zero",
            "activations": "Dynamic block32 maxabs/127 scale stored FP16; nearest-even integer rounding matching this x86 AVX2 kernel",
            "accumulation": "Dequantized FP32 matmul; mathematically equivalent block dots, different reduction order",
            "limitations": "Small TRAIN smoke only; native full-model parity and optimized kernel performance require separate validation",
        }
        metadata["restore_policy"] = args.restore_policy
        if restore_policy is not None:
            metadata["variant"] = (
                "ggml-q8_0-second-quantization-native-h4-f32-v1-emulated"
            )
            metadata["compute_policy_sha256"] = sha256(output / "compute_policy.json")
        if parity_path is not None:
            metadata["native_box_protocol_sha256"] = sha256(parity_path)
            (output / "native_parity_protocol.json").write_bytes(
                parity_path.read_bytes()
            )
        if args.manifest_reference:
            metadata["manifest_reference_sha256"] = sha256(
                external_path(args.manifest_reference)
            )
        print(
            json.dumps(
                {
                    "event": "candidate_installed",
                    "selected": manifest["selected_count"],
                    "fallback": manifest["fallback_count"],
                }
            ),
            flush=True,
        )
        return model, metadata

    evaluation = SimpleNamespace(
        checkpoint=args.checkpoint,
        work_dir=args.work_dir,
        name=args.name,
        frame_count=17,
        objects=args.objects,
        threads=args.threads,
        stock_sam2=False,
        variant="fp32",
        max_group_size=256,
        include=None,
        exclude=None,
        reference_dir=args.reference_dir,
        gguf=None,
        export_gguf=None,
        initial_box=args.initial_box,
        dump_temporal=args.dump_temporal,
    )
    run(evaluation, model_transform=transform)


if __name__ == "__main__":
    main()
