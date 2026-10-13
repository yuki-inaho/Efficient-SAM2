from types import SimpleNamespace

import pytest
import torch
from tools.quantization.evaluate_q8_0_cpu import (
    apply_restore_policy,
    install_candidate,
    Q8ReferenceLinear,
    validate_box_protocol,
    verify_torch_fixtures,
    weight_fallback_reason,
)


def test_dense_restore_policy_is_identity_and_needs_no_module():
    artifact = object()
    assert apply_restore_policy(artifact, "dense") == (artifact, None)


def test_native_restore_policy_is_explicit_and_preserves_source():
    source, restored = object(), object()
    policy = {"restore_policy": "native-h4-f32-v1"}
    module = SimpleNamespace(POLICY=policy, native_h4_artifact=lambda a: restored)
    actual, metadata = apply_restore_policy(source, "native-h4-f32-v1", module)
    assert actual is restored and metadata == policy and metadata is not policy


@pytest.mark.parametrize(
    "policy,module",
    [
        ("unknown", None),
        ("native-h4-f32-v1", None),
        ("native-h4-f32-v1", SimpleNamespace(POLICY={"restore_policy": "wrong"})),
    ],
)
def test_missing_or_mismatched_restore_policy_is_rejected(policy, module):
    with pytest.raises(ValueError, match="policy"):
        apply_restore_policy(object(), policy, module)


@pytest.mark.parametrize(
    "width,reason", [(32, None), (112, "input-width-not-block-aligned")]
)
def test_weight_width_policy(width, reason):
    assert weight_fallback_reason(torch.ones(3, width)) == reason


def test_weight_scale_underflow_requires_whole_layer_fallback():
    weights = torch.ones(3, 64)
    weights[0, :32] = 1e-9
    assert weight_fallback_reason(weights) == "nonzero-weight-scale-fp16-underflow"


def test_weight_scale_overflow_requires_whole_layer_fallback():
    assert (
        weight_fallback_reason(torch.full((3, 64), 1e8)) == "weight-scale-fp16-overflow"
    )


def test_nonfinite_weights_are_rejected():
    with pytest.raises(ValueError, match="Non-finite"):
        weight_fallback_reason(torch.full((3, 64), float("nan")))


def test_zero_weights_are_valid_q8_blocks():
    assert weight_fallback_reason(torch.zeros(3, 64)) is None


def test_cached_linear_quantizes_weight_once_and_activation_per_call():
    calls = []

    def fake(value, *, weights):
        calls.append(weights)
        return value.detach().clone()

    source = torch.nn.Linear(32, 7)
    replacement = Q8ReferenceLinear(source, SimpleNamespace(fake_q8_0=fake))
    for value in [torch.randn(2, 32), torch.empty(0, 14, 14, 32)]:
        assert torch.equal(source(value), replacement(value))
    assert calls == [True, False, False]


def test_missing_kernel_fixtures_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="six preregistered"):
        verify_torch_fixtures(None, tmp_path)


def test_source_activation_mode_cannot_be_silently_ignored():
    artifact = SimpleNamespace(
        metadata={
            "efficient_sam2.quantization": '{"layers":{"a":{"activation_mode":"w8a8"}}}'
        }
    )
    with pytest.raises(ValueError, match="W8A16 source"):
        install_candidate(None, artifact, None, None)


@pytest.mark.parametrize("changed", [None, "box", "frames", "threads"])
def test_box_protocol_binds_exact_inputs(tmp_path, changed):
    import hashlib

    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"fixture")
    args = SimpleNamespace(
        initial_box=[1, 2, 3, 4], objects=[2], threads=4, gguf=str(artifact)
    )
    hashes = [str(i) for i in range(17)]
    protocol = {
        "raw_frames": list(range(17)),
        "cpu_threads": 4,
        "image_size": 1024,
        "multimask_output": False,
        "logit_threshold": 0.0,
        "artifact_sha256": hashlib.sha256(b"fixture").hexdigest(),
        "frame_sha256": hashes.copy(),
        "objects": [0, 1, 2, 4],
        "boxes_xyxy_original_pixels": {"2": [1, 2, 3, 4]},
    }
    if changed == "box":
        protocol["boxes_xyxy_original_pixels"] = {"2": [1, 2, 3, 5]}
    elif changed == "frames":
        protocol["frame_sha256"] = ["wrong"] * 17
    elif changed == "threads":
        protocol["cpu_threads"] = 8
    if changed is None:
        validate_box_protocol(protocol, args, hashes)
    else:
        with pytest.raises(ValueError, match="protocol"):
            validate_box_protocol(protocol, args, hashes)
