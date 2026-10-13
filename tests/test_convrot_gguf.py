import json
import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")
from efficient_sam2_quant import (
    export_gguf,
    load_gguf,
    load_gguf_model,
    quantize_linear_modules,
)
from efficient_sam2_quant.gguf_io import ARCHITECTURE, PREFIX, stored_name


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(112, 32)
        self.last = torch.nn.Linear(32, 7, bias=False)
        self.conv = torch.nn.Conv2d(3, 4, 3)
        self.norm = torch.nn.LayerNorm(32)
        self.register_buffer("counter", torch.tensor(9, dtype=torch.int64))
        self.register_buffer(
            "five_axes", torch.arange(32, dtype=torch.float32).reshape(2, 2, 2, 2, 2)
        )
        self.register_buffer("flags", torch.tensor([True, False]))
        self.register_buffer("bf16", torch.tensor([0.25, -0.5], dtype=torch.bfloat16))

    def forward(self, value):
        return self.last(self.norm(self.first(value)))


@pytest.mark.parametrize("mode", ["w8a16", "w8a8"])
def test_standard_gguf_reader_and_reference_reload_parity(tmp_path, mode):
    torch.manual_seed(5)
    model = TinyModel().eval()
    reference = quantize_linear_modules(model, mode)
    path = tmp_path / "weights.gguf"
    config = {"variant": "base_plus", "image_size": 1024, "sam2_version": "2.1"}
    stats = export_gguf(reference, path, model_config=config, test_fixture=True)
    assert stats["verified_payload_roundtrip"] is True
    reader = gguf.GGUFReader(path)
    assert (
        reader.fields["general.architecture"].contents()
        == ARCHITECTURE
        == "efficientsam2"
    )
    assert reader.fields["general.quantization_version"].contents() == 1
    assert json.loads(reader.fields[PREFIX + "model_config"].contents()) == config
    physical = {t.name: t for t in reader.tensors}
    assert (
        physical[stored_name("first.weight")].tensor_type
        == gguf.GGMLQuantizationType.I8
    )
    assert (
        physical[stored_name("first.weight_scale")].tensor_type
        == gguf.GGMLQuantizationType.F32
    )
    assert tuple(physical[stored_name("first.weight")].shape) == (112, 32)
    assert all(len(name) <= 63 for name in physical)
    artifact = load_gguf(path)
    restored = load_gguf_model(model, path, mode)
    value = torch.randn(2, 3, 112)
    torch.testing.assert_close(restored(value), reference(value), atol=0, rtol=0)
    for name in ["conv.weight", "conv.bias", "counter", "five_axes", "flags", "bf16"]:
        torch.testing.assert_close(
            artifact.state_dict()[name], model.state_dict()[name], atol=0, rtol=0
        )
    assert isinstance(model.first, torch.nn.Linear)
    np.testing.assert_array_equal(
        artifact.quantized["first"].weight_int8, reference.first.weight_int8.numpy()
    )
    if mode == "w8a16":
        # The stock float model needs no online rotation after inverse-rotation.
        model.load_state_dict(artifact.state_dict(), strict=True)
        torch.testing.assert_close(model(value), reference(value), atol=2e-6, rtol=2e-5)


def test_root_linear_odd_width_zero_and_long_names(tmp_path):
    model = torch.nn.Linear(5, 3)
    with torch.no_grad():
        model.weight.zero_()
    q = quantize_linear_modules(model)
    path = tmp_path / "root.gguf"
    export_gguf(q, path, test_fixture=True)
    artifact = load_gguf(path)
    assert artifact.quantized[""].group_size == 1
    assert artifact.quantized[""].marker() == {
        "format": "int8_tensorwise",
        "convrot": False,
        "per_row": True,
    }
    reloaded = load_gguf_model(model, path)
    torch.testing.assert_close(
        q(torch.ones(5)), reloaded(torch.ones(5)), atol=0, rtol=0
    )
    long = torch.nn.ModuleDict({"a" * 200: torch.nn.Linear(4, 3)})
    export_gguf(
        quantize_linear_modules(long), tmp_path / "long.gguf", test_fixture=True
    )
    assert "a" * 200 in load_gguf(tmp_path / "long.gguf").quantized


def test_export_refuses_clobber_and_nonquantized_model(tmp_path):
    path = tmp_path / "preserve.gguf"
    with pytest.raises(ValueError, match="no ConvRotLinear"):
        export_gguf(torch.nn.Linear(4, 3), path)
    q = quantize_linear_modules(torch.nn.Linear(4, 3))
    export_gguf(q, path, test_fixture=True)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        export_gguf(q, path)
    assert path.read_bytes() == original
    export_gguf(q, path, overwrite=True, test_fixture=True)
    with pytest.raises(ValueError, match="SHA-256"):
        export_gguf(q, tmp_path / "bad.gguf", source_sha256="not-a-hash")


def test_reject_wrong_target_without_modifying_it(tmp_path):
    q = quantize_linear_modules(torch.nn.Linear(4, 3))
    path = tmp_path / "test.gguf"
    export_gguf(q, path, test_fixture=True)
    wrong = torch.nn.Linear(5, 3)
    original = wrong.weight.detach().clone()
    with pytest.raises(ValueError, match="shapes"):
        load_gguf_model(wrong, path, inplace=True)
    torch.testing.assert_close(wrong.weight, original, atol=0, rtol=0)


@pytest.mark.parametrize(
    "replace_from,replace_to,error",
    [
        (b"efficientsam2", b"wrongarch0000", "architecture"),
        (b"regular-h4-kron", b"wrong-basis0000", "basis"),
        (b'"group_size":4', b'"group_size":2', "power of four"),
        (b'"has_bias":true', b'"has_bias":null', "metadata"),
        (b'"in_features":4', b'"in_features":5', "shape"),
    ],
)
def test_reject_corrupt_metadata(tmp_path, replace_from, replace_to, error):
    assert len(replace_from) == len(replace_to)
    path = tmp_path / "bad.gguf"
    export_gguf(quantize_linear_modules(torch.nn.Linear(4, 3)), path, test_fixture=True)
    data = path.read_bytes()
    assert replace_from in data
    path.write_bytes(data.replace(replace_from, replace_to, 1))
    with pytest.raises(ValueError, match=error):
        load_gguf(path)


def test_reject_invalid_scale_and_truncated_file(tmp_path):
    path = tmp_path / "bad.gguf"
    export_gguf(quantize_linear_modules(torch.nn.Linear(4, 3)), path, test_fixture=True)
    reader = gguf.GGUFReader(path)
    scale = next(t for t in reader.tensors if t.name == stored_name("weight_scale"))
    offset = scale.data_offset
    del reader, scale
    with path.open("r+b") as target:
        target.seek(offset)
        target.write(struct.pack("<f", 0.0))
    with pytest.raises(ValueError, match="positive"):
        load_gguf(path)
    path.write_bytes(b"GGUF")
    with pytest.raises(ValueError, match="truncated"):
        load_gguf(path)


def test_shared_linear_aliases_survive_gguf_reload(tmp_path):
    shared = torch.nn.Linear(16, 4)
    original = torch.nn.ModuleDict({"a": shared, "b": shared})
    path = tmp_path / "aliases.gguf"
    export_gguf(quantize_linear_modules(original), path, test_fixture=True)
    restored = load_gguf_model(original, path)
    assert restored["a"] is restored["b"]
