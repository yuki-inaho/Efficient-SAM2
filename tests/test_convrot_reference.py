import numpy as np
import pytest

from efficient_sam2_quant import (
    choose_group_size,
    quantize_per_token,
    quantize_weight,
    QuantizedLinear,
    regular_hadamard,
    rotate_numpy,
)


@pytest.mark.parametrize("group", [1, 4, 16, 64, 256])
def test_regular_hadamard_is_symmetric_orthogonal_and_regular(group):
    rotation = regular_hadamard(group, np.float64)
    np.testing.assert_array_equal(rotation, rotation.T)
    np.testing.assert_array_equal(rotation @ rotation.T, np.eye(group))
    np.testing.assert_array_equal(rotation.sum(axis=0), np.ones(group))
    np.testing.assert_array_equal(rotation.sum(axis=1), np.ones(group))


def test_basis_is_regular_h4_not_sylvester():
    np.testing.assert_array_equal(
        regular_hadamard(4),
        np.array(
            [
                [1, 1, 1, -1],
                [1, 1, -1, 1],
                [1, -1, 1, 1],
                [-1, 1, 1, 1],
            ],
            dtype=np.float32,
        )
        / 2,
    )
    # A constant token is unchanged, rather than concentrated into one channel.
    np.testing.assert_array_equal(rotate_numpy(np.ones((2, 256), np.float32), 256), 1)


@pytest.mark.parametrize(
    "width,expected",
    [(1, 1), (3, 1), (12, 4), (112, 16), (224, 16), (448, 64), (896, 64), (1024, 256)],
)
def test_adaptive_group_selection(width, expected):
    assert choose_group_size(width) == expected


@pytest.mark.parametrize("width", [3, 12, 112, 224, 448, 896, 1024])
def test_unquantized_linear_invariance(width):
    rng = np.random.default_rng(width)
    value = rng.normal(size=(2, 3, width)).astype(np.float64)
    weight = rng.normal(size=(7, width)).astype(np.float64)
    bias = rng.normal(size=7)
    group = choose_group_size(width)
    np.testing.assert_allclose(
        rotate_numpy(value, group) @ rotate_numpy(weight, group).T + bias,
        value @ weight.T + bias,
        atol=2e-13,
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        rotate_numpy(rotate_numpy(value, group), group), value, atol=1e-14
    )


def test_noncontiguous_and_empty_batches_preserve_shape():
    array = np.arange(64, dtype=np.float32).reshape(4, 16).T
    assert not array.flags.c_contiguous
    np.testing.assert_allclose(rotate_numpy(rotate_numpy(array, 4), 4), array)
    assert rotate_numpy(np.zeros((0, 16), np.float32), 16).shape == (0, 16)
    assert rotate_numpy(np.zeros((2, 0, 16), np.float32), 16).shape == (2, 0, 16)


@pytest.mark.parametrize("group", [0, -1, 2, 8, 32, 3.0, True])
def test_reject_invalid_groups(group):
    with pytest.raises(ValueError):
        regular_hadamard(group)


def test_never_drop_or_pad_feature_tails():
    with pytest.raises(ValueError, match="divide"):
        rotate_numpy(np.ones((2, 6), np.float32), 4)
    record = quantize_weight(np.arange(15, dtype=np.float32).reshape(3, 5))
    assert record.group_size == 1
    assert record.weight_int8.shape == (3, 5)
    assert record.dequantized_weight()[2, -1] > 0


def test_zero_weights_zero_tokens_and_bias():
    bias = np.array([2, -3], dtype=np.float32)
    record = quantize_weight(np.zeros((2, 16), np.float32), bias)
    assert np.count_nonzero(record.weight_int8) == 0
    assert np.all(record.weight_scale == np.float32(1e-30))
    for mode in ["w8a16", "w8a8"]:
        np.testing.assert_array_equal(
            record.forward(np.zeros((3, 16), np.float32), mode),
            np.broadcast_to(bias, (3, 2)),
        )
    quantized, scale = quantize_per_token(np.zeros((0, 16), np.float32))
    assert quantized.shape == (0, 16)
    assert scale.shape == (0, 1)


def test_weight_quantization_matches_official_numeric_convention():
    rng = np.random.default_rng(22)
    weight = rng.normal(size=(8, 256)).astype(np.float32)
    record = quantize_weight(weight)
    rotated = (weight.reshape(8, 1, 256) @ regular_hadamard(256).T).reshape(8, 256)
    scales = np.maximum(
        np.abs(rotated).max(axis=1, keepdims=True) / 127, np.float32(1e-30)
    )
    expected = np.rint(rotated / scales).clip(-128, 127).astype(np.int8)
    np.testing.assert_array_equal(record.weight_int8, expected)
    np.testing.assert_array_equal(record.weight_scale, scales)
    # Rotation preserves L2 error; each rotated element is <= half a scale away.
    error = record.rotated_weight() - rotated
    assert np.all(np.abs(error) <= scales / 2 + 1e-6)
    np.testing.assert_allclose(
        np.linalg.norm(record.dequantized_weight() - weight),
        np.linalg.norm(error),
        atol=1e-5,
    )


def test_rounding_is_ties_to_even_and_activations_are_per_token():
    value = np.array(
        [
            [127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, -127],
            [254, 1, 3, 5, -1, -3, -5, -254],
        ],
        np.float32,
    )
    quantized, scale = quantize_per_token(value)
    np.testing.assert_array_equal(scale, [[1], [2]])
    np.testing.assert_array_equal(quantized[0], [127, 0, 2, 2, 0, -2, -2, -127])
    np.testing.assert_array_equal(quantized[0], quantized[1])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_weights_and_activations_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        quantize_weight(np.array([[bad]], np.float32))
    with pytest.raises(ValueError, match="finite"):
        quantize_per_token(np.array([[bad]], np.float32))


def test_invalid_shapes_scales_and_modes_rejected():
    with pytest.raises(ValueError):
        quantize_weight(np.zeros((2, 0), np.float32))
    with pytest.raises(ValueError):
        quantize_weight(np.ones(8, np.float32))
    with pytest.raises(TypeError):
        quantize_weight(np.ones((2, 4), np.int8))
    with pytest.raises(ValueError):
        QuantizedLinear(np.zeros((2, 4), np.int8), np.zeros((2, 1), np.float32), 4)
    with pytest.raises(ValueError):
        QuantizedLinear(np.zeros((2, 4), np.int8), np.ones((2,), np.float32), 4)
    with pytest.raises(ValueError):
        QuantizedLinear(
            np.zeros((2, 4), np.int8),
            np.ones((2, 1), np.float32),
            4,
            np.ones(3, np.float32),
        )
    record = quantize_weight(np.ones((2, 4), np.float32))
    with pytest.raises(ValueError):
        record.forward(np.ones((1, 3), np.float32))
    with pytest.raises(ValueError):
        record.forward(np.ones((1, 4), np.float32), "w4a4")


def test_torch_reference_agrees_with_numpy_integer_oracle():
    torch = pytest.importorskip("torch")
    from efficient_sam2_quant import ConvRotLinear

    rng = np.random.default_rng(31)
    weight = rng.normal(size=(11, 112)).astype(np.float32)
    bias = rng.normal(size=11).astype(np.float32)
    value = rng.normal(size=(2, 3, 112)).astype(np.float32)
    record = quantize_weight(weight, bias)
    for mode in ["w8a16", "w8a8"]:
        module = ConvRotLinear(record, mode)
        np.testing.assert_allclose(
            module(torch.from_numpy(value)).numpy(),
            record.forward(value, mode),
            atol=1e-5,
            rtol=1e-5,
        )
        assert module(torch.from_numpy(value).half()).dtype == torch.float16
        assert module(torch.empty((0, 112))).shape == (0, 11)
    with pytest.raises(ValueError):
        module(torch.zeros(111))
    with pytest.raises(TypeError):
        module(torch.zeros(112, dtype=torch.int8))


def test_linear_replacement_selection_sharing_and_conv2d_unchanged():
    torch = pytest.importorskip("torch")
    from efficient_sam2_quant import ConvRotLinear, quantize_linear_modules

    shared = torch.nn.Linear(16, 8)
    model = torch.nn.ModuleDict(
        {
            "conv": torch.nn.Conv2d(3, 8, 3),
            "first": shared,
            "alias": shared,
            "last": torch.nn.Linear(8, 4),
        }
    )
    q = quantize_linear_modules(model, exclude=["last"])
    assert isinstance(q["first"], ConvRotLinear)
    assert q["first"] is q["alias"]
    assert isinstance(q["last"], torch.nn.Linear)
    assert isinstance(model["first"], torch.nn.Linear)
    torch.testing.assert_close(q["conv"].weight, model["conv"].weight, atol=0, rtol=0)
    with pytest.raises(ValueError, match="aliases"):
        quantize_linear_modules(model, include=["first"])
    assert isinstance(quantize_linear_modules(torch.nn.Linear(4, 2)), ConvRotLinear)


def test_torch_dtype_conversion_keeps_scales_fp32_and_rejects_nonfinite_w8a8():
    torch = pytest.importorskip("torch")
    from efficient_sam2_quant import ConvRotLinear

    record = quantize_weight(np.zeros((2, 4), np.float32), np.ones(2, np.float32))
    module = ConvRotLinear(record, "w8a8")
    original = module.weight_scale.clone()
    module.half()
    assert module.weight_scale.dtype == torch.float32
    assert module.rotation.dtype == torch.float32
    assert module.bias.dtype == torch.float32
    torch.testing.assert_close(module.weight_scale, original, atol=0, rtol=0)
    torch.testing.assert_close(
        module(torch.zeros(4, dtype=torch.float16)), torch.ones(2, dtype=torch.float16)
    )
    with pytest.raises(ValueError, match="finite"):
        module(torch.full((4,), float("inf")))
