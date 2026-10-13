"""Regular-Hadamard group rotation from ConvRot (arXiv:2512.03673).

This uses the regular H4 basis, not the Sylvester Walsh-Hadamard basis.
No channels are padded or discarded: an explicit group must divide the width.
"""

from __future__ import annotations

import numpy as np


def validate_group_size(group_size: int) -> int:
    if type(group_size) is not int or group_size < 1:
        raise ValueError("group_size must be a positive power of four (including 1)")
    value = group_size
    while value > 1 and value % 4 == 0:
        value //= 4
    if value != 1:
        raise ValueError("group_size must be a power of four")
    return group_size


def choose_group_size(in_features: int, max_group_size: int = 256) -> int:
    """Largest power of four <= max_group_size that divides in_features.

    Widths 112, 224, 448, and 896 select 16, 16, 64, and 64. Widths not
    divisible by four select identity (1), so odd feature tails are preserved.
    """
    if type(in_features) is not int or in_features < 1:
        raise ValueError("in_features must be a positive integer")
    validate_group_size(max_group_size)
    group = 1
    while group * 4 <= max_group_size and in_features % (group * 4) == 0:
        group *= 4
    return group


def regular_hadamard(group_size: int, dtype=np.float32) -> np.ndarray:
    """Return normalized H4 Kronecker power; R.T == R and R @ R == I."""
    validate_group_size(group_size)
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise TypeError("rotation dtype must be floating point")
    base = (
        np.array(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype
        )
        / 2
    )
    matrix = np.ones((1, 1), dtype=dtype)
    while matrix.shape[0] < group_size:
        matrix = np.kron(matrix, base)
    return matrix


def rotate_numpy(value: np.ndarray, group_size: int) -> np.ndarray:
    """Apply x @ R to contiguous feature groups on the last axis."""
    validate_group_size(group_size)
    value = np.asarray(value)
    if value.ndim < 1 or value.shape[-1] < 1:
        raise ValueError("rotation requires a nonempty feature axis")
    if value.dtype.kind != "f":
        raise TypeError("rotation input must be floating point")
    if value.shape[-1] % group_size:
        raise ValueError("group_size must divide the feature width; no tail is dropped")
    if group_size == 1:
        return value.copy()
    shape = (*value.shape[:-1], value.shape[-1] // group_size, group_size)
    return (value.reshape(shape) @ regular_hadamard(group_size, value.dtype).T).reshape(
        value.shape
    )
