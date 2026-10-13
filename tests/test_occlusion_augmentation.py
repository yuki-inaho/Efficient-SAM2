import numpy as np
import pytest
from tools.workflow.occlusion_augmentation import augment_thin_green


def sample():
    y, x = np.mgrid[:128, :160]
    mask = ((x - 80) / 40) ** 2 + ((y - 64) / 30) ** 2 <= 1
    rgb = np.full((128, 160, 3), 60, np.uint8)
    rgb[mask] = [210, 60, 35]
    return rgb, mask


def test_target_and_original_are_unchanged_but_object_is_occluded():
    rgb, mask = sample()
    before = rgb.copy()
    augmented, target, occluder, meta = augment_thin_green(rgb, mask, seed=10, probability=1)
    assert np.array_equal(rgb, before)
    assert np.array_equal(target, mask)
    assert np.any(np.any(augmented != rgb, axis=-1) & mask)
    assert np.all(augmented[occluder == 0] == rgb[occluder == 0])
    assert 0.025 <= meta["object_occlusion_fraction"] <= 0.25
    assert meta["applied"]


def test_seed_reproduces_pixels_and_metadata():
    rgb, mask = sample()
    a = augment_thin_green(rgb, mask, seed=42, probability=1)
    b = augment_thin_green(rgb, mask, seed=42, probability=1)
    for x, y in zip(a[:3], b[:3], strict=True):
        assert np.array_equal(x, y)
    assert a[3] == b[3]


def test_clean_control_and_varied_seeds():
    rgb, mask = sample()
    clean = augment_thin_green(rgb, mask, seed=1, probability=0)
    assert np.array_equal(clean[0], rgb)
    assert not clean[2].any()
    assert not clean[3]["applied"]
    a = augment_thin_green(rgb, mask, seed=1, probability=1)
    b = augment_thin_green(rgb, mask, seed=2, probability=1)
    assert not np.array_equal(a[0], b[0])


@pytest.mark.parametrize("kind", ["empty", "shape", "wrong_rgb_type", "not_binary"])
def test_invalid_training_pair_is_rejected(kind):
    rgb, mask = sample()
    if kind == "empty":
        mask[:] = False
    if kind == "shape":
        mask = mask[:10]
    if kind == "wrong_rgb_type":
        rgb = rgb.astype(float)
    if kind == "not_binary":
        mask = mask.astype(np.uint8) * 2
    with pytest.raises(ValueError):
        augment_thin_green(rgb, mask, seed=0)
