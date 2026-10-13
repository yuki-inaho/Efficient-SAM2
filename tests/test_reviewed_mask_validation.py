"""Expanded crops must map masks and prompts back to the same image coordinates."""

import copy

import numpy as np
import pytest
from PIL import Image

from tools.workflow.reviewed_mask_validation import sha, validate_replacement


def fixture(tmp_path):
    mask = np.zeros((6, 6), np.uint8)
    mask[2:5, 2:5] = 255
    path = tmp_path / "mask.png"
    Image.fromarray(mask).save(path)
    old = {"candidate_id": "a", "crop": [4, 4, 2, 2], "center": [4, 4]}
    new = {
        "candidate_id": "a",
        "crop": [2, 2, 6, 6],
        "center": [4, 4],
        "mask_path": str(path),
        "mask_sha256": sha(path),
        "area": 9,
        "score": 0.9,
        "points": [[2, 2], [4, 4], [0, 0]],
        "labels": [1, 1, 0],
        "own_center": True,
        "positive_violations": 0,
        "negative_violations": 0,
    }
    return old, new


def test_expanded_origin_retains_original_global_position(tmp_path):
    old, new = fixture(tmp_path)
    validate_replacement(new, old, (10, 10))
    np.testing.assert_array_equal(np.asarray(new["points"])[0] + new["crop"][:2], [4, 4])


def test_changed_mask_bytes_rejected(tmp_path):
    old, new = fixture(tmp_path)
    Image.new("L", (6, 6)).save(new["mask_path"])
    with pytest.raises(ValueError, match="changed replacement mask"):
        validate_replacement(new, old, (10, 10))


@pytest.mark.parametrize(
    "change",
    [
        {"crop": [5, 5, 6, 6]},
        {"crop": [2.5, 2, 6, 6]},
        {"crop": [2, 2, 5, 6]},
        {"center": [5, 4]},
        {"points": [[2, 2], [4, 4], [3, 3]]},
        {"area": 1},
        {"negative_violations": 1},
    ],
)
def test_invalid_crop_or_stale_claim_rejected(tmp_path, change):
    old, new = fixture(tmp_path)
    new.update(copy.deepcopy(change))
    with pytest.raises(ValueError):
        validate_replacement(new, old, (10, 10))
