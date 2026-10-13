import hashlib
import json

import numpy as np
import pytest
from PIL import Image
from tools.workflow.occlusion_training_data import load_reviewed_pairs


def fixture(tmp_path):
    image = tmp_path / "rgb.png"
    mask = tmp_path / "mask.png"
    Image.fromarray(np.zeros((10, 10, 3), np.uint8)).save(image)
    m = np.zeros((10, 10), np.uint8)
    m[3:7, 3:7] = 255
    Image.fromarray(m).save(mask)
    row = {
        "object_id": "one",
        "split": "train",
        "image_path": str(image),
        "mask_path": str(mask),
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
        "review_status": "accepted_pilot",
        "reviewer": "test",
        "observations": "single object",
        "prompt_points": [[5, 5], [0, 0]],
        "prompt_labels": [1, 0],
        "local_box": [2, 2, 8, 8],
    }
    path = tmp_path / "manifest.json"
    return path, {"status": "REVIEWED_PILOT_ONLY", "items": [row]}


@pytest.mark.parametrize(
    "failure", ["review", "hash", "negative_inside", "duplicate", "missing_split"]
)
def test_reject_unreviewed_changed_conflicting_or_leaking_pairs(tmp_path, failure):
    path, payload = fixture(tmp_path)
    row = payload["items"][0]
    if failure == "review":
        row["review_status"] = "unreviewed"
    elif failure == "hash":
        row["mask_sha256"] = "stale"
    elif failure == "negative_inside":
        row["prompt_points"][1] = [4, 4]
    elif failure == "duplicate":
        payload["items"].append({**row, "split": "test"})
    else:
        row.pop("split")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_reviewed_pairs(path)


def test_loads_only_explicitly_reviewed_unchanged_pairs(tmp_path):
    path, payload = fixture(tmp_path)
    path.write_text(json.dumps(payload))
    pairs = load_reviewed_pairs(path)
    assert len(pairs) == 1
    assert pairs[0]["target"].sum() == 16
