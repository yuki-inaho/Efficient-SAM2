"""Quality must be proved independently of the number of generated files."""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image
from tools.workflow.annotation_quality_gate import audit_dataset, inspect_instances


def candidates():
    return [
        {"candidate_id": "a", "center": [2, 2]},
        {"candidate_id": "b", "center": [7, 2]},
    ]


def good_mask():
    a = np.zeros((10, 10), np.uint16)
    a[1:4, 1:4] = 1
    a[1:4, 6:9] = 2
    return a


def test_missing_object():
    a = good_mask()
    a[a == 2] = 0
    assert inspect_instances(a, candidates())["missing"] == ["b"]


def test_cluster_two_centers():
    a = good_mask()
    a[a == 2] = 1
    assert inspect_instances(a, candidates())["merged"] == {"1": ["a", "b"]}


def test_equal_count_does_not_prove_bijection():
    a = good_mask()
    a[a == 2] = 1
    a[8, 8] = 2
    result = inspect_instances(a, candidates())
    assert result["mask_count"] == result["candidate_count"] == 2
    assert not result["passed"]
    assert result["unreferenced"] == [2]


@pytest.mark.parametrize("bad", [-1, 0.5, np.nan])
def test_invalid_label(bad):
    a = good_mask().astype(float)
    a[0, 0] = bad
    with pytest.raises(ValueError):
        inspect_instances(a, candidates())


def test_duplicate_candidate():
    with pytest.raises(ValueError):
        inspect_instances(good_mask(), [candidates()[0]] * 2)


@pytest.mark.parametrize("center", [[np.nan, 2], [-1, 2], [100, 2]])
def test_invalid_center(center):
    with pytest.raises(ValueError):
        inspect_instances(good_mask(), [{"candidate_id": "a", "center": center}])


def test_empty_reference():
    assert not inspect_instances(np.zeros((10, 10), np.uint16), [])["passed"]


@pytest.fixture
def dataset(tmp_path):
    rgb = tmp_path / "rgb"
    masks = tmp_path / "masks"
    rgb.mkdir()
    masks.mkdir()
    im = rgb / "frame.jpg"
    mask = masks / "frame_0000_idmap.png"
    Image.new("RGB", (10, 10), (40, 60, 80)).save(im)
    Image.fromarray(good_mask()).save(mask)
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    ref = {
        "expected_frames": 1,
        "frames": [
            {
                "frame_index": 0,
                "source_path": str(im),
                "source_sha256": sha(im),
                "width": 10,
                "height": 10,
                "detections": candidates(),
            }
        ],
    }
    refpath = tmp_path / "reference.json"
    refpath.write_text(json.dumps(ref))
    review = {
        "frames": [
            {
                "frame_index": 0,
                "reviewer": "test-fixture",
                "observations": "Two separate visible objects; each has its own mask.",
                "status": "accepted",
                "image_sha256": sha(im),
                "mask_sha256": sha(mask),
                "reference_sha256": sha(refpath),
                "unresolved_items": [],
            }
        ]
    }
    return refpath, masks, review, im


def test_good_requires_geometry_and_visual_review(dataset):
    ref, masks, review, _ = dataset
    assert audit_dataset(ref, masks, review)["status"] == "PASS"


def test_missing_visual_review(dataset):
    ref, masks, _, _ = dataset
    assert audit_dataset(ref, masks, None)["status"] != "PASS"


def test_missing_frame(dataset):
    ref, masks, review, _ = dataset
    data = json.loads(ref.read_text())
    data["expected_frames"] = 2
    ref.write_text(json.dumps(data))
    assert audit_dataset(ref, masks, review)["status"] == "FAIL"


def test_changed_image(dataset):
    ref, masks, review, im = dataset
    Image.new("RGB", (10, 10), (90, 60, 80)).save(im)
    assert audit_dataset(ref, masks, review)["status"] == "FAIL"


def test_stale_review(dataset):
    ref, masks, review, _ = dataset
    review["frames"][0]["mask_sha256"] = "stale"
    assert audit_dataset(ref, masks, review)["status"] != "PASS"


def test_unresolved_review(dataset):
    ref, masks, review, _ = dataset
    review["frames"][0]["unresolved_items"] = ["identity ambiguous"]
    assert audit_dataset(ref, masks, review)["status"] != "PASS"


def test_duplicate_review_frame(dataset):
    ref, masks, review, _ = dataset
    review["frames"] *= 2
    assert audit_dataset(ref, masks, review)["status"] != "PASS"
