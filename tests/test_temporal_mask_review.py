"""Prevent identity switches and missing reviews from becoming draft labels."""

import copy
import hashlib
from pathlib import Path

import pytest

from tools.workflow.temporal_mask_review import select_reviewed_temporal_masks


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def case(tmp_path):
    evidence = tmp_path / "comparison.jpg"
    evidence.write_bytes(b"original and predicted comparison")
    tracks = {"seed_ids": ["object-a"], "frames": []}
    review = {"reviewer": "reviewer", "decisions": []}
    for i, (area, decision) in enumerate(
        [
            (10, "accept_draft"),
            (9, "reject_identity_switch"),
            (0, "missed_visible"),
            (0, "visually_absent"),
        ]
    ):
        mask = tmp_path / f"mask{i}.png"
        mask.write_bytes(f"immutable mask {i}".encode())
        obj = {
            "seed_id": "object-a",
            "area": area,
            "mask_path": str(mask),
            "mask_sha256": sha(mask),
        }
        tracks["frames"].append({"frame_index": i, "image_sha256": str(i) * 64, "instances": [obj]})
        review["decisions"].append(
            {
                "frame_index": i,
                "seed_id": "object-a",
                "decision": decision,
                "observations": "Individual visual decision, not a quality approval.",
                "image_sha256": str(i) * 64,
                "mask_sha256": sha(mask),
                "evidence": [
                    {"path": str(evidence), "sha256": sha(evidence), "frames": [0, 1, 2, 3]}
                ],
            }
        )
    return tracks, review


def select(case):
    return select_reviewed_temporal_masks(*case, expected_frames=range(4))


def test_nonempty_identity_switch_excluded_and_missing_visible_retained(case):
    before = copy.deepcopy(case)
    assert select(case) == {(0, "object-a")}
    assert case == before


@pytest.mark.parametrize(
    "failure",
    [
        "missing_decision",
        "duplicate_decision",
        "extra_seed",
        "missing_frame",
        "duplicate_frame",
        "missing_object",
        "duplicate_object",
        "unknown_status",
        "empty_evidence",
        "stale_evidence",
        "wrong_evidence_frame",
        "stale_mask",
        "stale_image",
        "empty_reviewer",
        "empty_observations",
        "changed_rejected_mask",
    ],
)
def test_incomplete_or_stale_review_fails_closed(case, failure):
    tracks, review = case
    row = review["decisions"][0]
    if failure == "missing_decision":
        review["decisions"].pop()
    elif failure == "duplicate_decision":
        review["decisions"].append(copy.deepcopy(row))
    elif failure == "extra_seed":
        row["seed_id"] = "unrequested-object"
    elif failure == "missing_frame":
        tracks["frames"].pop()
    elif failure == "duplicate_frame":
        tracks["frames"].append(copy.deepcopy(tracks["frames"][0]))
    elif failure == "missing_object":
        tracks["frames"][0]["instances"] = []
    elif failure == "duplicate_object":
        tracks["frames"][0]["instances"] *= 2
    elif failure == "unknown_status":
        row["decision"] = "UNREVIEWED"
    elif failure == "empty_evidence":
        row["evidence"] = []
    elif failure == "stale_evidence":
        Path(row["evidence"][0]["path"]).write_bytes(b"changed")
    elif failure == "wrong_evidence_frame":
        row["evidence"][0]["frames"] = [3]
    elif failure == "stale_mask":
        row["mask_sha256"] = "f" * 64
    elif failure == "stale_image":
        row["image_sha256"] = "f" * 64
    elif failure == "empty_reviewer":
        review["reviewer"] = " "
    elif failure == "empty_observations":
        row["observations"] = " "
    elif failure == "changed_rejected_mask":
        Path(tracks["frames"][1]["instances"][0]["mask_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        select(case)


@pytest.mark.parametrize(
    "frame,decision",
    [
        (2, "accept_draft"),
        (0, "visually_absent"),
        (0, "missed_visible"),
        (3, "reject_identity_switch"),
    ],
)
def test_decision_must_match_prediction_presence(case, frame, decision):
    case[1]["decisions"][frame]["decision"] = decision
    with pytest.raises(ValueError):
        select(case)


def test_deferred_mask_is_retained_but_never_selected(case):
    case[1]["decisions"][0]["decision"] = "defer_mask"
    assert select(case) == set()


def test_empty_scope_is_not_reviewed():
    with pytest.raises(ValueError):
        select_reviewed_temporal_masks(
            {"seed_ids": [], "frames": []}, {"reviewer": "r", "decisions": []}, expected_frames=[]
        )
