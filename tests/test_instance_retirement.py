"""Retirement must survive rebuilding pixel IDs without deleting raw evidence."""

import copy

import pytest

from tools.workflow.instance_retirement import active_instances


def test_retired_records_keep_evidence_but_lose_pixel_ids():
    rows = [
        {"candidate_id": "a", "status": "UNREVIEWED", "local_id": 1},
        {"candidate_id": "b", "alias_of": "a", "local_id": 2, "mask_path": "b.png"},
        {
            "candidate_id": "c",
            "status": "REVIEWED_REJECTED",
            "local_id": 3,
            "mask_sha256": "old-evidence",
            "review": "rejected after inspection",
        },
    ]
    before = copy.deepcopy(rows)
    assert active_instances(rows) == [rows[0]]
    for index in (1, 2):
        expected = before[index]
        expected.pop("local_id")
        assert rows[index] == expected
    assert len(rows) == 3


@pytest.mark.parametrize("parent", ["missing", "b", "c", "d"])
def test_invalid_parent_fails_without_partial_mutation(parent):
    rows = [
        {"candidate_id": "a"},
        {"candidate_id": "b", "alias_of": parent, "local_id": 2},
        {"candidate_id": "c", "status": "REVIEWED_REJECTED", "local_id": 3},
        {"candidate_id": "d", "alias_of": "a", "local_id": 4},
    ]
    before = copy.deepcopy(rows)
    with pytest.raises(ValueError):
        active_instances(rows)
    assert rows == before


def test_explicit_rejection_status_and_duplicate_identity():
    rows = [{"candidate_id": "a", "status": "CUSTOM_REJECTION", "local_id": 8}]
    assert active_instances(rows, rejected_statuses=("CUSTOM_REJECTION",)) == []
    assert "local_id" not in rows[0]
    with pytest.raises(ValueError, match="duplicate"):
        active_instances([{"candidate_id": "a"}, {"candidate_id": "a"}])
