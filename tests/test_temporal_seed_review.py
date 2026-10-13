import copy
import hashlib

import pytest

from tools.workflow.temporal_seed_review import validate_seed_review


@pytest.fixture
def draft(tmp_path):
    evidence = tmp_path / "comparison.jpg"
    evidence.write_bytes(b"reviewed comparison")
    return {
        "review_status": "accepted_draft_for_temporal_probe",
        "reviewer": "local reviewer",
        "observations": "One object identified; boundary remains unapproved.",
        "review_evidence": [
            {
                "path": str(evidence),
                "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            }
        ],
    }


def test_draft_requires_explicit_probe_permission(draft):
    with pytest.raises(ValueError, match="explicit"):
        validate_seed_review(draft)


def test_reviewed_probe_does_not_promote_status_or_mutate_evidence(draft):
    before = copy.deepcopy(draft)
    validate_seed_review(draft, allow_draft=True)
    assert draft == before


@pytest.mark.parametrize("kind", ["empty", "changed", "missing", "reviewer", "observations"])
def test_unverifiable_draft_is_rejected(draft, kind):
    if kind == "empty":
        draft["review_evidence"] = []
    elif kind == "changed":
        draft["review_evidence"][0]["sha256"] = "0" * 64
    elif kind == "missing":
        draft["review_evidence"][0]["path"] += ".missing"
    else:
        draft[kind] = " "
    with pytest.raises(ValueError):
        validate_seed_review(draft, allow_draft=True)


def test_unknown_review_status_is_rejected_even_with_permission(draft):
    draft["review_status"] = "UNREVIEWED"
    with pytest.raises(ValueError):
        validate_seed_review(draft, allow_draft=True)


def test_existing_pilot_review_remains_compatible():
    validate_seed_review(
        {
            "review_status": "accepted_pilot",
            "reviewer": "local reviewer",
            "observations": "Previously reviewed pilot seed.",
        }
    )
