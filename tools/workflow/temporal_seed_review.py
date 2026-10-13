"""Require explicit experiment scope and intact evidence for draft video seeds."""

import hashlib
from pathlib import Path


def validate_seed_review(row, *, allow_draft=False):
    """Validate review provenance without changing a seed's approval status."""
    for key in ("reviewer", "observations"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"missing seed {key}")
    status = row.get("review_status")
    if status == "accepted_pilot":
        return
    if status != "accepted_draft_for_temporal_probe":
        raise ValueError("unreviewed seed")
    if not allow_draft:
        raise ValueError("draft seeds require explicit temporal probe permission")
    evidence = row.get("review_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("missing draft seed visual evidence")
    for item in evidence:
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise ValueError("invalid draft seed evidence")
        try:
            digest = hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError("missing draft seed evidence file") from exc
        if digest != item["sha256"]:
            raise ValueError("changed draft seed evidence")
