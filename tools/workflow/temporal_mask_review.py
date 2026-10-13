"""Select draft temporal masks only after explicit per-frame identity review."""

import hashlib
from pathlib import Path


def select_reviewed_temporal_masks(tracks, review, *, expected_frames):
    """Return accepted (frame, seed) pairs; preserve rejected/missing hypotheses.

    This validates review completeness and provenance, not segmentation quality.
    The caller must additionally bind the review to the track manifest hash and
    validate image lineage, binary mask geometry, and duplicate-object aliases.
    Empty predictions still require a decision; they do not establish absence.
    """
    frames = list(expected_frames)
    seeds = tracks.get("seed_ids", [])
    if (
        not frames
        or any(type(i) is not int or i < 0 for i in frames)
        or len(set(frames)) != len(frames)
        or not seeds
        or any(not isinstance(s, str) or not s.strip() for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("empty or invalid temporal review scope")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("missing temporal reviewer")
    predictions = {}
    seen_frames = set()
    for frame in tracks.get("frames", []):
        i = frame["frame_index"]
        if type(i) is not int or i not in frames or i in seen_frames:
            raise ValueError("unexpected or duplicate temporal frame")
        seen_frames.add(i)
        for obj in frame.get("instances", []):
            key = (i, obj["seed_id"])
            if key in predictions or obj["seed_id"] not in seeds:
                raise ValueError("unexpected or duplicate temporal object")
            if type(obj.get("area")) is not int or obj["area"] < 0:
                raise ValueError("invalid temporal prediction area")
            predictions[key] = (frame, obj)
    expected = {(i, s) for i in frames for s in seeds}
    if set(predictions) != expected or seen_frames != set(frames):
        raise ValueError("incomplete temporal prediction scope")

    verified = set()

    def verify_file(path, digest):
        if not isinstance(path, str) or not path or not isinstance(digest, str):
            raise ValueError("missing temporal evidence path/hash")
        pair = (path, digest)
        if pair not in verified:
            try:
                actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            except OSError as exc:
                raise ValueError("missing temporal evidence file") from exc
            if actual != digest:
                raise ValueError("changed temporal evidence file")
            verified.add(pair)

    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("missing individual temporal decisions")  # noqa: TRY004
    seen, accepted = set(), set()
    allowed = {
        "accept_draft",
        "visually_absent",
        "reject_identity_switch",
        "defer_mask",
        "missed_visible",
    }
    for row in decisions:
        key = (row.get("frame_index"), row.get("seed_id"))
        if type(key[0]) is not int or key not in expected or key in seen:
            raise ValueError("unexpected or duplicate temporal decision")
        seen.add(key)
        frame, obj = predictions[key]
        if (
            row.get("mask_sha256") != obj.get("mask_sha256")
            or not frame.get("image_sha256")
            or row.get("image_sha256") != frame["image_sha256"]
        ):
            raise ValueError("stale temporal decision provenance")
        verify_file(obj.get("mask_path"), obj.get("mask_sha256"))
        decision = row.get("decision")
        if decision not in allowed:
            raise ValueError("unknown or unreviewed temporal decision")
        if not isinstance(row.get("observations"), str) or not row["observations"].strip():
            raise ValueError("missing individual temporal observations")
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("missing individual visual evidence")
        for item in evidence:
            if key[0] not in item.get("frames", []):
                raise ValueError("visual evidence does not cover decision frame")
            verify_file(item.get("path"), item.get("sha256"))
        present = obj["area"] > 0
        if (decision in {"accept_draft", "reject_identity_switch"} and not present) or (
            decision in {"visually_absent", "missed_visible"} and present
        ):
            raise ValueError("decision contradicts prediction presence")
        if decision == "accept_draft":
            accepted.add(key)
    if seen != expected:
        raise ValueError("incomplete individual temporal review")
    return accepted
