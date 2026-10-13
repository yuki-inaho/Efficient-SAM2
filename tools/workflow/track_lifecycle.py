"""Causal per-frame visibility state, independent of depth filtering."""

from __future__ import annotations


def add_lifecycle_states(frames: list[dict], gone_gap: int = 25) -> dict[str, str]:
    if gone_gap < 0:
        raise ValueError("gone_gap must be nonnegative")
    indices = [f["frame"] for f in frames]
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("frame indices must be strictly increasing")
    last_seen: dict[str, int] = {}
    states: dict[str, str] = {}
    for frame in frames:
        index = frame["frame"]
        visible = {str(oid) for oid, info in frame["objects"].items() if info["area"] > 0}
        for oid in visible:
            last_seen[oid] = index
        states = {
            oid: "visible" if oid in visible else "lost" if index - last <= gone_gap else "gone"
            for oid, last in sorted(last_seen.items(), key=lambda item: int(item[0]))
        }
        frame["states"] = states
    return states.copy()
