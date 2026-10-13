import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.workflow.track_lifecycle import add_lifecycle_states  # noqa: E402


def test_causal_loss_boundary_and_reappearance():
    frames = [{"frame": i, "objects": {"1": {"area": 3}} if i in (1, 5) else {}} for i in range(6)]
    states = add_lifecycle_states(frames, gone_gap=2)
    assert frames[0]["states"] == {}
    assert [f["states"].get("1") for f in frames] == [
        None,
        "visible",
        "lost",
        "lost",
        "gone",
        "visible",
    ]
    assert states == {"1": "visible"}


def test_empty_zero_area_and_invalid_order():
    assert add_lifecycle_states([], gone_gap=2) == {}
    frames = [{"frame": 0, "objects": {"2": {"area": 0}}}]
    assert add_lifecycle_states(frames, gone_gap=2) == {}
    with pytest.raises(ValueError):
        add_lifecycle_states([{"frame": 1, "objects": {}}, {"frame": 0, "objects": {}}])
