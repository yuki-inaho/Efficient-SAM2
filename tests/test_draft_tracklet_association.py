import numpy as np
from tools.workflow.draft_tracklet_association import associate, deduplicate


def rect(x0, y0, x1, y1):
    m = np.zeros((60, 80), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_duplicate_not_adjacent_fruit():
    a = rect(5, 5, 20, 20)
    b = rect(20, 5, 35, 20)
    kept, suppressed = deduplicate(
        [
            {"candidate_id": "a", "small": a},
            {"candidate_id": "dup", "small": a.copy()},
            {"candidate_id": "b", "small": b},
        ]
    )
    assert [c["candidate_id"] for c in kept] == ["a", "b"]
    assert suppressed[0]["candidate_id"] == "dup"


def test_one_to_one_and_persistent_ids_not_input_order():
    a = rect(5, 5, 20, 20)
    b = rect(20, 5, 35, 20)
    matches, _ = associate([{"small": b}, {"small": a}, {"small": a.copy()}], {78: a, 12: b})
    assert matches == {0: 12, 1: 78}


def test_cropped_overlap_equals_full_image_overlap():
    from tools.workflow.draft_tracklet_association import geometry, gated_overlap

    rng = np.random.default_rng(37)
    for _ in range(100):
        x, y, u, v = rng.integers(0, 40, size=4)
        a = rect(x, y, x + 15, y + 15)
        b = rect(u, v, u + 20, v + 20)
        ga, gb = geometry(a), geometry(b)
        fast = gated_overlap(a, b, ga, gb)
        inter = int(np.count_nonzero(a & b))
        distance = np.hypot(ga[5] - gb[5], ga[6] - gb[6])
        if inter and distance <= 0.75 * (np.sqrt(a.sum()) + np.sqrt(b.sum())):
            assert fast == (inter / np.count_nonzero(a | b), inter / min(a.sum(), b.sum()))
        else:
            assert fast is None or fast[0] == 0
