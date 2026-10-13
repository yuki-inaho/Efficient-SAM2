import numpy as np
import pytest

from tools.workflow.sam_crop_prompts import choose_crop, local_prompts


def test_boundary_crop_retains_box_and_pixel_coordinates():
    rect = choose_crop([3000, 2200, 3199, 2399], (3200, 2400))
    assert rect == (2176, 1376, 1024, 1024)
    pts, labs, box = local_prompts(
        [
            {
                "center": [3100, 2300],
                "box": [3000, 2200, 3199, 2399],
                "negative_points": [[2990, 2190]],
            }
        ],
        0,
        rect,
    )
    assert np.array_equal(pts[0], [924, 924])
    assert list(labs) == [1, 0]
    assert np.array_equal(box, [824, 824, 1023, 1023])


def test_close_duplicate_is_not_used_as_negative_but_other_object_is():
    dets = [
        {"center": [400, 400], "box": [350, 350, 450, 450]},
        {"center": [410, 405], "box": [351, 351, 451, 451]},
        {"center": [510, 400], "box": [460, 350, 560, 450]},
    ]
    pts, labs, _ = local_prompts(dets, 0, (0, 0, 1024, 1024))
    assert pts.tolist() == [[400, 400], [510, 400]]
    assert labs.tolist() == [1, 0]


def test_large_object_is_not_silently_clipped_to_small_tile():
    x, y, w, h = choose_crop([300, 300, 2000, 2100], (3200, 2400))
    assert x <= 300 and y <= 300 and x + w > 2000 and y + h > 2100


def test_invalid_box_is_rejected():
    with pytest.raises(ValueError):
        choose_crop([30, 30, 20, 40], (3200, 2400))


def test_verified_negatives_are_not_overridden_by_detector_hypotheses():
    dets = [
        {
            "center": [400, 400],
            "box": [350, 350, 450, 450],
            "verified_negatives_only": True,
            "negative_points": [[490, 400], [490, 400]],
        },
        {"center": [510, 400], "box": [460, 350, 560, 450]},
    ]
    points, labels, _ = local_prompts(dets, 0, (0, 0, 1024, 1024))
    assert points.tolist() == [[400, 400], [490, 400]]
    assert labels.tolist() == [1, 0]


def test_explicit_negative_survives_nearer_automatic_candidates():
    detections = [
        {"center": [400, 400], "box": [350, 350, 450, 450], "negative_points": [[600, 550]]},
        *[{"center": p} for p in [[480, 400], [320, 400], [400, 480], [400, 320], [510, 400]]],
    ]
    points, labels, _ = local_prompts(detections, 0, (0, 0, 1024, 1024))
    assert points[1].tolist() == [600, 550]
    assert len(points) == 6  # positive + explicit + at most four automatic negatives
    assert labels.tolist() == [1, 0, 0, 0, 0, 0]


def test_explicit_negative_at_positive_point_is_rejected():
    with pytest.raises(ValueError, match="conflicting"):
        local_prompts(
            [{"center": [400, 400], "box": [350, 350, 450, 450], "negative_points": [[400, 400]]}],
            0,
            (0, 0, 1024, 1024),
        )


def test_partial_duplicate_inside_larger_candidate_is_not_a_negative():
    detections = [
        {"center": [410, 450], "box": [370, 400, 450, 480]},
        {"center": [410, 350], "box": [320, 250, 500, 500]},
        {"center": [540, 450], "box": [510, 410, 570, 490]},
    ]
    points, labels, _ = local_prompts(detections, 0, (0, 0, 1024, 1024))
    assert points.tolist() == [[410, 450], [540, 450]]
    assert labels.tolist() == [1, 0]


def test_candidate_center_inside_target_box_is_not_an_automatic_negative():
    detections = [
        {"center": [400, 400], "box": [300, 350, 500, 450]},
        {"center": [485, 410], "box": [465, 390, 505, 430]},
    ]
    points, labels, _ = local_prompts(detections, 0, (0, 0, 1024, 1024))
    assert points.tolist() == [[400, 400]]
    assert labels.tolist() == [1]
