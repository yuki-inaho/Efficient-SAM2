from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sam2.sam2_video_predictor import SAM2VideoPredictor
from tools.quantization.evaluate_cpu import (
    compare_logits,
    external_path,
    mask_metrics,
    ROOT,
)
from tools.quantization.evaluate_identity import gated_assignment
from tools.quantization.evaluate_quality_gate import quality_gate


def test_memory_info_respects_cpu_device():
    predictor = SimpleNamespace(
        device=torch.device("cpu"),
        image_encoder=SimpleNamespace(trunk=SimpleNamespace(window_spec=(8, 4, 14, 7))),
        num_maskmem=7,
        num_frame_to_prune=2,
    )
    SAM2VideoPredictor.init_memory_info(predictor, enable_MeP_info=True)
    assert predictor.uniform_mode.device.type == "cpu"
    assert predictor.uniform_mode.shape == (4096,)
    assert predictor.uniform_mode.sum().item() == 64


@pytest.mark.parametrize("foreground", [False, True])
def test_identical_masks_have_perfect_metrics(foreground):
    mask = np.full((20, 24), foreground)
    assert mask_metrics(mask, mask) == {"j": 1.0, "f": 1.0, "j_and_f": 1.0}


def test_disjoint_masks_have_zero_j():
    a = np.zeros((40, 40), dtype=bool)
    b = a.copy()
    a[2:8, 2:8] = True
    b[30:36, 30:36] = True
    assert mask_metrics(a, b)["j"] == 0.0
    assert mask_metrics(a, b)["f"] == 0.0


def test_logit_comparison_reports_empty_flip():
    current = -np.ones((2, 3), dtype=np.float32)
    reference = np.ones((2, 3), dtype=np.float32)
    metrics = compare_logits(current, reference)
    assert metrics["mask_iou"] == 0.0
    assert metrics["pixel_disagreement"] == 1.0
    assert metrics["logit_mae"] == 2.0
    assert metrics["reference_nonempty_to_empty"] is True


def test_logit_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        compare_logits(np.ones((2, 3)), np.ones((3, 2)))


def test_outputs_cannot_be_written_in_checkout():
    with pytest.raises(ValueError, match="outside"):
        external_path(ROOT / "outputs")
    assert external_path("/tmp/evaluation") == Path("/tmp/evaluation")


def test_identity_assignment_is_semantic_not_emitted_id():
    assert gated_assignment(np.array([[0.01, 0.9], [0.95, 0.02]])) == [(0, 1), (1, 0)]


def test_identity_gate_allows_unmatched_and_duplicate_candidates():
    result = gated_assignment(np.array([[0.9, 0.1], [0.8, 0.2]]))
    assert result == [(0, 0)]


def test_identity_gate_is_inclusive_and_does_not_match_empty_pairs():
    assert gated_assignment(np.array([[0.5, 0.0], [0.0, 0.49]])) == [(0, 0)]


def passing_gate_report():
    from copy import deepcopy

    pairs = [(frame, obj) for frame in [4, 8, 12, 16] for obj in [0, 1, 2, 4]]
    protocol = {
        "evaluated_raw_frame_ids": [4, 8, 12, 16],
        "object_ids": [0, 1, 2, 4],
        "frame_count": 17,
        "image_size": 1024,
        "independent_object_tracks": True,
        "efficient_swr_smr": True,
    }
    reference = {
        "protocol": deepcopy(protocol),
        "per_object_metrics": [
            {"frame": frame, "object": obj, "j": 1.0, "f": 1.0, "j_and_f": 1.0}
            for frame, obj in pairs
        ],
        "mean_metrics": {"j": 1.0, "f": 1.0, "j_and_f": 1.0},
    }
    report = {
        "protocol": protocol,
        "agreement": [
            {
                "frame": frame,
                "object": obj,
                "mask_iou": 0.98,
                "reference_nonempty_to_empty": False,
                "pixel_disagreement": 0.0,
                "logit_mae": 0.0,
            }
            for frame, obj in pairs
        ],
        "per_object_metrics": [
            {"frame": frame, "object": obj, "j": 0.99, "f": 0.99, "j_and_f": 0.99}
            for frame, obj in pairs
        ],
        "mean_metrics": {"j": 0.99, "f": 0.99, "j_and_f": 0.99},
        "agreement_summary": {
            "mean_mask_iou": 0.98,
            "p5_mask_iou": 0.98,
            "min_mask_iou": 0.98,
            "catastrophic_empty_rate": 0.0,
            "mean_pixel_disagreement": 0.0,
            "mean_logit_mae": 0.0,
        },
        "paired_mean_metric_delta": {"j": -0.01, "f": -0.01, "j_and_f": -0.01},
    }
    return report, reference


def test_quality_gate_accepts_consistent_rows_at_mean_boundary():
    report, reference = passing_gate_report()
    assert quality_gate(report, reference)["passed"] is True


def test_quality_gate_rejects_incomplete_subset_and_nan():
    report, reference = passing_gate_report()
    report["agreement"].pop()
    assert quality_gate(report, reference)["passed"] is False
    report, reference = passing_gate_report()
    report["agreement_summary"]["mean_mask_iou"] = float("nan")
    assert quality_gate(report, reference)["passed"] is False
    report, reference = passing_gate_report()
    report["protocol"]["image_size"] = 512
    assert quality_gate(report, reference)["checks"]["fixed_protocol_v1"] is False


def test_quality_gate_rejects_forged_summary_and_out_of_range_iou():
    report, reference = passing_gate_report()
    report["agreement_summary"]["mean_mask_iou"] = 1.0
    assert (
        quality_gate(report, reference)["checks"]["summary_consistent_with_rows"]
        is False
    )
    report, reference = passing_gate_report()
    report["agreement"][0]["mask_iou"] = 1.1
    assert quality_gate(report, reference)["checks"]["iou_values_in_range"] is False


def test_quality_gate_derives_delta_from_reference_rows():
    report, reference = passing_gate_report()
    reference["per_object_metrics"][0]["j"] = 0.2
    assert (
        quality_gate(report, reference)["checks"][
            "delta_consistent_with_paired_reference"
        ]
        is False
    )
