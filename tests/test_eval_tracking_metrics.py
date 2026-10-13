import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from tools.workflow.eval_tracking_metrics import evaluate_sequence  # noqa: E402


def test_id_swap_is_flagged_despite_label_presence():
    gt = np.array([[1, 1, 2, 2]], dtype=np.uint8)
    swapped = np.array([[2, 2, 1, 1]], dtype=np.uint8)
    r = evaluate_sequence([gt, swapped], [gt, gt])
    assert r["continuation_rate"] == 1.0
    assert r["switch_candidates"] == 2


def test_missing_id_counts_against_continuation():
    gt = np.array([[1, 2]], dtype=np.uint8)
    pred = np.array([[1, 0]], dtype=np.uint8)
    r = evaluate_sequence([gt, pred], [gt, gt])
    assert r["continuation_rate"] == 0.5
    assert r["switch_candidates"] == 0


def test_empty_reference_has_no_denominator():
    z = np.zeros((2, 2), dtype=np.uint8)
    r = evaluate_sequence([z, z], [z, z])
    assert r["continuation_rate"] is None
    assert r["reference_pairs"] == 0


def test_shapes_and_missing_frames_fail():
    z = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        evaluate_sequence([z], [z, z])
    with pytest.raises(ValueError):
        evaluate_sequence([z], [z[:, :1]])


def test_tied_match_is_not_a_switch():
    gt = np.ones((1, 4), dtype=np.uint8)
    pred = np.array([[1, 1, 2, 2]], dtype=np.uint8)
    r = evaluate_sequence([gt, pred], [gt, gt])
    assert r["ambiguous_matches"] == 1
    assert r["switch_candidates"] == 0
