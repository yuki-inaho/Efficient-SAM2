"""Label continuity and spatial ID-switch candidates against a pseudo reference.

These diagnostics do not constitute human-verified identity ground truth.
"""

from __future__ import annotations

import numpy as np


def evaluate_sequence(predictions: list[np.ndarray], references: list[np.ndarray]) -> dict:
    if not predictions or len(predictions) != len(references):
        raise ValueError("prediction/reference frame counts must be equal and nonzero")
    pairs = continued = switches = ambiguous = comparable = 0
    prev_gt, prev_pred, prev_assignment = set(), set(), {}
    details = []
    for index, (pred, gt) in enumerate(zip(predictions, references)):
        if pred.shape != gt.shape or pred.ndim != 2:
            raise ValueError("prediction/reference shapes must be equal 2D arrays")
        gt_ids = set(int(v) for v in np.unique(gt) if v > 0)
        pred_ids = set(int(v) for v in np.unique(pred) if v > 0)
        assignment = {}
        for gid in gt_ids:
            scores = {}
            ref = gt == gid
            for pid in pred_ids:
                candidate = pred == pid
                intersection = np.count_nonzero(ref & candidate)
                if intersection:
                    scores[pid] = intersection / np.count_nonzero(ref | candidate)
            if scores:
                best = max(scores.values())
                tied = [pid for pid, score in scores.items() if abs(score - best) < 1e-12]
                if len(tied) == 1:
                    assignment[gid] = tied[0]
                else:
                    ambiguous += 1
        common = gt_ids & prev_gt
        pairs += len(common)
        continued += len(common & pred_ids & prev_pred)
        for gid in sorted(common & set(assignment) & set(prev_assignment)):
            comparable += 1
            if assignment[gid] != prev_assignment[gid]:
                switches += 1
                details.append(
                    {
                        "frame": index,
                        "reference_id": gid,
                        "previous_prediction_id": prev_assignment[gid],
                        "prediction_id": assignment[gid],
                    }
                )
        prev_gt, prev_pred, prev_assignment = gt_ids, pred_ids, assignment
    return {
        "frames": len(predictions),
        "reference_pairs": pairs,
        "continued_pairs": continued,
        "continuation_rate": continued / pairs if pairs else None,
        "switch_candidates": switches,
        "comparable_assignment_pairs": comparable,
        "ambiguous_matches": ambiguous,
        "switch_candidate_details": details,
        "reference": "pseudo_reference_not_identity_ground_truth",
        "definitions": {
            "continuation": (
                "same label present in both predictions when reference label is visible "
                "in adjacent frames; not a spatial accuracy score"
            ),
            "switch_candidate": (
                "change in unique maximum positive-IoU prediction ID for a reference ID "
                "in adjacent visible frames"
            ),
            "ties": "excluded from switch comparisons and counted as ambiguous",
        },
    }
