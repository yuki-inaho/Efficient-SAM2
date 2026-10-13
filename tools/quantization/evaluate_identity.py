"""Sampled semantic-ID audit using fixed-gate Hungarian IoU matching.

Uses the same public manual-mask training fixture as evaluate_cpu.py. Emitted
integer IDs are tested against manual object identities, not assumed correct.
Four sampled frames do not establish whole-video identity or occlusion reentry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from pycocotools import mask as coco_mask
from scipy.optimize import linear_sum_assignment

from tools.quantization.evaluate_cpu import external_path, sha256


def gated_assignment(iou: np.ndarray, gate: float = 0.5):
    """Maximum-cardinality valid assignment, then maximum summed IoU.

    Dummy rows/columns allow unmatched objects. Invalid edges cannot consume a
    real match, and the cardinality bonus exceeds any possible IoU tie-break sum.
    """
    rows, cols = iou.shape
    score = np.zeros((rows + cols, rows + cols), dtype=np.float64)
    score[:rows, :cols] = np.where(iou >= gate, max(rows, cols) + 1 + iou, -1.0)
    a, b = linear_sum_assignment(score, maximize=True)
    return [
        (int(r), int(c))
        for r, c in zip(a, b)
        if r < rows and c < cols and iou[r, c] >= gate
    ]


def evaluate(report_dir: Path):
    report = json.loads((report_dir / "report.json").read_text())
    protocol = report["protocol"]
    annotation = ROOT / "sav_dataset/example/sav_000001_manual.json"
    if protocol["annotations_sha256"] != sha256(annotation):
        raise ValueError("Manual annotation checksum differs from evaluation report")
    manual = json.loads(annotation.read_text())
    object_ids = protocol["object_ids"]
    last_match = {i: {"manual_id": i, "frame": 0} for i in object_ids}
    switches = []
    wrong_initial = []
    frame_reports = []
    for frame in protocol["evaluated_raw_frame_ids"]:
        predictions, ground_truth = [], []
        for object_id in object_ids:
            logits = np.load(
                report_dir / f"frame_{frame:05d}_object_{object_id:03d}.npy",
                allow_pickle=False,
            )
            if not np.isfinite(logits).all():
                raise ValueError("Non-finite prediction logits")
            predictions.append(logits > 0)
            ground_truth.append(
                coco_mask.decode(manual["masklet"][frame // 4][object_id]).astype(bool)
            )
        iou = np.zeros((len(object_ids), len(object_ids)), dtype=np.float64)
        for row, pred in enumerate(predictions):
            for col, gt in enumerate(ground_truth):
                if pred.shape != gt.shape:
                    raise ValueError("Prediction and manual mask shape mismatch")
                union = np.logical_or(pred, gt).sum()
                iou[row, col] = np.logical_and(pred, gt).sum() / union if union else 0.0
        assignments = gated_assignment(iou, gate=0.5)
        matched_pred = {row for row, _ in assignments}
        matched_gt = {col for _, col in assignments}
        matches = []
        for row, col in assignments:
            pred_id, gt_id = object_ids[row], object_ids[col]
            matches.append(
                {
                    "predicted_id": pred_id,
                    "manual_id": gt_id,
                    "iou": float(iou[row, col]),
                }
            )
            if pred_id in last_match and last_match[pred_id]["manual_id"] != gt_id:
                switches.append(
                    {
                        "frame": frame,
                        "predicted_id": pred_id,
                        "previous_frame": last_match[pred_id]["frame"],
                        "from_manual_id": last_match[pred_id]["manual_id"],
                        "to_manual_id": gt_id,
                    }
                )
            if pred_id != gt_id:
                wrong_initial.append(
                    {"frame": frame, "predicted_id": pred_id, "manual_id": gt_id}
                )
            last_match[pred_id] = {"manual_id": gt_id, "frame": frame}
        duplicate_candidates = [
            {
                "manual_id": object_ids[col],
                "predicted_ids": [
                    object_ids[row]
                    for row in range(len(object_ids))
                    if iou[row, col] >= 0.5
                ],
            }
            for col in range(len(object_ids))
            if np.count_nonzero(iou[:, col] >= 0.5) > 1
        ]
        frame_reports.append(
            {
                "frame": frame,
                "prediction_rows": object_ids,
                "manual_columns": object_ids,
                "all_pair_ious": iou.tolist(),
                "matches": matches,
                "unmatched_prediction_ids": [
                    object_ids[r]
                    for r in range(len(object_ids))
                    if r not in matched_pred and predictions[r].any()
                ],
                "unmatched_manual_ids": [
                    object_ids[c]
                    for c in range(len(object_ids))
                    if c not in matched_gt and ground_truth[c].any()
                ],
                "empty_prediction_ids": [
                    object_ids[r]
                    for r in range(len(object_ids))
                    if not predictions[r].any()
                ],
                "absent_manual_ids": [
                    object_ids[c]
                    for c in range(len(object_ids))
                    if not ground_truth[c].any()
                ],
                "duplicate_candidates": duplicate_candidates,
            }
        )
    result = {
        "scope": "public SA-V TRAIN sample; only the four predeclared evaluated frames",
        "gate": 0.5,
        "assignment": "Hungarian, maximum valid-match count then summed IoU; dummy unmatched nodes",
        "switch_definition": "A predicted ID changes its manual identity from the known frame-0 prompt mapping or a prior sampled matched observation; unmatched gaps remain visible",
        "initial_id_mapping": {str(i): i for i in object_ids},
        "untracked_manual_ids": sorted(set(manual["masklet_id"]) - set(object_ids)),
        "sampled_id_switch_count": len(switches),
        "sampled_id_switches": switches,
        "wrong_initial_mapping_count": len(wrong_initial),
        "wrong_initial_mapping": wrong_initial,
        "unmatched_prediction_count": sum(
            len(f["unmatched_prediction_ids"]) for f in frame_reports
        ),
        "unmatched_manual_count": sum(
            len(f["unmatched_manual_ids"]) for f in frame_reports
        ),
        "duplicate_candidate_count": sum(
            len(f["duplicate_candidates"]) for f in frame_reports
        ),
        "occlusion_reentry_status": "not established by this short sampled fixture",
        "frames": frame_reports,
    }
    (report_dir / "identity_report.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    result = evaluate(external_path(args.report_dir))
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}), flush=True)


if __name__ == "__main__":
    main()
