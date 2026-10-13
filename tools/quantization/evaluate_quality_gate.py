"""Validate paired evidence and apply the preregistered TRAIN-smoke gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from tools.quantization.evaluate_cpu import external_path, sha256


def quality_gate(report: dict, reference: dict, reference_sha256: str | None = None):
    """Derive metrics from rows; duplicated summaries are validation targets."""
    try:
        protocol = report["protocol"]
        expected = {
            (f, o)
            for f in protocol["evaluated_raw_frame_ids"]
            for o in protocol["object_ids"]
        }
        agreement = report["agreement"]
        scores = report["per_object_metrics"]
        reference_scores = reference["per_object_metrics"]
        summary = report["agreement_summary"]
        delta = report["paired_mean_metric_delta"]
        ious = np.asarray([r["mask_iou"] for r in agreement], dtype=np.float64)
        empty = [r["reference_nonempty_to_empty"] for r in agreement]
        derived = {
            "mean_mask_iou": float(np.mean(ious)),
            "p5_mask_iou": float(np.percentile(ious, 5)),
            "min_mask_iou": float(np.min(ious)),
            "catastrophic_empty_rate": float(np.mean(empty)),
            "mean_pixel_disagreement": float(
                np.mean([r["pixel_disagreement"] for r in agreement])
            ),
            "mean_logit_mae": float(np.mean([r["logit_mae"] for r in agreement])),
        }
        candidate_mean = {
            k: float(np.mean([r[k] for r in scores])) for k in ("j", "f", "j_and_f")
        }
        reference_mean = {
            k: float(np.mean([r[k] for r in reference_scores])) for k in candidate_mean
        }
        derived_delta = {
            k: candidate_mean[k] - reference_mean[k] for k in candidate_mean
        }
        all_values = [
            value
            for rows in [scores, reference_scores, agreement]
            for row in rows
            for key, value in row.items()
            if key not in {"frame", "object"}
        ]
        all_values += list(summary.values()) + list(delta.values())
        finite = all(
            isinstance(v, (int, float, bool)) and math.isfinite(v) for v in all_values
        )

        def complete(rows):
            return (
                len(rows) == len(expected)
                and {(r["frame"], r["object"]) for r in rows} == expected
            )

        def close(a, b):
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)

        checks = {
            "fixed_protocol_v1": protocol["evaluated_raw_frame_ids"] == [4, 8, 12, 16]
            and protocol["object_ids"] == [0, 1, 2, 4]
            and protocol.get("frame_count") == 17
            and protocol.get("image_size") == 1024
            and protocol.get("independent_object_tracks") is True
            and protocol.get("efficient_swr_smr") is True,
            "reference_protocol_matches": reference["protocol"] == protocol,
            "complete_paired_cases": complete(agreement),
            "complete_manual_cases": complete(scores),
            "complete_reference_cases": complete(reference_scores),
            "finite_metrics": finite,
            "iou_values_in_range": bool(np.all((ious >= 0) & (ious <= 1))),
            "manual_metrics_in_range": all(
                0 <= r[k] <= 1
                for rows in [scores, reference_scores]
                for r in rows
                for k in ("j", "f", "j_and_f")
            ),
            "empty_flags_are_boolean": all(isinstance(v, bool) for v in empty),
            "summary_consistent_with_rows": all(
                close(summary[k], v) for k, v in derived.items()
            ),
            "candidate_means_consistent": all(
                close(report["mean_metrics"][k], v) for k, v in candidate_mean.items()
            ),
            "reference_means_consistent": all(
                close(reference["mean_metrics"][k], v)
                for k, v in reference_mean.items()
            ),
            "delta_consistent_with_paired_reference": all(
                close(delta[k], v) for k, v in derived_delta.items()
            ),
            "mean_mask_iou_ge_0_98": derived["mean_mask_iou"] >= 0.98,
            "p5_mask_iou_ge_0_95": derived["p5_mask_iou"] >= 0.95,
            "min_mask_iou_ge_0_90": derived["min_mask_iou"] >= 0.90,
            "mean_j_drop_le_0_01": derived_delta["j"] >= -0.01 - 1e-12,
            "mean_f_drop_le_0_01": derived_delta["f"] >= -0.01 - 1e-12,
            "no_catastrophic_empty_flips": derived["catastrophic_empty_rate"] == 0.0,
        }
    except (KeyError, TypeError, ValueError, IndexError) as error:
        return {
            "passed": False,
            "checks": {"well_formed_paired_evidence": False},
            "error": str(error),
        }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "derived_agreement_summary": derived,
        "derived_paired_mean_metric_delta": derived_delta,
        "frozen_protocol_sha256": hashlib.sha256(
            (json.dumps(protocol, indent=2) + "\n").encode()
        ).hexdigest(),
        "reference_report_sha256": reference_sha256,
        "thresholds": {
            "mean_mask_iou_min": 0.98,
            "p5_mask_iou_min": 0.95,
            "min_mask_iou_min": 0.90,
            "mean_j_delta_min": -0.01,
            "mean_f_delta_min": -0.01,
            "catastrophic_empty_rate_max": 0.0,
        },
        "threshold_origin": "Agreed in the workdoc before quantized evaluation; this result file is produced after evaluation",
        "scope": "Only the fixed paired public TRAIN-smoke fixture; not general accuracy, downstream quality, native runtime parity, or video ID stability",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()
    report_dir = external_path(args.report_dir)
    reference_path = external_path(args.reference_dir) / "report.json"
    result = quality_gate(
        json.loads((report_dir / "report.json").read_text()),
        json.loads(reference_path.read_text()),
        sha256(reference_path),
    )
    (report_dir / "quality_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
