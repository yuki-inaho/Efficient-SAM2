"""Fail closed on missing/merged instances or absent artifact-bound visual review.

Center matching is a necessary geometric check, not proof of segmentation quality.
The independent RGB review must cover detector misses and mask boundaries as well.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_instances(idmap: np.ndarray, candidates: list[dict]) -> dict:
    a = np.asarray(idmap)
    if a.ndim != 2 or a.dtype.kind not in "ui" or (a < 0).any():
        raise ValueError("idmap must be a nonnegative 2D integer array")
    ids = set(int(i) for i in np.unique(a) if i)
    seen = set()
    assigned = defaultdict(list)
    matches, missing = {}, []
    for candidate in candidates:
        cid = candidate["candidate_id"]
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError("candidate IDs must be nonempty and unique")
        seen.add(cid)
        xy = np.asarray(candidate["center"], dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("candidate center must be finite x,y")
        if not (0 <= xy[0] < a.shape[1] and 0 <= xy[1] < a.shape[0]):
            raise ValueError("candidate center outside image")
        # A center may be within the continuous image but round outside its last pixel.
        x, y = np.minimum(np.rint(xy).astype(int), [a.shape[1] - 1, a.shape[0] - 1])
        oid = int(a[y, x])
        matches[cid] = oid
        if oid:
            assigned[oid].append(cid)
        else:
            missing.append(cid)
    merged = {str(i): names for i, names in assigned.items() if len(names) > 1}
    unreferenced = sorted(ids - set(assigned))
    return {
        "passed": bool(candidates) and not (missing or merged or unreferenced),
        "candidate_count": len(candidates),
        "mask_count": len(ids),
        "matches": matches,
        "missing": missing,
        "merged": merged,
        "unreferenced": unreferenced,
    }


def audit_dataset(reference_path: Path, masks_dir: Path, review: dict | None) -> dict:
    reference_path, masks_dir = Path(reference_path), Path(masks_dir)
    ref = json.loads(reference_path.read_text())
    errors, rows, unreviewed = [], [], []
    n = ref.get("expected_frames")
    frames = ref.get("frames", [])
    if not isinstance(n, int) or n <= 0:
        errors.append("invalid expected_frames")
    elif sorted(f.get("frame_index", -1) for f in frames) != list(range(n)):
        errors.append("missing or duplicate reference frames")
    if not frames:
        errors.append("empty reference")
    review_frames = (review or {}).get("frames", [])
    reviewed = {f.get("frame_index"): f for f in review_frames}
    if len(reviewed) != len(review_frames):
        errors.append("duplicate review frames")
    if review is not None and set(reviewed) != set(range(n or 0)):
        errors.append("missing or extra review frames")
    ref_sha = sha256(reference_path)
    for frame in frames:
        i = frame["frame_index"]
        rgb = Path(frame["source_path"])
        mask = masks_dir / f"frame_{i:04d}_idmap.png"
        if not rgb.is_file() or sha256(rgb) != frame.get("source_sha256"):
            errors.append(f"frame {i}: missing or changed RGB")
            continue
        if not mask.is_file():
            errors.append(f"frame {i}: missing mask")
            continue
        a = np.asarray(Image.open(mask))
        if a.shape != (frame["height"], frame["width"]):
            errors.append(f"frame {i}: mask shape mismatch")
            continue
        try:
            result = inspect_instances(a, frame["detections"])
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"frame {i}: {exc}")
            continue
        result["frame_index"] = i
        result["mask_sha256"] = sha256(mask)
        rows.append(result)
        if not result["passed"]:
            errors.append(f"frame {i}: instance correspondence failed")
        r = reviewed.get(i, {})
        proof = (
            r.get("status") == "accepted"
            and isinstance(r.get("reviewer"), str)
            and bool(r["reviewer"].strip())
            and isinstance(r.get("observations"), str)
            and bool(r["observations"].strip())
            and r.get("unresolved_items") == []
            and r.get("image_sha256") == frame["source_sha256"]
            and r.get("mask_sha256") == result["mask_sha256"]
            and r.get("reference_sha256") == ref_sha
        )
        if not proof:
            unreviewed.append(i)
    return {
        "status": "FAIL" if errors else "UNREVIEWED" if unreviewed else "PASS",
        "reference_sha256": ref_sha,
        "errors": errors,
        "unreviewed_frames": unreviewed,
        "totals": {
            "candidates": sum(r["candidate_count"] for r in rows),
            "masks": sum(r["mask_count"] for r in rows),
            "missing_candidates": sum(len(r["missing"]) for r in rows),
            "merged_masks": sum(len(r["merged"]) for r in rows),
            "unreferenced_masks": sum(len(r["unreferenced"]) for r in rows),
        },
        "frames": rows,
        "limitation": "Center correspondence does not establish boundary or RGB completeness.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--masks-dir", type=Path, required=True)
    p.add_argument("--review", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    review = json.loads(args.review.read_text()) if args.review else None
    report = audit_dataset(args.reference, args.masks_dir, review)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "totals")}))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
