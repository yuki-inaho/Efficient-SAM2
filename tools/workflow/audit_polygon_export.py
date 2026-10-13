"""Measure polygon coverage and added pixels against canonical PNG centers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from matplotlib.path import Path as PolygonPath

from tools.workflow.eval_masks import load_mask


def raster_polygons(polygons: list[dict], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    height, width = shape
    for poly in polygons:
        points = np.asarray(poly["vertices"], dtype=float)
        lo = np.maximum(np.floor(points.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.ceil(points.max(axis=0)).astype(int), [width - 1, height - 1])
        if np.any(hi < lo):
            continue
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        centers = np.column_stack([xx.ravel(), yy.ravel()])
        path = PolygonPath(points)
        covered = path.contains_points(centers, radius=1e-7) | path.contains_points(
            centers, radius=-1e-7
        )
        # Matplotlib can exclude a tangent vertex on a self-touching contour.
        # OpenCV's signed point test includes such zero-area contour boundaries.
        contour = points.astype(np.float32)
        indices = np.flatnonzero(~covered)
        for i in indices:
            if cv2.pointPolygonTest(contour, tuple(map(float, centers[i])), False) >= 0:
                covered[i] = True
        result[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1] |= covered.reshape(xx.shape)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--idmap-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frames = json.loads(args.labels.read_text())
    masks = sorted(args.idmap_dir.glob("frame_*_idmap.png"))
    if len(frames) != len(masks):
        raise ValueError("frame count mismatch")
    rows = []
    for frame, path in zip(frames, masks):
        mask = load_mask(path)
        expected = set(int(x) for x in np.unique(mask) if x)
        if expected != {label["id"] for label in frame["labels"]}:
            raise ValueError("export missing/extra object labels: " + frame["name"])
        for label in frame["labels"]:
            truth = mask == label["id"]
            raster = raster_polygons(label["poly2d"], mask.shape)
            rows.append(
                {
                    "frame": frame["name"],
                    "id": label["id"],
                    "png_pixels": int(truth.sum()),
                    "missing_pixels": int((truth & ~raster).sum()),
                    "extra_pixels": int((raster & ~truth).sum()),
                    "has_holes": label["attributes"]["has_holes"],
                }
            )
    summary = {
        key: sum(r[key] for r in rows) for key in ("png_pixels", "missing_pixels", "extra_pixels")
    }
    summary.update(
        frames=len(frames), labels=len(rows), labels_with_holes=sum(r["has_holes"] for r in rows)
    )
    result = {
        "summary": summary,
        "labels": rows,
        "definition": "union of polygons sampled at PNG pixel centers; boundaries included",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
