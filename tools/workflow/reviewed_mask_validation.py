"""Verify a reviewed replacement mask before recomposing an image."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .sam_crop_prompts import prompt_coverage


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_replacement(new, old, image_wh):
    """Allow reviewed crop expansion while retaining full-image identity."""
    if sha(new["mask_path"]) != new["mask_sha256"]:
        raise ValueError("changed replacement mask")
    crop = np.asarray(new["crop"])
    if crop.shape != (4,) or not np.isfinite(crop).all() or not np.equal(crop, np.rint(crop)).all():
        raise ValueError("invalid replacement crop")
    x, y, w, h = crop.astype(int)
    width, height = image_wh
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise ValueError("replacement crop outside image")
    raw = np.array(Image.open(new["mask_path"]))
    if raw.shape != (h, w) or not np.isin(raw, [0, 255]).all():
        raise ValueError("invalid replacement mask dimensions/values")
    if new["candidate_id"] != old["candidate_id"]:
        raise ValueError("replacement changed target identity")
    points = np.asarray(new["points"], float)
    if not len(points) or not np.allclose(points[0] + [x, y], new["center"], atol=1e-3, rtol=0):
        raise ValueError("replacement prompt origin mismatch")
    coverage = prompt_coverage(raw > 0, points, new["labels"])
    if coverage["positive_violations"] or coverage["negative_violations"]:
        raise ValueError("replacement violates point constraints")
    if any(new.get(key) != value for key, value in coverage.items()):
        raise ValueError("stale replacement prompt metrics")
    if np.count_nonzero(raw) != new["area"]:
        raise ValueError("stale replacement area")

