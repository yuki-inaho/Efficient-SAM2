"""Fail-closed loading of individually reviewed segmentation training pairs."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_reviewed_pairs(manifest_path):
    payload = json.loads(Path(manifest_path).read_text())
    if payload.get("status") not in ("REVIEWED_PILOT_ONLY", "REVIEWED_DATASET"):
        raise ValueError("unreviewed dataset")
    seen, pairs = set(), []
    for item in payload["items"]:
        if item.get("review_status") != "accepted_pilot" or not all(
            item.get(k) for k in ("reviewer", "observations")
        ):
            raise ValueError("missing individual visual review")
        name = item["object_id"]
        if name in seen or item.get("split") not in ("train", "val", "test"):
            raise ValueError("duplicate object identity or invalid split")
        seen.add(name)
        for kind in ("image", "mask"):
            path = Path(item[kind + "_path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != item[kind + "_sha256"]:
                raise ValueError("changed image or teacher mask")
        rgb = np.array(Image.open(item["image_path"]).convert("RGB"))
        mask = np.array(Image.open(item["mask_path"]))
        if (
            mask.ndim != 2
            or mask.shape != rgb.shape[:2]
            or not np.isin(mask, [0, 255]).all()
            or not mask.any()
        ):
            raise ValueError("invalid single-object binary teacher")
        target = mask > 0
        points = np.asarray(item["prompt_points"], float)
        labels = np.asarray(item["prompt_labels"])
        h, w = mask.shape
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or labels.shape != points.shape[:1]
            or not np.isfinite(points).all()
        ):
            raise ValueError("invalid point prompt")
        if not np.isin(labels, [0, 1]).all() or not np.any(labels == 1):
            raise ValueError("invalid positive/negative labels")
        p = np.rint(points).astype(int)
        if np.any(p < 0) or np.any(p[:, 0] >= w) or np.any(p[:, 1] >= h):
            raise ValueError("prompt outside crop")
        if not np.array_equal(target[p[:, 1], p[:, 0]], labels.astype(bool)):
            raise ValueError("positive/negative prompt conflicts with reviewed teacher")
        box = np.asarray(item["local_box"], float)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
            raise ValueError("invalid box")
        pairs.append({**item, "rgb": rgb, "target": target})
    if not pairs:
        raise ValueError("empty dataset")
    return pairs
