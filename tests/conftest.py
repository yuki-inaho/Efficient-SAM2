"""Synthetic approval fixtures; these never approve application datasets."""

import json

import numpy as np
import pytest
from PIL import Image

from tools.workflow.annotation_quality_gate import sha256


@pytest.fixture
def approve_dataset():
    def approve(masks, rgb):
        files = sorted(masks.glob("frame_*_idmap.png"))
        images = sorted(rgb.glob("*.jpg"))
        frames = []
        for i, (mask, image) in enumerate(zip(files, images, strict=True)):
            array = np.array(Image.open(mask))
            candidates = []
            for oid in np.unique(array):
                if not oid:
                    continue
                yy, xx = np.where(array == oid)
                j = len(xx) // 2
                candidates.append(
                    {
                        "candidate_id": f"object-{oid}",
                        "center": [int(xx[j]), int(yy[j])],
                    }
                )
            frames.append(
                {
                    "frame_index": i,
                    "source_path": str(image.resolve()),
                    "source_sha256": sha256(image),
                    "width": array.shape[1],
                    "height": array.shape[0],
                    "detections": candidates,
                }
            )
        root = masks.parent
        ref = root / "synthetic_reference.json"
        ref.write_text(json.dumps({"expected_frames": len(frames), "frames": frames}))
        reviews = [
            {
                "frame_index": i,
                "status": "accepted",
                "reviewer": "synthetic test fixture",
                "observations": "Synthetic rectangles with stable IDs, no unresolved objects.",
                "unresolved_items": [],
                "image_sha256": f["source_sha256"],
                "mask_sha256": sha256(files[i]),
                "reference_sha256": sha256(ref),
                "rgb_inventory_complete": True,
                "boundaries_accepted": True,
                "tracks_accepted": True,
                "evidence": [{"path": f["source_path"], "sha256": f["source_sha256"]}],
            }
            for i, f in enumerate(frames)
        ]
        review = root / "synthetic_review.json"
        review.write_text(json.dumps({"frames": reviews}))
        bundle = root / "synthetic_approval.json"
        bundle.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "reference_path": str(ref.resolve()),
                    "reference_sha256": sha256(ref),
                    "review_path": str(review.resolve()),
                    "review_sha256": sha256(review),
                    "masks_dir": str(masks.resolve()),
                }
            )
        )
        return bundle

    return approve
