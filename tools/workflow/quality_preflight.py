"""Recompute annotation approval and bind exported training pixels to its inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .annotation_quality_gate import audit_dataset, sha256


def require_annotation_approval(bundle_path: Path | None) -> dict:
    if bundle_path is None:
        raise ValueError("quality approval bundle is required")
    bundle_path = Path(bundle_path).resolve()
    try:
        bundle = json.loads(bundle_path.read_text())
        if bundle.get("schema_version") != 1:
            raise ValueError("quality approval schema must be 1")
        refpath = Path(bundle["reference_path"]).resolve()
        reviewpath = Path(bundle["review_path"]).resolve()
        masks = Path(bundle["masks_dir"]).resolve()
        if sha256(refpath) != bundle["reference_sha256"]:
            raise ValueError("quality reference changed")
        if sha256(reviewpath) != bundle["review_sha256"]:
            raise ValueError("quality review changed")
        ref = json.loads(refpath.read_text())
        review = json.loads(reviewpath.read_text())
        report = audit_dataset(refpath, masks, review)
        if report["status"] != "PASS":
            raise ValueError(f"quality approval failed: {report['status']}")
        for row in review["frames"]:
            for key in (
                "rgb_inventory_complete",
                "boundaries_accepted",
                "tracks_accepted",
            ):
                if row.get(key) is not True:
                    raise ValueError(f"quality review missing {key}")
            evidence = row.get("evidence", [])
            if not isinstance(evidence, list) or not evidence:
                raise ValueError("quality review missing image evidence")
            for item in evidence:
                if sha256(Path(item["path"])) != item["sha256"]:
                    raise ValueError("quality review evidence changed")
        frames = sorted(ref["frames"], key=lambda r: r["frame_index"])
        expected = {masks / f"frame_{r['frame_index']:04d}_idmap.png" for r in frames}
        if set(masks.glob("frame_*_idmap.png")) != expected:
            raise ValueError("quality idmap frame set mismatch")
        for row in frames:
            if Image.open(row["source_path"]).size != (row["width"], row["height"]):
                raise ValueError("quality RGB dimensions mismatch")
        return {
            "approval_path": str(bundle_path),
            "approval_sha256": sha256(bundle_path),
            "reference_path": str(refpath),
            "reference_sha256": sha256(refpath),
            "review_sha256": sha256(reviewpath),
            "masks_dir": str(masks),
            "frames": frames,
        }
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid or missing quality approval: {exc}") from exc


def validate_export_sources(
    approval: dict, idmaps: list[Path], rgbs: list[Path]
) -> None:
    frames = approval["frames"]
    if len(idmaps) != len(frames) or len(rgbs) != len(frames):
        raise ValueError("quality source count mismatch")
    for row, mask, rgb in zip(frames, idmaps, rgbs, strict=True):
        expected = (
            Path(approval["masks_dir"]) / f"frame_{row['frame_index']:04d}_idmap.png"
        )
        if (
            mask.resolve() != expected
            or rgb.resolve() != Path(row["source_path"]).resolve()
        ):
            raise ValueError("quality export sources differ from approved inputs")
        if sha256(rgb) != row["source_sha256"]:
            raise ValueError("quality export RGB changed")


def verify_training_inputs(data: dict) -> dict:
    """Check every listed dataset before a run directory, GPU wait or trainer launch."""
    try:
        rgb_base = Path(data["base_video_dir"]).resolve()
        ann_base = Path(data["input_mask_dir"]).resolve()
        if rgb_base.parent != ann_base.parent:
            raise ValueError("quality dataset roots differ")
        names = Path(data["video_list_file"]).read_text().splitlines()
        names = [n.strip() for n in names if n.strip() and not n.startswith("__pad_")]
        if not names or len(names) != len(set(names)):
            raise ValueError("quality dataset list empty or duplicated")
        results = []
        for name in names:
            if Path(name).name != name:
                raise ValueError("quality video name must be a single component")
            mp = rgb_base.parent / f"manifest_{name}.json"
            manifest = json.loads(mp.read_text())
            receipt = manifest["quality_receipt"]
            approval = require_annotation_approval(Path(receipt["approval_path"]))
            if any(
                receipt.get(k) != approval[k]
                for k in (
                    "approval_sha256",
                    "reference_sha256",
                    "review_sha256",
                )
            ):
                raise ValueError("quality export receipt is stale")
            indices = receipt["frame_indices"]
            n = len(approval["frames"])
            if (
                not indices
                or any(type(i) is not int or i < 0 or i >= n for i in indices)
                or indices != list(range(indices[0], indices[-1] + 1))
            ):
                raise ValueError("quality export frame range invalid")
            files = manifest["files"]
            all_ids = receipt["object_ids"]
            if (
                not isinstance(all_ids, list)
                or any(type(oid) is not int or oid <= 0 for oid in all_ids)
                or len(all_ids) != len(set(all_ids))
            ):
                raise ValueError("quality export object IDs invalid")
            observed_ids = set()
            if manifest["video_name"] != name or manifest["frames"] != len(indices):
                raise ValueError("quality export frame count mismatch")
            expected_rgb, expected_ann = set(), set()
            expected_keys = []
            for index in indices:
                ref = approval["frames"][index]
                source = Path(ref["source_path"])
                key = (
                    str(len(expected_keys))
                    if manifest["integer_names"]
                    else source.stem
                )
                expected_keys.append(key)
                record = files[key]
                maskpath = Path(approval["masks_dir"]) / f"frame_{index:04d}_idmap.png"
                if (
                    record["source_idmap"] != maskpath.name
                    or record["source_rgb"] != source.name
                ):
                    raise ValueError("quality source mapping changed")
                rgb = rgb_base / name / f"{key}.jpg"
                expected_rgb.add(rgb)
                if sha256(rgb) != ref["source_sha256"]:
                    raise ValueError(
                        "quality training RGB differs from approved source"
                    )
                mask = np.array(Image.open(maskpath))
                ids = sorted(int(v) for v in np.unique(mask) if v)
                observed_ids.update(ids)
                if manifest["layout"] == "packed":
                    paths = [(ann_base / name / f"{key}.png", mask)]
                elif manifest["layout"] == "per_object":
                    # Include zero masks for objects absent in this frame.
                    if not set(ids).issubset(all_ids):
                        raise ValueError("quality visible IDs removed from export")
                    paths = [
                        (
                            ann_base / name / f"{oid:03d}" / f"{key}.png",
                            (mask == oid).astype(np.uint8) * 255,
                        )
                        for oid in all_ids
                    ]
                else:
                    raise ValueError("quality unknown annotation layout")
                for path, expected in paths:
                    expected_ann.add(path)
                    if not np.array_equal(np.array(Image.open(path)), expected):
                        raise ValueError(
                            "quality training mask differs from approved source"
                        )
            if set(files) != set(expected_keys):
                raise ValueError("quality export frame records mismatch")
            if observed_ids != set(all_ids):
                raise ValueError(
                    "quality export object IDs differ from visible objects"
                )
            if set((rgb_base / name).glob("*")) != expected_rgb:
                raise ValueError("quality training RGB file set mismatch")
            if set((ann_base / name).rglob("*.png")) != expected_ann:
                raise ValueError("quality training mask file set mismatch")
            results.append(
                {
                    "video_name": name,
                    "manifest_sha256": sha256(mp),
                    "approval_sha256": approval["approval_sha256"],
                }
            )
        return {"status": "VERIFIED_CURRENT_INPUTS", "datasets": results}
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid or missing quality training provenance: {exc}"
        ) from exc
