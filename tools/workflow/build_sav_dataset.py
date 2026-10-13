"""Export reviewed uint16 instance maps as SA-V per-object or packed masks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from .annotation_quality_gate import sha256 as sha256_file
from .quality_preflight import require_annotation_approval, validate_export_sources


def make_palette() -> list[int]:
    palette: list[int] = []
    for i in range(256):
        palette.extend([(i * 37) % 256, (i * 91) % 256, (i * 53) % 256])
    return palette


def write_ann_png(path: Path, mask_u8) -> None:
    if mask_u8.dtype.name != "uint8" or mask_u8.ndim != 2:
        raise ValueError(
            f"annotation mask must be uint8 2D, got {mask_u8.dtype}/{mask_u8.shape}"
        )
    img = Image.fromarray(mask_u8)
    img.putpalette(make_palette())
    img.save(path)


def build_sav_dataset(
    idmap_dir: Path,
    rgb_dir: Path,
    out_root: Path,
    video_name: str,
    force: bool = False,
    frame_range: tuple[int, int] | None = None,
    integer_names: bool = True,
    layout: str = "per_object",
    restrict_to_first_frame_objects: bool = False,
    include_object_ids: set[int] | None = None,
    quality_approval: Path | None = None,
) -> dict:
    approval = require_annotation_approval(quality_approval)
    if layout not in {"per_object", "packed"}:
        raise ValueError(f"unknown layout: {layout}")
    idmaps = sorted(idmap_dir.glob("frame_*_idmap.png"))
    rgbs = sorted(rgb_dir.glob("*.jpg"))
    validate_export_sources(approval, idmaps, rgbs)
    indices = list(range(len(idmaps)))
    if frame_range is not None:
        lo, hi = frame_range
        if not (0 <= lo <= hi < len(idmaps)):
            raise ValueError("invalid frame range")
        indices = indices[lo : hi + 1]
        idmaps = idmaps[lo : hi + 1]
        rgbs = rgbs[lo : hi + 1]
    if len(idmaps) != len(rgbs):
        raise ValueError(f"idmap/rgb count mismatch: {len(idmaps)} vs {len(rgbs)}")

    import cv2  # 遅延 import(idmap 読み込み用)

    np_mod = __import__("numpy")
    first_ids: set[int] = set()
    if include_object_ids is not None:
        # Explicit IDs must still account for every visible object.
        first_ids = {int(v) for v in include_object_ids}
    elif restrict_to_first_frame_objects:
        first = cv2.imread(str(idmaps[0]), cv2.IMREAD_UNCHANGED)
        if first is None:
            raise FileNotFoundError(idmaps[0])
        first_ids = {int(v) for v in np_mod.unique(first) if v > 0}
    else:
        for p in idmaps:
            m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if m is None:
                raise FileNotFoundError(p)
            first_ids |= {int(v) for v in np_mod.unique(m) if v > 0}
    # Validate the entire selected scope before creating any output.
    observed_ids: set[int] = set()
    for path in idmaps:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.dtype != np_mod.uint16:
            raise ValueError(f"idmap must be uint16: {path}")
        ids = {int(v) for v in np_mod.unique(mask) if v}
        observed_ids |= ids
        if ids - first_ids:
            raise ValueError("quality export would drop visible object IDs")
        if layout == "packed" and ids and max(ids) > 255:
            raise ValueError("packed annotation cannot preserve IDs above 255")
    if first_ids != observed_ids:
        raise ValueError("quality export object IDs differ from visible objects")
    if not video_name or Path(video_name).name != video_name:
        raise ValueError("video name must be a single component")
    jpeg_dir = out_root / "JPEGImages" / video_name
    ann_dir = out_root / "Annotations" / video_name
    if jpeg_dir.exists() and any(jpeg_dir.iterdir()) and not force:
        raise FileExistsError(f"refusing to overwrite non-empty {jpeg_dir}")
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    dropped_ids: set[int] = set()
    for idx, (idmap_path, rgb_path) in enumerate(zip(idmaps, rgbs, strict=True)):
        idmap = cv2.imread(str(idmap_path), cv2.IMREAD_UNCHANGED)
        if idmap is None or idmap.dtype != np_mod.uint16:
            raise ValueError(f"idmap must be uint16: {idmap_path}")
        ids = sorted(int(v) for v in np_mod.unique(idmap) if v > 0)
        if set(ids) - first_ids:
            raise ValueError("quality visible IDs changed during export")
        if layout == "packed" and ids and max(ids) > 255:
            raise ValueError(f"object id > 255 not supported: {max(ids)}")
        frame_name = f"{idx}" if integer_names else rgb_path.stem
        dst_rgb = jpeg_dir / f"{frame_name}.jpg"
        shutil.copy2(rgb_path, dst_rgb)
        if layout == "packed":
            dst_ann = ann_dir / f"{frame_name}.png"
            write_ann_png(dst_ann, idmap.astype("uint8"))
            max_id = max(ids) if ids else 0
            ann_records = {"packed": {"sha256": sha256_file(dst_ann), "max_id": max_id}}
        else:
            ann_records = {}
            for obj_id in sorted(first_ids):
                obj_dir = ann_dir / f"{obj_id:03d}"
                obj_dir.mkdir(parents=True, exist_ok=True)
                dst_ann = obj_dir / f"{frame_name}.png"
                binary = ((idmap == obj_id).astype("uint8")) * 255
                if not cv2.imwrite(str(dst_ann), binary):
                    raise OSError(f"failed to write {dst_ann}")
                ann_records[f"{obj_id:03d}"] = {"sha256": sha256_file(dst_ann)}
        files[frame_name] = {
            "source_rgb": rgb_path.name,
            "source_idmap": idmap_path.name,
            "missing_object_ids": sorted(first_ids - set(ids)),
            "rgb": {"sha256": sha256_file(dst_rgb)},
            "ann": ann_records,
        }
    manifest = {
        "schema_version": 2,
        "integer_names": integer_names,
        "quality_receipt": {
            **{
                k: approval[k]
                for k in (
                    "approval_path",
                    "approval_sha256",
                    "reference_sha256",
                    "review_sha256",
                )
            },
            "frame_indices": indices,
            "object_ids": sorted(first_ids),
        },
        "video_name": video_name,
        "frames": len(files),
        "layout": layout,
        "restrict_to_first_frame_objects": restrict_to_first_frame_objects,
        "include_object_ids": sorted(first_ids)
        if include_object_ids is not None
        else None,
        "first_frame_objects": sorted(first_ids),
        "dropped_object_ids": sorted(dropped_ids),
        "missing_any_frame": sorted(
            {o for rec in files.values() for o in rec["missing_object_ids"]}
        ),
        "idmap_dir": str(idmap_dir),
        "rgb_dir": str(rgb_dir),
        "files": files,
    }
    (out_root / f"manifest_{video_name}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SA-V style dataset from teacher idmaps"
    )
    parser.add_argument("--idmap-dir", required=True)
    parser.add_argument("--rgb-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--video-name", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--frame-range", nargs=2, type=int, default=None)
    parser.add_argument(
        "--no-integer-names",
        action="store_true",
        help="ファイル名を元 stem のままにする(既定は整数連番=学習 loader 仕様)",
    )
    parser.add_argument(
        "--layout",
        choices=["per_object", "packed"],
        default="per_object",
        help="per_object=SA-V 形式(既定)、packed=DAVIS 形式(1フレーム1PNG)",
    )
    parser.add_argument(
        "--no-restrict-first-frame",
        action="store_true",
        help="互換オプション。正式出力は常に全可視IDを保持する",
    )
    parser.add_argument("--quality-approval", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_sav_dataset(
        Path(args.idmap_dir),
        Path(args.rgb_dir),
        Path(args.out_root),
        args.video_name,
        force=args.force,
        frame_range=tuple(args.frame_range) if args.frame_range else None,
        integer_names=not args.no_integer_names,
        layout=args.layout,
        restrict_to_first_frame_objects=False,
        quality_approval=args.quality_approval,
    )
    print(f"wrote {args.out_root}: video={args.video_name} frames={manifest['frames']}")


if __name__ == "__main__":
    main()
