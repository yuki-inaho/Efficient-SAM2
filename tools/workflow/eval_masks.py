"""手順19: 学生/教師のマスク比較評価(idmap 同士 or パレット PNG 同士)。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def load_mask(path: Path) -> np.ndarray:
    """PNG をパレット index のまま読む(cv2 はパレット PNG を RGB 展開するため PIL を使う)。"""
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D (palette/grayscale), got {arr.shape} for {path}")
    if arr.dtype == np.uint32:
        arr = arr.astype(np.uint16)
    return arr


def frame_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    pred_ids = sorted(int(v) for v in np.unique(pred) if v > 0)
    gt_ids = sorted(int(v) for v in np.unique(gt) if v > 0)
    common = sorted(set(pred_ids) & set(gt_ids))
    id_ious = {str(i): iou(pred == i, gt == i) for i in common}
    missing = [i for i in gt_ids if i not in common]
    extra = [i for i in pred_ids if i not in common]
    # 仕様: 同じID体系では「教師にあって予測にない物体」は IoU=0 として評価に含める
    id_ious_with_missing = dict(id_ious)
    for i in missing:
        id_ious_with_missing[str(i)] = 0.0
    return {
        "fg_iou": iou(pred > 0, gt > 0),
        "n_pred_objects": len(pred_ids),
        "n_gt_objects": len(gt_ids),
        "matched_objects": len(common),
        "missing_objects": missing,
        "extra_objects": extra,
        "id_ious": id_ious,
        "id_ious_with_missing": id_ious_with_missing,
    }


def _natural_key(p: Path) -> tuple:
    import re as _re

    nums = tuple(int(n) for n in _re.findall(r"\d+", p.stem))
    return (nums, p.stem)


def _manifest_entries(
    pred_dir: Path, gt_dir: Path, manifest_path: Path
) -> tuple[list[dict], list[dict]]:
    """manifest の対応表で pred/gt を結び付ける(ファイル名は manifest 由来)。"""
    import json as _json

    manifest = _json.loads(manifest_path.read_text())
    entries: list[dict] = []
    skipped: list[dict] = []
    for frame_name, record in manifest["files"].items():
        pf = pred_dir / f"{frame_name}.png"
        gf = gt_dir / f"{frame_name}.png"
        if not pf.exists() or not gf.exists():
            skipped.append(
                {
                    "frame": frame_name,
                    "source_rgb": record.get("source_rgb"),
                    "reason": "pred_or_gt_file_missing",
                }
            )
            continue
        entries.append({"frame": frame_name, "pred": pf, "gt": gf})
    entries.sort(key=lambda e: _natural_key(Path(e["frame"])))
    return entries, skipped


def evaluate(
    pred_dir: Path,
    gt_dir: Path,
    limit: int | None = None,
    skip_empty_gt: bool = True,
    manifest_path: Path | None = None,
    empty_gt_as_background: bool = False,
) -> dict:
    entries: list[dict] = []
    correspondence_skipped: list[dict] = []
    if manifest_path is not None:
        entries, correspondence_skipped = _manifest_entries(pred_dir, gt_dir, manifest_path)
        if limit is not None:
            entries = entries[:limit]
    else:
        pred_files = sorted(pred_dir.glob("*.png"), key=_natural_key)
        gt_files = sorted(gt_dir.glob("*.png"), key=_natural_key)
        if not pred_files or not gt_files:
            raise FileNotFoundError(f"no masks: {pred_dir} / {gt_dir}")
        if limit is not None:
            pred_files = pred_files[:limit]
            gt_files = gt_files[:limit]
        if len(pred_files) != len(gt_files):
            raise ValueError(f"frame count mismatch: {len(pred_files)} vs {len(gt_files)}")
        entries = [
            {"frame": pf.stem, "pred": pf, "gt": gf}
            for pf, gf in zip(pred_files, gt_files, strict=True)
        ]
    if not entries:
        raise FileNotFoundError(f"no evaluable frames: {pred_dir} / {gt_dir}")

    frames = []
    skipped = 0
    background_frames = 0
    background_fp_frames = 0
    for e in entries:
        gt_mask = load_mask(e["gt"])
        pred_mask = load_mask(e["pred"])
        if not (gt_mask > 0).any():
            if not empty_gt_as_background and skip_empty_gt:
                # 未注釈(背景確定ではない)として除外
                skipped += 1
                continue
            metrics = frame_metrics(pred_mask, gt_mask)
            kind = "background" if empty_gt_as_background else "empty_gt_included"
            if kind == "background":
                background_frames += 1
                background_fp_frames += int(metrics["fg_iou"] < 1.0)
            frames.append({"pred": e["pred"].name, "gt": e["gt"].name, "kind": kind, **metrics})
            continue
        metrics = frame_metrics(pred_mask, gt_mask)
        frames.append({"pred": e["pred"].name, "gt": e["gt"].name, **metrics})
    if not frames:
        raise ValueError("all frames skipped (empty GT); disable skip_empty_gt")
    annotated = [f for f in frames if f.get("kind") != "background"]
    fg = [f["fg_iou"] for f in annotated]
    matched = [f["matched_objects"] for f in annotated]
    id_ious_all = [v for f in annotated for v in f["id_ious"].values()]
    id_ious_with_missing = [v for f in annotated for v in f["id_ious_with_missing"].values()]
    extras = [len(f["extra_objects"]) for f in annotated]
    missing_total = sum(len(f["missing_objects"]) for f in annotated)
    return {
        "frames": len(frames),
        "annotated_frames": len(annotated),
        "skipped_empty_gt": skipped,
        "skipped_correspondence": len(correspondence_skipped),
        "correspondence_skipped_detail": correspondence_skipped,
        "bbox_assumption_empty_gt_background": empty_gt_as_background,
        "mean_fg_iou": float(np.mean(fg)) if fg else None,
        "min_fg_iou": float(np.min(fg)) if fg else None,
        "mean_matched_objects": float(np.mean(matched)) if matched else None,
        "mean_id_iou": float(np.mean(id_ious_all)) if id_ious_all else None,
        "mean_id_iou_with_missing": (
            float(np.mean(id_ious_with_missing)) if id_ious_with_missing else None
        ),
        "total_missing_objects": int(missing_total),
        "mean_extra_objects": float(np.mean(extras)) if extras else None,
        "background_frames": background_frames,
        "background_false_positives": background_fp_frames,
        "reference": {
            "gt_kind": "teacher_pseudo_gt",
            "note": "教師(疑似正解)との一致度。人手正解(実精度)とは区別して報告する",
        },
        "per_frame": frames,
    }


def _fmt(value, spec: str = ".3f") -> str:  # noqa: ANN001
    return "n/a" if value is None else format(value, spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted masks against GT idmaps")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-empty-gt",
        action="store_true",
        help="GT が空のフレームも評価に含める(既定はスキップ=未注釈扱い)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="対応表 manifest(指定時はファイル名の並び順に依存せずマッピングする)",
    )
    parser.add_argument(
        "--empty-gt-as-background",
        action="store_true",
        help="空GTフレームを『確認済み背景』とみなして誤検出を評価する(仮定を出力に記録)",
    )
    args = parser.parse_args()
    result = evaluate(
        Path(args.pred_dir),
        Path(args.gt_dir),
        args.limit,
        skip_empty_gt=not args.include_empty_gt,
        manifest_path=Path(args.manifest) if args.manifest else None,
        empty_gt_as_background=args.empty_gt_as_background,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"frames={result['frames']} skipped_empty_gt={result['skipped_empty_gt']} "
        f"mean_fg_iou={_fmt(result['mean_fg_iou'])} "
        f"matched_objects={_fmt(result['mean_matched_objects'], '.1f')} "
        f"mean_id_iou={_fmt(result['mean_id_iou'])} "
        f"mean_id_iou_with_missing={_fmt(result['mean_id_iou_with_missing'])} "
        f"missing_objects={result['total_missing_objects']} "
        f"mean_extra_objects={_fmt(result['mean_extra_objects'], '.1f')}"
    )


if __name__ == "__main__":
    main()
