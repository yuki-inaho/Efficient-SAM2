"""手順19: eval_masks の単体テスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.workflow.eval_masks import evaluate, frame_metrics, iou, load_mask  # noqa: E402


def test_iou() -> None:
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[:2] = True
    b[:4] = True
    assert iou(a, b) == pytest.approx(0.5)
    assert iou(a, a) == 1.0
    assert iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == 1.0


def test_frame_metrics_counts() -> None:
    pred = np.zeros((4, 4), dtype=np.uint16)
    pred[:, :2] = 1
    pred[:, 2:] = 2
    gt = pred.copy()
    m = frame_metrics(pred, gt)
    assert m["fg_iou"] == 1.0
    assert m["matched_objects"] == 2
    assert m["id_ious"]["1"] == 1.0


def test_frame_metrics_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        frame_metrics(np.zeros((2, 2), np.uint16), np.zeros((3, 3), np.uint16))


def test_evaluate_dir(tmp_path: Path) -> None:
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    for i in range(3):
        gt = np.zeros((4, 4), dtype=np.uint16)
        gt[:, :2] = 1
        cv2.imwrite(str(gt_dir / f"frame_{i:04d}_idmap.png"), gt)
        pred = gt.copy()
        cv2.imwrite(str(pred_dir / f"frame_{i:04d}_idmap.png"), pred)
    result = evaluate(pred_dir, gt_dir)
    assert result["frames"] == 3
    assert result["mean_fg_iou"] == 1.0


def test_load_mask_palette_png(tmp_path: Path) -> None:
    from PIL import Image

    arr = np.zeros((4, 4), dtype=np.uint8)
    arr[:, :2] = 1
    img = Image.fromarray(arr)
    img.putpalette([i for i in range(256)] * 3)
    p = tmp_path / "pal.png"
    img.save(p)
    out = load_mask(p)
    assert out.ndim == 2
    assert set(np.unique(out).tolist()) == {0, 1}


def test_evaluate_skips_empty_gt(tmp_path: Path) -> None:
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    for i in range(3):
        gt = np.zeros((4, 4), dtype=np.uint16)
        if i != 1:
            gt[:, :2] = 1
        cv2.imwrite(str(gt_dir / f"{i}.png"), gt)
        cv2.imwrite(str(pred_dir / f"{i}.png"), gt.copy())
    result = evaluate(pred_dir, gt_dir)
    assert result["frames"] == 2
    assert result["skipped_empty_gt"] == 1


def test_evaluate_count_mismatch(tmp_path: Path) -> None:
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    cv2.imwrite(str(gt_dir / "frame_0000_idmap.png"), np.zeros((4, 4), np.uint16))
    for i in range(2):
        cv2.imwrite(str(pred_dir / f"frame_{i:04d}_idmap.png"), np.zeros((4, 4), np.uint16))
    with pytest.raises(ValueError, match="count mismatch"):
        evaluate(pred_dir, gt_dir)


def test_frame_metrics_missing_object_gets_zero_iou() -> None:
    gt = np.zeros((4, 4), dtype=np.uint16)
    gt[:, :2] = 1
    gt[:, 2:] = 2
    pred = gt.copy()
    pred[pred == 2] = 0  # 物体2が予測から消失
    m = frame_metrics(pred, gt)
    assert m["missing_objects"] == [2]
    assert m["extra_objects"] == []
    assert m["id_ious_with_missing"]["2"] == 0.0
    assert m["id_ious"]["1"] == 1.0


def test_evaluate_missing_object_reflected_in_metric(tmp_path: Path) -> None:
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    for i in range(2):
        gt = np.zeros((4, 4), dtype=np.uint16)
        gt[:, :2] = 1
        gt[:, 2:] = 2
        cv2.imwrite(str(gt_dir / f"{i}.png"), gt)
        pred = gt.copy()
        pred[pred == 2] = 0
        cv2.imwrite(str(pred_dir / f"{i}.png"), pred)
    result = evaluate(pred_dir, gt_dir)
    assert result["mean_id_iou"] == pytest.approx(1.0)  # 一致物体のみ
    assert result["mean_id_iou_with_missing"] == pytest.approx(0.5)  # 欠落=0で平均
    assert result["total_missing_objects"] == 2


def test_evaluate_manifest_correspondence(tmp_path: Path) -> None:
    import json as _json

    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    for i in range(2):
        gt = np.zeros((4, 4), dtype=np.uint16)
        gt[:, :2] = 1
        cv2.imwrite(str(gt_dir / f"{i}.png"), gt)
        cv2.imwrite(str(pred_dir / f"{i}.png"), gt.copy())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        _json.dumps(
            {
                "files": {
                    "0": {"source_rgb": "a.jpg"},
                    "1": {"source_rgb": "b.jpg"},
                    "2": {"source_rgb": "c.jpg"},  # pred/gt に無い -> correspondence skip
                }
            }
        )
    )
    result = evaluate(pred_dir, gt_dir, manifest_path=manifest)
    assert result["frames"] == 2
    assert result["skipped_correspondence"] == 1
    assert result["correspondence_skipped_detail"][0]["frame"] == "2"


def test_evaluate_empty_gt_as_background(tmp_path: Path) -> None:
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    cv2.imwrite(str(gt_dir / "0.png"), np.zeros((4, 4), np.uint16))
    pred = np.zeros((4, 4), dtype=np.uint16)
    pred[:, :2] = 1
    cv2.imwrite(str(pred_dir / "0.png"), pred)
    result = evaluate(pred_dir, gt_dir, empty_gt_as_background=True)
    assert result["background_frames"] == 1
    assert result["background_false_positives"] == 1
    assert result["per_frame"][0]["kind"] == "background"
    assert result["mean_fg_iou"] is None  # 背景フレームは FG 平均から除外


def test_evaluate_all_background_cli_does_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    for i in range(2):
        cv2.imwrite(str(gt_dir / f"{i}.png"), np.zeros((4, 4), np.uint16))
        pred = np.zeros((4, 4), dtype=np.uint16)
        pred[:, :2] = 1
        cv2.imwrite(str(pred_dir / f"{i}.png"), pred)
    result = evaluate(pred_dir, gt_dir, empty_gt_as_background=True)
    assert result["mean_fg_iou"] is None
    assert result["mean_matched_objects"] is None
    # CLI 経路(整形)でも例外にならない
    from tools.workflow.eval_masks import main as eval_main

    out = tmp_path / "out.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_masks",
            "--pred-dir",
            str(pred_dir),
            "--gt-dir",
            str(gt_dir),
            "--empty-gt-as-background",
            "--out",
            str(out),
        ],
    )
    eval_main()
    captured = capsys.readouterr()
    assert "mean_fg_iou=n/a" in captured.out
    assert out.exists()
