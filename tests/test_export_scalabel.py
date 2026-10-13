"""手順13: export_scalabel の単体テスト。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.workflow.export_scalabel import (  # noqa: E402
    build_image_list,
    count_holes,
    frame_labels,
    mask_to_polygon,
    mask_to_polygons,
    validate_labels,
)


def test_mask_to_polygon_rectangle() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    poly = mask_to_polygon(mask, eps=1.0)
    assert len(poly) >= 4
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    assert min(xs) == 5 and max(xs) == 14
    assert min(ys) == 5 and max(ys) == 14


def test_mask_to_polygon_empty() -> None:
    assert mask_to_polygon(np.zeros((4, 4), dtype=bool)) == []


def test_frame_labels() -> None:
    idmap = np.zeros((10, 10), dtype=np.uint16)
    idmap[1:5, 1:5] = 1
    idmap[5:9, 5:9] = 2
    frame = frame_labels(idmap, {1: "object_a", 2: "object_b"}, "f.jpg", "clip", eps=1.0)
    assert frame["name"] == "f.jpg"
    assert frame["videoName"] == "clip"
    assert [label["id"] for label in frame["labels"]] == [1, 2]
    assert frame["labels"][1]["category"] == "object_b"  # Scalabel 公式は文字列
    assert frame["labels"][0]["poly2d"][0]["closed"] is True


def test_build_image_list_format() -> None:
    text = build_image_list([("0001_rgb.jpg", "clipA")])
    data = yaml.safe_load(text)
    assert data[0]["url"] == "http://localhost:8686/items/clipA/0001_rgb.jpg"
    assert data[0]["videoName"] == "clipA"


def test_multiple_components_are_kept_for_same_id() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:6, 2:6] = True
    mask[12:18, 12:18] = True
    polys, dropped = mask_to_polygons(mask, eps=1.0)
    assert len(polys) == 2
    assert dropped == 0
    frame = frame_labels(
        np.where(mask, 3, 0).astype(np.uint16), {3: "object_a"}, "f.jpg", "clip", eps=1.0
    )
    label = frame["labels"][0]
    assert len(label["poly2d"]) == 2  # 同一IDで複数polygon
    assert label["attributes"]["n_components"] == 2


def test_small_components_are_retained_not_dropped() -> None:
    # 大きな成分 + 小成分 + 孤立1画素(矩形の外)を全て保持する
    mask = np.zeros((24, 24), dtype=bool)
    mask[2:8, 2:8] = True
    mask[12:16, 12:16] = True
    mask[20, 20] = True  # 孤立1画素(他成分の外)
    polys, dropped = mask_to_polygons(mask, eps=2.0)
    assert len(polys) == 3
    assert dropped == 0
    frame = frame_labels(
        np.where(mask, 1, 0).astype(np.uint16), {1: "object_a"}, "f.jpg", "clip", eps=2.0
    )
    attrs = frame["labels"][0]["attributes"]
    assert attrs["n_components"] == 3


def test_component_inside_hole_is_retained() -> None:
    # 輪(穴あり) + 穴の内部にある独立成分 → 2成分として保持
    mask = np.zeros((24, 24), dtype=bool)
    mask[4:20, 4:20] = True
    mask[8:16, 8:16] = False  # 穴
    mask[10:14, 10:14] = True  # 穴の中の島
    import cv2 as _cv2

    n, _ = _cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    polys, dropped = mask_to_polygons(mask, eps=1.0)
    assert n - 1 == 2
    assert len(polys) == 2
    assert dropped == 0


def test_degenerate_components_fully_covered() -> None:
    import cv2 as _cv2

    mask = np.zeros((24, 24), dtype=bool)
    mask[10, 10] = True  # 孤立1画素(退化)
    mask[20, 20] = True
    n, labels = _cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    assert n - 1 == 2
    polys, dropped = mask_to_polygons(mask, eps=2.0)
    assert len(polys) == 2
    assert dropped == 0
    for k, poly in enumerate(polys):
        pts = np.asarray(poly, dtype=np.float32)
        assert _cv2.contourArea(pts) > 0  # 面積0の polygon を残さない
        ys, xs = np.nonzero(labels == k + 1)
        assert pts[:, 0].min() <= xs.min() and pts[:, 0].max() >= xs.max()
        assert pts[:, 1].min() <= ys.min() and pts[:, 1].max() >= ys.max()


def test_polygon_versions_cover_component_bbox() -> None:
    import cv2 as _cv2

    mask = np.zeros((16, 16), dtype=bool)
    mask[3:7, 3:7] = True
    mask[11, 11] = True
    polys, _ = mask_to_polygons(mask, eps=2.0)
    ys, xs = np.nonzero(mask)
    joined = np.vstack([np.asarray(p, dtype=np.float32) for p in polys])
    assert joined[:, 0].min() <= xs.min()
    assert joined[:, 0].max() >= xs.max()
    assert joined[:, 1].min() <= ys.min()
    assert joined[:, 1].max() >= ys.max()
    assert _cv2.contourArea(np.asarray(polys[0], dtype=np.float32)) > 0


def test_origin_vs_shifted_shape_are_equally_covered() -> None:
    import cv2 as _cv2

    def make(oy: int, ox: int) -> np.ndarray:
        m = np.zeros((24, 24), dtype=bool)
        m[oy : oy + 8, ox : ox + 8] = True
        m[oy : oy + 3, ox : ox + 3] = False  # 左上3x3を除去(L字)
        return m

    for oy, ox in ((0, 0), (10, 10)):
        mask = make(oy, ox)
        polys, _ = mask_to_polygons(mask, eps=2.0)
        joined = [np.asarray(p, dtype=np.float32) for p in polys]
        ys, xs = np.nonzero(mask)
        outside = 0
        for x, y in zip(xs, ys):
            if not any(_cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for p in joined):
                outside += 1
        assert outside == 0, f"origin/shift mismatch at ({oy},{ox}): {outside} outside"


def test_validate_labels_rejects_official_schema_violations() -> None:
    good = [
        {
            "name": "0.jpg",
            "videoName": "clip",
            "labels": [
                {
                    "id": 1,
                    "category": "object_a",
                    "box2d": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
                    "poly2d": [
                        {"vertices": [[0, 0], [1, 0], [1, 1]], "closed": True, "types": "LLL"}
                    ],
                }
            ],
        }
    ]
    validate_labels(good)
    # category は文字列でなければならない(公式スキーマ)
    bad_cat = json.loads(json.dumps(good))
    bad_cat[0]["labels"][0]["category"] = {"name": "object_a"}
    with pytest.raises(ValueError, match="category must be a string"):
        validate_labels(bad_cat)
    # 座標が文字列なら拒否
    bad_coord = json.loads(json.dumps(good))
    bad_coord[0]["labels"][0]["box2d"]["x1"] = "0"
    with pytest.raises(ValueError, match="x1 must be a number"):
        validate_labels(bad_coord)
    bad_vert = json.loads(json.dumps(good))
    bad_vert[0]["labels"][0]["poly2d"][0]["vertices"][0] = ["a", 0]
    with pytest.raises(ValueError, match="numeric"):
        validate_labels(bad_vert)
    bad_closed = json.loads(json.dumps(good))
    bad_closed[0]["labels"][0]["poly2d"][0]["closed"] = False
    with pytest.raises(ValueError, match="closed"):
        validate_labels(bad_closed)


def test_hole_is_reported_and_source_is_png() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:18, 2:18] = True
    mask[8:12, 8:12] = False  # 穴
    assert count_holes(mask) == 1
    frame = frame_labels(
        np.where(mask, 1, 0).astype(np.uint16), {1: "object_a"}, "f.jpg", "clip", eps=1.0
    )
    attrs = frame["labels"][0]["attributes"]
    assert attrs["has_holes"] is True
    assert attrs["source"] == "png_idmap(正本)"
    assert attrs["polygon_approximation"] is True


def test_validate_labels_missing_box2d() -> None:
    good = [
        {
            "name": "0.jpg",
            "videoName": "clip",
            "labels": [
                {
                    "id": 1,
                    "category": "object_a",
                    "box2d": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
                    "poly2d": [
                        {"vertices": [[0, 0], [1, 0], [1, 1]], "closed": True, "types": "LLL"}
                    ],
                }
            ],
        }
    ]
    bad2 = json.loads(json.dumps(good))
    del bad2[0]["labels"][0]["box2d"]
    with pytest.raises(ValueError, match="box2d"):
        validate_labels(bad2)


def test_degenerate_polygon_adds_no_neighboring_pixel_centers() -> None:
    from tools.workflow.audit_polygon_export import raster_polygons

    for points in ([(10, 10)], [(10, x) for x in range(5, 16)], [(x, x) for x in range(5, 16)]):
        mask = np.zeros((24, 24), dtype=bool)
        for y, x in points:
            mask[y, x] = True
        polygons, _ = mask_to_polygons(mask)
        raster = raster_polygons([{"vertices": p} for p in polygons], mask.shape)
        np.testing.assert_array_equal(raster, mask)
