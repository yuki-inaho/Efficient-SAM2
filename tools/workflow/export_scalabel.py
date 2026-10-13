"""手順13 / P1-7: teacher idmap を Scalabel 形式(polygon 派生+image_list.yml)へエクスポートする。

- 同一物体IDの複数の連結成分は、poly2d に複数 polygon として保持する。
- 穴は Scalabel polygon で厳密には保持できないため、正本は PNG マスクとし、
  polygon は近似表示であること(has_holes 属性+README)を明記する。
- UI(docker/redis)は本環境で起動不可のため、構造スキーマ検証までを担う。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def _pixel_run_rectangles(mask: np.ndarray, ox: int, oy: int) -> list[list[list[float]]]:
    """Exact pixel-center coverage for degenerate components, without background inflation."""
    rectangles = []
    for y, row in enumerate(mask):
        padded = np.pad(row.astype(np.int8), (1, 1))
        edges = np.diff(padded)
        for x0, x1 in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            left, right = float(x0 + ox) - 0.5, float(x1 + ox) - 0.5
            top, bottom = float(y + oy) - 0.5, float(y + oy) + 0.5
            rectangles.append([[left, top], [right, top], [right, bottom], [left, bottom]])
    return rectangles


def _polygon_area(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def mask_to_polygons(mask: np.ndarray, eps: float = 2.0) -> tuple[list[list[float]], int]:
    """mask の全連結成分(8近傍)を polygon 化して返す。

    穴の内部にある独立成分も含め、connectedComponentsWithStats で列挙する。
    3点未満または面積0に退化する成分は各行の連続画素を矩形で表す。
    1成分が複数polygonになることがあるが、画素中心の余剰/欠落は生まない。
    """
    n, labels, _stats, _cent = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    polys: list[list[float]] = []
    for k in range(1, n):  # 0 は背景
        comp = labels == k
        ys, xs = np.nonzero(comp)
        if xs.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        crop = comp[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8)
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts: np.ndarray | None = None
        if contours:
            c = max(contours, key=cv2.contourArea)
            raw = c.reshape(-1, 2).astype(np.float64)
            if len(raw) >= 3:
                # 画素被覆を最優先し、approx による縮退は使わない(eps は互換引数)
                pts = raw
        if pts is None or _polygon_area(pts) < 0.5:
            polys.extend(_pixel_run_rectangles(crop, x0, y0))
            continue
        pts = pts.copy()
        pts[:, 0] += x0
        pts[:, 1] += y0
        polys.append([[float(a), float(b)] for a, b in pts])
    return polys, 0


def count_holes(mask: np.ndarray) -> int:
    """RETR_CCOMP の子輪郭(穴)数を数える。"""
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return 0
    return int(sum(1 for h in hierarchy[0] if h[3] != -1))


def mask_to_polygon(mask: np.ndarray, eps: float = 2.0) -> list[list[float]]:
    """[後方互換] 最大連結成分の polygon のみを返す。"""
    polys, _ = mask_to_polygons(mask, eps)
    return polys[0] if polys else []


def frame_labels(
    idmap: np.ndarray,
    object_labels: dict[int, str],
    frame_name: str,
    video_name: str,
    eps: float = 2.0,
) -> dict:
    labels = []
    for obj_id in sorted(int(v) for v in np.unique(idmap) if v > 0):
        mask = idmap == obj_id
        polygons, n_dropped = mask_to_polygons(mask, eps)
        if not polygons:
            continue
        pts = np.vstack([np.asarray(p) for p in polygons])
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        n_holes = count_holes(mask)
        labels.append(
            {
                "id": obj_id,
                # Scalabel 公式スキーマでは category は文字列
                "category": object_labels.get(obj_id, "object"),
                "box2d": {
                    "x1": float(x0),
                    "y1": float(y0),
                    "x2": float(x1),
                    "y2": float(y1),
                },
                "poly2d": [
                    {
                        "vertices": poly,
                        "closed": True,
                        "types": "L" * len(poly),
                    }
                    for poly in polygons
                ],
                "attributes": {
                    "n_components": int(
                        cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1
                    ),
                    "n_polygons": len(polygons),
                    "n_small_components_dropped": n_dropped,
                    "has_holes": bool(n_holes),
                    "source": "png_idmap(正本)",
                    "polygon_approximation": True,
                },
            }
        )
    return {"name": frame_name, "videoName": video_name, "labels": labels}


def _is_number(v) -> bool:  # noqa: ANN001
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_labels(frames: list[dict]) -> None:
    """Scalabel 公式スキーマ準拠の構造検証(UI が使えないため形式検証まで)。"""
    if not isinstance(frames, list):
        raise ValueError("frames must be a list")
    for frame in frames:
        for key in ("name", "videoName", "labels"):
            if key not in frame:
                raise ValueError(f"frame missing key: {key}")
        for label in frame["labels"]:
            for key in ("id", "category", "box2d", "poly2d"):
                if key not in label:
                    raise ValueError(f"label missing key: {key}")
            if not isinstance(label["id"], int):
                raise ValueError(f"label id must be int: {label['id']!r}")
            if not isinstance(label["category"], str):
                raise ValueError("category must be a string (official schema)")
            for key in ("x1", "y1", "x2", "y2"):
                if key not in label["box2d"]:
                    raise ValueError(f"box2d missing {key}")
                if not _is_number(label["box2d"][key]):
                    raise ValueError(f"box2d.{key} must be a number")
            if not label["poly2d"]:
                raise ValueError("poly2d must not be empty")
            for poly in label["poly2d"]:
                if not poly.get("closed", False):
                    raise ValueError("poly2d.closed must be True")
                verts = poly.get("vertices", [])
                if len(verts) < 3:
                    raise ValueError("poly2d.vertices must have >=3 points")
                for px in verts:
                    if len(px) != 2 or not (_is_number(px[0]) and _is_number(px[1])):
                        raise ValueError("poly2d.vertices must be numeric [x, y]")
                if len(poly.get("types", "")) != len(verts):
                    raise ValueError("poly2d.types length must match vertices")


def build_image_list(
    entries: list[tuple[str, str]], url_base: str = "http://localhost:8686/items"
) -> str:
    data = [
        {"url": f"{url_base}/{video}/{file_name}", "videoName": video}
        for file_name, video in entries
    ]
    return yaml.dump(data, allow_unicode=True, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export teacher idmaps to Scalabel format")
    parser.add_argument("--idmap-dir", required=True)
    parser.add_argument("--rgb-dir", required=True)
    parser.add_argument("--meta-json", required=True, help="propagation video_tracks.json")
    parser.add_argument("--video-name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eps", type=float, default=2.0)
    args = parser.parse_args()

    meta = json.loads(Path(args.meta_json).read_text())
    object_labels = {int(k): v.get("label", "object") for k, v in meta.get("objects", {}).items()}

    idmaps = sorted(Path(args.idmap_dir).glob("frame_*_idmap.png"))
    rgbs = sorted(Path(args.rgb_dir).glob("*.jpg"))
    if len(idmaps) != len(rgbs):
        raise ValueError(f"idmap/rgb count mismatch: {len(idmaps)} vs {len(rgbs)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    entries = []
    for idmap_path, rgb_path in zip(idmaps, rgbs, strict=True):
        idmap = cv2.imread(str(idmap_path), cv2.IMREAD_UNCHANGED)
        if idmap is None or idmap.dtype != np.uint16:
            raise ValueError(f"idmap must be uint16: {idmap_path}")
        frames.append(frame_labels(idmap, object_labels, rgb_path.name, args.video_name, args.eps))
        entries.append((rgb_path.name, args.video_name))
    validate_labels(frames)
    (out_dir / "labels.json").write_text(json.dumps(frames, indent=2) + "\n")
    (out_dir / "image_list.yml").write_text(build_image_list(entries))
    (out_dir / "README.md").write_text(
        "# Scalabel export (MVP)\n\n"
        "- `labels.json`: フレームごとの polygon/box2d + track id(正本は PNG マスク)。\n"
        "- 複数の連結成分は保持する。退化成分は画素行の矩形で表す。"
        "n_componentsとn_polygonsは区別する。\n"
        "- 穴は polygon 形式で厳密に保持できないため、PNG を正本とし、polygon は近似"
        "(attributes.has_holes を参照)。\n"
        "- `image_list.yml`: 既存 scalabel_workspace の generate_image_list.py 互換。\n"
        "- `labels.json` は構造スキーマ検証済み。UI(docker/redis)起動不可のため"
        " UI 往復確認は行っていない(形式検証まで)。\n"
    )
    n_labels = sum(len(f["labels"]) for f in frames)
    print(f"wrote {out_dir}: frames={len(frames)} labels={n_labels}")


if __name__ == "__main__":
    main()
