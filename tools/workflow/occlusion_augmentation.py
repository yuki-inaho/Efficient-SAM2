"""Synthetic thin green occluders; the pre-occlusion object target stays unchanged.

Apply to vetted training pairs only. Validation/test object identities and temporal
groups must be split before augmentation to avoid leaking the original object.
"""

import cv2
import numpy as np


def augment_thin_green(
    rgb: np.ndarray,
    target: np.ndarray,
    *,
    seed: int,
    probability: float = 0.7,
    width_ratio: tuple[float, float] = (0.025, 0.08),
    coverage: tuple[float, float] = (0.025, 0.25),
    max_attempts: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return RGB, unchanged target, uint8 opacity map, reproducibility metadata."""
    rgb, target = np.asarray(rgb), np.asarray(target)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("RGB must be uint8 H,W,3")
    if target.shape != rgb.shape[:2] or not np.isin(target, [0, 1]).all():
        raise ValueError("single binary object target must match RGB")
    mask = target.astype(bool)
    if not mask.any():
        raise ValueError("empty object target")
    if not 0 <= probability <= 1 or not 0 < coverage[0] <= coverage[1] < 1:
        raise ValueError("invalid probability or occlusion fraction bounds")
    if not 0 < width_ratio[0] <= width_ratio[1] < 1 or max_attempts < 1:
        raise ValueError("invalid thin-occluder width or attempts")
    rng = np.random.default_rng(seed)
    meta = {
        "seed": int(seed),
        "applied": False,
        "object_occlusion_fraction": 0.0,
        "teacher_policy": "unchanged_pre_occlusion_target",
    }
    if rng.random() >= probability:
        return rgb.copy(), target.copy(), np.zeros(mask.shape, np.uint8), meta
    yy, xx = np.where(mask)
    box_w, box_h = int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)
    span = float(np.hypot(box_w, box_h)) * 1.3
    for attempt in range(max_attempts):
        anchor_index = int(rng.integers(len(xx)))
        anchor = np.array([xx[anchor_index], yy[anchor_index]], dtype=float)
        angle = float(rng.uniform(0, np.pi))
        along = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-along[1], along[0]])
        bend = float(rng.uniform(-0.3, 0.3) * min(box_w, box_h))
        t = np.linspace(-1, 1, 65)
        curve = anchor + t[:, None] * span * along + (t * t)[:, None] * bend * normal
        pts = np.rint(curve).astype(np.int32)
        width = max(1, round(float(rng.uniform(*width_ratio)) * min(box_w, box_h)))
        alpha = np.zeros(mask.shape, np.uint8)
        cv2.polylines(alpha, [pts], False, 255, width, cv2.LINE_AA)
        fraction = float(np.count_nonzero((alpha > 0) & mask) / mask.sum())
        if not coverage[0] <= fraction <= coverage[1]:
            continue
        color = np.array(
            [rng.integers(20, 65), rng.integers(70, 140), rng.integers(15, 55)], np.float32
        )
        highlight = np.zeros(mask.shape, np.uint8)
        cv2.polylines(highlight, [pts], False, 255, max(1, width // 3), cv2.LINE_AA)
        shade = 0.75 + 0.3 * highlight.astype(np.float32) / 255
        tube_rgb = color[None, None, :] * shade[..., None]
        opacity = alpha.astype(np.float32)[..., None] / 255
        augmented = np.rint(rgb * (1 - opacity) + tube_rgb * opacity).clip(0, 255).astype(np.uint8)
        meta.update(
            {
                "applied": True,
                "object_occlusion_fraction": fraction,
                "width_pixels": width,
                "angle_rad": angle,
                "bend_pixels": bend,
                "anchor_xy": anchor.tolist(),
                "color_rgb": color.astype(int).tolist(),
                "attempts": attempt + 1,
                "curve_xy": pts.tolist(),
            }
        )
        return augmented, target.copy(), alpha, meta
    raise ValueError("Could not place a thin occluder within the requested coverage bounds")
