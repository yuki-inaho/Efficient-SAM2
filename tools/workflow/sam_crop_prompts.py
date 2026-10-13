"""Local SAM geometry and explicit target/other-object prompt separation."""

import numpy as np


def box_of(detection):
    if "box" in detection:
        return np.asarray(detection["box"], float)
    corners = np.asarray(detection["corners"], float)
    return np.r_[corners.min(axis=0), corners.max(axis=0)]


def choose_crop(box, image_wh, side=1024):
    box = np.asarray(box, float)
    width, height = image_wh
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("invalid box")
    lo = np.maximum(box[:2], 0)
    hi = np.minimum(box[2:], [width - 1, height - 1])
    if np.any(hi <= lo):
        raise ValueError("box outside image")
    side = max(side, int(np.ceil(max(hi - lo) + 96)))
    cw, ch = min(side, width), min(side, height)
    xs = sorted({*range(0, width - cw + 1, max(1, cw // 2)), width - cw})
    ys = sorted({*range(0, height - ch + 1, max(1, ch // 2)), height - ch})
    choices = [
        (x, y, cw, ch)
        for y in ys
        for x in xs
        if x <= lo[0] and y <= lo[1] and x + cw > hi[0] and y + ch > hi[1]
    ]
    center = (lo + hi) / 2
    if not choices:
        x = max(0, min(width - cw, int(center[0] - cw / 2)))
        y = max(0, min(height - ch, int(center[1] - ch / 2)))
        choices = [(x, y, cw, ch)]
    return min(
        choices,
        key=lambda r: (
            (r[0] + r[2] / 2 - center[0]) ** 2 + (r[1] + r[3] / 2 - center[1]) ** 2
        ),
    )


def local_prompts(detections, index, rectangle):
    d = detections[index]
    box = box_of(d)
    center = np.asarray(d["center"], float)
    raw_positive = np.concatenate(
        [center[None], np.asarray(d.get("positive_points", []), float).reshape(-1, 2)]
    )
    if not np.isfinite(raw_positive).all():
        raise ValueError("invalid positive point")
    positives = []
    for point in raw_positive:
        if not any(np.array_equal(np.rint(point), np.rint(q)) for q in positives):
            positives.append(point)
    positive = np.asarray(positives)
    x, y, w, h = rectangle
    other = []
    for j, candidate in enumerate(detections):
        if d.get("verified_negatives_only"):
            break
        if j == index or (
            candidate.get("alias_of") is not None
            and candidate["alias_of"] == d.get("candidate_id", d.get("object_id"))
        ):
            continue
        point = np.asarray(candidate["center"], float)
        if any(np.array_equal(np.rint(point), np.rint(q)) for q in positive):
            continue
        # Partial and full detections of one object may have far-apart centers.
        # Ambiguous overlap is not enough evidence to label the other center background.
        if np.all(point >= box[:2]) and np.all(point <= box[2:]):
            continue
        if "box" in candidate or "corners" in candidate:
            candidate_box = box_of(candidate)
            if any(
                np.all(q >= candidate_box[:2]) and np.all(q <= candidate_box[2:])
                for q in positive
            ):
                continue
        distance = np.linalg.norm(point - center)
        # Nearby hypotheses could be the same object; do not carve it apart.
        if max(3, 0.6 * min(box[2:] - box[:2])) < distance < 2 * max(box[2:] - box[:2]):
            other.append(point)
    explicit = np.asarray(d.get("negative_points", []), float).reshape(-1, 2)
    if not np.isfinite(explicit).all():
        raise ValueError("invalid explicit negative point")
    if any(
        np.array_equal(np.rint(point), np.rint(q))
        for point in explicit
        for q in positive
    ):
        raise ValueError("conflicting positive and explicit negative point")
    unique = []
    automatic_count = 0
    # Explicit exclusions survive distance ranking; only automatic additions are capped.
    for is_explicit, pool in (
        (True, explicit),
        (False, sorted(other, key=lambda p: np.linalg.norm(p - center))),
    ):
        for point in pool:
            if not (x <= point[0] < x + w and y <= point[1] < y + h):
                continue
            if any(np.linalg.norm(point - q) < 4 for q in unique):
                continue
            if not is_explicit:
                if automatic_count >= 4:
                    break
                automatic_count += 1
            unique.append(point)
    neg = np.asarray(unique, float).reshape(-1, 2)
    points = np.concatenate([positive, neg]) - [x, y]
    labels = np.r_[np.ones(len(positive), int), np.zeros(len(neg), int)]
    pp = points[labels == 1]
    if np.any(pp < 0) or np.any(pp[:, 0] >= w) or np.any(pp[:, 1] >= h):
        raise ValueError("positive point outside crop")
    return (
        points.astype(np.float32),
        labels.astype(np.int32),
        (box - [x, y, x, y]).astype(np.float32),
    )


def prompt_coverage(mask, points, labels):
    """Measure all positive/negative constraints, retaining the primary center check."""
    mask = np.asarray(mask, bool)
    points, labels = np.asarray(points, float), np.asarray(labels)
    if (
        mask.ndim != 2
        or points.ndim != 2
        or points.shape[1] != 2
        or labels.shape != points.shape[:1]
        or not len(points)
        or not np.isfinite(points).all()
        or not np.isin(labels, [0, 1]).all()
        or labels[0] != 1
    ):
        raise ValueError("invalid mask or labeled point prompts")
    h, w = mask.shape
    if np.any(points < 0) or np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h):
        raise ValueError("point outside mask")
    coords = np.minimum(np.rint(points).astype(int), [w - 1, h - 1])
    values = mask[coords[:, 1], coords[:, 0]]
    return {
        "own_center": bool(values[0]),
        "positive_violations": int((~values[labels == 1]).sum()),
        "negative_violations": int(values[labels == 0].sum()),
    }
