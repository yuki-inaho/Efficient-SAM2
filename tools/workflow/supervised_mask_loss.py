"""Recover artificial occlusion while explicitly suppressing verified negatives."""

import numpy as np
import torch
import torch.nn.functional as F


def supervised_mask_loss(
    logits,
    target,
    occluder,
    points,
    labels,
    negative_weight=8.0,
    boundary_background_weight=0.0,
):
    target = np.asarray(target, bool)
    points, labels = np.asarray(points, float), np.asarray(labels)
    h, w = target.shape
    if logits.shape != (1, 1, h, w) or np.shape(occluder) != target.shape:
        raise ValueError("target/logit/occluder shape mismatch")
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or labels.shape != points.shape[:1]
        or not np.isfinite(points).all()
    ):
        raise ValueError("invalid prompt points")
    if not np.isin(labels, [0, 1]).all() or negative_weight < 0:
        raise ValueError("invalid negative labels/weight")
    if not np.isfinite(boundary_background_weight) or boundary_background_weight < 0:
        raise ValueError("invalid boundary background weight")
    p = np.rint(points).astype(int)
    if np.any(p < 0) or np.any(p[:, 0] >= w) or np.any(p[:, 1] >= h):
        raise ValueError("prompt out of bounds")
    neg = p[labels == 0]
    if len(neg) and target[neg[:, 1], neg[:, 0]].any():
        raise ValueError("negative point inside teacher object")
    truth = torch.as_tensor(target, device=logits.device, dtype=torch.float32)[
        None, None
    ]
    hidden = torch.as_tensor(
        (occluder > 0) & target, device=logits.device, dtype=torch.float32
    )[None, None]
    weights = 1 + 4 * hidden
    hard_background = np.zeros((h, w), bool)
    yy, xx = np.ogrid[:h, :w]
    radius = max(2, round(min(h, w) * 0.01))
    for x, y in neg:
        hard_background |= (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    hard_background &= ~target
    weights = (
        weights
        + negative_weight
        * torch.as_tensor(hard_background, device=logits.device)[None, None]
    )
    bce = (
        F.binary_cross_entropy_with_logits(logits, truth, reduction="none") * weights
    ).sum() / weights.sum()
    prob = logits.sigmoid()
    dice = 1 - (2 * (prob * truth).sum() + 1) / (prob.sum() + truth.sum() + 1)
    point_loss = logits.sum() * 0
    if len(neg) and negative_weight:
        point_loss = F.softplus(logits[0, 0, neg[:, 1], neg[:, 0]]).mean()
    background_loss = logits.sum() * 0
    if boundary_background_weight:
        # Protect foreground and its uncertain three-pixel boundary. Penalize
        # confident exterior predictions, including synthetic lines extending outside.
        protected = F.max_pool2d(truth, 7, stride=1, padding=3) > 0
        near_target = F.max_pool2d(truth, 25, stride=1, padding=12) > 0
        synthetic = torch.as_tensor(occluder > 0, device=logits.device)[None, None]
        exterior = (near_target | synthetic) & ~protected
        if exterior.any():
            background_loss = F.softplus(logits[exterior]).mean()
    return bce + dice + point_loss + boundary_background_weight * background_loss
