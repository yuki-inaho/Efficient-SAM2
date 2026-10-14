"""Greedy mask association for unapproved video annotation drafts.

Candidates have ``candidate_id`` and a boolean ``small`` mask. Predictions map
persistent integer tracklet IDs to boolean masks in the same coordinate system.
Returned candidate indices refer to the input list. Edges preserve ambiguity;
one-to-one matches do not establish physical identity or annotation approval.
Thresholds are initial proposal settings, not validated accuracy criteria.
"""

import numpy as np


def geometry(mask):
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    return (
        int(xx.min()),
        int(yy.min()),
        int(xx.max()) + 1,
        int(yy.max()) + 1,
        len(xx),
        float(xx.mean()),
        float(yy.mean()),
    )


def gated_overlap(a, b, ga, gb):
    if ga is None or gb is None:
        return None
    x0 = max(ga[0], gb[0])
    y0 = max(ga[1], gb[1])
    x1 = min(ga[2], gb[2])
    y1 = min(ga[3], gb[3])
    if x0 >= x1 or y0 >= y1:
        return None
    if np.hypot(ga[5] - gb[5], ga[6] - gb[6]) > 0.75 * (np.sqrt(ga[4]) + np.sqrt(gb[4])):
        return None
    inter = int(np.count_nonzero(a[y0:y1, x0:x1] & b[y0:y1, x0:x1]))
    return inter / max(1, ga[4] + gb[4] - inter), inter / max(1, min(ga[4], gb[4]))


def deduplicate(candidates):
    kept = []
    suppressed = []
    geoms = {id(c): geometry(c["small"]) for c in candidates}
    for c in sorted(candidates, key=lambda c: -(geoms[id(c)][4] if geoms[id(c)] else 0)):
        for k in kept:
            metric = gated_overlap(c["small"], k["small"], geoms[id(c)], geoms[id(k)])
            if metric and (metric[0] >= 0.60 or metric[1] >= 0.90):
                suppressed.append(
                    dict(
                        candidate_id=c["candidate_id"],
                        kept=k["candidate_id"],
                        iou=metric[0],
                        containment=metric[1],
                        reason="draft_duplicate_seed",
                    )
                )
                break
        else:
            kept.append(c)
    return kept, suppressed


def associate(candidates, predicted):
    edges = []
    pg = {tid: geometry(mask) for tid, mask in predicted.items()}
    for ci, c in enumerate(candidates):
        cg = geometry(c["small"])
        for tid, mask in predicted.items():
            metric = gated_overlap(c["small"], mask, cg, pg[tid])
            if metric and (metric[0] >= 0.15 or metric[1] >= 0.65):
                edges.append(
                    dict(candidate_index=ci, track_id=tid, iou=metric[0], containment=metric[1])
                )
    matches = {}
    used = set()
    for e in sorted(edges, key=lambda e: (-e["iou"], -e["containment"], e["track_id"])):
        if e["candidate_index"] not in matches and e["track_id"] not in used:
            matches[e["candidate_index"]] = e["track_id"]
            used.add(e["track_id"])
    return matches, edges
