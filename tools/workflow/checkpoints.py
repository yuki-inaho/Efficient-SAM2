"""Select a checkpoint by measured validation score within one explicit run."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path

EPOCH_RE = re.compile(r"_EP_(\d+)\.pth$")


def select_best(files: list[Path], scores: dict[str, float]) -> Path:
    if not files:
        raise ValueError("no checkpoints found")
    if set(scores) != {p.name for p in files}:
        raise ValueError("scores must cover exactly the checkpoints in this run")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in scores.values()
    ):
        raise ValueError("scores must be finite numbers")
    if any(EPOCH_RE.search(p.name) is None for p in files):
        raise ValueError("checkpoint filename has no epoch")
    # Lower epoch wins exact ties; no mtime or unrelated-run dependence.
    return max(files, key=lambda p: (scores[p.name], -int(EPOCH_RE.search(p.name)[1])))


def keep_best(
    ckpt_dir: Path,
    *,
    run_id: str,
    bypass_type: str,
    scores: dict[str, float],
    metric: str = "mean_id_iou_with_missing",
    superseded_dir: Path | None = None,
) -> dict:
    for value in (run_id, bypass_type):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("invalid run_id or bypass_type")
    pattern = re.compile(rf"bypass_{re.escape(bypass_type)}_{re.escape(run_id)}_EP_\d+\.pth")
    files = sorted(p for p in ckpt_dir.iterdir() if pattern.fullmatch(p.name) and p.is_file())
    best = select_best(files, scores)
    adapters = [p.with_suffix(".lora.pt") for p in files]
    has_adapters = any(p.exists() for p in adapters)
    if has_adapters and not all(p.is_file() for p in adapters):
        raise ValueError("incomplete adapter/checkpoint pairs in run")
    dest = superseded_dir or ckpt_dir / "_superseded" / run_id
    others = [p for p in files if p != best]
    if has_adapters:
        others += [p.with_suffix(".lora.pt") for p in others]
    # Preflight the entire operation; never partially archive on known collision.
    for p in others:
        if (dest / p.name).exists():
            raise FileExistsError(dest / p.name)
    manifest = ckpt_dir / "best_selection.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    result = {
        "run_id": run_id,
        "metric": metric,
        "mode": "max",
        "tie_break": "lowest_epoch",
        "scores": scores,
        "best": str(best),
        "sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
        "superseded": [str(dest / p.name) for p in others],
    }
    if has_adapters:
        adapter = best.with_suffix(".lora.pt")
        result.update(
            adapter=str(adapter), adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest()
        )
    for p in others:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest / p.name))
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    return result
