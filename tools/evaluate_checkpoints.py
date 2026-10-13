"""Evaluate every checkpoint in a run on one fixed validation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tools.workflow.eval_masks import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args()
    run = args.run_dir.absolute()
    repo = args.repo.absolute()
    dataset = args.dataset_root.absolute()
    manifest = dataset / f"manifest_{args.video}.json"
    expected = len(json.loads(manifest.read_text())["files"])
    metadata = json.loads((run / "run.json").read_text())
    checkpoints = [Path(p) for p in metadata["checkpoints"]]
    dest = run / "validation"
    dest.mkdir(exist_ok=False)
    scores, records = {}, []
    for ckpt in checkpoints:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.workflow.gpu_gate",
                "--min-free-gb",
                "12",
                "--wait-seconds",
                "600",
            ],
            check=True,
        )
        out = dest / ckpt.stem
        command = [
            str(repo / ".venv/bin/python"),
            str(repo / "tools/vos_inference_custom.py"),
            "--dataset",
            "custom",
            "--sam2_model",
            "base+",
            "--base_video_dir",
            str(dataset / "JPEGImages"),
            "--input_mask_dir",
            str(dataset / "Annotations"),
            "--video_list_file",
            str(dataset / "val_sel.txt"),
            "--output_mask_dir",
            str(out),
            "--apply_bypass",
            "--apply_WB",
            "--bypass_ckpt_base",
            str(ckpt),
        ]
        if args.lora:
            adapter = ckpt.with_suffix(".lora.pt")
            if not adapter.is_file():
                raise FileNotFoundError(adapter)
            command += ["--lora_checkpoint", str(adapter)]
        env = dict(os.environ, PYTHONPATH=str(repo))
        log = dest / f"{ckpt.stem}.log"
        start = time.monotonic()
        with log.open("w") as handle:
            subprocess.run(
                command, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True
            )
        wall = time.monotonic() - start
        result = evaluate(
            out / args.video, dataset / "Annotations" / args.video, manifest_path=manifest
        )
        if result["frames"] != expected or result["skipped_correspondence"]:
            raise RuntimeError("evaluation coverage differs from validation manifest")
        (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        scores[ckpt.name] = result["mean_id_iou_with_missing"]
        records.append(
            {
                "checkpoint": str(ckpt),
                "command": command,
                "log": str(log),
                "metrics": str(out / "metrics.json"),
                "process_wall_seconds": wall,
                "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
            }
        )
        print(ckpt.name, scores[ckpt.name], flush=True)
    payload = {
        "run_id": metadata["run_id"],
        "metric": "mean_id_iou_with_missing",
        "mode": "max",
        "scores": scores,
        "candidates": records,
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "reference": "teacher_pseudo_gt",
        "split": "validation_for_selection",
    }
    (run / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
