"""Render every evaluated case, including failures, as a labeled comparison movie."""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    cv2.setNumThreads(1)
    report = json.loads((args.run / "report.json").read_text())
    ep = args.run / "evaluation"
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1536x640",
        "-r",
        "25",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for base, tuned in zip(report["base_rows"], report["tuned_rows"], strict=True):
        if (base["object_id"], base["seed"]) != (tuned["object_id"], tuned["seed"]):
            raise ValueError("evaluation pair mismatch")
        name = f"{tuned['object_id']}_{'clean' if tuned['seed'] is None else tuned['seed']}"
        rgb = np.array(Image.open(ep / f"input_{name}.png"))
        truth = np.array(Image.open(ep / f"target_{name}.png")) > 0
        panel = Image.new("RGB", (1536, 640), "#17202b")
        draw = ImageDraw.Draw(panel)
        draw.text(
            (16, 8),
            "PILOT: synthetic thin green occlusion / target-disjoint objects in one scene",
            font=font,
            fill="white",
        )
        draw.text(
            (16, 38),
            f"{name}   |   Experimental comparison: visual review remains required",
            font=font,
            fill="#ffcf80",
        )
        for j, tag in enumerate(("input", "base", "tuned")):
            view = rgb.copy()
            if tag != "input":
                mask = np.array(Image.open(ep / f"{tag}_{name}_pred.png")) > 0
                view[mask] = (view[mask] * 0.5 + np.array([255, 0, 255]) * 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(
                    truth.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(view, contours, -1, (255, 255, 0), 2)
            panel.paste(Image.fromarray(view).resize((512, 512)), (j * 512, 80))
            label = [
                "Input image",
                f"Base SAM   IoU {base['iou']:.3f}",
                f"Fine-tuned SAM   IoU {tuned['iou']:.3f}",
            ][j]
            draw.text((j * 512 + 10, 602), label, font=font, fill="white")
        raw = np.array(panel).tobytes()
        for _ in range(25):
            proc.stdin.write(raw)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
