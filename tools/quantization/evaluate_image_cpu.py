"""Dump one all-window Efficient WB_Hiera box-prompt reference for native parity.

This is image-component validation. It does not exercise temporal SWR or SMR.
Use the same image bytes, original-pixel XYXY box, model, and artifact in the
native implementation. All outputs must be outside the checkout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from sam2.sam2_image_predictor import SAM2ImagePredictor
from tools.quantization.evaluate_cpu import build_predictor, external_path, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--box", type=float, nargs=4, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gguf")
    parser.add_argument("--mode", choices=["w8a16", "w8a8"], default="w8a16")
    args = parser.parse_args()
    output = external_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1234)
    model = build_predictor(external_path(args.checkpoint), efficient=True)
    if args.gguf:
        from efficient_sam2_quant import load_gguf_model

        model = load_gguf_model(
            model, external_path(args.gguf), mode=args.mode, inplace=True
        )
    predictor = SAM2ImagePredictor(model, max_hole_area=0, max_sprinkle_area=0)
    image_path = Path(args.image).resolve()
    image = np.asarray(Image.open(image_path).convert("RGB"))
    raw_decoder = {}

    def capture_decoder(_module, _args, result):
        raw_decoder["low_res_logits"] = result[0].detach().cpu().float().numpy().copy()
        raw_decoder["iou_scores"] = result[1].detach().cpu().float().numpy().copy()

    handle = model.sam_mask_decoder.register_forward_hook(capture_decoder)
    start = time.perf_counter()
    with torch.inference_mode():
        preprocessed = (
            predictor._transforms(image).unsqueeze(0).cpu().numpy().astype("<f4")
        )
        np.save(output / "preprocessed_image.npy", preprocessed, allow_pickle=False)
        preprocessed.tofile(output / "preprocessed_image.bin")
        np.save(output / "input_rgb.npy", image, allow_pickle=False)
        predictor.set_image(image)
        for index, feature in enumerate(predictor._features["high_res_feats"]):
            np.save(
                output / f"high_res_feature_{index}.npy",
                feature.cpu().numpy(),
                allow_pickle=False,
            )
        np.save(
            output / "image_embedding.npy",
            predictor.get_image_embedding().cpu().numpy(),
            allow_pickle=False,
        )
        logits, scores, clamped_low_res = predictor.predict(
            box=np.asarray(args.box, dtype=np.float32),
            multimask_output=False,
            return_logits=True,
            normalize_coords=True,
        )
    elapsed = time.perf_counter() - start
    handle.remove()
    for name, value in raw_decoder.items():
        np.save(output / f"{name}.npy", value, allow_pickle=False)
    np.save(output / "original_res_logits.npy", logits, allow_pickle=False)
    np.save(output / "clamped_low_res_logits.npy", clamped_low_res, allow_pickle=False)
    np.save(output / "iou_scores_public.npy", scores, allow_pickle=False)
    Image.fromarray((logits[0] > 0).astype(np.uint8) * 255).save(output / "mask.png")
    metadata = {
        "purpose": "Efficient WB_Hiera first-image all-25-packed-window box parity; no temporal memory propagation",
        "input_image_sha256": sha256(image_path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "box_xyxy_original_pixels": args.box,
        "multimask_output": False,
        "mask_threshold": 0.0,
        "max_hole_area": 0,
        "max_sprinkle_area": 0,
        "model": "SAM2.1 Hiera base+",
        "image_size": 1024,
        "variant": args.mode if args.gguf else "fp32",
        "checkpoint_sha256": sha256(Path(args.checkpoint)),
        "gguf_sha256": sha256(Path(args.gguf)) if args.gguf else None,
        "preprocessed_layout": "NCHW little-endian float32 [1,3,1024,1024]",
        "preprocessed_sha256": sha256(output / "preprocessed_image.bin"),
        "scores": scores.tolist(),
        "torch": torch.__version__,
        "elapsed_including_dumps_seconds": elapsed,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
