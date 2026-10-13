"""Reproducible CPU video smoke evaluation on the bundled manual SA-V example.

Outputs and downloaded weights must live outside the checkout. This small TRAIN
sample is a regression fixture, not a held-out benchmark or downstream dataset validation.
The Efficient path uses the upstream SWR/SMR propagation method at native 1024.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def external_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError("Evaluation outputs/weights must be outside the repository")
    return resolved


def prepare_fixture(work_dir: Path, frame_count: int, object_ids: list[int]):
    video = ROOT / "sav_dataset/example/sav_000001.mp4"
    annotation = ROOT / "sav_dataset/example/sav_000001_manual.json"
    manual = json.loads(annotation.read_text())
    if manual["video_split"] != "train" or set(manual["masklet_type"]) != {"manual"}:
        raise ValueError("Unexpected fixture annotation provenance")
    if not 5 <= frame_count <= int(manual["video_frame_count"]):
        raise ValueError(
            "frame-count must cover at least one unprompted annotated frame"
        )
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("Duplicate object IDs")
    for object_id in object_ids:
        if object_id not in manual["masklet_id"]:
            raise ValueError(f"Unknown manual object ID {object_id}")
        if not coco_mask.decode(manual["masklet"][0][object_id]).any():
            raise ValueError(f"Object {object_id} is absent on prompt frame 0")
    frames = work_dir / "fixture" / f"frames_{frame_count}"
    frames.mkdir(parents=True, exist_ok=True)
    if len(list(frames.glob("*.jpg"))) != frame_count:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video),
                "-frames:v",
                str(frame_count),
                "-q:v",
                "2",
                "-start_number",
                "0",
                str(frames / "%05d.jpg"),
            ],
            check=True,
        )
    frame_files = sorted(frames.glob("*.jpg"))
    if len(frame_files) != frame_count:
        raise RuntimeError("Decoded frame count does not match requested protocol")
    eval_frames = list(range(4, min(frame_count, 4 * (len(manual["masklet"]) - 1)), 4))
    protocol = {
        "fixture": "sav_000001",
        "split": "train",
        "annotation_source": "manual",
        "video_sha256": sha256(video),
        "annotations_sha256": sha256(annotation),
        "frame_count": frame_count,
        "raw_frame_ids": list(range(frame_count)),
        "independent_object_tracks": True,
        "frame_sha256": [sha256(p) for p in frame_files],
        "annotation_sample_rate": 4,
        "object_ids": object_ids,
        "prompt": "full manual mask on raw frame 0; no subsequent prompts",
        "evaluated_raw_frame_ids": eval_frames,
        "mask_threshold": 0.0,
        "image_size": 1024,
        "limitations": [
            "Small public TRAIN fixture, not a held-out accuracy benchmark",
            "Prompt frame 0 excluded; the full-video final annotation is excluded",
            "No downstream ground truth and no downstream quality claim",
            "Single measured run; latency is a smoke measurement, not a benchmark",
        ],
    }
    return frames, manual, protocol


def build_predictor(checkpoint: Path, efficient: bool = True):
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        str(checkpoint),
        device="cpu",
        apply_postprocessing=False,
    )
    settings = dict(
        memory_temporal_stride_for_eval=1,
        print_WS=False,
        random_mask=False,
        uniform_mask=False,
        topk_mask=True,
        set_drop_ratio=0.95,
        MTP_theta=0.5,
        disable_WB=not efficient,
        sam2_model="base+",
        win_sel_layer=[6],
        fpn_feat_layer=None,
        scale_layer=None,
        Mem_Frame_Prune=False,
        num_frame_to_prune=0,
        WB_theta=0.7,
        obj_theta=5,
        sel_max_iou=False,
        dilate_mask=True,
        dilate_kernel_size=5,
        prune_memory=efficient,
    )
    for name, value in settings.items():
        setattr(predictor, name, value)
    predictor.init_memory_info(enable_MeP_info=efficient)
    if efficient:
        from build_model.window_bypass import build_WB_model
        from bypass.bypass_modeling import build_bypass_model

        args = SimpleNamespace(
            **settings,
            apply_WB=True,
            selected_layers=list(range(6, 21)),
            final_global_layer=20,
            inner_channel=448,
            WB_all_layer=False,
            small_bypass=False,
            Mem_filter=False,
            pool_memory=False,
        )
        build_WB_model(args, predictor, selected_layers=args.selected_layers)
        build_bypass_model(args, predictor, adapter_type="bottleneck", training=False)
        bypass_state = torch.load(
            ROOT / "bypass/ckpt/bypass_bottleneck_base.pth",
            map_location="cpu",
            weights_only=True,
        )
        predictor.image_encoder.trunk.blocks[20].bypass_branch.load_state_dict(
            bypass_state
        )
    predictor.eval()
    predictor.time_log = {}
    predictor.FW_time_log = {}
    predictor.WS_log = {"Sel": [], "Mask": [], "PF": []}
    for module in [
        predictor.image_encoder,
        predictor.memory_attention,
        predictor.memory_encoder,
        predictor.sam_mask_decoder,
    ]:
        module.FW_time_log = []
    predictor.image_encoder.trunk.FW_time_log = {}
    for block in predictor.image_encoder.trunk.blocks:
        block.FW_time_log = []
    for layer in predictor.memory_attention.layers:
        layer.cross_attn_image.drop_ratio_log = []
    return predictor


def mask_metrics(prediction: np.ndarray, ground_truth: np.ndarray):
    from sav_dataset.utils.sav_benchmark import Evaluator

    evaluator = Evaluator(name="fixture", obj_id="object")
    evaluator.feed_frame(prediction.astype(np.uint8), ground_truth.astype(np.uint8))
    # Read the one binary object's entries directly to include all-empty pairs.
    j = evaluator.object_iou.get(1, [1.0])[0]
    f = evaluator.boundary_f.get(1, [1.0])[0]
    return {"j": float(j), "f": float(f), "j_and_f": float((j + f) / 2)}


def compare_logits(current: np.ndarray, reference: np.ndarray):
    if current.shape != reference.shape:
        raise ValueError("Reference logit shape mismatch")
    mask, baseline = current > 0, reference > 0
    union = np.logical_or(mask, baseline).sum()
    return {
        "mask_iou": (
            float(np.logical_and(mask, baseline).sum() / union) if union else 1.0
        ),
        "pixel_disagreement": float(np.mean(mask != baseline)),
        "logit_mae": float(np.abs(current - reference).mean()),
        "logit_max_abs_error": float(np.abs(current - reference).max()),
        "reference_nonempty_to_empty": bool(baseline.any() and not mask.any()),
    }


def temporal_snapshot(predictor):
    info = predictor.mem_info
    windows = info.get("sel_win_id")
    if isinstance(windows, torch.Tensor):
        windows = windows.detach().cpu().tolist()
    used_masks = {
        str(obj): [
            mask.nonzero(as_tuple=False).flatten().cpu().tolist() for mask in layers
        ]
        for obj, layers in info.get("mem_masks", {}).items()
    }
    newest_masks = {
        str(obj): [
            mask.nonzero(as_tuple=False).flatten().cpu().tolist() for mask in stack[-1]
        ]
        for obj, stack in info.get("mem_masks_stack", {}).items()
        if stack
    }
    return {
        "selected_windows": windows,
        "none_means_all_25_windows": True,
        "memory_pruning_enabled": bool(info.get("enable_mem_prune", False)),
        "memory_mask_indices_by_internal_object": used_masks,
        "newest_memory_mask_indices_by_internal_object": newest_masks,
    }


def run(args, model_transform=None):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1234)
    np.random.seed(1234)
    work_dir = external_path(args.work_dir)
    checkpoint = external_path(args.checkpoint)
    output_dir = work_dir / "results" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    frames, manual, protocol = prepare_fixture(work_dir, args.frame_count, args.objects)
    protocol["efficient_swr_smr"] = not args.stock_sam2
    if args.initial_box is not None:
        if len(args.objects) != 1:
            raise ValueError("Box parity requires one explicitly selected object")
        protocol["prompt"] = (
            "box on raw frame 0; no subsequent prompts; separate temporal-parity protocol"
        )
        protocol["initial_box_xyxy_original_pixels"] = args.initial_box
    reference = external_path(args.reference_dir) if args.reference_dir else None
    if reference:
        ref_report = json.loads((reference / "report.json").read_text())
        # Informational scope wording is not an input or an inference setting.
        # Preserve the original frozen protocol verbatim after verifying every
        # behavioral field, even if a later release clarifies its limitations.
        comparison = {
            key: value for key, value in protocol.items() if key != "limitations"
        }
        reference_comparison = {
            key: value
            for key, value in ref_report["protocol"].items()
            if key != "limitations"
        }
        if reference_comparison != comparison:
            raise ValueError("Reference and candidate protocols differ")
        protocol = ref_report["protocol"]
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    start = time.perf_counter()
    predictor = build_predictor(checkpoint, efficient=not args.stock_sam2)
    build_seconds = time.perf_counter() - start
    variant_metadata = {"variant": args.variant, "execution": "CPU FP32"}
    if model_transform is not None:
        if args.variant != "fp32" or args.gguf or args.export_gguf:
            raise ValueError(
                "A model transform cannot be combined with built-in quantization"
            )
        predictor, variant_metadata = model_transform(predictor, output_dir)
    if args.variant != "fp32" or args.gguf:
        from efficient_sam2_quant import load_gguf_model, quantize_linear_modules

        mode = args.variant.removeprefix("convrot-")
        if mode not in {"w8a16", "w8a8"}:
            raise ValueError("A GGUF load requires a convrot variant")
        if args.gguf:
            predictor = load_gguf_model(
                predictor, external_path(args.gguf), mode=mode, inplace=True
            )
            variant_metadata["gguf_loaded_sha256"] = sha256(external_path(args.gguf))
        else:
            predictor = quantize_linear_modules(
                predictor,
                mode=mode,
                max_group_size=args.max_group_size,
                inplace=True,
                include=args.include,
                exclude=args.exclude,
            )
        variant_metadata["execution"] = (
            "CPU float-dequantized ConvRot correctness reference; not an INT8 speed claim"
        )
        variant_metadata["include"] = args.include
        variant_metadata["exclude"] = args.exclude
    if args.export_gguf:
        from efficient_sam2_quant import export_gguf

        variant_metadata["gguf_export"] = export_gguf(
            predictor,
            external_path(args.export_gguf),
            model_config={
                "variant": "base_plus",
                "image_size": 1024,
                "sam2_version": "2.1",
                "hydra_config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
                "efficient_swr": not args.stock_sam2,
                "efficient_smr": not args.stock_sam2,
                "bypass_type": "bottleneck",
                "selected_layers": list(range(6, 21)),
                "final_global_layer": 20,
                "memory_temporal_stride": 1,
                "memory_drop_ratio": 0.95,
                "wb_theta": 0.7,
                "dilate_mask": True,
                "dilate_kernel_size": 5,
            },
            source_sha256=sha256(checkpoint),
        )
    state_nbytes = sum(
        v.numel() * v.element_size() for v in predictor.state_dict().values()
    )
    timings, per_object_metrics, agreement = [], [], []
    initialize_seconds = prompt_seconds = 0.0
    with torch.inference_mode():
        for tracked_object in args.objects:
            init_start = time.perf_counter()
            state = predictor.init_state(
                str(frames), offload_video_to_cpu=True, offload_state_to_cpu=True
            )
            initialize_seconds += time.perf_counter() - init_start
            prompt_start = time.perf_counter()
            for object_id in [tracked_object]:
                if args.initial_box is None:
                    predictor.add_new_mask(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=object_id,
                        mask=coco_mask.decode(manual["masklet"][0][object_id]).astype(
                            bool
                        ),
                    )
                else:
                    predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=object_id,
                        box=np.asarray(args.initial_box, dtype=np.float32),
                    )
            prompt_seconds += time.perf_counter() - prompt_start
            propagation = (
                predictor.propagate_in_video
                if args.stock_sam2
                else predictor.propagate_in_video_wj
            )
            iterator = iter(propagation(state))
            while True:
                before_trace = (
                    temporal_snapshot(predictor) if args.dump_temporal else None
                )
                frame_start = time.perf_counter()
                try:
                    frame_id, object_ids, logits = next(iterator)
                except StopIteration:
                    break
                timings.append(
                    {
                        "frame": frame_id,
                        "object": tracked_object,
                        "seconds": time.perf_counter() - frame_start,
                    }
                )
                if args.dump_temporal:
                    trace = {
                        "frame": frame_id,
                        "before_step": before_trace,
                        "after_step": temporal_snapshot(predictor),
                    }
                    for index, object_id in enumerate(object_ids):
                        basename = f"frame_{frame_id:05d}_object_{object_id:03d}"
                        per_obj = state["output_dict_per_obj"][
                            state["obj_id_to_idx"][object_id]
                        ]
                        current = per_obj["cond_frame_outputs"].get(frame_id)
                        if current is None:
                            current = per_obj["non_cond_frame_outputs"][frame_id]
                        np.save(
                            output_dir / f"{basename}_low_res.npy",
                            current["pred_masks"].detach().cpu().float().numpy(),
                            allow_pickle=False,
                        )
                        raw = logits[index, 0].detach().cpu().float().numpy()
                        np.save(output_dir / f"{basename}.npy", raw, allow_pickle=False)
                        Image.fromarray((raw > 0).astype(np.uint8) * 255).save(
                            output_dir / f"{basename}.png"
                        )
                        trace["decoder_scalars"] = {
                            key: current[key].detach().cpu().float().tolist()
                            for key in ["object_score_logits", "best_iou"]
                            if key in current
                        }
                        (output_dir / f"{basename}_trace.json").write_text(
                            json.dumps(trace, indent=2) + "\n"
                        )
                if frame_id not in protocol["evaluated_raw_frame_ids"]:
                    continue
                for object_index, object_id in enumerate(object_ids):
                    raw_logits = (
                        logits[object_index, 0].cpu().numpy().astype(np.float32)
                    )
                    if not np.isfinite(raw_logits).all():
                        raise ValueError("Non-finite output logits")
                    basename = f"frame_{frame_id:05d}_object_{object_id:03d}"
                    np.save(
                        output_dir / f"{basename}.npy", raw_logits, allow_pickle=False
                    )
                    Image.fromarray((raw_logits > 0).astype(np.uint8) * 255).save(
                        output_dir / f"{basename}.png"
                    )
                    ground_truth = coco_mask.decode(
                        manual["masklet"][frame_id // 4][object_id]
                    ).astype(bool)
                    item = {
                        "frame": frame_id,
                        "object": object_id,
                        **mask_metrics(raw_logits > 0, ground_truth),
                    }
                    per_object_metrics.append(item)
                    if reference:
                        ref_logits = np.load(
                            reference / f"{basename}.npy", allow_pickle=False
                        )
                        agreement.append(
                            {
                                "frame": frame_id,
                                "object": object_id,
                                **compare_logits(raw_logits, ref_logits),
                            }
                        )
                    print(json.dumps({"event": "evaluated", **item}), flush=True)
    report = {
        "protocol": protocol,
        "model": "Efficient-SAM2 SAM2.1 Hiera base+",
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "bypass_sha256": sha256(ROOT / "bypass/ckpt/bypass_bottleneck_base.pth"),
        "state_tensor_bytes": state_nbytes,
        "variant": variant_metadata,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cpu_threads": args.threads,
            "platform": platform.platform(),
            "timing_context": "Shared machine with concurrent work; wall times are functional observations, not controlled speed comparisons",
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "build_seconds": build_seconds,
        "initialize_seconds": initialize_seconds,
        "prompt_seconds": prompt_seconds,
        "propagation_timings": timings,
        "propagation_seconds": sum(row["seconds"] for row in timings),
        "per_object_metrics": per_object_metrics,
        "mean_metrics": {
            key: float(np.mean([row[key] for row in per_object_metrics]))
            for key in ["j", "f", "j_and_f"]
        },
        "agreement": agreement,
    }
    if agreement:
        ious = [row["mask_iou"] for row in agreement]
        report["agreement_summary"] = {
            "mean_mask_iou": float(np.mean(ious)),
            "p5_mask_iou": float(np.percentile(ious, 5)),
            "min_mask_iou": float(min(ious)),
            "catastrophic_empty_rate": float(
                np.mean([row["reference_nonempty_to_empty"] for row in agreement])
            ),
            "mean_pixel_disagreement": float(
                np.mean([row["pixel_disagreement"] for row in agreement])
            ),
            "mean_logit_mae": float(np.mean([row["logit_mae"] for row in agreement])),
        }
        report["paired_mean_metric_delta"] = {
            key: report["mean_metrics"][key] - ref_report["mean_metrics"][key]
            for key in report["mean_metrics"]
        }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(output_dir / "report.json"),
                "mean_metrics": report["mean_metrics"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--name", default="efficient-fp32")
    parser.add_argument("--frame-count", type=int, default=17)
    parser.add_argument("--objects", type=int, nargs="+", default=[1])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stock-sam2", action="store_true")
    parser.add_argument(
        "--variant", choices=["fp32", "convrot-w8a16", "convrot-w8a8"], default="fp32"
    )
    parser.add_argument("--max-group-size", type=int, default=256)
    parser.add_argument(
        "--include",
        nargs="+",
        help="Quantized module-name globs (default: all Linear modules)",
    )
    parser.add_argument("--exclude", nargs="+", help="Excluded module-name globs")
    parser.add_argument("--reference-dir")
    parser.add_argument("--gguf")
    parser.add_argument("--export-gguf")
    parser.add_argument(
        "--initial-box",
        type=float,
        nargs=4,
        help="Separate box-initialized temporal parity protocol (one object only)",
    )
    parser.add_argument(
        "--dump-temporal",
        action="store_true",
        help="Dump every frame's logits/mask and SWR/SMR index trace",
    )
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("--name must be a simple directory name")
    run(args)


if __name__ == "__main__":
    main()
