"""Supervised SAM2 decoder training on reviewed targets under synthetic occlusion.

Uses explicit positive/negative points and boxes. Keeps pre-occlusion targets.
Validation selects checkpoints; target-disjoint test pairs never update weights.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.workflow.occlusion_augmentation import augment_thin_green  # noqa: E402
from tools.workflow.occlusion_training_data import load_reviewed_pairs  # noqa: E402
from tools.workflow.supervised_mask_loss import supervised_mask_loss  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forward_mask(predictor, row, rgb):
    """Encoder is frozen; invoke decoder directly, preserving its autograd graph."""
    predictor.set_image(rgb)
    model = predictor.model
    points = torch.tensor(row["prompt_points"], device="cuda", dtype=torch.float32)
    labels = torch.tensor(row["prompt_labels"], device="cuda", dtype=torch.int32)
    box = torch.tensor(row["local_box"], device="cuda", dtype=torch.float32).reshape(2, 2)
    coords = (
        torch.cat([box, points], dim=0)
        / torch.tensor([rgb.shape[1], rgb.shape[0]], device="cuda")
        * model.image_size
    )
    labels = torch.cat([torch.tensor([2, 3], device="cuda"), labels])[None]
    with torch.no_grad():
        sparse, dense = model.sam_prompt_encoder(
            points=(coords[None], labels), boxes=None, masks=None
        )
    masks, _, _, _ = model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"],
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        high_res_features=predictor._features["high_res_feats"],
    )
    return F.interpolate(masks.float(), size=rgb.shape[:2], mode="bilinear", align_corners=False)


def metrics(pred, target, occ):
    inter = np.count_nonzero(pred & target)
    hidden = (occ > 0) & target
    _, _, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8))
    return {
        "iou": float(inter / np.count_nonzero(pred | target)),
        "dice": float(2 * inter / (pred.sum() + target.sum())),
        "hidden_recall": float(np.count_nonzero(pred & hidden) / hidden.sum())
        if hidden.any()
        else None,
        "outside_target_pixels": int(np.count_nonzero(pred & ~target)),
        "substantial_components": int(
            sum(s[cv2.CC_STAT_AREA] >= max(5, target.sum() * 0.01) for s in stats[1:])
        ),
    }


def evaluate(predictor, rows, *, seeds, out=None, tag="eval"):
    results = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in rows:
            for seed in [None, *seeds]:
                if seed is None:
                    rgb, target, occ = (
                        row["rgb"],
                        row["target"],
                        np.zeros(row["target"].shape, np.uint8),
                    )
                else:
                    rgb, target, occ, _ = augment_thin_green(
                        row["rgb"], row["target"], seed=seed, probability=1
                    )
                pred = forward_mask(predictor, row, rgb)[0, 0].cpu().numpy() > 0
                result = {
                    "object_id": row["object_id"],
                    "condition": "clean" if seed is None else "synthetic",
                    "seed": seed,
                    **metrics(pred, target, occ),
                }
                results.append(result)
                if out is not None:
                    name = f"{row['object_id']}_{'clean' if seed is None else seed}"
                    Image.fromarray(pred.astype(np.uint8) * 255).save(
                        out / f"{tag}_{name}_pred.png"
                    )
                    if tag == "base":
                        Image.fromarray(rgb).save(out / f"input_{name}.png")
                        Image.fromarray(target.astype(np.uint8) * 255).save(
                            out / f"target_{name}.png"
                        )
                        Image.fromarray(occ).save(out / f"occluder_{name}.png")
    return results


def summarize(rows):
    summary = {}
    for condition in ("clean", "synthetic"):
        selected = [r for r in rows if r["condition"] == condition]
        summary[condition] = {
            k: float(np.mean([r[k] for r in selected]))
            for k in ("iou", "dice", "outside_target_pixels", "substantial_components")
        }
        if condition == "synthetic":
            summary[condition]["hidden_recall"] = float(
                np.mean([r["hidden_recall"] for r in selected])
            )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=320)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negative-weight", type=float, default=8.0)
    p.add_argument(
        "--evaluation-role", choices=["heldout", "reused_development"], default="heldout"
    )
    args = p.parse_args()
    if args.steps < 1 or args.lr <= 0:
        raise ValueError("positive steps/lr required")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pairs = load_reviewed_pairs(args.manifest)
    split = {s: [r for r in pairs if r["split"] == s] for s in ("train", "val", "test")}
    if any(not v for v in split.values()):
        raise ValueError("all three disjoint splits are required")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "evaluation").mkdir()
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        str(args.checkpoint),
        device="cuda",
        apply_postprocessing=False,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    decoder = model.sam_mask_decoder
    for name, param in decoder.named_parameters():
        # These projections run inside frozen forward_image, outside decoder forward.
        if not name.startswith(("conv_s0.", "conv_s1.")):
            param.requires_grad_(True)
    predictor = SAM2ImagePredictor(model)
    optimizer = torch.optim.AdamW(
        [p for p in decoder.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    writer = SummaryWriter(str(args.out / "tensorboard"))
    initial = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
    (args.out / "run.json").write_text(
        json.dumps(
            {
                "status": "TRAINING_PILOT",
                "arguments": {
                    k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                },
                "manifest_sha256": sha(args.manifest),
                "base_checkpoint_sha256": sha(args.checkpoint),
                "split_objects": {s: [r["object_id"] for r in v] for s, v in split.items()},
                "loss": "hidden-weighted BCE + Dice + explicit negative neighborhoods/point loss",
                "runner_sha256": sha(Path(__file__)),
                "trainable": "mask decoder excluding cached high-res projections; encoders frozen",
                "negative_points": (
                    "preserved original scene prompts; never mark synthetic occluder as background"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    # Validation only during optimization. Test is evaluated after selection is frozen.
    val_base = evaluate(predictor, split["val"], seeds=[10001, 10002])
    best_value = summarize(val_base)["synthetic"]["iou"]
    best_step = 0
    torch.save(
        {"decoder": initial, "step": 0, "manifest_sha256": sha(args.manifest)},
        args.out / "best_decoder.pt",
    )
    rng = np.random.default_rng(args.seed)
    history = []
    started = time.monotonic()
    gradient_evidence = None
    for step in range(1, args.steps + 1):
        row = split["train"][int(rng.integers(len(split["train"])))]
        seed = args.seed * 100000 + step
        rgb, target, occ, aug = augment_thin_green(
            row["rgb"], row["target"], seed=seed, probability=0.7
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = forward_mask(predictor, row, rgb)
            loss = supervised_mask_loss(
                logits,
                target,
                occ,
                row["prompt_points"],
                row["prompt_labels"],
                negative_weight=args.negative_weight,
            )
        if not torch.isfinite(loss):
            raise ValueError("non-finite supervised loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in decoder.parameters() if p.requires_grad], 1.0
        )
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("missing/nonfinite decoder gradient")
        optimizer.step()
        if step == 1:
            difference = sum(
                float((v.detach().cpu() - initial[k]).abs().sum())
                for k, v in decoder.state_dict().items()
            )
            if difference <= 0:
                raise ValueError("decoder did not update")
            gradient_evidence = {
                "gradient_norm": float(norm),
                "parameter_l1_change": difference,
                "loss": float(loss),
            }
        entry = {
            "step": step,
            "object_id": row["object_id"],
            "seed": seed,
            "loss": float(loss.detach()),
            "gradient_norm": float(norm),
            "augmentation": aug,
        }
        history.append(entry)
        writer.add_scalar("train/supervised_loss", entry["loss"], step)
        if step % 20 == 0 or step == 1:
            print(
                f"step {step}/{args.steps} loss={entry['loss']:.5f} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
        if step % 80 == 0 or step == args.steps:
            val = evaluate(predictor, split["val"], seeds=[10001, 10002])
            summary = summarize(val)
            score = summary["synthetic"]["iou"]
            writer.add_scalar("val/synthetic_iou", score, step)
            clean_drop = summarize(val_base)["clean"]["iou"] - summary["clean"]["iou"]
            if score > best_value and clean_drop <= 0.02:
                best_value = score
                best_step = step
                torch.save(
                    {
                        "decoder": decoder.state_dict(),
                        "step": step,
                        "manifest_sha256": sha(args.manifest),
                    },
                    args.out / "best_decoder.pt",
                )
            (args.out / f"validation_{step:04d}.json").write_text(
                json.dumps({"summary": summary, "rows": val}, indent=2) + "\n"
            )
            (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    torch.save(
        {
            "decoder": decoder.state_dict(),
            "step": args.steps,
            "manifest_sha256": sha(args.manifest),
        },
        args.out / "last_decoder.pt",
    )
    # Freeze selection before inspecting held-out test results.
    selected = torch.load(args.out / "best_decoder.pt", map_location="cpu", weights_only=True)
    decoder.load_state_dict(initial)
    base = evaluate(
        predictor,
        split["test"],
        seeds=[20001, 20002, 20003, 20004],
        out=args.out / "evaluation",
        tag="base",
    )
    decoder.load_state_dict(selected["decoder"])
    tuned = evaluate(
        predictor,
        split["test"],
        seeds=[20001, 20002, 20003, 20004],
        out=args.out / "evaluation",
        tag="tuned",
    )
    # Re-load from disk and reproduce the same tensor, not just successful loading.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        row = split["test"][0]
        before = forward_mask(predictor, row, row["rgb"]).clone()
        decoder.load_state_dict(
            torch.load(args.out / "best_decoder.pt", map_location="cpu", weights_only=True)[
                "decoder"
            ]
        )
        after = forward_mask(predictor, row, row["rgb"])
    if not torch.equal(before, after):
        raise ValueError("checkpoint reload changed prediction")
    report = {
        "status": "PILOT_EVALUATED_NOT_FULL_DATASET_APPROVAL",
        "best_step": best_step,
        "evaluation_role": args.evaluation_role,
        "base": summarize(base),
        "tuned": summarize(tuned),
        "base_rows": base,
        "tuned_rows": tuned,
        "gradient_evidence": gradient_evidence,
        "reload_prediction_equal": True,
        "wall_seconds": time.monotonic() - started,
        "checkpoint_sha256": sha(args.out / "best_decoder.pt"),
        "limitations": [
            "same scene, target-disjoint pilot only",
            "visually corrected SAM teachers; not independent benchmark",
            "real thin-occlusion and independent-scene evaluation still required",
        ],
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    run_path = args.out / "run.json"
    run_record = json.loads(run_path.read_text())
    run_record.update(status=report["status"], report_path="report.json", selected_step=best_step)
    run_path.write_text(json.dumps(run_record, indent=2) + "\n")
    writer.close()
    print(
        json.dumps({k: report[k] for k in ("status", "best_step", "base", "tuned")}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
