"""Real entry points must reject unreviewed data before writing or launching."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.workflow.build_sav_dataset import build_sav_dataset


def inputs(tmp_path):
    masks, rgb = tmp_path / "masks", tmp_path / "rgb"
    masks.mkdir()
    rgb.mkdir()
    for i in range(2):
        a = np.zeros((8, 8), np.uint16)
        a[:4, :4] = 1
        if i:
            a[4:, 4:] = 2
        Image.fromarray(a).save(masks / f"frame_{i:04d}_idmap.png")
        Image.new("RGB", (8, 8), (i, 40, 80)).save(rgb / f"{i}.jpg")
    return masks, rgb


def test_export_missing_approval_rejected_before_output(tmp_path):
    masks, rgb = inputs(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="approval"):
        build_sav_dataset(masks, rgb, out, "clip")
    assert not out.exists()


def test_late_visible_object_not_silently_dropped(tmp_path, approve_dataset):
    masks, rgb = inputs(tmp_path)
    approval = approve_dataset(masks, rgb)
    manifest = build_sav_dataset(
        masks, rgb, tmp_path / "out", "clip", quality_approval=approval
    )
    assert manifest["dropped_object_ids"] == []
    assert (tmp_path / "out/Annotations/clip/002/1.png").exists()


@pytest.mark.parametrize("layout", ["per_object", "packed"])
def test_approved_export_reverified_from_actual_pixels(
    tmp_path, approve_dataset, layout
):
    from tools.workflow.quality_preflight import verify_training_inputs

    masks, rgb = inputs(tmp_path)
    approval = approve_dataset(masks, rgb)
    out = tmp_path / "out"
    build_sav_dataset(masks, rgb, out, "clip", layout=layout, quality_approval=approval)
    selection = out / "train.txt"
    selection.write_text("__pad_000__\nclip\n")
    data = {
        "base_video_dir": str(out / "JPEGImages"),
        "input_mask_dir": str(out / "Annotations"),
        "video_list_file": str(selection),
    }
    assert verify_training_inputs(data)["status"] == "VERIFIED_CURRENT_INPUTS"
    target = out / "Annotations/clip" / ("1.png" if layout == "packed" else "002/1.png")
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(target)
    with pytest.raises(ValueError, match="differs from approved"):
        verify_training_inputs(data)


@pytest.mark.parametrize(
    "change",
    [
        "source_rgb",
        "source_mask",
        "review",
        "evidence",
        "inventory",
        "boundary",
        "tracks",
        "unresolved",
        "extra_mask",
    ],
)
def test_approval_recomputed_and_bound(tmp_path, approve_dataset, change):
    from tools.workflow.annotation_quality_gate import sha256
    from tools.workflow.quality_preflight import require_annotation_approval

    masks, rgb = inputs(tmp_path)
    approval = approve_dataset(masks, rgb)
    bundle = json.loads(approval.read_text())
    review_path = Path(bundle["review_path"])
    review = json.loads(review_path.read_text())
    if change == "source_rgb":
        Image.new("RGB", (8, 8), "red").save(rgb / "0.jpg")
    elif change == "source_mask":
        Image.fromarray(np.ones((8, 8), np.uint16)).save(masks / "frame_0000_idmap.png")
    elif change == "review":
        review["frames"][0]["status"] = "draft"
        review_path.write_text(json.dumps(review))
    elif change == "extra_mask":
        Image.fromarray(np.ones((8, 8), np.uint16)).save(masks / "frame_0002_idmap.png")
    else:
        if change == "evidence":
            review["frames"][0]["evidence"][0]["sha256"] = "stale"
        if change == "inventory":
            review["frames"][0]["rgb_inventory_complete"] = False
        if change == "boundary":
            review["frames"][0]["boundaries_accepted"] = False
        if change == "tracks":
            review["frames"][0]["tracks_accepted"] = False
        if change == "unresolved":
            review["frames"][0]["unresolved_items"] = ["unknown fruit"]
        review_path.write_text(json.dumps(review))
        bundle["review_sha256"] = sha256(review_path)
        approval.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="quality"):
        require_annotation_approval(approval)
    with pytest.raises(ValueError, match="quality"):
        build_sav_dataset(
            masks, rgb, tmp_path / "out", "clip", quality_approval=approval
        )
    assert not (tmp_path / "out").exists()


def test_cached_pass_not_an_approval(tmp_path):
    masks, rgb = inputs(tmp_path)
    approval = tmp_path / "fake.json"
    approval.write_text(json.dumps({"status": "PASS", "integrity": "PASS"}))
    with pytest.raises(ValueError, match="quality"):
        build_sav_dataset(
            masks, rgb, tmp_path / "out", "clip", quality_approval=approval
        )
    assert not (tmp_path / "out").exists()


def test_different_rgb_directory_rejected(tmp_path, approve_dataset):
    import shutil

    masks, rgb = inputs(tmp_path)
    approval = approve_dataset(masks, rgb)
    other = tmp_path / "other"
    shutil.copytree(rgb, other)
    with pytest.raises(ValueError, match="sources differ"):
        build_sav_dataset(
            masks, other, tmp_path / "out", "clip", quality_approval=approval
        )
    assert not (tmp_path / "out").exists()


def test_training_source_mask_changed_after_export(tmp_path, approve_dataset):
    from tools.workflow.quality_preflight import verify_training_inputs

    masks, rgb = inputs(tmp_path)
    approval = approve_dataset(masks, rgb)
    out = tmp_path / "out"
    build_sav_dataset(masks, rgb, out, "clip", quality_approval=approval)
    selection = out / "train.txt"
    selection.write_text("clip\n")
    Image.fromarray(np.ones((8, 8), np.uint16)).save(masks / "frame_0000_idmap.png")
    with pytest.raises(ValueError, match="quality approval failed"):
        verify_training_inputs(
            {
                "base_video_dir": str(out / "JPEGImages"),
                "input_mask_dir": str(out / "Annotations"),
                "video_list_file": str(selection),
            }
        )


def test_cli_cannot_export_without_approval(tmp_path):
    import subprocess

    masks, rgb = inputs(tmp_path)
    out = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.workflow.build_sav_dataset",
            "--idmap-dir",
            str(masks),
            "--rgb-dir",
            str(rgb),
            "--out-root",
            str(out),
            "--video-name",
            "clip",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0 and "--quality-approval" in result.stderr
    assert not out.exists()


@pytest.mark.parametrize("change", ["rgb", "extra_mask", "object_ids", "receipt"])
def test_training_export_tampering_rejected(tmp_path, approve_dataset, change):
    from tools.workflow.quality_preflight import verify_training_inputs

    masks, rgb = inputs(tmp_path)
    out = tmp_path / "out"
    approval = approve_dataset(masks, rgb)
    build_sav_dataset(masks, rgb, out, "clip", quality_approval=approval)
    selection = out / "train.txt"
    selection.write_text("clip\n")
    if change == "rgb":
        Image.new("RGB", (8, 8), "red").save(out / "JPEGImages/clip/0.jpg")
    elif change == "extra_mask":
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(
            out / "Annotations/clip/001/9.png"
        )
    else:
        path = out / "manifest_clip.json"
        manifest = json.loads(path.read_text())
        if change == "object_ids":
            manifest["quality_receipt"]["object_ids"] = [1]
        else:
            manifest["quality_receipt"]["approval_sha256"] = "old"
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="quality"):
        verify_training_inputs(
            {
                "base_video_dir": str(out / "JPEGImages"),
                "input_mask_dir": str(out / "Annotations"),
                "video_list_file": str(selection),
            }
        )
