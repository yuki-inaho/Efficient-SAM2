"""手順15: train_student / keep_best_ckpt の単体テスト。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.train_experiment import absolutize_train_command, build_train_command  # noqa: E402
from tools.workflow.checkpoints import keep_best, select_best  # noqa: E402

CFG = {
    "repo_root": "repos/Efficient-SAM2",
    "python": "repos/Efficient-SAM2/.venv/bin/python",
    "train_script": "tools/train_bypass_all.py",
    "data": {
        "base_video_dir": "data/sav_format/JPEGImages",
        "input_mask_dir": "data/sav_format/Annotations",
        "video_list_file": "data/sav_format/train_sel.txt",
        "dataset": "sav_train",
    },
    "model": {"sam2_model": "base+", "bypass_type": "bottleneck"},
    "train": {"epochs": 1, "steps": 2, "lr": 1.0e-4},
    "flags": {"apply_bypass": True, "apply_WB": True, "use_wandb": False},
}


def test_build_train_command_contains_required_args() -> None:
    cmd = build_train_command(CFG, "runs/x/pred_masks")
    assert cmd[0].endswith("python")
    assert "--apply_bypass" in cmd and "--apply_WB" in cmd
    for flag in (
        "--train_epoch",
        "--train_step",
        "--lr",
        "--base_video_dir",
        "--input_mask_dir",
        "--video_list_file",
        "--output_mask_dir",
        "--dataset",
        "--sam2_model",
        "--bypass_type",
    ):
        assert flag in cmd, flag
    assert cmd[cmd.index("--sam2_model") + 1] == "base+"


def test_build_train_command_without_flags() -> None:
    cfg = {**CFG, "flags": {"apply_bypass": False, "apply_WB": False, "use_wandb": False}}
    cmd = build_train_command(cfg, "out")
    assert "--apply_bypass" not in cmd and "--apply_WB" not in cmd


def test_select_best_uses_score_not_epoch(tmp_path: Path) -> None:
    files = [tmp_path / f"bypass_bottleneck_run_EP_{i}.pth" for i in range(3)]
    for f in files:
        f.write_bytes(b"x")
    scores = dict(zip([f.name for f in files], [0.9, 0.6, 0.9]))
    assert select_best(files, scores) == files[0]


def test_keep_best_isolates_runs(tmp_path: Path) -> None:
    names = [
        "bypass_bottleneck_old_EP_9.pth",
        "bypass_bottleneck_run_EP_0.pth",
        "bypass_bottleneck_run_EP_1.pth",
        "bypass_bottleneck_base.pth",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"x")
    result = keep_best(
        tmp_path, run_id="run", bypass_type="bottleneck", scores={names[1]: 0.9, names[2]: 0.8}
    )
    assert result["best"].endswith(names[1])
    assert (tmp_path / names[0]).exists()
    assert (tmp_path / names[3]).exists()
    assert not (tmp_path / names[2]).exists()
    assert (tmp_path / "best_selection.json").exists()


@pytest.mark.parametrize("scores", [{}, {"bypass_bottleneck_run_EP_0.pth": float("nan")}])
def test_missing_or_nonfinite_score_preserves_files(tmp_path: Path, scores: dict) -> None:
    p = tmp_path / "bypass_bottleneck_run_EP_0.pth"
    p.write_bytes(b"x")
    with pytest.raises(ValueError):
        keep_best(tmp_path, run_id="run", bypass_type="bottleneck", scores=scores)
    assert p.read_bytes() == b"x"
    assert not (tmp_path / "best_selection.json").exists()


def test_archive_collision_fails_before_any_move(tmp_path: Path) -> None:
    names = [f"bypass_bottleneck_run_EP_{i}.pth" for i in range(3)]
    for name in names:
        (tmp_path / name).write_bytes(b"x")
    archive = tmp_path / "_superseded" / "run"
    archive.mkdir(parents=True)
    (archive / names[2]).write_bytes(b"old")
    with pytest.raises(FileExistsError):
        keep_best(
            tmp_path,
            run_id="run",
            bypass_type="bottleneck",
            scores=dict(zip(names, [0.9, 0.8, 0.7])),
        )
    assert all((tmp_path / name).exists() for name in names)
    assert (archive / names[2]).read_bytes() == b"old"


def test_keep_best_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no checkpoints"):
        keep_best(tmp_path, run_id="run", bypass_type="bottleneck", scores={})


def test_build_env_sets_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.train_experiment import build_env

    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = build_env("/repo/root")
    assert env["PYTHONPATH"] == "/repo/root"
    monkeypatch.setenv("PYTHONPATH", "/existing")
    env2 = build_env("/repo/root")
    assert env2["PYTHONPATH"] == f"/repo/root{os.pathsep}/existing"


def test_absolutize_train_command(tmp_path: Path) -> None:
    wd = tmp_path / "wd"
    repo = wd / "repos" / "X"
    repo.mkdir(parents=True)
    cmd = build_train_command(CFG, "runs/x/pred_masks")
    out = absolutize_train_command(cmd, wd, Path("repos/X"))
    assert out[0] == os.path.abspath(wd / CFG["python"])
    assert out[1] == os.path.abspath(wd / "repos/X" / "tools/train_bypass_all.py")
    idx = out.index("--base_video_dir")
    assert out[idx + 1] == os.path.abspath(wd / CFG["data"]["base_video_dir"])
    idx = out.index("--output_mask_dir")
    assert out[idx + 1] == os.path.abspath(wd / "runs/x/pred_masks")


def test_keep_best_archives_adapter_pairs(tmp_path: Path) -> None:
    names = [f"bypass_bottleneck_run_EP_{i}.pth" for i in range(2)]
    for name in names:
        p = tmp_path / name
        p.write_bytes(b"weights")
        p.with_suffix(".lora.pt").write_bytes(b"adapter")
    result = keep_best(
        tmp_path, run_id="run", bypass_type="bottleneck", scores={names[0]: 0.9, names[1]: 0.8}
    )
    assert Path(result["adapter"]).exists()
    assert not (tmp_path / names[1]).with_suffix(".lora.pt").exists()
    assert len(result["superseded"]) == 2


def test_incomplete_adapter_pair_prevents_selection(tmp_path: Path) -> None:
    names = [f"bypass_bottleneck_run_EP_{i}.pth" for i in range(2)]
    for name in names:
        (tmp_path / name).write_bytes(b"weights")
    (tmp_path / names[0]).with_suffix(".lora.pt").write_bytes(b"adapter")
    with pytest.raises(ValueError, match="adapter"):
        keep_best(
            tmp_path, run_id="run", bypass_type="bottleneck", scores={names[0]: 0.9, names[1]: 0.8}
        )
    assert all((tmp_path / n).exists() for n in names)


def test_wandb_flag_is_forwarded_when_enabled() -> None:
    cfg = {**CFG, "flags": {**CFG["flags"], "use_wandb": True}}
    assert "--use_wandb" in build_train_command(cfg, "out")
