"""手順15: Hydra で Efficient-SAM2 bypass 学習を起動・管理するラッパー。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import hydra
from omegaconf import DictConfig, OmegaConf

# スクリプトとして直接実行(python src/train_student.py)された場合でも
# `from src.keep_best_ckpt import ...` を解決できるようにする
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_train_command(cfg: dict, run_outputs: str) -> list[str]:
    """設定から repos/Efficient-SAM2/tools/train_bypass_all.py の引数列を組み立てる。"""
    cmd = [cfg["python"], cfg["train_script"]]
    if cfg["flags"]["apply_bypass"]:
        cmd.append("--apply_bypass")
    if cfg["flags"]["apply_WB"]:
        cmd.append("--apply_WB")
    if cfg["flags"].get("use_wandb", False):
        cmd.append("--use_wandb")
    cmd += [
        "--train_epoch",
        str(int(cfg["train"]["epochs"])),
        "--train_step",
        str(int(cfg["train"]["steps"])),
        "--lr",
        str(float(cfg["train"]["lr"])),
        "--base_video_dir",
        cfg["data"]["base_video_dir"],
        "--input_mask_dir",
        cfg["data"]["input_mask_dir"],
        "--video_list_file",
        cfg["data"]["video_list_file"],
        "--output_mask_dir",
        run_outputs,
        "--dataset",
        cfg["data"]["dataset"],
        "--sam2_model",
        cfg["model"]["sam2_model"],
        "--bypass_type",
        cfg["model"]["bypass_type"],
    ]
    return cmd


def absolutize_train_command(cmd: list[str], workdir: Path, repo_root: Path) -> list[str]:
    """subprocess(cwd=repo_root) で安全に実行できるようパス引数を絶対化する。

    注意: venv の python はシンボリックリンクのため `resolve()` を使わない(環境が壊れる)。
    """
    out = list(cmd)
    out[0] = os.path.abspath(workdir / out[0])
    out[1] = os.path.abspath(workdir / repo_root / out[1])
    path_flags = {"--base_video_dir", "--input_mask_dir", "--video_list_file", "--output_mask_dir"}
    for i, arg in enumerate(out):
        if i > 0 and out[i - 1] in path_flags:
            out[i] = os.path.abspath(workdir / out[i])
    return out


def build_env(repo_root_abs: str) -> dict:
    """train スクリプトが `build_model` 等を repo ルートから import できるよう
    PYTHONPATH を付与する。"""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root_abs}{os.pathsep}{existing}" if existing else repo_root_abs
    return env


def _check_workdir() -> None:
    if not Path("pyproject.toml").exists() or not Path("tools/train_bypass_all.py").exists():
        raise RuntimeError(
            "このラッパーは作業ディレクトリ(workdir)を cwd として実行する必要があります"
        )


@hydra.main(version_base=None, config_path="conf", config_name="train_bypass")
def main(cfg: DictConfig) -> None:
    _check_workdir()
    flat = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(flat, dict)
    run_dir = Path(flat["runs_root"]) / flat["run_name"]
    run_outputs = str(run_dir / "pred_masks")
    workdir = Path.cwd()
    repo_root = Path(flat["repo_root"])
    command = absolutize_train_command(build_train_command(flat, run_outputs), workdir, repo_root)
    print("train command:", " ".join(command), flush=True)
    if flat.get("dry_run", False):
        print("dry_run=true のため実行しません", flush=True)
        return

    # Reject accidental reuse before touching any outputs.
    run_dir.mkdir(parents=True, exist_ok=False)
    run_dir = run_dir.absolute()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    (run_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "started",
                "selection_status": "pending_validation",
            },
            indent=2,
        )
        + "\n"
    )
    gate = flat["gate"]
    if gate.get("enabled", True):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.workflow.gpu_gate",
                "--min-free-gb",
                str(gate["min_free_gb"]),
                "--wait-seconds",
                str(gate["wait_seconds"]),
            ],
            check=True,
        )

    log_path = run_dir / "train.log"
    repo_root_abs = os.path.abspath(Path.cwd() / repo_root)
    # train スクリプトは ./result/tmp/<timestamp>.log を書く(環境前準備として作成)
    (Path(repo_root_abs) / "result" / "tmp").mkdir(parents=True, exist_ok=True)
    env = build_env(repo_root_abs)
    env["EFFICIENTSAM_RUN_DIR"] = str(run_dir)
    env["EFFICIENTSAM_RUN_ID"] = run_id
    with log_path.open("w") as log:
        proc = subprocess.run(
            command,
            cwd=repo_root_abs,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    print(f"training exit={proc.returncode} log={log_path}", flush=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    checkpoints = sorted((run_dir / "checkpoints").glob("*_EP_*.pth"))
    if len(checkpoints) != int(flat["train"]["epochs"]):
        raise RuntimeError("training did not produce one checkpoint per epoch in this run")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "training_complete",
                "selection_status": "pending_validation",
                "checkpoints": [str(p) for p in checkpoints],
                "log": str(log_path),
                "tensorboard": str(run_dir / "tensorboard"),
            },
            indent=2,
        )
        + "\n"
    )
    print("Training complete; best selection requires validation scores.", flush=True)


if __name__ == "__main__":
    main()
