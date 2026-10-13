"""GPU 競合ゲート: Codex 優先ポリシーの下で重い GPU ジョブの可否を判定する。

方針(ユーザー指示 2026-10-13): Codex が GPU を使用していて VRAM が競合しそうな場合は
Codex を優先し、本ジョブは待機/直列化する。暗黙 fallback はせず、判定不能・時間切れは明示エラー。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def parse_smi_output(text: str) -> tuple[int, int]:
    lines = text.strip().splitlines()
    if not lines:
        raise ValueError("empty nvidia-smi output")
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) < 2:
        raise ValueError(f"unexpected nvidia-smi line: {lines[0]!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"non-integer nvidia-smi values: {lines[0]!r}") from exc


def should_proceed(used_mb: int, total_mb: int, min_free_mb: int) -> bool:
    return (total_mb - used_mb) >= min_free_mb


def query_gpu() -> tuple[int, int]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return parse_smi_output(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate heavy GPU jobs under Codex-first policy")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--wait-seconds", type=int, default=600)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    min_free_mb = int(args.min_free_gb * 1024)
    deadline = time.monotonic() + args.wait_seconds
    while True:
        used, total = query_gpu()
        if should_proceed(used, total, min_free_mb):
            print(f"GPU gate open: free={total - used}MB >= {min_free_mb}MB")
            return
        if time.monotonic() >= deadline:
            print(
                f"GPU gate timeout: used={used}MB/{total}MB, "
                f"need free>={min_free_mb}MB (Codex 優先ポリシー) — 明示的に中断",
                file=sys.stderr,
            )
            raise SystemExit(2)
        print(
            f"GPU gate waiting: used={used}MB/{total}MB "
            f"(free<{min_free_mb}MB) — Codex 優先のため待機",
            flush=True,
        )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
