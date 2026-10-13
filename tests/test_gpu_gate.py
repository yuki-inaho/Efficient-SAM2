"""GPU ゲートの単体テスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.workflow.gpu_gate import parse_smi_output, should_proceed  # noqa: E402


def test_should_proceed() -> None:
    assert should_proceed(used_mb=1000, total_mb=24564, min_free_mb=20000)
    assert not should_proceed(used_mb=10000, total_mb=24564, min_free_mb=20000)


def test_parse_smi_output() -> None:
    used, total = parse_smi_output("1234, 24564\n")
    assert (used, total) == (1234, 24564)


def test_parse_smi_output_invalid() -> None:
    with pytest.raises(ValueError):
        parse_smi_output("")
