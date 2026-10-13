"""P1-1 回帰テスト: propagate_in_video が非condフレームで常に推論されること。

repo の関数を AST で取り出し、決定的スタブで実行する(CPU・torch不要)。
レビュー指摘を反映し、逆方向・開始位置>10・物体ごとのcond位置差・
既存non_cond出力がある再伝播も検証する。
"""

from __future__ import annotations

import ast
import types
from pathlib import Path

REPO_FILE = (
    Path(__file__).resolve().parents[1]
    / "sam2"
    / "sam2_video_predictor.py"
)


class FakeMask:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def to(self, device, non_blocking: bool = True):  # noqa: ANN001
        return self


def _extract_function(name: str) -> str:
    src = REPO_FILE.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    seg = ast.get_source_segment(src, f)
                    assert seg, f"cannot extract {name}"
                    return seg
    raise AssertionError(f"function {name} not found")


def _load_fn() -> dict:
    fn_src = _extract_function("propagate_in_video")
    ns: dict = {
        "tqdm": lambda x, **kw: x,
        "logger": types.SimpleNamespace(info=lambda *a, **k: None),
        "torch": types.SimpleNamespace(cat=lambda xs, dim=0: xs),
    }
    exec(fn_src, ns)  # noqa: S102
    return ns


def _make_env(cond0=(0,), cond1=(0,), num_frames: int = 13, stale=None):
    """stale: {obj_name: {frame: tag}} を既存 non_cond 出力として差し込む。"""
    calls: list[tuple[int, int]] = []
    stale = stale or {}

    class Stub:
        clear_non_cond_mem_around_input = False
        image_encoder = types.SimpleNamespace(trunk=types.SimpleNamespace())

        def __init__(self) -> None:
            self.time_log: dict = {}

        def propagate_in_video_preflight(self, state):  # noqa: ANN001
            return None

        def _get_obj_num(self, state):  # noqa: ANN001
            return 2

        def _run_single_frame_inference(self, **kw):  # noqa: ANN003
            calls.append((int(kw["frame_idx"]), int(kw["output_dict"]["name"])))
            tag = f"f{kw['frame_idx']}_o{kw['output_dict']['name']}"
            return ({"pred_masks": FakeMask(tag)}, FakeMask(tag))

        def _get_orig_video_res_output(self, state, masks):  # noqa: ANN001
            return None, masks

    def _obj(name: int, conds) -> dict:
        return {
            "name": name,
            "cond_frame_outputs": {
                c: {"pred_masks": FakeMask(f"cond{c}_o{name}")} for c in conds
            },
            "non_cond_frame_outputs": {
                f: {"pred_masks": FakeMask(tag)}
                for f, tag in stale.get(name, {}).items()
            },
        }

    state = {
        "obj_ids": [1, 2],
        "num_frames": num_frames,
        "device": "cpu",
        "output_dict_per_obj": {0: _obj(0, cond0), 1: _obj(1, cond1)},
        "frames_tracked_per_obj": {0: {}, 1: {}},
    }
    return Stub(), state, calls


def _expected_calls(order, obj0_skip, obj1_skip) -> list[tuple[int, int]]:
    out = []
    for f in order:
        if f not in obj0_skip:
            out.append((f, 0))
        if f not in obj1_skip:
            out.append((f, 1))
    return out


def test_source_has_no_frame_skip_branch() -> None:
    src = REPO_FILE.read_text()
    start = src.index("    def propagate_in_video(\n")
    end = src.index("    def propagate_in_video_wj_1(\n")
    assert "frame_idx > 10" not in src[start:end]


def test_forward_all_frames_call_inference() -> None:
    ns = _load_fn()
    stub, state, calls = _make_env()
    outs = list(ns["propagate_in_video"](stub, state))

    expected = _expected_calls(range(13), {0}, {0})
    assert calls == expected, f"expected {len(expected)} calls, got {len(calls)}"
    assert len(outs) == 13

    by_frame = {fi: masks for fi, _ids, masks in outs}
    assert by_frame[0][0].tag == "cond0_o0"
    assert by_frame[0][1].tag == "cond0_o1"
    assert by_frame[11][0].tag == "f11_o0"
    assert by_frame[11][1].tag == "f11_o1"
    assert by_frame[12][0].tag == "f12_o0"


def test_reverse_propagation_infers_every_non_cond_frame() -> None:
    ns = _load_fn()
    stub, state, calls = _make_env()
    outs = list(ns["propagate_in_video"](stub, state, start_frame_idx=12, reverse=True))

    order = list(range(12, -1, -1))
    expected = _expected_calls(order, {0}, {0})
    assert calls == expected
    assert [fi for fi, _ids, _m in outs] == order

    by_frame = {fi: masks for fi, _ids, masks in outs}
    assert by_frame[12][0].tag == "f12_o0"
    assert by_frame[1][1].tag == "f1_o1"
    assert by_frame[0][0].tag == "cond0_o0"


def test_start_frame_above_10_infers() -> None:
    ns = _load_fn()
    stub, state, calls = _make_env()
    outs = list(ns["propagate_in_video"](stub, state, start_frame_idx=12))

    assert calls == [(12, 0), (12, 1)]
    assert [fi for fi, _ids, _m in outs] == [12]
    by_frame = {fi: masks for fi, _ids, masks in outs}
    assert by_frame[12][1].tag == "f12_o1"


def test_object_specific_cond_positions() -> None:
    ns = _load_fn()
    stub, state, calls = _make_env(cond0=(0,), cond1=(2,))
    outs = list(ns["propagate_in_video"](stub, state))

    expected = _expected_calls(range(13), {0}, {2})
    assert calls == expected
    assert len(calls) == 24

    by_frame = {fi: masks for fi, _ids, masks in outs}
    # obj1 は frame2 が cond、obj0 は推論結果(取り違えなし)
    assert by_frame[2][0].tag == "f2_o0"
    assert by_frame[2][1].tag == "cond2_o1"


def test_existing_non_cond_output_is_overwritten_by_fresh_inference() -> None:
    ns = _load_fn()
    stale = {0: {1: "stale_f1_o0"}}
    stub, state, calls = _make_env(stale=stale)
    outs = list(ns["propagate_in_video"](stub, state))

    assert (1, 0) in calls
    by_frame = {fi: masks for fi, _ids, masks in outs}
    assert by_frame[1][0].tag == "f1_o0"
    assert state["output_dict_per_obj"][0]["non_cond_frame_outputs"][1][
        "pred_masks"
    ].tag == "f1_o0"
