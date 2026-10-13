"""The custom dataset output path must be stable for downstream evaluation."""

import ast
from pathlib import Path
from types import SimpleNamespace


def test_custom_output_path_preserved():
    path = (
        Path(__file__).resolve().parents[1] / "tools/vos_inference_custom.py"
    )
    tree = ast.parse(path.read_text())
    # Execute the actual dataset dispatch in isolation.
    dispatch = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "args.dataset == 'custom'"
    )
    args = SimpleNamespace(dataset="custom", output_mask_dir="rewritten/name")
    exec(
        compile(ast.Module(body=[dispatch], type_ignores=[]), str(path), "exec"),
        {"args": args, "output_mask_dir": "/requested/output"},
    )
    assert args.output_mask_dir == "/requested/output"
