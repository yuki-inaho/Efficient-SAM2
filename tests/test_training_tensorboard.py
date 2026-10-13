"""Exercise the real scalar logging block without loading CUDA models."""

import ast
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]


def test_training_loss_is_logged_without_wandb():
    path = REPO / "sam2/sam2_video_predictor_with_bypass.py"
    tree = ast.parse(path.read_text())
    block = None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for i, child in enumerate(body):
            if isinstance(child, ast.If) and ast.unparse(child.test) == "self.use_wandb":
                # Include any unconditional scalar immediately following the conditional.
                fragment = body[i : i + 2]
                if any(
                    isinstance(n, ast.Attribute) and n.attr == "add_scalar"
                    for item in fragment
                    for n in ast.walk(item)
                ):
                    block = fragment
    assert block is not None
    scalars = []
    obj = SimpleNamespace(
        use_wandb=False,
        epoch=0,
        iters=3,
        writer=SimpleNamespace(add_scalar=lambda *args: scalars.append(args)),
    )
    exec(
        compile(ast.Module(body=block, type_ignores=[]), str(path), "exec"),
        {"self": obj, "avg_loss": 0.25},
    )
    assert scalars == [("training loss", 0.25, 3)]
