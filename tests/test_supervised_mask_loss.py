import numpy as np
import pytest

pytest.importorskip("torch.nn", reason="requires real PyTorch, not a workflow test stub")
import torch  # noqa: E402
from tools.workflow.supervised_mask_loss import supervised_mask_loss  # noqa: E402


def test_positive_and_explicit_negative_gradients_point_in_opposite_directions():
    logits = torch.zeros((1, 1, 16, 16), requires_grad=True)
    target = np.zeros((16, 16), bool)
    target[4:12, 4:12] = True
    occ = np.zeros((16, 16), np.uint8)
    occ[6:8] = 255
    loss = supervised_mask_loss(logits, target, occ, [[0, 0], [8, 8]], [0, 1])
    loss.backward()
    assert logits.grad[0, 0, 0, 0] > 0  # gradient descent suppresses background negative
    assert logits.grad[0, 0, 8, 8] < 0  # preserves the object positive
    assert logits.grad[0, 0, 7, 8] < 0  # recovers object under synthetic occlusion
    assert logits.grad[0, 0, 0, 0] > logits.grad[0, 0, 0, 15] * 5


def test_negative_on_teacher_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        supervised_mask_loss(
            torch.zeros((1, 1, 4, 4)),
            np.ones((4, 4), bool),
            np.zeros((4, 4), np.uint8),
            [[1, 1]],
            [0],
        )
