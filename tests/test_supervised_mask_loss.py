import numpy as np
import pytest

pytest.importorskip(
    "torch.nn", reason="requires real PyTorch, not a workflow test stub"
)
import torch

from tools.workflow.supervised_mask_loss import supervised_mask_loss


def test_positive_and_explicit_negative_gradients_point_in_opposite_directions():
    logits = torch.zeros((1, 1, 16, 16), requires_grad=True)
    target = np.zeros((16, 16), bool)
    target[4:12, 4:12] = True
    occ = np.zeros((16, 16), np.uint8)
    occ[6:8] = 255
    loss = supervised_mask_loss(logits, target, occ, [[0, 0], [8, 8]], [0, 1])
    loss.backward()
    assert (
        logits.grad[0, 0, 0, 0] > 0
    )  # gradient descent suppresses background negative
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


def test_background_regularizer_preserves_foreground_and_boundary_uncertainty():
    target = np.zeros((48, 48), bool)
    target[16:32, 16:32] = True
    occ = np.zeros((48, 48), np.uint8)
    occ[:, 24] = 255
    base = torch.zeros((1, 1, 48, 48), requires_grad=True)
    tuned = torch.zeros_like(base, requires_grad=True)
    supervised_mask_loss(base, target, occ, [[24, 24]], [1]).backward()
    supervised_mask_loss(
        tuned, target, occ, [[24, 24]], [1], boundary_background_weight=0.5
    ).backward()
    delta = tuned.grad - base.grad
    assert (
        delta[0, 0, 24, 24] == 0
    )  # foreground under artificial occlusion remains foreground
    assert (
        delta[0, 0, 15, 24] == 0
    )  # one pixel outside teacher: uncertain boundary protected
    assert delta[0, 0, 9, 24] > 0  # confident exterior near target
    assert delta[0, 0, 0, 24] > 0  # artificial occluder outside the target
    assert delta[0, 0, 0, 0] == 0  # distant unaffected background


def test_zero_background_weight_retains_baseline_loss():
    target = np.zeros((32, 32), bool)
    target[8:24, 8:24] = True
    logits = torch.zeros((1, 1, 32, 32))
    args = (logits, target, np.zeros_like(target), [[16, 16]], [1])
    assert torch.equal(
        supervised_mask_loss(*args),
        supervised_mask_loss(*args, boundary_background_weight=0),
    )
