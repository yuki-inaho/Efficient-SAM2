"""CPU contract test, run with the model venv (torch + peft)."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lora_adapter import frozen_reference, inject_image_lora, load_image_lora, save_image_lora


class AdapterContract(unittest.TestCase):
    def test_update_and_restore(self):
        torch.manual_seed(13)
        trunk = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        original = copy.deepcopy(trunk)
        predictor = SimpleNamespace(image_encoder=SimpleNamespace(trunk=trunk))
        inject_image_lora(predictor, ["0", "2"])
        x = torch.randn(3, 4)
        opt = torch.optim.AdamW([p for p in trunk.parameters() if p.requires_grad], lr=0.01)
        trunk(x).square().mean().backward()
        opt.step()
        with frozen_reference(trunk):
            torch.testing.assert_close(trunk(x), original(x))
            self.assertFalse(trunk(x).requires_grad)
        self.assertTrue(trunk(x).requires_grad)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "adapter.pt"
            info = save_image_lora(predictor, path)
            self.assertGreater(info["changed_tensors"], 0)
            self.assertGreater(info["nonzero_gradient_calls"], 0)
            restored = SimpleNamespace(image_encoder=SimpleNamespace(trunk=original))
            load_image_lora(restored, path)
            torch.testing.assert_close(trunk(x), restored.image_encoder.trunk(x))
            blob = torch.load(path, weights_only=True)
            blob["adapter_state"].pop(next(iter(blob["adapter_state"])))
            torch.save(blob, path)
            another = SimpleNamespace(
                image_encoder=SimpleNamespace(
                    trunk=nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
                )
            )
            with self.assertRaisesRegex(ValueError, "keys"):
                load_image_lora(another, path)

    def test_missing_target_fails(self):
        model = SimpleNamespace(image_encoder=SimpleNamespace(trunk=nn.Sequential(nn.Linear(4, 4))))
        with self.assertRaises(ValueError):
            inject_image_lora(model, ["not_present"])


if __name__ == "__main__":
    unittest.main()
