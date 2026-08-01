import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from pcft.adapters.control import ControlLoRA
from pcft.io.state import load_control_state_if_any, save_control_state
from pcft.metrics.adaptive_rank import AdaptiveRankAllocatorCallback
from pcft.wrapping.wrap import wrap_model_with_double_control


class FakeAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        return (self.projection(hidden_states),)


class FakeLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn = FakeAttention(hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size, bias=False)


class FakeBackbone(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(hidden_size), FakeLayer(hidden_size)])


class FakeModel(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.model = FakeBackbone(hidden_size)
        self.config = type("Config", (), {"hidden_size": hidden_size})()


class ControlTests(unittest.TestCase):
    def test_control_starts_as_identity_residual(self):
        control = ControlLoRA(hidden_size=16, rank=8, alpha=2.0, dropout=0.0)
        output = control(torch.randn(2, 4, 16))
        self.assertEqual(torch.count_nonzero(output), 0)

    def test_active_rank_masks_components(self):
        control = ControlLoRA(hidden_size=16, rank=8, alpha=1.0, dropout=0.0)
        control.set_active_rank(3)
        self.assertEqual(control.active_rank, 3)
        self.assertEqual(control.rank_mask.tolist(), [1, 1, 1, 0, 0, 0, 0, 0])

    def test_allocator_preserves_bounded_budget(self):
        allocator = AdaptiveRankAllocatorCallback(
            results_dir="unused",
            total_budget=12,
            rank_min=2,
            rank_max=6,
        )
        ranks = allocator._allocate([0.1, 0.9, 0.5])
        self.assertEqual(sum(ranks), 12)
        self.assertTrue(all(2 <= rank <= 6 for rank in ranks))

    def test_double_wrapper_and_checkpoint_round_trip(self):
        model = wrap_model_with_double_control(
            FakeModel(), ranks=[8, 8], alphas=[1.0, 1.0], dropout=0.0
        )
        output = model.model.layers[0](torch.randn(2, 3, 16))
        self.assertEqual(output.shape, (2, 3, 16))
        model.model.layers[0].ctrl_attn.set_active_rank(3)

        with tempfile.TemporaryDirectory() as directory:
            save_control_state(model, directory, {"architecture": "double_control"})
            restored = wrap_model_with_double_control(
                FakeModel(), ranks=[8, 8], alphas=[1.0, 1.0], dropout=0.0
            )
            self.assertIsNone(load_control_state_if_any(restored, directory))
            self.assertEqual(restored.model.layers[0].ctrl_attn.active_rank, 3)
            self.assertTrue((Path(directory) / "control_config.json").is_file())

    def test_huggingface_llama_forward_and_backward(self):
        config = LlamaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        model = wrap_model_with_double_control(
            LlamaForCausalLM(config), ranks=[8, 8], alphas=[1.0, 1.0], dropout=0.0
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for layer in model.model.layers:
            for control in (layer.ctrl_attn, layer.ctrl_mlp):
                for parameter in control.parameters():
                    parameter.requires_grad_(True)
        input_ids = torch.randint(0, 128, (2, 12))
        output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        output.loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertEqual(output.logits.shape, (2, 12, 128))
        self.assertTrue(all(gradient is not None for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
