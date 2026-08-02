import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from pcft.adapters.control import LowRankControl
from pcft.io.state import (
    export_compact_adapter,
    load_control_state_if_any,
    save_control_state,
    verify_compact_adapter,
)
from pcft.rank import BranchKey, get_rank_map, override_rank_map, set_rank_map, validate_rank_map
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


def wrap_fake(rank_max=16):
    return wrap_model_with_double_control(
        FakeModel(),
        ranks_attn=[8, 8],
        ranks_mlp=[8, 8],
        rank_max_attn=[rank_max, rank_max],
        rank_max_mlp=[rank_max, rank_max],
        alpha=2.0,
        dropout=0.0,
        rank_quantum=8,
    )


class ControlTests(unittest.TestCase):
    def test_control_starts_as_zero_residual(self):
        control = LowRankControl(16, rank_max=16, active_rank=8, alpha=2.0, dropout=0.0)
        output = control(torch.randn(2, 4, 16))
        self.assertEqual(torch.count_nonzero(output), 0)

    def test_nested_prefix_rank_and_temporary_override(self):
        control = LowRankControl(16, rank_max=16, active_rank=8, alpha=2.0, dropout=0.0)
        control.set_active_rank(16)
        self.assertEqual(control.active_rank, 16)
        self.assertEqual(control.rank_gates.tolist(), [1.0] * 16)
        with control.override_rank(8):
            self.assertEqual(control.effective_forward_rank, 8)
        self.assertEqual(control.effective_forward_rank, 16)

    def test_branch_rank_map_preserves_global_budget(self):
        model = wrap_fake()
        ranks = get_rank_map(model)
        ranks[BranchKey(0, "attn")] = 16
        ranks[BranchKey(1, "mlp")] = 0
        with self.assertRaises(ValueError):
            validate_rank_map(ranks, 8, 16, 8, 32)
        ranks[BranchKey(1, "mlp")] = 8
        ranks[BranchKey(0, "mlp")] = 8
        ranks[BranchKey(1, "attn")] = 8
        validate_rank_map(ranks, 8, 16, 8, 40)
        set_rank_map(model, ranks)
        self.assertEqual(model.model.layers[0].ctrl_attn.active_rank, 16)
        self.assertEqual(model.model.layers[0].ctrl_mlp.active_rank, 8)

    def test_rank_map_override_restores_committed_ranks(self):
        model = wrap_fake()
        committed = get_rank_map(model)
        candidate = dict(committed)
        candidate[BranchKey(0, "attn")] = 16
        with override_rank_map(model, candidate):
            self.assertEqual(model.model.layers[0].ctrl_attn.effective_forward_rank, 16)
        self.assertEqual(get_rank_map(model), committed)
        self.assertEqual(model.model.layers[0].ctrl_attn.effective_forward_rank, 8)

    def test_checkpoint_round_trip_and_compact_equivalence(self):
        model = wrap_fake()
        model.model.layers[0].ctrl_attn.set_active_rank(16)
        config = {
            "architecture": "id_dr_double_control_v2",
            "base_model": "fake/model",
            "hidden_size": 16,
            "rank_min": 8,
            "rank_max": 16,
            "rank_quantum": 8,
            "global_rank_budget": 40,
            "alpha": 2.0,
            "scaling_mode": "alpha_over_sqrt_rank",
        }
        with tempfile.TemporaryDirectory() as directory:
            compact = Path(directory) / "compact"
            save_control_state(model, directory, config)
            restored = wrap_fake()
            self.assertIsNone(load_control_state_if_any(restored, directory))
            self.assertEqual(restored.model.layers[0].ctrl_attn.active_rank, 16)
            export_compact_adapter(directory, compact)
            self.assertLessEqual(verify_compact_adapter(directory, compact), 1e-6)
            self.assertTrue((compact / "control_config.json").is_file())

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
            LlamaForCausalLM(config), ranks=[8, 8], alpha=1.0, dropout=0.0
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
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertEqual(output.logits.shape, (2, 12, 128))
        self.assertTrue(all(gradient is not None for gradient in gradients))

    def test_attention_accepts_optimized_none_mask(self):
        config = LlamaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        model = wrap_model_with_double_control(
            LlamaForCausalLM(config), ranks=[8], alpha=1.0, dropout=0.0
        )
        hidden_states = torch.randn(2, 8, 32)
        position_ids = torch.arange(8).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
        output = model.model.layers[0](
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        self.assertEqual(output.shape, hidden_states.shape)


if __name__ == "__main__":
    unittest.main()
