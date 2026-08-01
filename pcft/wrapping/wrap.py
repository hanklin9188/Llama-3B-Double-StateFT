from inspect import signature
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from ..adapters.control import ControlLoRA


def _get_base_model_after_peft(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    return model


def _find_llama_like_layers_container(model: nn.Module):
    base = _get_base_model_after_peft(model)
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base.model, "layers", base.model.layers, base
    return None, None, None, base


class DoubleControlDecoderLayer(nn.Module):
    """Frozen decoder layer with parallel controls on attention and MLP residuals."""

    def __init__(self, base_layer, hidden_size, rank, alpha, dropout, checkpoint_base=True):
        super().__init__()
        self.base = base_layer
        self.checkpoint_base = checkpoint_base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        self.ctrl_attn = ControlLoRA(hidden_size, rank, alpha, dropout)
        self.ctrl_mlp = ControlLoRA(hidden_size, rank, alpha, dropout)
        reference = next(self.base.parameters())
        dtype = reference.dtype if reference.is_floating_point() else torch.float16
        self.ctrl_attn.to(device=reference.device, dtype=dtype)
        self.ctrl_mlp.to(device=reference.device, dtype=dtype)

    def _call_attention(self, hidden_states, attention_mask, position_ids, past_key_value, kwargs):
        attention = self.base.self_attn
        accepted = set(signature(attention.forward).parameters)
        call_args = {"hidden_states": hidden_states}
        optional = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_value": past_key_value,
            "output_attentions": False,
            "use_cache": False,
            "cache_position": kwargs.get("cache_position"),
            "position_embeddings": kwargs.get("position_embeddings"),
        }
        for name, value in optional.items():
            if name in accepted and value is not None:
                call_args[name] = value
        output = attention(**call_args)
        return output[0] if isinstance(output, (tuple, list)) else output

    def _run(self, function, tensor):
        if self.training and self.checkpoint_base:
            return torch.utils.checkpoint.checkpoint(function, tensor, use_reentrant=False)
        return function(tensor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        **kwargs: Any,
    ):
        if isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

        attention_input = hidden_states
        normalized = self.base.input_layernorm(hidden_states)
        attention_output = self._run(
            lambda x: self._call_attention(x, attention_mask, position_ids, past_key_value, kwargs),
            normalized,
        )
        hidden_states = attention_input + attention_output + self.ctrl_attn(attention_input)

        mlp_input = hidden_states
        normalized = self.base.post_attention_layernorm(hidden_states)
        mlp_output = self._run(self.base.mlp, normalized)
        return mlp_input + mlp_output + self.ctrl_mlp(mlp_input)


def wrap_model_with_double_control(
    model: nn.Module,
    ranks,
    alphas,
    dropout: float = 0.05,
    checkpoint_base: bool = True,
):
    container, attribute, layers, base = _find_llama_like_layers_container(model)
    if layers is None:
        raise RuntimeError("Expected a Llama-like model with model.layers")
    if len(ranks) != len(layers) or len(alphas) != len(layers):
        raise ValueError("Rank and alpha schedules must match the number of decoder layers")
    hidden_size = base.config.hidden_size
    wrapped = [
        DoubleControlDecoderLayer(
            layer,
            hidden_size,
            int(ranks[index]),
            float(alphas[index]),
            dropout,
            checkpoint_base,
        )
        for index, layer in enumerate(layers)
    ]
    setattr(container, attribute, nn.ModuleList(wrapped))
    return model
