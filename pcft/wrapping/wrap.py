from inspect import signature
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from ..adapters.control import LowRankControl


def _find_llama_like_layers_container(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model, "layers", model.model.layers, model
    return None, None, None, model


class DoubleControlDecoderLayer(nn.Module):
    def __init__(
        self,
        base_layer,
        hidden_size,
        rank_max_attn,
        rank_max_mlp,
        active_rank_attn,
        active_rank_mlp,
        alpha,
        dropout,
        rank_quantum,
        scaling_mode,
        checkpoint_base=True,
    ):
        super().__init__()
        self.base = base_layer
        self.checkpoint_base = checkpoint_base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        common = dict(
            hidden_size=hidden_size,
            alpha=alpha,
            dropout=dropout,
            rank_quantum=rank_quantum,
            scaling_mode=scaling_mode,
        )
        self.ctrl_attn = LowRankControl(
            rank_max=rank_max_attn, active_rank=active_rank_attn, **common
        )
        self.ctrl_mlp = LowRankControl(
            rank_max=rank_max_mlp, active_rank=active_rank_mlp, **common
        )
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
        required_even_when_none = {"attention_mask", "position_embeddings"}
        for name, value in optional.items():
            if name in accepted and (value is not None or name in required_even_when_none):
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
        attention_output = self._run(
            lambda value: self._call_attention(
                value, attention_mask, position_ids, past_key_value, kwargs
            ),
            self.base.input_layernorm(hidden_states),
        )
        hidden_states = attention_input + attention_output + self.ctrl_attn(attention_input)
        mlp_input = hidden_states
        mlp_output = self._run(self.base.mlp, self.base.post_attention_layernorm(hidden_states))
        return mlp_input + mlp_output + self.ctrl_mlp(mlp_input)


def wrap_model_with_double_control(
    model: nn.Module,
    ranks_attn=None,
    ranks_mlp=None,
    rank_max_attn=None,
    rank_max_mlp=None,
    ranks=None,
    alphas=None,
    alpha: float = 16.0,
    dropout: float = 0.05,
    rank_quantum: int = 8,
    scaling_mode: str = "alpha_over_sqrt_rank",
    checkpoint_base: bool = True,
):
    container, attribute, layers, base = _find_llama_like_layers_container(model)
    if layers is None:
        raise RuntimeError("Expected a Llama-like model with model.layers")
    count = len(layers)
    if ranks is not None:
        ranks_attn = ranks_mlp = list(ranks)
    ranks_attn = list(ranks_attn or [64] * count)
    ranks_mlp = list(ranks_mlp or [64] * count)
    rank_max_attn = list(rank_max_attn or ranks_attn)
    rank_max_mlp = list(rank_max_mlp or ranks_mlp)
    alphas = list(alphas or [alpha] * count)
    arrays = (ranks_attn, ranks_mlp, rank_max_attn, rank_max_mlp, alphas)
    if any(len(array) != count for array in arrays):
        raise ValueError("All rank and alpha lists must match decoder layer count")
    hidden_size = base.config.hidden_size
    wrapped = []
    for index, layer in enumerate(layers):
        wrapped.append(
            DoubleControlDecoderLayer(
                layer,
                hidden_size,
                rank_max_attn[index],
                rank_max_mlp[index],
                ranks_attn[index],
                ranks_mlp[index],
                alphas[index],
                dropout,
                rank_quantum,
                scaling_mode,
                checkpoint_base,
            )
        )
    setattr(container, attribute, nn.ModuleList(wrapped))
    return model
