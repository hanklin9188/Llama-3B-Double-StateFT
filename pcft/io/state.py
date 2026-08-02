import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..adapters.control import LowRankControl
from ..rank import get_rank_map, iter_branch_controls, serialize_rank_map


def iter_control_modules(module: nn.Module):
    for _, control in iter_branch_controls(module):
        yield control


def build_control_config(model, base_config: Dict[str, Any]):
    config = dict(base_config)
    controls = dict(iter_branch_controls(model))
    rank_map = get_rank_map(model)
    layers = serialize_rank_map(rank_map)
    for row in layers:
        layer = int(row["layer"])
        row["rank_max_attn"] = controls[next(key for key in controls if key.layer == layer and key.branch == "attn")].rank_max
        row["rank_max_mlp"] = controls[next(key for key in controls if key.layer == layer and key.branch == "mlp")].rank_max
    config["layers"] = layers
    config["global_rank_budget"] = sum(rank_map.values())
    config["compact"] = bool(config.get("compact", False))
    return config


def save_control_state(model: nn.Module, out_dir: str, cfg: Dict[str, Any]):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {key.name: control.state_dict() for key, control in iter_branch_controls(model)}
    torch.save(state, out_dir / "control_state.pt")
    with (out_dir / "control_config.json").open("w", encoding="utf-8") as handle:
        json.dump(build_control_config(model, cfg), handle, indent=2)


def load_control_state_if_any(model: nn.Module, maybe_dir: Optional[str]) -> Optional[str]:
    if not maybe_dir:
        return None
    path = Path(maybe_dir) / "control_state.pt"
    if not path.is_file():
        return maybe_dir
    state = torch.load(path, map_location="cpu", weights_only=True)
    controls = dict(iter_branch_controls(model))
    expected = {key.name for key in controls}
    if set(state) != expected:
        raise ValueError(f"Control keys mismatch: checkpoint={set(state)}, model={expected}")
    for key, control in controls.items():
        control.load_state_dict(state[key.name], strict=True)
    print(f"[Resume] Loaded control state from {path}")
    return None


def export_compact_adapter(source_dir: str, output_dir: str):
    source = Path(source_dir)
    output = Path(output_dir)
    with (source / "control_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    state = torch.load(source / "control_state.pt", map_location="cpu", weights_only=True)
    compact_state = {}
    ranks = {}
    for row in config["layers"]:
        ranks[f"layer_{int(row['layer']):02d}.attn"] = int(row["rank_attn"])
        ranks[f"layer_{int(row['layer']):02d}.mlp"] = int(row["rank_mlp"])
        row["rank_max_attn"] = int(row["rank_attn"])
        row["rank_max_mlp"] = int(row["rank_mlp"])
    quantum = int(config["rank_quantum"])
    for name, branch_state in state.items():
        rank = ranks[name]
        compact_state[name] = {
            "lora_A": branch_state["lora_A"][:rank].clone(),
            "lora_B": branch_state["lora_B"][:, :rank].clone(),
            "active_rank_tensor": torch.tensor(rank, dtype=torch.int32),
            "rank_gates": branch_state["rank_gates"][:rank].clone(),
            "block_activation_count": branch_state["block_activation_count"][: rank // quantum].clone(),
            "block_last_active_step": branch_state["block_last_active_step"][: rank // quantum].clone(),
        }
    config["compact"] = True
    config["rank_max"] = max(ranks.values())
    config["configured_parameters"] = sum(
        2 * int(config["hidden_size"]) * rank for rank in ranks.values()
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(compact_state, output / "control_state.pt")
    with (output / "control_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return output


def _scaling(config, rank):
    alpha = float(config["alpha"])
    mode = config["scaling_mode"]
    if mode == "alpha":
        return alpha
    if mode == "alpha_over_rank":
        return alpha / float(rank)
    return alpha / float(rank) ** 0.5


@torch.no_grad()
def verify_compact_adapter(source_dir: str, compact_dir: str, seed: int = 42):
    """Verify every branch without loading a second copy of the base model."""
    source_dir = Path(source_dir)
    compact_dir = Path(compact_dir)
    with (source_dir / "control_config.json").open(encoding="utf-8") as handle:
        source_config = json.load(handle)
    with (compact_dir / "control_config.json").open(encoding="utf-8") as handle:
        compact_config = json.load(handle)
    source = torch.load(source_dir / "control_state.pt", map_location="cpu", weights_only=True)
    compact = torch.load(compact_dir / "control_state.pt", map_location="cpu", weights_only=True)
    if set(source) != set(compact):
        raise ValueError("Supernet and compact branch keys differ")

    generator = torch.Generator().manual_seed(seed)
    hidden_size = int(source_config["hidden_size"])
    maximum_error = 0.0
    for name in sorted(source):
        rank = int(compact[name]["active_rank_tensor"].item())
        source_gates = source[name]["rank_gates"]
        if not torch.equal(source_gates[:rank], torch.ones_like(source_gates[:rank])):
            raise ValueError(f"{name} has an unfinished growing transition")
        if torch.count_nonzero(source_gates[rank:]):
            raise ValueError(f"{name} has an unfinished shrinking transition")
        x = torch.randn(3, hidden_size, generator=generator, dtype=torch.float32)
        source_a = source[name]["lora_A"][:rank].float()
        source_b = source[name]["lora_B"][:, :rank].float()
        compact_a = compact[name]["lora_A"].float()
        compact_b = compact[name]["lora_B"].float()
        source_reduced = F.linear(x, source_a)
        source_output = F.linear(
            source_reduced * source_gates[:rank].float(), source_b
        ) * _scaling(source_config, rank)
        compact_reduced = F.linear(x, compact_a)
        compact_output = F.linear(
            compact_reduced * compact[name]["rank_gates"].float(), compact_b
        ) * _scaling(compact_config, rank)
        maximum_error = max(maximum_error, float((source_output - compact_output).abs().max()))
    if maximum_error > 1e-6:
        raise ValueError(f"Compact adapter mismatch: max_abs_error={maximum_error:.3e}")
    return maximum_error
