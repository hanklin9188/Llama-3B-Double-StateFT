from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Tuple

import torch.nn as nn

from .adapters.control import LowRankControl


@dataclass(frozen=True, order=True)
class BranchKey:
    layer: int
    branch: str

    def __post_init__(self):
        if self.branch not in {"attn", "mlp"}:
            raise ValueError(f"Invalid branch: {self.branch}")

    @property
    def name(self) -> str:
        return f"layer_{self.layer:02d}.{self.branch}"


RankMap = Dict[BranchKey, int]


def iter_branch_controls(model: nn.Module) -> Iterator[Tuple[BranchKey, LowRankControl]]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Expected model.model.layers")
    for layer_index, layer in enumerate(layers):
        yield BranchKey(layer_index, "attn"), layer.ctrl_attn
        yield BranchKey(layer_index, "mlp"), layer.ctrl_mlp


def get_rank_map(model: nn.Module) -> RankMap:
    return {key: control.active_rank for key, control in iter_branch_controls(model)}


def set_rank_map(model: nn.Module, rank_map: RankMap, step: int = 0, transition_steps: int = 0):
    controls = dict(iter_branch_controls(model))
    if set(rank_map) != set(controls):
        missing = set(controls) - set(rank_map)
        extra = set(rank_map) - set(controls)
        raise ValueError(f"Rank-map keys mismatch; missing={missing}, extra={extra}")
    for key, rank in rank_map.items():
        if transition_steps > 0:
            controls[key].begin_rank_transition(rank, transition_steps, step=step)
        else:
            controls[key].set_active_rank(rank, step=step)


def validate_rank_map(rank_map: RankMap, rank_min: int, rank_max: int, quantum: int, budget: int):
    if any(rank < rank_min or rank > rank_max or rank % quantum for rank in rank_map.values()):
        raise ValueError("Rank map violates rank bounds or quantum")
    if sum(rank_map.values()) != int(budget):
        raise ValueError(f"Rank budget violated: {sum(rank_map.values())} != {budget}")


@contextmanager
def override_rank_map(model: nn.Module, rank_map: RankMap):
    controls = dict(iter_branch_controls(model))
    with ExitStack() as stack:
        for key, rank in rank_map.items():
            stack.enter_context(controls[key].override_rank(rank))
        yield model


def serialize_rank_map(rank_map: RankMap):
    by_layer = {}
    for key, rank in rank_map.items():
        row = by_layer.setdefault(key.layer, {"layer": key.layer})
        row[f"rank_{key.branch}"] = int(rank)
    return [by_layer[index] for index in sorted(by_layer)]


def deserialize_rank_map(layers) -> RankMap:
    rank_map = {}
    for row in layers:
        rank_map[BranchKey(int(row["layer"]), "attn")] = int(row["rank_attn"])
        rank_map[BranchKey(int(row["layer"]), "mlp")] = int(row["rank_mlp"])
    return rank_map
