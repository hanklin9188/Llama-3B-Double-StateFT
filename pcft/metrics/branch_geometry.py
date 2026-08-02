import csv
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from ..rank import BranchKey, iter_branch_controls
from .metrics import _effective_rank_from_ctrl, _id_twonN_and_gride


@dataclass
class BranchGeometry:
    step: int
    layer: int
    branch: str
    active_rank: int
    id_input_median: float
    id_input_mad: float
    id_input_lcb: float
    id_input_ema: float
    id_output_median: float
    id_output_mad: float
    output_energy: float
    output_id_valid: bool
    output_id_saturation: float
    effective_rank: float
    parameter_rank_saturation: float
    num_examples: int
    wall_time: float

    @property
    def key(self):
        return BranchKey(self.layer, self.branch)


def _last_token_indices(input_ids, attention_mask, eos_token_id):
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
    valid = attention_mask.bool()
    if eos_token_id is not None:
        valid = valid & input_ids.ne(eos_token_id)
    indices = torch.where(valid, positions, torch.full_like(positions, -1)).max(dim=1).values
    fallback = torch.where(
        attention_mask.bool(), positions, torch.full_like(positions, -1)
    ).max(dim=1).values
    return torch.where(indices >= 0, indices, fallback)


def _selected_token(states, indices):
    batch = torch.arange(states.shape[0], device=states.device)
    return states[batch, indices].detach().to(device="cpu", dtype=torch.float32)


def estimate_id_with_bootstrap(
    states: torch.Tensor,
    n_bootstrap: int = 20,
    sample_fraction: float = 0.8,
    uncertainty_lambda: float = 1.0,
    seed: int = 42,
):
    array = states.detach().cpu().float().numpy()
    if len(array) < 32 or float(np.std(array)) < 1e-12:
        return {"median": float("nan"), "mad": float("nan"), "lcb": 0.0}
    rng = np.random.default_rng(seed)
    repeats = max(1, int(n_bootstrap))
    size = max(32, int(len(array) * sample_fraction))
    estimates = []
    for _ in range(repeats):
        indices = rng.choice(len(array), size=size, replace=True)
        try:
            _, estimate = _id_twonN_and_gride(array[indices], range_max=64, k_lo=8, k_hi=32)
            if math.isfinite(estimate):
                estimates.append(float(estimate))
        except Exception:
            continue
    if not estimates:
        return {"median": float("nan"), "mad": float("nan"), "lcb": 0.0}
    median = float(np.median(estimates))
    mad = float(1.4826 * np.median(np.abs(np.asarray(estimates) - median)))
    return {
        "median": median,
        "mad": mad,
        "lcb": max(0.0, median - float(uncertainty_lambda) * mad),
    }


@torch.no_grad()
def collect_branch_states(model, dataloader, processing_class, sample_size: int):
    controls = dict(iter_branch_controls(model))
    inputs: Dict[BranchKey, List[torch.Tensor]] = {key: [] for key in controls}
    outputs: Dict[BranchKey, List[torch.Tensor]] = {key: [] for key in controls}
    current_indices = None
    hooks = []

    def hook_for(key):
        def hook(module, args, output):
            inputs[key].append(_selected_token(args[0], current_indices))
            outputs[key].append(_selected_token(output, current_indices))
        return hook

    for key, control in controls.items():
        hooks.append(control.register_forward_hook(hook_for(key)))

    was_training = model.training
    model.eval()
    count = 0
    try:
        for batch in dataloader:
            batch = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in batch.items()}
            current_indices = _last_token_indices(
                batch["input_ids"],
                batch["attention_mask"],
                getattr(processing_class, "eos_token_id", None),
            )
            model(**batch, use_cache=False)
            count += batch["input_ids"].shape[0]
            if count >= sample_size:
                break
    finally:
        for hook in hooks:
            hook.remove()
        if was_training:
            model.train()
    return (
        {key: torch.cat(value)[:sample_size] for key, value in inputs.items()},
        {key: torch.cat(value)[:sample_size] for key, value in outputs.items()},
    )


def measure_branch_geometry(
    model,
    dataloader,
    processing_class,
    step: int,
    sample_size: int = 256,
    bootstrap_repeats: int = 20,
    bootstrap_fraction: float = 0.8,
    uncertainty_lambda: float = 1.0,
    output_energy_threshold: float = 1e-8,
    seed: int = 42,
):
    started = time.perf_counter()
    inputs, outputs = collect_branch_states(model, dataloader, processing_class, sample_size)
    controls = dict(iter_branch_controls(model))
    rows = []
    for offset, key in enumerate(sorted(controls)):
        control = controls[key]
        id_input = estimate_id_with_bootstrap(
            inputs[key], bootstrap_repeats, bootstrap_fraction, uncertainty_lambda, seed + offset
        )
        energy = float(outputs[key].square().mean().sqrt().item())
        output_valid = energy > output_energy_threshold
        id_output = (
            estimate_id_with_bootstrap(
                outputs[key], bootstrap_repeats, bootstrap_fraction, uncertainty_lambda, seed + 1000 + offset
            )
            if output_valid
            else {"median": float("nan"), "mad": float("nan"), "lcb": 0.0}
        )
        effective_rank, active_rank = _effective_rank_from_ctrl(control)
        output_saturation = (
            min(1.0, id_output["median"] / active_rank)
            if output_valid and math.isfinite(id_output["median"])
            else 0.0
        )
        rows.append(
            BranchGeometry(
                step=step,
                layer=key.layer,
                branch=key.branch,
                active_rank=active_rank,
                id_input_median=id_input["median"],
                id_input_mad=id_input["mad"],
                id_input_lcb=id_input["lcb"],
                id_input_ema=id_input["lcb"],
                id_output_median=id_output["median"],
                id_output_mad=id_output["mad"],
                output_energy=energy,
                output_id_valid=output_valid,
                output_id_saturation=output_saturation,
                effective_rank=effective_rank,
                parameter_rank_saturation=min(1.0, effective_rank / active_rank),
                num_examples=len(inputs[key]),
                wall_time=time.perf_counter() - started,
            )
        )
    return rows


def append_geometry_csv(path, rows: Iterable[BranchGeometry]):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        if new_file:
            writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
