import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch

from ..rank import BranchKey, RankMap, override_rank_map


@dataclass
class RankProbeResult:
    step: int
    probe_type: str
    layer: int
    branch: str
    peer_layer: Optional[int]
    peer_branch: Optional[str]
    current_rank: int
    candidate_rank: int
    baseline_loss: float
    candidate_loss: float
    marginal_gain: float
    num_examples: int
    probe_wall_time: float


@torch.no_grad()
def calibration_loss(model, dataloader, rank_map: RankMap, max_examples: int):
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    count = 0
    with override_rank_map(model, rank_map):
        for batch in dataloader:
            batch = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in batch.items()}
            batch_size = batch["input_ids"].shape[0]
            loss = model(**batch, use_cache=False).loss
            loss_sum += float(loss.item()) * batch_size
            count += batch_size
            if count >= max_examples:
                break
    if was_training:
        model.train()
    return loss_sum / max(1, count), count


def probe_single(
    model,
    dataloader,
    current_map,
    key: BranchKey,
    candidate_rank: int,
    baseline_loss: float,
    max_examples: int,
    step: int,
    probe_type: str,
):
    started = time.perf_counter()
    candidate = dict(current_map)
    candidate[key] = candidate_rank
    candidate_loss, count = calibration_loss(model, dataloader, candidate, max_examples)
    gain = baseline_loss - candidate_loss if probe_type == "add" else candidate_loss - baseline_loss
    return RankProbeResult(
        step, probe_type, key.layer, key.branch, None, None,
        current_map[key], candidate_rank, baseline_loss, candidate_loss,
        gain, count, time.perf_counter() - started,
    )


def probe_pair(
    model,
    dataloader,
    current_map,
    receiver: BranchKey,
    donor: BranchKey,
    quantum: int,
    baseline_loss: float,
    max_examples: int,
    step: int,
):
    started = time.perf_counter()
    direct_baseline, _ = calibration_loss(model, dataloader, current_map, max_examples)
    candidate = dict(current_map)
    candidate[receiver] += quantum
    candidate[donor] -= quantum
    candidate_loss, count = calibration_loss(model, dataloader, candidate, max_examples)
    return RankProbeResult(
        step, "direct_pair", receiver.layer, receiver.branch, donor.layer, donor.branch,
        current_map[receiver], candidate[receiver], direct_baseline, candidate_loss,
        direct_baseline - candidate_loss, count, time.perf_counter() - started,
    )


def append_probe_csv(path, rows: Iterable[RankProbeResult]):
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
