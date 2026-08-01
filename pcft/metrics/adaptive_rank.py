import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from ..rank import BranchKey, RankMap, validate_rank_map
from .branch_geometry import BranchGeometry
from .rank_probe import (
    RankProbeResult,
    append_probe_csv,
    calibration_loss,
    probe_pair,
    probe_single,
)


@dataclass
class RankTransferResult:
    step: int
    move_index: int
    receiver_layer: int
    receiver_branch: str
    receiver_rank_before: int
    receiver_rank_after: int
    donor_layer: int
    donor_branch: str
    donor_rank_before: int
    donor_rank_after: int
    approx_gain: float
    direct_gain: float
    kl_before: float
    kl_after: float
    switch_penalty: float
    final_score: float
    accepted: bool


def robust_normalize(values):
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array)
    array = np.where(finite, array, np.median(array[finite]))
    median = np.median(array)
    mad = np.median(np.abs(array - median))
    if mad < 1e-12:
        return np.zeros_like(array)
    return np.clip((array - median) / (1.4826 * mad), -2.5, 2.5)


def rank_distribution(rank_map: RankMap, rank_min: int):
    extra = np.asarray([rank - rank_min for _, rank in sorted(rank_map.items())], dtype=np.float64)
    total = extra.sum()
    return np.full_like(extra, 1.0 / len(extra)) if total <= 0 else extra / total


def kl_divergence(q, p, epsilon=1e-12):
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    mask = q > 0
    return float(np.sum(q[mask] * np.log((q[mask] + epsilon) / (p[mask] + epsilon))))


class IDRegularizedRankExchangeOptimizer:
    """Calibration-loss rank exchange with an uncertainty-aware branch-ID prior."""

    def __init__(
        self,
        rank_min=8,
        rank_max=128,
        rank_quantum=8,
        global_budget=3584,
        id_prior_beta=2.0,
        id_regularization_tau=0.05,
        switch_cost=0.001,
        move_threshold=0.0,
        receiver_count=10,
        donor_count=10,
        direct_verify_pairs=6,
        max_transfers=4,
        id_ema_alpha=0.3,
        use_id_prior=True,
    ):
        self.rank_min = int(rank_min)
        self.rank_max = int(rank_max)
        self.quantum = int(rank_quantum)
        self.global_budget = int(global_budget)
        self.beta = float(id_prior_beta)
        self.tau = float(id_regularization_tau) if use_id_prior else 0.0
        self.switch_cost = float(switch_cost)
        self.move_threshold = float(move_threshold)
        self.receiver_count = int(receiver_count)
        self.donor_count = int(donor_count)
        self.direct_verify_pairs = int(direct_verify_pairs)
        self.max_transfers = int(max_transfers)
        self.ema_alpha = float(id_ema_alpha)
        self.use_id_prior = bool(use_id_prior)
        self.id_ema: Dict[BranchKey, float] = {}

    def id_prior(self, geometry: Dict[BranchKey, BranchGeometry]):
        keys = sorted(geometry)
        values = []
        for key in keys:
            current = geometry[key].id_input_ema
            self.id_ema[key] = current
            values.append(current)
        if not self.use_id_prior:
            uniform = 1.0 / max(1, len(keys))
            return keys, {key: uniform for key in keys}
        logits = self.beta * robust_normalize(values)
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        return keys, {key: float(value) for key, value in zip(keys, probabilities)}

    def _kl(self, rank_map, ordered_keys, prior):
        q = rank_distribution(rank_map, self.rank_min)
        p = np.asarray([prior[key] for key in ordered_keys])
        return kl_divergence(q, p)

    def shortlist(self, rank_map, geometry, prior, gradient_ema=None):
        gradient_ema = gradient_ema or {}
        receivers = []
        donors = []
        for key, rank in rank_map.items():
            row = geometry[key]
            gradient = max(0.0, float(gradient_ema.get(key, 0.0)))
            if self.use_id_prior:
                receiver_score = (
                    math.log(prior[key] + 1e-12)
                    + row.output_id_saturation
                    + row.parameter_rank_saturation
                    + 0.05 * math.log1p(gradient)
                )
                donor_score = (
                    -math.log(prior[key] + 1e-12)
                    + (1.0 - row.output_id_saturation)
                    + (1.0 - row.parameter_rank_saturation)
                )
            else:
                receiver_score = row.parameter_rank_saturation + math.log1p(gradient)
                donor_score = 1.0 - row.parameter_rank_saturation
            if rank + self.quantum <= self.rank_max:
                receivers.append((receiver_score, key))
            if rank - self.quantum >= self.rank_min:
                donors.append((donor_score, key))
        receivers = [key for _, key in sorted(receivers, reverse=True)[: self.receiver_count]]
        donors = [key for _, key in sorted(donors, reverse=True)[: self.donor_count]]
        return receivers, donors

    def run_event(
        self,
        model,
        dataloader,
        current_map: RankMap,
        geometry_rows: List[BranchGeometry],
        step: int,
        probe_examples: int,
        direct_examples: int,
        metrics_dir,
        gradient_ema=None,
    ):
        current = dict(current_map)
        validate_rank_map(current, self.rank_min, self.rank_max, self.quantum, self.global_budget)
        geometry = {row.key: row for row in geometry_rows}
        ordered_keys, prior = self.id_prior(geometry)
        baseline_loss, _ = calibration_loss(model, dataloader, current, probe_examples)
        all_probes: List[RankProbeResult] = []
        transfers: List[RankTransferResult] = []
        rank_before_event = dict(current)

        for move_index in range(self.max_transfers):
            receivers, donors = self.shortlist(current, geometry, prior, gradient_ema)
            if not receivers or not donors:
                break
            add_results = {
                key: probe_single(
                    model, dataloader, current, key, current[key] + self.quantum,
                    baseline_loss, probe_examples, step, "add"
                )
                for key in receivers
            }
            remove_results = {
                key: probe_single(
                    model, dataloader, current, key, current[key] - self.quantum,
                    baseline_loss, probe_examples, step, "remove"
                )
                for key in donors
            }
            all_probes.extend(add_results.values())
            all_probes.extend(remove_results.values())
            kl_before = self._kl(current, ordered_keys, prior)
            approximate_raw = []
            for receiver in receivers:
                for donor in donors:
                    if receiver == donor:
                        continue
                    candidate = dict(current)
                    candidate[receiver] += self.quantum
                    candidate[donor] -= self.quantum
                    kl_after = self._kl(candidate, ordered_keys, prior)
                    approximate_gain = (
                        add_results[receiver].marginal_gain - remove_results[donor].marginal_gain
                    )
                    penalty = self.switch_cost * 2 * self.quantum
                    approximate_raw.append(
                        (receiver, donor, approximate_gain, kl_after, penalty)
                    )
            if not approximate_raw:
                break

            gain_scale = float(
                np.median(np.abs([row[2] for row in approximate_raw]))
            )
            gain_scale = max(gain_scale, 1e-6)
            approximate = []
            for receiver, donor, approximate_gain, kl_after, penalty in approximate_raw:
                score = (
                    approximate_gain / gain_scale
                    - self.tau * (kl_after - kl_before)
                    - penalty
                )
                approximate.append((score, receiver, donor, approximate_gain, kl_after, penalty))

            verified = []
            for approx in sorted(approximate, reverse=True)[: self.direct_verify_pairs]:
                _, receiver, donor, approximate_gain, kl_after, penalty = approx
                direct = probe_pair(
                    model, dataloader, current, receiver, donor, self.quantum,
                    baseline_loss, direct_examples, step,
                )
                all_probes.append(direct)
                direct_score = (
                    direct.marginal_gain / gain_scale
                    - self.tau * (kl_after - kl_before)
                    - penalty
                )
                verified.append(
                    (direct_score, receiver, donor, approximate_gain, kl_after, penalty, direct)
                )
            best = max(verified, default=None, key=lambda row: row[0])
            if best is None:
                break
            score, receiver, donor, approximate_gain, kl_after, penalty, direct = best
            accepted = score > self.move_threshold
            transfers.append(
                RankTransferResult(
                    step, move_index, receiver.layer, receiver.branch, current[receiver],
                    current[receiver] + self.quantum, donor.layer, donor.branch, current[donor],
                    current[donor] - self.quantum, approximate_gain, direct.marginal_gain,
                    kl_before, kl_after, penalty, score, accepted,
                )
            )
            if not accepted:
                break
            current[receiver] += self.quantum
            current[donor] -= self.quantum
            baseline_loss, _ = calibration_loss(model, dataloader, current, probe_examples)
            validate_rank_map(current, self.rank_min, self.rank_max, self.quantum, self.global_budget)

        metrics_dir = Path(metrics_dir)
        append_probe_csv(metrics_dir / "rank_probe_all.csv", all_probes)
        append_transfer_csv(metrics_dir / "rank_transfer_all.csv", transfers)
        append_rank_map_csv(
            metrics_dir / "rank_all.csv", step, rank_before_event, current,
            prior, geometry, self.global_budget,
        )
        return current, prior, transfers


def append_transfer_csv(path, rows: Iterable[RankTransferResult]):
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


def append_rank_map_csv(path, step, before, after, prior, geometry, budget):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step", "layer", "branch", "rank_before", "rank_after", "rank_change",
        "id_prior", "output_id_saturation", "parameter_rank_saturation",
        "sum_rank_before", "sum_rank_after", "global_budget", "budget_valid",
    ]
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for key in sorted(after):
            writer.writerow(
                {
                    "step": step,
                    "layer": key.layer,
                    "branch": key.branch,
                    "rank_before": before[key],
                    "rank_after": after[key],
                    "rank_change": after[key] - before[key],
                    "id_prior": prior[key],
                    "output_id_saturation": geometry[key].output_id_saturation,
                    "parameter_rank_saturation": geometry[key].parameter_rank_saturation,
                    "sum_rank_before": sum(before.values()),
                    "sum_rank_after": sum(after.values()),
                    "global_budget": budget,
                    "budget_valid": sum(after.values()) == budget,
                }
            )
